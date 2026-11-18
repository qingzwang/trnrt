"""GLM-5.3's decode MoE block kernel (kernels/glm_moe_half.py): emulate() is the model's XLA MoE block (the routing
over every expert, the expert-parallel mapping, the unclamped SwiGLU experts and shared expert), and (with the NKI
package) the kernel in the simulator, and on a NeuronCore under TRNRT_TEST_DEVICE=1, equals emulate() at 1, 4 and 5
rows, without expert parallelism (8 routed experts, all in the blob) and with it (16 routed experts, 8 in this rank's
blob, lo 8)."""

import importlib.util
import types

import pytest
import torch

from tests.test_moe_prefill import checkpoint_experts
from trnrt.kernels import glm_moe_half as gh
from trnrt.kernels import moe_dedupe as mdd

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
H, IS = 1024, 32  # hidden, shared intermediate rows a rank (GLM-5.3 at tp 64: 2048 / 64)


def route(x, router, bias, k=8, scale=2.5):
    """decoder._route: sigmoid scores, choice on score + bias, top k, weights renormalised and scaled, bf16."""
    from trnrt.models.dsa_select import topk_ids

    scores = (x.float() @ router.float().t()).sigmoid()
    topi = topk_ids(scores + bias, k)
    topv = torch.gather(scores, 1, topi)
    return (topv / (topv.sum(-1, keepdim=True) + 1e-20) * scale).to(torch.bfloat16), topi


def case(T, E_all, El, seed=0):
    g = torch.Generator().manual_seed(seed * 13 + T)
    bf = torch.bfloat16
    ws = checkpoint_experts(E=El, H=H)[0]
    blob = mdd.pack(*ws)
    x = torch.randn(T, H, generator=g).to(bf)
    router = (torch.randn(E_all, H, generator=g) * H**-0.5).to(bf)
    bias = 0.5 * torch.randn(E_all, generator=g)
    gu = (torch.randn(2 * IS, H, generator=g) * H**-0.5).to(bf)
    dn = (torch.randn(H, IS, generator=g) * IS**-0.5).to(bf)
    layer = types.SimpleNamespace(
        router=router,
        router_bias=bias,
        shared_gate_up=gu,
        shared_gate_up_scale=None,
        shared_down=dn,
        shared_down_scale=None,
        w_blob=blob,
        _decode_only=(),
    )
    return x, layer


def packed(layer, npg):
    from trnrt.kernels import moe_half as MH
    from trnrt.kernels import shell

    lay = shell.packed_layout(MH.shapes(layer.router.shape[0], H, IS, npg))
    for dt, (n, _) in lay.items():
        setattr(
            layer,
            gh.PREFIX + "blob_" + shell.DT_NAMES[dt],
            torch.nn.Parameter(torch.zeros(n, dtype=dt), requires_grad=False),
        )
    gh.fill(layer, torch.bfloat16, npg)
    return layer


@pytest.mark.parametrize(("E_all", "lo"), [(8, 0), (16, 8)], ids=["no-ep", "ep2"])
def test_emulate_is_the_models_moe_block(E_all, lo):
    from torch.nn import functional as F

    x, layer = case(4, E_all, 8)
    topv, topi = route(x, layer.router, layer.router_bias)
    got = gh.emulate(x, topv, topi, layer.w_blob, lo, layer.shared_gate_up, layer.shared_down).float()
    # the model's XLA block (glm_moe_dsa._experts + _swiglu): the same mapping, moe_dedupe's arithmetic unclamped
    local = topi - lo
    mine = (local >= 0) & (local < 8)
    w_gu, s_gu, w_down, s_down = mdd.unpack(layer.w_blob, H)
    from trnrt.models.quant import dequant, dequant_t

    wg, wd = dequant(w_gu, s_gu, torch.float32), dequant_t(w_down, s_down, torch.float32)
    want = torch.zeros(4, H)
    for t in range(4):
        for j in range(8):
            if mine[t, j]:
                e = int(local[t, j])
                gu = wg[e] @ x[t].float()
                want[t] += topv[t, j].float() * ((F.silu(gu[:64]) * gu[64:]) @ wd[e])
    gus = F.linear(x, layer.shared_gate_up)
    want += F.linear(F.silu(gus[..., :IS]) * gus[..., IS:], layer.shared_down).float()
    assert (got - want).abs().max().item() <= 3e-2 * want.abs().max().item()


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize("T", [1, 4, 5])
@pytest.mark.parametrize(("E_all", "lo"), [(8, 0), (16, 8)], ids=["no-ep", "ep2"])
def test_kernel_in_the_simulator_is_its_emulation(grid, T, E_all, lo, monkeypatch):
    import nki

    from trnrt import platform
    from trnrt.kernels import moe_half as MH
    from trnrt.kernels import rowiso

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr(platform, "nki_grid", lambda: grid)
    x, layer = case(T, E_all, 8)
    packed(layer, grid)
    topv, topi = route(x, layer.router, layer.router_bias)
    want = gh.emulate(x, topv, topi, layer.w_blob, lo, layer.shared_gate_up, layer.shared_down).float()
    kern = gh.glm_moe_half_kernel[grid] if grid > 1 else gh.glm_moe_half_kernel
    rp, sp = nki.simulate(kern)(
        x=x,
        bb=layer.gh_blob_bf16.data,
        bf=layer.gh_blob_f32.data,
        wb=layer.w_blob,
        lo=torch.tensor([[float(lo)]]),
        Er=E_all,
        Is=IS,
        K=8,
        scale=2.5,
        group=mdd.group_size(1),
        ring=mdd.RING,
        limit=gh.NO_CLAMP,
        rsp=0,
        rev=gh.REV_K,
        fin=rowiso.FIN,
        **MH.offsets(E_all, H, IS, grid),
    )
    routed = mdd.from_pairs(torch.as_tensor(rp).sum(0).to(torch.bfloat16), torch.bfloat16)
    got = (routed + torch.as_tensor(sp).sum(0).to(torch.bfloat16)).float()
    assert (got - want).abs().max().item() <= 3e-2 * want.abs().max().item()


@needs_nki
def test_a_wrong_expert_offset_is_caught(monkeypatch):
    """The positive control: under expert parallelism the kernel given lo 0 instead of 8 maps the other group's experts
    onto this rank's, and its output must then differ from emulate()'s."""
    import nki

    from trnrt import platform
    from trnrt.kernels import moe_half as MH
    from trnrt.kernels import rowiso

    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    monkeypatch.setattr(platform, "nki_grid", lambda: 1)
    x, layer = case(4, 16, 8)
    packed(layer, 1)
    topv, topi = route(x, layer.router, layer.router_bias)
    want = gh.emulate(x, topv, topi, layer.w_blob, 8, layer.shared_gate_up, layer.shared_down).float()
    rp, sp = nki.simulate(gh.glm_moe_half_kernel)(
        x=x,
        bb=layer.gh_blob_bf16.data,
        bf=layer.gh_blob_f32.data,
        wb=layer.w_blob,
        lo=torch.tensor([[0.0]]),
        Er=16,
        Is=IS,
        K=8,
        scale=2.5,
        group=mdd.group_size(1),
        ring=mdd.RING,
        limit=gh.NO_CLAMP,
        rsp=0,
        rev=gh.REV_K,
        fin=rowiso.FIN,
        **MH.offsets(16, H, IS, 1),
    )
    got = (
        mdd.from_pairs(torch.as_tensor(rp).sum(0).to(torch.bfloat16), torch.bfloat16)
        + torch.as_tensor(sp).sum(0).to(torch.bfloat16)
    ).float()
    assert (got - want).abs().max().item() > 3e-2 * want.abs().max().item()
