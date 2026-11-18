"""GLM-5.3's decode attention front kernel (kernels/glm_attn_front.py): its emulate() in the model's decode (at one
head a rank, tp 4 on the tiny model, TRNRT_GLM_FRONT_HOST=1) generates what tp 1's XLA path generates, and (with the
NKI package) the kernel in the simulator equals emulate() at 1 and 3 rows, on a full and a shared layer, at grid 1 and
2, with a positive control (a wrong placement must be caught)."""

import importlib.util

import pytest
import torch

from tests.test_glm_moe_dsa import build, engine, prompts
from trnrt.kernels import gemv_tiled as GT
from trnrt.kernels import glm_attn_front as GF

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
BF = torch.bfloat16
H, QL, R, DR, DN, DI, N = 512, 256, 128, 64, 64, 128, 112  # W = QL + R + DR = 448: four shards of 112 columns
W = QL + R + DR


def test_kernel_path_in_the_engine_matches_tp1(tmp_path, monkeypatch):
    """At tp 4 (one head a rank, decode's w_a in column shards) the model's decode through the front's emulate()
    generates tp 1's greedy tokens."""
    from trnrt.engine.request import SamplingParams
    from trnrt.models import glm_moe_dsa

    monkeypatch.setattr(glm_moe_dsa, "MOE_EP", 1)
    path = str(tmp_path)
    build(path, layers=4)
    ps = prompts(11, (9, 30))
    sp = SamplingParams(max_new_tokens=6, ignore_eos=True)
    want = [r.output_ids for r in engine(path, max_num_seqs=2).generate(ps, sp)]
    monkeypatch.setenv("TRNRT_GLM_FRONT_HOST", "1")
    eng = engine(path, tp=4, max_num_seqs=2)
    try:
        assert GF.takes(eng.model, eng.model.layers[0], 1)
        got = [r.output_ids for r in eng.generate(ps, sp)]
    finally:
        eng.close()
    assert got == want


def case(T, full: bool, lo: int, seed=0):
    g = torch.Generator().manual_seed(seed * 7 + T + 100 * int(full))
    h = (torch.randn(T, H, generator=g) * 3).to(BF)
    gin = (1 + 0.1 * torch.randn(H, generator=g)).to(BF)
    gq = (1 + 0.1 * torch.randn(QL, generator=g)).to(BF)
    gkv = (1 + 0.1 * torch.randn(R, generator=g)).to(BF)
    wa = GT.pack(torch.randn(N, H, generator=g) * H**-0.5, BF)
    wqb = GT.pack(torch.randn(DN + DR, QL, generator=g) * QL**-0.5, BF)
    wik = GT.pack(torch.randn(DI, H, generator=g) * H**-0.5, BF) if full else None
    knw = 1 + 0.1 * torch.randn(DI, generator=g)
    knb = 0.1 * torch.randn(DI, generator=g)
    ang = torch.rand(T, DR // 2, generator=g) * 6.28
    cos, sin = ang.cos().to(BF), ang.sin().to(BF)
    place = (torch.arange(W).view(1, W) == torch.arange(N).view(N, 1) + lo).to(BF)
    return h, gin, gq, gkv, wa, wqb, wik, knw, knb, cos, sin, place


def run_sim(grid, T, full, lo, place_lo=None):
    import nki

    h, gin, gq, gkv, wa, wqb, wik, knw, knb, cos, sin, place = case(T, full, lo if place_lo is None else place_lo)
    kern = GF.glm_attn_front_kernel[grid] if grid > 1 else GF.glm_attn_front_kernel
    kn = torch.stack([knw, knb]) if full else torch.zeros(2, 1)
    got = nki.simulate(kern)(
        h=h,
        gin=gin.view(1, -1),
        wa=wa,
        place=place,
        gq=gq.view(1, -1),
        gkv=gkv.view(1, -1),
        wqb=wqb,
        wik=wik if full else wqb,
        kn=kn,
        cs=torch.cat([cos, sin], dim=-1),
        ident=torch.eye(128, dtype=BF),
        eps=1e-5,
        eps_q=1e-6,
        ql=QL,
        r=R,
        dn=DN,
        dr=DR,
        world=1,
        cc=0,
        full=int(full),
        qres=1,
        rev=GF.REV,
    )
    h, gin, gq, gkv, wa, wqb, wik, knw, knb, cos, sin, _ = case(T, full, lo)
    want = GF.emulate(h, gin, wa, lo, W, gq, gkv, wqb, wik, knw, knb, cos, sin, 1e-5, 1e-6, QL, R, DN, DR)
    return [torch.as_tensor(x).float() for x in got], want


def close(got, want, rel=2e-2):
    want = want.float()
    return (got - want).abs().max().item() <= rel * max(want.abs().max().item(), 1e-3)


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize("T", [1, 3])
@pytest.mark.parametrize("full", [True, False], ids=["full", "shared"])
@pytest.mark.parametrize("lo", [224, 336])  # the shard across q's and c's parts, across c's and k_pe's
def test_kernel_in_the_simulator_is_its_emulation(grid, T, full, lo, monkeypatch):
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    (qn, qp, kv, ki, qr), (wqn, wqp, wkv, wki, wqr) = run_sim(grid, T, full, lo=lo)
    assert close(qn, wqn) and close(qp, wqp) and close(kv, wkv) and close(qr, wqr)
    if full:
        assert close(ki, wki)


@needs_nki
def test_a_wrong_placement_is_caught(monkeypatch):
    """The positive control: the shard placed at column 0 instead of 224 puts its columns in q's part, not c's."""
    monkeypatch.setenv("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
    got, want = run_sim(1, 3, True, lo=224, place_lo=0)
    assert not close(got[2], want[2])  # the cache row
