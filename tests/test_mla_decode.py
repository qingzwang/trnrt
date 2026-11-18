"""GLM-5.3's one-head MLA decode attention kernel (kernels/mla_decode.py): emulate() is the model's absorbed decode
attention (models/glm_moe_dsa.py _attend) up to bf16 rounding, and (with the NKI package) the kernel in the simulator,
and on a NeuronCore under TRNRT_TEST_DEVICE=1, equals emulate() at GLM-5.3's per-rank shapes (192 nope + 64 rope dims,
a 512 latent, 256 value dims), at 1, 4 and 5 rows, over 1 and several 128-token tiles, with masked keys."""

import importlib.util
import types

import pytest
import torch

from trnrt.kernels import mla_decode as md

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")
DN, DR, R, DV = 192, 64, 512, 256


def case(B, L, seed=0, masked=0):
    g = torch.Generator().manual_seed(seed * 101 + B * 7 + L)
    bf = torch.bfloat16
    qn, qp = torch.randn(B, DN, generator=g).to(bf), torch.randn(B, DR, generator=g).to(bf)
    kc = torch.randn(B, L, R + DR, generator=g).to(bf)
    w_uk = (torch.randn(DN, R, generator=g) * DN**-0.5).to(bf)
    w_uv = (torch.randn(DV, R, generator=g) * R**-0.5).to(bf)
    bias = torch.zeros(B, L)
    if masked:  # the last `masked` keys invisible (a shorter context in the bucket), and every 7th (a DSA selection)
        bias[:, L - masked :] = md.NEG_INF
        bias[:, ::7] = md.NEG_INF
    return qn, qp, kc, bias, w_uk, w_uv


def test_emulate_is_the_models_absorbed_attention():
    from trnrt.models import glm_moe_dsa

    qn, qp, kc, bias, w_uk, w_uv = case(3, 256, masked=40)
    scale = 256**-0.5
    got = md.emulate(qn, qp, kc, bias, w_uk, w_uv.t().contiguous(), scale).float()
    model = types.SimpleNamespace(dtype=torch.bfloat16, _w=getattr)
    m = types.SimpleNamespace(
        qk_nope_head_dim=DN, qk_rope_head_dim=DR, v_head_dim=DV, kv_lora_rank=R, softmax_scale=scale
    )
    layer = types.SimpleNamespace(nh=1, spec=types.SimpleNamespace(mla=m), w_uk=w_uk, w_uv=w_uv)
    want = glm_moe_dsa._attend(
        model, layer, qn.view(3, 1, DN), qp.view(3, 1, DR), kc, bias.view(3, 1, -1), 3, 1, absorb=True
    ).float()
    assert (got - want).abs().max().item() <= 2e-2 * want.abs().max().item()


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize(("B", "L", "masked"), [(1, 128, 0), (4, 512, 100), (5, 384, 30)])
def test_kernel_in_the_simulator_is_its_emulation(grid, B, L, masked):
    import nki

    qn, qp, kc, bias, w_uk, w_uv = case(B, L, masked=masked)
    w_uv_t = w_uv.t().contiguous()
    kern = md.mla_decode_kernel[grid] if grid > 1 else md.mla_decode_kernel
    got = torch.as_tensor(
        nki.simulate(kern)(
            q_nope=qn,
            q_pe=qp,
            kc=kc,
            bias=bias,
            w_uk=w_uk,
            w_uv_t=w_uv_t,
            ident=torch.eye(128, dtype=torch.bfloat16),
            scale=256**-0.5,
            rev=md.REV,
        )
    ).float()
    want = md.emulate(qn, qp, kc, bias, w_uk, w_uv_t, 256**-0.5).float()
    assert (got - want).abs().max().item() <= 2e-2 * want.abs().max().item()


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
def test_a_verify_batch_attends_its_sequences_rows(grid):
    """rep: a verify batch of 2 sequences x 3 positions, each position its own visibility, over its sequence's rows
    (kc [2, L, W]); and the positive control: the same rows over the wrong sequences' rows differ."""
    import nki

    B, Q, L = 2, 3, 256
    qn, qp, _, bias, w_uk, w_uv = case(B * Q, L, masked=20)
    kc = case(B, L, seed=1)[2]
    for j in range(B * Q):  # position j of a sequence sees one key more than position j - 1
        bias[j, L - 40 + j % Q :] = md.NEG_INF
    w_uv_t = w_uv.t().contiguous()
    kern = md.mla_decode_kernel[grid] if grid > 1 else md.mla_decode_kernel
    got = torch.as_tensor(
        nki.simulate(kern)(
            q_nope=qn,
            q_pe=qp,
            kc=kc,
            bias=bias,
            w_uk=w_uk,
            w_uv_t=w_uv_t,
            ident=torch.eye(128, dtype=torch.bfloat16),
            scale=256**-0.5,
            rev=md.REV,
            rep=Q,
        )
    ).float()
    want = md.emulate(qn, qp, kc, bias, w_uk, w_uv_t, 256**-0.5, rep=Q).float()
    assert (got - want).abs().max().item() <= 2e-2 * want.abs().max().item()
    wrong = md.emulate(qn, qp, kc.flip(0), bias, w_uk, w_uv_t, 256**-0.5, rep=Q).float()
    assert (got - wrong).abs().max().item() > 2e-2 * want.abs().max().item()
