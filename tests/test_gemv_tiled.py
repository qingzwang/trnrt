"""The decode projection kernel (kernels/gemv_tiled.py): the blob layout is W^T tile for tile, emulate() is x @ W^T in
fp32 rounded once, and (with the NKI package) the kernel in the simulator, and on a NeuronCore under TRNRT_TEST_DEVICE=1,
equals emulate() at the shapes GLM-5.3's decode takes (K a multiple of 128 and not, N within one chunk and across
several, one and two programs)."""

import importlib.util

import pytest
import torch

from trnrt.kernels import gemv_tiled as gt

needs_nki = pytest.mark.skipif(importlib.util.find_spec("nki") is None, reason="needs the NKI package (Neuron venv)")


def test_pack_lays_w_transposed_with_k_on_the_partitions():
    w = torch.randn(300, 200)
    blob = gt.pack(w, torch.float32)
    assert blob.shape == (gt.P, 2, 300)
    for i, c, j in ((0, 0, 0), (5, 1, 7), (71, 1, 299)):
        assert blob[i, c, j] == w[j, c * gt.P + i]
    assert torch.all(blob[72:, 1] == 0)  # K 200: tile 1 holds rows 128 .. 199, the rest zero


def test_emulate_is_the_projection():
    g = torch.Generator().manual_seed(0)
    x, w = torch.randn(4, 2624, generator=g), torch.randn(700, 2624, generator=g)
    got = gt.emulate(x, gt.pack(w, torch.float32))
    assert torch.allclose(got, x @ w.t(), rtol=1e-5, atol=1e-3)


@needs_nki
@pytest.mark.parametrize("grid", [1, 2])
@pytest.mark.parametrize(("T", "K", "N"), [(1, 6144, 2624), (4, 2048, 256), (4, 192, 6144), (8, 256, 6144)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
def test_kernel_in_the_simulator_is_its_emulation(grid, T, K, N, dtype):
    import nki

    g = torch.Generator().manual_seed(T * 7 + K)
    x = torch.randn(T, K, generator=g).to(dtype)
    blob = gt.pack(torch.randn(N, K, generator=g) * K**-0.5, dtype)
    kern = gt.gemv_tiled_kernel[grid] if grid > 1 else gt.gemv_tiled_kernel
    got = torch.as_tensor(nki.simulate(kern)(x=x, blob=blob, rev=gt.REV)).float()
    want = gt.emulate(x, blob).float()
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    assert (got - want).abs().max().item() <= tol * want.abs().max().item()
