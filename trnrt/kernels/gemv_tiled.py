"""A decode step's projection y = x W^T for a few rows (T <= 128: a decode or verify batch) as one NKI kernel, over W
dequantized once at load and pre-tiled with its input dimension on the partitions (GLM-5.3's decode weights,
models/glm_moe_dsa.py DECODE_COPIES).

Why a kernel: XLA lowers F.linear(x, W) with W [out, in] by transposing W on the tensor engine in every step (W's input
dimension must be on the partitions for the PE's contraction): profiled on trn2 (neuron-explorer, GLM-5.3 batch-1 decode,
78 layers, rank 0 of 64), 301 of the step's 314 GFLOP were those transposes, 174.6 ms a step. A copy stored [in, out]
removed the transposes but neuronx-cc read it back through 81.9 GB of DMA a step (323 ms): XLA gives no say over the
layout it loads a weight in. Here the layout is the kernel's: blob[i, c, j] = W[j, c 128 + i] (C = ceil(K / 128) tiles of
128 input rows, the last zero-padded), so a chunk of output columns is one DMA of [128, C, n] that is contiguous on every
partition, and the PE reads it as the moving operand directly: no transpose, every weight byte read once.

Weights load in super-chunks of G chunks (as many as fit LOAD_BYTES a partition: a small K's chunk alone is mostly
overhead), each load in SPLIT pieces on the scalar and sync engines' hardware DMA queues. Per chunk of NCH output columns
(one PSUM bank of fp32): C matmuls stationary = x^T tile [128, T], moving = the chunk's [128, n] of tile c, accumulating
in one PSUM tile (accumulate=(c > 0), every flag set), then one copy to SBUF in x's dtype and one DMA out. At LNC 2 the
programs take alternate super-chunks; they share nothing. x^T [128, C, T] is loaded once, tile by tile (x is a few rows). Arithmetic: products and sums in the PE's fp32,
rounded once to x's dtype; emulate() is that in torch (fp32 matmul, one rounding), so it is close to F.linear's, not
bit-equal (the PE's K grouping is not XLA's).
"""

from __future__ import annotations

import torch

P = 128
NCH = 512  # output columns per chunk: one 2 KB PSUM bank of fp32 a partition
SPLIT = 4  # pieces of each weight load (benchmarked on trn2 at GLM-5.3's w_a: 1 -> 89.2 us, 4 on two queues -> 72.5 us)
LOAD_BYTES = 32768  # a weight load's bytes on a partition, at most (several chunks a load when K is small)


def tiles(K: int) -> int:
    return -(-K // P)


def pack(w: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """W [N, K] (any float dtype) as the kernel's blob [P, C, N] in `dtype`: blob[i, c, j] = W[j, c P + i], zero past K."""
    N, K = w.shape
    C = tiles(K)
    wt = torch.zeros(C * P, N, dtype=torch.float32)
    wt[:K] = w.float().t()
    return wt.view(C, P, N).permute(1, 0, 2).contiguous().to(dtype)


def emulate(x: torch.Tensor, blob: torch.Tensor) -> torch.Tensor:
    """The kernel's arithmetic in torch: x [T, K] @ the blob's W^T in fp32, rounded once to x's dtype: [T, N]."""
    _, C, N = blob.shape
    w = blob.float().permute(1, 0, 2).reshape(C * P, N)[: x.shape[-1]]
    return (x.float() @ w).to(x.dtype)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# -- the kernel (its span is what REV hashes)
if nki is not None:
    @nki.jit
    def gemv_tiled_kernel(x, blob, rev: int):
        """y [T, N] = x [T, K] @ W^T for the blob [P, C, N] (pack) in x's dtype; the programs take alternate chunks of
        NCH output columns."""
        T, K = x.shape
        _, C, N = blob.shape
        dt = x.dtype
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        out = nl.ndarray((T, N), dtype=dt, buffer=nl.shared_hbm)
        xt = nl.ndarray((P, C, T), dtype=dt, buffer=nl.sbuf)  # x^T: element (i, c, t) = x[t, c P + i]
        if C * P != K:  # the last tile's rows past K stay 0 (the blob's are 0 too)
            nisa.memset(dst=xt[:, C - 1, :], value=0.0)
        for c in range(C):
            kc = min(P, K - c * P)
            nisa.dma_copy(dst=xt[0:kc, c, :], src=x.ap(pattern=[[1, kc], [K, T]], offset=c * P))
        nq = -(-N // NCH)
        # Super-chunks of G chunks per weight load: as many as fit LOAD_BYTES on a partition (a small K's chunk alone
        # is mostly per-chunk overhead), at most every program's share.
        G = max(1, min(LOAD_BYTES // (C * NCH * 2), -(-nq // npg)))
        ns = -(-nq // G)
        for j in range(-(-ns // npg)):
            sq = j * npg + pid
            if sq < ns:
                m0 = sq * G * NCH
                mw = min(G * NCH, N - m0)
                wt = nl.ndarray((P, C, mw), dtype=dt, buffer=nl.sbuf)
                # The load in SPLIT pieces over the K tiles, alternating the scalar and sync engines' hardware DGE
                # queues (one queue held a w_a call to 361 GB/s on trn2, four pieces on two reached 445 GB/s).
                cs = -(-C // SPLIT)
                for s_ in range(SPLIT):
                    c0 = s_ * cs
                    c1 = min(C, c0 + cs)
                    if c0 < c1:
                        nisa.dma_copy(dst=wt[:, c0:c1, :],
                                      src=blob.ap(pattern=[[C * N, P], [N, c1 - c0], [1, mw]], offset=c0 * N + m0),
                                      engine=nisa.engine.scalar if s_ % 2 == 0 else nisa.engine.sync,
                                      dge_mode=nisa.dge_mode.hwdge)
                for u in range(-(-mw // NCH)):
                    u0 = u * NCH
                    nw = min(NCH, mw - u0)
                    ps = nl.ndarray((T, nw), dtype=nl.float32, buffer=nl.psum)
                    for c in range(C):
                        nisa.nc_matmul(dst=ps, stationary=xt[:, c, :], moving=wt[:, c, u0:u0 + nw], accumulate=(c > 0))
                    ot = nl.ndarray((T, nw), dtype=dt, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ot, src=ps, engine=nisa.vector_engine)
                    nisa.dma_copy(dst=out.ap(pattern=[[N, T], [1, nw]], offset=m0 + u0), src=ot)
        return out
else:
    gemv_tiled_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of the kernel's source span and the constants it reads."""
    import zlib

    src = open(__file__).read()
    a = src.index("# -- the kernel (its span is what REV hashes)")
    b = src.index("    gemv_tiled_kernel = None", a)
    return zlib.crc32(repr((P, NCH, SPLIT, LOAD_BYTES)).encode(), zlib.crc32(src[a:b].encode()))


REV = _kernel_rev()


def takes(T: int) -> bool:
    """Whether a call of T rows can take the kernel (x^T's tile is the PE's stationary operand: at most P columns)."""
    return 0 < T <= P


def gemv(x: torch.Tensor, blob: torch.Tensor) -> torch.Tensor:
    """x [T, K] @ W^T for W packed in blob (pack): the kernel at the runtime's grid on a Neuron device (or under
    shell.HOST_SIM), emulate() on the host."""
    from . import shell as SH

    if x.device.type == "cpu" and not SH.HOST_SIM:
        return emulate(x, blob)
    if gemv_tiled_kernel is None:
        raise RuntimeError("the NKI decode projection kernel needs the nki package (the Neuron venv)")
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    return wrap_nki(gemv_tiled_kernel)[platform.nki_grid()](x=x.to(blob.dtype).contiguous(), blob=blob, rev=REV)
