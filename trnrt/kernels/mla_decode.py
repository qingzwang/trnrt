"""GLM-5.3's absorbed-MLA decode attention for one head per rank (tp 64: 64 heads over 64 ranks), as one NKI kernel:
from a row's q (nope and rotated rope parts) and its context's cache rows [c | k_pe] to the head's attention output,
the o_proj input (models/glm_moe_dsa.py _attend's absorb branch).

Why a kernel: XLA spelled the head's einsums (q_lat = q_nope W_UK, the scores over [c | k_pe], softmax, p c, W_UV) as
~510 small tensor-engine instructions a layer, most of them layout transposes (profiled on trn2: 39,746 matmuls of 220 ns
and as many weight loads in a 78-layer batch-1 decode step, beside the decode projection kernel's).

Per row b (static; at LNC 2 the programs take alternate rows):
- q_lat^T [512, 1] = W_UK^T q_nope directly in column form: stationary = W_UK's [d, 128 r] tiles (W_UK is stored [d, r],
  its input rows on the partitions), moving = q_nope^T [d, 1]; one PSUM tile per 128-row block of r (4 blocks over the
  two d tiles, 192 = 128 + 64). q_pe^T [64, 1] is loaded straight onto the partitions. Both scaled by `scale` on the
  way to bf16, so the scores need no scaling.
- per 128-token tile of the context: K^T [576, 128] by 5 PE transposes (4 x 128 latent dims, 64 rope dims) of the
  tile's rows [128, 576] against the identity (exact through fp32 PSUM), then the tile's scores [1, 128] by 5 matmuls
  accumulating in one PSUM tile, plus the row's bias (0 / NEG_INF: visibility and the DSA selection).
- softmax over the row's scores [1, L] on one partition: the max, exp(s - max) with its sum in the same instruction
  (fp32), p rounded to bf16 unnormalised.
- o_lat^T [512, 1] = sum over tiles of c_tile^T p_tile: each tile's p^T [128, 1] by one matmul against a 1 x 1 one,
  then 4 matmuls a tile accumulating into 4 separate PSUM tiles (one per 128 dims: accumulating into slices of ONE PSUM
  tile is computed wrongly on the device, docs/MEGAKERNELS.md).
- out [1, 256] = o_lat W_UV^T: stationary = o_lat^T's [128, 1] tiles, moving = W_UV^T's [128 r, 256] tiles, then
  times 1 / sum, rounded to bf16 once.
Every context tile stays in SBUF from its scores to its p c (128 tokens x 576 x 2 B = 1.15 KB a partition a tile).

Numerics: emulate() is this arithmetic in torch. It is not _attend's bit for bit: p is normalised once at the end
instead of before its bf16 rounding, and the PE's K grouping is not XLA's einsums'.
"""

from __future__ import annotations

import torch

P = 128
NEG_INF = -1e30


def emulate(q_nope: torch.Tensor, q_pe: torch.Tensor, kc: torch.Tensor, bias: torch.Tensor, w_uk: torch.Tensor,
            w_uv_t: torch.Tensor, scale: float) -> torch.Tensor:
    """The kernel's arithmetic: q_nope [B, dn], q_pe [B, dr], kc [B, L, r + dr] ([c | k_pe]), bias [B, L] fp32, w_uk
    [dn, r], w_uv_t [r, dv] (all bf16) -> [B, dv] bf16."""
    bf = torch.bfloat16
    r = w_uk.shape[1]
    q_lat = (torch.einsum("bd,dr->br", q_nope.float(), w_uk.float()) * scale).to(bf)
    q = torch.cat([q_lat, (q_pe.float() * scale).to(bf)], dim=-1).float()
    s = torch.einsum("br,blr->bl", q, kc.float()) + bias.float()
    e = torch.exp(s - s.amax(-1, keepdim=True))
    p, tot = e.to(bf).float(), e.sum(-1, keepdim=True)
    o_lat = torch.einsum("bl,blr->br", p, kc[..., :r].float()).to(bf).float()
    return (o_lat @ w_uv_t.float() / tot).to(bf)


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# -- the kernel (its span is what REV hashes)
if nki is not None:
    @nki.jit
    def mla_decode_kernel(q_nope, q_pe, kc, bias, w_uk, w_uv_t, ident, scale: float, rev: int):
        """emulate()'s attention, rows split over the LNC programs. q_nope [B, dn], q_pe [B, dr], kc [B, L, r + dr],
        bias [B, L] fp32, w_uk [dn, r], w_uv_t [r, dv], ident [128, 128] (all bf16 but bias); L a multiple of 128,
        r a multiple of 128, dn and dr at most 2 x 128 and 128. Returns bf16 [B, dv]."""
        f32, bf16 = nl.float32, nl.bfloat16
        B, dn = q_nope.shape
        dr = q_pe.shape[1]
        L, W = kc.shape[1], kc.shape[2]
        r, dv = w_uk.shape[1], w_uv_t.shape[1]
        RB, NT = r // P, L // P
        DT = -(-dn // P)
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        out = nl.ndarray((B, dv), dtype=bf16, buffer=nl.shared_hbm)
        I = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        nisa.dma_copy(dst=I, src=ident)
        one = nl.ndarray((1, 1), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=one, value=1.0)
        WK = nl.ndarray((P, DT, r), dtype=bf16, buffer=nl.sbuf)  # W_UK's d rows on the partitions, tile by tile
        for t in range(DT):
            dk = min(P, dn - t * P)
            nisa.dma_copy(dst=WK[0:dk, t, :], src=w_uk.ap(pattern=[[r, dk], [1, r]], offset=t * P * r))
        WV = nl.ndarray((P, RB, dv), dtype=bf16, buffer=nl.sbuf)  # W_UV^T's r rows on the partitions
        nisa.dma_copy(dst=WV, src=w_uv_t.ap(pattern=[[dv, P], [P * dv, RB], [1, dv]], offset=0))
        for j in range(-(-B // npg)):
            b = j * npg + pid
            if b < B:
                # q_nope^T [d, 1] per d tile and q_pe^T [dr, 1], straight onto the partitions
                qn = nl.ndarray((P, DT, 1), dtype=bf16, buffer=nl.sbuf)
                for t in range(DT):
                    dk = min(P, dn - t * P)
                    nisa.dma_copy(dst=qn[0:dk, t, :], src=q_nope.ap(pattern=[[1, dk], [1, 1]], offset=b * dn + t * P))
                qpf = nl.ndarray((dr, 1), dtype=bf16, buffer=nl.sbuf)
                nisa.dma_copy(dst=qpf, src=q_pe.ap(pattern=[[1, dr], [1, 1]], offset=b * dr))
                qp = nl.ndarray((dr, 1), dtype=bf16, buffer=nl.sbuf)
                nisa.activation(dst=qp, op=nl.copy, data=qpf, scale=scale)
                # q_lat^T [r, 1] = W_UK^T q_nope, one PSUM tile per 128 rows of r, scaled into bf16
                ql = nl.ndarray((P, RB, 1), dtype=bf16, buffer=nl.sbuf)
                for v in range(RB):
                    pq = nl.ndarray((P, 1), dtype=f32, buffer=nl.psum)
                    for t in range(DT):
                        dk = min(P, dn - t * P)
                        nisa.nc_matmul(dst=pq, stationary=WK[0:dk, t, v * P:(v + 1) * P], moving=qn[0:dk, t, :],
                                       accumulate=(t > 0))
                    nisa.activation(dst=ql[:, v, :], op=nl.copy, data=pq, scale=scale)
                # the row's bias [1, L] and every context tile [128 tokens, W], kept for p c
                s = nl.ndarray((1, L), dtype=f32, buffer=nl.sbuf)
                nisa.dma_copy(dst=s, src=bias.ap(pattern=[[L, 1], [1, L]], offset=b * L))
                KT = []
                for i in range(NT):
                    kt = nl.ndarray((P, W), dtype=bf16, buffer=nl.sbuf)
                    nisa.dma_copy(dst=kt, src=kc.ap(pattern=[[W, P], [1, W]], offset=(b * L + i * P) * W))
                    KT.append(kt)
                    # K^T [W, 128] in RB + 1 blocks of dims, then the tile's scores [1, 128]
                    kts = []
                    for u in range(RB + 1):
                        du = dr
                        if u < RB:
                            du = P
                        pt = nl.ndarray((du, P), dtype=f32, buffer=nl.psum)
                        nisa.nc_matmul(dst=pt, stationary=kt[:, u * P:u * P + du], moving=I, accumulate=False)
                        ktt = nl.ndarray((du, P), dtype=bf16, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=ktt, src=pt, engine=nisa.vector_engine)
                        kts.append(ktt)
                    ps = nl.ndarray((1, P), dtype=f32, buffer=nl.psum)
                    for u in range(RB):
                        nisa.nc_matmul(dst=ps, stationary=ql[:, u, :], moving=kts[u], accumulate=(u > 0))
                    nisa.nc_matmul(dst=ps, stationary=qp, moving=kts[RB], accumulate=True)
                    nisa.tensor_tensor(dst=s[:, i * P:(i + 1) * P], data1=ps, data2=s[:, i * P:(i + 1) * P],
                                       op=nl.add, engine=nisa.vector_engine)
                # softmax over [1, L]: exp(s - max) with its sum, p bf16 unnormalised
                mx = nl.ndarray((1, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=mx, op=nl.maximum, data=s, axis=1)
                nmx = nl.ndarray((1, 1), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=nmx, data=mx, op0=nl.multiply, operand0=-1.0, engine=nisa.vector_engine)
                p = nl.ndarray((1, L), dtype=bf16, buffer=nl.sbuf)
                tot = nl.ndarray((1, 1), dtype=f32, buffer=nl.sbuf)
                nisa.activation_reduce(dst=p, op=nl.exp, data=s, bias=nmx, reduce_op=nl.add, reduce_res=tot)
                # o_lat^T [r, 1] = sum_i c_i^T p_i: p_i^T [128, 1] by a K = 1 matmul, 4 accumulators (one PSUM tile each)
                po = []
                for v in range(RB):
                    po.append(nl.ndarray((P, 1), dtype=f32, buffer=nl.psum))
                for i in range(NT):
                    pp = nl.ndarray((P, 1), dtype=f32, buffer=nl.psum)
                    nisa.nc_matmul(dst=pp, stationary=p[:, i * P:(i + 1) * P], moving=one, accumulate=False)
                    pt_ = nl.ndarray((P, 1), dtype=bf16, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=pt_, src=pp, engine=nisa.vector_engine)
                    for v in range(RB):
                        nisa.nc_matmul(dst=po[v], stationary=KT[i][:, v * P:(v + 1) * P], moving=pt_,
                                       accumulate=(i > 0))
                ol = nl.ndarray((P, RB, 1), dtype=bf16, buffer=nl.sbuf)
                for v in range(RB):
                    nisa.tensor_copy(dst=ol[:, v, :], src=po[v], engine=nisa.vector_engine)
                # out [1, dv] = o_lat W_UV^T, times 1 / sum
                pv = nl.ndarray((1, dv), dtype=f32, buffer=nl.psum)
                for v in range(RB):
                    nisa.nc_matmul(dst=pv, stationary=ol[:, v, :], moving=WV[:, v, :], accumulate=(v > 0))
                rs = nl.ndarray((1, 1), dtype=f32, buffer=nl.sbuf)
                nisa.reciprocal(dst=rs, data=tot)
                ob = nl.ndarray((1, dv), dtype=bf16, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=ob, data=pv, op0=nl.multiply, operand0=rs, engine=nisa.vector_engine)
                nisa.dma_copy(dst=out.ap(pattern=[[dv, 1], [1, dv]], offset=b * dv), src=ob)
        return out
else:
    mla_decode_kernel = None


def _kernel_rev() -> int:
    """CRC-32 of the kernel's source span and P."""
    import zlib

    src = open(__file__).read()
    a = src.index("# -- the kernel (its span is what REV hashes)")
    b = src.index("    mla_decode_kernel = None", a)
    return zlib.crc32(repr((P,)).encode(), zlib.crc32(src[a:b].encode()))


REV = _kernel_rev()


def takes(B: int, L: int, dn: int, dr: int, r: int) -> bool:
    """Whether a decode call of B rows over L keys fits the kernel (static per bucket): L and r whole 128-row tiles, dn at
    most two tiles, dr at most one, the context's tiles in SBUF (at most 32 tiles: 4,096 keys)."""
    return 0 < B <= 128 and L % P == 0 and 0 < L // P <= 32 and r % P == 0 and dn <= 2 * P and 0 < dr <= P


def attend(q_nope, q_pe, kc, bias, w_uk, w_uv_t, scale: float) -> torch.Tensor:
    """The kernel at the runtime's grid on a Neuron device (or under shell.HOST_SIM), emulate() on the host."""
    from . import shell as SH

    if q_nope.device.type == "cpu" and not SH.HOST_SIM:
        return emulate(q_nope, q_pe, kc, bias, w_uk, w_uv_t, scale)
    if mla_decode_kernel is None:
        raise RuntimeError("the NKI MLA decode kernel needs the nki package (the Neuron venv)")
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    bf = torch.bfloat16
    ident = torch.eye(P, dtype=bf, device=q_nope.device)
    return wrap_nki(mla_decode_kernel)[platform.nki_grid()](
        q_nope=q_nope.to(bf).contiguous(), q_pe=q_pe.to(bf).contiguous(), kc=kc.to(bf).contiguous(),
        bias=bias.float().contiguous(), w_uk=w_uk, w_uv_t=w_uv_t, ident=ident, scale=float(scale), rev=REV)
