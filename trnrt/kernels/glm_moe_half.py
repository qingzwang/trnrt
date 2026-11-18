"""GLM-5.3's decode MoE block (router, top 8, routed experts, shared expert) as ONE kernel: kernels/moe_half.py's kernel
(GLM-5.3-Flash's plain-layer decode MoE half) for GLM-5.3. Its source below is moe_half's NKI section with three changes,
so GLM-5.3-Flash's own kernel binary does not move:
- the router's experts (Er, every expert: the routing is global) apart from the blob's (this rank's, under expert
  parallelism E / ep: models/glm_moe_dsa.py MOE_EP);
- after the top 8 and their weights, the ids as this rank's local experts (lo, this rank's first expert, a tensor so every
  rank runs one NEFF), a pair routed to the other expert group as local expert 0 with weight 0, as the XLA path maps them
  (glm_moe_dsa._experts);
- GLM-5.3's SwiGLU is not clamped: the caller passes limit 3e38, so the kernel's clamps (min / max against limit) are
  the identity on every finite value.
Its layout and packing (shapes, router_t, offsets) are moe_half's, imported.

Why: at one decode row the XLA MoE block of a GLM-5.3 layer was the fp32 router, its top-k, gather and normalisation,
the EP mapping, the shared expert's two projections and the adds around the experts' kernel, a few hundred instructions
and several kernel calls a layer (profiled on trn2: ~1,100 kernel calls in a 78-layer batch-1 step).
"""

from __future__ import annotations

import zlib

import torch

from . import moe_dedupe as mdd
from . import moe_half as MH
from . import rowiso
from . import shell

P = 128
KTOP = MH.KTOP
PREFIX = "gh_"
BF16, F32 = torch.bfloat16, torch.float32
NO_CLAMP = 3.0e38  # the SwiGLU limit that clamps nothing finite


def register(layer, cfg, npg: int) -> None:
    """The layer's packed decode-only blobs (moe_half's layout at every expert's router), listed in _decode_only."""
    E, H = layer.router.shape
    Is = layer.shared_gate_up.shape[0] // 2
    shell.register_packed(layer, PREFIX, MH.shapes(E, H, Is, npg))


def fill(layer, dtype: torch.dtype, npg: int) -> None:
    """The blobs from the loaded router, its bias and the shared expert (moe_half.fill's arithmetic)."""
    from ..models.quant import dequant

    E, H = layer.router.shape
    Is = layer.shared_gate_up.shape[0] // 2
    ih, C = Is // npg, H // P
    v = shell.packed_views(layer, PREFIX, MH.shapes(E, H, Is, npg))
    rt, eh = MH.router_t(layer.router.data), E // npg
    for p in range(npg):
        v["rt"][p].copy_(rt[:, :, p * eh:(p + 1) * eh])
    sgu, sdn = layer.shared_gate_up_scale, layer.shared_down_scale
    w = dequant(layer.shared_gate_up.data, sgu.data if sgu is not None else None, dtype).to(BF16)  # [2 Is, H]
    d = dequant(layer.shared_down.data, sdn.data if sdn is not None else None, dtype).to(BF16)  # [H, Is]
    for p in range(npg):
        rows = torch.cat([w[p * ih:(p + 1) * ih], w[Is + p * ih:Is + (p + 1) * ih]])  # [2 ih, H]
        v["gu"][p].copy_(rows.view(2 * ih, C, P).permute(2, 1, 0))
        v["dn"][p].copy_(d[:, p * ih:(p + 1) * ih].t())
    v["rb"].copy_(layer.router_bias.data.float().view(1, E))


def emulate(x, topv, topi, blob, lo: int, w_gu_shared, w_dn_shared) -> torch.Tensor:
    """The block in torch: the routing (topv, topi over every expert) mapped to this rank's experts as the kernel maps
    them, the pairs kernel's arithmetic unclamped (moe_dedupe.emulate_pairs, act silu), the shared expert in bf16 where
    torch rounds (F.linear's outputs); routed + shared in bf16 [T, H]."""
    El = blob.shape[0]
    loc = topi - lo
    mine = (loc >= 0) & (loc < El)
    ti = torch.where(mine, loc, torch.zeros_like(loc))
    tv = torch.where(mine, topv, torch.zeros_like(topv))
    routed = mdd.emulate_pairs(x, tv, ti, blob, act=mdd.ACTS["silu"]).to(x.dtype)
    gu = torch.nn.functional.linear(x, w_gu_shared)
    n = gu.shape[-1] // 2
    a = torch.nn.functional.silu(gu[..., :n]) * gu[..., n:]
    return routed + torch.nn.functional.linear(a, w_dn_shared)


# -- the decode kernel (NKI)
try:  # the Neuron venv
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

if nki is not None:
    def half_body(x, bb, bf, wb, lo, Er: int, Is: int, K: int, scale: float, group: int, ring: int, limit: float,
                  rsp: int, o_rt: int, o_gu: int, o_dn: int, o_rb: int, xt_in=None, fin: int = 0):
        """moe_half_kernel's body (the module docstring). x HBM [T, H] as the kernel takes it, or None with xt_in, the
        input already in an SBUF bf16 tile [T, H] (the kernel's fin 2 path hands it the row-isolated copy). Returns
        (rp, sp, i8f, pvb, ch): the kernel's two outputs, then the final ids (fp32 [T, 8], sanitised and clamped), the
        weights (bf16 [T, 8]) and the choice the top 8 read (fp32 [T, ER]: sigmoid + bias, raised to -3e38, as max8 /
        nc_find_index8 consume it; ER = E at rsp 0). fin: the input's rows with a NaN / inf zeroed before the
        transposes (1: shell.finite_or_zero); 0 where the caller isolated them already (xt_in)."""
        f32, bf16, i32, u32 = nl.float32, nl.bfloat16, nl.int32, nl.uint32
        T, H = x.shape if xt_in is None else xt_in.shape
        E = Er  # the router's experts (every expert: the routing is global)
        El = wb.shape[0]  # this rank's experts in the blob (expert parallelism: E / ep)
        C = H // P
        DW = H // 2
        C2 = DW // P
        F = wb.shape[2]
        SB = (F - H - DW) // (2 * C)  # bytes per tile scale: 2 (bf16) or 4 (fp32)
        o_dw, o_sg, o_sd = H, H + DW, H + DW + C * SB
        sdt = nl.bfloat16 if SB == 2 else nl.float32
        Q = group
        N = T * K
        NQ = -(-N // Q)
        NP = NQ * Q  # pairs rounded to whole groups (the extra ones load expert 0, weight 0)
        NR = ring
        PD = NR - 2
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        assert K == KTOP and T <= P and E <= 512 and (npg == 1 or NQ >= 2)
        ih = Is // npg
        J = 2 * ih
        Eh = E // npg
        rsp_on = rsp != 0 and npg == 2
        ER = Eh if rsp_on else E  # the experts this program scores
        e0 = pid * Eh if rsp_on else 0
        o_gup = o_gu + pid * P * C * J  # this program's shared expert rows and columns
        o_dnp = o_dn + pid * ih * H
        # outputs first, the same in both programs' traces
        rp = nl.ndarray((npg, P, T, C), dtype=f32, buffer=nl.shared_hbm)
        sp = nl.ndarray((npg, T, H), dtype=f32, buffer=nl.shared_hbm)
        if rsp_on:  # allocated first, the same in both programs' traces: the peer's sendrecv lands here
            rcv = nl.ndarray((T, 3 * KTOP), dtype=f32, buffer=nl.sbuf)
        # this program's contiguous half of the pair groups (moe_tiles_pairs_v1's split)
        split = npg == 2
        q_lo, q_hi = (0, NQ) if not split else ((0, (NQ + 1) // 2) if pid == 0 else ((NQ + 1) // 2, NQ))
        NL = q_hi - q_lo

        # --- loads: the input, the router's [k, e] copy, this program's shared expert rows and columns, the bias
        if xt_in is None:
            xt = nl.ndarray((T, H), dtype=bf16, buffer=nl.sbuf)
            shell.dma_static(dst=xt, src=x)
        else:  # the normalised input already in SBUF, rows on the partitions
            xt = xt_in
        xs = xt  # what the transposes read: x, or (fin) x with each row's NaN / inf zeroed
        if fin:  # x's rows are the contraction of the identity matmuls below (xc, a^T, the pair sums)
            xz = nl.ndarray((T, H), dtype=f32, buffer=nl.sbuf)
            x16 = nl.ndarray((T, H), dtype=nl.uint16, buffer=nl.sbuf)
            shell.finite_or_zero(xt.view(nl.uint16), x16, xz)
            xs = x16.view(bf16)
        r_sb = nl.ndarray((P, C, ER), dtype=bf16, buffer=nl.sbuf)
        if rsp_on:
            shell.dma_static(dst=r_sb,
                             src=bb.ap(pattern=[[C * Eh, P], [Eh, C], [1, Eh]], offset=o_rt + pid * P * C * Eh))
        else:
            for q in range(npg):
                shell.dma_static(dst=r_sb[:, :, q * Eh:(q + 1) * Eh],
                                 src=bb.ap(pattern=[[C * Eh, P], [Eh, C], [1, Eh]], offset=o_rt + q * P * C * Eh))
        g_sb = nl.ndarray((P, C, J), dtype=bf16, buffer=nl.sbuf)
        shell.dma_static(dst=g_sb, src=bb.ap(pattern=[[C * J, P], [J, C], [1, J]], offset=o_gup))
        d_sb = nl.ndarray((ih, H), dtype=bf16, buffer=nl.sbuf)
        shell.dma_static(dst=d_sb, src=bb.ap(pattern=[[H, ih], [1, H]], offset=o_dnp))
        rb = nl.ndarray((T, ER), dtype=f32, buffer=nl.sbuf)
        shell.dma_static(dst=rb, src=bf.ap(pattern=[[0, T], [1, ER]], offset=o_rb + e0))

        # --- constants: the partition index, the identity [T, T] (rows 0 .. T - 1 of eye), expert ids along e
        ipi = nl.ndarray((P, 1), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ipi, pattern=[[0, 1]], offset=0, channel_multiplier=1)
        ip = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ip, src=ipi)
        jti = nl.ndarray((P, T), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=jti, pattern=[[1, T]], offset=0, channel_multiplier=0)
        eye = nl.ndarray((P, T), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eye, src=jti)
        nisa.tensor_scalar(dst=eye, data=eye, op0=nl.equal, operand0=ip)
        eyeb = nl.ndarray((P, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=eyeb, src=eye)
        ioei = nl.ndarray((P, ER), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=ioei, pattern=[[1, ER]], offset=0, channel_multiplier=0)
        ioe = nl.ndarray((P, ER), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ioe, src=ioei)

        # --- x in the k-major layout: xc[i, c, t] = x[t, c 128 + i], by matmuls with the identity (exact)
        xps = nl.ndarray((P, C, T), dtype=f32, buffer=nl.psum)
        for c in range(C):
            nisa.nc_matmul(dst=xps[:, c, :], stationary=xs[0:T, c * P:(c + 1) * P], moving=eyeb[0:T, 0:T],
                           accumulate=False)
        xc = nl.ndarray((P, C, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xc, src=xps, engine=nisa.vector_engine)

        # --- the router: logits [T, ER], sigmoid, + bias, top 8 (RSPLIT: of this program's ER, then merged), weights
        pl = nl.ndarray((T, ER), dtype=f32, buffer=nl.psum)
        for c in range(C):
            nisa.nc_matmul(dst=pl, stationary=xc[:, c, :], moving=r_sb[:, c, :], accumulate=(c > 0))
        ex = nl.ndarray((T, ER), dtype=f32, buffer=nl.sbuf)
        sc = nl.ndarray((T, ER), dtype=f32, buffer=nl.sbuf)
        shell.sigmoid_exp(dst=sc, src=pl, e=ex)
        ch = nl.ndarray((T, ER), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ch, data1=sc, data2=rb, op=nl.add, engine=nisa.vector_engine)
        nisa.tensor_scalar(dst=ch, data=ch, op0=nl.maximum, operand0=-3.0e38, engine=nisa.vector_engine)
        v8 = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
        nisa.max8(dst=v8, src=ch)
        i8 = nl.ndarray((T, KTOP), dtype=u32, buffer=nl.sbuf)
        nisa.nc_find_index8(dst=i8, data=ch, vals=v8)
        i8f = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=i8f, src=i8)
        nisa.tensor_scalar(dst=i8f, data=i8f, op0=nl.maximum, operand0=0.0, op1=nl.minimum, operand1=float(ER - 1),
                           engine=nisa.vector_engine)
        tv = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)  # the chosen experts' scores: sum_e [e == id] sc[e]
        mk = nl.ndarray((T, ER), dtype=f32, buffer=nl.sbuf)
        for j in range(KTOP):
            nisa.tensor_scalar(dst=mk, data=ioe[0:T, :], op0=nl.equal, operand0=i8f[:, j:j + 1],
                               engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=mk, data1=mk, data2=sc, op=nl.multiply, engine=nisa.vector_engine)
            nisa.tensor_reduce(dst=tv[:, j:j + 1], op=nl.add, data=mk, axis=1)
        if rsp_on:  # this program's 8 candidates (choice, global id, score) swapped with the peer's; the top 8 of 16
            cand = nl.ndarray((T, 3 * KTOP), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cand[:, 0:KTOP], src=v8, engine=nisa.vector_engine)
            nisa.tensor_scalar(dst=cand[:, KTOP:2 * KTOP], data=i8f, op0=nl.add, operand0=float(e0),
                               engine=nisa.vector_engine)
            nisa.tensor_copy(dst=cand[:, 2 * KTOP:3 * KTOP], src=tv, engine=nisa.vector_engine)
            if rsp == 2:
                nisa.sendrecv(src=cand, dst=rcv, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0,
                              dma_engine=nisa.dma_engine.gpsimd_dma)
            else:
                nisa.sendrecv(src=cand, dst=rcv, send_to_rank=1 - pid, recv_from_rank=1 - pid, pipe_id=0)
            mg = nl.ndarray((T, 3, 2 * KTOP), dtype=f32, buffer=nl.sbuf)  # [choice | id | score] x program 0's, 1's
            for k3 in range(3):
                if pid == 0:
                    nisa.tensor_copy(dst=mg[:, k3, 0:KTOP], src=cand[:, k3 * KTOP:(k3 + 1) * KTOP])
                    nisa.tensor_copy(dst=mg[:, k3, KTOP:2 * KTOP], src=rcv[:, k3 * KTOP:(k3 + 1) * KTOP])
                else:
                    nisa.tensor_copy(dst=mg[:, k3, 0:KTOP], src=rcv[:, k3 * KTOP:(k3 + 1) * KTOP])
                    nisa.tensor_copy(dst=mg[:, k3, KTOP:2 * KTOP], src=cand[:, k3 * KTOP:(k3 + 1) * KTOP])
            nisa.max8(dst=v8, src=mg[:, 0, :])
            p8 = nl.ndarray((T, KTOP), dtype=u32, buffer=nl.sbuf)
            nisa.nc_find_index8(dst=p8, data=mg[:, 0, :], vals=v8)
            p8f = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=p8f, src=p8)
            io16i = nl.ndarray((P, 2 * KTOP), dtype=i32, buffer=nl.sbuf)
            nisa.iota(dst=io16i, pattern=[[1, 2 * KTOP]], offset=0, channel_multiplier=0)
            io16 = nl.ndarray((P, 2 * KTOP), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=io16, src=io16i)
            m16 = nl.ndarray((T, 2 * KTOP), dtype=f32, buffer=nl.sbuf)
            g16 = nl.ndarray((T, 2 * KTOP), dtype=f32, buffer=nl.sbuf)
            for j in range(KTOP):
                nisa.tensor_scalar(dst=m16, data=io16[0:T, :], op0=nl.equal, operand0=p8f[:, j:j + 1],
                                   engine=nisa.vector_engine)
                nisa.tensor_tensor(dst=g16, data1=m16, data2=mg[:, 1, :], op=nl.multiply, engine=nisa.vector_engine)
                nisa.tensor_reduce(dst=i8f[:, j:j + 1], op=nl.add, data=g16, axis=1)
                nisa.tensor_tensor(dst=g16, data1=m16, data2=mg[:, 2, :], op=nl.multiply, engine=nisa.vector_engine)
                nisa.tensor_reduce(dst=tv[:, j:j + 1], op=nl.add, data=g16, axis=1)
        ssum = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ssum, op=nl.add, data=tv, axis=1)
        nisa.tensor_scalar(dst=ssum, data=ssum, op0=nl.add, operand0=1e-20, engine=nisa.vector_engine)
        rs = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
        nisa.reciprocal(dst=rs, data=ssum)
        pv = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=pv, data=tv, op0=nl.multiply, operand0=rs, op1=nl.multiply, operand1=scale,
                           engine=nisa.vector_engine)
        pvb = nl.ndarray((T, KTOP), dtype=bf16, buffer=nl.sbuf)  # the graph's topv is bf16
        nisa.tensor_copy(dst=pvb, src=pv)
        if El < E:  # expert parallelism: the ids as this rank's local experts, the other group's pairs as expert 0, weight 0
            lo_sb = nl.ndarray((T, 1), dtype=f32, buffer=nl.sbuf)
            shell.dma_static(dst=lo_sb, src=lo.ap(pattern=[[0, T], [1, 1]], offset=0))
            loc = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=loc, data=i8f, op0=nl.subtract, operand0=lo_sb, engine=nisa.vector_engine)
            mine = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mine, data=loc, op0=nl.greater_equal, operand0=0.0, engine=nisa.vector_engine)
            mhi = nl.ndarray((T, KTOP), dtype=f32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=mhi, data=loc, op0=nl.less, operand0=float(El), engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=mine, data1=mine, data2=mhi, op=nl.multiply, engine=nisa.vector_engine)
            nisa.tensor_tensor(dst=i8f, data1=loc, data2=mine, op=nl.multiply, engine=nisa.vector_engine)
            mineb = nl.ndarray((T, KTOP), dtype=bf16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=mineb, src=mine)
            nisa.tensor_tensor(dst=pvb, data1=pvb, data2=mineb, op=nl.multiply, engine=nisa.vector_engine)
        # the pairs u = t K + k: ids on partition 0 (e_sb), weights on every partition (wb_), by matmuls that sum
        # M[t', t, k] = value[t', k] [t' == t] over t'
        mi = nl.ndarray((T, T, KTOP), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mi, data1=i8f.expand_dim(1).broadcast(1, T),
                           data2=eye[0:T].expand_dim(2).broadcast(2, KTOP), op=nl.multiply)
        mv = nl.ndarray((T, T, KTOP), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=mv, data1=pvb.expand_dim(1).broadcast(1, T),
                           data2=eyeb[0:T].expand_dim(2).broadcast(2, KTOP), op=nl.multiply)
        ones_c = nl.ndarray((P, 1), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=ones_c, value=1.0)
        pe = nl.ndarray((1, N), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pe, stationary=ones_c[0:T, :], moving=mi.flatten_dims(1, 2), accumulate=False)
        e_sb = nl.ndarray((1, NP), dtype=i32, buffer=nl.sbuf)
        nisa.memset(dst=e_sb, value=0)
        nisa.tensor_copy(dst=e_sb[:, 0:N], src=pe)
        ones_b = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        nisa.memset(dst=ones_b, value=1.0)
        pwb = nl.ndarray((P, N), dtype=f32, buffer=nl.psum)
        nisa.nc_matmul(dst=pwb, stationary=ones_b[0:T, :], moving=mv.flatten_dims(1, 2), accumulate=False)
        wv = nl.ndarray((P, NP), dtype=f32, buffer=nl.sbuf)
        nisa.memset(dst=wv, value=0.0)
        nisa.tensor_copy(dst=wv[:, 0:N], src=pwb)

        # --- the shared expert: this program's Is / npg intermediate rows (hybrid._swiglu, bf16 where torch rounds)
        gps = nl.ndarray((T, J), dtype=f32, buffer=nl.psum)
        for c in range(C):
            nisa.nc_matmul(dst=gps, stationary=xc[:, c, :], moving=g_sb[:, c, :], accumulate=(c > 0))
        gub = nl.ndarray((T, J), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=gub, src=gps, engine=nisa.vector_engine)
        gcl = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gcl, data=gub[:, 0:ih], op0=nl.minimum, operand0=limit, engine=nisa.vector_engine)
        ucl = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=ucl, data=gub[:, ih:J], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                           operand1=limit, engine=nisa.vector_engine)
        slf = nl.ndarray((T, ih), dtype=f32, buffer=nl.sbuf)
        nisa.activation(dst=slf, op=nl.silu, data=gcl)
        slb = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=slb, src=slf, engine=nisa.vector_engine)
        ab = nl.ndarray((T, ih), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=ab, data1=slb, data2=ucl, op=nl.multiply, engine=nisa.vector_engine)
        aps = nl.ndarray((ih, T), dtype=f32, buffer=nl.psum)  # a^T: the down projection's stationary
        nisa.nc_matmul(dst=aps, stationary=ab, moving=eyeb[0:T, 0:T], accumulate=False)
        at = nl.ndarray((ih, T), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=at, src=aps, engine=nisa.vector_engine)
        ssb = nl.ndarray((T, H), dtype=f32, buffer=nl.sbuf)
        dps = nl.ndarray((T, 512), dtype=f32, buffer=nl.psum)
        for h0 in range(0, H, 512):
            nisa.nc_matmul(dst=dps, stationary=at, moving=d_sb[:, h0:h0 + 512], accumulate=False)
            nisa.tensor_copy(dst=ssb[:, h0:h0 + 512], src=dps, engine=nisa.vector_engine)
        shell.dma_static(dst=sp.ap(pattern=[[H, T], [1, H]], offset=pid * T * H), src=ssb)

        # --- the routed experts: moe_tiles_pairs_v1's pairs on this program's groups
        dd_i = nl.ndarray((P, P), dtype=i32, buffer=nl.sbuf)
        nisa.iota(dst=dd_i, pattern=[[1, P]], offset=0, channel_multiplier=0)
        dd = nl.ndarray((P, P), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=dd, src=dd_i)
        nisa.tensor_scalar(dst=dd, data=dd, op0=nl.subtract, operand0=ip)
        fd = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        f1 = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        f2 = nl.ndarray((P, P), dtype=bf16, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=fd, data=dd, op0=nl.equal, operand0=0.0)
        nisa.tensor_scalar(dst=f1, data=dd, op0=nl.equal, operand0=64.0)
        nisa.tensor_scalar(dst=f2, data=dd, op0=nl.equal, operand0=-64.0)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f1, op=nl.add)
        nisa.tensor_tensor(dst=fd, data1=fd, data2=f2, op=nl.add)
        mh = nl.ndarray((P, 2), dtype=f32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=mh[:, 0:1], data=ip, op0=nl.less, operand0=64.0)
        nisa.tensor_scalar(dst=mh[:, 1:2], data=ip, op0=nl.greater_equal, operand0=64.0)
        acc = nl.ndarray((P, T, C), dtype=f32, buffer=nl.sbuf)  # [p, t, g]: g = 2 c' + half
        nisa.memset(dst=acc, value=0.0)
        bufs = (nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf), nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf),
                nl.ndarray((P, Q, F), dtype=nl.uint8, buffer=nl.sbuf))[:NR]
        sg_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf))
        sd_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf))
        q_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((P, Q, C), dtype=f32, buffer=nl.sbuf))
        g_r = (nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf))
        mm_r = (nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf))
        sl_r = (nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf))
        au_r = (nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf),
                nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf))
        a2_r = (nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf), nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf),
                nl.ndarray((P, Q, 2), dtype=bf16, buffer=nl.sbuf))
        y_r = (nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.sbuf), nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.sbuf),
               nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.sbuf))
        pg_r = (nl.ndarray((P, Q, C), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, C), dtype=f32, buffer=nl.psum),
                nl.ndarray((P, Q, C), dtype=f32, buffer=nl.psum))
        pf_r = (nl.ndarray((P, Q, 2), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, 2), dtype=f32, buffer=nl.psum),
                nl.ndarray((P, Q, 2), dtype=f32, buffer=nl.psum))
        pd_r = (nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.psum), nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.psum),
                nl.ndarray((P, Q, C2, 2), dtype=f32, buffer=nl.psum))
        # step i: the loads of group i + PD, stage A of group i, stages B and C of group i - 1 (the pairs kernel's)
        for it in range(-PD, NL + 1):
            gl = it + PD
            if 0 <= gl < NL:
                bq = bufs[gl % NR]
                for s in range(Q):
                    e = e_sb.ap(pattern=[[NP, 1], [1, 1]], offset=(q_lo + gl) * Q + s)
                    nisa.dma_copy(dst=bq[:, s, :], src=wb.select(0, e))
            if 0 <= it < NL:  # stage A: gate_up, scales
                gq = it
                bq = bufs[gq % NR]
                pg = pg_r[gq % 3]
                for s in range(Q):
                    t = min(T - 1, ((q_lo + gq) * Q + s) // K)
                    wg = bq[:, s, 0:H].view(nl.float8_e4m3)
                    for c in range(C):
                        nisa.nc_matmul(dst=pg[:, s, c:c + 1], stationary=wg[:, c * P:(c + 1) * P],
                                       moving=xc[:, c, t:t + 1], accumulate=False)
                nisa.activation(dst=sg_r[gq % 3], op=nl.copy, data=bq[:, :, o_sg:o_sd].view(sdt))
                nisa.activation(dst=sd_r[gq % 3], op=nl.copy, data=bq[:, :, o_sd:F].view(sdt))
                q = q_r[gq % 3]
                nisa.tensor_tensor(dst=q, data1=pg, data2=sg_r[gq % 3], op=nl.multiply)
                g = g_r[gq % 3]
                nisa.tensor_reduce(dst=g, op=nl.add, data=q, axis=2)
                nisa.tensor_tensor(dst=mm_r[gq % 3], data1=g.expand_dim(2).broadcast(2, 2),
                                   data2=mh.expand_dim(1).broadcast(1, Q), op=nl.multiply)
            if 1 <= it <= NL:  # stages B and C of group it - 1
                gq = it - 1
                pf = pf_r[gq % 3]
                nisa.nc_matmul(dst=pf.flatten_dims(1, 2), stationary=fd, moving=mm_r[gq % 3].flatten_dims(1, 2),
                               accumulate=False)
                sl = sl_r[gq % 3]
                au = au_r[gq % 3]
                gc = nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf)  # the clamped SwiGLU (ACTS silu_clamp)
                nisa.tensor_scalar(dst=gc, data=pf[:, :, 0], op0=nl.minimum, operand0=limit)
                uc = nl.ndarray((P, Q), dtype=f32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=uc, data=pf[:, :, 1], op0=nl.maximum, operand0=-limit, op1=nl.minimum,
                                   operand1=limit)
                nisa.activation(dst=sl, op=nl.silu, data=gc)
                nisa.tensor_tensor(dst=au, data1=sl, data2=uc, op=nl.multiply)
                nisa.tensor_tensor(dst=a2_r[gq % 3], data1=au.expand_dim(2).broadcast(2, 2),
                                   data2=mh.expand_dim(1).broadcast(1, Q), op=nl.multiply)
                bq = bufs[gq % NR]
                pd = pd_r[gq % 3]
                for s in range(Q):
                    wd = bq[:, s, o_dw:o_sg].view(nl.float8_e4m3)
                    for c in range(C2):
                        nisa.nc_matmul(dst=pd[:, s, c, :], stationary=wd[:, c * P:(c + 1) * P],
                                       moving=a2_r[gq % 3][:, s, :], accumulate=False)
                y = y_r[gq % 3]
                nisa.tensor_tensor(dst=y.flatten_dims(2, 3), data1=pd.flatten_dims(2, 3), data2=sd_r[gq % 3],
                                   op=nl.multiply)
                for s in range(Q):
                    u = (q_lo + gq) * Q + s
                    if u < N:
                        t = u // K
                        nisa.scalar_tensor_tensor(dst=acc[:, t, :], data=y[:, s].flatten_dims(1, 2), op0=nl.multiply,
                                                  operand0=wv[:, u:u + 1], op1=nl.add, operand1=acc[:, t, :])
        shell.dma_static(dst=rp.ap(pattern=[[T * C, P], [C, T], [1, C]], offset=pid * P * T * C), src=acc)
        if npg > 1:  # both programs' partials written before either ends
            nisa.core_barrier(data=rp, cores=(0, 1))
            nisa.core_barrier(data=sp, cores=(0, 1))
        return rp, sp, i8f, pvb, ch

    @nki.jit
    def glm_moe_half_kernel(x, bb, bf, wb, lo, Er: int, Is: int, K: int, scale: float, group: int, ring: int,
                            limit: float, rsp: int, o_rt: int, o_gu: int, o_dn: int, o_rb: int, rev: int, fin: int = 1):
        """x bf16 [T, H] (the MoE half's normalised input, T <= 128); bb / bf: the layer's packed bf16 / fp32
        decode-only blobs (shapes()); wb uint8 [E, 128, F]: the layer's w_blob (moe_dedupe's tile layout); Is the
        shared expert's intermediate rows per rank; K = 8; scale the routed scaling factor; group / ring: the pairs
        kernel's (moe_dedupe.group_size(1), RING); limit the SwiGLU clamp; rsp: RSPLIT; o_rt / o_gu / o_dn / o_rb: the
        packed tensors' offsets (offsets(), computed on the host: no dict in the kernel body, which the trn1 tracer
        refused); rev: this module's kernel source revision; fin (default 1): each row of x with a NaN / inf
        zeroed first, so the identity matmuls that contract over x's rows cannot carry it into other rows; 0 none;
        2 rowiso's (the served form, rowiso.FIN): a row with any NaN / inf +0 whole (row_keep, zero_rows).
        Returns (rp fp32 [npg, 128, T, C], sp fp32 [npg, T, H]): each program's partial of the routed experts (the
        pairs kernel's [p, t, g] layout) and of the shared expert."""
        xt_in = None
        if fin == 2:  # rowiso: a row with any NaN / inf +0 whole (its flag read on 128 lanes, then one pass over
            T, H = x.shape  # [T, H]), handed to the body as its SBUF input (half_body's xt_in)
            xt = nl.ndarray((T, H), dtype=nl.bfloat16, buffer=nl.sbuf)
            shell.dma_static(dst=xt, src=x)
            keep = rowiso.row_keep(x, T, H)
            x16 = nl.ndarray((T, H), dtype=nl.uint16, buffer=nl.sbuf)
            rowiso.zero_rows(xt.view(nl.uint16), x16, keep)
            xt_in = x16.view(nl.bfloat16)
        rp, sp, _, _, _ = half_body(x, bb, bf, wb, lo, Er, Is, K, scale, group, ring, limit, rsp, o_rt, o_gu, o_dn,
                                    o_rb, xt_in=xt_in, fin=fin if fin < 2 else 0)
        return rp, sp
else:
    glm_moe_half_kernel = None




REV_K = zlib.crc32(repr((shell.kernel_rev(__file__, "# -- the decode kernel (NKI)", "    glm_moe_half_kernel = None"),
                         shell.REV, rowiso.REV, rowiso.REV_ROWS, KTOP, P)).encode())


def takes(layer, T: int) -> bool:
    """A decode call of T rows of a packed MoE layer with a shared expert and the router's bias (moe_half's limits)."""
    return (0 < T <= P and getattr(layer, "moe_blob", False) and getattr(layer, "gh_blob_bf16", None) is not None
            and getattr(layer, "router_bias", None) is not None)


def decode_half(model, layer, x: torch.Tensor) -> torch.Tensor:
    """The MoE block's output [T, H] before its all-reduce: the kernel's partials added and rounded as moe_half's
    caller adds them, routed + shared."""
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

    from .. import platform

    if glm_moe_half_kernel is None:
        raise RuntimeError("the NKI GLM MoE half kernel needs the nki package (the Neuron venv)")
    cfg = model.cfg
    npg = platform.nki_grid()
    E, H = layer.router.shape
    Is = int(layer.shared_gate_up.shape[0] // 2)
    lo = model.ep_lo_t.to(torch.float32).view(1, 1) if getattr(model, "moe_ep", 1) > 1 else torch.zeros(
        1, 1, dtype=torch.float32, device=x.device)
    rp, sp = wrap_nki(glm_moe_half_kernel)[npg](
        x=x.to(BF16).contiguous(), bb=layer.gh_blob_bf16, bf=layer.gh_blob_f32, wb=layer.w_blob, lo=lo, Er=int(E),
        Is=Is, K=int(cfg.num_experts_per_tok), scale=float(cfg.routed_scaling_factor), group=mdd.group_size(1),
        ring=mdd.RING, limit=NO_CLAMP, rsp=0, rev=REV_K, fin=rowiso.FIN, **MH.offsets(E, H, Is, npg))
    routed = mdd.from_pairs(shell.add_partials(rp, BF16), x.dtype)
    return routed + shell.add_partials(sp, BF16).to(x.dtype)
