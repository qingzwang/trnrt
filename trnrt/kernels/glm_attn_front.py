"""GLM-5.3's decode attention front for one head a rank (tp 64), as ONE kernel: from the layer's residual h to the
head's query (q_nope, the rotated q_pe), the token's cache row [c | k_pe] and, on a full layer, its indexer key
(models/glm_moe_dsa.py _project and _indexer's key, with decode's tp column shards of q_a / kv_a, _w_a_sharded).

Why: at one decode row the XLA front of a layer was the input norm, the w_a shard's projection call, the one-hot
placement and its all-reduce, the two latent norms, q_b's and idx_wk's projection calls, three RoPEs and the key's
LayerNorm: four kernel calls and an XLA all-reduce, with the small DMAs and semaphore waits between them (profiled on
trn2, GLM-5.3 batch-1 decode, 78 layers, rank 0 of 64: 6.5 ms of XLA-only instructions, 9,900 DMA_DIRECT2D of ~0.66 us,
and 6.6 ms idle in a 24.5 ms step).

The kernel (rows T on the partitions where a row is short, a row's H on 128 partitions where it is contracted):
- both programs: each row of h DMA'd as [H / 128, 128] and moved onto the partitions by one PE transpose (exact
  through fp32 PSUM), its sum of squares, rsqrt broadcast to the partitions (shell.rsqrt_bcast), then x^T =
  bf16(bf16(h r) g) as rms_norm rounds it;
- program 0: the rank's w_a columns in column form, local^T [n, T] = W^T x^T (rounded to bf16 as the projection
  kernel rounds), placed at the rank's offset by a matmul against the one-hot place [n, W] (one 1 a column: exact),
  and summed over the world by cc_world's all-reduce (every column has one non-zero term: exact);
- program 1 (the last program), under the all-reduce: the indexer key, x idx_wk^T rounded to bf16, the LayerNorm in
  fp32 (eps 1e-6, with bias) rounded once, RoPE on its first dr dims;
- program 0, after the sum: q_resid = rms_norm(a[:, :ql]) on the partitions (one transpose a row), q = q_resid q_b^T,
  q_pe and k_pe rotated (interleaved pairs read with a stride of 2, rope()'s bf16 roundings), c = rms_norm(a[:, ql:ql +
  r]) in row form; with qres, q_resid back in row form for the indexer's query (a bucket of more than index_topk keys).

Numerics: emulate() is this arithmetic in torch (models' rms_norm, rope, F.layer_norm and the projection kernel's
emulation). The norms' sums are the PE's and the vector engine's, not torch's order, so a rounding can differ in the
last bf16 place.
"""

from __future__ import annotations

import os
import zlib

import torch
import torch.nn.functional as F

from . import cc_world as CW
from . import gemv_tiled as GT
from . import shell

P = 128
BF16, F32 = torch.bfloat16, torch.float32
LN_EPS = 1e-6  # the indexer key's LayerNorm (GlmMoeDsaIndexer)



def host() -> bool:
    """TRNRT_GLM_FRONT_HOST=1: the model takes this kernel's path on the host too, through emulate() (tests; an
    environment variable so that an engine's rank processes see it)."""
    return os.environ.get("TRNRT_GLM_FRONT_HOST", "0") == "1"


def emulate(h, gin, wa, lo: int, place_w: int, gq, gkv, wqb, wik, knw, knb, cos, sin, eps: float, eps_q: float,
            ql: int, r: int, dn: int, dr: int, reduce=None):
    """The kernel's arithmetic in torch: h [T, H] bf16, the norms' weights gin [H], gq [ql], gkv [r], the projection
    blobs wa [P, C, n] (this rank's w_a columns from lo), wqb [P, CQ, dn + dr], wik [P, C, Di] or None (a shared
    layer), the key's LayerNorm knw / knb [Di], cos / sin [T, dr / 2]. place_w: the latent row's width (ql + r + dr);
    reduce: the all-reduce of the placed row over the ranks (None: one rank). Returns (q_nope [T, dn], q_pe [T, dr], kv [T, r + dr], ki [T, Di] or None, q_resid [T, ql])."""
    from ..models.decoder import rms_norm
    from ..models.glm_moe_dsa import rope

    T = h.shape[0]
    x = rms_norm(h, gin, eps)
    local = GT.emulate(x, wa)
    a = torch.zeros(T, place_w, dtype=h.dtype)
    n = min(local.shape[1], place_w - lo)
    a[:, lo:lo + n] = local[:, :n]
    if reduce is not None:
        a = reduce(a)
    q_resid = rms_norm(a[:, :ql], gq, eps_q)
    q = GT.emulate(q_resid, wqb)
    c = rms_norm(a[:, ql:ql + r], gkv, eps_q)
    kv = torch.cat([c, rope(a[:, ql + r:ql + r + dr], cos, sin)], dim=-1)
    ki = None
    if wik is not None:
        k = GT.emulate(x, wik)
        k = F.layer_norm(k.float(), (k.shape[-1],), knw.float(), knb.float(), LN_EPS).to(h.dtype)
        ki = torch.cat([rope(k[:, :dr], cos, sin), k[:, dr:]], dim=-1)
    return q[:, :dn], rope(q[:, dn:dn + dr], cos, sin), kv, ki, q_resid


try:  # the Neuron venv; elsewhere (CPU tests, a laptop) there is no kernel to build
    import nki
    import nki.isa as nisa
    import nki.language as nl
except ImportError:
    nki = None

# -- the kernel (its span is what REV hashes)
if nki is not None:
    from .cc_world import ar_world_alloc, ar_world_issue, ar_world_zero1
    from .shell import rsqrt_bcast

    def _sb(shape, dtype):
        return nl.ndarray(shape, dtype=dtype, buffer=nl.sbuf)

    def _ps(shape):
        return nl.ndarray(shape, dtype=nl.float32, buffer=nl.psum)

    def _load_w(dst, blob, C: int, N: int):
        """A projection blob [P, C, N] into SBUF, in two pieces on the scalar and sync engines' hardware DGE queues
        (gemv_tiled's split)."""
        c1 = -(-C // 2)
        nisa.dma_copy(dst=dst[:, 0:c1, :], src=blob.ap(pattern=[[C * N, P], [N, c1], [1, N]], offset=0),
                      engine=nisa.engine.scalar, dge_mode=nisa.dge_mode.hwdge)
        if c1 < C:
            nisa.dma_copy(dst=dst[:, c1:C, :], src=blob.ap(pattern=[[C * N, P], [N, C - c1], [1, N]], offset=c1 * N),
                          engine=nisa.engine.sync, dge_mode=nisa.dge_mode.hwdge)

    def _rsqrt_rows(a, T: int):
        """rsqrt(a) of a [T, 1] fp32: the scalar engine's rsqrt, then one Newton step (shell.rsqrt_bcast's)."""
        VE = nisa.vector_engine
        r0 = _sb((T, 1), nl.float32)
        nisa.activation(dst=r0, op=nl.rsqrt, data=a)
        nt = _sb((T, 1), nl.float32)
        nisa.tensor_tensor(dst=nt, data1=r0, data2=r0, op=nl.multiply, engine=VE)
        nisa.tensor_tensor(dst=nt, data1=nt, data2=a, op=nl.multiply, engine=VE)
        nisa.tensor_scalar(dst=nt, data=nt, op0=nl.multiply, operand0=-0.5, op1=nl.add, operand1=1.5, engine=VE)
        rr = _sb((T, 1), nl.float32)
        nisa.tensor_tensor(dst=rr, data1=r0, data2=nt, op=nl.multiply, engine=VE)
        return rr

    def _rope(dst, do: int, src, so: int, sw: int, cst, T: int, hd: int):
        """dst[:, do:do + 2 hd] = rope() of src's interleaved pairs at so (src a [T, sw] bf16 tile): [x1 cos - x2 sin,
        x2 cos + x1 sin], each product and the sum rounded to bf16 (rope()'s ops in the model dtype)."""
        VE = nisa.vector_engine
        x1 = src.ap(pattern=[[sw, T], [2, hd]], offset=so)
        x2 = src.ap(pattern=[[sw, T], [2, hd]], offset=so + 1)
        cos = cst[:, 0:hd]
        sin = cst[:, hd:2 * hd]
        t1 = _sb((T, hd), nl.bfloat16)
        t2 = _sb((T, hd), nl.bfloat16)
        nisa.tensor_tensor(dst=t1, data1=x1, data2=cos, op=nl.multiply, engine=VE)
        nisa.tensor_tensor(dst=t2, data1=x2, data2=sin, op=nl.multiply, engine=VE)
        nisa.tensor_tensor(dst=dst[:, do:do + hd], data1=t1, data2=t2, op=nl.subtract, engine=VE)
        t3 = _sb((T, hd), nl.bfloat16)
        t4 = _sb((T, hd), nl.bfloat16)
        nisa.tensor_tensor(dst=t3, data1=x2, data2=cos, op=nl.multiply, engine=VE)
        nisa.tensor_tensor(dst=t4, data1=x1, data2=sin, op=nl.multiply, engine=VE)
        nisa.tensor_tensor(dst=dst[:, do + hd:do + 2 * hd], data1=t3, data2=t4, op=nl.add, engine=VE)

    def _norm_t(src, ws, wd: int, I, ones_c, ones_r, g, gd: int, T: int, eps: float, xT):
        """xT [P, wd / P, T] = rms_norm of the rows src[b, ws:ws + wd] (src in HBM, rows gd wide, each read as [wd / P,
        P]) with weight g [1, wd] (HBM), on the partitions: one PE transpose a row, the sum of squares broadcast as
        rsqrt, then bf16(bf16(x r) g)."""
        VE = nisa.vector_engine
        C = wd // P
        g48 = _sb((C, P), nl.bfloat16)
        nisa.dma_copy(dst=g48, src=g.ap(pattern=[[P, C], [1, P]], offset=0))
        pg = _ps((P, C))
        nisa.nc_matmul(dst=pg, stationary=g48, moving=I[0:C, 0:C], accumulate=False)
        gT = _sb((P, C), nl.bfloat16)
        nisa.tensor_copy(dst=gT, src=pg, engine=VE)
        hT = _sb((P, T, C), nl.bfloat16)
        sq = _sb((P, T), nl.float32)
        for b in range(T):
            hr = _sb((C, P), nl.bfloat16)
            nisa.dma_copy(dst=hr, src=src.ap(pattern=[[P, C], [1, P]], offset=b * gd + ws))
            ph = _ps((P, C))
            nisa.nc_matmul(dst=ph, stationary=hr, moving=I[0:C, 0:C], accumulate=False)
            nisa.tensor_copy(dst=hT[:, b, :], src=ph, engine=VE)
            junk = _sb((P, C), nl.float32)
            nisa.activation_reduce(dst=junk, op=nl.square, data=hT[:, b, :], reduce_op=nl.add,
                                   reduce_res=sq[:, b:b + 1])
        rb = rsqrt_bcast(ones_c, ones_r, sq, T, 1.0 / wd, eps)
        for b in range(T):
            tb = _sb((P, C), nl.bfloat16)
            nisa.tensor_scalar(dst=tb, data=hT[:, b, :], op0=nl.multiply, operand0=rb[:, b:b + 1], engine=VE)
            nisa.tensor_tensor(dst=xT[:, :, b], data1=tb, data2=gT, op=nl.multiply, engine=VE)

    @nki.jit
    def glm_attn_front_kernel(h, gin, wa, place, gq, gkv, wqb, wik, kn, cs, ident, eps: float, eps_q: float, ql: int,
                              r: int, dn: int, dr: int, world: int, cc: int, full: int, qres: int, rev: int):
        """h bf16 [T, H] (the layer's residual), gin [1, H] / gq [1, ql] / gkv [1, r] the norms' weights, wa [P, C, n]
        this rank's w_a columns (gemv_tiled's blob), place [n, W] their one-hot placement in the latent row, wqb [P,
        ql / P, dn + dr] q_b's blob (one head), wik [P, C, Di] idx_wk's (any blob on a shared layer: full 0), kn fp32
        [2, Di] the key's LayerNorm weight and bias, cs bf16 [T, dr] (cos | sin), ident bf16 [P, P]; world the
        all-reduce's ranks (cc 0: no collective, world 1); qres 1: q_resid out too. Returns bf16 (q_nope [T, dn],
        q_pe [T, dr], kv [T, r + dr], ki [T, Di], q_resid [T, ql]); ki is written on a full layer only, q_resid
        under qres only."""
        BF, FP = nl.bfloat16, nl.float32
        VE = nisa.vector_engine
        T, H = h.shape
        C, n = wa.shape[1], wa.shape[2]
        W = place.shape[1]
        CQ, NQ = wqb.shape[1], wqb.shape[2]
        Di = wik.shape[2]
        hd = dr // 2
        npg, pid = (nl.num_programs(axes=0), nl.program_id(axis=0)) if nl.program_ndim() != 0 else (1, 0)
        qn = nl.ndarray((T, dn), dtype=BF, buffer=nl.shared_hbm)
        qp = nl.ndarray((T, dr), dtype=BF, buffer=nl.shared_hbm)
        kv = nl.ndarray((T, r + dr), dtype=BF, buffer=nl.shared_hbm)
        ki = nl.ndarray((T, Di), dtype=BF, buffer=nl.shared_hbm)
        qr = nl.ndarray((T, ql), dtype=BF, buffer=nl.shared_hbm)
        Y, S0, S1, Z1 = ar_world_alloc(T, W, BF)
        ar_world_zero1(S1, T, W, pid)
        I = _sb((P, P), BF)
        nisa.dma_copy(dst=I, src=ident)
        ones_c = _sb((P, 1), FP)
        nisa.memset(dst=ones_c, value=1.0)
        ones_r = _sb((1, P), FP)
        nisa.memset(dst=ones_r, value=1.0)
        cst = _sb((T, dr), BF)
        nisa.dma_copy(dst=cst, src=cs)
        # x^T [P, C, T]: rms_norm(h) on the partitions (both programs: w_a's and the key's input)
        xT = _sb((P, C, T), BF)
        _norm_t(h, 0, H, I, ones_c, ones_r, gin, H, T, eps, xT)
        wqt = _sb((P, CQ, NQ), BF)
        if pid == 0:
            wat = _sb((P, C, n), BF)
            _load_w(wat, wa, C, n)
            _load_w(wqt, wqb, CQ, NQ)  # q_b's weights, early: they land under the all-reduce
            pl = _ps((n, T))
            for c in range(C):
                nisa.nc_matmul(dst=pl, stationary=wat[:, c, :], moving=xT[:, c, :], accumulate=(c > 0))
            lt = _sb((n, T), BF)
            nisa.tensor_copy(dst=lt, src=pl, engine=VE)
            plc = _sb((n, W), BF)
            nisa.dma_copy(dst=plc, src=place)
            row = _sb((T, W), BF)
            for u in range(-(-W // 512)):
                u0 = u * 512
                nw = min(512, W - u0)
                pp = _ps((T, nw))
                nisa.nc_matmul(dst=pp, stationary=lt, moving=plc[:, u0:u0 + nw], accumulate=False)
                nisa.tensor_copy(dst=row[:, u0:u0 + nw], src=pp, engine=VE)
            nisa.dma_copy(dst=S0, src=row, dge_mode=nisa.dge_mode.none)
        ar_world_issue(Y, S0, S1, Z1, world, pid, cc=cc, npg=npg)
        # the indexer key, on the last program, under the all-reduce
        if full:
            if pid == npg - 1:
                wkt = _sb((P, C, Di), BF)
                _load_w(wkt, wik, C, Di)
                pk = _ps((T, Di))
                for c in range(C):
                    nisa.nc_matmul(dst=pk, stationary=xT[:, c, :], moving=wkt[:, c, :], accumulate=(c > 0))
                kb = _sb((T, Di), BF)
                nisa.tensor_copy(dst=kb, src=pk, engine=VE)
                kf = _sb((T, Di), FP)
                nisa.tensor_copy(dst=kf, src=kb, engine=VE)
                s1 = _sb((T, 1), FP)
                nisa.tensor_reduce(dst=s1, op=nl.add, data=kf, axis=1)
                nm = _sb((T, 1), FP)
                nisa.tensor_scalar(dst=nm, data=s1, op0=nl.multiply, operand0=-1.0 / Di, engine=VE)
                d = _sb((T, Di), FP)
                nisa.tensor_scalar(dst=d, data=kf, op0=nl.add, operand0=nm, engine=VE)
                junk = _sb((T, Di), FP)
                s2 = _sb((T, 1), FP)
                nisa.activation_reduce(dst=junk, op=nl.square, data=d, reduce_op=nl.add, reduce_res=s2)
                va = _sb((T, 1), FP)
                nisa.tensor_scalar(dst=va, data=s2, op0=nl.multiply, operand0=1.0 / Di, op1=nl.add,
                                   operand1=LN_EPS, engine=VE)
                rr = _rsqrt_rows(va, T)
                knt = _sb((T, 2, Di), FP)
                nisa.dma_copy(dst=knt, src=kn.ap(pattern=[[0, T], [Di, 2], [1, Di]], offset=0))
                y = _sb((T, Di), FP)
                nisa.tensor_scalar(dst=y, data=d, op0=nl.multiply, operand0=rr, engine=VE)
                nisa.tensor_tensor(dst=y, data1=y, data2=knt[:, 0, :], op=nl.multiply, engine=VE)
                yb = _sb((T, Di), BF)
                nisa.tensor_tensor(dst=yb, data1=y, data2=knt[:, 1, :], op=nl.add, engine=VE)
                ko = _sb((T, Di), BF)
                _rope(ko, 0, yb, 0, Di, cst, T, hd)
                nisa.tensor_copy(dst=ko[:, dr:Di], src=yb[:, dr:Di], engine=VE)
                nisa.dma_copy(dst=ki, src=ko)
        if npg > 1:
            nisa.core_barrier(data=Y, cores=(0, 1))  # program 0's sum landed (cc_world's ar_world_close)
        if pid == 0:
            # q = rms_norm(a[:, :ql]) q_b^T, q_pe rotated
            qT = _sb((P, CQ, T), BF)
            _norm_t(Y, 0, ql, I, ones_c, ones_r, gq, W, T, eps_q, qT)
            pq = _ps((T, NQ))
            for c in range(CQ):
                nisa.nc_matmul(dst=pq, stationary=qT[:, c, :], moving=wqt[:, c, :], accumulate=(c > 0))
            qb = _sb((T, NQ), BF)
            nisa.tensor_copy(dst=qb, src=pq, engine=VE)
            nisa.dma_copy(dst=qn, src=qb[:, 0:dn])
            qpo = _sb((T, dr), BF)
            _rope(qpo, 0, qb, dn, NQ, cst, T, hd)
            nisa.dma_copy(dst=qp, src=qpo)
            # the cache row: rms_norm(a[:, ql:ql + r]) and the rotated k_pe, in row form
            ak = _sb((T, r + dr), BF)
            nisa.dma_copy(dst=ak, src=Y.ap(pattern=[[W, T], [1, r + dr]], offset=ql))
            junk2 = _sb((T, r), FP)
            sc = _sb((T, 1), FP)
            nisa.activation_reduce(dst=junk2, op=nl.square, data=ak[:, 0:r], reduce_op=nl.add, reduce_res=sc)
            vc = _sb((T, 1), FP)
            nisa.tensor_scalar(dst=vc, data=sc, op0=nl.multiply, operand0=1.0 / r, op1=nl.add, operand1=eps_q,
                               engine=VE)
            rc = _rsqrt_rows(vc, T)
            gk = _sb((T, r), BF)
            nisa.dma_copy(dst=gk, src=gkv.ap(pattern=[[0, T], [1, r]], offset=0))
            tc = _sb((T, r), BF)
            nisa.tensor_scalar(dst=tc, data=ak[:, 0:r], op0=nl.multiply, operand0=rc, engine=VE)
            kvo = _sb((T, r + dr), BF)
            nisa.tensor_tensor(dst=kvo[:, 0:r], data1=tc, data2=gk, op=nl.multiply, engine=VE)
            _rope(kvo, r, ak, r, r + dr, cst, T, hd)
            nisa.dma_copy(dst=kv, src=kvo)
            if qres:  # q_resid in row form: one PE transpose a row back
                for b in range(T):
                    pr = _ps((CQ, P))
                    nisa.nc_matmul(dst=pr, stationary=qT[:, :, b], moving=I, accumulate=False)
                    rq = _sb((CQ, P), BF)
                    nisa.tensor_copy(dst=rq, src=pr, engine=VE)
                    nisa.dma_copy(dst=qr.ap(pattern=[[P, CQ], [1, P]], offset=b * ql), src=rq)
        return qn, qp, kv, ki, qr
else:
    glm_attn_front_kernel = None


REV = zlib.crc32(repr((shell.kernel_rev(__file__, "# -- the kernel (its span is what REV hashes)",
                                        "    glm_attn_front_kernel = None"), CW.REV, shell.REV, P, LN_EPS)).encode())


def takes(model, layer, T: int) -> bool:
    """A decode call of T rows the kernel takes: one head a rank, decode's w_a shard and its placement, q_b's (and on a
    full layer idx_wk's) blob, whole 128-row tiles of H and ql."""
    m = layer.spec.mla
    full = not m.index_shared
    return (0 < T <= 16 and layer.nh == 1 and getattr(layer, "dt_w_a_shard", None) is not None
            and getattr(model, "w_a_place", None) is not None and getattr(layer, "dt_q_b", None) is not None
            and (not full or getattr(layer, "dt_idx_wk", None) is not None)
            and model.cfg.hidden_size % P == 0 and m.q_lora_rank % P == 0 and m.q_lora_rank // P <= P)


def front(model, layer, h: torch.Tensor, positions: torch.Tensor, qres: bool):
    """(q_nope [T, 1, dn], q_pe [T, 1, dr], kv [T, r + dr], ki [T, Di] or None, q_resid [T, ql] or None) of the layer
    for the residual h [T, H]: the kernel at the runtime's grid on a Neuron device, emulate() on the host (host())."""
    m = layer.spec.mla
    cfg = model.cfg
    T = h.shape[0]
    dn, dr, r, ql = m.qk_nope_head_dim, m.qk_rope_head_dim, m.kv_lora_rank, m.q_lora_rank
    full = not m.index_shared
    cos, sin = model._rope(positions)
    wik = layer.dt_idx_wk if full else None
    if h.device.type == "cpu":
        lo = int(model.w_a_lo_t)
        qn, qp, kv, ki, q_res = emulate(h, layer.in_norm, layer.dt_w_a_shard, lo, layer.w_a_cols, layer.q_a_norm,
                                        layer.kv_a_norm, layer.dt_q_b, wik, getattr(layer, "idx_knorm_w", None),
                                        getattr(layer, "idx_knorm_b", None), cos, sin, cfg.rms_norm_eps, m.norm_eps,
                                        ql, r, dn, dr, reduce=model._all_reduce)
    else:
        if glm_attn_front_kernel is None:
            raise RuntimeError("the NKI GLM attention front kernel needs the nki package (the Neuron venv)")
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki

        from .. import platform

        dev = h.device
        kn = (torch.stack([layer.idx_knorm_w.float(), layer.idx_knorm_b.float()]) if full
              else torch.zeros(2, 1, dtype=F32, device=dev))
        world = model.tp_size
        qn, qp, kv, ki, q_res = wrap_nki(glm_attn_front_kernel)[platform.nki_grid()](
            h=h.to(BF16).contiguous(), gin=layer.in_norm.to(BF16).view(1, -1), wa=layer.dt_w_a_shard,
            place=model.w_a_place, gq=layer.q_a_norm.to(BF16).view(1, -1), gkv=layer.kv_a_norm.to(BF16).view(1, -1),
            wqb=layer.dt_q_b, wik=wik if full else layer.dt_q_b, kn=kn,
            cs=torch.cat([cos, sin], dim=-1).to(BF16).contiguous(), ident=torch.eye(P, dtype=BF16, device=dev),
            eps=float(cfg.rms_norm_eps), eps_q=float(m.norm_eps), ql=ql, r=r, dn=dn, dr=dr, world=world,
            cc=int(world > 1), full=int(full), qres=int(qres), rev=REV)
        if not full:
            ki = None
    return qn.view(T, 1, dn), qp.view(T, 1, dr), kv, ki, q_res if qres else None
