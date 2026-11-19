"""Decode context parallelism (DCP) of GLM-5.3's attention (models/glm_moe_dsa.py): each of the cp ranks of a group
holds 1 / cp of every sequence's latent cache rows [c | k_pe] and indexer keys, so a context of L tokens costs L / cp
rows a rank instead of L (93 KB a token replicated at GLM-5.3's shapes: 23 GiB a rank at 256K, past a NeuronCore's
24 GiB with the weights). The reference (XLA) form, for every batch form (decode, verify, prefill chunk).

Layout: token j of a page lives on rank j % cp at local slot page (ps / cp) + j // cp (models/dsa_long.py
cp_local_slots at kpool 1), so the allocator, the prefix cache and the host tier keep their pages (a rank holds ps / cp
rows of each, engine/model_runner.py lps). Consecutive tokens sit on consecutive ranks, which spreads a selection's
recent window evenly.

Per layer, on every rank of the group:
1. the token's rows are written by the rank that owns its slot (the others write the dump slot);
2. the context's local rows are gathered (L / cp of them);
3. a full layer past index_topk keys selects (select): its local indexer scores, the local top-m (m = min(M_LOCAL,
   L / cp)) with their values and positions all-gathered over the group, the global top-k of that union by (value
   descending, position ascending): the rule kernels/dsa_topk.py and the non-parallel path keep. Exact when every rank
   whose list was cut holds no value at or above the k-th (a certificate, counted in dcp_inexact); M_LOCAL is 8x the
   2048 / 64 = 32 keys a rank holds on average. A shared layer reuses its full layer's local candidates and mask;
4. the partial attention of every head of the group over the rank's candidates (the queries all-gathered over the
   group), exp(s - max) with its max and sum, the maxima and sums all-gathered, each partial rescaled to the global
   softmax, and the partials summed over the group; each rank keeps its own heads, then W_UV and o_proj as before.
All-gathers are zero-padded all-reduces (each rank's piece at its index), as models/mla.py _cp_gather builds them.
"""

from __future__ import annotations

import os

import torch

from .decoder import NEG_INF

M_LOCAL = 256  # local candidates a rank sends per query row (the exactness certificate's margin, see above)


def m_local() -> int:
    """M_LOCAL, or TRNRT_DCP_M_LOCAL (tests: an environment variable, so an engine's rank processes see it too)."""
    return int(os.environ.get("TRNRT_DCP_M_LOCAL", M_LOCAL))
BIG_POS = float(2**24)


def enabled(model) -> bool:
    return getattr(model, "cp", 1) > 1


def rank_buffers(model, device=None) -> None:
    """dcp_rank_t [1] int64 (this rank's index in its group) and dcp_onehot [cp] fp32, buffers so that every rank traces
    one graph (vocab_start_t's rule)."""
    A = model.cp
    g = model.attn_rank % A
    oh = torch.zeros(A, dtype=torch.float32)
    oh[g] = 1
    model.register_buffer("dcp_rank_t", torch.tensor([g], dtype=torch.int64).to(device), persistent=False)
    model.register_buffer("dcp_onehot", oh.to(device), persistent=False)
    model.register_buffer("dcp_inexact", torch.zeros(2, dtype=torch.float32).to(device), persistent=False)


def gather(model, x: torch.Tensor) -> torch.Tensor:
    """[cp, *x.shape]: every group rank's x (a zero-padded all-reduce, exact: each element is one rank's value plus
    zeros)."""
    A = model.cp
    oh = model.dcp_onehot.to(x.dtype).view(A, *([1] * x.dim()))
    return model._all_reduce(x.unsqueeze(0) * oh, group=model.cp_group)


def local_rows(model, table: torch.Tensor):
    """(rows [B, Lloc] int64 local cache rows, pos [Lloc] fp32 their positions) of the pages a table [B, P] addresses:
    page i's local rows i (ps / cp) .. + ps / cp - 1 hold positions i ps + k cp + rank."""
    ps, A = model.page_size, model.cp
    lps = ps // A
    P = table.shape[-1]
    k = torch.arange(lps, device=table.device)
    rows = (table.unsqueeze(-1) * lps + k).flatten(-2)
    base = (torch.arange(P, device=table.device).unsqueeze(-1) * ps + k * A).flatten().to(torch.float32)
    return rows, base + model.dcp_rank_t.to(torch.float32)


def local_slots(model, positions: torch.Tensor, rowtab: torch.Tensor, slot_mapping: torch.Tensor) -> torch.Tensor:
    """[T] the local cache slot each row's token is written to: the dump slot on the ranks that do not own it, and for
    a padded row (its slot_mapping in the null page, below page_size: a chunk's or a batch's padding), which would
    otherwise overwrite a real token's row at its position."""
    from . import dsa_long

    ps = model.page_size
    loc = dsa_long.cp_local_slots(positions, rowtab, ps, 1, model.cp, model.dcp_rank_t)
    return torch.where(slot_mapping >= ps, loc, torch.full_like(loc, dsa_long.cp_dump_slot(ps, model.cp)))


def additive(keep: torch.Tensor) -> torch.Tensor:
    """0 where keep, NEG_INF elsewhere, fp32: from fp32 tensors, never python scalars (a where of two python floats
    traced as f64 under torch-xla, which neuronx-cc refuses on trn2: NCC_ESPP004)."""
    z = torch.zeros(keep.shape, dtype=torch.float32, device=keep.device)
    return torch.where(keep, z, torch.full_like(z, NEG_INF))


def select(model, scores: torch.Tensor, seen: torch.Tensor, pos: torch.Tensor, keep: int):
    """The global top-keep of the group's visible keys for each query row. scores [B, Q, Lloc] fp32 this rank's local
    indexer scores, seen [B, Q, Lloc] bool the visible ones, pos [Lloc] the local keys' positions. Returns (idx [B, Q,
    m] local key indices, mask [B, Q, m] additive: 0 at a selected visible key)."""
    from .dsa_select import topk_ids

    Lloc = scores.shape[-1]
    m = min(m_local(), Lloc)
    v = scores.float() + additive(seen)
    idx = topk_ids(v, m)  # [B, Q, m], descending
    tv = v.gather(-1, idx)
    tp = pos[idx]
    A = model.cp
    gv = gather(model, tv)  # [A, B, Q, m]
    gp = gather(model, tp)
    B, Q = v.shape[:2]
    allv = gv.permute(1, 2, 0, 3).reshape(B, Q, A * m)
    allp = gp.permute(1, 2, 0, 3).reshape(B, Q, A * m)
    k = min(keep, A * m)
    tau = allv.gather(-1, topk_ids(allv, k)[..., k - 1 :])  # [B, Q, 1] the k-th value
    need = k - (allv > tau).sum(-1, keepdim=True)
    tiep = torch.where(allv == tau, allp, torch.full_like(allp, BIG_POS))
    # the last tied position kept: the need-th smallest position among the ties (a top-k of the negated positions, in
    # ascending order: trn2 has no sort, NCC_EVRF029; need <= k)
    neg = -tiep
    ptau = -neg.gather(-1, topk_ids(neg, k)).gather(-1, (need - 1).clamp(min=0))
    ok = seen.gather(-1, idx)
    sel = ok & ((tv > tau) | ((tv == tau) & (tp <= ptau)))
    if Lloc > m:  # the certificate: every cut list's last value is below the k-th
        cut = gv[..., m - 1].permute(1, 2, 0)  # [B, Q, A]
        bad = (cut >= tau).any(-1) & (tau > NEG_INF / 2).squeeze(-1)
        model.dcp_inexact.add_(torch.stack([bad.float().sum(), torch.ones_like(bad, dtype=torch.float32).sum()]))
    return idx, additive(sel)


def attend(model, layer, q_nope, q_pe, kc, idx, mask, B: int, Q: int) -> torch.Tensor:
    """The group's attention over every rank's candidates, this rank's heads out: q_nope [B Q, nh, dn], q_pe [B Q, nh,
    dr] this rank's heads, kc [B, Lloc, r + dr] the local rows, idx / mask [B, Q, m] the local candidates and their
    additive mask. Returns the o_proj input [B Q, nh dv]."""
    m_ = layer.spec.mla
    nh, dn, dv, r = layer.nh, m_.qk_nope_head_dim, m_.v_head_dim, m_.kv_lora_rank
    A = model.cp
    w_uk = model._w(layer, "w_uk").view(nh, dn, r)
    w_uv = model._w(layer, "w_uv").view(nh, dv, r)
    qn = q_nope.reshape(B, Q, nh, dn)
    qp = q_pe.reshape(B, Q, nh, -1)
    q = torch.cat([torch.einsum("bqhd,hdr->bqhr", qn, w_uk), qp], dim=-1)  # [B, Q, nh, r + dr]
    qa = gather(model, q)  # [A, B, Q, nh, W]
    Hg = A * nh
    qa = qa.permute(1, 2, 0, 3, 4).reshape(B, Q, Hg, -1)
    W = kc.shape[-1]
    m = idx.shape[-1]
    kcs = kc.unsqueeze(1).expand(B, Q, kc.shape[1], W).gather(2, idx.unsqueeze(-1).expand(B, Q, m, W))  # [B, Q, m, W]
    s = torch.einsum("bqhw,bqmw->bqhm", qa, kcs).float() * m_.softmax_scale + mask.float().unsqueeze(2)
    mx = s.amax(-1, keepdim=True)  # [B, Q, Hg, 1]
    e = torch.exp(s - mx)
    tot = e.sum(-1, keepdim=True)
    o = torch.einsum("bqhm,bqmr->bqhr", e.to(model.dtype), kcs[..., :r]).float()  # [B, Q, Hg, r]
    st = gather(model, torch.cat([mx, tot], dim=-1))  # [A, B, Q, Hg, 2]
    M = st[..., 0].amax(0)
    total = (st[..., 1] * torch.exp(st[..., 0] - M)).sum(0)
    mine = torch.exp(mx.squeeze(-1) - M) / total  # [B, Q, Hg]
    o = model._all_reduce(o * mine.unsqueeze(-1), group=model.cp_group)  # the group's sum: the softmax output
    heads = model.dcp_rank_t * nh + torch.arange(nh, device=o.device)
    own = o.index_select(2, heads).to(model.dtype)  # [B, Q, nh, r]
    return torch.einsum("bqhr,hvr->bqhv", own, w_uv).reshape(B * Q, nh * dv)
