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
3. a full layer past index_topk keys selects (select): its local indexer scores, the local top-m (m = min(4096 / cp,
   L / cp)) all-gathered over the group, the global top-k of that union by kernels/dsa_topk.py (value descending, the
   lowest positions among ties while every local key is a candidate: the non-parallel path's rule). Exact when every
   rank whose list was cut holds no value at or above the k-th (a certificate, counted in dcp_inexact): m is 2x the
   2048 / cp keys a rank holds on average. A shared layer reuses its full layer's local candidates and mask;
4. the partial attention of every head of the group over the rank's candidates (the queries all-gathered over the
   group), exp(s - max) with its max and sum, the maxima and sums all-gathered, each partial rescaled to the global
   softmax, and the partials summed over the group; each rank keeps its own heads, then W_UV and o_proj as before.
All-gathers are zero-padded all-reduces (each rank's piece at its index), as models/mla.py _cp_gather builds them.
"""

from __future__ import annotations

import os

import torch

from .decoder import NEG_INF

MAX_UNION = 4096  # the group's candidates a query row (kernels/dsa_topk.py MAX_W): a rank sends 4096 / cp of them


def m_local(model) -> int:
    """The local candidates a rank sends: TRNRT_DCP_M_LOCAL if set (tests: an environment variable, so an engine's rank
    processes see it too), else 4096 / cp (the group's union fits kernels/dsa_topk.py's 4,096 scores a row)."""
    env = os.environ.get("TRNRT_DCP_M_LOCAL")
    return int(env) if env else max(1, MAX_UNION // model.cp)


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


def local_pos(model, P: int, device) -> torch.Tensor:
    """[P ps / cp] fp32: the positions of a context's local rows over P pages (page i's local rows i (ps / cp) .. + ps /
    cp - 1 hold positions i ps + k cp + rank)."""
    ps, A = model.page_size, model.cp
    k = torch.arange(ps // A, device=device)
    base = (torch.arange(P, device=device).unsqueeze(-1) * ps + k * A).flatten().to(torch.float32)
    return base + model.dcp_rank_t.to(torch.float32)


def local_load(model, cache: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """[B, P ps / cp, D] the local rows of the pages a table [B, P] addresses, by whole pages (the cache viewed as
    [pages, ps / cp, D]: one gather of P rows a sequence; a gather of token rows from a [B, P, ps / cp] index made
    neuronx-cc fail with NCC_ITEN406 at dcp 16), in the model dtype."""
    lps = model.page_size // model.cp
    B, P = table.shape
    pages = cache.view(-1, lps, cache.shape[-1])[table]  # [B, P, lps, D]
    return pages.reshape(B, P * lps, cache.shape[-1]).to(model.dtype)


def local_slots(model, positions: torch.Tensor, table: torch.Tensor, Q: int, slot_mapping: torch.Tensor) -> torch.Tensor:
    """[T] the local cache slot each row's token is written to (models/dsa_long.py cp_local_slots at kpool 1): row t is
    position positions[t] of sequence t // Q of table [B, P]; the dump slot on the ranks that do not own it, and for a
    padded row (its slot_mapping in the null page, below page_size: a chunk's or a batch's padding), which would
    otherwise overwrite a real token's row at its position. The page is read from the flat table at (t // Q) P + page:
    a table repeated Q times a row and gathered row by row made neuronx-cc fail with NCC_ITEN406 (a strided partition
    access) at dcp 16. Exact float arithmetic: page_size and cp are powers of two, positions below 2^24."""
    from . import dsa_long

    ps, A = model.page_size, model.cp
    T = positions.shape[0]
    P = table.shape[-1]
    pf = positions.to(torch.float32)
    i = torch.floor(pf * (1.0 / ps))
    within = pf - i * ps
    jA = torch.floor(within * (1.0 / A))
    owner = (within - jA * A).to(torch.int64)
    seq = torch.arange(T, device=positions.device) // Q
    page = table.reshape(-1)[seq * P + i.to(torch.int64)]
    local = page * (ps // A) + jA.to(torch.int64)
    keep = (owner == model.dcp_rank_t.view(())) & (slot_mapping >= ps)
    return torch.where(keep, local, torch.full_like(local, dsa_long.cp_dump_slot(ps, A)))


def additive(keep: torch.Tensor) -> torch.Tensor:
    """0 where keep, NEG_INF elsewhere, fp32: from fp32 tensors, never python scalars (a where of two python floats
    traced as f64 under torch-xla, which neuronx-cc refuses on trn2: NCC_ESPP004)."""
    z = torch.zeros(keep.shape, dtype=torch.float32, device=keep.device)
    return torch.where(keep, z, torch.full_like(z, NEG_INF))


def select(model, scores: torch.Tensor, seen: torch.Tensor, pos: torch.Tensor, keep: int):
    """The global top-keep of the group's visible keys for each query row. scores [B, Q, Lloc] fp32 this rank's local
    indexer scores, seen [B, Q, Lloc] bool the visible ones, pos [Lloc] the local keys' positions (unused: the union's
    column order is the position order when every local key is a candidate). Returns (idx [B, Q, m] local key indices,
    or None when every local key is a candidate (m = Lloc), mask [B, Q, m] additive: 0 at a selected visible key).

    The union of the group's candidates is laid out [m, cp] (local index, then rank), which with every local key a
    candidate is the context's position order; kernels/dsa_topk.py then selects its top-keep (lowest index among ties:
    the lowest positions, as the non-parallel path), exact, on the device in the kernel (m cp <= its MAX_W of 4,096 a
    row: m_local's default is 4096 / cp). A cut candidate list orders its union by value, so its ties break by rank
    (the certificate still bounds what a cut list may miss)."""
    from ..kernels import dsa_topk

    Lloc = scores.shape[-1]
    m = min(m_local(model), Lloc)
    v = scores.float() + additive(seen)
    B, Q = v.shape[:2]
    if m < Lloc:
        from .dsa_select import topk_ids

        idx = topk_ids(v, m)  # [B, Q, m], descending
        tv = v.gather(-1, idx)
    else:  # every local key is a candidate: no local top-k
        idx, tv = None, v
    A = model.cp
    gv = gather(model, tv)  # [A, B, Q, m]
    union = gv.permute(1, 2, 3, 0).reshape(B, Q, m * A)
    k = min(keep, m * A)
    if union.device.type == "cpu" or dsa_topk.supported(B * Q, m * A):
        mu = dsa_topk.select(union, k, vis_only=True)  # [B, Q, m A] additive
    else:  # wider than the kernel takes: torch.topk in the graph, any valid top-k among exact ties
        from .glm_moe_dsa import _xla_selection

        mu = _xla_selection(union, torch.zeros_like(union), k)
    mine = mu.view(B, Q, m, A).index_select(-1, model.dcp_rank_t).squeeze(-1)  # [B, Q, m]
    if m < Lloc:  # the certificate: every cut list's last value is below the k-th
        half = torch.full_like(union, NEG_INF / 2)  # comparisons against tensors: one against a python float traced
        big = torch.full_like(union, 3.0e38)  # as f64 under torch-xla, which trn2 refuses (NCC_ESPP004)
        tau = torch.where(mu > half, union, big).amin(-1, keepdim=True)  # [B, Q, 1] the k-th value
        cut = gv[..., m - 1].permute(1, 2, 0)  # [B, Q, A]
        real = (tau > half[..., :1]) & (tau < big[..., :1])
        bad = (cut >= tau).any(-1) & real.squeeze(-1)
        model.dcp_inexact.add_(torch.stack([bad.float().sum(), torch.ones_like(bad, dtype=torch.float32).sum()]))
    return idx, mine


def attend(model, layer, q_nope, q_pe, kc, idx, mask, B: int, Q: int) -> torch.Tensor:
    """The group's attention over every rank's candidates, this rank's heads out: q_nope [B Q, nh, dn], q_pe [B Q, nh,
    dr] this rank's heads, kc [B, Lloc, r + dr] the local rows, idx / mask [B, Q, m] the local candidates (None: every
    local row) and their additive mask. Returns the o_proj input [B Q, nh dv]."""
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
    if idx is None:  # every local row is a candidate: no gather
        s = torch.einsum("bqhw,bmw->bqhm", qa, kc).float() * m_.softmax_scale + mask.float().unsqueeze(2)
    else:
        m = idx.shape[-1]
        kcs = kc.unsqueeze(1).expand(B, Q, kc.shape[1], W).gather(2, idx.unsqueeze(-1).expand(B, Q, m, W))
        s = torch.einsum("bqhw,bqmw->bqhm", qa, kcs).float() * m_.softmax_scale + mask.float().unsqueeze(2)
    mx = s.amax(-1, keepdim=True)  # [B, Q, Hg, 1]
    e = torch.exp(s - mx)
    tot = e.sum(-1, keepdim=True)
    if idx is None:
        o = torch.einsum("bqhm,bmr->bqhr", e.to(model.dtype), kc[..., :r]).float()  # [B, Q, Hg, r]
    else:
        o = torch.einsum("bqhm,bqmr->bqhr", e.to(model.dtype), kcs[..., :r]).float()
    # the maxima and sums stacked on a LEADING axis: their concatenation on the last one, split again after the
    # gather, made neuronx-cc fail with NCC_ITEN406 (a strided partition access) at dcp 16, batch 4
    st = gather(model, torch.stack([mx.squeeze(-1), tot.squeeze(-1)], dim=0))  # [A, 2, B, Q, Hg]
    M = st[:, 0].amax(0)
    total = (st[:, 1] * torch.exp(st[:, 0] - M)).sum(0)
    mine = torch.exp(mx.squeeze(-1) - M) / total  # [B, Q, Hg]
    o = model._all_reduce(o * mine.unsqueeze(-1), group=model.cp_group)  # the group's sum: the softmax output
    heads = model.dcp_rank_t * nh + torch.arange(nh, device=o.device)
    own = o.index_select(2, heads).to(model.dtype)  # [B, Q, nh, r]
    return torch.einsum("bqhr,hvr->bqhv", own, w_uv).reshape(B * Q, nh * dv)
