"""GLM-5.3 (`glm_moe_dsa`, GlmMoeDsaForCausalLM): the text model.

78 layers on a plain pre-norm residual (no hyper-connection streams, no linear attention, so no recurrent state). Every
layer is MLA with RoPE and DeepSeek sparse attention (DSA): 64 heads of 192 nope + 64 rope query / key dims and 256
value dims over a 512 latent, q_lora 2048, softmax scale 256 ** -0.5. RoPE is the interleaved (DeepSeek) form, theta
8e6, over the 64 rope dims of q / k and the first 64 dims of the indexer's q / k. The indexer scores TOKENS (no pools):
index[t, s] = sum_h w[t, h] relu(q_I[t, h] . k_I[s] / sqrt(Di)), with w = weights_proj(x) / sqrt(Hi), and the query
attends to its index_topk (2048) best visible tokens. Only the "full" layers run an indexer (indexer_types: layers 0, 1,
2, then every 4th from 6); each "shared" layer attends with the selection of the full layer before it. The MLP is a
plain SwiGLU (no clamp), first 3 layers dense (intermediate 12288), then 256 routed experts (sigmoid routing with a
selection bias, top 8, routed_scaling_factor 2.5) plus one shared expert. The MTP layer (layers.78) is not loaded.

Numerics follow transformers v5.17.0 src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py
(https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py),
config keys its configuration_glm_moe_dsa.py; the checkpoint's config.json:
https://huggingface.co/zai-org/GLM-5.3/raw/ba32f8c19c8738e2c57c71798c601d477a7bf844/config.json

The paged cache, per layer and token: K holds the latent c then the rotated rope key k_pe (512 + 64, the layout of vLLM's
and FlashMLA's MLA caches), V the rotated indexer key (128) on a full layer and nothing on a shared one. Decode attends
over the latent by weight absorption ([W_UK^T q_nope, q_pe] . [c, k_pe]); a prefill chunk decompresses its context.

The selection of a full layer is an additive mask [rows, keys] (models/glm5_next.py block_mask at kpool 1: the
kernels/dsa_topk.py selection, its NKI kernel on the device). A shared layer reads it from the Python carry of the graph
it runs in (model._sel) or, when its full layer ran in an earlier graph of the step, from the selection scratch every
DSA layer shares (models/mla.py bind_scratch), which the full layer wrote. When a bucket holds at most index_topk keys
every visible key is selected, so there the layer is exactly dense causal MLA and nothing is selected or staged.

This is the host (and plain XLA device) path: it is exact, not fast. The decode megakernels, context-parallel DSA and
the sequence-parallel prefill streams of GLM-5.3-Flash do not take these layers yet; the MTP layer is not built.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from . import mla as _mla
from .config import AttnSpec, ModelConfig
from .decoder import NEG_INF, DecoderForCausalLM, rms_norm
from .glm5_next import _ids, _indexer_types, block_mask, pool_index
from .mla import DSASpec, MLASpec
from .quant import FP8, dequant

ARCHITECTURES = ("GlmMoeDsaForCausalLM",)


@dataclass(frozen=True)
class RopeMLASpec(MLASpec):
    """A GLM-5.3 layer's MLA: qk_rope_head_dim > 0, rotated with interleaved RoPE of rope_theta; index_shared: the layer
    runs no indexer and attends with the selection of the full layer before it (indexer_types "shared")."""

    rope_theta: float = 10000.0
    index_shared: bool = False


def cache_widths(m: RopeMLASpec) -> tuple[int, int]:
    """(K, V) widths of the paged cache: the latent and the rope key; the indexer key, on a full layer only."""
    return m.kv_lora_rank + m.qk_rope_head_dim, 0 if m.index_shared else m.dsa.head_dim


def _attn_spec(t: dict, theta: float, shared: bool) -> AttnSpec:
    dn, dr = t["qk_nope_head_dim"], t["qk_rope_head_dim"]
    if t["index_head_dim"] < dr:
        raise ValueError(f"index_head_dim {t['index_head_dim']} is below qk_rope_head_dim {dr}")
    dsa = DSASpec(t["index_n_heads"], t["index_head_dim"], t["index_topk"], kpool=1, kpool_tail=False)
    # q_a_layernorm / kv_a_layernorm are GlmMoeDsaRMSNorm(dim) with its default eps 1e-6, not rms_norm_eps
    # (GlmMoeDsaAttention.__init__); the softmax scale is qk_head_dim ** -0.5 (no YaRN mscale: default RoPE only).
    m = RopeMLASpec(t["q_lora_rank"], t["kv_lora_rank"], dn, dr, t["v_head_dim"], (dn + dr) ** -0.5, norm_eps=1e-6,
                    dsa=dsa, rope_theta=theta, index_shared=shared)
    kw, vw = cache_widths(m)
    return AttnSpec(t["num_attention_heads"], 1, kw, vw, dr, theta, mla=m)


def config_from_hf(cls, c: dict, eos_ids: tuple[int, ...]) -> ModelConfig:
    """Keys of GlmMoeDsaConfig (configuration_glm_moe_dsa.py, v5.17.0)."""
    t = c
    n = t["num_hidden_layers"]
    if t.get("hidden_act", "silu") != "silu":
        raise NotImplementedError(f"hidden_act {t.get('hidden_act')!r}")
    if t.get("q_lora_rank") is None:
        raise NotImplementedError("glm_moe_dsa needs q_lora_rank")
    if not t.get("qk_rope_head_dim"):
        raise NotImplementedError("glm_moe_dsa layers without RoPE (qk_rope_head_dim 0)")
    if t["qk_rope_head_dim"] % 2:
        raise ValueError(f"qk_rope_head_dim {t['qk_rope_head_dim']} is odd")
    if t.get("scoring_func", "sigmoid") != "sigmoid" or t.get("topk_method", "noaux_tc") != "noaux_tc":
        raise NotImplementedError("only sigmoid / noaux_tc routing (GlmMoeDsaTopkRouter)")
    if not t.get("rope_interleave", True) or not t.get("indexer_rope_interleave", True):
        raise NotImplementedError("only interleaved RoPE (rope_interleave, indexer_rope_interleave)")
    rp = t.get("rope_parameters") or {"rope_type": "default", "rope_theta": t.get("rope_theta", 10000.0)}
    if rp.get("rope_type", "default") != "default":
        raise NotImplementedError(f"rope_type {rp.get('rope_type')!r} (default RoPE only)")
    theta = float(rp.get("rope_theta", t.get("rope_theta", 10000.0)))
    # GlmMoeDsaConfig.__post_init__: the first first_k_dense_replace layers are dense, the rest MoE.
    k_dense = min(t.get("first_k_dense_replace", 3), n)
    mlp = t.get("mlp_layer_types") or ["dense"] * k_dense + ["sparse"] * (n - k_dense)
    types = t.get("layer_types") or ["deepseek_sparse_attention"] * n
    if any(x != "deepseek_sparse_attention" for x in types[:n]):
        raise NotImplementedError(f"layer types {sorted(set(types[:n]))} (DSA layers only)")
    itypes = _indexer_types(t, n)
    if itypes[0] != "full":
        raise ValueError("indexer_types: layer 0 is 'shared', but a shared layer reuses an earlier layer's selection")
    if any(x not in ("full", "shared") for x in itypes):
        raise ValueError(f"indexer_types {sorted(set(itypes))}")
    specs = tuple(_attn_spec(t, theta, itypes[i] == "shared") for i in range(n))
    experts = t.get("n_routed_experts") or 0
    # The MTP layer (model.layers.<n>, with its own indexer) is not built here (GlmMoeDsaForCausalLM refuses mtp); its
    # spec is kept so the checkpoint's layer count reads right.
    mtp = {}
    if int(t.get("num_nextn_predict_layers") or 0):
        mtp = dict(mtp_layers=int(t["num_nextn_predict_layers"]), mtp_spec=_attn_spec(t, theta, False),
                   mtp_prefix=f"model.layers.{n}", mtp_moe=bool(experts))
    eos = eos_ids or _ids(t.get("eos_token_id"))
    return cls(
        architecture=c["architectures"][0], vocab_size=t["vocab_size"], hidden_size=t["hidden_size"],
        intermediate_size=t["intermediate_size"], num_layers=n, num_heads=t["num_attention_heads"], num_kv_heads=1,
        head_dim=specs[0].head_dim, rms_norm_eps=t.get("rms_norm_eps", 1e-5), rope_theta=theta,
        max_position_embeddings=t.get("max_position_embeddings", 1048576),
        tie_word_embeddings=bool(t.get("tie_word_embeddings", False)), eos_token_ids=eos,
        num_experts=experts, num_experts_per_tok=t.get("num_experts_per_tok", 0) or 0,
        moe_intermediate_size=t.get("moe_intermediate_size", 0) or 0, norm_topk_prob=t.get("norm_topk_prob", True),
        moe_layers=tuple(i for i in range(n) if experts and mlp[i] == "sparse"), attn_layers=specs,
        router_scoring="sigmoid", router_bias=bool(experts),
        routed_scaling_factor=float(t.get("routed_scaling_factor") or 1.0),
        n_shared_experts=(t.get("n_shared_experts") or 0) if experts else 0, n_group=t.get("n_group") or 1,
        topk_group=t.get("topk_group") or 1, hybrid=None, **mtp, **cls._quant(c))


# -- RoPE ------------------------------------------------------------------------------


def rope_table(dr: int, theta: float, n: int) -> torch.Tensor:
    """[n, dr] fp32: cos then sin of position p's dr / 2 angles p * theta ** (-2 i / dr) (GlmMoeDsaRotaryEmbedding with
    head_dim = qk_rope_head_dim, attention_scaling 1). Built on the host once: an in-graph cos of an angle near 1e6
    rad would be as exact as the device's range reduction, a table read is exact."""
    inv = 1.0 / (theta ** (torch.arange(0, dr, 2, dtype=torch.float32, device="cpu") / dr))
    f = torch.arange(n, dtype=torch.float32, device="cpu").unsqueeze(1) * inv.unsqueeze(0)
    return torch.cat([f.cos(), f.sin()], dim=-1)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE of x [..., dr] (pairs (x0, x1), (x2, x3), ... each rotated by one angle), output
    de-interleaved: [even', odd'] (apply_rotary_pos_emb_interleave). cos / sin [..., dr / 2] broadcast over x's
    leading dims, in x's dtype. Every q and k this rotates comes out in the same order, so their products are the
    interleaved ones."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# -- parameters ------------------------------------------------------------------------

_INDEXER = ("idx_wq", "idx_wq_scale", "idx_wk", "idx_wk_scale", "idx_knorm_w", "idx_knorm_b", "idx_wproj")


def _fix_indexer(model, layer, prefix: str) -> None:
    """models/mla.py init_layer builds GLM-5.3-Flash's pooled indexer: drop its pool parameters, give a full layer's
    wq_b its full width (index_n_heads x index_head_dim: the RoPE dims are part of each head here), and give a shared
    layer no indexer at all."""
    m = layer.spec.mla
    d = m.dsa
    for name in ("idx_pool_ape", "idx_pool_gate"):
        delattr(layer, name)
    if m.index_shared:
        for name in _INDEXER:
            delattr(layer, name)
        return
    old = layer.idx_wq
    rows, cols = d.n_heads * d.head_dim, m.q_lora_rank
    layer.idx_wq = nn.Parameter(torch.empty(rows, cols, dtype=old.dtype, device=old.device), requires_grad=False)
    if layer.idx_wq_scale is not None:
        bk = model.cfg.quant_block
        layer.idx_wq_scale = nn.Parameter(torch.empty(rows, -(-cols // bk), dtype=torch.float32, device=old.device),
                                          requires_grad=False)


def load_attention(layer, ck, pre: str, cfg: ModelConfig, rank: int, dtype, assign, concat, put) -> None:
    """Fill a layer's attention parameters from the checkpoint (loader._load_decoder), as attention rank `rank`: the
    MLA projections as models/mla.py load_layer lays them out, then a full layer's indexer (tensor names:
    zai-org/GLM-5.3 model.safetensors.index.json, self_attn.indexer.wq_b / .wk / .k_norm.weight / .k_norm.bias /
    .weights_proj)."""
    m = layer.spec.mla
    a = pre + "self_attn."
    blk = cfg.quant_block
    nh, dn, dr, dv = layer.nh, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim
    assign(layer.w_a, layer.w_a_scale, *concat([ck.linear(a + "q_a_proj", block=blk),
                                                ck.linear(a + "kv_a_proj_with_mqa", block=blk)]))
    put(layer.q_a_norm, a + "q_a_layernorm.weight")
    put(layer.kv_a_norm, a + "kv_a_layernorm.weight")
    qd = dn + dr
    w, s = ck.linear(a + "q_b_proj", slice(rank * nh * qd, (rank + 1) * nh * qd), block=blk)
    assign(layer.q_b, layer.q_b_scale, w, s)
    kd = dn + dv
    w, s = ck.linear(a + "kv_b_proj", slice(rank * nh * kd, (rank + 1) * nh * kd), block=blk)
    w3 = w.view(nh, kd, -1)
    s3 = s.view(nh, kd, -1) if s is not None else None
    assign(layer.w_uk, layer.w_uk_scale, w3[:, :dn].reshape(nh * dn, -1),
           s3[:, :dn].reshape(nh * dn, -1) if s3 is not None else None)
    assign(layer.w_uv, layer.w_uv_scale, w3[:, dn:].reshape(nh * dv, -1),
           s3[:, dn:].reshape(nh * dv, -1) if s3 is not None else None)
    w, s = ck.linear(a + "o_proj", cols=slice(rank * nh * dv, (rank + 1) * nh * dv), block=blk)
    assign(layer.o, layer.o_scale, w, s)
    if m.index_shared:
        return
    w, s = ck.linear(a + "indexer.wq_b", block=blk)
    assign(layer.idx_wq, layer.idx_wq_scale, w, s)
    w, s = ck.linear(a + "indexer.wk", block=blk)
    assign(layer.idx_wk, layer.idx_wk_scale, w, s)
    layer.idx_knorm_w.data.copy_(ck.get(a + "indexer.k_norm.weight").to(dtype))
    layer.idx_knorm_b.data.copy_(ck.get(a + "indexer.k_norm.bias").to(dtype))
    w, s = ck.linear(a + "indexer.weights_proj", block=blk)  # fp32 (_keep_in_fp32_modules)
    layer.idx_wproj.data.copy_(w.float() if s is None else dequant(w, s, torch.float32))


# -- forward ---------------------------------------------------------------------------


# Decode's weight copies: every weight a decode step multiplies by, dequantized at load and pre-tiled for the decode
# projection kernel (dt_<name>, kernels/gemv_tiled.py pack: input rows on the partitions; in the model dtype, fp32 for the
# fp32 ones), read only by the whole-step decode graph (DecoderLayer._decode_only, so no prefill graph's inputs move).
# XLA's F.linear(x, W) transposed every [out, in] weight on the tensor engine in every step: profiled on trn2
# (neuron-explorer, batch 1, 78 layers, rank 0 of 64), 301 of the step's 314 GFLOP, 174.6 ms a step.
DECODE_COPIES = ("w_a", "q_b", "o", "idx_wq", "idx_wk", "gate_up", "down", "shared_gate_up", "shared_down")
DECODE_COPIES_F32 = ("idx_wproj", "router")


def _lin(model, layer, x, name: str, dec: bool):
    """x @ W^T for the layer's weight `name`: in a decode call the projection kernel over its pre-tiled copy
    (DECODE_COPIES)."""
    from ..kernels import gemv_tiled

    t = getattr(layer, "dt_" + name, None) if dec else None
    if t is not None and gemv_tiled.takes(x.shape[0]):
        return gemv_tiled.gemv(x.to(t.dtype), t)
    if name in DECODE_COPIES_F32:
        return F.linear(x.float(), getattr(layer, name).float())
    return F.linear(x, model._w(layer, name))


def _w_a_sharded(model, layer, x):
    """Decode's q_a + kv_a projection [T, ql + r + dr] (replicated on every rank in the prefill graphs, vLLM's and
    SGLang's MLA layout) as tp column shards: this rank's columns through the projection kernel, placed at its offset in
    a zero row by a one-hot product (decoder._head's form: the offset is a buffer, so every rank traces one graph) and
    all-reduced. At tp 64 each rank reads 41 of the 2,624 columns' weights instead of all of them: 32 MB of the 3.55 GB a
    78-layer batch-1 step read (profiled on trn2), for one more all-reduce a layer. The sum adds zeros to each column's
    one value, so the row is the replicated projection's."""
    from ..kernels import gemv_tiled

    T, tp = x.shape[0], model.tp_size
    blob = layer.dt_w_a_shard
    local = gemv_tiled.gemv(x.to(blob.dtype), blob)  # [T, n] this rank's columns
    n = local.shape[1]
    mine = (torch.arange(tp, device=x.device) * n == model.w_a_lo_t).to(local.dtype)
    full = (local.unsqueeze(1) * mine.view(1, -1, 1)).reshape(T, tp * n)
    return model._all_reduce(full)[:, : layer.w_a_cols]


def _project(model, layer, x, positions, dec: bool = False):
    """q_nope [T, nh, dn], the rotated q_pe [T, nh, dr], latent c [T, r], the rotated k_pe [T, dr], the normalised q
    latent (the indexer's input) and the angles' (cos, sin) [T, dr / 2] (GlmMoeDsaAttention.forward): models/mla.py
    _project's arithmetic, with decode's weight copies."""
    m = layer.spec.mla
    T = x.shape[0]
    dn, dr, r, ql = m.qk_nope_head_dim, m.qk_rope_head_dim, m.kv_lora_rank, m.q_lora_rank
    if dec and getattr(layer, "dt_w_a_shard", None) is not None:
        a = _w_a_sharded(model, layer, x)
    else:
        a = _lin(model, layer, x, "w_a", dec)
    q_resid = rms_norm(a[:, :ql], layer.q_a_norm, m.norm_eps)
    kv = a[:, ql:]
    q = _lin(model, layer, q_resid, "q_b", dec).view(T, layer.nh, dn + dr)
    q_nope, q_pe = q[..., :dn], q[..., dn:]
    c = rms_norm(kv[:, :r], layer.kv_a_norm, m.norm_eps)
    k_pe = kv[:, r:]
    cos, sin = model._rope(positions)
    return q_nope, rope(q_pe, cos.unsqueeze(1), sin.unsqueeze(1)), c, rope(k_pe, cos, sin), q_resid, (cos, sin)


def _indexer(model, layer, x, q_resid, cs, dec: bool = False, scores: bool = True):
    """The indexer's query [T, Hi, Di], key [T, Di] and head weights [T, Hi] fp32 (GlmMoeDsaIndexer.forward before
    scoring): RoPE on the first qk_rope_head_dim dims of each, LayerNorm (eps 1e-6, with bias) on the key first.
    scores False (a bucket of at most index_topk keys selects nothing): the key alone, for the cache; the query and
    weights are None, and their weights (16.8 MB of wq_b a full layer at GLM-5.3's shapes) are not read."""
    m = layer.spec.mla
    d = m.dsa
    T, dr = x.shape[0], m.qk_rope_head_dim
    cos, sin = cs
    k = _lin(model, layer, x, "idx_wk", dec)
    k = F.layer_norm(k.float(), (d.head_dim,), layer.idx_knorm_w.float(), layer.idx_knorm_b.float(), 1e-6).to(x.dtype)
    k = torch.cat([rope(k[..., :dr], cos, sin), k[..., dr:]], dim=-1)
    if not scores:
        return None, k, None
    q = _lin(model, layer, q_resid, "idx_wq", dec).view(T, d.n_heads, d.head_dim)
    q = torch.cat([rope(q[..., :dr], cos.unsqueeze(1), sin.unsqueeze(1)), q[..., dr:]], dim=-1)
    w = _lin(model, layer, x, "idx_wproj", dec) * d.n_heads ** -0.5
    return q, k, w


# The device's top-k of a full layer's selection: "nki" is kernels/dsa_topk.py (its emulation on the host, the tie rule
# exact), falling back to "xla" for a width the kernel does not take (dsa_topk.supported); "xla" is torch.topk in the
# graph (dsa_select.topk_ids), any valid top-k among exact ties. On libtorch-neuronx-lite past SDK 2.32 the kernel needs
# the native compile backend (platform.compile_backend).
DEVICE_TOPK = "nki"

# Expert parallelism of the routed experts (EngineConfig.moe_ep, set on every rank by engine.set_decode_routes before
# the shard is built): the tp ranks form moe_ep groups of tp / moe_ep consecutive ranks; group g holds experts
# [g E / ep, (g + 1) E / ep), each split over its ranks by intermediate columns (I / (tp / ep) a rank). At tp 64 and ep 2
# a rank holds 128 experts x 64 columns: 2 x 64 = 128 gate + up rows, the width the NKI MoE kernels take
# (moe_dedupe.supports), where tp 64 alone leaves 64. Tokens are replicated (the attention block's all-reduce), so a rank
# runs its own experts on every token, a pair routed to the other group as local expert 0 with weight 0 (a valid lane
# of moe_dedupe's plan, adding 0), and the one all-reduce after the MLP sums the groups as it sums the column shards.
MOE_EP = 1


def _xla_selection(index: torch.Tensor, vis: torch.Tensor, keep: int) -> torch.Tensor:
    """Additive [B, Q, L]: 0 at each query's `keep` best-scored visible keys (all visible ones if fewer), else NEG_INF."""
    from .dsa_select import topk_ids

    s = index.float() + vis.float()
    top = topk_ids(s, min(keep, s.shape[-1]))
    return torch.full_like(s, NEG_INF).scatter(-1, top, 0.0)


def _selection(model, layer, idx, kI, vis, B: int, Q: int) -> torch.Tensor:
    """The additive selection [B, Q, L] of B x Q queries: a full layer's own (idx: its _indexer query and weights, kI
    [B, L, Di] the context's indexer keys), published for the shared layers after it; a shared layer's (idx None),
    from the graph's carry or the scratch."""
    m = layer.spec.mla
    d = m.dsa
    T, L = B * Q, vis.shape[-1]
    if m.index_shared:
        sel = getattr(model, "_sel", None)
        if sel is not None and tuple(sel.shape) == (B, Q, L):
            return sel
        if not layer.dsa_stage or T > layer.dsa_mask.shape[0] or L > layer.dsa_mask.shape[1]:
            raise RuntimeError("a shared DSA layer has no selection to reuse: its full layer ran in another graph and "
                               "the selection scratch is missing or too small")
        return _mla._shared(layer, T, L).view(B, Q, L)
    q, w = idx
    index = pool_index(d, (None, q.reshape(B, Q, d.n_heads, d.head_dim)), w.view(B, Q, d.n_heads), kI)
    from ..kernels import dsa_topk

    if index.device.type != "cpu" and (DEVICE_TOPK == "xla" or not dsa_topk.supported(B * Q, vis.shape[-1])):
        sel = _xla_selection(index, vis, d.topk)
    else:
        sel = block_mask(index, vis, 1, d.topk, False)
    model._sel = sel
    if layer.dsa_stage and T <= layer.dsa_mask.shape[0] and L <= layer.dsa_mask.shape[1]:
        _mla._share(layer, sel, T)
    return sel


def _attend(model, layer, q_nope, q_pe, kc, vis, B: int, Q: int, absorb: bool) -> torch.Tensor:
    """B x Q queries over L cached rows kc [B, L, r + dr] ([c, k_pe]) with additive visibility vis [B, Q, L]. absorb:
    attend over the latent (decode), else decompress the context into per-head keys and values (a chunk, the
    reference). Returns the o_proj input [B Q, nh dv]."""
    m = layer.spec.mla
    nh, dn, dr, dv, r = layer.nh, m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim, m.kv_lora_rank
    w_uk = model._w(layer, "w_uk").view(nh, dn, r)
    w_uv = model._w(layer, "w_uv").view(nh, dv, r)
    qn = q_nope.reshape(B, Q, nh, dn)
    qp = q_pe.reshape(B, Q, nh, dr)
    c = kc[..., :r]
    if absorb:
        q = torch.cat([torch.einsum("bqhd,hdr->bqhr", qn, w_uk), qp], dim=-1)
        s = torch.einsum("bqhr,blr->bhql", q, kc)
        p = torch.softmax(s.float() * m.softmax_scale + vis.unsqueeze(1), dim=-1).to(model.dtype)
        o = torch.einsum("bqhr,hvr->bqhv", torch.einsum("bhql,blr->bqhr", p, c), w_uv)
    else:
        L = kc.shape[1]
        k = torch.cat([torch.einsum("blr,hdr->blhd", c, w_uk), kc[..., r:].unsqueeze(2).expand(B, L, nh, dr)], dim=-1)
        v = torch.einsum("blr,hvr->blhv", c, w_uv)
        s = torch.einsum("bqhd,blhd->bhql", torch.cat([qn, qp], dim=-1), k)
        p = torch.softmax(s.float() * m.softmax_scale + vis.unsqueeze(1), dim=-1).to(model.dtype)
        o = torch.einsum("bhql,blhv->bqhv", p, v)
    return o.reshape(B * Q, nh * dv)


def attention(model, layer, x, positions, slot_mapping, table, bias):
    """The attention block over the paged cache, for a decode batch (table [B, P], positions [B]), a prefill chunk
    (table [P], positions [C]) or a verify batch (table [B, P], bias [B, 1, 1, Q, L]); x is already normalised.
    Writes the token's cache rows first, then attends (selecting in a bucket of more than index_topk keys). Returns
    the o_proj output before the tensor-parallel all-reduce."""
    m = layer.spec.mla
    d = m.dsa
    T = x.shape[0]
    dec = table.dim() == 2  # a decode or verify batch: the whole-step graph, with decode's weight copies
    q_nope, q_pe, c, k_pe, q_resid, cs = _project(model, layer, x, positions, dec)
    model._store(layer.k_cache, slot_mapping, torch.cat([c, k_pe], dim=-1).unsqueeze(1))
    idx = None
    L = table.shape[-1] * model.page_size  # the bucket's keys (static)
    if not m.index_shared:
        qi, ki, wi = _indexer(model, layer, x, q_resid, cs, dec, scores=L > d.topk)
        model._store(layer.v_cache, slot_mapping, ki.unsqueeze(1))
        idx = (qi, wi)
    kc = model._load(layer.k_cache, table).squeeze(-2)  # [(B,) L, r + dr]
    if table.dim() == 1:  # chunk: one sequence, C queries
        B, Q = 1, T
        kc = kc.unsqueeze(0)
    elif bias.dim() == 5:  # a verify batch: B sequences of Q rows
        B, Q = table.shape[0], bias.shape[3]
    else:  # decode
        B, Q = T, 1
    L = kc.shape[1]
    vis = bias.reshape(B, Q, L)
    if L > d.topk:  # static per bucket: below it every visible key is kept
        kI = None if idx is None else model._load(layer.v_cache, table).squeeze(-2).reshape(B, L, d.head_dim)
        vis = vis + _selection(model, layer, idx, kI, vis, B, Q)
    from ..kernels import mla_decode

    dk = getattr(layer, "dk_w_uk", None)
    if (dk is not None and bias.dim() == 4 and x.device.type != "cpu"
            and mla_decode.takes(B, L, m.qk_nope_head_dim, m.qk_rope_head_dim, m.kv_lora_rank)):
        # a decode batch on the device, one head a rank: kernels/mla_decode.py over the context's rows
        out = mla_decode.attend(q_nope.reshape(B, -1), q_pe.reshape(B, -1), kc, vis.reshape(B, L), dk,
                                layer.dk_w_uv_t, m.softmax_scale)
    else:
        out = _attend(model, layer, q_nope, q_pe, kc, vis, B, Q, absorb=table.dim() != 1)
    return _lin(model, layer, out, "o", dec)


def reference(model, layer, x, positions):
    """The attention block over a whole sequence without a cache (forward_logits, tests)."""
    m = layer.spec.mla
    T = x.shape[0]
    q_nope, q_pe, c, k_pe, q_resid, cs = _project(model, layer, x, positions)
    vis = torch.where(positions.unsqueeze(0) <= positions.unsqueeze(1), 0.0, NEG_INF).view(1, T, T)
    if T > m.dsa.topk:
        idx, kI = None, None
        if not m.index_shared:
            qi, ki, wi = _indexer(model, layer, x, q_resid, cs)
            idx, kI = (qi, wi), ki.unsqueeze(0)
        vis = vis + _selection(model, layer, idx, kI, vis, 1, T)
    out = _attend(model, layer, q_nope, q_pe, torch.cat([c, k_pe], dim=-1).unsqueeze(0), vis, 1, T, absorb=False)
    return F.linear(out, model._w(layer, "o"))


def _route(model, layer, x: torch.Tensor, dec: bool):
    """DecoderForCausalLM._route (sigmoid scores, choice on score + bias, the device-safe top-k, weights renormalised and
    scaled), its fp32 logits through decode's router copy."""
    from .dsa_select import topk_ids

    if not dec:
        return model._route(layer, x)
    cfg = model.cfg
    k = cfg.num_experts_per_tok
    scores = _lin(model, layer, x, "router", dec).sigmoid()  # [T, E] fp32
    choice = scores + layer.router_bias if layer.router_bias is not None else scores
    topi = topk_ids(choice, k)
    topv = torch.gather(scores, 1, topi)
    if cfg.norm_topk_prob and k > 1:
        topv = topv / (topv.sum(dim=-1, keepdim=True) + 1e-20)
    return (topv * cfg.routed_scaling_factor).to(model.dtype), topi


def _experts(model, layer, x: torch.Tensor, dec: bool = False) -> torch.Tensor:
    """The routed experts with GlmMoeDsaExperts' plain SwiGLU: models/hybrid.py _moe_clamped without the clamp (the
    NKI kernels with act "silu" for a packed layer on the device)."""
    cfg = model.cfg
    topv, topi = _route(model, layer, x, dec)
    if getattr(model, "moe_ep", 1) > 1:  # this rank's experts only: the other group's pairs as expert 0, weight 0
        local = topi - model.ep_lo_t
        mine = (local >= 0) & (local < layer.ep_experts)
        topi = torch.where(mine, local, torch.zeros_like(local))
        topv = torch.where(mine, topv, torch.zeros_like(topv))
    T, H = x.shape
    k, Im = cfg.num_experts_per_tok, model.moe_inter
    if getattr(layer, "moe_blob", False) and x.device.type != "cpu":
        from ..kernels.moe_dedupe import ACTS, moe_dedupe
        from .decoder import MOE_PREFILL_MIN_TOKENS, moe_prefill_down

        if T >= MOE_PREFILL_MIN_TOKENS:
            from ..kernels.moe_prefill_tsb import moe_prefill_tsb

            return moe_prefill_tsb(x, topv, topi, layer.w_blob, act=ACTS["silu"], down=moe_prefill_down(layer))
        return moe_dedupe(x, topv, topi, layer.w_blob, act=ACTS["silu"])

    def act(gu):
        return F.silu(gu[..., :Im]) * gu[..., Im:]

    if T * k <= model.MOE_GATHER_MAX_PAIRS:
        flat = topi.reshape(T * k)
        xs = x.unsqueeze(1).expand(T, k, H).reshape(T * k, H, 1)
        a = act(torch.bmm(model._experts(layer, "w_gu", flat), xs).squeeze(-1))
        y = torch.bmm(a.unsqueeze(1), model._experts(layer, "w_down", flat)).squeeze(1)
        return (y.view(T, k, H) * topv.unsqueeze(-1)).sum(dim=1)
    a = act(torch.einsum("th,eih->eti", x, model._experts(layer, "w_gu")))
    ye = torch.einsum("eti,eih->eth", a, model._experts(layer, "w_down"))
    w = torch.zeros(T, cfg.num_experts, dtype=model.dtype, device=x.device).scatter(1, topi, topv)
    return torch.einsum("eth,te->th", ye, w)


def _swiglu(model, layer, x, gate_up: str, down: str, dec: bool) -> torch.Tensor:
    gu = _lin(model, layer, x, gate_up, dec)
    n = gu.shape[-1] // 2
    return _lin(model, layer, F.silu(gu[..., :n]) * gu[..., n:], down, dec)


def mlp(model, layer, x: torch.Tensor, dec: bool = False) -> torch.Tensor:
    """The MLP block on its normalised input x [T, H], after its all-reduce (GlmMoeDsaMLP / GlmMoeDsaMoE)."""
    if not layer.moe:
        return model._out_reduce(_swiglu(model, layer, x, "gate_up", "down", dec))
    y = _experts(model, layer, x, dec)
    if layer.shared_gate_up is not None:
        y = y + _swiglu(model, layer, x, "shared_gate_up", "shared_down", dec)
    return model._out_reduce(y)


class GlmMoeDsaForCausalLM(DecoderForCausalLM):
    """GLM-5.3 on DecoderForCausalLM's paged cache, tensor parallelism, sampler and graph entry points, with every layer
    a plain residual block (DecoderLayer plain) of this module's attention and MLP."""

    def __init__(self, cfg: ModelConfig, dtype: torch.dtype, max_positions: int, tp_rank: int = 0, tp_size: int = 1,
                 tp_group=None, keep_fp8: bool = False, vocab_parallel: bool = False, moe_kernel: str = "xla",
                 attn_tp: int | None = None, attn_group=None, mtp: bool = False):
        if mtp:
            raise NotImplementedError("GLM-5.3's MTP layer (speculative decoding) is not supported yet")
        if cfg.hybrid is not None:
            raise ValueError("GLM-5.3 has no hyper-connection streams")
        ep = MOE_EP if cfg.num_experts else 1
        # Under EP the experts are built at tp's width first and rebuilt as this rank's group below (_init_ep): an NKI
        # moe_kernel would refuse tp's width (moe_dedupe.supports) before then.
        super().__init__(cfg, dtype, max_positions, tp_rank, tp_size, tp_group, keep_fp8=keep_fp8,
                         vocab_parallel=vocab_parallel, moe_kernel="xla" if ep > 1 else moe_kernel, attn_tp=attn_tp,
                         attn_group=attn_group)
        self.moe_ep = ep
        if ep > 1:
            self._init_ep(moe_kernel)
        # Sequence-parallel prefill streams and context-parallel DSA are GLM-5.3-Flash's paths (models/hybrid.py,
        # models/mla.py attention_cp); every rank here runs every row of a chunk over the replicated latent.
        self.prefill_sp = False
        for name in ("sp_index", "sp_onehot"):  # registered by DecoderForCausalLM.__init__ while prefill_sp was on
            self._buffers.pop(name, None)
        for i, layer in enumerate(self.layers):
            _fix_indexer(self, layer, f"model.layers.{i}")
        self.rope_positions = max_positions
        self._sel = None  # the selection a full layer published for the shared layers of the same graph (_selection)
        self._rope_buffer(None)
        for layer in self.layers:
            self._init_decode_copies(layer)
        shard = next((l.dt_w_a_shard for l in self.layers if getattr(l, "dt_w_a_shard", None) is not None), None)
        self.w_a_lo = self.tp_rank * shard.shape[2] if shard is not None else None
        if self.w_a_lo is not None:
            self._w_a_buffer(None)

    def _init_decode_copies(self, layer) -> None:
        """The layer's dt_<name> decode copies (DECODE_COPIES, the projection kernel's blobs), filled at load
        (after_layer_load)."""
        from ..kernels import gemv_tiled

        names = []
        for name in (*DECODE_COPIES, *DECODE_COPIES_F32):
            w = getattr(layer, name, None)
            if not isinstance(w, torch.Tensor):
                continue
            if name == "w_a" and self.tp_size > 1:  # decode reads its column shard only (_w_a_sharded)
                n_out, n_in = w.shape
                n = -(-n_out // self.tp_size)
                layer.w_a_cols = n_out
                layer.dt_w_a_shard = nn.Parameter(torch.empty(gemv_tiled.P, gemv_tiled.tiles(n_in), n, dtype=self.dtype,
                                                              device=w.device), requires_grad=False)
                names.append("dt_w_a_shard")
                continue
            from ..kernels import gemv_tiled

            dt = torch.float32 if name in DECODE_COPIES_F32 else self.dtype
            n_out, n_in = w.shape
            blob = torch.empty(gemv_tiled.P, gemv_tiled.tiles(n_in), n_out, dtype=dt, device=w.device)
            setattr(layer, "dt_" + name, nn.Parameter(blob, requires_grad=False))
            names.append("dt_" + name)
        if layer.nh == 1:  # one head a rank: the MLA decode kernel's bf16 W_UK [dn, r] and W_UV^T [r, dv]
            m = layer.spec.mla
            dn, dv, r = m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
            dev = layer.w_uk.device
            layer.dk_w_uk = nn.Parameter(torch.empty(dn, r, dtype=self.dtype, device=dev), requires_grad=False)
            layer.dk_w_uv_t = nn.Parameter(torch.empty(r, dv, dtype=self.dtype, device=dev), requires_grad=False)
            names += ["dk_w_uk", "dk_w_uv_t"]
        layer._decode_only = (*getattr(layer, "_decode_only", ()), *names)

    def after_layer_load(self, layer) -> None:
        """Fill the layer's decode copies from the weights the loader just filled (loader._load_decoder), on the host,
        before the layer moves to the device."""
        if getattr(layer, "dt_w_a_shard", None) is not None:
            from ..kernels import gemv_tiled

            t = layer.dt_w_a_shard
            n = t.shape[2]
            sc = layer.w_a_scale
            w = dequant(layer.w_a.data, sc.data if sc is not None else None, torch.float32)
            part = torch.zeros(n, w.shape[1])
            rows = w[self.tp_rank * n : (self.tp_rank + 1) * n]
            part[: rows.shape[0]] = rows
            t.data.copy_(gemv_tiled.pack(part, t.dtype))
        for name in (*DECODE_COPIES, *DECODE_COPIES_F32):
            t = getattr(layer, "dt_" + name, None)
            if t is None:
                continue
            from ..kernels import gemv_tiled

            sc = getattr(layer, name + "_scale", None)
            w = dequant(getattr(layer, name).data, sc.data if sc is not None else None, torch.float32)
            t.data.copy_(gemv_tiled.pack(w, t.dtype))
        if getattr(layer, "dk_w_uk", None) is not None:
            for name, dst, tr in (("w_uk", layer.dk_w_uk, False), ("w_uv", layer.dk_w_uv_t, True)):
                sc = getattr(layer, name + "_scale", None)
                w = dequant(getattr(layer, name).data, sc.data if sc is not None else None, torch.float32)
                dst.data.copy_((w.t() if tr else w).to(dst.dtype))

    def _init_ep(self, moe_kernel: str) -> None:
        """This rank's routed experts under expert parallelism (MOE_EP): experts [g E / ep, (g + 1) E / ep) of its group
        g = rank // (tp / ep), each at its intermediate columns within the group (the loader reads the shard from
        _ep_first / _ep_tp), the MoE kernels' blobs packed from them as at tp alone."""
        cfg, tp, ep = self.cfg, self.tp_size, self.moe_ep
        E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
        sub = tp // ep
        if tp % ep or E % ep or I % sub:
            raise ValueError(f"moe_ep {ep}: tp {tp}, {E} experts and moe_intermediate_size {I} must divide by it "
                             f"(and I by tp / ep = {sub})")
        group, inner = self.tp_rank // sub, self.tp_rank % sub
        El, Im = E // ep, I // sub
        self.moe_inter = Im
        self.ep_lo = group * El
        for i, layer in enumerate(self.layers):
            if not layer.moe:
                continue
            ref = layer.router
            quant = self.keep_fp8 and cfg.is_quantized(f"model.layers.{i}.mlp.experts", True)
            bk = cfg.quant_expert_block

            def param(*shape, dt):
                return nn.Parameter(torch.empty(*shape, dtype=dt, device=ref.device), requires_grad=False)

            if quant:
                layer.w_gu, layer.w_gu_scale = param(El, 2 * Im, H, dt=FP8), param(El, 2 * Im, -(-H // bk), dt=torch.float32)
                layer.w_down, layer.w_down_scale = param(El, Im, H, dt=FP8), param(El, -(-Im // bk), H, dt=torch.float32)
            else:
                layer.w_gu, layer.w_down = param(El, 2 * Im, H, dt=self.dtype), param(El, Im, H, dt=self.dtype)
                layer.w_gu_scale = layer.w_down_scale = None
            layer.ep_experts = El
            layer._ep_first, layer._ep_tp = group * El, (inner, sub)
            if moe_kernel == "nki":
                from ..kernels import moe_dedupe

                if not moe_dedupe.supports(layer.w_gu, layer.w_gu_scale, layer.w_down, layer.w_down_scale, True):
                    raise ValueError(f"layer {i}: the NKI MoE kernels do not take this rank's experts "
                                     f"({El} x 2 x {Im} x {H}, {layer.w_gu.dtype})")
                layer.pack_tiles = layer.pack_moe = True
            elif moe_kernel != "xla":
                raise ValueError(f"moe_kernel must be xla or nki, not {moe_kernel!r}")
        self._ep_buffer(None)

    def _ep_buffer(self, device) -> None:
        """ep_lo_t: this rank's first expert as a buffer, so that every rank traces one graph (vocab_start_t's rule)."""
        self.register_buffer("ep_lo_t", torch.tensor(self.ep_lo, dtype=torch.int64, device="cpu").to(device),
                             persistent=False)

    def _rope_buffer(self, device) -> None:
        m = self.cfg.attn_layers[0].mla
        self.register_buffer("rope_cs", rope_table(m.qk_rope_head_dim, m.rope_theta, self.rope_positions).to(device),
                             persistent=False)

    def rank_buffers(self, device=None) -> None:
        super().rank_buffers(device)
        if getattr(self, "rope_positions", None):  # not yet during DecoderForCausalLM.__init__
            self._rope_buffer(device)
        if getattr(self, "moe_ep", 1) > 1:
            self._ep_buffer(device)
        if getattr(self, "w_a_lo", None) is not None:
            self._w_a_buffer(device)

    def _w_a_buffer(self, device) -> None:
        """w_a_lo_t: this rank's first column of decode's w_a shard (_w_a_sharded), a buffer (vocab_start_t's rule)."""
        self.register_buffer("w_a_lo_t", torch.tensor(self.w_a_lo, dtype=torch.int64, device="cpu").to(device),
                             persistent=False)

    load_attention = staticmethod(load_attention)

    def _rope(self, positions: torch.Tensor):
        """(cos, sin) [T, dr / 2] of positions [T], in the model dtype (transformers casts them to the activations')."""
        cs = self.rope_cs.index_select(0, positions.reshape(-1).clamp(0, self.rope_positions - 1)).to(self.dtype)
        half = cs.shape[-1] // 2
        return cs[:, :half], cs[:, half:]

    def _final(self, h: torch.Tensor) -> torch.Tensor:
        return rms_norm(h, self.norm, self.cfg.rms_norm_eps)

    def _hidden_in(self, input_ids: torch.Tensor, sp: bool = False) -> torch.Tensor:
        e = self._embed(input_ids)
        return self._sp_rows(e) if sp else e

    def _layer(self, layer, h, positions, slot_mapping, table, bias, state_slot=None, **mixin):
        """GlmMoeDsaDecoderLayer.forward: attention, then the MLP, each on a pre-norm residual."""
        if mixin:
            raise ValueError(f"hybrid.layer's {sorted(mixin)} reached a GLM-5.3 layer")
        eps = self.cfg.rms_norm_eps
        x = rms_norm(h, layer.in_norm, eps)
        h = h + self._attn_all_reduce(attention(self, layer, x, positions, slot_mapping, table, bias))
        return h + mlp(self, layer, rms_norm(h, layer.post_norm, eps), dec=table.dim() == 2)

    def _run_layers(self, h, positions, slot_mapping, attn, state_slot=None):
        bias, table = attn[None]
        self._sel = None
        try:
            for layer in self.layers:
                h = self._layer(layer, h, positions, slot_mapping, table, bias)
        finally:
            self._sel = None
        return h

    def group_fn(self, idxs):
        """DecoderForCausalLM.group_fn, with the selection carry cleared around each call: a graph's shared layers
        take a selection their own graph made, or the scratch."""
        f = super().group_fn(idxs)

        @functools.wraps(f)
        def g(*args, **kw):
            self._sel = None
            try:
                return f(*args, **kw)
            finally:
                self._sel = None

        return g

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Logits of a whole sequence without a cache (tests)."""
        T = input_ids.shape[0]
        positions = torch.arange(T, device=input_ids.device)
        h = self._hidden_in(input_ids)
        eps = self.cfg.rms_norm_eps
        self._sel = None
        try:
            for layer in self.layers:
                h = h + self._attn_all_reduce(reference(self, layer, rms_norm(h, layer.in_norm, eps), positions))
                h = h + mlp(self, layer, rms_norm(h, layer.post_norm, eps))
        finally:
            self._sel = None
        return self._head(self._final(h))
