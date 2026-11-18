"""GLM-5.3 (`glm_moe_dsa`: MLA with interleaved RoPE, a DSA indexer over tokens shared across layers, a plain residual,
shared-expert MoE) against transformers' GlmMoeDsaForCausalLM, on CPU in fp32 with identical weights.

Every model is built by transformers itself from GLM-5.3's published geometry
(https://huggingface.co/zai-org/GLM-5.3/raw/ba32f8c19c8738e2c57c71798c601d477a7bf844/config.json) with fewer layers,
heads and experts, a small hidden size and vocabulary: MLA (q_lora 2048, kv_lora 512, 192 nope + 64 rope + 256 value
dims, RoPE theta 8e6, interleaved), the indexer (32 heads of 128 dims, RoPE on the first 64), the published
indexer_types pattern (full, full, full, shared, shared, shared, full, shared: layers 0-7), 3 dense layers then MoE
(sigmoid, bias, scale 2.5, one shared expert). index_topk is cut to SPARSE_TOPK so that short sequences select.

The indexer keeps its 32 heads: with few heads a token's score is exactly 0 (every head's ReLU zero) often enough to
tie at the top-k boundary (tests/test_glm5_next.py). transformers' own choice among exact ties is torch.topk's, which on
CPU is not lowest-index-first, so the reference runs with Tensor.topk replaced by a stable sort (lowest index first among
ties, kernels/dsa_topk.py's rule); every other value is untouched.

The helpers (build, engine, prompts) are shared by the engine tests.
"""

import contextlib
import os

import pytest
import torch

pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")

SPARSE_TOPK = 16
INDEXER_TYPES = ["full", "full", "full", "shared", "shared", "shared", "full", "shared"]


def text_config(layers=8, **extra) -> dict:
    c = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "hidden_size": 128,
        "intermediate_size": 256,
        "moe_intermediate_size": 128,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 384,
        "n_routed_experts": 8,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "first_k_dense_replace": 3,
        "q_lora_rank": 2048,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 192,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_topk": SPARSE_TOPK,
        "indexer_types": INDEXER_TYPES[:layers],
        "rope_interleave": True,
        "indexer_rope_interleave": True,
        "rope_parameters": {"rope_theta": 8000000, "rope_type": "default"},
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 1024,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "hidden_act": "silu",
        "pad_token_id": None,
        "eos_token_id": 1,
        "num_nextn_predict_layers": 0,
    }
    c.update(extra)
    return c


def randomize(model, seed=0):
    """Weights at scales that exercise every piece: unit-variance matmul outputs, routing biases that move the
    expert choice, LayerNorm biases on the indexer key."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, p in model.named_parameters():
            r = torch.randn(p.shape, generator=g)
            if "norm" in n and n.endswith("weight"):
                p.copy_(1 + 0.1 * r)
            elif n.endswith("bias"):
                p.copy_(0.1 * r)
            elif "embed_tokens" in n:
                p.copy_(r)
            else:
                p.copy_(r * p.shape[-1] ** -0.5)
        for n, b in model.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(0.5 * torch.randn(b.shape, generator=g))


def build(path, seed=0, layers=8, **extra):
    """A random GLM-5.3-architecture model saved to `path` by transformers; returns the HF model."""
    import transformers as tf

    c = text_config(layers, **extra)
    cfg = tf.GlmMoeDsaConfig(**{k: v for k, v in c.items() if k != "architectures"})
    cfg.architectures = c["architectures"]
    cfg._attn_implementation = "eager"
    model = tf.GlmMoeDsaForCausalLM(cfg)
    randomize(model, seed)
    model.eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


@contextlib.contextmanager
def stable_topk():
    """Tensor.topk as a stable descending sort: the lowest index first among ties (module docstring)."""
    orig = torch.Tensor.topk

    def topk(self, k, dim=-1, largest=True, sorted=True):  # noqa: A002  torch's own parameter name
        if not largest:
            return orig(self, k, dim, largest, sorted)
        v, i = torch.sort(self, dim=dim, descending=True, stable=True)
        return torch.return_types.topk((v.narrow(dim, 0, k), i.narrow(dim, 0, k)))

    torch.Tensor.topk = topk
    try:
        yield
    finally:
        torch.Tensor.topk = orig


def hf_logits(model, ids):
    with torch.no_grad(), stable_topk():
        return model(torch.tensor([ids]), use_cache=False).logits[0].float()


def hf_greedy(model, ids, n):
    out = list(ids)
    with torch.no_grad(), stable_topk():
        for _ in range(n):
            out.append(int(model(torch.tensor([out]), use_cache=False).logits[0, -1].argmax()))
    return out[len(ids) :]


def ours(path, layers=None):
    from trnrt.models.config import ModelConfig
    from trnrt.models.loader import load_model

    cfg = ModelConfig.from_pretrained(path)
    if layers is not None:
        cfg = cfg.truncated(layers)
    return load_model(path, cfg, torch.float32, torch.device("cpu"), 256)


def engine(path, **kw):
    """A CPU engine on the model at `path`: fp32, pages of 4 tokens, prefill chunks of 8."""
    from trnrt.config import EngineConfig
    from trnrt.engine.engine import LLMEngine

    base = {
        "model_path": path,
        "device": "cpu",
        "dtype": torch.float32,
        "page_size": 4,
        "num_pages": 256,
        "max_num_seqs": 3,
        "max_model_len": 256,
        "max_prefill_tokens": 8,
    }
    base.update(kw)
    return LLMEngine(EngineConfig(**base))


def prompts(seed, lengths, vocab=384):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(2, vocab, (n,), generator=g).tolist() for n in lengths]


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("glm53"))
    hf = build(path)
    return path, hf


def test_config_reads_the_published_geometry(model_dir):
    from trnrt.models.config import ModelConfig
    from trnrt.models.glm_moe_dsa import RopeMLASpec

    path, _ = model_dir
    cfg = ModelConfig.from_pretrained(path)
    assert cfg.hybrid is None and cfg.moe_layers == (3, 4, 5, 6, 7)
    specs = cfg.attn_layers
    assert all(isinstance(s.mla, RopeMLASpec) for s in specs)
    assert [s.mla.index_shared for s in specs] == [t == "shared" for t in INDEXER_TYPES]
    m = specs[0].mla
    assert (m.qk_nope_head_dim, m.qk_rope_head_dim, m.v_head_dim, m.kv_lora_rank) == (192, 64, 256, 512)
    assert m.softmax_scale == pytest.approx(256**-0.5) and m.norm_eps == 1e-6 and m.rope_theta == 8e6
    # K: the latent then the rope key; V: the indexer key on a full layer, nothing on a shared one
    assert [(s.head_dim, s.v_head_dim) for s in specs[2:4]] == [(576, 128), (576, 0)]


@pytest.mark.parametrize("n", [12, 40], ids=["dense", "sparse"])
def test_logits_match_transformers(model_dir, n):
    """n = 12 holds at most index_topk tokens (dense causal MLA everywhere); n = 40 selects 16 of up to 40 per query,
    and the shared layers reuse it."""
    path, hf = model_dir
    ids = prompts(1, [n])[0]
    ref = hf_logits(hf, ids)
    with torch.no_grad():
        got = ours(path).forward_logits(torch.tensor(ids))
    assert got.shape == ref.shape
    err = (got - ref).abs().max().item() / ref.abs().max().item()
    assert err < 1e-4, err
    assert torch.equal(got.argmax(-1), ref.argmax(-1))


def test_shared_layers_attend_with_their_full_layers_selection(model_dir, monkeypatch):
    """Each shared layer attends with the very selection the full layer before it made (indexer_types: layers 3-5 take
    layer 2's, layer 7 takes layer 6's); a full layer makes its own."""
    from trnrt.models import glm_moe_dsa

    path, _ = model_dir
    m = ours(path)
    made = []
    orig = glm_moe_dsa._selection

    def spy(model, layer, *a):
        sel = orig(model, layer, *a)
        made.append((layer.spec.mla.index_shared, sel))
        return sel

    monkeypatch.setattr(glm_moe_dsa, "_selection", spy)
    with torch.no_grad():
        m.forward_logits(torch.tensor(prompts(2, [40])[0]))
    assert [s for s, _ in made] == [t == "shared" for t in INDEXER_TYPES]
    full = None
    for shared, sel in made:
        if shared:
            assert sel is full
        else:
            assert sel is not full
            full = sel


def test_paged_chunked_greedy_matches(model_dir, monkeypatch):
    """Prompts of 1 to 61 tokens prefilled in chunks of 8, four requests for three running slots, batched decode over
    page buckets past index_topk; nothing leaks. On the CPU each MoE layer's prefill is a graph of its own
    (model_runner.piecewise_groups), so the shared MoE layers read their full layer's selection from the scratch,
    while a decode step (one graph) passes it in the graph's carry."""
    from trnrt.engine.request import SamplingParams
    from trnrt.models import mla

    path, hf = model_dir
    eng = engine(path)
    reads = []
    shared = mla._shared
    monkeypatch.setattr(mla, "_shared", lambda *a: reads.append(a[1:]) or shared(*a))
    ps = prompts(3, (5, 19, 61, 1))
    want = [hf_greedy(hf, p, 10) for p in ps]
    reqs = eng.generate(ps, SamplingParams(max_new_tokens=10, ignore_eos=True))
    assert [r.output_ids for r in reqs] == want
    assert reads  # the shared layers of the prefill graphs read the scratch
    assert eng.runner.state is None and eng.scheduler.stateless
    assert eng.pool.num_free + eng.radix.total_pages() == eng.pool.num_usable


def test_prefix_hit_resumes_from_any_cached_page(model_dir):
    """Without recurrent state a hit resumes from the whole cached KV prefix (no checkpoint needed): a second request
    sharing 40 of its tokens with a finished one computes only the rest, and both match transformers."""
    from trnrt.engine.request import SamplingParams

    path, hf = model_dir
    eng = engine(path)
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True)
    first = prompts(4, (45,))[0]
    second = first[:40] + prompts(5, (7,))[0]
    (a,) = eng.generate([first], sp)
    (b,) = eng.generate([second], sp)
    assert a.output_ids == hf_greedy(hf, first, 8)
    assert b.output_ids == hf_greedy(hf, second, 8)
    assert b.num_cached_tokens == 40  # every whole page of the shared prefix (pages of 4)
    assert eng.radix.num_ckpts == 0


def test_fp8_kv_cache_is_the_rounded_reference(model_dir, monkeypatch):
    """The served cache form: K ([c, k_pe]) and V (the indexer key) in FP8 e4m3 (scale 1, clamped to the format's max),
    nothing for a shared layer's V. Greedy decoding equals the cache-free reference with exactly those values rounded
    through e4m3: the FP8 cache changes nothing else. (On this random model the rounding alone moves the argmax from
    the first token on, so the fp32 reference is not the comparison.)"""
    from trnrt.engine.request import SamplingParams
    from trnrt.models import glm_moe_dsa

    path, _ = model_dir
    eng = engine(path, kv_cache_dtype="fp8")
    v = [layer.v_cache.shape[-1] for layer in eng.model.kv_layers()]
    assert v == [0 if t == "shared" else 128 for t in INDEXER_TYPES]
    p = prompts(6, (41,))[0]
    (got,) = eng.generate([p], SamplingParams(max_new_tokens=6, ignore_eos=True))
    fmax = eng.model.fp8_max

    def r8(x):
        return x.float().clamp(-fmax, fmax).to(torch.float8_e4m3fn).to(x.dtype)

    project, indexer = glm_moe_dsa._project, glm_moe_dsa._indexer

    def rounded_project(*a):
        q_nope, q_pe, c, k_pe, q_resid, cs = project(*a)
        return q_nope, q_pe, r8(c), r8(k_pe), q_resid, cs

    def rounded_indexer(*a):
        q, k, w = indexer(*a)
        return q, r8(k), w

    monkeypatch.setattr(glm_moe_dsa, "_project", rounded_project)
    monkeypatch.setattr(glm_moe_dsa, "_indexer", rounded_indexer)
    m, ids, want = ours(path), list(p), []
    with torch.no_grad():
        for _ in range(6):
            want.append(int(m.forward_logits(torch.tensor(ids))[-1].argmax()))
            ids.append(want[-1])
    assert got.output_ids == want


def test_host_tier_restores_evicted_pages(model_dir):
    """A device pool too small to keep a finished prompt evicts it to the host tier (no checkpoint part without
    recurrent state); the next request on that prefix copies the pages back and matches transformers."""
    from trnrt.engine.request import SamplingParams

    path, hf = model_dir
    eng = engine(path, num_pages=24, host_cache_gb=0.01)
    assert eng.host_tier is not None and eng.host_tier.ckpt_capacity == 0
    sp = SamplingParams(max_new_tokens=4, ignore_eos=True)
    first, other = prompts(8, (41, 60))
    eng.generate([first], sp)
    eng.generate([other], sp)  # evicts the first prompt's pages to the host
    (again,) = eng.generate([first], sp)
    assert eng.host_tier.restored > 0
    assert again.num_cached_tokens > 0
    assert again.output_ids == hf_greedy(hf, first, 4)


def test_tp2_matches_tp1(model_dir):
    """MLA heads, the MLP and the experts' intermediate split over two ranks; the latent, the rope key and the indexer
    replicated."""
    from trnrt.engine.request import SamplingParams

    path, _ = model_dir
    ps = prompts(9, (11, 41))
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True)
    kw = {"max_num_seqs": 2}
    want = [r.output_ids for r in engine(path, **kw).generate(ps, sp)]
    two = engine(path, tp=2, **kw)
    try:
        got = [r.output_ids for r in two.generate(ps, sp)]
    finally:
        two.close()
    assert got == want


def test_no_sequence_parallel_buffers(model_dir):
    """Sequence-parallel prefill streams are off, so their rank buffers are not registered: loader.load_model moves
    only what rank_buffers re-registers, and on a device they were left on the host."""
    from trnrt.models.config import ModelConfig
    from trnrt.models.glm_moe_dsa import GlmMoeDsaForCausalLM

    path, _ = model_dir
    m = GlmMoeDsaForCausalLM(ModelConfig.from_pretrained(path), torch.float32, 64, tp_rank=1, tp_size=2)
    assert not m.prefill_sp
    assert not {"sp_index", "sp_onehot"} & {n for n, _ in m.named_buffers()}


def test_xla_selection_keeps_the_top_visible_keys():
    """The device's torch.topk form of the selection (DEVICE_TOPK "xla") keeps the same keys as the kernel's exact rule
    when there are no ties, and every visible key when fewer than keep are visible."""
    from trnrt.models.glm5_next import block_mask
    from trnrt.models.glm_moe_dsa import _xla_selection

    g = torch.Generator().manual_seed(0)
    B, Q, L, keep = 2, 5, 40, 16
    index = torch.randn(B, Q, L, generator=g)
    pos = torch.tensor([3, 10, 20, 30, 39])
    vis = torch.where(torch.arange(L).view(1, 1, L) <= pos.view(1, Q, 1), 0.0, -1e30).expand(B, Q, L)
    got = _xla_selection(index, vis, keep) + vis
    want = block_mask(index, vis, 1, keep, False) + vis
    assert torch.equal(got > -1e29, want > -1e29)


@pytest.mark.parametrize(("tp", "ep"), [(2, 2), (4, 2)], ids=["tp2-ep2", "tp4-ep2"])
def test_expert_parallel_matches_tp1(model_dir, tp, ep, monkeypatch):
    """Expert parallelism (glm_moe_dsa.MOE_EP): ep groups of tp / ep ranks, each holding E / ep experts at its columns
    within the group; a pair routed to the other group runs as local expert 0 with weight 0, and the MLP's all-reduce
    sums the groups. Greedy decoding equals tp 1's. (An engine sets the route on every build, engine.set_decode_routes;
    monkeypatch restores it for the tests that build a model without one.)"""
    from trnrt.engine.request import SamplingParams
    from trnrt.models import glm_moe_dsa

    monkeypatch.setattr(glm_moe_dsa, "MOE_EP", 1)

    path, _ = model_dir
    ps = prompts(10, (11, 41))
    sp = SamplingParams(max_new_tokens=8, ignore_eos=True)
    kw = {"max_num_seqs": 2}
    want = [r.output_ids for r in engine(path, **kw).generate(ps, sp)]
    eng = engine(path, tp=tp, moe_ep=ep, **kw)
    try:
        layer = next(x for x in eng.model.layers if x.moe)
        assert eng.model.moe_ep == ep and layer.ep_experts == 8 // ep
        assert tuple(layer.w_gu.shape[:2]) == (8 // ep, 2 * 128 // (tp // ep))
        got = [r.output_ids for r in eng.generate(ps, sp)]
    finally:
        eng.close()
    assert got == want


def test_expert_parallel_is_refused_for_other_models(tmp_path):
    """moe_ep is GLM-5.3's: another architecture refuses it instead of ignoring it."""
    from trnrt.config import EngineConfig
    from trnrt.engine.engine import LLMEngine

    with open(os.path.join(os.path.dirname(__file__), "fixtures", "GLM-5.3-Flash-config.json")) as f:
        (tmp_path / "config.json").write_text(f.read())
    with pytest.raises(ValueError, match=r"GLM-5\.3 only"):
        LLMEngine(EngineConfig(model_path=str(tmp_path), device="cpu", moe_ep=2, tp=2))


def test_decode_reads_pretiled_copies_prefill_graphs_do_not(model_dir):
    """Decode's weight copies (glm_moe_dsa.DECODE_COPIES): each dt_<name> is its weight dequantized and packed for the
    decode projection kernel (kernels/gemv_tiled.py), read only by the whole-step decode graph, so no prefill layer
    graph's tensor list holds one."""
    from trnrt.kernels import gemv_tiled
    from trnrt.models.quant import dequant

    path, _ = model_dir
    m = ours(path)
    full, shared = m.layers[0], m.layers[3]
    for layer, name in ((full, "w_a"), (full, "idx_wq"), (full, "gate_up"), (shared, "router"), (shared, "o")):
        w = getattr(layer, name)
        sc = getattr(layer, name + "_scale", None)
        t = getattr(layer, "dt_" + name)
        assert torch.equal(
            t, gemv_tiled.pack(dequant(w.data, sc.data if sc is not None else None, torch.float32), t.dtype)
        )
    assert not hasattr(shared, "dt_idx_wq")  # a shared layer has no indexer
    for i in range(len(m.layers)):
        names = set(m._layer_split(i)[1])
        assert names and not any(n.startswith("dt_") for n in names)
