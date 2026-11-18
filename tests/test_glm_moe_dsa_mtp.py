"""GLM-5.3's MTP speculative decoding (models/glm_moe_dsa.py with DecoderForCausalLM's MTP passes, the engine's blind
verify of engine/spec_async.py, on a model without recurrent state): greedy generation with 2 drafts a step emits exactly
the tokens of plain decoding, at tp 1 and tp 2, and the MTP layer is built from the checkpoint's model.layers.<n>."""

import json
import os

import pytest
import torch

from tests.test_glm_moe_dsa import INDEXER_TYPES, build, engine, prompts

pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")

LAYERS = 4  # the target's layers; the checkpoint's layer 4 (a full DSA layer with the MoE) is the MTP layer


def build_mtp(path, seed=0):
    """A random GLM-5.3-architecture checkpoint with one MTP layer: transformers builds LAYERS + 1 layers, the last is
    renamed into the MTP layer's place (model.layers.<LAYERS>, its own indexer and MoE), and the MTP head's enorm, hnorm,
    eh_proj and shared_head.norm are added; the config then counts LAYERS layers and one next-n layer."""
    from safetensors.torch import load_file, save_file

    types = INDEXER_TYPES[:LAYERS] + ["full"]
    build(path, seed=seed, layers=LAYERS + 1, indexer_types=types)
    f = os.path.join(path, "model.safetensors")
    sd = load_file(f)
    H = sd["model.norm.weight"].shape[0]
    g = torch.Generator().manual_seed(seed + 99)
    pre = f"model.layers.{LAYERS}."
    sd[pre + "enorm.weight"] = (1 + 0.1 * torch.randn(H, generator=g)).to(sd["model.norm.weight"].dtype)
    sd[pre + "hnorm.weight"] = (1 + 0.1 * torch.randn(H, generator=g)).to(sd["model.norm.weight"].dtype)
    sd[pre + "shared_head.norm.weight"] = sd["model.norm.weight"].clone()
    sd[pre + "eh_proj.weight"] = (torch.randn(H, 2 * H, generator=g) * (2 * H) ** -0.5).to(sd["model.norm.weight"].dtype)
    save_file(sd, f, metadata={"format": "pt"})
    c = json.load(open(os.path.join(path, "config.json")))
    c["num_hidden_layers"] = LAYERS
    c["num_nextn_predict_layers"] = 1
    for k, v in list(c.items()):  # every per-layer list (indexer_types, mlp_layer_types, layer_types) counts LAYERS
        if isinstance(v, list) and len(v) == LAYERS + 1:
            c[k] = v[:LAYERS]
    json.dump(c, open(os.path.join(path, "config.json"), "w"))


@pytest.fixture(scope="module")
def mtp_dir(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("glm_mtp"))
    build_mtp(path)
    return path


def test_the_mtp_layer_is_built_from_the_checkpoint(mtp_dir):
    from trnrt.models.config import ModelConfig
    from trnrt.models.loader import load_model

    cfg = ModelConfig.from_pretrained(mtp_dir)
    assert cfg.num_layers == LAYERS and cfg.mtp_layers == 1 and cfg.mtp_prefix == f"model.layers.{LAYERS}"
    m = load_model(mtp_dir, cfg, torch.float32, torch.device("cpu"), 64, mtp=True)
    assert m.mtp is not None and m.mtp.plain and m.mtp.moe and not m.mtp.spec.mla.index_shared
    assert m.mtp.idx_wq.shape[0] == 32 * 128  # the full-width indexer query (_fix_indexer)
    assert float(m.mtp_eh.abs().sum()) > 0 and m.mtp in m.kv_layers()


@pytest.mark.parametrize(("tp", "front"), [(1, False), (2, False), (4, True)], ids=["tp1", "tp2", "tp4-front"])
def test_speculative_greedy_is_plain_greedy(mtp_dir, tp, front, monkeypatch):
    """tp4-front: one head a rank, the verify and the MTP passes through the attention front kernel's path
    (kernels/glm_attn_front.py emulate(), TRNRT_GLM_FRONT_HOST=1) in both engines."""
    from trnrt.engine.request import SamplingParams
    from trnrt.models import glm_moe_dsa

    monkeypatch.setattr(glm_moe_dsa, "MOE_EP", 1)
    if front:
        monkeypatch.setenv("TRNRT_GLM_FRONT_HOST", "1")
    ps = prompts(12, (9, 30))
    sp = SamplingParams(max_new_tokens=12, ignore_eos=True)
    plain = engine(mtp_dir, max_num_seqs=2, tp=tp)
    try:
        want = [r.output_ids for r in plain.generate(ps, sp)]
    finally:
        plain.close()
    eng = engine(mtp_dir, max_num_seqs=2, tp=tp, spec_method="mtp", spec_k=2)
    try:
        got = [r.output_ids for r in eng.generate(ps, sp)]
        assert eng.spec_verifies > 0  # the steps were verifies, with drafts
    finally:
        eng.close()
    assert got == want
