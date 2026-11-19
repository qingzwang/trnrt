"""GLM-5.3's decode context parallelism (models/glm_dcp.py): with each rank of a group holding 1 / dcp of every
sequence's cache rows, greedy generation (prefill chunks, decode, a selection past index_topk keys, shared layers
reusing it) emits exactly tp 1's tokens, over the whole group (tp 4, dcp 4) and in row groups (tp 4, dcp 2); the local
candidate list cut short (M_LOCAL) still selects exactly where its certificate holds."""

import pytest

from tests.test_glm_moe_dsa import engine, prompts

pytest.importorskip("transformers.models.glm_moe_dsa.modeling_glm_moe_dsa")


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from tests.test_glm_moe_dsa import build

    path = str(tmp_path_factory.mktemp("glm_dcp"))
    build(path, layers=8)
    return path


def greedy(path, ps, n, **kw):
    from trnrt.engine.request import SamplingParams

    eng = engine(path, max_num_seqs=4, **kw)
    try:
        out = [r.output_ids for r in eng.generate(ps, SamplingParams(max_new_tokens=n, ignore_eos=True))]
        return out, eng
    finally:
        eng.close()


@pytest.mark.parametrize(("tp", "dcp"), [(2, 2), (4, 4), (4, 2)], ids=["tp2-dcp2", "tp4-dcp4", "tp4-dcp2-rows"])
def test_dcp_greedy_is_tp1_greedy(model_dir, tp, dcp, monkeypatch):
    from trnrt.models import glm_moe_dsa

    monkeypatch.setattr(glm_moe_dsa, "MOE_EP", 1)
    monkeypatch.setattr(glm_moe_dsa, "DCP", 1)  # the engine sets it on this rank too: restored after the test
    # the 41-token prompt selects (index_topk 16) from its first decode step on; 20 and 30 end in a part-filled prefill
    # chunk (chunks of 8), whose padded rows must write nothing a real token owns
    ps = prompts(21, (9, 41)) + prompts(24, (20, 30))
    want, _ = greedy(model_dir, ps, 10)
    got, eng = greedy(model_dir, ps, 10, tp=tp, dcp=dcp)
    assert eng.runner.lps == eng.runner.ps // dcp
    assert got == want


def test_dcp_with_a_short_candidate_list(model_dir, monkeypatch):
    """M_LOCAL 8 (each rank sends 8 candidates of its ~11 local keys at dcp 4, TRNRT_DCP_M_LOCAL): the selection stays
    exact where every cut list ends below the k-th value; the tokens equal tp 1's, and the certificate was checked."""
    from trnrt.models import glm_moe_dsa

    monkeypatch.setattr(glm_moe_dsa, "MOE_EP", 1)
    monkeypatch.setattr(glm_moe_dsa, "DCP", 1)  # the engine sets it on this rank too: restored after the test
    monkeypatch.setenv("TRNRT_DCP_M_LOCAL", "8")
    ps = prompts(22, (41,))
    want, _ = greedy(model_dir, ps, 6)
    got, eng = greedy(model_dir, ps, 6, tp=4, dcp=4)
    bad, rows = eng.model.dcp_inexact.tolist()  # rank 0's certificate counts: [rows that failed, rows checked]
    assert rows > 0 and bad == 0
    assert got == want


def test_dcp_with_mtp_and_the_front_kernel_path(tmp_path, monkeypatch):
    """dcp 4 at tp 4 (one head a rank) with MTP speculation (2 drafts) and the attention front kernel's path
    (TRNRT_GLM_FRONT_HOST=1): the verify batches and the MTP layer attend through the group too; tokens equal tp 1's
    plain greedy tokens."""
    from tests.test_glm_moe_dsa_mtp import build_mtp
    from trnrt.models import glm_moe_dsa

    monkeypatch.setattr(glm_moe_dsa, "MOE_EP", 1)
    monkeypatch.setattr(glm_moe_dsa, "DCP", 1)  # the engine sets it on this rank too: restored after the test
    path = str(tmp_path)
    build_mtp(path)
    ps = prompts(23, (9, 30))
    want, _ = greedy(path, ps, 10)
    monkeypatch.setenv("TRNRT_GLM_FRONT_HOST", "1")
    got, eng = greedy(path, ps, 10, tp=4, dcp=4, spec_method="mtp", spec_k=2)
    assert eng.spec_verifies > 0
    assert got == want
