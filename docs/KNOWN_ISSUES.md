# Known issues

## Verification status

The published numbers were measured with an earlier build of TrnRT. This repository passes these checks against that
build ([RESULTS.md](RESULTS.md)):

| Check | Result |
|---|---|
| Graph equivalence (`tools/graph_equiv.py`), all 32 ranks, default batch-12 configuration | EQUAL for all 157 graphs that build compiles |
| Outputs on the device | Byte-identical to that build's on 384 of 384 requests |
| The README's two-engine serve | Passes end to end |

## Prompt processing: a small share of sparse-attention selections are inexact

In the sparse-attention (DSA) layers each prompt token selects 512 blocks of its earlier context to attend to. The
context is split across the eight ranks of a context-parallel group, so the selection merges each rank's shortlist (its
top 120 blocks, `CP_LOCAL_K` in `trnrt/models/mla.py`) and then checks a certificate: that the merge provably holds the
true top set. Rows that fail it, up to 128 per chunk and row group (`CP_LOCAL_F`), are selected again exactly. Rows
past that keep the merged shortlist: it always contains the true top 120 blocks, but up to 392 of the 512 may differ
from the exact selection.

- **How often.** In a run with real coding agents, none of 13 million selections failed the certificate. On AgentX's
  synthetic prompt content (the runs at 4, 8 and 16 agents), 2.8% of selections failed it and 1.2% kept the merged
  shortlist; in the batch-size runs, 2.8%, 2.9% and 2.5% failed and 1.2%, 1.3% and 1.1% kept it at batch sizes 4, 8
  and 12.
- **Effect on outputs.** AgentX prompts of 48K, 180K and 420K tokens were run through the engine twice: with this
  selection, and with every failed selection redone exactly. Scored on the same text, the merged shortlist moved the
  model's next-token log-probabilities by 0.012 to 0.021 nats on average, and changed the most likely next token at 3
  of 448 positions. Generating 64 tokens greedily, 2 of 6 prompts took a different continuation, from the 6th and 18th
  token on; a greedy continuation can diverge from a single near-tie.
- **Effect on the published times.** The benchmark results in [RESULTS.md](RESULTS.md) ran with this selection. On the
  1.2% of selections that kept the merged shortlist, an exact re-selection would have added an estimated 0.4 to 0.7%
  to total prompt-processing time, and up to 29 to 50% on the worst-affected prompt chunks. Per-request impact was not
  measured, so on requests built from such chunks the first-token and end-to-end times may be optimistic.
- **Watching it.** Each engine counts these rows on rank 0's row group, warmup included: `/metrics` reports
  `trnrt:dsa_local_k_rows_total`, `trnrt:dsa_local_k_failed_rows_total` (rows that failed the certificate) and
  `trnrt:dsa_local_k_inexact_rows_total` (rows past the limit, which kept the merged shortlist), and the engine logs
  the same counts when it closes (stderr), EXACT only if no row went past the limit: at INFO when EXACT, at WARNING
  when NOT EXACT, so `TRNRT_LOG_LEVEL=WARNING` still shows an inexact run.

## GLM-5.3: early, not yet tuned

GLM-5.3 (`zai-org/GLM-5.3`, `GlmMoeDsaForCausalLM`) loads, prefills and decodes through the engine. It matches
transformers 5.17.0 on CPU (`tests/test_glm_moe_dsa.py`) and on the real checkpoint's first 4 layers (fp32, CPU). The
full model runs on one trn2.48xlarge as one engine of 64 ranks, from a cache built by `trnrt compile`.

Measured once each (no repeats), 2026-11-17 and 2026-11-18: trn2.48xlarge, the image of `docker/Dockerfile` (Neuron SDK
2.32, neuronx-cc 2.27), `zai-org/GLM-5.3` at ba32f8c19c8738e2c57c71798c601d477a7bf844, bf16 activations with the
checkpoint's FP8 weights, a bf16 KV cache, `--tp 64 --moe-kernel nki --moe-ep 2 --page-size 32 --max-num-seqs 4
--max-model-len 4096 --decode-buckets 1,4 --prefill-buckets 512 --page-buckets 16,128`, no speculation. Decode is the
steady-state step (the difference of a 72- and an 8-token greedy generation over 64-token prompts, so prefill cancels
out); prefill is one 2,000-token prompt that hits nothing in the prefix cache. Correctness was checked by reading two
greedy chat answers. The attention front kernel changes decode's roundings (its norms sum in another order), so its
greedy text leaves the previous row's at near-ties: forced along the previous build's tokens (3 prompts, 24 decode
steps), its top-1 agreed at every step, with chosen-token logprobs 0.018 apart on average (at most 0.27).

| Configuration | Decode, batch 1 | Decode, batch 4 | Prefill, 2,000 tokens |
|---|---|---|---|
| XLA MoE, tp 64 | 190 ms/step | 258 ms/step | 508 s |
| NKI MoE, tp 64 / ep 2 | 174.5 ms/step | 192.9 ms/step | 3.17 s |
| + decode projection kernel (`kernels/gemv_tiled.py`) | 33.9 ms/step | 48.6 ms/step | 3.04 s |
| + its weight loads on two DMA queues, the one-head MLA decode kernel (`kernels/mla_decode.py`) | 31.0 ms/step | 44.4 ms/step | 3.04 s |
| + decode's q_a / kv_a projection as tp column shards, no indexer query below index_topk keys | 29.3 ms/step | 42.9 ms/step | 3.04 s |
| + the decode MoE block (router, top 8, experts, shared expert) as one kernel (`kernels/glm_moe_half.py`) | 24.4 ms/step | 36.9 ms/step | 3.03 s |
| + the decode attention front (norm, q_a / kv_a shard and its all-reduce, q_b, RoPE, indexer key) as one kernel (`kernels/glm_attn_front.py`) | 19.2 ms/step | 28.0 ms/step | 3.03 s |
| + MTP speculative decoding, 2 drafts a step (`--spec-method mtp --spec-k 2`): mean acceptance length 2.58 (batch 1) and 2.72 (batch 4) | 16.9 ms a token (59.3 tok/s) | 16.8 ms a token a sequence (238.2 tok/s) | 3.16 s |
| + decode context parallelism over all 64 ranks (`--dcp 64 --page-size 64`, models/glm_dcp.py; XLA attention): a rank holds 1 / 64 of every sequence's KV, mean acceptance length 2.60 / 2.73 | 16.5 ms a token (60.8 tok/s) | 20.6 ms a token a sequence (194.5 tok/s) | 9.90 s |
| + decode context parallelism in 4 groups of 16 ranks (`--dcp 16 --page-size 64`), the selection by kernels/dsa_topk.py over the group's candidates: a rank holds 1 / 16 of every sequence's KV, mean acceptance length 2.67 / 2.77 | 14.4 ms a token (69.7 tok/s) | 18.9 ms a token a sequence (211.9 tok/s) | 4.39 s |

| Missing | Why it matters |
|---|---|
| Decode megakernels and context-parallel DSA | Flash's fused layer kernels assume NoPE MLA with a pooled indexer. GLM-5.3's attention is XLA: each rank holds every sequence's whole latent and attends over the whole context with a mask. |
| FP8 decode weights | The decode kernels read bf16 copies made at load (+1.5 GB a rank). GLM-5.3's FP8 block scales are not powers of two, so the copies cannot be FP8 without a second rounding of every weight. |
| Collectives and glue between kernels | A decode step is a few hundred kernel calls with XLA ops and 236 all-reduces between them (2 a layer, plus decode's sharded q_a / kv_a). The MoE block and the attention front are one kernel each; the attention itself (cache rows, the selection above index_topk keys, MLA, o_proj) is still several kernels and XLA ops, and the block outputs' all-reduces are XLA's. A back kernel (MLA, o_proj, the all-reduce inside the kernel, the residual and the post-attention norm in one call) measured slower, 19.6 / 29.6 ms/step at batch 1 / 4, so it is not used. Its first form, with program 1 writing half of the all-reduce's send buffer, decoded garbage on trn2 although the simulator computed it right. |
| Sequence-parallel prefill streams | Off for GLM-5.3: every rank runs every row of a chunk. |
| MTP tuning | A verify step (3 rows a sequence, then the MTP layer's draft passes) costs about 2.3 plain decode steps at batch 1, so MTP gains 14% there and 67% at batch 4. The checkpoint's `index_share_for_mtp_iteration` is not used: each draft pass runs the MTP layer's own indexer, which changes drafts, never the verified tokens. |
| FP8 KV precision | The FP8 cache rounds the rope key and the indexer key to e4m3 (scale 1). On a random tiny model that changes greedy output from the first token on. Not measured on the real weights. |
| Long contexts | Measured at up to 4,096 tokens only. Without `--dcp` every rank holds every sequence's whole KV (93 KB a token: 23 GiB a rank at 256K tokens, past a NeuronCore's HBM with the weights); with `--dcp 64` a rank holds 1 / 64 of it. |
| DCP prefill and kernels | Under `--dcp` the attention is XLA (no MLA decode kernel), and a prefill chunk all-gathers the group's heads' queries and all-reduces their partial outputs: a 2,000-token prefill takes 4.39 s at dcp 16 (9.90 s at dcp 64 with the selection in XLA top-k) against 3.16 s without. Under `--dcp` neuronx-cc 2.27 compiles the MTP board graph at -O1 (at -O2 it failed with NCC_ITEN406 / NCC_ILSA901 at dcp 16, batch 4). A selection is exact where its certificate holds (each rank's cut candidate list ends below the k-th value; `dcp_inexact` counts the rows that fail). |

## Kernels on Neuron SDKs after 2.32

On the Neuron vLLM venv's libtorch-neuronx-lite 2.13 (neuronx-cc 2.28), trnrt compiles NKI kernels through its native
backend (`platform.compile_backend`); the XLA backend's lowering does not find their registration. 19 of the 22 kernel test
files pass on a NeuronCore there (`TRNRT_TEST_DEVICE=1`). `kda_decode` and the FFN half at grid 2, and the NKI MoE inside
a whole Flash decode graph, fail to compile with neuronx-cc's internal error NCC_ILMM003 ("Matmult psum accumulation
flags need to be all set or all unset"). They compile and run on SDK 2.32. The native backend also ignores
`NEURON_LIBTORCH_ASSERT_CACHE_HIT`, so a start without `--allow-compile` compiles a missing graph instead of failing.

## Batch 12 beyond 32 agents: the host-side prompt cache fills

Past its peak load, batch size 12 runs into the host-side tier of the prompt cache. In a run at 48 agents, one
instance's host tier was full before the measured hour began (7,339 of 7,340 pages) and had dropped 93,548 pages, so
more of each prompt was processed again: the share read from cache fell from 95% to 88%, and throughput from 3,720 to
2,642 tokens per second per chip. That run was cut at 44 of its 60 minutes, so it is not a valid AgentX submission and
is not in the results. Past that point, cache capacity, not the batch size, sets the limit. The host tier's size is
`--host-cache-gb` (4 GiB per rank by default); a larger tier has not been measured.

## `tests/test_dsa_long.py::test_cp_engine_matches_replicated`: logprobs outside 1e-4

On the CPU test model, context parallelism at tp 2 and 4 is compared with the replicated long path at tp 1. Every greedy
token agrees. The logprobs differ by up to 2.2e-3 at tp 2 and 1.3e-3 at tp 4, against the test's absolute tolerance of
1e-4, on x86 CPUs.

What is known:

- The failure is not new: an earlier version of this code fails the same tests with the same values.
- The difference is spread over nearly every logprob (9 of a request's 10), not concentrated in one token.
- It does not move with the CPU math library's threading or its reproducible mode: 2.17e-3 to 2.19e-3 at tp 2.
- The comparison runs with the FP8 KV cache, which context parallelism requires. Under FP8, a change in summation order
  that moves a cached value across a rounding boundary changes it by a whole FP8 step. The replicated path alone, with only
  its prefill chunk size changed (16 tokens to 12), already moves logprobs by up to 5.7e-4, above the 1e-4 tolerance.

Not shown: that context parallelism adds nothing beyond that ordering noise. Its difference is about 4 times the
chunk-size floor, so the tolerance is not loosened. The test is marked as an expected failure, strict: if a change makes it
pass, the suite fails until this entry and the mark are revisited.

This CPU test does not measure served accuracy. The served engine's accuracy was checked on the device, against a
high-precision reference, with the build the published numbers were measured on. On the device, this repository's
outputs were byte-identical to that build's on 384 of 384 requests ([Verification status](#verification-status)).
