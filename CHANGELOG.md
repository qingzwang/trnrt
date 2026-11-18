# Changelog

All notable changes to TrnRT are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/),
and versions follow [Semantic Versioning](https://semver.org/). Rebuild the compile cache for each release.

## [Unreleased]

### Added

- GLM-5.3 (`zai-org/GLM-5.3`, `GlmMoeDsaForCausalLM`): MLA with interleaved RoPE, a token-level DSA indexer whose selection
  the "shared" layers reuse, a plain residual and the shared-expert MoE, on the existing paged cache, scheduler, sampler
  and tensor parallelism (`trnrt/models/glm_moe_dsa.py`). It matches transformers 5.17.0 on CPU and serves on one
  trn2.48xlarge as one engine of 64 ranks; its numbers and limits are in
  [docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md#glm-53-early-not-yet-tuned).
- Expert parallelism of the routed experts, `--moe-ep` (GLM-5.3): groups of tp / ep ranks, each holding E / ep experts, so
  the NKI MoE kernels take GLM-5.3 at 64 ranks. `trnrt compile decisions --moe-ep` writes its decisions.
- `kernels/gemv_tiled.py`: a decode step's projections over weights pre-tiled at load (GLM-5.3's decode).
- `kernels/mla_decode.py`: GLM-5.3's absorbed-MLA decode attention for one head per rank, as one kernel.
- `kernels/glm_moe_half.py`: GLM-5.3's decode MoE block (router, top 8, routed experts under expert parallelism, shared
  expert) as one kernel, from `kernels/moe_half.py`'s.
- The engine serves models without recurrent state: a prefix hit resumes from any cached page, with no state
  checkpoints, and the host-memory tier holds KV pages only.
- `TRNRT_TEST_DEVICE=1`: the kernel tests' simulator calls run on a NeuronCore ([docs/DEVELOPMENT.md](docs/DEVELOPMENT.md#tests)).
- Neuron SDKs after 2.32: NKI kernels compile through libtorch-neuronx-lite's native backend, and the compile cache's key
  follows its compiler-argument functions.

## [0.1.0] - 2026-11-08

The first public release.

- GLM-5.3-Flash on one trn2.48xlarge: two engines of 32 tensor-parallel ranks behind a session-affine router, an
  OpenAI-compatible API, 1M-token context, a prefix cache with a host-memory tier, and NKI decode megakernels.
  These describe the configuration, not a measurement; [docs/RESULTS.md](docs/RESULTS.md) has the measured numbers with
  their context, and [docs/DEPLOY.md](docs/DEPLOY.md) the deployment.

### Added

- MTP speculative decoding with GLM-5.3-Flash's own MTP layer: 2 draft tokens a step, verified by the target in the next
  step, scheduled before the previous step is read back. Presets `mtp-always`, `mtp-batch8`, `mtp-batch12` and
  `mtp-batch24` ([docs/CONFIGURATION.md](docs/CONFIGURATION.md)), a dynamic mode that speculates only while few requests
  decode, and `trnrt:spec_decode_*` metrics. A benchmark mode (`--spec-synth-al`) reproduces InferenceX's synthetic
  acceptance; it is off by default ([docs/AGENTX.md](docs/AGENTX.md)).
- `TRNRT_DMA_HW`: the kernels' static DMA helpers in hardware DGE. The `mtp-*` presets set it, as in the builds behind
  the published MTP numbers; the other configurations leave it unset.
- Public release: README, docs, contributing and security policies, issue and pull request templates.
- CI: pre-commit (ruff lint and format), mypy, the CPU test suite, a package build check, Dockerfile lint and CodeQL.
  Release workflow.
- `tools/check_docs.py`: every relative Markdown link and image resolves.
- Branches: pull requests target `development` and are squash-merged; `main` takes promotions of `development` for
  releases only, as merge commits. Both need a maintainer's approval (`.github/CODEOWNERS`) and every review
  conversation resolved. Merging a promotion that bumps the version into `main` is the release: it tags `vX.Y.Z` on the
  merge commit and publishes, once per version (`tools/release_plan.py`).
- Contributor CI: actionlint and zizmor on the workflows (pre-commit); advisory PR checks (title format, a warning
  when traced code changes); dependency review; OpenSSF Scorecard; a weekly external-link check; a release dry run
  that builds the package and the container image.
- Area labels on every PR from the paths it changes (`.github/labeler.yml`), forks included.
- AI review on every ready-for-review PR into `development` (CodeRabbit, cubic, Greptile), never a required check.
- Contributing: a three-step quick start, issue forms for performance regressions and docs, Discussions for questions,
  `SUPPORT.md`.

### Changed

- A request without `temperature`, `top_p`, `top_k` or `min_p` takes the checkpoint's `generation_config.json`
  values, as vLLM does by default. For GLM-5.3-Flash that is top_p 0.95; it was 1.0.
- The display name is TrnRT. The package, the `trnrt` command, environment variables and metric names are unchanged.
- Server, engine and router messages go through Python `logging`, on stderr. Their text is unchanged, and
  `TRNRT_LOG_LEVEL` sets the level. They used to be printed, some on stdout. A hang report still reaches stderr when
  nothing has configured logging. The CLIs' own start-up status lines (`engine up in`, `bucket warmup`,
  `engine N ready after`) are still printed, as is one status line in the frozen `model_runner.py`.
- Code style: the ruff formatter and a strict ruff rule set over all host code, tests and tools, plus mypy with zero
  errors. All enforced in pre-commit and CI. Traced code (kernels, models) is excluded, so no compiled graph changes.

### Fixed

- The router matches and inserts a sessionless request's text in its prefix trees in a worker thread, not on its event
  loop, which every relayed stream shares: a 1.1 MB (~250K-token) request stalled the loop for ~300 ms; now ~12 ms.
  The routing decisions are unchanged.
- A non-streaming client that disconnects from the router now aborts its request on the engine. The router kept its
  own connection to the engine open, so the engine generated to `max_tokens` and held a decode seat.
- The router answers a request body that is not a JSON object with 400, as each engine does, instead of 500.
