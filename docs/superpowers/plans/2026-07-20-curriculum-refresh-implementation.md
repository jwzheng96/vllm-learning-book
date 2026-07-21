# vLLM Curriculum Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Review all 50 existing chapters against one current official vLLM main commit, add 10 complete beginner/production/interview chapters, and make the 60-chapter book buildable, traceable, practical, and interview-ready.

**Architecture:** The source-sync plan provides a pinned commit, semantic links, chapter inventory, impact report, and pending review ledger. This plan processes content in coherent source domains, upgrades review entries only after evidence checks, adds ten full chapters, then switches every CI and deployment gate from contract-only to full semantic validation. A final upstream refresh rechecks only newly affected chapters before release evidence is recorded.

**Tech Stack:** Markdown, Mermaid, shell/curl examples, Python 3.9 documentation tooling, vLLM V1 source at the pinned submodule SHA, official documentation from that same checkout, `unittest`, HTML/PDF/EPUB builders, GitHub Actions.

## Global Constraints

- Complete `2026-07-20-source-sync-implementation.md` Tasks 1-7 first; this plan consumes its CLI, lock, inventories, and workflows.
- All project commands, tests, builds, installs, and mutating Git commands run on `rlocal` at `/Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning`.
- All vLLM source and official-documentation facts come from the commit in `source.lock.json`; current web documentation may supplement but must not override the pinned source.
- Use V1 as the canonical implementation. Mention V0 only for compatibility or migration and label it explicitly.
- Never invent benchmark values or label expected behavior as measured hardware output.
- A chapter is reviewed only when its semantic links resolve, source-area diff was examined, commands/metrics/diagrams were checked, and its review ledger entry points to the current lock SHA.
- Every command block states working directory, prerequisites, success evidence, and cleanup or rollback.
- Every configuration recommendation states workload, expected metric movement, risk, and rollback condition.
- No hand edits to `_site/` or `vllm-learning-html/`.
- The curriculum design is `docs/superpowers/specs/2026-07-20-curriculum-refresh-design.md` and is authoritative for scope and acceptance.

---

## Shared Chapter Review Protocol

Every content task uses this exact protocol before marking a review complete:

1. Read the full chapter, every semantic source target, every file matching its `source_areas`, and the chapter's rows in `artifacts/source-sync/latest-impact.md`.
2. Search the pinned submodule for every named class, function, CLI option, environment variable, metric, default, backend, and failure behavior.
3. Classify every technical claim as pinned-source fact, official-doc fact, paper fact, measured result, or engineering heuristic; add the appropriate evidence or qualification.
4. Correct the prose, Mermaid data flow, tables, code excerpts, self-check answers, and interview answers together.
5. Ensure the chapter has: reader, prerequisites, time, difficulty/environment, observable outcomes, current source trail, experiment or trace exercise, production tradeoff, failure evidence, summary, self-check, interview expression, and next step.
6. Run source-contract validation and build HTML into a temporary directory.
7. Set the chapter review entry to `status="reviewed"`, current full SHA, current UTC time, and the four booleans true only after their evidence exists. Leave `hardware_verified=[]` unless a real indexed hardware run was performed.
8. Run full validation for only reviewed entries plus contracts; do not suppress pending chapters until the final task.

The review note for an unmeasured GPU chapter must include `"static source review complete; current-SHA GPU run not performed"`.

---

### Task 1: Review Inventory Report and Book-Level Learning Paths

**Files:**
- Create: `artifacts/content-review/baseline-audit.md`
- Modify: `README.md`
- Modify: `curriculum.toml`
- Modify: `content-review.toml`
- Modify: `build_html.py`
- Modify: `build_pdf_epub.py`
- Create: `tests/source_sync/test_build_inventory.py`

**Interfaces:**
- Consumes: 50-chapter inventory, pinned lock, pending review ledger.
- Produces: baseline audit, inventory-driven chapter discovery, and four reader-visible learning paths.

- [ ] **Step 1: Add failing inventory/build tests**

Test that both builders discover chapters in `curriculum.toml` order, reject a missing listed file, reject an unlisted chapter file, and calculate chapter count from inventory. The shared assertion is:

```python
self.assertEqual(
    [path.relative_to(repo_root).as_posix() for path in discover_chapter_files(repo_root)],
    [chapter.path.as_posix() for chapter in load_curriculum(repo_root / "curriculum.toml")],
)
```

Move shared discovery to `tools/source_sync/inventory.py`; both builders import it. README remains first for PDF/EPUB and is not counted as a chapter.

- [ ] **Step 2: Run tests to verify the old hard-coded discovery fails**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest tests.source_sync.test_build_inventory -v'
```

Expected: failure because builders do not yet consume `curriculum.toml`.

- [ ] **Step 3: Make builders inventory-driven**

Add `discover_chapter_files(repo_root: Path) -> tuple[Path, ...]` to `inventory.py`. Replace both builders' hard-coded section glob loops with this function. Keep `SECTIONS_META` only for display labels. In `combine_files`, derive the subtitle from the actual chapter count and total source line count; remove the stale literal `15K+`.

- [ ] **Step 4: Generate the baseline content audit**

Create `artifacts/content-review/baseline-audit.md` with exact sections:

- Locked Source Version;
- Inventory Reconciliation (`50` source chapters before additions, README's old `49` claim);
- Pending Reviews by Part;
- Affected Chapters from Upstream Diff;
- Unmanaged or Unresolved Source Contracts;
- Named Symbols/Flags/Metrics Requiring Review;
- GPU Verification Availability;
- Planned Ten Chapters.

Populate every list from CLI output and repository scans; do not manually estimate counts.

- [ ] **Step 5: Rewrite README's navigation around four tracks**

Keep existing chapter links, but replace stale totals and add exact tracks:

- `30 分钟理解` for prerequisites, what-is-vLLM, architecture, and first API service;
- `源码主线` from entry/input through scheduler/KV/runner/attention/sampling/output;
- `工业实战` from setup through benchmark/tuning/deployment/SLO/security/upgrade/capstone;
- `面试冲刺` from common questions through calculations, system design, troubleshooting, and mock rubric.

Display the lock SHA and validation time through the generated version block. State that hardware badges require indexed runs.

- [ ] **Step 6: Run tests/build and commit the book-level foundation**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest discover -s tests/source_sync -v && python3 -m tools.source_sync validate --profile contracts && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && test "$(find "$site_dir" -name "*.html" | wc -l | tr -d " ")" -ge 51'
```

Expected: tests/contracts pass and the site contains README plus all 50 current chapter pages.

Commit:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add artifacts/content-review/baseline-audit.md README.md curriculum.toml content-review.toml build_html.py build_pdf_epub.py tools/source_sync/inventory.py tests/source_sync/test_build_inventory.py && git commit -m "docs: establish inventory-driven learning paths"'
```

---

### Task 2: Beginner Foundations and Core Concepts Review

**Files:**
- Modify: `01-overview/00-prerequisites.md`
- Modify: `01-overview/01-what-is-vllm.md`
- Modify: `01-overview/02-architecture.md`
- Modify: `01-overview/03-v0-vs-v1.md`
- Modify: `01-overview/04-project-structure.md`
- Modify: `01-overview/05-process-and-ipc-internals.md`
- Modify: `02-core-concepts/01-paged-attention.md`
- Modify: `02-core-concepts/02-continuous-batching.md`
- Modify: `02-core-concepts/03-kv-cache-management.md`
- Modify: `02-core-concepts/04-prefix-caching.md`
- Modify: `02-core-concepts/05-chunked-prefill.md`
- Modify: `content-review.toml`
- Modify: `artifacts/content-review/baseline-audit.md`

**Interfaces:**
- Consumes: pinned V1 engine/config/scheduler/KV source areas and Shared Chapter Review Protocol.
- Produces: 11 reviewed beginner/core chapters with consistent terminology and current source evidence.

- [ ] **Step 1: Review the six overview chapters**

Verify and correct:

- token/KV/TTFT/TPOT/ITL and TP/PP/DP definitions;
- current public entry points and V1 process/IPC topology;
- V0/V1 status and migration language;
- project structure counts and module map;
- `EngineCore`, core client, executor, worker, ZMQ/shared-memory message flow.

Add one request-lifecycle Mermaid that uses the exact current class names and one no-GPU trace exercise that asks the reader to locate each contract without executing vLLM.

- [ ] **Step 2: Review the five core-concept chapters**

Verify PagedAttention block terminology, continuous batching semantics, block pool allocation/free/refcount behavior, prefix hashing inputs/collision checks, and chunked-prefill scheduling/defaults. Every default value must link to current config source; conceptual paper claims must remain distinguished from current implementation.

- [ ] **Step 3: Update review entries and validate this batch**

Update exactly the 11 corresponding `[[review]]` entries. Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && test -s "$site_dir/01-overview/02-architecture.html" && test -s "$site_dir/02-core-concepts/04-prefix-caching.html"'
```

Expected: contracts/build pass; only untouched chapters remain pending in full validation.

- [ ] **Step 4: Commit foundations/core review**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add 01-overview 02-core-concepts content-review.toml artifacts/content-review/baseline-audit.md && git commit -m "docs: refresh vLLM foundations and core concepts"'
```

---

### Task 3: Request Entry, Scheduling, and KV Source Review

**Files:**
- Modify: `03-code-walkthrough/01-entry-points.md`
- Modify: `03-code-walkthrough/02-scheduler.md`
- Modify: `03-code-walkthrough/02b-scheduling-policies.md`
- Modify: `03-code-walkthrough/03-kv-cache-manager.md`
- Create: `03-code-walkthrough/08-input-processing-and-tokenization.md`
- Modify: `curriculum.toml`
- Modify: `content-review.toml`

**Interfaces:**
- Consumes: current OpenAI entrypoint, `LLM`/`AsyncLLM`, input-processing, EngineCore, scheduler, request queue, KV manager, block pool, and hash source.
- Produces: four refreshed chapters plus a complete input-processing chapter.

- [ ] **Step 1: Trace and correct the current request path**

Build a source-backed table with columns HTTP/API object, internal request object, process boundary, queue/state transition, and observable evidence. Verify cancellation and error paths in addition to success. Replace any V0-first call chain.

- [ ] **Step 2: Reconstruct `Scheduler.schedule()` from current contracts**

Explain current token budget, running/waiting/deferred paths, preemption, structured output/spec decode/KV connector constraints, priority policy, and `SchedulerOutput`. Use focused semantic anchors rather than linking the entire method. Include one hand-worked schedule step with explicit token counts and one interview explanation.

- [ ] **Step 3: Reconstruct KV lifecycle**

Verify allocation, cache hit accounting, block hashes, reference counts, free queues, eviction/reuse, preemption, and any KV connector/offload boundary. Include one block-table example and one failure diagnosis based on current metrics.

- [ ] **Step 4: Write the full input-processing chapter**

`08-input-processing-and-tokenization.md` must include:

1. lesson metadata and outcomes;
2. OpenAI request to internal request data flow;
3. validation, tokenization, chat template, prompt embeddings, LoRA/prompt adapter, and multimodal preprocessing boundaries;
4. CPU cost/backpressure and async safety;
5. current symbols and semantic links;
6. a no-GPU source trace and a remote-endpoint request experiment;
7. invalid-input, tokenizer mismatch, oversized prompt, multimodal limit, and cancellation failures;
8. production checklist, self-check, 30-second/3-minute interview answers, next step.

- [ ] **Step 5: Register and review the five chapters**

Add the new chapter to `curriculum.toml` immediately after `07-model-architectures.md` temporarily; Task 4 will append output processing after it. Add a reviewed ledger row with current SHA and no hardware claim. Validate/build.

- [ ] **Step 6: Commit entry/scheduler/KV content**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && test -s "$site_dir/03-code-walkthrough/08-input-processing-and-tokenization.html" && git add 03-code-walkthrough curriculum.toml content-review.toml && git commit -m "docs: refresh request scheduling and kv internals"'
```

---

### Task 4: Model Execution, Attention, Kernels, and Output Review

**Files:**
- Modify: `03-code-walkthrough/04-model-runner.md`
- Modify: `03-code-walkthrough/05-attention-backends.md`
- Modify: `03-code-walkthrough/06-cuda-kernels.md`
- Modify: `03-code-walkthrough/07-model-architectures.md`
- Create: `03-code-walkthrough/09-output-processing-and-streaming.md`
- Modify: `curriculum.toml`
- Modify: `content-review.toml`

**Interfaces:**
- Consumes: current GPU model runner/input batch, attention selection/backends, PagedAttention and auxiliary kernels, model registry, sampler/output processor/detokenizer/API streaming source.
- Produces: complete source journey from scheduled batch to streamed output.

- [ ] **Step 1: Refresh model-runner and input-batch data flow**

Verify persistent input batch structures, block tables, positions, sampling/structured-output metadata, compile/CUDA Graph dispatch, forward call, sampler boundary, and async output behavior. Separate prefill/decode conceptual labels from the scheduler's unified computed-token model.

- [ ] **Step 2: Refresh backend and kernel selection**

Document the current selection inputs, supported head/layout/data-type/model combinations, fallback behavior, and platform boundaries. Do not present a static ranking of FlashAttention/FlashInfer/Triton/MLA without workload and support constraints. Re-anchor PagedAttention, RoPE, RMSNorm, quantization, and sampler kernel examples to current code.

- [ ] **Step 3: Refresh architecture coverage**

Verify model registry/loading contracts and current distinctions among dense/GQA/MLA/MoE/Mamba-style models. Explain how model-specific layers meet common runner and cache contracts; remove stale file-count claims or generate them.

- [ ] **Step 4: Write the full output-processing chapter**

`09-output-processing-and-streaming.md` must cover sampler outputs, logprobs, stop conditions, detokenization, usage accounting, streaming chunks, client cancellation, network backpressure, and error propagation. Include one streaming timeline, one curl/Python client exercise, and separate TTFT, ITL, server scheduling, and network latency evidence.

- [ ] **Step 5: Register, validate, and commit**

Place chapter 09 after chapter 08 in inventory; update five review rows. Run contracts/build, then commit:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && test -s "$site_dir/03-code-walkthrough/09-output-processing-and-streaming.html" && git add 03-code-walkthrough curriculum.toml content-review.toml && git commit -m "docs: refresh model execution and output internals"'
```

---

### Task 5: Optimization Review

**Files:**
- Modify: `04-optimizations/01-quantization.md`
- Modify: `04-optimizations/02-speculative-decoding.md`
- Modify: `04-optimizations/03-cudagraph-and-compile.md`
- Modify: `04-optimizations/04-compilation-internals.md`
- Modify: `04-optimizations/05-roofline-and-arithmetic-intensity.md`
- Modify: `content-review.toml`

**Interfaces:**
- Consumes: current quantization registry/configs, speculative decoding, compilation, CUDA Graph, and performance-estimation source.
- Produces: five reviewed optimization chapters with decision and measurement discipline.

- [ ] **Step 1: Rebuild the optimization decision matrix**

For quantization, speculative decoding, compile/CUDA Graph, and roofline claims, record support constraints, accuracy/latency/throughput/memory tradeoffs, workload assumptions, current flags, current fallback path, metric evidence, and rollback condition. Remove unsupported absolute recommendations.

- [ ] **Step 2: Verify compilation internals and performance metrics**

Trace the current compilation manager/backend/pass boundaries, piecewise compilation, splitting ops, cache keys, shape specialization, graph capture/dispatch, and failure fallback. Verify every `estimated_flops/read_bytes/write_bytes` claim and qualify estimates versus hardware counters.

- [ ] **Step 3: Add one-variable experiments and interview expressions**

Each chapter gets a baseline/variant/result template. Quantization includes quality evaluation; speculative decoding includes acceptance rate; compile includes warmup and cache; roofline includes units and sanity checks.

- [ ] **Step 4: Update five reviews, validate, and commit**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && git add 04-optimizations content-review.toml && git commit -m "docs: refresh vLLM optimization internals"'
```

---

### Task 6: Distributed and Large-Scale Review

**Files:**
- Modify: `05-distributed/01-tp-pp-ep.md`
- Modify: `05-distributed/02-disaggregated.md`
- Modify: `05-distributed/03-expert-parallel-deep-dive.md`
- Modify: `05-distributed/04-context-parallel.md`
- Modify: `05-distributed/05-large-scale-cluster-inference.md`
- Modify: `content-review.toml`

**Interfaces:**
- Consumes: current parallel config/state, executors, collective communication, EPLB, CP, KV transfer/offload, and disaggregated serving source.
- Produces: five reviewed distributed chapters with explicit topology, collective, and failure contracts.

- [ ] **Step 1: Verify all parallel dimensions and world-size equations**

For TP, PP, DP, EP, DCP/PCP/CP, specify what is partitioned, process groups, collective type, divisibility constraints, memory effect, latency effect, and incompatible/conditional combinations. Derive equations with units and validate current config checks.

- [ ] **Step 2: Verify MoE, EPLB, and communication behavior**

Trace logical-to-physical expert mapping, token dispatch/combine, AllToAll backend selection, load tracking/rebalancing, DP padding and microbatch overlap from current code. Remove claims of online elasticity unless the current source implements the full behavior.

- [ ] **Step 3: Verify disaggregation and cluster failure guidance**

Trace KV connector interfaces, control/data planes, supported transports, ownership, backpressure, timeout, partial failure, and compatibility. Recheck every large-scale recommendation against current implementation or label it as architecture guidance rather than built-in behavior.

- [ ] **Step 4: Update five reviews, validate, and commit**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && git add 05-distributed content-review.toml && git commit -m "docs: refresh distributed inference internals"'
```

---

### Task 7: Existing Hands-On Review and API Service Chapter

**Files:**
- Modify: `07-hands-on/01-setup.md`
- Modify: `07-hands-on/02-trace-a-request.md`
- Modify: `07-hands-on/03-mini-experiments.md`
- Modify: `07-hands-on/04-profiling-and-debugging.md`
- Create: `07-hands-on/05-serve-openai-api.md`
- Modify: `curriculum.toml`
- Modify: `content-review.toml`

**Interfaces:**
- Consumes: pinned official installation/platform docs, examples, server CLI, metrics endpoint, benchmark clients, profiling hooks.
- Produces: current setup/trace/profiling material and a beginner-complete service lab.

- [ ] **Step 1: Refresh setup and environment matrix**

Separate Linux NVIDIA GPU, supported CPU platforms, source build, precompiled package, container, and remote endpoint routes. State that macOS is not assumed to run vLLM locally. Every route includes version checks, success evidence, and cleanup.

- [ ] **Step 2: Refresh trace, experiment, and profiling commands**

Verify module paths and current commands for request tracing, logs, debugger/py-spy, torch.profiler, NVTX, memory snapshots, and metrics. Keep GPU-only tools labeled. Each mini experiment changes one variable and records expected metric direction, not fabricated values.

- [ ] **Step 3: Write `05-serve-openai-api.md` as a complete lab**

Required sections:

1. choose local supported CPU, remote endpoint, or NVIDIA GPU route;
2. define `MODEL_ID`, `VLLM_BASE_URL`, `VLLM_API_KEY`, and output directory;
3. start server using the pinned CLI and verify health/models;
4. send chat, completion if supported by the model, streaming, and concurrent requests;
5. inspect `/metrics` and identify running/waiting/KV/TTFT/token counters or histograms by current names;
6. deliberately trigger invalid model/input/auth/limit cases appropriate to the route;
7. stop/drain and clean artifacts;
8. produce a one-page evidence report;
9. self-check and interview explanation.

Never place a real key in the chapter. Use environment variables and redact headers from logs.

- [ ] **Step 4: Register/review five chapters, build, and commit**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && test -s "$site_dir/07-hands-on/05-serve-openai-api.html" && git add 07-hands-on curriculum.toml content-review.toml && git commit -m "docs: add current vLLM serving lab"'
```

---

### Task 8: Benchmark, Tuning, and Production Capstone Chapters

**Files:**
- Create: `07-hands-on/06-benchmark-methodology.md`
- Create: `07-hands-on/07-tuning-playbook.md`
- Create: `07-hands-on/08-production-capstone.md`
- Create: `07-hands-on/templates/experiment-report.md`
- Create: `07-hands-on/templates/capacity-plan.md`
- Create: `07-hands-on/templates/incident-review.md`
- Modify: `curriculum.toml`
- Modify: `content-review.toml`

**Interfaces:**
- Consumes: current benchmark CLI/options, metrics, optimization/distributed chapters, production chapters.
- Produces: reproducible experimental method, evidence-based tuning flow, and portfolio/interview capstone.

- [ ] **Step 1: Write benchmark methodology with runnable command families**

Cover open-loop versus closed-loop, request-rate versus concurrency, warmup, request distributions, tokenizer/model consistency, input/output lengths, TTFT/TPOT/ITL/E2E/throughput/goodput/error percentiles, raw artifact retention, and comparison validity. Use current benchmark command names from the pinned checkout. Include a dry no-GPU analysis route using supplied example JSON in the chapter and a real endpoint route.

- [ ] **Step 2: Write the tuning playbook**

Organize by symptom: high TTFT, high TPOT, low throughput, KV pressure/preemption, compile startup, CPU/tokenizer bottleneck, communication bottleneck, cache miss, tail latency. Every branch includes evidence, likely causes, one-variable experiments, expected metric direction, confounders, and rollback.

- [ ] **Step 3: Write the capstone and three complete templates**

The capstone requires the ten deliverables in the approved design: requirements, architecture, deployment, baseline plus two experiments, capacity, SLO dashboard/alerts, two incidents, upgrade/rollback, retrospective, and five-minute interview narrative. Templates must contain concrete fields, formulas, unit columns, evidence links, and acceptance rubric; they must not contain `TBD` or `TODO`.

- [ ] **Step 4: Register/review/build and commit**

After adding these chapters, inventory count must be 56. Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && python3 -c "from pathlib import Path; from tools.source_sync.inventory import load_curriculum; assert len(load_curriculum(Path(\"curriculum.toml\"))) == 56" && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && test -s "$site_dir/07-hands-on/08-production-capstone.html" && git add 07-hands-on curriculum.toml content-review.toml && git commit -m "docs: add benchmark tuning and capstone labs"'
```

---

### Task 9: Production Deployment Review and Lifecycle Chapters

**Files:**
- Modify: `08-production-deployment/01-deployment-architectures.md`
- Modify: `08-production-deployment/02-smart-routing-and-load-balancing.md`
- Modify: `08-production-deployment/03-gateway-and-service-mesh.md`
- Modify: `08-production-deployment/04-autoscaling-and-capacity.md`
- Modify: `08-production-deployment/05-slo-and-observability.md`
- Modify: `08-production-deployment/06-reliability-and-failure-modes.md`
- Modify: `08-production-deployment/07-incident-playbook.md`
- Modify: `08-production-deployment/08-monitoring-cookbook.md`
- Modify: `08-production-deployment/09-vllm-doctor-skill.md`
- Modify: `08-production-deployment/10-gpu-utilization-and-tail-latency.md`
- Create: `08-production-deployment/11-security-and-multi-tenancy.md`
- Create: `08-production-deployment/12-upgrades-rollbacks-and-compatibility.md`
- Modify: `curriculum.toml`
- Modify: `content-review.toml`

**Interfaces:**
- Consumes: current API server/metrics/tracing source, official production docs, existing routing/autoscaling/SLO/reliability/runbook chapters.
- Produces: 12 reviewed production chapters covering architecture through secure lifecycle operation.

- [ ] **Step 1: Review architecture, routing, gateway, and autoscaling**

Verify which capabilities live in vLLM versus Production Stack/llm-d/AIBrix/gateway/service mesh. Recheck prefix-aware routing signals, readiness/drain behavior, metrics used for scaling, cold-start phases, queue/capacity formulas, and failure behavior. External project claims require primary project docs and access dates.

- [ ] **Step 2: Review SLO, reliability, incidents, monitoring, doctor skill, and performance diagnosis**

Verify every metric against current registration and exposed Prometheus naming. Ensure PromQL uses correct counter suffixes and histogram aggregation. Align incident detection/mitigation/recovery with current failure paths. Keep the doctor skill's automated permissions and remediation levels fail-safe. Recheck MBU/MFU estimates and label estimated versus hardware-measured quantities.

- [ ] **Step 3: Write security and multi-tenancy**

Cover threat model, auth placement, TLS boundary, request/output limits, rate/quota isolation, model/LoRA trust, tokenizer/chat-template input risk, log/trace redaction, network policy, filesystem/cache concerns, dependency/image provenance, and incident audit. Explicitly assign each control to vLLM, gateway, orchestrator, secret manager, or platform.

- [ ] **Step 4: Write upgrades, rollbacks, and compatibility**

Cover compatibility matrix fields for vLLM SHA, model revision, tokenizer, quantization format, GPU architecture, driver/CUDA/PyTorch, attention backend, API behavior, config, and client. Provide golden requests, quality/performance regression gates, shadow/canary, drain, rollback, cache implications, and change record. Link the source-sync workflow as the documentation-side counterpart.

- [ ] **Step 5: Register/review/build and commit**

Inventory count must be 58. Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && python3 -c "from pathlib import Path; from tools.source_sync.inventory import load_curriculum; assert len(load_curriculum(Path(\"curriculum.toml\"))) == 58" && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && test -s "$site_dir/08-production-deployment/12-upgrades-rollbacks-and-compatibility.html" && git add 08-production-deployment curriculum.toml content-review.toml && git commit -m "docs: refresh production operations and lifecycle"'
```

---

### Task 10: Advanced Features Review

**Files:**
- Modify: `09-advanced-features/01-sampling-and-logits.md`
- Modify: `09-advanced-features/02-structured-output.md`
- Modify: `09-advanced-features/03-multimodal.md`
- Modify: `09-advanced-features/04-lora-serving.md`
- Modify: `09-advanced-features/05-embedding-and-pooling.md`
- Modify: `content-review.toml`

**Interfaces:**
- Consumes: current sampling, structured output, multimodal, LoRA, embedding/pooling source and API contracts.
- Produces: five reviewed feature chapters connected to the core request and production paths.

- [ ] **Step 1: Verify sampling and structured output**

Check current sampling order, logits processors, penalties, logprobs, seed/determinism limits, structured-output backends, grammar compilation/cache, fallback/error behavior, and scheduling impact. Remove obsolete backend names or flags.

- [ ] **Step 2: Verify multimodal and LoRA**

Trace preprocessing, encoder cache, multimodal limits/placeholders, batching constraints, adapter loading/activation/eviction, Punica kernels, isolation, and failure paths. Include production memory/concurrency considerations and security boundary links.

- [ ] **Step 3: Verify embedding and pooling**

Check supported runner/task selection, pooling contracts, normalization, token-level versus sequence-level outputs, OpenAI-compatible endpoints where applicable, batching, and evaluation pitfalls.

- [ ] **Step 4: Update five reviews, validate, and commit**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && git add 09-advanced-features content-review.toml && git commit -m "docs: refresh advanced serving features"'
```

---

### Task 11: Interview Review, Calculation Drills, and Mock Rubric

**Files:**
- Modify: `06-interview/01-common-questions.md`
- Modify: `06-interview/02-system-design.md`
- Create: `06-interview/03-capacity-and-troubleshooting-drills.md`
- Create: `06-interview/04-mock-interview-and-rubric.md`
- Modify: `curriculum.toml`
- Modify: `content-review.toml`

**Interfaces:**
- Consumes: every reviewed technical chapter and capstone evidence model.
- Produces: four reviewed interview chapters, 60 total chapters, and reusable scoring rubrics.

- [ ] **Step 1: Convert common questions to three-layer answers**

For each existing question, provide 30-second conclusion, 3-minute mechanism/source answer, production tradeoff, verification signal, rollback/failure limit, and at least two follow-ups. Correct any answer changed by the pinned source review.

- [ ] **Step 2: Refresh system design around requirements first**

Force candidates to ask model/workload/SLO/hardware/quality/security/change-window questions before architecture. Include capacity math, single-node and distributed alternatives, routing/cache tradeoffs, failure domains, observability, deployment, cost, and explicit rejection criteria.

- [ ] **Step 3: Write calculation and troubleshooting drills**

Include solved problems for weight memory, KV bytes/token, request KV, batch capacity, TP/PP/DP/EP world sizes, throughput/goodput, replica count, and failure reserve. Every solution shows formula, units, assumptions, intermediate values, and sanity check. Include evidence-first scenarios for TTFT, TPOT, OOM, preemption, NCCL, cache hit collapse, tokenizer CPU, and retry storms.

- [ ] **Step 4: Write mock interview and rubric**

Provide five rounds: concept, source trace, calculation, system design, incident. Each round has interviewer prompt, expected evidence, follow-ups, strong/weak signals, and a 1-5 rubric for accuracy, evidence, tradeoffs, validation/rollback, and communication. Include a final project-experience narrative template without fictitious results.

- [ ] **Step 5: Register/review and verify 60 chapters**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile contracts && python3 -c "from pathlib import Path; from tools.source_sync.inventory import load_curriculum; chapters = load_curriculum(Path(\"curriculum.toml\")); assert len(chapters) == 60; assert len({c.path for c in chapters}) == 60" && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && test -s "$site_dir/06-interview/04-mock-interview-and-rubric.html"'
```

Expected: 60 unique chapters and a successful HTML build.

- [ ] **Step 6: Commit interview content**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add 06-interview curriculum.toml content-review.toml && git commit -m "docs: add inference interview drills and rubric"'
```

---

### Task 12: Full Semantic Gate, Final Upstream Refresh, and Release Evidence

**Files:**
- Modify: `.github/workflows/validate.yml`
- Modify: `.github/workflows/pages.yml`
- Modify: `.github/workflows/sync-upstream.yml`
- Create: `.github/workflows/gpu-validation.yml`
- Create: `scripts/gpu-validation.sh`
- Modify: `tests/source_sync/test_workflows.py`
- Modify: `README.md`
- Modify: `DEPLOY.md`
- Modify: `source.lock.json`
- Modify: `content-review.toml`
- Modify as identified by final impact: affected chapter Markdown and semantic links only
- Create: `artifacts/content-review/final-verification.md`

**Interfaces:**
- Consumes: 60 reviewed chapters, all source contracts, official current main at final verification.
- Produces: full fail-closed CI/deployment gate and evidence-backed final baseline.

- [ ] **Step 1: Prove every review is complete before changing CI**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m tools.source_sync validate --profile full'
```

Expected: `OK: full source and content review is valid`. If any pending row, stale SHA, false check, unmanaged link, or missing chapter remains, fix that chapter under the Shared Chapter Review Protocol; do not weaken validation.

- [ ] **Step 2: Switch every workflow from contracts to full validation**

Replace `validate --profile contracts` with `validate --profile full` in pull-request, upstream-sync, and Pages workflows. Keep `--require-committed` in PR/Pages checkout jobs. Update README and DEPLOY so the public definition of “validated” includes semantic review, not only link resolution.

- [ ] **Step 3: Add an evidence-producing manual GPU workflow**

Create `scripts/gpu-validation.sh` with positional arguments `MODEL_ID` and `TENSOR_PARALLEL_SIZE`. Reject a model ID outside `^[A-Za-z0-9._/-]+$` and a parallel size outside `1|2|4|8`. The script must:

1. create an output directory named by UTC time and vLLM SHA;
2. record `git rev-parse HEAD`, `nvidia-smi -q`, `nvidia-smi topo -m`, Python/package/CUDA/PyTorch versions, model ID, and exact launch command;
3. start `vllm serve "$MODEL_ID" --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" --api-key "$VLLM_API_KEY"` with stdout/stderr captured and a cleanup trap that terminates the server;
4. poll `/health` for at most 300 seconds and fail with server logs on timeout;
5. request `/v1/models`, one deterministic short `/v1/chat/completions` request, one streaming request, and `/metrics`, saving headers separately from redacted bodies;
6. store start/end GPU snapshots and a `result.json` containing `status`, `source_sha`, `model_id`, `tensor_parallel_size`, `started_at`, and `finished_at`;
7. never print or archive the API key.

Create `.github/workflows/gpu-validation.yml` with `workflow_dispatch` inputs `model_id` and choice `tensor_parallel_size` (`1`, `2`, `4`, `8`), `runs-on: [self-hosted, linux, x64, nvidia-gpu]`, checkout of the tutorial and pinned submodule, `VLLM_API_KEY` from an environment secret, execution of the script, and `actions/upload-artifact@v4` under `if: always()`. It does not install vLLM or GPU drivers; the runner contract requires the pinned submodule version to be installed and records the actual installed version.

Extend `tests/source_sync/test_workflows.py` to parse `gpu-validation.yml`, assert it is manual-only, uses the four-label self-hosted runner, reads the API key from `secrets`, calls only `scripts/gpu-validation.sh`, and uploads artifacts under `if: always()`.

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && bash -n scripts/gpu-validation.sh && python3 -m unittest tests.source_sync.test_workflows -v'
```

Expected: shell syntax and workflow contract tests pass without requiring a GPU. Do not execute the hardware workflow unless a matching self-hosted runner and model access are available.

- [ ] **Step 4: Query official main and refresh if it advanced**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && official_sha="$(git ls-remote https://github.com/vllm-project/vllm.git refs/heads/main | awk "{print \$1}")" && test "${#official_sha}" -eq 40 && baseline_sha="$(python3 -c "import json; print(json.load(open(\"source.lock.json\"))[\"commit\"])")" && if test "$official_sha" != "$baseline_sha"; then git -C vllm fetch origin main && git -C vllm checkout --detach "$official_sha" && python3 -m tools.source_sync impact --baseline "$baseline_sha" --candidate "$official_sha" --output artifacts/source-sync/latest-impact.md; fi'
```

Expected: either no change, or a new impact report and detached candidate checkout. If advanced, repeat `refresh`, review every affected/uncovered file, and update those chapters. For every non-affected chapter, first verify that neither its direct contracts nor any `source_areas` glob matched the diff, then carry its review forward with a note naming the old and new SHAs. Only after that evidence may every review row use the new candidate SHA. Do not mass-change review SHAs without reading the impact diff.

- [ ] **Step 5: Update independent vLLM main to the final verified SHA without touching the bugfix branch**

If Step 4 advanced, use the atomic fast-forward sequence from source-sync Task 6 Step 2 with the new official SHA. Verify the active branch and commits `004b8601c`, `d1cd0162d`, and `090fd61d8` again.

- [ ] **Step 6: Run complete automated verification**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest discover -s tests/source_sync -v && python3 -m tools.source_sync validate --profile full && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && test "$(find "$site_dir" -name "*.html" | wc -l | tr -d " ")" -ge 61 && test -s "$site_dir/search-index.json" && git diff --check'
```

Expected: all tests and full validation pass; site contains README plus 60 chapter pages; search index exists; diff check is clean.

- [ ] **Step 7: Run optional publication builders in the equipped environment**

First inspect tools:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && command -v pandoc || true && command -v xelatex || true'
```

If both exist, run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && output_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$output_dir" python3 build_pdf_epub.py && test -s "$output_dir/vllm-learning.pdf" && test -s "$output_dir/vllm-learning.epub"'
```

If either is absent, record exactly which executable is missing in final verification; do not install a multi-gigabyte TeX distribution without separate user authorization and do not claim PDF/EPUB passed.

- [ ] **Step 8: Record final verification evidence**

Create `artifacts/content-review/final-verification.md` with:

- official repository/branch/full SHA and upstream commit time;
- validation UTC time;
- 60 chapter count and source line count;
- exact test/build commands and exit results;
- source contract count, affected chapter count, and zero unresolved/unmanaged counts;
- all review rows current and complete;
- HTML output page/search evidence;
- PDF/EPUB result or explicit missing tools;
- GPU runs indexed by ID, or the exact statement `No current-SHA GPU hardware run was performed; performance examples are labeled expected or illustrative.`;
- preserved bugfix branch and three reachable commit SHAs;
- known external constraints that do not contradict static completion.

- [ ] **Step 9: Commit the full gate and final evidence**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && git add .github/workflows/validate.yml .github/workflows/pages.yml .github/workflows/sync-upstream.yml .github/workflows/gpu-validation.yml scripts/gpu-validation.sh tests/source_sync/test_workflows.py README.md DEPLOY.md source.lock.json curriculum.toml content-review.toml artifacts/source-sync/latest-impact.md artifacts/content-review/final-verification.md vllm 01-overview 02-core-concepts 03-code-walkthrough 04-optimizations 05-distributed 06-interview 07-hands-on 08-production-deployment 09-advanced-features && git commit -m "docs: complete latest-main inference curriculum"'
```

- [ ] **Step 10: Re-run verification from committed state**

Run:

```bash
ssh rlocal 'cd /Users/zjw/Documents/LLM/inference-engine/vllm/vllm-learning && python3 -m unittest discover -s tests/source_sync -v && python3 -m tools.source_sync validate --profile full --require-committed && site_dir="$(mktemp -d)" && VLLM_LEARNING_DST="$site_dir" python3 build_html.py && git status --short --branch && official_sha="$(git ls-remote https://github.com/vllm-project/vllm.git refs/heads/main | awk "{print \$1}")" && test "$(git -C vllm rev-parse HEAD)" = "$official_sha" && test "$(git -C ../vllm rev-parse main)" = "$official_sha"'
```

Expected: all checks pass from committed content, worktree is clean, and official/local/submodule main SHAs match at the recorded verification moment.
