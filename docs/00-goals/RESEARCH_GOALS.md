---
title: "AM-LLM — research objectives and acceptance framework (A–D)"
version: "1.0"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "GOALS"
source_of_truth: true
supersedes: ["docs/archive/Phase3 backlog §0 goal sentence"]
superseded_by: []
---

**English** | [Magyar](RESEARCH_GOALS.hu.md)

# AM-LLM — research objectives and acceptance framework (A–D)


**Version:** 1.0 — 2026-10-23  
**Status:** proposed current canonical research objectives; does not retrospectively change preregistrations.  
**Scope:** long-term AM-LLM research programme. New hypotheses and experiments must refer to this framework.  
**Original historical specification:** the original first experiment assignment (`goal.md`, internal) §2.  
**Evidence:** `../../results/EXPERIMENT_REPORT.md` and `../../results/overnight_v2_20261022/NIGHT_RESEARCH_REPORT_HU.md`.

## 1. Preserve the original question

The original research question was whether a hierarchical module router, an expert router within each module, and dynamic processing depth can attain Dense Transformer / fixed-depth MoE predictive quality **with less active computation**. This remains objective **A**, not a claim already demonstrated.

The original first-series targets were approximately ≤5% quality regression, ≥15% executed-compute savings **or** better quality at matched compute, ≤10% measured latency regression, meaningful router learning, variable-depth behaviour, and comparable evidence over three seeds. These are **historical first-series gates**, not retroactive criteria for all future phases. The original `private/goal.md` is an archival experiment assignment, not a living global-goal file.

## 2. The current A–B–C–D framework

### A — Scientific core: quality per actual unit of compute

**Question:** can learned module/expert routing, recurrent module use, and content-dependent computation achieve comparable or better **predictive/task quality with fewer actually executed operations** than properly matched Dense/MoE baselines?

- **Evaluate:** validation NLL/perplexity and independent tasks versus **executed** FLOPs/token, actual attention/FFN rows, kernel operations and model/batch/context scale.
- **Evidence required:** preregistered, reproducible quality–compute Pareto improvement; neither *ideal* FLOP estimates nor reduced average hops alone meet A.
- **Examples:** dynamic depth, learned selective expert/attention calls, Top-K routing, modular recurrence, structured block skipping.
- **Counterexample:** a fixed eight-hop variant nearly matches MoE loss using more computation; this is potentially useful architecture evidence, **not a demonstration of A**.

### B — Systems memory: small real GPU working set

**Question:** can infrequently used module/expert weights reside in CPU RAM or storage while only the active weights, KV cache and runtime scratch occupy the GPU, with tolerable cache-miss/transfer overhead?

- **Long-term deployment target:** commodity GPU with **16 GB VRAM**, not a success requirement for 17M toy experiments.
- **Evaluate:** measured inference peak VRAM with resident weights, KV cache and scratch separately; cache hit/miss, host↔device bytes/token and p95 miss latency.
- **Evidence required:** real working selective loading/offload and lower measured peak VRAM at comparable quality. Simulated cache traces alone are insufficient.
- **Important:** smaller **total** model weight count is *not mandatory*. Full parameter count and disk/RAM footprint remain metrics.

### C — Practical throughput and latency

**Question:** can dynamic execution yield **at least MoE-near, eventually better real generation speed** after routing, launch, synchronization, caching and paging overhead?

- **Evaluate:** autoregressive decode tokens/sec (batch 1 and concurrent), TTFT, prefill, p50/p95 latency under matched GPU, quantization, context, backend and task-quality conditions.
- **Long-term engineering target:** **≥60 decoded tokens/sec** on a useful larger coding model in a declared commodity-hardware/32K-context setting; not extrapolated from toy-model speed.
- **Evidence required:** measured generation, not training throughput or full-sequence-forward latency alone.

### D — Ultimate vision: substantially larger-model capabilities on commodity hardware

**Vision:** a high-total-capacity AM-LLM whose task-conditional active path achieves coding, implementation and debugging capability near substantially larger Dense/MoE models **while jointly meeting A–C**.

- **Illustrative research scale hypotheses, not commitments or existing results:** a ~27B AM-LLM approaching much larger-model coding ability; eventually a ~200B-total-capacity model challenging a ~377B-scale MoE if training and evaluation justify it.
- **Full model weight count can exceed the GPU's resident set** when conditional execution and a memory hierarchy make that practical. Fewer total weights are a welcome bonus, not an architectural requirement.
- **Evaluate:** leakage-resistant TypeScript/React/NestJS coding, bug fixing, implementation and pass@1 benchmarks, later at matching prompts and test protocols against much larger external baselines.
- **Status:** distant, **unproven** goal. Small byte and 17M BPE models do not establish D.

**Relationship:** A is the original scientific hypothesis, B and C are complementary practical requirements, and D is the ultimate capability goal. Improvements in one dimension do not prove the others.

### 2.1 Architecture family versus the original hypothesis (decision, 2026-10-23)

The fixed-depth variant with per-hop attention belongs to the AM-LLM architecture family. The **architecture family** and the **original full hypothesis** (dynamic compute) are, however, not the same thing.

| Component | Definition |
|---|---|
| Reusable module bank | core of the architecture |
| Hierarchical module and expert routing | core of the architecture |
| Multiple processing hops with learned routes | core of the architecture |
| Dynamic depth / halting | part of the original main hypothesis, not mandatory in every control variant |
| Per-hop attention | permitted architectural variant |
| Fixed 4/6/8 hops | permitted control and research variant |

Naming: **AM-LLM-FixedDepth** — hierarchical routing and a reusable module bank with fixed depth; **AM-LLM-AdaptiveDepth** — the same with learned, input-dependent compute depth.

- A FixedDepth variant may approach or exceed MoE quality, but **does not by itself prove** the original dynamic-compute hypothesis (A): that needs AdaptiveDepth.
- If both hierarchical routing and the reusable module bank are dropped, the result is **a different architecture**, not an AM-LLM variant.

## 3. Non-goals versus tools

- **Not mandatory:** reducing total unique parameters in every revision; targeting exactly 27B/400 modules/100 experts; reducing hop count for every token.
- **Not sufficient:** lower ideal FLOPs, faster training, better loss requiring more compute, smaller training VRAM without inference measurements, or a faster AM-specific backend without similarly optimized MoE controls.
- **Potential techniques:** hierarchical routing, adaptive Top-K and halting, partially shared/per-hop attention, recurrent modules, selective attention, static buckets, fused/Triton kernels, KV cache and RAM–VRAM paging.
- Future design studies **may revise v0.1 topology**, but the original design must remain reproducible as a historical control.

## 4. Evidence snapshot and limitations (2026-10-23)

**E00–E09:** learned routing beat random routing by 19.9% validation loss over three seeds; per-hop attention improved quality. The original E07 go/no-go gate **failed** (ADR-0011). Later variants cannot retroactively change that finding.

**F5 (~17M BPE):** the preregistered F5 Q gate **failed**. Soft and hard halting diverged substantially; pass@1 was eventually evaluated but was zero/nearly zero on these small models and does not imply meaningful coding proficiency.

**Overnight 2026-10-22/23:** 36 completed trainings (12 configurations × three seeds). The fixed eight-hop AM-LLM reached mean validation NLL **2.1212** versus MoE **2.1166** (ratio 1.002) but required **6.9% more total parameters and more computation**. No useful lower-hop hard-halting quality trade-off was found. Batch-1 decode: AM 632, MoE 884, Dense 2,075 tokens/sec on these *small* benchmarks. Source: `../../results/overnight_v2_20261022/NIGHT_RESEARCH_REPORT_HU.md`.

**F6:** currently **NO-GO** for a new runtime according to the overnight report. Historic F6 plans do not imply implementation or success.

**Overall:** A is **currently not supported at the 17M scale with the architectural variants tested**: the two best variants reached MoE-close quality but needed 24–31% more FLOP, which counts against the efficiency hypothesis in that range. This does not show that A is unreachable with other routing, attention or adaptive-depth solutions. B and C have not been demonstrated at target scale; D remains a research vision. Router learning is a meaningful sub-result but not an A–D success claim.

## 5. Experimental governance

- Every new task identifies the primary **A/B/C/D** objective, optional secondary objectives, explicit hypothesis and matching controls. A local component-level success is not an overall project success.
- New experiments have **new preregistrations**; original first-series thresholds, the E07 decision and F5 Q result are never retrospectively altered.
- Keep train/validation/test splits distinct, paired seeds, matched parameter/compute/quality controls and actual inference measurements separate from FLOP estimates.
- The 2026-10-22/23 planner had access to test losses; subsequent experiment selection must use **validation only**, and final confirmations should use a fresh independent holdout where feasible.
- Large-scale trials (27B and beyond) remain contingent on new quality, memory and decode gates.
- Report negative, incomplete and inconclusive results and retain code, data and hardware provenance.

## 6. Document precedence

1. [`RESEARCH_GOALS.md`](RESEARCH_GOALS.md) — current canonical A–D objectives; paired with [`RESEARCH_GOALS.hu.md`](RESEARCH_GOALS.hu.md).
2. the original first experiment assignment (`goal.md`, internal) — archived first-series task, preserved as written.
3. `RESEARCH_OVERVIEW.md` — concise hypotheses and status.
4. [`../02-methodology/METHODOLOGY.md`](../02-methodology/METHODOLOGY.md) — original method and extended A–D measurements.
5. [`../01-architecture/ARCHITECTURE.md`](../01-architecture/ARCHITECTURE.md) — historical v0.2.1 implementation, not automatic specification for later experimental branches.
6. `../archive/PHASE2_PLAN.md` — original Phase 2 plan/preregistration and gates, historical.
7. `../archive/Phase3/AM_LLM_ADAPTIV_EJSZAKAI_KUTATASI_BACKLOG_V2_HU.md` — 2026-10-22 adaptive backlog; this document **supersedes the old V2 introductory goal statement**.
8. Experiment reports and ADRs remain immutable, date-specific evidence; current goals must not rewrite historical runs or decisions.
