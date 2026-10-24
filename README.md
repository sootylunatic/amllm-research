**English** | [Magyar](README.hu.md)

# AM-LLM — Adaptive Modular Language Model

![status](https://img.shields.io/badge/status-research%20prototype%20%E2%80%94%20main%20hypothesis%20not%20yet%20supported-orange)
[![tests](https://github.com/sootylunatic/amllm-research/actions/workflows/tests.yml/badge.svg)](https://github.com/sootylunatic/amllm-research/actions) ![license](https://img.shields.io/badge/license-MIT-blue)

**An open research programme, not a product.** AM-LLM asks whether a language model can learn *which reusable processing units to run, in which order and how many times* for each token, and so approach the quality of a Dense Transformer or a Mixture-of-Experts (MoE) with less **actually executed** compute. This repository holds the prototype code, the tests, the decisions taken along the way and an honest account of what has and has not been shown. Negative results are part of the record.

## The main goal

Run a **large, genuinely useful language model — first of all for coding — on a consumer GPU**: much of the quality of a far bigger Dense/MoE model with a fraction of the *executed* compute, a small *inference* memory footprint (long-term target: 16 GB VRAM) and fast local generation (long-term target: ≥ 60 decoded tokens/s). Illustrative long-term scales such as ~27B stored parameters in hundreds of modules are **aspirations, not results**. Everything below is the step-by-step test of whether this is even possible, starting small and refusing to scale up before the small model shows a gain.

## The hypothesis

> **H (central):** a model with a **shared bank of reusable modules**, **two-level routing** (module, then expert) and **learned per-token depth** can reach the quality of a Dense Transformer or a fixed-depth MoE with **less actually executed compute**.
>
> **H₀ (null):** the quality loss, routing overhead and memory traffic cancel the benefit. The programme accepts this outcome if the data say so.

It is stated so that it can fail. The first round was judged against thresholds fixed in advance: validation loss at most ~5% worse than the control, at least ~15% less executed compute (or better quality at equal compute), at most ~10% latency regression, a router that demonstrably learns, working dynamic depth, and the same conclusion on three seeds.

| Part of the hypothesis | Status |
|---|---|
| **H1** the router learns stable, input-dependent routes | **Supported** (small models) |
| **H2** same quality with less executed compute | **Not supported at ~17M** |
| **H3** easy tokens need fewer steps (adaptive depth) | Partly: depth varies with content, but the current halting mechanism loses quality |
| **H4** specialisation helps coding quality | Cannot be decided at this scale |
| **H5** predictable module loading lowers memory | Not implemented, not tested |

## The idea

```text
Dense / MoE                          AM-LLM
token ─► layer 1 ─► layer 2 ─► …     token ─► [ module router ] ─► module ─► [ expert router ] ─► experts
         (each layer has its own            └────────── one shared bank, used again in every hop ◄────────┘
          weights, fixed depth)             ─► hop 1 ─► hop 2 ─► … ─► (optional) per-token halting: another hop?
```

- **A shared module bank:** M modules, each a few small expert networks, reused in every processing step ("hop").
- **Two-level routing:** per token and hop a router picks modules, and inside them an expert router picks experts.
- **Adaptive depth (optional):** a halting head decides per token whether another hop is needed.

Two members of the family are distinguished throughout: **FixedDepth** (the bank and the routing with a fixed number of hops) and **AdaptiveDepth** (the same with learned, input-dependent depth). FixedDepth is a legitimate control and research subject, but it does **not** by itself test the dynamic-compute part of the hypothesis. [Architecture guide](docs/ARCHITECTURE_GUIDE.md) · [v0.2.1 implementation](docs/01-architecture/ARCHITECTURE.md).

## The four goals A–B–C–D

The main goal is split into four claims that are proven separately, because a good result on one is easily mistaken for another.

| | The question | What counts as proof | What does **not** count | Status |
|---|---|---|---|---|
| **A** Quality per executed compute | Same quality with less *executed* compute than a matched Dense/MoE? (the original scientific hypothesis, H2) | A pre-registered, reproduced quality–compute advantage over matched controls, on independent data | Lower *ideal* FLOP that is not executed; fewer average hops alone; better loss with more compute | **Not supported at ~17M**: the best variants reach MoE loss only with more compute or more parameters; at 0.85× routed FLOP the AM-LLM loses to an equal-parameter MoE (V3 W2) |
| **B** Inference memory | A smaller real GPU working set, with rarely used weights kept in RAM or on disk? | Measured inference VRAM (weights, KV-cache, buffers) and host→GPU traffic with a working offload | Training memory; a cache *simulation*; fewer total parameters as such | **Not shown** |
| **C** Generation speed | Real autoregressive decode at least MoE-like, later better? | Measured prefill/decode tokens/s, TTFT, latency percentiles, same GPU and backend as the MoE control | Training tokens/s; full-sequence forward time | **Not shown**; first measurement: AM-LLM 632 vs MoE 884 tok/s at batch 1 |
| **D** Capability | Useful coding and problem-solving ability of a much larger model on consumer hardware, with A, B and C together? | Independent, leakage-free coding benchmarks against much larger models | A low loss at 17M extrapolated to 27B or 200B | **Long-term ambition, not tested** |

How they relate (as in the canonical goal document): **A is the original scientific *why*; B and C are two independent engineering proofs of feasibility; D is the final *what for* and needs A, B and C together.** No goal substitutes for another's measurement. [How results are judged](docs/00-goals/RESEARCH_GOALS.md) · [Evidence status](docs/RESEARCH_STATUS.md).

## The research programme at a glance

| Phase | Scale | What was asked | What came out |
|---|---|---|---|
| **v0.1 prototype** (2026-10-20) | byte-level, tiny | Does the mechanism run at all? | Yes: routing, halting, tests (4). |
| **Phase 1 — E00–E09** | byte-level, 0.3–2.5M parameters | Does it match Dense/MoE with less compute? Controlled comparison, 3 seeds, audited. | The router **learns** (19.9% lower loss than random routing). Per-hop attention is the biggest quality factor. At equal active compute AM-LLM was 6.8% worse than Dense and 14.5% worse than MoE, and slower on the GPU. The preregistered go/no-go gate **failed**. |
| **Phase 2 — F1–F4** | byte-level | Can the paper saving become a real one? | Top-1 routing with a gradient works; bucketed skipping and kernel work (Triton, CUDA Graph, KV-cache decode) gave partial results and no general decode win. |
| **F5** (exploratory) | ~17M, 8,192 BPE | Do the effects survive a larger model and real code? | The quality gate **failed**; learned halting loses a lot of quality when truncated at inference. |
| **Overnight round** (2026-10-22/23) | ~17M | 12 configurations × 3 seeds | The best FixedDepth variants approach MoE loss with **24–31% more** estimated FLOP; no variant matches MoE with less compute. |
| **Round V3 — W0, W1** (closed 2026-10-23) | ~17M | Why does halting fail? Which part of the family costs quality: expert tying, hierarchical routing, shared normalisation, the router's training rule? | **W1, 48 preregistered runs:** expert tying across hops and hierarchical routing each cost quality (separate banks −0.020…−0.036 nat, flat router −0.033 nat against FixedDepth); for shared normalisation and the training rule no effect was detected (inconclusive, not equivalence). An anytime (multi-exit) loss repairs the hard halting path (−0.70…−0.88 nat) but stays ≥ 1.18× the MoE loss. [W1 report](results/round_v3_w1/W1_REPORT.md) |
| **Round V3 — W2** (closed 2026-10-24) | ~17M | Attention sharing, FLOP-matched depth/Top-K/granularity points with matched MoE/Dense controls, adaptive K | **66 preregistered runs:** separate per-hop attention matters (sharing costs 0.11–0.17 nat). **At 0.85× routed FLOP the AM-LLM loses to an equal-parameter MoE by 0.044–0.054 nat.** The W1 combination (3 banks + flat routing, *FDC1*) is the only model below MoE (2.087 vs 2.118), but with 1.83× parameters and 1.38× executed FLOP; its parameter-matched MoE control is the next test. Adaptive K beats fixed K = 1; the halting lane stays ≥ 1.19× MoE. [W2 report](results/round_v3_w2/W2_REPORT.md) |

The detailed numbers, confidence intervals and limitations are in the [research report](results/RESEARCH_REPORT.md); the decisions behind each step are the [technical records](docs/04-decisions/README.md).

## What we have learned so far

1. **Routing is learned and used.** In the small models the learned routing reaches a ~20% lower loss than random routing with the same parameters and compute.
2. **Hop-specific attention matters most** in the small models; it is not a capacity artefact. W2 confirmed it at ~17M: sharing attention across hops costs 0.11–0.17 nat even with per-hop adapters.
3. **At ~17M no variant gets MoE quality with fewer executed FLOP.** The six-hop FixedDepth is structurally an *expert-tied MoE*; expert tying, two-level routing, shared normalisation and the router's training rule (`straight_through` vs `topk_softmax`) change together; the W1 round separated them: tying and hierarchical routing cost quality, for normalisation and the training rule no effect was detected (inconclusive at n = 3). The best FixedDepth variants (2.122–2.125) reach the dense control but not MoE (2.119).
4. **Current learned halting does not give a usable trade-off:** it is trained softly (all hops run) and applied hard (tokens stop), and that mismatch costs a lot. This is evidence against this *mechanism*, not against adaptive computation in general. An anytime (multi-exit) loss largely repairs the hard path (W1), but costs final quality and still saves little compute.
5. **Real speed has barely been measured:** one early batch-1 decode run had AM-LLM at 632 and MoE at 884 tokens/s.

## What is *not* claimed

No large-model result, no VRAM saving, no ≥ 60 tok/s, no superiority over MoE or Dense. The old test split was visible to an automated experiment planner and is not an untouched holdout. With three seeds the confidence intervals are wide: an interval that contains zero does **not** show equivalence. Estimated FLOP is not measured GPU work. Byte-level and BPE losses are not comparable.

## Where you can help

Controlled ablations, hard-aware halting, matched controls, profiling, replication and negative results are all welcome: [open research questions](docs/OPEN_QUESTIONS.md) · [how to contribute](CONTRIBUTING.md) · [Issues](https://github.com/sootylunatic/amllm-research/issues).

## Repository map

```text
src/amllm/   model, routing, data, training, experiments, profiling, fast inference paths, generation, evaluation
tests/       pytest suite (CPU; extra CUDA tests run when a GPU is present)
docs/        goals, architecture guide, methodology, technical decision records (English and Hungarian)
results/     research report, E00–E09 summary table
data/        a tiny toy file only; corpora are rebuilt locally (data/README.md)
```

## Quick start (smoke test, not a reproduction of the reported runs)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[dev,analysis]'
.venv/bin/python -m pytest -q
.venv/bin/python -m amllm.train --data data/toy_typescript.txt --steps 40 --batch 4 --max-seq-len 64 --run-dir runs/smoke
```

The full corpus, model weights and per-run logs are **not released**, so the public source alone does not reproduce the reported runs; the [report](results/RESEARCH_REPORT.md) is an audited secondary summary. MIT licence, see [LICENSE](LICENSE). Claude and Codex assisted implementation and review; all findings remain open to independent scrutiny.
