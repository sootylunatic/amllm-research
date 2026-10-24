**English** | [Magyar](OPEN_QUESTIONS.hu.md)

# Open research questions — contributions welcome

**Evidence snapshot: 2026-10-24 (after V3 W3).** Proposed contribution tracks with the current evidence. Check [Issues](https://github.com/sootylunatic/amllm-research/issues) before duplicating work.

| ID | Question | Minimum useful evidence | Status (V3) |
|---|---|---|---|
| **R-01** | Does tying all 24 experts across hops help? | Tied vs untied controls; routing/attention constant, FLOPs/params reported separately | **W1: tying costs quality** (separate banks −0.020…−0.036 nat; the equal-parameter result is a package effect). Open: at larger scale, with matched executed compute |
| **R-02** | Is hierarchical module→expert routing better than flat routing? | Flat 24-expert control, matched Top-K/FLOP, verified router gradients | **W1: flat is better** (−0.033 nat, equal params and routed FLOP; executed FLOP higher). Open: why, and whether it holds at scale |
| **R-03** | Is shared bank-LayerNorm a source of loss? | Shared vs hop-specific norms; only one factor varied | W1: no effect detected (inconclusive; CIs −0.008…+0.013 and −0.005…+0.013) — more seeds needed |
| **R-04** | Can hard-aware halting reduce the soft/hard quality gap? | Hard-path correctness, intermediate NLL, actual FLOPs, fixed-depth control | **W1: an anytime loss cuts the gap strongly** (−0.70…−0.88 nat), still ≥ 1.18× MoE. W2: anytime 0.5 and adaptive K (better than fixed K = 1) still ≥ 1.19× MoE. Open |
| **R-05** | Can grouped/shared attention or adapters retain quality? | Separate-per-hop positive control and matched compute/parameters | **W2: no** at ~17M — shared attention +0.17, with per-hop adapter +0.13, every second hop expert-only +0.11 nat; 3 attention sets inconclusive (+0.02, CI up to +0.05) |
| **R-06** | Why is decode slower than MoE? | Same GPU/backend decode, prefill, TTFT, VRAM and profiler | **W3 (exploratory): mostly execution** — a per-call expert-weight conversion and sort/scatter dispatch; fixed paths reach ~95% of MoE at batch 1. Open: large batches, a preregistered measurement |
| **R-07** | Does smaller-expert H04 give an efficient Pareto point? | 0.85×-FLOP MoE/Dense controls and uncertainty intervals | **W2: no** — at 0.85× routed FLOP the AM-LLM is 0.044–0.054 nat worse than an equal-parameter MoE and worse than the matched dense control |
| **R-08** | Do findings generalize? | License-audited data, family-level deduplication, paired seeds, external distribution | Open |
| **R-09** | Does the router training rule explain part of the AM–MoE gap? (MoE `topk_softmax`, AM-LLM `straight_through`) | Both rules on both architectures, everything else equal, paired seeds | W1: no effect detected (inconclusive; MoE CI −0.036…+0.066) — more seeds needed |
| **R-10** | Is the FDC1 advantage (3 banks + flat routing, 2.087 vs MoE 2.118) structural or just more parameters? | A parameter- and routed-FLOP-matched MoE (12 experts/layer, 31.4M), single-factor controls, longer training, executed cost on a padding-free path | **W3: mostly parameters** — FDC1 loses to the equal-size MoE12 at 9k steps (+0.019); at 18k not distinguishable (+0.001, inconclusive, repeated data). Open: a larger corpus |

A useful PR can contain a correctness test, a reproducible CPU counterexample, a preregistered pilot or a controlled multi-seed experiment. **Negative findings count.** Identify your hypothesis, baseline, constant factors, success/failure rule and evaluation budget. Use validation for iteration, **never** the historical planner-exposed test or a sealed future holdout. Never publish unpublished corpora, weights, credentials or private research systems. [Contributing](../CONTRIBUTING.md) · [Architecture](ARCHITECTURE_GUIDE.md) · [Results](../results/RESEARCH_REPORT.md).
