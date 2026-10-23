**English** | [Magyar](OPEN_QUESTIONS.hu.md)

# Open research questions — contributions welcome

**Evidence snapshot: 2026-10-23.** Proposed contribution tracks, **not a claim that current private W1 runs are finished**. Check [Issues](https://github.com/sootylunatic/amllm-research/issues) before duplicating work.

| ID | Question | Minimum useful evidence |
|---|---|---|
| **R-01** | Does tying all 24 experts across hops help? | Tied vs untied controls; routing/attention constant, FLOPs/params reported separately |
| **R-02** | Is hierarchical module→expert routing better than flat routing? | Flat 24-expert control, matched Top-K/FLOP, verified router gradients |
| **R-03** | Is shared bank-LayerNorm a source of loss? | Shared vs hop-specific norms; only one factor varied |
| **R-04** | Can hard-aware halting reduce the soft/hard quality gap? | Hard-path correctness, intermediate NLL, actual FLOPs, fixed-depth control |
| **R-05** | Can grouped/shared attention or adapters retain quality? | Separate-per-hop positive control and matched compute/parameters |
| **R-06** | Why is decode slower than MoE? | Same GPU/backend decode, prefill, TTFT, VRAM and profiler |
| **R-07** | Does smaller-expert H04 give an efficient Pareto point? | 0.85×-FLOP MoE/Dense controls and uncertainty intervals |
| **R-08** | Do findings generalize? | License-audited data, family-level deduplication, paired seeds, external distribution |

A useful PR can contain a correctness test, a reproducible CPU counterexample, a preregistered pilot or a controlled multi-seed experiment. **Negative findings count.** Identify your hypothesis, baseline, constant factors, success/failure rule and evaluation budget. Use validation for iteration, **never** the historical planner-exposed test or a sealed future holdout. Never publish unpublished corpora, weights, credentials or private research systems. [Contributing](../CONTRIBUTING.md) · [Architecture](ARCHITECTURE_GUIDE.md) · [Results](../results/RESEARCH_REPORT.md).
