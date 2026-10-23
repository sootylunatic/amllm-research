**English** | [Magyar](RESEARCH_STATUS.hu.md)

# Research status — evidence snapshot 2026-10-23

| Objective | Evidence level | Missing proof |
|---|---|---|
| **A: quality per executed FLOP** | **Not supported at ~17M**: best AM approaches MoE NLL with +24–31% estimated FLOP; original E07 gate failed | Compute-matched Pareto improvement on independent data |
| **B: inference GPU memory** | Not shown; training VRAM is not working-set savings | Actual paging/offload, allocation and PCIe metrics |
| **C: decode speed** | Not shown: earlier F5 b=1 AM 632 vs MoE 884 decoded tok/s; overnight winners not fully benchmarked | Equal-backend prefill/decode/TTFT/VRAM |
| **D: larger-model capabilities** | Long-term ambition, not proven | Scale and coding-task evaluations |

Supported observations: early E09 learned routing was 19.9% better in relative validation loss than **random routing**, not MoE. Per-hop unshared attention improved small-model loss while increasing capacity. The 36 overnight runs (12 configs × 3 seeds) showed H05_h8 at NLL 2.1212, MoE 2.1166, with 1.24× FLOP and +6.9% parameters. The CI includes zero; **absence of significant difference is not equivalence**. The current soft-survival/hard-truncation mismatch causes severe quality loss at shorter depth. H04 at 0.85× FLOP remains **inconclusive**.

Limits: n=3 seeds; byte-level NLL is not directly comparable to BPE nat/token; estimated FLOP is not actual GPU kernel work or generated tokens/s; the historic planner saw test losses, contaminating that holdout. Public data lack full corpora, weights and per-run JSON, so this is an audited **secondary report**, not independent reproduction. [Full summary](../results/RESEARCH_REPORT.md) · [Architecture](ARCHITECTURE_GUIDE.md) · [Research questions](OPEN_QUESTIONS.md).
