**English** | [Magyar](RESEARCH_REPORT.hu.md)

# Public research evidence report — snapshot 2026-10-23

This is a **curated, secondary summary** of closed internal experiments, not an independent rerun or a checkpoint release. Current private W1 experiments are **not** included.

## Series and interpretations

| Series | Scope | Interpretation |
|---|---|---|
| E00–E09 | Byte-level, ~0.3–2.5M parameters | Router can learn; per-hop attention improves quality; original go/no-go **failed** |
| F1–F4 | Byte-level controls / kernels | Mixed partial improvements, no general decode win established |
| F5 | ~17M, 8,192 BPE | MoE val 2.1166, AM soft ~2.189, AM hard ~3.345; F5 Q gate **failed** |
| Overnight 2026-10-22/23 | Same corpus/scale, 12 configurations × 3 paired seeds | Strongest FixedDepth variants approach MoE loss at higher estimated FLOP |

Do **not** directly compare byte-level NLL with BPE nat/token.

## Complete 36-run aggregate (12 configurations × seeds 42,43,44)

Training: 9,000 steps, batch 32, context 512, same tokenization and split; mean NLL from 512 validation windows, default evaluation mode. Relative FLOPs are analytical forward estimates vs MoE, **not executed kernel FLOPs on every backend** and not generated tok/s.

| Configuration | Mean val NLL ↓ | Rel. FLOP ↓ | Total params | Evaluation mode |
|---|---:|---:|---:|---|
| B_moe_3090 | **2.1166** | 1.00× | 17.15M | fixed |
| H05_h8_fixed | 2.1212 | 1.24× | 18.33M | 8 hops fixed |
| H03_m2e2_fixed | 2.1366 | 1.31× | 17.15M | Top-2/Top-2 fixed |
| H03_m2e1_fixed | 2.1558 | 1.00× | 17.15M | Top-2/Top-1 fixed |
| H01_fixed | 2.1625 | 1.00× | 17.15M | 6 hops fixed |
| H04_e8h192_fixed | 2.1804 | 0.85× | 17.17M | narrow experts fixed |
| H01_p005 | 2.1846 | 1.00× | 17.15M | hard survival |
| H04_m12h192_fixed | 2.1886 | 0.85× | 17.17M | more narrow modules fixed |
| H03_m1e1_fixed | 2.1943 | 0.85× | 17.15M | Top-1/Top-1 fixed |
| H05_h4_fixed | 2.2542 | 0.76× | 15.97M | 4 hops fixed |
| H01_bias_m2 | 3.2753 | 0.88× | 17.15M | hard survival |
| H01_p05 | 6.3677 | 0.49× | 17.15M | hard survival |

Paired 95% confidence intervals for AM−MoE validation NLL (nat/token, n=3): **H05_h8 +0.005 [−0.035,+0.044]**; **H03_m2e2 +0.020 [−0.018,+0.058]**. Intervals crossing zero **do not prove equivalence**. No tested AM variant demonstrates similar quality with lower executed compute than an appropriately matched MoE; lower-compute H04 is **inconclusive** without matched MoE/Dense controls.

## Failure analysis and limitations

Current survival training runs all hops with soft weights; hard inference truncates token updates. F5 seed 42: soft val NLL ~2.183; hard threshold 0.5 ~3.426 at ~3.47 hops. Even when fewer hops are used, quality degrades substantially. The original architecture hypothesis is **not disproven for all possible implementations**, but this mechanism is unsuccessful at this scale.

Earlier F5 batch-1 KV decode recorded **AM 632 vs MoE 884 tokens/s**; this is not an L4 throughput measurement of the best overnight configurations. Training peak VRAM and training tok/s do not measure inference memory or generation. 17M coding-task pass@1 does not establish large-model coding ability.

## Reproducibility and data integrity

The public repo ships source, unit tests, a toy fixture, [E00–E09 summary.csv](summary.csv), and this reviewed aggregate. It **does not contain** the full third-party corpus, learned weights/checkpoints, the 36 individual run JSONs, or private orchestration state. Thus the overnight table is **not independently recomputable from public artifacts alone**. A follow-on replication should freeze source commit, code hash, split/tokenizer manifest, seeds, scheduler/optimizer, evaluation mode, and matched compute baselines. Internal study identifiers: training-code SHA prefix `406c4eef88f4ba69`, data-split ID `355afd73091925ef` (not dataset download links).

The original automated experiment planner saw old test-loss values, so that split cannot be treated as an independent untouched holdout. No future sealed holdout is published. Hypotheses must be selected on validation only.

[Architecture](../docs/ARCHITECTURE_GUIDE.md) · [Open questions](../docs/OPEN_QUESTIONS.md) · [Contribute](../CONTRIBUTING.md).
