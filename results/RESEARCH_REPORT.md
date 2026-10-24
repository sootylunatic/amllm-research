**English** | [Magyar](RESEARCH_REPORT.hu.md)

# Public research evidence report — snapshot 2026-10-23

This is a **curated, secondary summary** of closed internal experiments, not an independent rerun or a checkpoint release. The preregistered V3 round, waves W1–W4 (all closed), is summarised below.

## Series and interpretations

| Series | Scope | Interpretation |
|---|---|---|
| E00–E09 | Byte-level, ~0.3–2.5M parameters | Router can learn; per-hop attention improves quality; original go/no-go **failed** |
| F1–F4 | Byte-level controls / kernels | Mixed partial improvements, no general decode win established |
| F5 | ~17M, 8,192 BPE | MoE val 2.1166, AM soft ~2.189, AM hard ~3.345; F5 Q gate **failed** |
| Overnight 2026-10-22/23 | Same corpus/scale, 12 configurations × 3 paired seeds | Strongest FixedDepth variants approach MoE loss at higher estimated FLOP |
| V3 W1 2026-10-23 | Same corpus/scale, 16 configurations × 3 seeds, preregistered | Expert tying and hierarchical routing cost quality; anytime loss repairs the hard path but not to MoE level |
| V3 W2 2026-10-24 | Same corpus/scale, 22 configurations × 3 seeds, preregistered | Per-hop attention matters; AM loses to matched MoE at 0.85× routed FLOP; FDC1 below MoE only with 1.83× params |
| V3 W3 2026-10-24 | Same corpus, 17–31M, 5 configurations × 3 seeds + measurement, preregistered | Equal-size MoE beats FDC1 at 9k, not distinguishable at 18k; A gate not met; decode gap mostly execution (exploratory) |

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

## Round V3, W1 — preregistered ablations (48 runs, closed 2026-10-23)

Same corpus, scale and protocol as above (9,000 steps, seeds 42/43/44, validation NLL), new code version (`ba81b4fc02cd7b63`), [sealed preregistration](round_v3_w1/PREREGISTRATION.md) with a directional rule: `SUPPORTED` = Δ ≤ −0.010, 95% CI below 0, 3/3 seeds. Full per-seed tables: [W1 report](round_v3_w1/W1_REPORT.md).

| Question (Δ vs FixedDepth B_fixed 2.1585; MoE 2.1185, Dense 2.1253) | Δ nat | 95% CI | Verdict |
|---|---:|---|---|
| Separate bank per hop (3× params) | −0.036 | [−0.049, −0.023] | better |
| Separate banks, equal total params (package effect) | −0.020 | [−0.030, −0.011] | better |
| Flat 24-way router instead of hierarchical | −0.033 | [−0.043, −0.024] | better |
| Random routing (router-value control) | +0.129 | [+0.119, +0.139] | worse |
| Per-hop bank LayerNorm / per-hop scale-shift | +0.003 / +0.004 | CI contains 0 | inconclusive |
| Router rule: MoE with `straight_through` / AM with `full_softmax` | +0.015 / +0.007 | CI contains 0 / below threshold | inconclusive |
| Anytime loss 0.3 / 1.0 vs halting A0 (hard path) | −0.699 / −0.878 | below 0 | better; still 1.27× / 1.18× MoE |

"Equal FLOP" means routed FLOP; executed FLOP with bmm padding differs (reported in the W1 report). Inconclusive rows do not show the absence of an effect.

## Round V3, W2 — attention, matched points, FDC1, adaptive K (66 runs, closed 2026-10-24)

New code version (`cd909ea282660f0f`); anchors reproduce W1 within ±0.0015 (MoE 2.1180, Dense 2.1254, FixedDepth B_fixed 2.1599). [Sealed preregistration](round_v3_w2/PREREGISTRATION.md) · [W2 report](round_v3_w2/W2_REPORT.md) with all per-seed values.

| Question | Δ nat | 95% CI | Verdict |
|---|---:|---|---|
| One shared attention vs per-hop (B_fixed) | +0.174 | [+0.134, +0.214] | worse |
| Shared attention + per-hop adapter | +0.132 | [+0.110, +0.153] | worse |
| Every second hop expert-only (0.79× routed FLOP) | +0.114 | [+0.035, +0.192] | worse |
| 3 attention sets | +0.020 | [−0.012, +0.051] | inconclusive |
| 8 hops, equal routed FLOP and params | +0.008 | [−0.029, +0.045] | inconclusive |
| 4 hops, equal routed FLOP | +0.074 | [+0.031, +0.118] | worse |
| **AM finer experts − matched MoE, 0.85× routed FLOP, equal params** | **+0.054** | [+0.030, +0.077] | **worse** |
| **AM smaller experts − matched MoE, 0.86×, equal params** | **+0.044** | [+0.019, +0.069] | **worse** |
| FDC1 (3 banks + flat routing) − B_fixed | −0.073 | [−0.076, −0.070] | better |
| FDC1 − MoE (descriptive: 1.83× params, 1.38× executed FLOP) | −0.031 | [−0.052, −0.010] | lower loss, not matched |
| Anytime 0.5 vs halting A0 (hard path) | −0.859 | [−1.135, −0.583] | better; still 1.19× MoE |
| Adaptive K (p 0.7) − fixed K = 1 | −0.068 | [−0.096, −0.039] | better |

Equal FLOP means routed FLOP. The training split has 53M tokens, so every run sees it about 2.8 times; FDC1's lead over MoE shrinks during training (−0.127 at 2k steps, −0.031 at 9k). Inconclusive rows do not show the absence of an effect.

## Round V3, W3 — FDC1 against the equal-size MoE, longer training, practical measurement (closed 2026-10-24)

Training code unchanged from W2, so W2 runs are compared directly. [Sealed preregistration](round_v3_w3/PREREGISTRATION.md) · [W3 report](round_v3_w3/W3_REPORT.md) with per-seed values, curves and the measurement tables.

| Question | Δ nat | 95% CI | Verdict |
|---|---:|---|---|
| **FDC1 − MoE12** (equal params 31.4M, equal routed FLOP), 9k steps | **+0.019** | [+0.000, +0.038] | **worse** — the W2 lead came from parameters |
| FDC1 − MoE12 at 18k steps (data repeated ~5.6×) | +0.001 | [−0.018, +0.020] | inconclusive |
| Flat − hierarchical routing, 3 banks / 1 bank (equal params) | −0.046 / −0.034 | below 0 | better |
| 3 banks − 1 bank, hierarchical / flat (1.83× params) | −0.027 / −0.039 | below 0 | better |
| A success gate, halting models (padding-free executed FLOP ≤ 0.90× MoE **and** loss ≤ 1.03× MoE) | 0.80–0.86× FLOP, 1.18–1.23× loss | | **not met** (0/3) |

Exploratory decode benchmark (one RTX 3090, batch 1, not preregistered): fixing a per-call expert-weight conversion and the sort/scatter dispatch makes AM-LLM decoding 3–3.3× faster (FDC1 2,179 vs MoE 2,293 tokens/s) and the adaptive-depth model ~23× faster (88 → ~2,000 tokens/s); validation loss changes by ≤ 0.003. At batch 32 the MoE stays faster.

## Round V3, W4 — descriptive generalization check on an independent holdout (closed 2026-10-24)

A new holdout (H-A) of fresh npm packages, independent of all corpus splits by name, scope, name stem, repository and exact/near-duplicate content; part H1 (154 packages, 3.06M tokens) opened once for 36 frozen checkpoints; part H2 stays sealed. Not a confirmation (no AM candidate passed its gate). [Preregistration](round_v3_w4/PREREGISTRATION.md) · [W4 report](round_v3_w4/W4_REPORT.md).

| Pair | Validation Δ | Holdout Δ [package-bootstrap 95% CI] | Label |
|---|---:|---|---|
| FDC1 − MoE12 (equal size) | +0.019 | +0.017 [+0.014, +0.020] | transfers |
| Flat − hierarchical routing, 1 bank / 3 banks | −0.034 / −0.046 | −0.030 / −0.024 | transfers |
| 3 banks − 1 bank, hierarchical / flat | −0.027 / −0.039 | −0.008 / −0.002 (CI contains 0) | transfers / undecided |
| FDC1 − small MoE | −0.031 | +0.001 (CI contains 0) | undecided |
| Anytime loss vs halting A0 (hard path) | −0.859 | −0.937 | transfers |
| 18k vs 9k steps (MoE12 / FDC1) | −0.018 / −0.036 | holdout NLL rises 2.508 → 2.613 / 2.525 → 2.593 | longer training generalizes worse |

## Failure analysis and limitations

Current survival training runs all hops with soft weights; hard inference truncates token updates. F5 seed 42: soft val NLL ~2.183; hard threshold 0.5 ~3.426 at ~3.47 hops. Even when fewer hops are used, quality degrades substantially. The original architecture hypothesis is **not disproven for all possible implementations**, but this mechanism is unsuccessful at this scale.

Earlier F5 batch-1 KV decode recorded **AM 632 vs MoE 884 tokens/s**; this is not an L4 throughput measurement of the best overnight configurations. Training peak VRAM and training tok/s do not measure inference memory or generation. 17M coding-task pass@1 does not establish large-model coding ability.

## Reproducibility and data integrity

The public repo ships source, unit tests, a toy fixture, [E00–E09 summary.csv](summary.csv), and this reviewed aggregate. It **does not contain** the full third-party corpus, learned weights/checkpoints, the 36 individual run JSONs, or private orchestration state. Thus the overnight table is **not independently recomputable from public artifacts alone**. A follow-on replication should freeze source commit, code hash, split/tokenizer manifest, seeds, scheduler/optimizer, evaluation mode, and matched compute baselines. Internal study identifiers: training-code SHA prefix `406c4eef88f4ba69`, data-split ID `355afd73091925ef` (not dataset download links).

The original automated experiment planner saw old test-loss values, so that split cannot be treated as an independent untouched holdout. No future sealed holdout is published. Hypotheses must be selected on validation only.

[Architecture](../docs/ARCHITECTURE_GUIDE.md) · [Open questions](../docs/OPEN_QUESTIONS.md) · [Contribute](../CONTRIBUTING.md).
