---
title: "W4 results — descriptive generalization check on the independent holdout H1"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
---

**English** | [Magyar](W4_REPORT.hu.md)

# W4 results — descriptive generalization check on the independent holdout H1

**Status:** final. Per the [sealed preregistration](PREREGISTRATION.md) this is **not a confirmation** (no AM candidate passed its gate, so the confirmatory N = 0); it checks whether V3's validation findings carry over to fresh packages. Holdout H1 was opened **once** (one evaluator call, hash-chained log `holdout_eval/calls.log`), with the sealed holdout (`SHA256SUMS` `6bc776c0…`) and the frozen list of 36 checkpoints (`690fd214…`); both hashes were verified by the evaluator before it ran. The holdout files were deleted from the GPU machine afterwards; H2 stays sealed. One RTX 3090, bf16; results backed up and SHA-256 verified.

## 1. Levels and ranking

| Config | val NLL | holdout H1 NLL | holdout − val |
|---|---|---|---|
| MoE12 | 2.0680 | **2.5084** | +0.440 |
| B_dense | 2.1254 | 2.5173 | +0.392 |
| B_moe | 2.1180 | 2.5244 | +0.406 |
| FDC1 | 2.0870 | 2.5254 | +0.438 |
| B1F | 2.1263 | 2.5270 | +0.401 |
| B3H | 2.1332 | 2.5492 | +0.416 |
| B_fixed | 2.1599 | 2.5567 | +0.397 |
| FDC1_18k | 2.0510 | 2.5934 | +0.542 |
| MoE12_18k | 2.0499 | 2.6128 | +0.563 |
| AD02a-3 | 2.5237 | 2.9994 | +0.476 |
| AD03b | 2.5409 | 3.0142 | +0.473 |
| A0 | 3.3828 | 3.9367 | +0.554 |

The holdout is ~0.4 nat harder than validation for every model (another package distribution, as expected). Spearman correlation of the 12 validation and holdout means: **0.52**. The main reason the ranking moves: **the 18k-step models, best on validation, are among the worst on fresh code** — they got *worse* than their own 9k versions (MoE12 2.508 → 2.613, FDC1 2.525 → 2.593). With 53M training tokens, 18k steps (5.6 epochs) fit the training distribution and generalize worse; the W3 18k comparison measured mostly that regime.

## 2. The preregistered pairs

Holdout: token-weighted paired difference over 154 packages (seed means), 95% CI by package bootstrap (10,000). Validation: the W2/W3 paired-seed results. Labels as preregistered: *transfers* = same sign and holdout CI excludes 0; *not reproduced* = opposite sign and CI excludes 0; otherwise *undecided*.

| Pair | val Δ (verdict) | holdout Δ [95% CI] | packages where a is better | Label |
|---|---|---|---|---|
| FDC1 − MoE12 | +0.019 (worse) | **+0.017** [+0.014; +0.020] | 20% | transfers |
| FDC1_18k − MoE12_18k | +0.001 (inconclusive) | −0.019 [−0.025; −0.015] | 75% | not reproduced |
| B1F − B_fixed (flat routing, 1 bank) | −0.034 (better) | **−0.030** [−0.033; −0.026] | 93% | transfers |
| FDC1 − B3H (flat routing, 3 banks) | −0.046 (better) | **−0.024** [−0.031; −0.016] | 90% | transfers |
| B3H − B_fixed (3 banks, hierarchical) | −0.027 (better) | −0.008 [−0.013; −0.002] | 73% | transfers |
| FDC1 − B1F (3 banks, flat) | −0.039 (better) | −0.002 [−0.011; +0.010] | 64% | undecided |
| MoE12 − B_moe | −0.050 (better) | −0.016 [−0.024; −0.007] | 81% | transfers |
| B_dense − B_moe | +0.007 (inconclusive) | −0.007 [−0.016; −0.001] | 51% | not reproduced |
| FDC1 − B_moe | −0.031 (better) | +0.001 [−0.007; +0.010] | 63% | undecided |
| AD02a-3 − A0 (anytime loss) | −0.859 (better) | **−0.937** [−0.975; −0.903] | 100% | transfers |
| AD03b − AD02a-3 (adaptive K) | +0.017 (inconclusive) | +0.015 [+0.012; +0.018] | 14% | transfers |

## 3. Reading
- **What carries over to fresh code:** flat routing beats hierarchical routing (with 1 and with 3 banks, ~0.024–0.030 nat, 90%+ of packages); the anytime loss repairs the hard halting path (−0.94 nat, every package); FDC1 stays worse than the equal-size MoE12; the larger MoE stays better than the small one; adaptive K costs a little quality (+0.015) in exchange for its compute saving.
- **What shrinks or disappears:** the gain from separate banks is much smaller on fresh code (−0.008 hierarchical, undecided with flat routing); FDC1's validation lead over the small MoE (−0.031) is gone (+0.001, undecided). Parameter-heavy gains measured on validation partly reflect the training distribution.
- **The 18k result reverses on fresh code:** FDC1_18k is better than MoE12_18k on the holdout (−0.019), but both are worse than at 9k; this reflects how the two overfit the repeated data, not a structural advantage, and it marks 9k steps (or a larger corpus) as the right regime for these comparisons.
- **Labels for two pairs whose validation result was inconclusive** ("not reproduced") follow the preregistered rule mechanically; they mean a clear holdout difference where validation saw none.
- No A evidence follows from any of this (no executed-compute claim; descriptive check).

## Files
`holdout_eval/H1_results.json` (per run and package: NLL sums and token counts — aggregates only), `holdout_eval/calls.log` (hash-chained), `holdout_eval/W4_comparison.json` / `.txt`, `candidates_frozen.json`, `pairs.json`, `val_per_seed.json`. The holdout content is not in the repository.
