---
title: "W4 preregistration — descriptive generalization check on the independent holdout"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
locked: true
---

**English** | [Magyar](PREREGISTRATION.hu.md)

# W4 preregistration — descriptive generalization check on the independent holdout

**Status:** APPROVED AND SEALED — approved by the owner on 2026-10-24 ("rendben"). The content is recorded in `PREREGISTRATION.sha256`; holdout part H1 is opened only after this, once. In case of a difference between the two languages the Hungarian text governs.

## 1. Why descriptive, not confirmatory
By the preregistered rule of the holdout plan a branch gets a candidate slot only if its candidate passed its own preregistered gate. At the end of V3 **no AM candidate passed** (FixedDepth: FDC1 lost to the equal-size MoE12, W3; AdaptiveDepth: the A gate 0/3, W2–W3), so the confirmatory evaluation has **N = 0** candidates. On 2026-10-24 the owner decided ("option 2, tests are always important") to use holdout part **H1** for a **descriptive generalization check**: do V3's validation findings carry over to fresh, independent packages? **This is not confirmation:** no model becomes "holdout-confirmed", and the result does not feed back into selection. **H2** (the sealed reserve) stays untouched for a future confirmatory evaluation.

## 2. The holdout
H-A, `tools/holdout/build_holdout.py`, built 2026-10-24 on the private backup server: fresh npm packages (latest versions), independent of all three `code_corpus_v2` splits by name, scope, name stem and repository, with per-file exact and near-duplicate (MinHash, Jaccard ≥ 0.8) filtering against every v2 document; the v2 tokenizer; at most 0.2 MB per package; H1 and H2 disjoint by family. **The sealed holdout:** H1 3,055,222 tokens, 154 packages, 1,980 files (the largest package holds 3.2% of the tokens, median 7,121 tokens per package); H2 3,054,781 tokens, 149 packages (sealed reserve). Excluded: 299 packages for a family match, 6 packages for leakage, 46 files as exact and 46 as near duplicates (out of 26,350 candidates). `SHA256SUMS` SHA-256: `6bc776c0dae7d2fc9d03b9c34c050fadbdf39d28f9012edb022278157b01d379`; with a different hash the evaluator refuses to run. Two earlier builds were discarded before sealing because they did not meet the plan (one package held too large a share); no model was run on them.

## 3. Frozen models (no selection: every main V3 configuration, 3 seeds)
Code hash `cd909ea282660f0f` (W2/W3). The checkpoints' SHA-256 are recorded in the frozen list (`candidates_frozen.json`, 36 checkpoints; the file's SHA-256 is `690fd214c8bd601e08cf834174a85b60b19367fc15431e69a86895f92caa910b`).

| Group | Configurations |
|---|---|
| Controls | B_moe, MoE12, B_dense |
| FixedDepth | B_fixed, B3H, B1F, FDC1 |
| Longer training | FDC1_18k, MoE12_18k |
| AdaptiveDepth | A0, AD02a-3, AD03b |

12 configurations × 3 seeds = 36 checkpoints.

## 4. Measurement and comparisons
- **Metric:** holdout NLL (nat/token) in the default mode (`fixed`; hard θ 0.5 for halting models), bf16, identical non-overlapping 512-token windows; bits/byte alongside.
- **Statistics:** paired difference per package (mean over seeds), token-weighted; 95% CI by **package-level bootstrap** (10,000 resamples) — the package is the independent unit.
- **Pairs** (each reported next to its validation counterpart):
  1. FDC1 − MoE12 · 2. FDC1_18k − MoE12_18k · 3. B1F − B_fixed · 4. FDC1 − B3H · 5. B3H − B_fixed · 6. FDC1 − B1F · 7. MoE12 − B_moe · 8. B_dense − B_moe · 9. FDC1 − B_moe · 10. AD02a-3 − A0 · 11. AD03b − AD02a-3
- **Transfer label (descriptive, no gate):** *transfers* if the holdout difference has the validation sign and its CI excludes 0; *not reproduced* if the sign is opposite and the CI excludes 0; otherwise *undecided*.
- **Ranking:** Spearman correlation of the 12 configurations' validation and holdout means.
- **Absolute level:** the holdout distribution may differ from validation; conclusions rest on the paired differences and the ranking, not the level.

## 5. Procedure, cost
- One evaluation on one GPU (RTX 3090 or similar) with `python -m amllm.holdout evaluate`; every call goes into a hash-chained log; the result is reported even if unfavourable.
- The holdout files are on the GPU machine only for the evaluation and are deleted afterwards; the content is not public (package list and hashes are).
- ~36 checkpoints × ~3M tokens: a few minutes of GPU time, < $0.5.

## 6. Not part of W4
Opening H2; any model selection or tuning based on the holdout; new training.
