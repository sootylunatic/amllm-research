---
title: "W2 results — attention sharing (FD04), matched quality–compute points (FD05), the W1 combination (FDC1), anytime weight and adaptive K (AD02a-3, AD03)"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
---

**English** | [Magyar](W2_REPORT.hu.md)

# W2 results — attention sharing (FD04), matched quality–compute points (FD05), the W1 combination (FDC1), anytime weight and adaptive K (AD02a-3, AD03)

**Status:** final for W2. All **66 runs** of the [sealed preregistration](PREREGISTRATION.md) finished with 9,000 steps and finite results; none failed. Code `train_code_sha256` = `cd909ea282660f0f` throughout. Every run directory (checkpoint included) is on the private backup server (SHA-256 verified per run); reports, logs and analyses are in this directory. Test split and holdout were not used. Machines: RunPod Community RTX 3090 Ti, RTX 3090 and RTX A4000 (the preregistration calls the third one "A40" — it was an **RTX A4000**; this only describes the queue balancing and enters no decision).
**Protocol and rule:** as preregistered — `code_corpus_v2`, 9,000 steps, batch 32 × 512, seeds 42/43/44, validation NLL (nat/token) in the default mode (`fixed`; `hard_surv_0.5` for halting models); `SUPPORTED` = Δ ≤ −0.010, 95% t-CI (n = 3) entirely below 0 and 3/3 seeds negative; `NOT_SUPPORTED_IN_SCOPE` = the mirror image; otherwise `INCONCLUSIVE`, which is **not** evidence of no effect. **"Equal / matched FLOP" always means routed FLOP**; executed FLOP (bmm dispatch, padding included) is reported next to it and differs.

## 1. Anchors — reproduce W1 within ±0.0015

| Config | val NLL | seeds 42 / 43 / 44 | W1 | params | routed MFLOP | executed MFLOP |
|---|---|---|---|---|---|---|
| B_moe | **2.1180** | 2.1120 / 2.1242 / 2.1177 | 2.1185 | 17.15M | 22.83 | 24.87 |
| B_dense | 2.1254 | 2.1328 / 2.1143 / 2.1291 | 2.1253 | 13.59M | 22.81 | 22.81 |
| B_fixed | 2.1599 | 2.1600 / 2.1577 / 2.1620 | 2.1585 | 17.15M | 22.86 | 31.60 |
| A0 (hard θ 0.5) | 3.3828 | 3.4404 / 3.4393 / 3.2687 | 3.3821 | 17.15M | 19.95 | 27.32 |

## 2. All configurations

Ratios are to B_moe (NLL, routed FLOP, executed FLOP).

| Config | val NLL | seeds | params | routed / executed MFLOP | NLL× | routed× | exec× |
|---|---|---|---|---|---|---|---|
| FD04a shared attention | 2.3337 | 2.3154 / 2.3433 / 2.3425 | 14.19M | 22.86 / 32.27 | 1.102 | 1.001 | 1.298 |
| FD04b 3 attention sets | 2.1796 | 2.1830 / 2.1882 / 2.1676 | 15.37M | 22.86 / 32.93 | 1.029 | 1.001 | 1.324 |
| FD04c shared + r=32 adapter | 2.2914 | 2.2871 / 2.2992 / 2.2880 | 14.31M | 23.10 / 32.44 | 1.082 | 1.012 | 1.304 |
| FD04d every 2nd hop expert-only | 2.2738 | 2.2521 / 2.3078 / 2.2615 | 15.37M | 18.14 / 29.22 | 1.074 | 0.794 | 1.175 |
| FD05a 8 hops | 2.1677 | 2.1746 / 2.1757 / 2.1527 | 17.17M | 22.90 / 31.21 | 1.023 | 1.003 | 1.255 |
| FD05b 4 hops | 2.2342 | 2.2543 / 2.2204 / 2.2277 | 16.55M | 22.83 / 29.23 | 1.055 | 1.000 | 1.175 |
| FD05c top-2 module / top-1 expert | 2.1616 | 2.1647 / 2.1516 / 2.1684 | 17.15M | 22.88 / 31.53 | 1.021 | 1.002 | 1.268 |
| FD05d top-2 / top-2, expert 192 | 2.1984 | 2.2045 / 2.2041 / 2.1865 | 13.61M | 22.88 / 31.72 | 1.038 | 1.002 | 1.276 |
| FD05e finer experts (8/module, 192) | 2.1817 | 2.1679 / 2.1856 / 2.1916 | 17.17M | 19.34 / 27.26 | 1.030 | 0.847 | 1.096 |
| FD05e_moe matched MoE | 2.1280 | 2.1223 / 2.1213 / 2.1403 | 17.17M | 19.31 / 21.30 | 1.005 | 0.846 | 0.857 |
| FD05f smaller experts (208) | 2.2061 | 2.2012 / 2.2207 / 2.1965 | 13.90M | 19.61 / 24.14 | 1.042 | 0.859 | 0.970 |
| FD05f_moe matched MoE | 2.1622 | 2.1628 / 2.1650 / 2.1588 | 13.91M | 19.59 / 20.71 | 1.021 | 0.858 | 0.833 |
| FD05f_dense matched dense | 2.1676 | 2.1724 / 2.1661 / 2.1642 | 11.97M | 19.57 / 19.57 | 1.023 | 0.857 | 0.787 |
| **FDC1** 3 banks + flat routing | **2.0870** | 2.0886 / 2.0842 / 2.0881 | **31.36M** | 22.93 / 34.23 | **0.985** | 1.004 | 1.376 |
| AD02a-3 anytime 0.5 (hard θ 0.5) | 2.5237 | 2.4938 / 2.5427 / 2.5347 | 17.15M | 19.72 / 28.48 | 1.192 | 0.864 | 1.145 |
| AD03a + adaptive K, p 0.5 | 2.5593 | 2.5008 / 2.6324 / 2.5446 | 17.15M | 17.80 / 23.42 | 1.208 | 0.779 | 0.942 |
| AD03b + adaptive K, p 0.7 | 2.5409 | 2.4830 / 2.5968 / 2.5428 | 17.15M | 18.28 / 25.54 | 1.200 | 0.800 | 1.027 |
| AD03c + fixed K = 1 | 2.6085 | 2.5389 / 2.6649 / 2.6217 | 17.15M | 17.68 / 20.97 | 1.232 | 0.774 | 0.843 |

## 3. FD04 — separate per-hop attention matters

| Pair (vs B_fixed) | Δ | 95% CI | Verdict |
|---|---|---|---|
| FD04a one shared attention | +0.1738 | [+0.1336; +0.2141] | NOT_SUPPORTED_IN_SCOPE (worse) |
| FD04b 3 attention sets | +0.0197 | [−0.0120; +0.0513] | INCONCLUSIVE (all three seeds positive) |
| FD04c shared + per-hop adapter | +0.1315 | [+0.1101; +0.1530] | NOT_SUPPORTED_IN_SCOPE (worse) |
| FD04d every second hop expert-only | +0.1139 | [+0.0354; +0.1924] | NOT_SUPPORTED_IN_SCOPE (worse) |

Sharing the attention across hops costs 0.11–0.17 nat even with a per-hop adapter, and skipping attention on every second hop (0.79× routed FLOP) costs 0.11 nat — 1.07× the MoE loss. Three attention sets (hops in pairs) are not distinguishable from separate per-hop attention at n = 3, but the CI allows up to +0.051 nat. The per-hop attention, which the MoE also has per layer, carries much of the quality; this agrees with the earlier small-model finding.

## 4. FD05 — matched quality–compute points; the A-relevant pairs fail

| Pair | Δ | 95% CI | Verdict |
|---|---|---|---|
| FD05a 8 hops − B_fixed (equal routed FLOP and params) | +0.0078 | [−0.0292; +0.0447] | INCONCLUSIVE |
| FD05b 4 hops − B_fixed | +0.0742 | [+0.0309; +0.1176] | NOT_SUPPORTED_IN_SCOPE (worse) |
| FD05c top-2/top-1 − B_fixed | +0.0016 | [−0.0153; +0.0185] | INCONCLUSIVE |
| FD05d top-2/top-2 − B_fixed (0.79× params) | +0.0385 | [+0.0084; +0.0685] | NOT_SUPPORTED_IN_SCOPE (worse) |
| FD05e finer experts − B_fixed | +0.0218 | [−0.0081; +0.0517] | INCONCLUSIVE |
| FD05f smaller experts − B_fixed | +0.0462 | [+0.0094; +0.0831] | NOT_SUPPORTED_IN_SCOPE (worse) |
| **FD05e − FD05e_moe** (A-relevant: 0.85× routed FLOP, equal params) | **+0.0537** | [+0.0301; +0.0774] | **NOT_SUPPORTED_IN_SCOPE (worse)** |
| **FD05f − FD05f_moe** (A-relevant: 0.86×, equal params) | **+0.0440** | [+0.0188; +0.0691] | **NOT_SUPPORTED_IN_SCOPE (worse)** |
| FD05f − FD05f_dense (0.86×) | +0.0386 | [+0.0039; +0.0733] | NOT_SUPPORTED_IN_SCOPE (worse) |

- **At 0.85× routed FLOP the AM-LLM is worse than an equal-parameter MoE by 0.044–0.054 nat**; FD05f is also worse than the matched dense control. The matched MoE loses little when made cheaper (2.1280 at 0.85× vs 2.1180 at 1.00×); the AM-LLM loses more. At this scale the A hypothesis is therefore not supported at the 0.85× point either (H04 is no longer `INCONCLUSIVE` for this design: it is `NOT_SUPPORTED_IN_SCOPE`).
- Depth: 8 hops at equal routed FLOP and parameters is not distinguishable from 6 hops (CI up to +0.045); 4 hops is clearly worse.
- Top-K: top-2 modules with top-1 expert is not distinguishable from the base (CI −0.015…+0.019, which is not equivalence); top-2/top-2 is worse, but it also has 0.79× the parameters, so the two cannot be separated.

## 5. FDC1 — the W1 combination has the lowest loss in W2, with 1.83× parameters

| Pair | Δ | 95% CI | Verdict |
|---|---|---|---|
| FDC1 − B_fixed | −0.0729 | [−0.0762; −0.0696] | **SUPPORTED** (better) |
| FDC1 − B_moe (descriptive, not matched) | −0.0310 | [−0.0519; −0.0101] | better on 3/3 seeds |

Partial bank sharing (3 banks, each used by two hops) with a flat 24-way router reaches **2.0870**, below MoE (2.1180) and dense (2.1254), at equal **routed** FLOP (1.004×). It is not an A result: it has **1.83× the parameters** (31.36M) and **1.38× the executed FLOP** of B_moe (34.23 vs 24.87 MFLOP; bmm padding over 72 experts). The missing control is a MoE with the same parameter count at equal routed FLOP (`moe_f5 n_experts=12`, top-2: 31.37M parameters at 1.002× routed FLOP, measured with `tools/fd05_match.py`). Whether the two W1 factors add up cannot be read from W2 alone: the single factors were measured in W1 under another code version, and no comparison across hashes is made.

## 6. AD — anytime weight 0.5 and adaptive K

| Pair | Δ | 95% CI | Verdict |
|---|---|---|---|
| AD02a-3 − A0 (hard θ 0.5) | −0.8591 | [−1.1352; −0.5829] | **SUPPORTED** (better) |
| AD03a (p 0.5) − AD02a-3 | +0.0355 | [−0.0811; +0.1522] | INCONCLUSIVE (−10% routed FLOP) |
| AD03b (p 0.7) − AD02a-3 | +0.0171 | [−0.0659; +0.1001] | INCONCLUSIVE (−7% routed FLOP) |
| AD03a − AD03c (fixed K = 1) | −0.0492 | [−0.1096; +0.0111] | INCONCLUSIVE (+0.7% routed FLOP) |
| **AD03b − AD03c** | **−0.0676** | [−0.0962; −0.0391] | **SUPPORTED** (better, +3.4% routed FLOP) |
| AD03c − AD02a-3 | +0.0848 | [−0.0111; +0.1807] | INCONCLUSIVE |

**Primary gate (A1 screening signal):** hard θ 0.5 NLL ≤ 1.03 × B_moe and routed FLOP ≤ 0.90 × B_moe on 3/3 seeds — **0/3 for every AD variant** (NLL 1.19–1.23× MoE at 0.77–0.86× routed FLOP). Not `PROMISING_EXPLORATORY`.

**Secondary (AD01, `tools/analysis/ad01_trace.py`, 128 validation windows, trace = forward exactly on all 15 checkpoints):**

| | soft read-out per hop 1→6 | hard θ 0.1 | θ 0.3 | θ 0.5 | θ 0.8 |
|---|---|---|---|---|---|
| A0 | 6.17 · 4.68 · 4.01 · 3.06 · 2.48 · **2.18** | 2.18 @ 5.90 | 2.29 @ 5.14 | 3.34 @ 3.55 | 6.17 @ 1.00 |
| AD02a-3 | 3.84 · 3.22 · 2.66 · 2.33 · 2.28 · **2.26** | 2.26 @ 5.33 | 2.31 @ 4.21 | 2.49 @ 3.35 | 3.42 @ 1.74 |
| AD03a | 3.84 · 3.25 · 2.58 · 2.34 · 2.30 · **2.27** | 2.27 @ 5.32 | 2.32 @ 4.18 | 2.52 @ 3.24 | 3.51 @ 1.57 |
| AD03b | 3.85 · 3.25 · 2.68 · 2.34 · 2.29 · **2.27** | 2.27 @ 5.37 | 2.31 @ 4.28 | 2.51 @ 3.36 | 3.49 @ 1.64 |
| AD03c | 3.87 · 3.25 · 2.64 · 2.39 · 2.32 · **2.29** | 2.30 @ 5.34 | 2.35 @ 4.20 | 2.57 @ 3.24 | 3.60 @ 1.48 |

- Against A0 (paired): the hop-5 read-out gap to the final soft value shrinks by ~0.28 nat and the hard–soft gap at θ 0.5 by 0.88–0.93 nat for all four variants (**SUPPORTED**); at θ 0.3 SUPPORTED for AD02a-3, AD03a and AD03b, INCONCLUSIVE for AD03c.
- Learned exit vs a **real random stop** (hard run, tokens stopped by a shuffle of the hard path's exit hops): the learned exit is better at θ 0.3 and 0.5 for all five configurations and at θ 0.1 for all but AD03c (SUPPORTED); at θ 0.8 INCONCLUSIVE everywhere.

**Reading:** the anytime weight 0.5 repairs the hard path (−0.86 nat against A0 within W2; descriptively its hard and soft values lie between those of W1's weights 0.3 and 1.0, but no comparison across hashes is made). Adaptive K saves compute against the full-K model (routed −10% / −7%, executed −18% / −10% for p 0.5 / 0.7); its quality difference to the full-K model is INCONCLUSIVE with wide CIs (up to +0.15 nat), so no claim of a cost-free saving is made. At p = 0.7 it is clearly better than fixed K = 1 for 3% more routed FLOP. None of this brings the AD lane near MoE quality: the best hard point at θ 0.5 is still 1.19× MoE.

## 7. What W2 does not establish
- No A evidence: the A-relevant matched pairs fail; FDC1 is better than MoE only with more parameters and more executed FLOP; no padding-free execution measurement exists (W3).
- One scale (17M), one corpus, 9k steps, n = 3: INCONCLUSIVE rows (FD04b, FD05a, FD05c, FD05e − B_fixed, the AD03 − AD02a-3 pairs) are not "no effect".
- FD05d mixes top-K with fewer parameters; FD05b has no expert choice inside its modules.

## 8. Consequences (proposal; the next preregistration decides)
- **FixedDepth:** keep separate per-hop attention (FD04). The only configuration that beats MoE is FDC1; the decisive next test is **FDC1 against a parameter- and routed-FLOP-matched MoE** (`moe_f5 n_experts=12`, 31.37M), plus its executed cost on a padding-free path. Single-factor controls (3 banks alone, flat routing alone) under the same hash would show whether the factors add up.
- **A at lower compute:** with this design the AM-LLM loses to matched MoE at 0.85× (FD05e/f). A further reduced-compute point is only worth testing on top of the FDC1 structure.
- **AdaptiveDepth:** the anytime loss (0.5) and adaptive K (p 0.7) are the current best AD base; the gap to MoE (1.19×) is the open problem. The next AD step should target the final quality, not further compute savings.

## Files
- `*/<preset>_s4?/`: `config.json`, `env.json`, `report.json`, `train.log` per run (checkpoints on the backup server only); `*/summary.csv`.
- `analysis/`: `ad01_trace__*.json` (AD01 with the real random-stop control) for A0, AD02a-3, AD03a/b/c; per-pod `STATUS_*`.
- `queues/`, `_queue_w2*/`: the executed queues and worker logs.
