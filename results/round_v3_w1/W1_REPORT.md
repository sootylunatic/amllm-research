---
title: "W1 results — anchors, FixedDepth factors (FD01–FD03, FD07), anytime loss (AD02a)"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
---

**English** | [Magyar](W1_REPORT.hu.md)

# W1 results — anchors, FixedDepth factors (FD01–FD03, FD07), anytime loss (AD02a)

**Status:** final for W1. All **48 runs** of the [preregistration](PREREGISTRATION.md) and its amendments [A1](PREREGISTRATION_A1.md), [A2](PREREGISTRATION_A2.md), [A3](PREREGISTRATION_A3.md) finished with the requested 9,000 steps and finite results; none failed. Code `train_code_sha256` = `ba81b4fc02cd7b63` throughout. Every run directory (checkpoint included) is on the private backup server (SHA-256 verified per run); reports, logs and the analyses are in this directory. Test split and holdout were not used.
**Protocol:** `code_corpus_v2`, 9,000 steps, batch 32 × 512, seeds 42/43/44, validation NLL (nat/token) in the default mode (`fixed`; `hard_surv_0.5` for halting models). Decision rule (preregistered, directional): `SUPPORTED` = Δ ≤ −0.010, the 95% t-CI (n = 3, t = 4.303) entirely below 0 and 3/3 seeds negative; `NOT_SUPPORTED_IN_SCOPE` = the mirror image; otherwise `INCONCLUSIVE` (which is not equivalence). Machines: RunPod Community RTX 3090 Ti / RTX 3090 / RTX A4000 (GPU type is recorded per run and does not enter any decision).

## 1. Anchors (sanity check passed)

| Config | val NLL mean | seeds 42 / 43 / 44 | params | routed MFLOP/tok | executed MFLOP/tok | hops |
|---|---|---|---|---|---|---|
| B_moe (`moe_f5`) | **2.1185** | 2.1070 / 2.1311 / 2.1172 | 17.15M | 22.83 | 25.02 | 6 |
| B_dense (`dense_f5`) | 2.1253 | 2.1320 / 2.1157 / 2.1281 | 13.59M | 22.81 | 22.81 | 6 |
| B_fixed (`amllm_f5`, no halting) | 2.1585 | 2.1574 / 2.1568 / 2.1612 | 17.15M | 22.86 | 31.56 | 6 |
| A0 (`amllm_f5`, survival λ 0.02), hard θ 0.5 | 3.3821 | 3.4576 / 3.4105 / 3.2782 | 17.15M | 19.94 | 27.22 | 3.54 |

B_moe and B_fixed differ from the earlier values (2.1166 / 2.1625) by +0.002 / −0.004 nat, inside the ±0.03 tolerance: the anchors reproduce. A0's soft-mode NLL is 2.187; its hard path is far worse, as before.

## 2. FD lane — which of the confounded factors costs quality

Paired Δ = variant − reference (nat/token), 3 seeds.

| Question | Variant | params | Δ | 95% CI | Verdict |
|---|---|---|---|---|---|
| FD01a separate bank per hop (pool 24 per hop) | 2.1221 | 52.69M | −0.0363 | [−0.0493; −0.0234] | **SUPPORTED** (better) |
| FD01b separate banks, **equal total params** (pool 4 per hop) | 2.1384 | 17.15M | −0.0200 | [−0.0296; −0.0105] | **SUPPORTED** — package effect (A1) |
| FD01c partial sharing (3 banks) | 2.1345 | 31.37M | −0.0240 | [−0.0440; −0.0040] | **SUPPORTED** |
| FD01d control: one shared 4-expert bank | 2.2359 | 11.23M | +0.0774 | [+0.0482; +0.1066] | NOT_SUPPORTED_IN_SCOPE (worse) |
| FD02a flat 24-way router | 2.1251 | 17.15M | −0.0334 | [−0.0430; −0.0237] | **SUPPORTED** |
| FD02b random routing (router-value control) | 2.2874 | 17.15M | +0.1289 | [+0.1192; +0.1386] | NOT_SUPPORTED_IN_SCOPE (worse) |
| FD03a per-hop bank LayerNorm | 2.1611 | 17.15M | +0.0026 | [−0.0079; +0.0132] | INCONCLUSIVE |
| FD03b shared LN + per-hop scale/shift | 2.1624 | 17.15M | +0.0040 | [−0.0050; +0.0129] | INCONCLUSIVE |

**FD01 2×2 (A1: causal statements only within an equal per-hop pool; no pair has equal total parameters):**

| Pair | What it isolates | Δ | 95% CI | Verdict | params |
|---|---|---|---|---|---|
| FD01b − FD01d | sharing, pool 4 | −0.0974 | [−0.1353; −0.0596] | **SUPPORTED** (separate better) | 17.15M vs 11.23M |
| FD01a − B_fixed | sharing, pool 24 | −0.0363 | [−0.0493; −0.0234] | **SUPPORTED** (separate better) | 52.69M vs 17.15M |
| B_fixed − FD01d | pool size, shared | −0.0774 | [−0.1066; −0.0482] | **SUPPORTED** (24 better than 4) | 17.15M vs 11.23M |
| FD01a − FD01b | pool size, separate | −0.0163 | [−0.0368; +0.0042] | INCONCLUSIVE | 52.69M vs 17.15M |

**FD07 router training rule (A3, within one architecture):**

| Pair | Δ | 95% CI | Verdict |
|---|---|---|---|
| FD07a MoE: `straight_through` − `topk_softmax` (B_moe_st 2.1335 − B_moe) | +0.0150 | [−0.0356; +0.0656] | INCONCLUSIVE (seeds +0.022 / −0.008 / +0.031) |
| FD07b AM: `full_softmax` − `straight_through` (B_fixed_fs 2.1653 − B_fixed) | +0.0068 | [+0.0004; +0.0132] | INCONCLUSIVE (|Δ| below 0.010) |

**Per-bank expert usage (A2; `tools/analysis/bank_usage.py`, validation, 3 seeds each)** — the `expert_counts`-derived fields of the FD01a/b/c reports are invalid (merged banks) and are not quoted; the correct values:

| Run | banks | distinct experts | dead (per seed) | entropy (bits) / max | max share | least balanced bank (entropy / max) |
|---|---|---|---|---|---|---|
| FD01a | 6 | 144 | 2 / 0 / 0 | 6.90 / 7.17 | 0.021 | 0.91 |
| FD01b | 6 | 24 | 0 / 0 / 0 | 4.57 / 4.58 | 0.050 | 0.98 |
| FD01c | 3 | 72 | 0 / 0 / 0 | 5.94 / 6.17 | 0.035 | 0.92 |

**Reading of the FD lane (in scope: 17M parameters, this corpus, 9k steps):**
- Two of the four confounded AM–MoE differences cost measurable quality: **expert tying across hops** (FD01) and **hierarchical routing** (FD02a: a flat 24-way router is better by 0.033 at equal parameters and equal **routed** FLOP; the executed FLOP is not equal — 38.45 vs 31.56 MFLOP with bmm padding). The 2×2 shows that with a fixed per-hop pool separate banks are better, and with a shared bank a larger pool is better; the equal-parameter FD01b is a package effect.
- For the other two no effect was detected: **bank normalisation** (FD03a/b) and **the router training rule** (FD07a/b) are INCONCLUSIVE. This is **not** evidence that they do not matter: with n = 3 the CIs still allow effects between −0.008 and +0.013 (FD03a), −0.005 and +0.013 (FD03b), −0.036 and +0.066 (FD07a) and +0.0004 and +0.013 (FD07b) nat. FD07b's CI is above 0, but the effect (+0.007) is below the preregistered 0.010 threshold.
- Learned routing matters: random routing is worse by 0.129 (FD02b).
- The FD01 gains are not explained by dead experts in the shared bank: the separate banks use all (or all but two of 144) experts, nearly uniformly.
- The best FixedDepth variants (FD01a 2.1221, FD02a 2.1251) reach the dense control (2.1253) but not B_moe (2.1185). "Better than B_fixed" is not an A result: A needs an executed-FLOP-matched comparison (FD05, W2). Executed FLOP is higher than routed for every AM variant because of bmm padding (e.g. FD02a 38.45 vs 22.93 MFLOP).

## 3. AD lane — anytime (multi-exit) loss

| Config | hard θ 0.5 NLL | seeds | hops | routed MFLOP | executed MFLOP | soft NLL | Δ vs A0 (hard) | 95% CI | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 3.3821 | 3.458 / 3.411 / 3.278 | 3.54 | 19.94 | 27.22 | 2.187 | — | — | — |
| AD02a-1 (weight 0.3) | 2.6834 | 2.716 / 2.673 / 2.661 | 3.42 | 19.80 | 27.55 | 2.233 | −0.6987 | [−0.8740; −0.5234] | **SUPPORTED** |
| AD02a-2 (weight 1.0) | 2.5041 | 2.570 / 2.509 / 2.433 | 3.42 | 19.80 | 28.14 | 2.342 | −0.8780 | [−0.9505; −0.8055] | **SUPPORTED** |

**Primary gate (A1 screening signal):** hard θ 0.5 NLL ≤ 1.03 × B_moe **and** routed FLOP ≤ 0.90 × B_moe on 3/3 seeds. Quality ratio: AD02a-1 **1.267×**, AD02a-2 **1.182×** (A0 1.596×); routed FLOP ratio 0.867×. **0/3 seeds pass: not `PROMISING_EXPLORATORY`.** Executed FLOP (with bmm padding) is 1.10–1.12× B_moe's executed, i.e. no executed saving with this dispatch.

**Secondary diagnostic (AD01 re-run, `tools/analysis/ad01_trace.py`, 128 validation windows, trace = forward exactly on all 9 checkpoints):**

| | soft read-out per hop 1→6 | oracle\* | hard θ 0.1 | θ 0.3 | θ 0.5 | θ 0.8 |
|---|---|---|---|---|---|---|
| A0 | 6.17 · 4.70 · 3.99 · 3.05 · 2.45 · **2.17** | 1.86 | 2.17 @ 5.90 | 2.28 @ 5.13 | 3.33 @ 3.54 | 6.17 @ 1.00 |
| AD02a-1 | 4.08 · 3.32 · 2.94 · 2.35 · 2.27 · **2.22** | 1.92 | 2.22 @ 5.59 | 2.28 @ 4.53 | 2.64 @ 3.42 | 3.82 @ 1.35 |
| AD02a-2 | 3.62 · 3.13 · 2.64 · 2.38 · 2.34 · **2.32** | 2.03 | 2.33 @ 5.08 | 2.36 @ 4.10 | 2.47 @ 3.42 | 3.05 @ 2.17 |

\* per-token best hop knowing the target: an analytic bound, not a policy.

Paired against A0 (3 seeds): the hop-5 read-out gap to the final soft value shrinks from 0.15–0.39 to 0.05 (AD02a-1) and 0.01–0.02 (AD02a-2) — **3/3 seeds**, but under the directional CI rule INCONCLUSIVE because A0 varies a lot between seeds. The hard–soft gap shrinks with a CI below zero at θ 0.3 (−0.047 / −0.073) and θ 0.5 (−0.733 / −1.009): **SUPPORTED** for both weights. The control for the learned exit is a **real random stop**: a hard run in which every token stops at a hop drawn by shuffling the hard path's exit hops over the tokens (same exit distribution; `random_exit_matched_nll_hard` in `analysis/`). The learned exit is better at θ 0.1, 0.3 and 0.5 for A0, AD02a-1 and AD02a-2 (paired, 3 seeds, all **SUPPORTED**; e.g. θ 0.5: A0 3.334 vs 3.640, AD02a-1 2.645 vs 2.769, AD02a-2 2.475 vs 2.565); at θ 0.8 it is INCONCLUSIVE. An earlier version of this paragraph used a read-out of the *soft* trace at the random hop as the control; that is not a stopped run (later tokens attend to unfrozen states). It differs from the real stopped run by up to 0.031 nat (A0, θ 0.5) and no conclusion rests on it.

**Reading of the AD lane:** the anytime loss does what it was meant to do — intermediate states become readable and the hard path no longer collapses (θ 0.3 gives A0's quality with 4.5 instead of 5.1 hops). It costs final quality (soft 2.22 / 2.32 vs 2.17), and hard points within about 1.05× of B_moe need ≥ 5.6 average hops (AD02a-1 θ 0.1: 1.048× at 5.59; A0 θ 0.1: 1.026× at 5.90), where little compute is saved. The AD lane is therefore alive but not near goal A; the next step is selective compute below whole-hop granularity (AD03) and a weight between 0.3 and 1.0 that keeps the final quality.

## 4. What W1 does not establish
- No A evidence: no executed-FLOP-matched comparison yet (FD05, W2) and no padding-free execution measurement (W3).
- Single scale (17M), single corpus, 9k steps, n = 3: INCONCLUSIVE rows are not "no effect".
- The FD01 equal-parameter verdict (FD01b) is a package effect; FD01a uses 3× parameters.

## 5. Consequences for W2 (proposal, the W2 preregistration decides)
- **FD04/FD05 build on the improved FixedDepth:** test the attention questions (FD04) and the matched quality–compute points (FD05: 8 hops at 1.00× FLOP, top-2, H04 at 0.85×) on the base *and* on the best W1 factor combination (separate/partially separate banks + flat routing), each against FLOP- and parameter-matched MoE/dense controls.
- **FD01c+FD02a combination** (partial bank sharing with flat routing) is a natural candidate; whether the two gains add up is untested.
- **FD03 and FD07 are not carried forward** as primary questions — a budget priority, not a conclusion that they have no effect (INCONCLUSIVE); they stay open and would need more seeds to be resolved.
- **AD:** AD03 (adaptive K, below-hop selective compute) with the anytime loss as a base; an anytime weight around 0.5 as a compromise between the hard path and the final quality.

## Files
- `*/amllm_f5_s4?/` etc.: `config.json`, `env.json`, `report.json`, `train.log` per run (checkpoints on vps2 only); `*/summary.csv`.
- `analysis/`: `bank_usage__*.json` (A2), `ad01_trace__*.json` (AD01 re-run), per-pod `STATUS_*`.
- `queues/`, `_queue_w1*/`: the executed queues and worker logs.
