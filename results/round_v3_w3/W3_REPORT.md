---
title: "W3 results — is FDC1's advantage structural, does it last, and what do the models cost when executed"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
---

**English** | [Magyar](W3_REPORT.hu.md)

# W3 results — is FDC1's advantage structural, does it last, and what do the models cost when executed

**Status:** final for W3. All **15 training runs** of the [sealed preregistration](PREREGISTRATION.md) finished with the requested steps and finite results; none failed. Training code unchanged from W2 (`train_code_sha256` = `cd909ea282660f0f`), so the W2 runs (B_moe, B_fixed, FDC1, AD02a-3, AD03b, B_dense) are compared directly. Every run directory (checkpoint included) is on the private backup server, SHA-256 verified per run; reports, logs and analysis outputs are in this directory. Test split and holdout were not used. Machines: five RunPod Community RTX 3090s; `w3a` only measured (it never trained), `w3b`–`w3e` trained.
**Rule:** as preregistered — Δ = variant − reference (validation NLL, nat/token, 3 paired seeds); `SUPPORTED` = Δ ≤ −0.010, 95% t-CI below 0, 3/3 seeds; `NOT_SUPPORTED_IN_SCOPE` = the mirror image; otherwise `INCONCLUSIVE` (not evidence of no effect). "Equal FLOP" = routed FLOP. Decision table: `analysis_decision_table.txt` (produced by `tools/analysis/w3_table.py` from the backed-up reports).

## 1. The primary question: FDC1 against the equal-size MoE

| Config | val NLL | seeds 42 / 43 / 44 | params | routed MFLOP | executed MFLOP (bmm) |
|---|---|---|---|---|---|
| B_moe (W2) | 2.1180 | 2.1120 / 2.1242 / 2.1177 | 17.15M | 22.83 | 24.87 |
| **MoE12** (`moe_f5 n_experts=12`) | **2.0680** | 2.0766 / 2.0572 / 2.0701 | 31.37M | 22.87 | 28.24 |
| **FDC1** (W2; 3 banks + flat routing) | 2.0870 | 2.0886 / 2.0842 / 2.0881 | 31.36M | 22.93 | 34.23 |

**FDC1 − MoE12 = +0.0190 nat**, 95% CI [+0.0003; +0.0377], all three seeds positive (+0.012 / +0.027 / +0.018) → **`NOT_SUPPORTED_IN_SCOPE`: FDC1 is worse than the MoE with the same parameter count and the same routed FLOP.** Its W2 advantage over the 17M MoE came from its parameters, not from its structure. The MoE itself gains 0.050 nat from the same parameter increase (MoE12 − B_moe, descriptive).

**W3b condition** (preregistered for the later ~300M experiment: mean FDC1 − MoE12 ≤ +0.010): +0.019 → **not met** at 9,000 steps (but see section 3).

## 2. Single factors (2×2: bank count × routing)

| Pair | What changes | Δ | 95% CI | Verdict |
|---|---|---|---|---|
| FDC1 − B3H | flat instead of hierarchical routing, 3 banks (equal params) | −0.0462 | [−0.0682; −0.0242] | **SUPPORTED** (flat better) |
| B1F − B_fixed | flat instead of hierarchical routing, 1 bank (equal params) | −0.0336 | [−0.0458; −0.0215] | **SUPPORTED** (flat better) |
| B3H − B_fixed | 3 banks instead of 1, hierarchical (1.83× params) | −0.0267 | [−0.0475; −0.0059] | **SUPPORTED** (3 banks better) |
| FDC1 − B1F | 3 banks instead of 1, flat (1.83× params) | −0.0393 | [−0.0546; −0.0240] | **SUPPORTED** (3 banks better) |
| Interaction (FDC1 − B3H) − (B1F − B_fixed) | do the two gains add up? | −0.0126 | [−0.0445; +0.0193] | descriptive; the CI contains 0 |

B3H 2.1332 (31.37M), B1F 2.1263 (17.15M). Both W1 findings reproduce under this code: flat routing is better than hierarchical routing at equal parameters (by 0.034–0.046 nat), and separate banks help (with more parameters). The data are compatible with the two gains adding up (the interaction CI contains 0; it is not shown to be zero). None of the AM variants reaches the equal-size MoE.

## 3. Longer training (18,000 steps)

| Config | val NLL | seeds | Δ vs its 9k run |
|---|---|---|---|
| FDC1_18k | 2.0510 | 2.0521 / 2.0600 / 2.0410 | −0.0359 (SUPPORTED) |
| MoE12_18k | 2.0499 | 2.0482 / 2.0529 / 2.0485 | −0.0181 (INCONCLUSIVE) |

**FDC1_18k − MoE12_18k = +0.0012**, 95% CI [−0.0179; +0.0203] → **`INCONCLUSIVE`**: after 18k steps the gap is no longer detectable (it was +0.019 at 9k), but equivalence is not shown either. Validation curves (mean of seeds):

| step | 2k | 6k | 9k | 12k | 15k | 16k | 17k |
|---|---|---|---|---|---|---|---|
| FDC1_18k | 2.854 | 2.194 | 2.107 | 2.066 | 2.052 | 2.052 | 2.052 |
| MoE12_18k | 2.831 | 2.174 | 2.089 | 2.059 | 2.048 | 2.050 | 2.050 |

Both curves flatten around 15k steps and MoE12's rises slightly afterwards — with 53M training tokens, 18k steps see the data ~5.6 times, so the late part of the curve is in the repetition regime. FDC1 improved more between 9k and 18k than MoE12 (−0.036 vs −0.018); whether this is a real late advantage of FDC1 or only the MoE saturating earlier on repeated data cannot be separated on this corpus.

## 4. Measurement (one RTX 3090, `w3a`, bf16, Triton expert dispatch; 21 checkpoints)

Means of 3 seeds; `analysis_measurement_table.txt`, raw: `measure/w3a_rtx3090.json`.

| Config | executed MFLOP padding-free / bmm | decode tok/s b1 / b8 / b32 | TTFT b1 | weights bf16 | LRU hit, 25% / 50% of experts cached | greedy loop share | sampled loop share |
|---|---|---|---|---|---|---|---|
| B_moe | 22.83 / 24.93 | 875 / 6,532 / 25,750 | 9.4 ms | 32.7 MB | 0.12 / 0.47 | 0.82 | 0.32 |
| MoE12 | 22.87 / 28.80 | 780 / 6,021 / 23,025 | 9.4 ms | 59.8 MB | 0.29 / 0.60 | 0.82 | 0.17 |
| B_dense | 22.81 / 22.81 | 2,077 / 14,972 / 53,017 | 3.7 ms | 25.9 MB | – | 0.80 | 0.41 |
| B_fixed | 22.86 / 31.82 | 653 / 4,881 / 19,253 | 10.1 ms | 32.7 MB | 0.52 / 0.74 | 0.82 | 0.34 |
| FDC1 | 22.93 / 35.71 | 634 / 4,992 / 19,523 | 10.1 ms | 59.8 MB | 0.56 / 0.76 | 0.77 | 0.24 |
| AD02a-3 | 19.71 / 27.61 | 625 / 4,700 / 18,288 | 10.9 ms | 32.7 MB | 0.58 / 0.80 | 0.54 | 0.06 |
| AD03b | 18.30 / 24.96 | 87 / 661 / 2,612 (eager) | 11.8 ms | 32.7 MB | 0.60 / 0.82 | 0.58 | 0.13 |

- **A success gate (A1):** AD02a-3 executes 0.86× and AD03b 0.80× of B_moe's padding-free FLOP — the compute condition is met — but their hard-path NLL is 1.18–1.23× MoE: **0/3 seeds, `NOT_SUPPORTED_IN_SCOPE`** for both. FDC1: its padding-free executed FLOP is 1.002× MoE12's, but the quality pair fails, so no combined claim is made. On the padding-free path the executed overhead of FDC1/B_fixed disappears (the bmm overhead was a dispatch property, as expected).
- **NLL by position** (last 64-token bucket of the window): FDC1 − MoE12 +0.028 and FDC1 − B_moe −0.022, both INCONCLUSIVE.
- **Generation (validation prompts, 384 new tokens; descriptive):** all 17–31M models loop very often under greedy decoding (77–82%; the halting models 54–58%, for reasons not analysed here). With sampling, the larger models loop less (MoE12 0.17, FDC1 0.24 vs B_moe 0.32 and Dense 0.41); FDC1 − MoE12 differs in sign between seeds. This is the first direct measurement of the looping the owner observed in larger MoE fine-tunes; at this size it says mainly that more parameters reduce sampled looping.
- **Hotset (B, descriptive):** with a quarter of the experts cached, the AM models hit 52–60% (bank reuse across hops makes their working set smaller), the MoEs 12–29%. This is a simulation, not a measured offload.

## 5. Goal C exploration (not preregistered): how fast can decoding get?

`c_decode/w3a_rtx3090_v2.json` (`tools/analysis/c_decode_bench.py`, seed 42, RTX 3090, prompt 128, teacher-forced validation text). The profile (`c_decode/profile_w3a_console.txt`) showed that the Triton expert kernel converted the whole expert stack to bf16 on every call (~150 copy kernels per decode step). Variants applied identically to every model: `bf16` (weights cast once), `gather` (gathered-expert banks, no sort/count/scatter, adaptive K capturable; `amllm.fastdecode`, fp64-equivalence tested), `compile` (gather + `torch.compile`, replayed as a CUDA Graph).

| Model | base b1 tok/s | best b1 (variant) | speed-up | b32 best (bf16) | val NLL base → best |
|---|---|---|---|---|---|
| Dense | 1,946 | 3,045 (bf16) | 1.6× | 72,191 | 2.1593 → 2.1600 |
| MoE | 911 | 2,293 (compile) | 2.5× | 32,698 | 2.1554 → 2.1553 |
| MoE12 | 825 | 2,158 (compile) | 2.6× | 31,511 | 2.0947 → 2.0947 |
| B_fixed | 673 | 2,013 (compile) | 3.0× | 27,655 | 2.1900 → 2.1876 |
| FDC1 | 668 | 2,179 (compile) | 3.3× | 27,882 | 2.0948 → 2.0945 |
| AD03b | 88 (eager) | 1,989 (compile) | ~23× | 17,374 (compile) | 2.5219 → 2.5222 |

Quality is preserved: validation NLL changes by ≤ 0.003 and the timed path agrees with the same variant run eagerly on 96–100% of teacher-forced steps (`compile` shows small numeric differences). At batch 1 the AM–MoE gap shrinks from FDC1 at 0.73× to 0.95× of MoE's speed, and adaptive K becomes usable (from 88 to ~2,000 tokens/s) because the gathered path is CUDA-Graph capturable. At batch 32 only bf16 helps (~30%), gather/compile slow down, and the MoE stays faster. Limits: exploratory, one seed, one GPU type, small models where kernel-launch overhead dominates. Two earlier runs of this benchmark are kept as `partial_v1_*` (stopped, before the quality checks were added) and `partial_v2_compile_fallback_*` (the compiler silently fell back to eager after a few models); only `_v2` is valid.

## 6. What W3 does not establish
- No A evidence: the A gate fails for both AD candidates (quality), and FDC1 loses the matched quality pair.
- One scale (17–31M), one corpus (53M training tokens), n = 3; INCONCLUSIVE rows are not "no effect".
- The 18k comparison is in the data-repetition regime.
- Goal C numbers are exploratory, not a preregistered decode result.

## 7. Consequences (proposal)
- **FDC1 is not the structural winner the W2 result suggested.** At equal size the plain MoE is better at 9k steps; at 18k the two are not distinguishable. The ~300M FDC1 experiment's preregistered condition is not met; the owner decides whether the 18k result changes that (and the data are needed anyway).
- **What stands from V3:** flat routing beats hierarchical routing; separate (per-hop or partially shared) banks help; per-hop attention is essential; the anytime loss repairs the hard halting path; adaptive K saves compute, and its quality difference to full K was INCONCLUSIVE (W2; not shown to be zero). None of these makes an AM variant beat the equal-size MoE at this scale.
- **Goal C:** most of the decode-speed gap was an execution problem; with the gathered path the AM models decode at ~95% of MoE speed at batch 1, and adaptive depth becomes practical. A proper preregistered decode measurement (several seeds, a second GPU type) would turn this into evidence.

## Files
- `*/<preset>_s4?/`: `config.json`, `env.json`, `report.json`, `train.log` per run (checkpoints on the backup server only).
- `analysis_decision_table.txt`, `analysis_measurement_table.txt`; `measure/w3a_rtx3090.json`; `c_decode/` (decode benchmark, profile, manifest).
- `queues/`, `_queue_w3*/`: executed queues and worker logs.
