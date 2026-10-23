---
title: "W1 preregistration (draft, awaiting owner approval)"
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

# W1 preregistration (draft, awaiting owner approval)

**Status:** APPROVED AND SEALED — approved by the owner on 2026-10-23 (with the corrected decision rules and the code version as stated below). From now on the gate numbers cannot be changed; the content of the file is recorded in `PREREGISTRATION.sha256`. The anchor runs (part 1) are controls and started before the approval; the runs of parts 2–3 started after it.

**Code version (frozen during W1):** `train_code_sha256` = `ba81b4fc02cd7b63` (`config.py`, `data.py`, `model.py`, `routing.py`, `train.py`); the training code must not change from this. No comparison with runs under the earlier hashes 406c4eef88f4ba69, e361ca5da3bf54ed and 83339a48c73ce5c0 (R2). A run is done only if its actual step count equals the requested one and every result is finite (NaN training stops with an error, leaves a FAILED.json marker and writes no report.json). (The e361… version trained nothing: after review the test-split evaluation and the FLOP report were corrected.)
**Data/protocol:** `code_corpus_v2` (split id `355afd73091925ef`), 9,000 steps, batch 32 × 512, lr 1e-3, the earlier F5 protocol; seeds 42, 43, 44; validation for decisions; **the test split and the holdout are not used: a run neither evaluates nor logs the test split (`eval_test=false`, `test_default_mode: null` in `report.json`; tested)**. Every run records its GPU; GPU type does not affect training loss (see RESEARCH_TASKS), timing is not used here.
**Primary metric:** validation NLL (nat/token) in the default mode (`fixed`; `hard_surv_0.5` for a halting model), paired (same-seed) difference, 95% t-CI (n = 3). With n = 3 the CI is wide: whatever contains 0 is **INCONCLUSIVE**, not equivalence. FLOP is reported as two numbers: `flops_per_token` (the routed work) and `flops_per_token_executed` (what was really executed, including the padding rows of the `bmm` dispatch; training and evaluation use `bmm`). The parameter count is reported for every run.

## 1. Anchors and controls (start at once)

| ID | Preset / setting | Seeds |
|---|---|---|
| B_moe | `moe_f5` | 42, 43, 44 |
| B_dense | `dense_f5` | 42, 43, 44 |
| B_fixed | `amllm_f5`, `halt_rule=none` (FixedDepth base) | 42, 43, 44 |
| A0 | `amllm_f5` (current survival halting, λ 0.02) — reproduction of L1 on the new code | 42, 43, 44 |

Purpose: anchors under the new hash and the A0 control. No gate; check: the B_moe and B_fixed means differ from the earlier 2.1166 / 2.1625 by at most ±0.03 nat (a sanity check of code and GPU independence; on a larger deviation the runs are not used and a cause is sought).

## 2. FD lane (FixedDepth; every variant is `amllm_f5`, `halt_rule=none` unless stated)

Each question changes **one** factor; the reference is B_fixed (shared bank, 6×4, top-1/top-2, own per-hop attention, shared bank norm).

| ID | Variant (`--set`) | Total parameters | MFLOP/token | Question |
|---|---|---|---|---|
| FD01a | `bank_groups=6` | 52.69M | 22.857 | equal active FLOP, 3× parameters: does a separate bank help? |
| FD01b | `bank_groups=6 n_modules=2 experts_per_module=2` | 17.15M | 22.830 | **equal total parameters** (24 experts), separate bank per hop |
| FD01c | `bank_groups=3` | 31.37M | 22.857 | partial sharing |
| FD02a | `n_modules=1 experts_per_module=24 module_topk=1` | 17.15M | 22.926 | flat 24-way router against the hierarchical one |
| FD02b | `router_mode=random` | 17.15M | 22.857 | random routing (training control) |
| FD03a | `bank_norm=per_hop` | 17.15M | 22.857 | per-hop bank LayerNorm |
| FD03b | `bank_norm=hop_affine` | 17.15M | 22.857 | shared LN + per-hop scale/shift |

**Decision rule (per question, paired Δ = variant − B_fixed, nat/token):**
The direction is fixed (a smaller loss is better):
- `SUPPORTED` (the variant is **better**): Δ ≤ −0.010 (≈ one seed SD) **and** the **upper** bound of the 95% CI is **< 0** (the CI lies entirely below zero) **and** the Δ of all three seeds is negative;
- `NOT_SUPPORTED_IN_SCOPE` (the variant is **worse**): Δ ≥ +0.010 **and** the **lower** bound of the 95% CI is **> 0** **and** the Δ of all three seeds is positive;
- otherwise `INCONCLUSIVE`.
A significantly worse variant therefore can never be `SUPPORTED`.
FD01a does not decide alone because parameters are not matched: FD01b is the equal-parameter verdict. FD02b (random) measures only the value of the router; it is not a competing candidate. "Improved" is not an A success: A needs an executed-FLOP-matched comparison (FD05, W2).

## 3. AD lane (AdaptiveDepth; `amllm_f5` survival halting, λ 0.02)

| ID | Variant (`--set`) | Reference |
|---|---|---|
| AD02a-1 | `anytime_weight=0.3` | A0 |
| AD02a-2 | `anytime_weight=1.0` | A0 |

**Decision rule:**
- Primary (towards goal A): in **hard_surv_0.5** mode val NLL ≤ 1.03 × the B_moe mean **and** the routed `flops_per_token` ≤ 0.90 × B_moe's (≥ 10% saving), on 3/3 seeds. The `flops_per_token_executed` (with bmm padding) **does not decide** but is reported with the padding ratio, because bmm padding is a property of the dispatch, not of the architecture; the saving counts as real only if a padding-free execution path exists (F3 result, to be verified in W3) — until then the verdict is at most `SUPPORTED_IN_SCOPE (routed FLOP)`; this is `SUPPORTED_IN_SCOPE` for the AD branch, not full A evidence.
- Secondary (diagnostic): the AD01 metrics re-run (per-hop read-out, hard–soft gap, threshold curve); an improvement if the read-out NLL of hop 5 gets closer to the final one and the θ = 0.3 / 0.5 points stay closer to the soft value than for A0 (paired, 3 seeds).
- Otherwise `INCONCLUSIVE` / `NOT_SUPPORTED_IN_SCOPE`. The AD result draws no conclusion for FixedDepth and vice versa.

## 4. Scope, cost, stopping
- 12 anchor + 21 FD + 6 AD runs = **39 runs** (≈ 11–12 GPU-hours on a 3090, ≈ $3–4 at the measured prices).
- The machines do not stop during W1; the queue is the pod-side worker's file (`tools/runpod/worker.sh`), to which jobs can be appended while it runs. Cost is reserved time × hourly rate; W1 stays within the proposed $20 round cap.
- Closure: every result saved to vps2 and the Mac, SHA verified; further questions (FD04, FD05, AD03) in W2.

## 5. Not part of W1
Holdout, test data, 18k runs, FD04/FD05, AD03/AD04, decode/VRAM measurement, any code change (the training code is frozen during W1).
