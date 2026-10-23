---
title: "W2 preregistration"
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

# W2 preregistration

**Status:** APPROVED AND SEALED — approved by the owner on 2026-10-23 ("after the fix W2 can start on the pods"). From now on the gate numbers cannot change (only by a dated amendment); the content of the file is recorded in `PREREGISTRATION.sha256`. The anchors and the runs of parts 2–4 start after sealing. In case of a difference between the two languages the Hungarian text governs.

**Code version (frozen during W2):** `train_code_sha256` = `cd909ea282660f0f` (recomputed after the W2 code changes; the diagnostic fixes — `profile.py`, `router_analysis.py`, `tools/analysis/` — are not part of the hash). **No comparison with W1 runs** (different hash): every W2 comparison is within W2, which is why the anchors run again.
**Data / protocol:** as in W1 — `code_corpus_v2`, 9,000 steps, batch 32 × 512, lr 1e-3, seeds 42/43/44, validation NLL (nat/token) in the default mode (`fixed`; `hard_surv_0.5` for a halting model); the test split and the holdout are not used (`eval_test=false`). A run is done only if its step count is the requested one and every result is finite. FLOP as two numbers: `flops_per_token` (routed) and `flops_per_token_executed` (with bmm padding); the parameter count for every run.
**Decision rule (every paired comparison, as in W1, directional):** Δ = variant − reference; `SUPPORTED` (better) = Δ ≤ −0.010 **and** the upper bound of the 95% t-CI < 0 **and** 3/3 seeds negative; `NOT_SUPPORTED_IN_SCOPE` (worse) = the mirror image; otherwise `INCONCLUSIVE` (not equivalence).

## 1. Anchors (new hash)

| ID | Preset / setting | Purpose |
|---|---|---|
| B_moe | `moe_f5` | MoE reference |
| B_dense | `dense_f5` | dense reference |
| B_fixed | `amllm_f5 halt_rule=none` | FD reference |
| A0 | `amllm_f5` (survival, λ 0.02) | AD reference |

Check: the B_moe and B_fixed means differ from W1's by at most ±0.03 nat (the code change is bitwise neutral by default; this is the GPU-independent reproduction). 4 × 3 = **12 runs**.

## 2. FD04 — is separate attention per hop needed (H02)

B_fixed uses separate attention per hop (`unshared_attention=true`). Reference: B_fixed. Every row uses fewer parameters and (FD04d) fewer FLOP; the question is how much quality is lost.

| ID | `--set` (all `halt_rule=none`) | FLOP× / params× B_moe | Question |
|---|---|---|---|
| FD04a | `unshared_attention=false` | 1.001 / 0.827 | one shared attention for all hops |
| FD04b | `unshared_attention=false attn_groups=3` | 1.001 / 0.896 | 3 attention weight sets (hops in pairs) |
| FD04c | `unshared_attention=false attn_adapter_rank=32` | 1.012 / 0.834 | shared attention + per-hop r=32 adapter |
| FD04d | `unshared_attention=false attn_every=2 attn_groups=3` | 0.794 / 0.896 | every second hop expert-only, own weights per attention hop |

Interpretation: `SUPPORTED` → the shared / sparser variant is better; `NOT_SUPPORTED_IN_SCOPE` → separate per-hop attention matters; `INCONCLUSIVE` → no difference detectable at this resolution, which is **not** equivalence (the CI is wide with n = 3). FD04d cuts FLOP to ~0.79×: its quality and FLOP ratios to B_moe are reported descriptively. 4 × 3 = **12 runs**.

## 3. FD05 — matched quality–compute points (H03/H04/H05) and the W1 combination

The matching comes from real forward passes of `tools/fd05_match.py` (routed FLOP × B_moe / params × B_moe). **Every "equal / matched FLOP" in this document refers to the routed work**; executed FLOP (bmm padding) differs between variants, is therefore reported separately for every comparison, and no verdict claims equal executed work.

| ID | Preset, `--set` | FLOP× / params× | Reference | Question |
|---|---|---|---|---|
| FD05a | `amllm_f5 halt_rule=none n_hops=8 expert_hidden=160 experts_per_module=8` | 1.003 / 1.001 | B_fixed | H05: 8 hops at equal routed FLOP and parameters |
| FD05b | `amllm_f5 halt_rule=none n_hops=4 expert_hidden=832 experts_per_module=2` | 1.000 / 0.965 | B_fixed | H05: 4 hops (no expert choice inside its modules) |
| FD05c | `amllm_f5 halt_rule=none module_topk=2 expert_topk=1` | 1.002 / 1.000 | B_fixed | H03: top-2 module / top-1 expert |
| FD05d | `amllm_f5 halt_rule=none module_topk=2 expert_hidden=192` | 1.002 / 0.793 | B_fixed | H03: top-2/top-2 (parameters cannot be matched) |
| FD05e | `amllm_f5 halt_rule=none experts_per_module=8 expert_hidden=192` | 0.847 / 1.001 | **FD05e_moe**, B_fixed | H04: finer experts at 0.85× |
| FD05e_moe | `moe_f5 n_experts=8 expert_hidden=192` | 0.846 / 1.001 | — | matched MoE control |
| FD05f | `amllm_f5 halt_rule=none expert_hidden=208` | 0.859 / 0.810 | **FD05f_moe**, B_fixed | H04: smaller experts at 0.86× |
| FD05f_moe | `moe_f5 expert_hidden=208` | 0.858 / 0.811 | — | matched MoE control |
| FD05f_dense | `dense_f5 ffn_hidden=416` | 0.857 / 0.698 | — | matched dense control |
| FDC1 | `amllm_f5 halt_rule=none bank_groups=3 n_modules=1 experts_per_module=24 module_topk=1` | 1.004 / 1.828 | B_fixed, B_moe (descriptive) | W1 combination: partial bank sharing + flat routing |

- FD05e − FD05e_moe and FD05f − FD05f_moe are the **A-relevant** pairs (equal routed FLOP, equal parameters), with the same decision rule. Executed FLOP is reported alongside; this does not give an A success either (by A1, executed work needs the W3 measurement).
- FDC1 has 1.83× the parameters (separate banks); stated on every row, not an equal-parameter comparison.
- 10 × 3 = **30 runs**.

## 4. AD — anytime weight and adaptive K (AD03)

All `amllm_f5`, survival halting, λ 0.02. Reference: A0 (W2 anchor).

| ID | `--set` | Question |
|---|---|---|
| AD02a-3 (directory `AD02a_3`) | `anytime_weight=0.5` | intermediate weight: hard path vs. final quality |
| AD03a | `anytime_weight=0.5 expert_topp=0.5` | adaptive K (the second expert only if the first has probability < 0.5) |
| AD03b | `anytime_weight=0.5 expert_topp=0.7` | adaptive K, milder |
| AD03c | `anytime_weight=0.5 expert_topk=1` | fixed K = 1 control (the lower bound of adaptive K) |

- Primary gate (A1 screening signal, unchanged): hard θ 0.5 val NLL ≤ 1.03 × B_moe **and** routed FLOP ≤ 0.90 × B_moe, on 3/3 seeds → `PROMISING_EXPLORATORY`; only the W3 execution measurement can give success.
- Paired comparisons: AD02a-3 − A0; AD03a/b − AD02a-3 (what adaptive K gives at the same weight); AD03a/b − AD03c **reporting FLOP as well** (adaptive K is worth it if its mean expert count and FLOP are close to AD03c and its quality close to AD02a-3).
- Secondary: the AD01 diagnostics (per-hop read-out, hard–soft gap) for all four variants and A0. The control for the learned exit is a **real random stop** (`random_exit_matched_nll_hard`: a hard run in which tokens stop according to a shuffle of the hard path's exit hops); the read-out from the soft trace (`…_from_soft_trace`) is only an approximation and no conclusion rests on it.
- Dropped routes of adaptive K appear as −1 in the trace; the router diagnostics skip them, and the MoE active-parameter count uses the routes actually dispatched.
- 4 × 3 = **12 runs**.

## 5. Scope, cost, stopping
- 12 anchors + 12 FD04 + 30 FD05 + 12 AD = **66 runs**; queues balanced on the per-pod run times measured in W1 (`queues/w2a.txt` 27, `w2b.txt` 27, `w2c.txt` 12 runs; the slower A40 gets fewer): estimated wall time **~7.5–8 h**, ~**$5.5** ($0.68/h for the three pods).
- The machines do not stop during W2; the queue is pod-side (`worker.sh`) and can be rebalanced while running. The slots are named w2a/w2b/w2c (separate pod-log directories). Every finished run is backed up to vps2 within ~2 minutes with its checkpoint, verified by SHA-256.
- Closure: evaluation (decision table, AD01 diagnostics) on the pods, then the final backup; the owner decides what happens to the machines.

## 6. Not part of W2
Holdout, test data, 18k runs, OOD, decode/VRAM measurement, padding-free execution measurement (W3), AD04 (routing × exit), any code change while the runs are going.
