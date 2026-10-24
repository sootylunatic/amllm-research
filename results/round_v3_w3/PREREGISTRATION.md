---
title: "W3 preregistration"
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

# W3 preregistration

**Status:** APPROVED AND SEALED — approved by the owner on 2026-10-24 ("approved, go"). From now on only a dated amendment can change it; the content is recorded in `PREREGISTRATION.sha256`. Training and measurement start after sealing. In case of a difference between the two languages the Hungarian text governs.

**Code version:** the training code is **unchanged** from W2: `train_code_sha256` = `cd909ea282660f0f`. W3 only extends the non-hashed execution code (`fastpath.py`, `generate.py` — multi-bank, per-hop-norm and adaptive-K models on the KV-cache and the static/bucketed hard paths, with fp64/fp32 equivalence tests; new measurement module `w3measure.py`). **The W2 runs (B_moe, B_dense, B_fixed, FDC1, AD02a-3, AD03b) are therefore directly comparable in W3** and are not rerun.
**Data / protocol:** as in W1/W2 — `code_corpus_v2`, batch 32 × 512, lr 1e-3, seeds 42/43/44, validation NLL; the test split and the holdout are not used (the measurement module also loads the validation split only). **Known limitation:** the training split has 53.1M tokens; 9,000 steps repeat it ~2.8×, 18,000 steps ~5.6×.
**Decision rule (every paired comparison, as in W1/W2, directional):** Δ = variant − reference; `SUPPORTED` (better) = Δ ≤ −0.010 **and** the upper bound of the 95% t-CI < 0 **and** 3/3 seeds negative; `NOT_SUPPORTED_IN_SCOPE` (worse) = the mirror image; otherwise `INCONCLUSIVE` (not equivalence). "Equal FLOP" = routed FLOP; executed FLOP is reported separately.

## 1. Training: is FDC1's advantage structural (9,000 steps, 3 seeds)

| ID | Preset, `--set` | Params | Routed FLOP× B_moe | Role |
|---|---|---|---|---|
| MoE12 | `moe_f5 n_experts=12` | 31.37M | 1.002 | **FDC1's parameter- and routed-FLOP-matched MoE control** (72 experts, 2 per layer per token) |
| B3H | `amllm_f5 halt_rule=none bank_groups=3` | 31.37M | 1.001 | 3 banks, hierarchical routing (only the bank count changes from B_fixed) |
| B1F | `amllm_f5 halt_rule=none n_modules=1 experts_per_module=24 module_topk=1` | 17.15M | 1.004 | 1 bank, flat routing (only the routing changes from B_fixed) |

From W2: B_fixed (1 bank, hierarchical, 17.15M), FDC1 (3 banks, flat, 31.36M), B_moe. This is a 2×2 (bank count × routing) plus the equal-size MoE.

**Primary question:** FDC1 − MoE12 (equal parameters, equal routed FLOP).
- `SUPPORTED` → at this scale FDC1's advantage is not only its parameter count.
- **The W3b condition (for the later ~300M experiment, which the owner decides separately):** FDC1 "holds" if the mean FDC1 − MoE12 Δ ≤ +0.010 (at most minimally worse). This is **not** A evidence, only the condition for going further; the 300M run is not part of W3.

**Secondary (same rule):** FDC1 − B3H (flat routing with 3 banks, equal params); B1F − B_fixed (flat routing with 1 bank, equal params); B3H − B_fixed and FDC1 − B1F (bank count; 1.83× params, stated on every row). Interaction, descriptive: (FDC1 − B3H) − (B1F − B_fixed), paired per seed, with CI.

9 runs.

## 2. Training: does the advantage last with longer training (18,000 steps, 3 seeds)

| ID | Preset, `--set` | Steps |
|---|---|---|
| FDC1_18k | `amllm_f5 halt_rule=none bank_groups=3 n_modules=1 experts_per_module=24 module_topk=1` | 18,000 |
| MoE12_18k | `moe_f5 n_experts=12` | 18,000 |

Question: FDC1_18k − MoE12_18k (same rule); descriptively the change of the difference between 9k and 18k. The 18k runs see the data ~5.6 times, so overfitting (a rising validation NLL) is a possible outcome; the full curve is reported. 6 runs.

## 3. Measurement (the plan's "practical measurement" wave)

**Models:** B_moe, B_dense, B_fixed, FDC1, AD02a-3, AD03b (W2) and MoE12 (W3, 9k) — 7 configurations × 3 seeds = 21 checkpoints. **On one machine** (the RTX 3090 of slot `w3a`, which does not train, so nothing disturbs the measurement), in one environment, with one backend: bf16, Triton grouped dispatch for everything except Dense (routed rows only); the measurement module never mixes environments in one file. Measurements (`amllm.w3measure`):

| Measurement | Content | Goal |
|---|---|---|
| Executed work | routed and executed FLOP/token, computed expert rows per dispatch (bmm vs. padding-free loop/Triton); for halting models the bucketed hard path's padding and the attention query fraction | **A** |
| Latency | full-sequence forward (8 × 512): eager and CUDA Graph where capturable | **C** |
| Decode | KV-cache, batch 1/8/32: prefill (TTFT), ms/token p50/p95, tokens/s (eager and CUDA Graph) | **C** |
| VRAM (K5) | weights fp32/bf16, KV cache for batch 1/2/4/8 × context 128/256/512, measured peak memory | **B** |
| Expert hotset (K6) | usage concentration per bank/layer, LRU cache simulation (25/50/75% of experts), bytes/token | **B** |
| NLL by position | 64-token buckets in the 512 window | quality over length |
| Generation | 32 validation prompts × 128 tokens, 384 new tokens, greedy and sampled (T 0.8, top-p 0.95): repeated 4-gram rate, distinct-1/2, share of sequences ending in a loop | looping |

**Gates and comparisons:**
- **A success gate (per A1, measurable for the first time):** an AD candidate (AD02a-3, AD03b) is `SUPPORTED_IN_SCOPE` if its padding-free executed FLOP/token ≤ 0.90 × B_moe's (both the `flops_per_token_executed` measured on the Triton-dispatch reference path: only the expert rows really computed; on this path attention runs for every query, because halted tokens remain keys/values — the conservative number; the bucketed path's packing ratio is reported alongside) **and** its hard θ 0.5 val NLL ≤ 1.03 × B_moe, on 3/3 seeds; otherwise `NOT_SUPPORTED_IN_SCOPE`. For FDC1: "better at equal parameters and equal executed compute" may only be said if FDC1 − MoE12 is `SUPPORTED` (part 1) **and** FDC1's padding-free executed FLOP ≤ 1.00 × MoE12's (on the same Triton path); otherwise both numbers are reported separately, without a combined claim.
- **C:** decode tokens/s, TTFT, p50/p95 relative to B_moe, same GPU and backend — descriptive, no gate.
- **B:** VRAM breakdown and hotset — descriptive.
- **NLL by position:** FDC1 − MoE12 and FDC1 − B_moe in the last bucket (448–512) with the decision rule (secondary); the other buckets descriptive.
- **Generation:** loop and repetition metrics of FDC1, MoE12, B_moe, B_dense, with paired per-seed differences and CIs — descriptive, no gate.

## 4. Scope, cost, stopping
- Five RTX 3090s (Community, $0.22/h): `w3b`–`w3e` train (9 + 6 runs, queues balanced on the run times measured in W1/W2: `queues/w3b.txt` 5, `w3c.txt` 4, `w3d.txt` 3, `w3e.txt` 3 runs, ~2–2.3 h wall time); `w3a` only measures: the 18 W2 checkpoints at once, MoE12's 3 checkpoints once they exist (~1–1.5 h). About $2.5–3 in total.
- The queue is pod-side and can be rebalanced while running; every finished run is backed up to the backup server within ~2 minutes with its checkpoint, SHA-256 verified. The W2 and MoE12 checkpoints reach the measuring machine from the backup server (the Mac's mobile link is not used).
- Closure: decision table, measurement tables, final backup, report; the owner decides about the machines.

## 5. Not part of W3
The ~300M FDC1 experiment (W3b; the data corpus is not ready, the owner decides separately), holdout and OOD (K7 — with W4), training-code changes, training longer than part 2.
