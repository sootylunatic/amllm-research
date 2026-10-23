---
title: "W0 (CPU) first results — AD01 trace, K1 curve analysis, K2 route trace"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
---

**English** | [Magyar](W0_REPORT.hu.md)

# W0 (CPU) first results — AD01 trace, K1 curve analysis, K2 route trace

**Platform:** local Mac (arm64, M4), torch 2.14.1, fp32, CPU. Data: the **val** split of `code_corpus_v2`, 128 evenly spaced windows × 512 tokens (65,536 tokens; the full evaluation uses 512 windows, so absolute values may differ from the reported val values by ≈ 0.01 nat). No test data used. The main tables are **seed 42**; AD01 is also repeated on three seeds (below): these are diagnostics, not confirmatory results. Tools: `tools/analysis/ad01_trace.py`, `k1_curves.py`, `k2_route_trace.py`; raw output: `AD01/`, `K1/`, `K2/`. Checkpoints verified (SHA-256 equals vps2).

## AD01 — where quality breaks in the halting models

**Correctness:** the tracer re-runs the model's actual hop loop and compares the final logits with `model.forward` output in soft and hard mode: **the largest difference is 0.0 on all 5 checkpoints**. The output is therefore not caused by a hidden mismatch; the soft/hard gap belongs to the method, not to an implementation bug.

**Per-hop read-out** (the state after a hop read through the final norm and head, nat/token):

| Checkpoint | hop 1 | hop 2 | hop 3 | hop 4 | hop 5 | hop 6 (final) | oracle: best hop per token\* |
|---|---|---|---|---|---|---|---|
| `H01_fixed` (no halting) | 6.24 | 5.78 | 5.20 | 4.20 | 3.57 | **2.14** | 1.86 |
| F5 base (λ 0.02) | 6.04 | 4.60 | 4.00 | 3.34 | 2.45 | **2.17** | 1.85 |
| `H01_bias_m2` | 6.09 | 4.59 | 3.95 | 3.24 | 2.45 | **2.18** | 1.86 |
| `H01_p005` | 6.32 | 5.65 | 4.72 | 4.29 | 2.71 | **2.15** | 1.86 |
| `H01_p05` | 6.23 | 4.48 | 3.43 | 2.63 | 2.36 | **2.23** | 1.91 |

\* The oracle picks, per token, the hop with the smallest NLL knowing the target byte: **an analytic bound, not a deployable policy**, and taking a minimum is optimistic in itself.

**Hard exit** (threshold → val NLL @ mean hops; "random" = the same hop distribution assigned to random tokens from the soft trace):

| Checkpoint | θ 0.1 | θ 0.3 | θ 0.5 | θ 0.8 |
|---|---|---|---|---|
| F5 base | 2.174 @ 5.91 (random 2.21) | 2.291 @ 5.16 (2.59) | 3.383 @ 3.47 (3.68) | 6.04 @ 1.0 |
| `H01_bias_m2` | 2.176 @ 5.91 | 2.290 @ 5.15 | 3.274 @ 3.58 (3.57) | 6.09 @ 1.0 |
| `H01_p005` | 2.153 @ 6.00 | 2.155 @ 5.98 | 2.177 @ 5.90 (2.22) | 3.08 @ 5.06 (2.93) |
| `H01_p05` | 2.255 @ 5.30 (2.37) | 3.487 @ 2.81 (3.76) | 6.21 @ 1.01 | 6.23 @ 1.0 |

**What this shows (for the measured points, 1 seed):**
1. **Intermediate states cannot be read out directly,** not even without halting: the fixed-depth model gives 4.20 NLL at hop 4 and 3.57 at hop 5, then jumps to 2.14 at the end. The hops are not trained to give early outputs. A threshold can therefore stop without the loss jumping only if the model gets its quality at the end anyway — this is the cause of the measured failure. The conclusion supports the "weak intermediate hop" branch of the AD05 decision tree and justifies **the first item of AD02 (anytime/multi-exit loss)**.
2. **The learned stop policy beats random exit at most measured points** (0.03–0.3 nat; exception `H01_p005` θ 0.8: 3.08 against random 2.93), so it carries information, but not enough to keep the loss in check. The "random" control reads from the soft trace, not from the hard path's states, so it is approximate.
3. **Fewer hops only at the cost of quality:** the hard value close to soft (θ 0.1) comes with ≈ 5.9 hops, i.e. hardly any saving. `p05` learns better intermediate states (hop 4: 2.63), but the threshold collapses there too.
4. **The oracle bound (≈ 1.85–1.91) lies below the final 2.14–2.23:** for some tokens an earlier hop is better than the final one. This is only a hypothesis about spare capacity; it is not usable as a policy and proves nothing about deployable gain.

**Repeated on three seeds** (42/43/44, the four AM-LLM rows of the overnight 12 configurations; mean, seed range in brackets; the tracer–`forward` difference is 0.0 everywhere):

| Configuration | hop 1 → 5 read-out (NLL) | final | oracle | θ 0.1 | θ 0.3 | θ 0.5 |
|---|---|---|---|---|---|---|
| `H01_fixed` | 6.29 / 5.78 / 5.21 / 4.15 / 3.43 | 2.149 | 1.87 | – | – | – |
| `H01_p005` | 6.31 / 5.66 / 4.63 / 4.03 / 2.99 | 2.144 | 1.85 | 2.144 @ 6.00 | 2.146 @ 5.98 | 2.171 @ 5.91 (random 2.22) |
| `H01_bias_m2` | 6.19 / 4.66 / 3.96 / 3.06 / 2.54 | 2.165 | 1.86 | 2.165 @ 5.88 | 2.268 @ 5.14 | 3.222 @ 3.66 (3.51) |
| `H01_p05` | 6.40 / 4.45 / 3.36 / 2.77 / 2.40 | 2.255 | 1.94 | 2.277 @ 5.24 | 3.368 @ 2.87 | 6.322 @ 1.06 |

The three seeds show the same as the single-seed rows: the intermediate read-outs are weak (hop 4 of the model without halting 4.15, hop 5 3.43), hard exit stays near the soft value only at ≈ 5.9 hops, and the learned policy beats random exit (θ 0.3–0.5: 0.05–0.3 nat) but the loss cannot be kept with fewer hops.

## AD01 — separate gradient audit (autograd tests and measurement)

**Autograd tests** (`tests/test_ad01_gradient_paths.py`, exact, not approximations):
1. **Hard exit:** the loss of an exited token gives **exactly zero** gradient to the hops after the exit, i.e. the truncated future state receives no LM gradient. The **threshold decision** is not differentiable, but the **survival weights** of the hops that did run are: a token that ran one hop (weight 1) gives the halt head no gradient, a token that ran two hops gives it a **non-zero** LM gradient through the weight of its second hop (two separate tests).
2. **Soft (training) path:** the halt head gets a gradient through the survival weights, and later hops get a gradient from every token, scaled by survival.
3. **Ponder penalty:** `d(ponder)/d(bias) < 0`, gradient descent raises the bias, i.e. pushes towards earlier stopping, even without the LM loss.
4. **Exit loss (AD02a):** the loss of the state after hop *h* reaches the parameters of hops *≤ h*, the shared bank and the shared head, and **exactly zero** for hops *> h*.

**Measurement on real checkpoints** (`tools/analysis/ad01_grad.py`; Mac, train split, 6 × 4 × 512 tokens, soft/train mode, seed 42, one checkpoint per configuration; gradient norm for the LM loss and separately for the aux loss (ponder + balance)):

| Configuration | halt head LM | halt head aux | aux/LM | attention hop 1 → 6 (LM) |
|---|---|---|---|---|
| `H01_p005` (λ 0.005) | 0.24 | 0.02 | 0.09 | 0.23 / 0.48 / 0.36 / 0.35 / 0.30 / 0.25 |
| F5 base (λ 0.02) | 0.79 | 0.37 | 0.46 | 0.35 / 0.43 / 0.29 / 0.22 / 0.25 / 0.18 |
| `H01_bias_m2` (λ 0.02) | 0.76 | 0.37 | 0.48 | 0.38 / 0.46 / 0.29 / 0.23 / 0.24 / 0.17 |
| `H01_p05` (λ 0.05) | 1.29 | 0.80 | 0.62 | 0.48 / 0.41 / 0.26 / 0.23 / 0.14 / 0.12 |

- The aux gradient on the halt head grows with λ, but at the final checkpoints the LM gradient is **larger** (aux/LM < 1): the ponder penalty does not dominate the final state. The collapse at λ = 0.05 is therefore not an aux dominance seen in a snapshot but a matter of the training dynamics (see the K1/AD01 curves).
- The attention gradient of later hops is smaller; at the strongest ponder the gradient of hop 6 is a quarter of hop 1's (0.12 vs 0.48): survival weighting trains later hops less, which supports AD02a (training intermediate hops directly).
- The aux gradient on the attention of later hops is practically zero (≤ 0.0007 at hop 6): the ponder pushes only the halt head and, through survival, the earlier hops.
- Limit: one seed, the final checkpoint, 6 mini-batches; no gradient-norm log during training (the W1 runs do not log it yet; that is a task for the next wave).

## K1 — the cause of the MoE 18k "degradation" (existing curves, F5_2x, seed 42)

| Step | 14,000 | 15,000 | 16,000 | 17,000 | final |
|---|---|---|---|---|---|
| MoE val | 2.046 | 2.056 | **2.254** | 2.253 | 2.318 |
| MoE training loss (500-step mean) | 1.27 | 1.26–1.33 | **1.41 → 1.53** | 1.54 | 1.58 |
| Dense training loss | 1.33 | 1.32–1.34 | 1.27–1.31 | 1.29 | 1.25 |

- The MoE validation loss **improves** until step 15,000 (2.046), then **jumps by +0.2 nat** between 15,000 and 16,000, and the **training loss rises too** (1.26 → 1.53–1.58). With overfitting the training loss would fall while validation rises; here both worsen: this is a **training instability event** (LR around 1.4·10⁻⁴), not overfitting.
- The final evaluation shows **no routing collapse** (dead experts 0; usage entropy 3.08 / 3.18, that of the 9k run 3.13). The trigger (e.g. attention-logit growth, bf16, router) cannot be identified from this; the Dense model did not break on the same schedule.
- One seed, one run: this is a **hypothesis**, not a confirmed cause. The new K1 runs (W2) can separate it (same configuration with more seeds, gradient-norm and logit-norm logging, a shorter schedule horizon).

## K2 — what does the router depend on? (H06/FD06 first CPU measurement)

**Method:** `tools/analysis/k2_route_trace.py`; 256 val windows (131,072 tokens), seeds 42/43/44 each, `H01_fixed` (FixedDepth AM-LLM) and `B_moe_3090` (MoE). The route label is the first-choice expert of the hop/layer. From the plug-in mutual information (bits) a permutation baseline (5 shuffles) is subtracted, so "excess" is the part above chance. Frequent tokens (≥ 30 occurrences) cover 80% of tokens; rare ones form one group. Platform: Mac arm64, fp32. Mean of the three seeds; the spread between seeds is ≤ 0.14 bit.

| | Route entropy (bits) | I(route; current token), excess | …as a share of route entropy | I(route; previous token) share | I(route; position) | I(route; token shape) |
|---|---|---|---|---|---|---|
| `H01_fixed` (24 experts/hop, max 4.58) | 4.19–4.40 | 1.87–2.35 | **45–54%** | 14–28% | ≈ 0.005–0.009 | 0.48–0.81 |
| MoE (4 experts/layer, max 2.00) | 1.82–1.98 | 1.30–1.51 | **69–78%** | 12–27% | ≈ 0.000–0.001 | 0.14–0.61 |

(The previous and the current token are correlated, so the shares do not add. Token shape: whitespace/newline, number, identifier-like, punctuation, other.)

**What it shows (one measurement family, 3 seeds, only for the variables tested):**
1. **In both models routing is strongly tied to the current token** (token identity explains 45–78% of the route entropy), the previous token adds a smaller further part, and position practically nothing.
2. **The AM-LLM's routing is less token-determined than the MoE's** as a share of route entropy (45–54% against 69–78%), i.e. a larger part is context-dependent or otherwise not token identity. The two route labels have different alphabets (24 vs 4), so the shares are an approximate comparison, not a proven difference.
3. **What it does not measure:** task, syntax or semantic context (labelled categories are needed), the causal role of the route (effect of shuffle/reinit on the loss) and the function of the experts. H06 is therefore **PARTIAL**; the other FD06 questions (shuffle, copying) remain.

## Consequences for the plan
- First method of AD02: **anytime/multi-exit loss**; the AD01 metrics (per-hop NLL, hard–soft gap) can be re-run with the same tool for the gate.
- Because of K1, **gradient-norm and logit-norm logging** is mandatory for every longer (18k) run, and the MoE anchor's 18k run must be repeated with more seeds before long training is used for comparison.
- Missing: AD01 on more seeds (the overnight 36 checkpoints are already on the Mac), the causal test of the route (shuffle/reinit), and an arm64–x86 numerical agreement check (platform independence).
