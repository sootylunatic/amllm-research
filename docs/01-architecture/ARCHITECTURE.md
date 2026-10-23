---
title: "AM-LLM — architecture and implementation (v0.2.1)"
version: "v0.2.1"
created: "2026-10-21"
updated: "2026-10-23"
status: "HISTORICAL"
scope: "VERSION"
source_of_truth: true
supersedes: []
superseded_by: []
---

**English** | [Magyar](ARCHITECTURE.hu.md)

> **Document scope (2026-10-23):** this is the exact **historical v0.2.1 implementation**; it is not an automatic specification for later F5 or overnight branches. Current **A–B–C–D research objectives**: [../00-goals/RESEARCH_GOALS.md](../00-goals/RESEARCH_GOALS.md). Actual active-compute savings (A), real inference memory (B), decode speed (C), and larger-model skill (D) require distinct evidence. The technical description below is preserved.


# AM-LLM — architecture and implementation (v0.2.1)


This document describes the code in `src/amllm/` exactly. The rationale for each decision is in the ADRs
([`adr/`](../04-decisions/README.md)); the measurement protocol is in [../02-methodology/METHODOLOGY.md](../02-methodology/METHODOLOGY.md).

## 1. Overview

```text
bytes ─► embedding + learned position ─► [ v0.1 topology: 1× attention ]
                                          │
                 ┌────────────────────────┴───── hop h = 1 … H ─────────────────────────┐
                 │  w_h = s_h · active_h            (survival weight; mask in hard mode)  │
                 │  x ← x + w_h · Attn_h(LN(x))      (shared, or per-hop weights)         │
                 │  x ← x + w_h · Bank(LN(x))        (runs only for active tokens)        │
                 │  p_h = σ(halt(x));  s_{h+1} = s_h · (1 − p_h)                          │
                 │  hard: active_{h+1} = active_h ∧ (s_{h+1} ≥ τ)                         │
                 └───────────────────────────────────────────────────────────────────────┘
                                          │
                                LN ─► lm_head ─► next-byte logits
```

Notation: `d` hidden size, `t` context length, `M` modules, `E` experts per module, `Km` module top-k,
`Ke` expert top-k, `H` maximum hops, `h_e` expert hidden width.

## 2. Components

### 2.1 Input and output (`_ByteLM`)
- `embed: Embedding(256, d)` + `positions: Embedding(max_seq_len, d)`.
- Output: `LayerNorm` → `Linear(d, 256, bias=False)` (not tied to the embedding).
- `diagnostics` flag (see 2.7).

### 2.2 Causal attention (`CausalAttention`)
`qkv: Linear(d, 3d)`, `out: Linear(d, d)`, `n_heads` heads, `F.scaled_dot_product_attention(is_causal=True)`.
Pre-LN: `x + Attn(LN(x))`.

### 2.3 Expert bank (`ExpertBank`)
`G = M·E` independent experts with stacked weights
`w1 ∈ R^{G×d×h_e}`, `b1 ∈ R^{G×h_e}`, `w2 ∈ R^{G×h_e×d}`, `b2 ∈ R^{G×d}`;
`FFN_g(x) = SiLU(x·w1_g + b1_g)·w2_g + b2_g`.

Input: a list of assignments `(token, expert, weight)`. Output: `y[token] = Σ weight · FFN_expert(x[token])`.

| `dispatch` | Behaviour | Rows computed | Host syncs |
|---|---|---|---|
| `loop` | sort by expert, one matmul pair per expert on its own rows | = number of assignments | 1 per call |
| `bmm` | pad every expert's rows to `cap = max(count)`, two `baddbmm` | `G · cap` (≥ assignments) | 1 per call |

**Combine.** With `routes_per_token = Km·Ke` (always the case inside the module bank) the per-token expert outputs are
summed with a fixed-order reduction (`index_copy` back to token order, then `view(N, k, d).sum(1)`) instead of an
atomic `index_add`, so CUDA forwards are bitwise reproducible.

Both dispatch paths give the same outputs and gradients (CPU and CUDA/bf16 tests). `rows_computed` and
`last_executed` are diagnostic counters.

### 2.4 Hierarchical module bank (`HierarchicalModuleBank`)
Input `x ∈ R^{N×d}` (only tokens that are still running).

1. Module scores `z = Wm·x + bm ∈ R^{N×M}`; select top-`Km` modules `m`.
2. Expert scores only for the selected modules: `u = x·We[m] + be[m] ∈ R^{N×Km×E}`; select top-`Ke` experts `e`.
3. Route weights (`route_weighting`):
   - `topk_softmax` (default): softmax over the **selected** scores. With `k = 1` the weight is identically 1 and that
     router receives no language-model gradient (ADR-0013).
   - `full_softmax`: softmax over **all** scores, gathered at the selected routes (Switch-Transformer style);
     every router receives gradient for every `k`. Weights then do not sum to 1.
4. Assignments `(token n, expert m·E + e, weight w_m·w_e)`: `Km·Ke` per token.
5. Load balancing (Switch style, `load` detached):
   module `M · Σ_m mean_n softmax(z)_m · load_m`, `load_m = count_m / (N·Km)`;
   expert `E · Σ_m load_m · Σ_e imp_{m,e} · load_{m,e}` (averaged over the tokens that chose the module).
6. `router_mode="random"`: uniform random routes with uniform weights `1/k` — a training-time "no learned
   routing" control with identical parameters and compute.
7. Evaluation diagnostics: `random_routing` (random choices, learned scores as weights), `last_route` (trace).

The fixed MoE control is the same class with `M = 1, Km = 1`.

### 2.5 AM-LLM recursion (`AdaptiveModularLM.forward`)
Per-token state: `s` (survival, initially 1), `active` (hard mode), `counts` (executed hops).

For each hop `h = 0 … H−1`:
1. Hard mode: `idx` = active tokens; if empty the loop ends (later hops **do not run**).
2. Weight `w = s` (`survival`) or `w = 1` (`step`, `none`); in hard mode `w ← w · active`.
3. With `attn_per_hop`: `x ← x + w · Attn_h(LN_h(x))`; `Attn_h` is shared, or for `h ≥ 1` has its own weights
   when `unshared_attention=True`. Attention runs for all tokens (halted tokens keep serving as keys/values).
4. Module bank: soft mode for all tokens, hard mode only for `idx`: `x[idx] += w[idx] · Bank(LN(x[idx]))`.
5. Halting (`h < H−1`, `halt_rule ≠ none`): `p = σ(halt_head(x))`, `s ← s·(1−p)`;
   hard `survival`: `active ← active ∧ (s ≥ τ)`; hard `step` (v0.1): `active ← active ∧ (p < τ)`.

Auxiliary loss: `aux = λ_ponder · P + λ_balance · mean_h(balance_h)`; soft mode `P = Σ_h mean(s_h)` (expected hops),
hard mode the mean executed hops. Halting head init: weight 0, bias `halt_bias_init` (−1 → `p ≈ 0.27`).

**Tested properties:** causal in every mode (fp64 exact on CPU and CUDA; bf16 within rounding on CUDA);
deterministic evaluation; hard ≡ soft for `τ → 0`; `module_counts = Σ hops · Km`,
`expert_counts = Σ hops · Km · Ke`; computed FFN rows = routed assignments (`loop`); fewer FLOPs on early exit.

### 2.6 Control models (`LayeredLM`)
`n_layers` blocks, each with its own attention and either a dense FFN (`d → ffn_hidden → d`, SiLU) or a MoE layer
(`n_experts` × `expert_hidden`, top-`moe_topk`). Pre-LN, residual. UT control: `arch="amllm"`, `M=E=Km=Ke=1`
(one shared block applied H times), with `halt_rule="none"` (fixed depth) or with halting (depth-only control).

### 2.7 Diagnostics and the inference fast path
With `diagnostics=True` (training and evaluation) every forward also computes the balance loss, module statistics
and an analytic FLOP report (which reads counts back to the host). `set_diagnostics(model, False)` skips them;
outputs are identical (tested). Remaining host syncs per hop: the expert counts of the dispatch and, in hard mode,
the active-token index. Latency is reported for both settings.

## 3. Configuration (`ModelConfig`)

| Field | Default | Meaning |
|---|---|---|
| `arch` | `amllm` | `amllm` / `dense` / `moe` |
| `vocab_size`, `max_seq_len` | 256, 128 | byte tokens, context (experiments use 256) |
| `d_model`, `n_heads` | 64, 4 | hidden size, heads (experiments use 128) |
| `n_layers`, `ffn_hidden` | 4, 256 | Dense/MoE depth, dense FFN width |
| `n_experts`, `moe_topk` | 8, 2 | MoE layer |
| `n_modules`, `experts_per_module` | 8, 4 | AM-LLM bank |
| `module_topk`, `expert_topk` | 2, 2 | top-k per level |
| `expert_hidden` | 128 | expert FFN width (AM-LLM and MoE) |
| `n_hops` | 4 | maximum hops |
| `attn_per_hop` | True | False = v0.1 topology (one attention before all hops) |
| `unshared_attention` | False | per-hop attention weights (+3 attention blocks for 4 hops) |
| `halt_rule`, `halt_threshold` | `survival`, 0.5 | ADR-0004 |
| `halt_bias_init` | −1.0 | halting head bias |
| `route_weighting` | `topk_softmax` | `topk_softmax` / `full_softmax` (ADR-0013) |
| `router_mode` | `learned` | `learned` / `random` (control) |
| `ponder_coefficient` | 0.002 | λ (main experiments: 0.01, chosen by a pre-registered rule) |
| `balance_coefficient` | 0.01 | load balancing |

## 4. FLOP estimate (forward, per token; multiply-add = 2 FLOPs)

| Term | Formula |
|---|---|
| attention application | `8d² + 2d(t+1)` (qkv + out projections; QKᵀ and AV with on average `(t+1)/2` keys) |
| expert / FFN call | `4·d·h` |
| module router | `2·d·M` per hop; expert router `2·d·E` per selected module |
| halting head | `2·d` per hop (except the last) |
| LM head | `2·d·256` |

- **Routed** (`flops_per_token`, called "implemented" in phase 1): attention in every executed hop for every token,
  plus the *selected* expert assignments only.
- **Executed** (`amllm.profile`): routed plus the padding rows the `bmm` dispatch also computes; equals routed for `loop`.
- **Ideal:** attention only for active query tokens — not implemented yet.
- **Profiled** (`amllm.profile`): `torch.profiler` matmul FLOPs; SDPA is not counted, so the difference to the estimate
  is exactly `attention calls · 2d(t+1)` (verified). With `bmm` dispatch the profiled FLOPs include the padding.

## 5. Modules

| File | Content |
|---|---|
| `config.py` | `ModelConfig` (validation, dict conversion) |
| `routing.py` | `count_ids`, `ExpertBank`, `HierarchicalModuleBank` |
| `model.py` | `AdaptiveModularLM`, `LayeredLM`, FLOP functions, `set_dispatch`, `set_diagnostics`, `set_random_routing` |
| `data.py` | corpus builder, package-disjoint split, batch sampling, byte classes |
| `synth.py` | synthetic tasks with category labels |
| `train.py` | training, evaluation modes, run directory |
| `experiment.py` | preset × seed matrix, `summary.csv` per experiment |
| `invariants.py` | correctness checks (CPU and CUDA), skipping measurement, route trace |
| `profile.py` | latency (diagnostics on/off), VRAM, profiled FLOPs, kernel and sync counts, active parameters |
| `router_analysis.py` | routing statistics, mutual information, untrained control |
| `report.py` | global summary, comparison JSON, figures (en/hu) |
| `bench.py`, `env.py` | microbenchmark, environment record (with path sanitising) |
