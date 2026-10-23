**English** | [Magyar](0004-survival-halting.hu.md)

# ADR-0004: Cumulative survival halting instead of a per-step threshold

- **Status:** accepted (supersedes the v0.1 per-step rule, which is kept for comparison)
- **Date:** 2026-10-21
- **Affects:** `ModelConfig.halt_rule` (`survival` | `step` | `none`), `AdaptiveModularLM.forward`

## Context
v0.1 trained with cumulative survival weights (`s_h = Π(1 − p_i)`) but stopped at inference on the *per-step*
condition `p_h ≥ 0.65`. Measured (E01): ~1.8 expected hops during training, **4.0** at inference, because the
learned `p_h ≈ 0.5` never reaches 0.65. The two rules do not implement the same model.

## Decision
`survival` rule: the hop update is weighted by `s_h` in both modes; at hard inference a token runs the next hop only
if `s_h ≥ τ`. Hard inference is therefore soft training **truncated** where the remaining weight is small. Ponder
penalty: `λ · Σ_h mean(s_h)`. Test: for `τ → 0` hard ≡ soft (`test_hard_with_tiny_threshold_equals_soft`).

## Consequences
- Calibratable: larger τ → fewer hops (the `step` rule behaved the other way round, E03).
- Measured: λ ≤ 0.01 barely halts (~1%); λ = 0.03 → 1.7 hops but +8% loss and a 4.8% soft/hard gap;
  8 hops + τ = 0.8 → collapse (1.46 → 2.89). Closing the gap is open work (NEXT_EXPERIMENTS N5).
- Halted tokens keep a frozen state but still serve as attention keys/values; attention itself is not skipped
  (ADR-0006, "ideal" FLOP).

## Alternatives considered
- ACT remainder weighting: similar, harder to test; the truncation reading is easier to verify.
- Hard (straight-through) halting during training: later experiment (N5).
