**English** | [Magyar](0005-attention-in-every-hop.hu.md)

# ADR-0005: Shared causal attention in every hop; per-hop attention as the next candidate

- **Status:** accepted (shared attention); per-hop attention **proposed** — amended by ADR-0016
- **Date:** 2026-10-21
- **Affects:** `ModelConfig.attn_per_hop`, `ModelConfig.unshared_attention`

## Context
In v0.1 a single attention ran before all hops; the hops were per-token FFN recursions and tokens could not see
each other between hops. The Dense/MoE controls have their own attention per layer.

## Decision
1. `attn_per_hop=True` is the default: the same (shared) causal attention runs at the start of every hop
   (Universal-Transformer-like recursion). The v0.1 topology (`attn_per_hop=False`) is kept as an ablation.
2. `unshared_attention=True` gives every hop its own attention weights (the module bank stays shared).
   This option was added after E07 as an exploratory variant.

## Consequences
- Measured (E06): the v0.1 topology is worse (1.622 vs. 1.526).
- Measured (exploratory, 3 seeds, equal FLOP): with per-hop attention 1.526 → 1.347 (MoE 1.333, Dense 1.430).
- Per-hop attention adds **198,912** parameters (three extra attention blocks with their LayerNorms).

**Correction (2026-10-21, audit):** the first version of the report attributed this gain to "un-sharing" alone
("largest single factor"). Un-sharing and the extra capacity change together, so the attribution was not
supported. E09 adds a capacity-matched control; the result is recorded in ADR-0016.
