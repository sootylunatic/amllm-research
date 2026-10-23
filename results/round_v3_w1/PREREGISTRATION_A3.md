---
title: "W1 preregistration — amendment A3 (a fourth confounded difference: the router training rule)"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
locked: true
---

**English** | [Magyar](PREREGISTRATION_A3.hu.md)

# W1 preregistration — amendment A3 (a fourth confounded difference: the router training rule)

**Status:** The sealed [W1 preregistration](PREREGISTRATION.md), [A1](PREREGISTRATION_A1.md) and [A2](PREREGISTRATION_A2.md) are not rewritten; this is a dated supplement motivated by a verified review finding. The code (`ba81b4fc02cd7b63`) is unchanged. The runs added here start after A3 is recorded.
**Date:** 2026-10-23.

## A3.1 The finding
By the saved configurations, the MoE control (`moe_f5`, `B_moe`) uses the default `route_weighting = topk_softmax`, the FixedDepth AM-LLM (`amllm_f5`, `B_fixed`) uses `straight_through`. The forward value is identical (a softmax over the selected scores), but **the gradient reaching the router differs** (`straight_through` also passes it through the full softmax probability). The inventory of the AM–MoE difference therefore holds **four** confounded factors: (1) expert tying, (2) hierarchical routing, (3) shared bank normalisation, (4) **the router training rule**. The documents are corrected accordingly.

## A3.2 New controls (FD07), 2 × 3 seeds
| ID | Setting | Reference | What it isolates |
|---|---|---|---|
| **B_moe_st** (FD07a) | `moe_f5`, `route_weighting=straight_through` | B_moe | the effect of the rule on the **MoE side** |
| **B_fixed_fs** (FD07b) | `amllm_f5`, `halt_rule=none route_weighting=full_softmax` | B_fixed | the effect of the rule on the **AM side** (a gradient-preserving, non-straight-through rule) |

In the AM-LLM `topk_softmax` gives a constant weight 1 for a top-1 module (the module router gets no LM gradient, ADR-0013): that cell is degenerate, so `full_softmax` is the appropriate control on the AM side. The decision rule (directional, |Δ| ≥ 0.010, 95% CI, 3/3 seeds) is the same for the pairwise comparisons. The result is two rows of a 2×2 layout (architecture × rule); **causal statements only from a pair within the same architecture**, and every comparison states its total parameter count.
- **Scope:** +6 runs; W1 totals **48 runs**.
