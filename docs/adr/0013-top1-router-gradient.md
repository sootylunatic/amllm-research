**English** | [Magyar](0013-top1-router-gradient.hu.md)

# ADR-0013: Top-1 routers get no LM gradient with renormalised weights; add full-softmax weighting

- **Status:** accepted (supersedes ADR-0010)
- **Date:** 2026-10-21
- **Affects:** `HierarchicalModuleBank._select`, `ModelConfig.route_weighting`, tests `test_topk_softmax_top1_router_gets_no_lm_gradient`, `test_full_softmax_weighting_gives_top1_routers_gradient`

## Context
The default route weight is a softmax over the *selected* scores. With `k = 1` it is identically 1, so the router of
that level receives gradient only from the load-balancing loss. An external audit pointed out that this holds for the
**expert** level (`Ke = 1`) as well as for the module level (`Km = 1`); verified by a gradient test on both levels.
Consequently the E06 rows "module top-1" and "expert top-1" measure a router that cannot learn from the language-model
loss, not the capability of top-1 routing.

## Decision
- Keep `route_weighting="topk_softmax"` as the default so that all E00–E08 results remain valid for the code they
  were produced with.
- Add `route_weighting="full_softmax"` (Switch-Transformer style): the weight is the softmax over **all** scores,
  gathered at the selected routes, so every router receives gradient for every `k`.
- E09 re-measures top-1 at both levels with `full_softmax`, against a top-2 `full_softmax` reference.
- The E06 top-1 rows are marked invalid as measurements of top-1 routing.

## Consequences
- The defect is pinned by tests on both levels; the fix is tested to deliver gradient.
- `full_softmax` changes the output scale (weights no longer sum to 1); compare `full_softmax` runs only with each other.
