**English** | [Magyar](0003-hierarchical-module-bank.hu.md)

# ADR-0003: Shared module bank with hierarchical top-k routing (module → expert)

- **Status:** accepted (subject of the research hypothesis)
- **Date:** 2026-10-21
- **Affects:** `amllm/routing.py` (`HierarchicalModuleBank`, `ExpertBank`), `amllm/model.py` (`AdaptiveModularLM`)

## Context
The research question is whether a model can choose, per token and per step, *which* processing units to run, and
thereby reach similar quality with less active compute. A standard Transformer runs every layer for every token;
a MoE chooses experts per layer, but the layer order is fixed.

## Decision
- One module bank **reused in every hop**: `M` modules × `E` experts (SiLU FFNs).
- Per hop and per token: a module router selects top-`Km` modules; the expert router of each selected module
  selects top-`Ke` experts. Output: `Σ_m Σ_e w_m · w_e · FFN_{m,e}(x)`.
- An expert router runs only for tokens that selected its module (conditional computation).
- Switch-style load-balancing loss (differentiable importance × detached load) on both levels.
- A plain MoE layer is the special case `M = 1, Km = 1`, so the control shares the same code.

## Consequences
- Measured (E07, 3 seeds): the router learns. Random routes with the same weights cost +2.99 nats, the
  route/byte-class mutual information is 2–4× that of an untrained control, and there is no collapse
  (normalised entropy 0.98–0.99).
- Measured: at equal active FLOP the quality is worse than Dense (+6.8%) and MoE (+14.5%) (ADR-0011).
- With the default weighting a top-1 router (module or expert) receives no LM gradient (ADR-0013).

## Alternatives considered
- One flat router over all `M·E` experts: loses the module as a unit that could later be cached per task.
- Dense soft mixing of all modules: not sparse, no compute saving.
