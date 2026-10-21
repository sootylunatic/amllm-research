**English** | [Magyar](0008-control-models.hu.md)

# ADR-0008: Control models matched on active FLOP and on total parameters

- **Status:** accepted — amended by ADR-0016
- **Date:** 2026-10-21
- **Affects:** `amllm/model.py: LayeredLM`, `amllm/experiment.py: PRESETS`

## Context
A sparse model easily "wins" if it has more parameters or more compute. We need two comparison axes and must
separate the effect of weight sharing from that of modularity.

## Decision
Common: d_model = 128, 4 heads, context 256, byte tokens, same optimiser and token budget.
- **Equal active FLOP (~1.9 MFLOP/token):** Dense (4 layers, FFN 512); MoE (4 layers, 8 experts × 256, top-2);
  UT (one shared block × 4 hops, no routing); AM-LLM (8×4 experts × 128, top-2/top-2, 4 hops).
- **Equal total parameters (~2.2–2.4M non-embedding):** wide Dense (FFN 2048), MoE, AM-LLM with 16 modules.
- **Reduced compute (~0.85–1.0 MFLOP/token):** 2-layer Dense, 2-layer MoE, strongly halting AM-LLM.
- FLOPs: analytic forward estimate (`attention_flops`, `ffn_flops`), "implemented" and "ideal", cross-checked
  with `torch.profiler`.

## Consequences
- Remaining differences (parameter counts, attention sharing) are marked in every table.

**Correction (2026-10-21, audit):** the first report attributed the UT → AM-LLM gain (−11% loss) to routing. UT has
297k and AM-LLM 1.23M parameters, so routing and stored capacity changed together. E09 adds a random-routing
control with identical parameters and FLOP; the result is recorded in ADR-0016.
