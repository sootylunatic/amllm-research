**English** | [Magyar](0009-measurement-protocol.hu.md)

# ADR-0009: Measurement protocol and pre-registered decision rules

- **Status:** accepted — amended by ADR-0014
- **Date:** 2026-10-21
- **Affects:** `amllm/train.py`, `amllm/profile.py`, `amllm/router_analysis.py`, `results/*/PREREGISTRATION.md`

## Context
For small models the variance between seeds is comparable to the differences between models, and adjusting
hyper-parameters after seeing results easily manufactures "success".

## Decision
- Validation: 512 fixed, evenly spaced windows of the validation split — identical for every model.
- Default inference mode: full depth; adaptive AM-LLM: hard survival halting, τ = 0.5.
- Main comparison with 3 seeds (42/43/44): mean ± std plus raw per-seed values; paired (same-seed) comparison.
- Hyper-parameter choices (e.g. ponder λ) follow a rule written down **before** running; it is not changed afterwards.
- Latency: CUDA-synchronised, warm-up + repeats, median and p95, full-sequence forward — **not** generated tokens/s.
  VRAM: only the measured peak; no saving is claimed (all weights stay on the GPU).
- Router: random-routing ablation (same weights) and mutual information with the byte class against an
  untrained control.
- Every run stores `config.json`, `env.json` (versions, hardware, source hash, git commit), `train.log`, `report.json`.

## Consequences
- The first-round targets (≤ 5% quality gap, ≥ 15% compute saving, ≤ 10% latency regression) were not adjusted to
  the results; the report states per criterion what was met.
