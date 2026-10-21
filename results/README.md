**English** | [Magyar](README.hu.md)

# Results (preliminary — under audit)

`summary.csv` holds one row per run and evaluation mode. It is the only published result artefact: run logs,
checkpoints and the detailed report are kept in the private development archive.

**Read before using the numbers:**

- The table is **preliminary**. An external audit of the first series found measurement and attribution issues;
  corrections are being re-measured.
- Compare models **only within the same `experiment`** — and therefore the same `code_version` — and the same
  `eval_mode`. E07 and E09 used different code versions; E09 retrains its own references.
- Rows with a non-empty `validity_note` do not measure what their preset name suggests. Example: top-1 routing with
  `topk_softmax` weighting, where the router gets no gradient (see ADR-0013).
- `eval_batch_ms_with_diagnostics` includes routing statistics and host synchronisation. It is **not** a latency result
  and not generated tokens/s.

| Column | Meaning |
|---|---|
| `experiment` | E01 smoke, E03 halting sweep, E06 ablations, E07 main comparison, E07b reduced compute / exploratory, E08 synthetic tasks, E09 audit controls |
| `code_version` | package version that produced the run |
| `preset`, `seed`, `arch` | configuration name (see `src/amllm/experiment.py`), random seed, model family |
| `eval_mode`, `is_default_mode` | evaluation mode (e.g. `fixed`, `soft`, `hard_surv_0.5`, `random_routes`); the default inference mode is marked |
| `validity_note` | why a row must not be read as its name suggests (empty if no known issue) |
| `params_total`, `params_non_embedding` | stored parameters |
| `val_loss`, `val_ppl`, `val_bpb` | validation cross-entropy (nat/byte), per-byte perplexity, bits per byte |
| `mean_hops`, `p95_hops` | executed hops per token (Dense/MoE: number of layers) |
| `est_flops_per_token_impl`, `est_flops_per_token_ideal` | analytic forward FLOPs per token as implemented / with attention only for active tokens |
| `eval_batch_ms_with_diagnostics` | evaluation batch time including diagnostics (not a latency result) |
| `expert_usage_entropy_norm`, `dead_experts`, `max_expert_share` | routing balance (empty for Dense) |
| `test_loss_default_mode` | test-split loss, default mode only |
| `train_tokens`, `train_seconds`, `train_tokens_per_s`, `train_peak_vram_mb` | training budget and cost |
