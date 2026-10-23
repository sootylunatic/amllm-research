**English** | [Magyar](0014-deterministic-combine-and-diagnostics.hu.md)

# ADR-0014: Deterministic expert combine, sync-free statistics and a diagnostics-off measurement path

- **Status:** accepted (amends ADR-0006 and ADR-0009)
- **Date:** 2026-10-21
- **Affects:** `ExpertBank.forward(routes_per_token=…)`, `count_ids`, `set_diagnostics`, `amllm.profile`, `tests/test_cuda.py`

## Context
An external audit found that (1) latency was measured with routing statistics, the balance loss and a host-side FLOP
report running inside every forward (each `float(tensor)` and `torch.bincount` is a host synchronisation on CUDA), and
(2) all correctness tests ran on CPU/fp32 although training and evaluation run on CUDA/bf16. Adding CUDA checks
showed (a) AM-LLM forwards were not bitwise deterministic on CUDA (atomic `index_add` of the `Km·Ke` expert outputs per
token; ~1e-15 in fp64, identical routes), and (b) in bf16 a prefix logit can change by 1 ulp when future tokens change,
because a GEMM over a different number of rows per expert rounds the same row differently — in fp64 the difference is
~1e-15, so this is batch-composition numerics, not information flow.

## Decision
- Expert outputs are combined with a fixed-order reduction (`index_copy` back to token order, then `view(N, k, D).sum(1)`);
  CUDA forwards are bitwise reproducible.
- Counting uses `scatter_add` (`count_ids`) instead of `torch.bincount` (no host sync).
- `set_diagnostics(model, False)`: no balance loss, no module statistics, no FLOP report → only the necessary syncs remain
  (one per hop for the dispatch counts, one per hop for the active-token index in hard mode).
- `amllm.profile` measures latency with diagnostics on **and** off, and counts host syncs (CUDA sync-debug mode, lower bound).
- CUDA tests: fp64 exact causality, bf16 causality within rounding, loop/bmm forward and all-gradient agreement, determinism.
  E02 invariants run on CPU and on CUDA (fp64).

## Consequences
- Training/eval numerics differ from the E00–E08 code at rounding level, so E09 retrains its references.
- The latency comparison now separates the model's necessary execution from the measurement code.
