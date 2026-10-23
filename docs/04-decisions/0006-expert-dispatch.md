**English** | [Magyar](0006-expert-dispatch.hu.md)

# ADR-0006: Stacked expert weights with two dispatch paths (`loop` and `bmm`)

- **Status:** accepted — amended by ADR-0014
- **Date:** 2026-10-21
- **Affects:** `amllm/routing.py: ExpertBank`, `amllm.model.set_dispatch`

## Context
v0.1 called `nonzero()` per module and per expert, i.e. many host–GPU synchronisations per hop. We also need to
show that sparse execution *really* skips work, not only that the statistics report fewer calls.

## Decision
- All experts of all modules live in one `[G, D, H]` / `[G, H, D]` weight tensor (`G = M·E`).
- `loop`: assignments are sorted by expert; one matmul pair per expert on its own rows only. One host sync per
  call (the counts). Computed rows = routed assignments.
- `bmm`: every expert's rows are padded to the largest group and one batched matmul runs (a grouped-GEMM stand-in).
  Few kernels, but padding and empty experts are computed too.
- Both paths are numerically equivalent (outputs and gradients, tested); `bmm` for training, `loop` to measure
  real skipping.

## Consequences
- Measured (E02): forcing 1 vs. 4 hops, `loop` computes 32,768 vs. 131,072 FFN rows, 2.2 vs. 8.2 ms → skipping is real.
- Measured (profile): an AM-LLM forward launches ~1,660 kernels (`loop`); halting cuts profiled FLOPs by 36% but
  wall-clock time by only 3.5%. `bmm`: 2.1–3× faster than `loop`, but computes 2.3–3× more FLOPs.

**Correction (2026-10-21, audit):** the report's statement that "~60% of wall-clock time is CPU launch overhead"
was derived by subtracting profiled kernel time from a separately measured latency. That difference mixes launch
overhead, host synchronisation, Python and diagnostics work and does not decompose them. ADR-0014 adds a
diagnostics-off path and sync counts; the report now states only what is measured.
