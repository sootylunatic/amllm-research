**English** | [Magyar](README.hu.md)

# Technical decision records (ADR)

Template: [0000-template.md](0000-template.md). Accepted records are not rewritten; changes get a new record, and factual
errors get a dated *Correction* note. Only technical records are published; missing numbers are project-internal
records. Mentions of those records, of `NEXT_EXPERIMENTS` items and of pre-registration files refer to internal documents.
Measured values quoted in the records are those of the first series at the time of each decision; see `results/README.md` for how to read them.

| # | Decision | Status |
|---|---|---|
| [0002](0002-byte-level-tokenization.md) | Byte-level tokens in the first experiment series | accepted |
| [0003](0003-hierarchical-module-bank.md) | Shared module bank with hierarchical top-k routing | accepted |
| [0004](0004-survival-halting.md) | Cumulative survival halting instead of a per-step threshold | accepted |
| [0005](0005-attention-in-every-hop.md) | Shared attention in every hop; per-hop attention as a candidate | accepted / proposed |
| [0006](0006-expert-dispatch.md) | Stacked experts, `loop` and `bmm` dispatch | accepted, amended by 0014 |
| [0007](0007-corpus-and-split.md) | Open-source, package-disjoint corpus; own code excluded | accepted |
| [0008](0008-control-models.md) | Controls matched on active FLOP and on total parameters | accepted |
| [0009](0009-measurement-protocol.md) | Measurement protocol and pre-registered decision rules | accepted, amended by 0014 |
| [0013](0013-top1-router-gradient.md) | Top-1 routers get no LM gradient; full-softmax weighting | accepted |
| [0014](0014-deterministic-combine-and-diagnostics.md) | Deterministic combine, sync-free statistics, diagnostics-off measurement | accepted |
