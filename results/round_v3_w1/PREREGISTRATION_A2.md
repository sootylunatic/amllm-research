---
title: "W1 preregistration — amendment A2 (diagnostic note, per-bank expert statistics)"
created: "2026-10-23"
updated: "2026-10-23"
status: "ACTIVE"
scope: "RESULT"
source_of_truth: true
supersedes: []
superseded_by: []
locked: true
---

**English** | [Magyar](PREREGISTRATION_A2.hu.md)

# W1 preregistration — amendment A2 (diagnostic note, per-bank expert statistics)

**Status:** The sealed [W1 preregistration](PREREGISTRATION.md) and [A1](PREREGISTRATION_A1.md) are not rewritten; this is a dated supplement motivated by a verified review finding. The W1 code (`train_code_sha256` = `ba81b4fc02cd7b63`) does **not change** because of it; the runs continue.
**Date:** 2026-10-23.

## A2.1 The defect and its scope
In the W1 code, `AdaptiveModularLM` sums the per-hop `module_counts` / `expert_counts` counters into one `[module, expert]` table without a bank identifier. With `bank_groups > 1` (FD01a, FD01b, FD01c), **experts with the same index in different banks are merged**: e.g. out of 12 distinct experts the report sees 4 categories.
- **Affected (diagnostics):** the `expert_counts`, `module_counts`, `dead_experts`, `expert_usage_entropy`, `max_expert_share` fields of `report.json` and `summary.csv` are **invalid** for the FD01a/b/c runs.
- **Not affected:** the validation loss, FLOP (`flops_per_token`, `flops_per_token_executed`), the parameter count and the training. The FD decision rules rest on the loss only, the AD gate on loss and FLOP; **no decision rule uses the expert statistics**. The counters of the `bank_groups = 1` runs (B_fixed, B_moe, A0, FD01d, FD02, FD03, AD02a) are correct.

## A2.2 Handling
1. The invalid fields of FD01a/b/c are not quoted in reports; the correct per-bank values come from **`tools/analysis/bank_usage.py`** run on the saved checkpoints (route trace, validation split, CPU), and these go into the FD01 results.
2. The fix (per-bank counters) is ready and tested on the **`w2-prep` branch** (for the next wave); the W1 code stays frozen, so the training code on `main` is not changed.
3. W1 itself does not change: no rerun, no new gate.
