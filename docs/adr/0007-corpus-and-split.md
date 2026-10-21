**English** | [Magyar](0007-corpus-and-split.hu.md)

# ADR-0007: Open-source, package-disjoint code corpus; own code excluded

- **Status:** accepted
- **Date:** 2026-10-21
- **Affects:** `amllm/data.py`, `amllm/synth.py`, `data/README.md`

## Context
v0.1 trained on a 986-byte file without any train/validation split, which only measures memorisation. Real code
is needed, but code from private projects may contain secrets and personal data.

## Decision
- Source: `.ts/.tsx` files (not `.d.ts`) of third-party open-source npm packages from a local `node_modules`
  folder. Private repositories are **not** included.
- Deduplication by content SHA-256; minified files are filtered.
- **Package-disjoint** split by hashing the package name (train/val/test ≈ 80/10/10%), so files of one library
  cannot leak into evaluation. The split id (SHA-256) is stored in every run's `config.json`.
- A separate synthetic set (repeat / copy / add) with per-byte task labels; val/test lines do not occur in train
  (except repeat lines: only 234 variants exist).
- Neither corpus nor manifest is published (local paths, mixed licences); generation is deterministic.

## Consequences
- Validation/test data never entered training; the test split is evaluated once per run.
- The corpus is small (13.6 MB train); 6,000 steps ≈ 3.6 epochs.
