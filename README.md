**English** | [Magyar](README.hu.md)

# AM-LLM — Adaptive Modular Language Model (research prototype)

[![tests](https://github.com/sootylunatic/amllm-research/actions/workflows/tests.yml/badge.svg)](https://github.com/sootylunatic/amllm-research/actions/workflows/tests.yml)
![status](https://img.shields.io/badge/status-research%20prototype%20%E2%80%94%20main%20hypothesis%20not%20yet%20supported-orange)
![license](https://img.shields.io/badge/license-MIT-blue)

A small, study-oriented research prototype. The models are tiny byte-level language models trained on a single
consumer GPU. This is not a product, and nothing here makes a claim about large-model performance.

> **Status (2026-10-23).** The first experiment series (E00–E09) was audited and corrected; its pre-registered
> go/no-go gate was **not** met. The later phases (F1–F5 and an overnight round on ~17M-parameter models) are
> exploratory. So far they do **not** support the efficiency hypothesis at this scale: the best variants reach
> MoE-close quality but need more compute, while learned routing does beat random routing. The goals and how results
> are judged are in [docs/00-goals/RESEARCH_GOALS.md](docs/00-goals/RESEARCH_GOALS.md).
> [`results/summary.csv`](results/README.md) covers the first series only; later results are not published here yet.

## Research question

Can a language model learn **which processing units to run, in which order and how many times** for each token,
and so match a Dense Transformer or a fixed-depth Mixture-of-Experts at **lower active compute**?

The prototype combines three mechanisms:

1. a **shared module bank** that is reused recurrently ("hops");
2. **hierarchical top-k routing**: a router picks modules, then each picked module picks experts;
3. **per-token adaptive depth**: a halting head decides whether another hop is needed.

## Contents

```text
src/amllm/   model, routing, data, training, experiments, profiling, analysis, fast inference paths, generation,
             coding evaluation, pre-registered gate evaluator
tests/       pytest suite (CPU; additional CUDA/bf16 tests run when a GPU is present)
docs/        architecture, methodology, technical decision records (English and Hungarian)
data/        a tiny toy file only; corpora are rebuilt locally (data/README.md)
results/     preliminary summary table (results/README.md)
```

Documentation: [research goals (A–D)](docs/00-goals/RESEARCH_GOALS.md) · [docs/01-architecture/ARCHITECTURE.md](docs/01-architecture/ARCHITECTURE.md) · [docs/02-methodology/METHODOLOGY.md](docs/02-methodology/METHODOLOGY.md) ·
[docs/04-decisions/](docs/04-decisions/README.md)

## Quick start

```bash
# reference environment: Python 3.12, PyTorch 2.11 (CUDA 12.8); the CPU-only test suite works without a GPU
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e '.[dev,analysis]'
.venv/bin/pytest -q

# 40-step smoke run on the bundled toy file
.venv/bin/python -m amllm.train --data data/toy_typescript.txt --steps 40 --batch 4 --max-seq-len 64 --run-dir runs/smoke
```

## Limitations

- Phase 1 used byte-level tokens and ~0.3–2.5M parameters, trained on ~14 MB of open-source TypeScript. The
  phase 2 tooling adds a byte-level BPE corpus builder (permissive licences only, package hold-out) for ~17M-parameter
  models; phase 2 results are not published yet.
- Generation uses a KV-cache (`amllm.generate`); the fast inference path (Triton grouped GEMM, static execution, CUDA
  Graph) is inference only. Speed gains measured so far come from execution, not from skipped compute.
- All weights stay in GPU memory, so no VRAM saving is claimed.
- The coding evaluation (`amllm.f5eval`) executes generated code only inside a bubblewrap sandbox and needs Node.js and
  a TypeScript package (`AMLLM_TYPESCRIPT`); the related tests are skipped without them.
- No training data and no model weights are published (see [data/README.md](data/README.md)).

## Notes

Claude and Codex models assisted with parts of the implementation and with reviewing the experiments.

## License

MIT, see [LICENSE](LICENSE).
