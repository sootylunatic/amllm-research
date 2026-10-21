**English** | [Magyar](README.hu.md)

# AM-LLM — Adaptive Modular Language Model (research prototype)

[![tests](https://github.com/sootylunatic/amllm-research/actions/workflows/tests.yml/badge.svg)](https://github.com/sootylunatic/amllm-research/actions/workflows/tests.yml)
![status](https://img.shields.io/badge/results-preliminary%20%E2%80%94%20under%20audit-orange)
![license](https://img.shields.io/badge/license-MIT-blue)

A small, study-oriented research prototype. The models are tiny byte-level language models trained on a single
consumer GPU. This is not a product, and nothing here makes a claim about large-model performance.

> **Preliminary results — under audit.** The first experiment series has been reviewed by an external audit,
> which found measurement and attribution issues. These are being corrected and re-measured. Until that is
> finished, [`results/summary.csv`](results/README.md) is preliminary and should not be cited as a finding.

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

Documentation: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · [docs/METHODOLOGY.md](docs/METHODOLOGY.md) ·
[docs/adr/](docs/adr/README.md)

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
