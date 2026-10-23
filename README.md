**English** | [Magyar](README.hu.md)

# AM-LLM — Adaptive Modular Language Model

**Open research prototype.** Shared expert banks, hierarchical routing, recurrent hops, and optional adaptive computation. [Tests](https://github.com/sootylunatic/amllm-research/actions) · [MIT License](LICENSE)

> **Evidence snapshot — 2026-10-23:** The original compute-efficiency hypothesis is **not supported at the tested ~17M parameter scale**. The best FixedDepth AM-LLM variants approach MoE validation loss with **24–31% more estimated FLOPs**. Current learned halting suffers from a soft-training / hard-inference discrepancy. These are findings, not a claim of superiority.

## What are we investigating?

Can a model learn **which reusable modules or experts to run, in which order and how often**, approaching Dense/MoE quality using less **actually executed compute**? The A–D programme also considers **B** small inference GPU working sets, **C** practical autoregressive generation speed, and **D** useful large-model capability on commodity hardware. A 16 GB VRAM target and ≥60 decoded tok/s are **aspirations, not measured achievements**. [Goals and evaluation](docs/00-goals/RESEARCH_GOALS.md).

## Results so far

F5 / overnight: ~17M parameters, 8,192 BPE vocabulary, context 512, 12 configurations × 3 seeds. Mean validation NLL (nat/token); FLOP ratios are **analytical estimates, not measured decode speed**.

| Configuration | Val NLL ↓ | Est. FLOPs / MoE ↓ |
|---|---:|---:|
| 6-layer MoE control | **2.1166** | 1.00× |
| 6-hop FixedDepth | 2.1625 | 1.00× |
| 8-hop FixedDepth | 2.1212 | 1.24× |
| 6-hop FixedDepth, Top-2/Top-2 | 2.1366 | 1.31× |
| FixedDepth, smaller experts | 2.1804 | 0.85× |

Paired 95% confidence intervals for the top two AM–MoE NLL differences include zero; **this does not establish equivalence**. The 0.85×-compute region remains inconclusive without matched controls. The old test split was visible to the experiment planner, so it is **not an untouched holdout**.

[Results and limitations](results/RESEARCH_REPORT.md) · [Evidence status](docs/RESEARCH_STATUS.md) · [Legacy E00–E09 summary](results/summary.csv)

## Two branches of the architecture

**FixedDepth** reuses one expert bank over fixed 4/6/8 hops; per-hop attention has separate weights. It is a useful control, not proof of adaptive compute. **AdaptiveDepth** learns token-wise stopping, but the existing soft/hard mismatch causes substantial degradation. The six-hop FixedDepth is structurally close to an **expert-tied MoE**: expert weight reuse, two-level routing, and shared normalization change together and **have not yet been individually ablated**. [Architecture guide](docs/ARCHITECTURE_GUIDE.md).

## Contributing

We welcome independent critiques, tests, controlled ablations, profiling, and **negative results**. See [specific open research questions](docs/OPEN_QUESTIONS.md), [contribution guide](CONTRIBUTING.md), and [Issues](https://github.com/sootylunatic/amllm-research/issues).

## Quick start (smoke test, not F5 reproduction)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[dev,analysis]'
.venv/bin/python -m pytest -q
.venv/bin/python -m amllm.train --data data/toy_typescript.txt --steps 40 --batch 4 --max-seq-len 64 --run-dir runs/smoke
```

The complete corpus, checkpoints and per-run logs are **not released**, so public source alone does not yet reproduce the 36 runs. No large-model VRAM or ≥60 tok/s claim is made. Claude and Codex assisted implementation and review; findings remain open to independent scrutiny.
