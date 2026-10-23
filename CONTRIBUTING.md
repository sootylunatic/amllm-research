**English** | [Magyar](CONTRIBUTING.hu.md)

# Contributing to AM-LLM research

This project values **well-controlled negative results**, reproducibility improvements, tests, reviewer feedback and architectural ablations. [Open questions](docs/OPEN_QUESTIONS.md) identify useful starting points.

1. Read [research goals](docs/00-goals/RESEARCH_GOALS.md) and [current results](results/RESEARCH_REPORT.md); check [Issues](https://github.com/sootylunatic/amllm-research/issues).
2. Open a proposal giving a falsifiable hypothesis, matching Dense/MoE control, fixed variables, expected metrics, correctness tests and smallest informative experiment. Coordinate before expensive GPU work.
3. Submit a PR with code/tests, exact source commit and train-code hash, corpus/tokenizer/split IDs, seed list, steps/tokens, model parameters, primary validation NLL, actual or estimated FLOP (clearly labelled), and uncertainty. Differentiate **training vs inference** throughput and VRAM.
4. An exploratory one-seed pilot is **not** a confirmed effect. Three-seed confidence intervals can still be wide. Historical gates cannot be reinterpreted after a run; report unsuccessful ablations too.

## Local preflight

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -e '.[dev,analysis]'
.venv/bin/python -m pytest -q
```

The included toy fixture is not the full F5 training corpus. The overnight aggregate is not publicly reproducible end-to-end until its dataset/checkpoints are released. **Do not tune on the old planner-exposed test set or any sealed holdout.** Do not upload secrets, unlicensed corpora, protected model weights or sensitive infrastructure details. Running generated code requires an isolated offline sandbox.

**PR checklist:** hypothesis and controls; correctness/causality tests; identifiers and seeds; paired results and negative cases; accurate compute/VRAM/decode claims; no holdout leakage; documentation updated.
