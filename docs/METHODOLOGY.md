**English** | [Magyar](METHODOLOGY.hu.md)

# AM-LLM — experimental methodology and reproduction

Related decisions: [ADR-0007](adr/0007-corpus-and-split.md) (data), [ADR-0008](adr/0008-control-models.md)
(controls), [ADR-0009](adr/0009-measurement-protocol.md) (measurement), [ADR-0014](adr/0014-deterministic-combine-and-diagnostics.md)
(determinism, diagnostics-off latency). Model details: [ARCHITECTURE.md](ARCHITECTURE.md).

> **Status: preliminary — under audit.** The first experiment series was reviewed by an external audit; measurement and
> attribution issues were found and are being re-measured (stage E09). Results published in
> [`results/summary.csv`](../results/README.md) are preliminary.

## 1. Stages

| Stage | Question | Tool |
|---|---|---|
| E00 | Are the environment and the code correct? | `pytest` (CPU; plus CUDA/bf16 tests when a GPU is present), `amllm.env` |
| E01 | Does training run, can checkpoints be saved and reloaded, does inference work? | `amllm.train` (40 steps) |
| E02 | Causality, determinism, counters, real skipping of computation | `amllm.invariants` (CPU fp32 and CUDA fp64) |
| E03 | How does halting trade quality against executed hops? | `amllm.experiment` (ponder sweep) + a pre-registered selection rule |
| E04 / E05 | Dense / fixed-depth MoE controls | `amllm.experiment` |
| E06 | One-factor ablations | `amllm.experiment` |
| E07 | Main comparison, 3 seeds | `amllm.experiment` |
| E08 | Synthetic tasks evaluated separately | `amllm.experiment --data data/synthetic` |
| E09 | Audit controls: capacity-matched per-hop attention, learned vs. random routing, depth-only, top-1 with gradient, 4 vs. 8 experts | `amllm.experiment` (references re-trained on the same code) |

## 2. Training (identical for all models)

| Setting | Value |
|---|---|
| Representation | bytes (vocabulary 256), context 256 |
| Batch × steps | 32 × 6,000 (E08: 3,000) → 49.2M byte predictions per run |
| Optimiser | AdamW, β = (0.9, 0.95), weight decay 0.05 (tensors with ≥ 2 dims only) |
| Learning rate | 1e-3, 100 linear warm-up steps, cosine decay to 10% |
| Precision | bf16 autocast (CUDA), gradient clipping 1.0 |
| Seeds | 42, 43, 44 (main comparison and E09 controls); 42 (ablations) |
| Dispatch | `bmm` for training (numerically equivalent to `loop`) |
| Hardware | one GPU (`CUDA_VISIBLE_DEVICES=0`), no other GPU load during measurements |

## 3. Evaluation

- **Validation:** 512 fixed, evenly spaced 256-byte windows of the validation split (131,072 predictions), identical
  for every model. **Test:** once, in the default inference mode.
- **Default mode:** full depth; adaptive AM-LLM: hard survival halting, τ = 0.5.
- **Additional AM-LLM modes:** soft; hard survival τ ∈ {0.3, 0.5, 0.65, 0.8}; v0.1 `step` τ ∈ {0.5, 0.65, 0.8};
  random routing with the same weights.
- **Per category:** class of the predicted byte (space, newline, letter, digit, punctuation, other) or synthetic task type:
  loss and mean hops.

## 4. Metrics and definitions

| Metric | Definition |
|---|---|
| val/test loss | mean cross-entropy in nat/byte; `bpb = loss / ln 2`; perplexity = `exp(loss)` (per byte) |
| mean hops | executed module-bank calls per token (Dense/MoE: number of layers) |
| estimated FLOP/token | analytic forward estimate: "routed" (selected assignments; "implemented" in phase 1), "executed" (incl. `bmm` padding) and "ideal" ([ARCHITECTURE.md §4](ARCHITECTURE.md#4-flop-estimate-forward-per-token-multiply-add--2-flops)) |
| profiled FLOP/token | `torch.profiler` matmul FLOPs of one forward (batch 32), per dispatch path |
| latency | full-sequence forward, CUDA-synchronised, 10 warm-up + 50 repeats, median and p95, batch 1 and 32, **with diagnostics on and off**, all 3 seeds; **not generated tokens/s** |
| host syncs | synchronisations per forward from CUDA sync-debug mode (a lower bound) |
| VRAM | peak allocation during the forward + fp32 weights, reported **per dispatch path**; training peak separately |
| active parameters/token | from the route trace: unique weights touched, and parameter applications using the **executed** attention calls (the ideal count is reported separately) |
| router specialisation | `I(selected module ; input byte class) / H(byte class)` against an untrained control; usage entropy, max share, dead units |
| random-routing Δ | increase of validation loss when learned routes are replaced by random ones with the same weights |
| training cost | wall-clock time, tokens/s, training peak VRAM |

Every reported row combines quality, compute, latency and memory of **one** execution path (dispatch, halting mode,
batch size, diagnostics setting); values from different paths are not mixed in one row.

## 5. Decision rules

- Hyper-parameters and interpretation rules are written down before running and are not changed afterwards.
- First-round targets: validation loss at most ~5% worse than the matching control; ≥ ~15% compute saving or better
  quality at equal compute; ≤ ~10% latency regression; a learning router; measurable dynamic depth; similar
  conclusions on 3 seeds.
- Component attributions require a control that changes only that component (same parameters and compute);
  otherwise the result is reported as confounded.
- A finished run is reused (`--skip-existing`) only if its model and training configuration, data split id and
  training-code hash (`train_code_sha256`) are identical to the requested run.
- Runs whose configuration makes a router untrainable (top-1 with `topk_softmax`, ADR-0013) are flagged as invalid for
  that question.

## 6. Reproduction

```bash
# environment (reference: Python 3.12, PyTorch 2.11 cu128)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e '.[dev,analysis]'
.venv/bin/pytest -q && .venv/bin/python -m amllm.env

# data (see data/README.md)
.venv/bin/python -m amllm.data build --src <node_modules folders> --out data/ts_corpus --max-mb 32 --val-frac 0.1 --test-frac 0.1
.venv/bin/python -m amllm.synth --out data/synthetic

# experiments: presets are defined in src/amllm/experiment.py
export CUDA_VISIBLE_DEVICES=0
.venv/bin/python -m amllm.experiment --name e07_seeds \
    --presets dense,dense_wide,moe,ut,amllm_fixed,amllm,amllm_m16 --seeds 42,43,44 --set ponder_coefficient=0.01

# analysis
.venv/bin/python -m amllm.invariants --checkpoint <checkpoint.pt>
.venv/bin/python -m amllm.profile --runs <run directories> --out results/profile.json
.venv/bin/python -m amllm.router_analysis --runs <run directories> --out results/router_analysis.json
.venv/bin/python -m amllm.report
```

Notes:
- With the deterministic expert combine (v0.2.1) evaluation forwards are bitwise reproducible; training on GPU is not
  bit-exact across runs (atomic gradient accumulation), so the same seed can differ by about ±0.002 validation loss.
- The code corpus depends on the local `node_modules` content and is reproducible byte-for-byte only from the same
  source; the split id is stored in every run's `config.json`.
