"""Goal C exploration (W3 side experiment): how much faster can KV-cache decoding of the current models run?

    python tools/analysis/c_decode_bench.py --runs <run_dir> ... --out results/round_v3_w3/c_decode/<gpu>.json

Variants, applied the same way to every model (Dense, MoE and AM-LLM):
  base      fp32 weights + bf16 autocast, Triton expert dispatch, CUDA Graph (as in the W3 measurement)
  bf16      weights cast to bf16 once (removes the per-call expert-weight conversion)
  gather    bf16 + gathered-expert banks (amllm.fastdecode): no sort/count/scatter, static shapes, adaptive K capturable
  compile   gather + torch.compile of the decode step (inductor fusion; still replayed as a CUDA Graph)
Per variant: the validation NLL of the variant (32 windows; quality against base). Per variant and batch size, on
teacher-forced validation text (prompt 128, then the real next tokens): decode tokens/s and ms/token p50/p95 of the timed
path (CUDA Graph replay of the compiled or plain step; eager where not capturable), the NLL that exact path assigns to the
real next tokens, and its per-step prediction agreement with the same variant run eagerly (no compile, no graph).
Validation prompts only. Exploratory: no preregistered gate.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from amllm import env, fastdecode  # noqa: E402
from amllm.data import eval_batches, load_split  # noqa: E402
from amllm.generate import Generator, GraphDecoder  # noqa: E402
from amllm.model import set_diagnostics, set_dispatch  # noqa: E402
from amllm.profile import load_run  # noqa: E402

VARIANTS = ("base", "bf16", "gather", "compile")


def prepare(run_dir: Path, variant: str, device):
    cfg, model = load_run(run_dir, device)
    set_diagnostics(model, False)
    if cfg.arch != "dense":
        set_dispatch(model, "triton")
    if variant != "base":
        fastdecode.to_bf16(model)
    if variant in ("gather", "compile") and cfg.arch != "dense":
        fastdecode.use_gather(model)
    return cfg, model


def stepper(gen: Generator, variant: str, autocast: bool):
    step = gen.decode_step
    if variant == "compile":
        # a fresh compiler state per model and batch: otherwise dynamo hits its recompile limit after a few models and
        # silently falls back to eager (measured: the compile variant then equals gather)
        torch._dynamo.reset()
        step = torch.compile(step, mode="max-autotune-no-cudagraphs", dynamic=False)

    def run(token):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return step(token)
    return run


@torch.no_grad()
def forced_decode(gen: Generator, step, ids: torch.Tensor, prompt: int, timed: bool):
    """Prefill ids[:, :prompt], then feed the real next tokens (teacher forcing). Returns per-step argmax [B,S], the NLL of
    the real next token per step [B,S] and the step times in ms (only if timed)."""
    first = gen.prefill(ids[:, :prompt])[:, -1]
    preds, nlls, times = [first.argmax(-1)], [F.cross_entropy(first.float(), ids[:, prompt], reduction="none")], []
    for i in range(prompt, ids.shape[1] - 1):
        if timed:
            torch.cuda.synchronize()
            s = time.perf_counter()
        logits = step(ids[:, i])
        if timed:
            torch.cuda.synchronize()
            times.append(1000 * (time.perf_counter() - s))
        preds.append(logits.argmax(-1))
        nlls.append(F.cross_entropy(logits.float(), ids[:, i + 1], reduction="none"))
    return torch.stack(preds, 1).cpu(), torch.stack(nlls, 1).cpu(), times


@torch.no_grad()
def bench(model, cfg, variant: str, val, device, batch: int, prompt: int = 128, new: int = 128) -> dict:
    """Times the variant's real execution path (compiled and/or CUDA Graph) on teacher-forced validation text and checks
    that this very path predicts what the same variant computes eagerly."""
    autocast = variant == "base"
    ids = next(eval_batches(val, batch, batch, prompt + new, device))[0]           # [B, prompt+new] real text
    gen = Generator(model, batch=batch, max_len=prompt + new, cache_dtype=torch.bfloat16)
    run = stepper(gen, variant, autocast)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        token = gen.prefill(ids[:, :prompt])[:, -1].argmax(-1)
    for _ in range(3):  # warm-up / compile
        token = run(token).argmax(-1)
    r = {}
    try:
        class _G(GraphDecoder):  # capture the variant's step (compiled or not) instead of the plain decode_step
            def _step(self_inner):
                return run(self_inner.token)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            gen.prefill(ids[:, :prompt])
        dec = _G(gen, ids[:, prompt], autocast_dtype=None)
        step, r["mode"] = dec, "cuda_graph"
    except Exception as e:  # not capturable: the eager path is the timed path
        step, r["mode"] = run, f"eager ({type(e).__name__})"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        preds, nll, times = forced_decode(gen, step, ids, prompt, timed=True)
    # reference: the same variant, plain eager decode_step (no compile, no graph), same inputs
    ref_gen = Generator(model, batch=batch, max_len=prompt + new, cache_dtype=torch.bfloat16)

    def eager(t):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return ref_gen.decode_step(t)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        ref_preds, ref_nll, _ = forced_decode(ref_gen, eager, ids, prompt, timed=False)
    if variant == "compile":
        r["dynamo_unique_graphs"] = int(torch._dynamo.utils.counters["stats"]["unique_graphs"])  # 0 = nothing compiled
    times.sort()
    med = statistics.median(times)
    r.update(tokens_per_s=batch * 1000 / med, ms_per_token_p50=med, ms_per_token_p95=times[int(0.95 * len(times))],
             timed_path_agreement_with_eager=float((preds == ref_preds).float().mean()),
             timed_path_nll=float(nll.mean()), eager_nll=float(ref_nll.mean()))
    return r


@torch.no_grad()
def val_nll(model, variant: str, val, device, windows: int = 32, batch: int = 8) -> float:
    """Full-sequence validation NLL with this variant's weights and expert path (quality check against base), computed by
    the KV-cache path's prefill: the same model as the reference hard/fixed forward (equivalence-tested) and the path the
    decode timing uses; it also runs halting models in pure bf16 (the reference forward is training code and is not
    changed for this exploration)."""
    autocast = variant == "base"
    tot, n = 0.0, 0
    for x, y, _ in eval_batches(val, windows, batch, model.cfg.max_seq_len, device):
        gen = Generator(model, batch=x.shape[0], max_len=x.shape[1], cache_dtype=torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = gen.prefill(x)
        tot += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
        n += y.numel()
    return tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data/code_corpus_v2")
    p.add_argument("--batches", type=int, nargs="*", default=[1, 8, 32])
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    a = p.parse_args()
    device = torch.device("cuda")
    val = load_split(a.data, "val")
    out = {"env": env.collect(), "runs": {}}
    for run in a.runs:
        run_dir = Path(run)
        key = f"{run_dir.parent.name}/{run_dir.name}"
        res = {}
        for variant in a.variants:
            cfg, model = prepare(run_dir, variant, device)
            v = {"val_nll_32_windows": val_nll(model, variant, val, device)}
            for b in a.batches:
                try:
                    v[f"b{b}"] = bench(model, cfg, variant, val, device, b)
                except Exception as e:
                    v[f"b{b}"] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
            res[variant] = v
            del model
            torch.cuda.empty_cache()
            print(key, variant, {k: (round(x["tokens_per_s"]) if isinstance(x, dict) and "tokens_per_s" in x else x)
                                 for k, x in v.items()}, flush=True)
        out["runs"][key] = res
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
