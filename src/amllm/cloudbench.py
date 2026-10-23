"""GPU benchmark for choosing cloud hardware (F5 cloud comparison): short training on the real F5 presets and data.

    python -m amllm.cloudbench --presets moe_f5,amllm_f5,dense_f5 --data data/code_corpus_v2 --out bench.json

Measurement tool, not training code (not part of `train_code_sha256`); it produces no F5 result. The training step
mirrors `train.run` exactly (same model, optimizer groups, cosine schedule of the 9000-step F5 run, bf16 autocast,
gradient clipping, data order of the seed); `tests/test_cloudbench.py` checks the losses against `train.run`.
Phases are timed separately per preset:
  build     model construction and transfer to the GPU
  warmup    the first --warmup-steps steps (CUDA context, cuBLAS/cuDNN heuristics, allocator growth)
  measured  --steps steps, each timed between cuda synchronizations as in train.run: median/p95/mean step time,
            tokens/s, steps/s, peak allocated and reserved memory
  eval      one validation pass in the default mode (eval_windows of the F5 runs)
  infer     Triton dispatch (JIT compile on first call) + B2 static/CUDA-Graph forward and CUDA-Graph decode, as used
            by f5eval: first-call (compile + capture) time and steady-state numbers
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from . import env
from .config import ModelConfig
from .data import load_split, random_batch
from .experiment import PRESETS
from .model import build_model, count_parameters, set_diagnostics, set_dispatch
from .train import TrainConfig, evaluate

F5_STEPS = 9000  # length of the F5 runs: the learning-rate schedule of the benchmark is the one of a real run


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def train_steps(cfg: ModelConfig, tc: TrainConfig, n_steps: int, device, schedule_steps: int) -> tuple:
    """The training loop of train.run for the first n_steps of a schedule_steps run. Returns (model, step seconds,
    losses per step, build seconds)."""
    torch.manual_seed(tc.seed)
    train_data = load_split(tc.data, "train")
    sync(device)
    t0 = time.perf_counter()
    model = build_model(cfg).to(device)
    set_dispatch(model, tc.dispatch)
    sync(device)
    build_s = time.perf_counter() - t0
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tc.lr, betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < tc.warmup:
            return tc.lr * (step + 1) / tc.warmup
        progress = (step - tc.warmup) / max(1, schedule_steps - tc.warmup)
        return tc.lr * (tc.min_lr_frac + (1 - tc.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))

    gen = torch.Generator().manual_seed(tc.seed)
    amp = tc.amp and device.type == "cuda"
    times, losses = [], []
    model.train()
    for step in range(n_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = random_batch(train_data, tc.batch, cfg.max_seq_len, gen, device)
        sync(device)
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, info = model(x)
            lm_loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
            loss = lm_loss + info["aux_loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sync(device)
        times.append(time.perf_counter() - start)
        losses.append(lm_loss.item())
    return model, times, losses, build_s


def step_stats(times: list[float], tokens_per_step: int) -> dict:
    s = sorted(times)
    mean = statistics.fmean(s)
    return {"steps": len(s), "median_step_s": statistics.median(s), "p95_step_s": s[min(len(s) - 1, int(0.95 * len(s)))],
            "mean_step_s": mean, "min_step_s": s[0], "max_step_s": s[-1], "steps_per_s": 1 / mean,
            "tokens_per_s": tokens_per_step / mean, "total_s": sum(s)}


def infer_phase(model, cfg: ModelConfig, data_dir: str, device) -> dict:
    """Triton JIT + CUDA Graph paths of f5eval: first call (compile/capture) and steady state."""
    from .f5eval import b2_latency, decode_speed
    val = load_split(data_dir, "val")
    model.eval()
    if cfg.arch != "dense":
        set_dispatch(model, "triton")
    set_diagnostics(model, False)
    out = {}
    for name, fn in (("b2_latency", lambda: b2_latency(model, val, device)),
                     ("decode", lambda: decode_speed(model, val, device, repeats=1))):
        sync(device)
        t0 = time.perf_counter()
        first = fn()
        sync(device)
        t1 = time.perf_counter()
        second = fn()
        sync(device)
        out[name] = {"first_call_s": t1 - t0, "second_call_s": time.perf_counter() - t1, "result": second}
    return out


def bench(preset: str, data: str, seed: int, warmup: int, steps: int, device, with_infer: bool) -> dict:
    cfg = ModelConfig(**PRESETS[preset])
    tc = TrainConfig(data=data, steps=F5_STEPS, batch=32, lr=1e-3, seed=seed, eval_every=1000, dispatch="bmm")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model, times, losses, build_s = train_steps(cfg, tc, warmup + steps, device, F5_STEPS)
    tokens_per_step = tc.batch * cfg.max_seq_len
    r = {"preset": preset, "arch": cfg.arch, "parameters": count_parameters(model), "seed": seed,
         "batch": tc.batch, "seq_len": cfg.max_seq_len, "tokens_per_step": tokens_per_step, "dispatch": tc.dispatch,
         "build_s": build_s,
         "warmup": {"steps": warmup, "total_s": sum(times[:warmup]), "first_step_s": times[0] if times else None},
         "measured": step_stats(times[warmup:], tokens_per_step),
         "loss_first": losses[0], "loss_last": losses[-1],
         "loss_mean_last50": statistics.fmean(losses[-50:])}
    if device.type == "cuda":
        r["train_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
        r["train_peak_reserved_mb"] = torch.cuda.max_memory_reserved() / 2**20
        torch.cuda.reset_peak_memory_stats()
    sync(device)
    t0 = time.perf_counter()
    default = {"hard_halt": True} if cfg.arch == "amllm" else {}
    e = evaluate(model, load_split(data, "val"), cfg, tc, device, **default)
    sync(device)
    r["eval"] = {"seconds": time.perf_counter() - t0, "windows": tc.eval_windows, "val_loss": e["val_loss"],
                 "median_batch_ms": e["median_batch_ms"]}
    if device.type == "cuda":
        r["eval_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    if with_infer and device.type == "cuda":
        r["infer"] = infer_phase(model, cfg, data, device)
    del model
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--presets", default="moe_f5,amllm_f5,dense_f5")
    p.add_argument("--data", default="data/code_corpus_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--no-infer", action="store_true")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"env": env.collect(), "protocol": {"warmup_steps": a.warmup_steps, "measured_steps": a.steps,
                                             "seed": a.seed, "data": a.data, "schedule_steps": F5_STEPS},
           "results": []}
    if device.type == "cuda":
        p0 = torch.cuda.get_device_properties(0)
        out["gpu"] = {"name": p0.name, "total_mb": p0.total_memory / 2**20, "capability": f"{p0.major}.{p0.minor}"}
    for preset in a.presets.split(","):
        r = bench(preset, a.data, a.seed, a.warmup_steps, a.steps, device, not a.no_infer)
        m = r["measured"]
        print(f"{preset}: median {1000 * m['median_step_s']:.1f} ms  p95 {1000 * m['p95_step_s']:.1f} ms  "
              f"{m['tokens_per_s']:,.0f} tok/s  peak {r.get('train_peak_allocated_mb', 0):.0f} MB  "
              f"loss {r['loss_first']:.3f}->{r['loss_mean_last50']:.3f}", flush=True)
        out["results"].append(r)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
