"""F3 measurement: the same checkpoints on three backends (results/phase2/preregistration.md, F3).

    python -m amllm.f3bench --runs <run dirs> --out results/phase2/F3/f3bench.json

  B0  bmm dispatch, eager, reference forward (AM-LLM: hard halting with per-hop nonzero)
  B1  triton dispatch, eager, reference forward
  B1s triton dispatch, eager, static masked hard execution (no host sync; informative)
  B2  triton dispatch, static execution, captured in one CUDA Graph
All in bf16 with diagnostics off, on real validation text. Per backend: median/p95 latency (batch 32 x 256, batch 1 x 256,
batch 32 x 64), host syncs and CUDA kernels per forward (eager paths), executed expert rows vs. routed assignments,
and the validation loss of the backend's own forward on the standard 512 validation windows.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.profiler import ProfilerActivity, profile

from . import env
from .data import eval_batches, load_split
from .fastpath import GraphRunner, static_forward
from .model import AdaptiveModularLM, expert_banks, set_diagnostics, set_dispatch
from .profile import load_run

LOADS = ((32, 256), (1, 256), (32, 64))


def timed(fn, x, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn(x)
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        s = time.perf_counter()
        fn(x)
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - s))
    times.sort()
    return {"median_ms": times[len(times) // 2], "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))]}


def syncs_and_kernels(fn, x) -> dict:
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn(x)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(x)
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return {"host_syncs": sum("synchroniz" in str(w.message).lower() for w in caught),
            "cuda_kernels": len(kernels), "memcpy_dtoh": sum("DtoH" in e.name for e in kernels)}


def val_loss(fn, val, max_seq_len: int, device) -> float:
    total, n = 0.0, 0
    for x, y, _ in eval_batches(val, 512, 64, max_seq_len, device):
        logits = fn(x)
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
        n += y.numel()
    return total / n


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data/ts_corpus")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    a = p.parse_args()
    device = torch.device("cuda")
    val = load_split(a.data, "val")
    out = {"env": env.collect(), "NOTE": "full-sequence forward latency, NOT generated tokens/s", "runs": {}}
    for run in a.runs:
        run_dir = Path(run)
        cfg, model = load_run(run_dir, device)
        set_diagnostics(model, False)
        is_amllm = isinstance(model, AdaptiveModularLM)

        def reference(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return model(x, hard_halt=True)[0] if is_amllm else model(x)[0]

        def static(x):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return static_forward(model, x)

        r = {"arch": cfg.arch, "backends": {}}
        for name, dispatch, fn, graph in (("B0_bmm", "bmm", reference, False), ("B1_triton", "triton", reference, False),
                                          ("B1s_triton_static", "triton", static, False),
                                          ("B2_triton_static_graph", "triton", static, True)):
            set_dispatch(model, dispatch)
            b = {"loads": {}}
            for batch, seq in LOADS:
                x = next(eval_batches(val, batch, batch, seq, device))[0]
                runner = GraphRunner(fn, x) if graph else fn
                b["loads"][f"b{batch}/t{seq}"] = timed(runner, x, a.warmup, a.repeats)
                if batch == 32 and seq == 256:
                    for bank in expert_banks(model):
                        bank.rows_computed = 0
                    fn(x)
                    rows = sum(bank.rows_computed for bank in expert_banks(model))
                    b["expert_rows_per_token"] = rows / x.numel()
                    if not graph:
                        b.update(syncs_and_kernels(fn, x))
                del runner
            # quality of this backend's own forward (graph replay = static eager bitwise, tested)
            b["val_loss"] = val_loss(fn, val, cfg.max_seq_len, device)
            r["backends"][name] = b
            l = b["loads"]["b32/t256"]
            print(f"{run_dir.parent.name}/{run_dir.name:24s} {name:24s} b32x256 median={l['median_ms']:.2f} p95={l['p95_ms']:.2f} "
                  f"b1={b['loads']['b1/t256']['median_ms']:.2f} t64={b['loads']['b32/t64']['median_ms']:.2f} "
                  f"syncs={b.get('host_syncs', '-')} kernels={b.get('cuda_kernels', '-')} rows/tok={b['expert_rows_per_token']:.2f} "
                  f"val={b['val_loss']:.4f}", flush=True)
        out["runs"][f"{run_dir.parent.name}/{run_dir.name}"] = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
