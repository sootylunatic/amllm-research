"""F2 gate measurement: bucketed hard execution vs. full-depth execution of the same checkpoint.

    python -m amllm.f2bench --runs results/phase2/F2/<run> ... --out results/phase2/F2/f2bench.json

For each checkpoint (`bmm` dispatch, diagnostics off, bf16, real validation text): median/p95 latency of
  - full depth (all hops, no halting)        -> reference of the F2 gate
  - reference hard halting (model.forward)   -> the current execution path
  - bucketed hard halting (fastpath)         -> F2-1
plus output agreement (bucketed vs. reference hard) and bucket/padding statistics.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from . import env
from .data import eval_batches, load_split
from .fastpath import bucketed_hard_forward
from .model import set_diagnostics, set_dispatch
from .profile import load_run


def timed(fn, device, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize(device)
        s = time.perf_counter()
        fn()
        torch.cuda.synchronize(device)
        times.append(1000 * (time.perf_counter() - s))
    times.sort()
    return {"median_ms": times[len(times) // 2], "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))]}


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
        set_dispatch(model, "bmm")
        set_diagnostics(model, False)
        r = {}
        for batch, seq in ((32, 256), (1, 256), (32, 64)):
            x = next(eval_batches(val, batch, batch, seq, device))[0]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ref, info = model(x, hard_halt=True)
                fast, stats = bucketed_hard_forward(model, x)
                agree = {"max_abs_logit_diff": float((fast.float() - ref.float()).abs().max()),
                         "argmax_agreement": float((fast.argmax(-1) == ref.argmax(-1)).float().mean()),
                         "hops_identical": bool(torch.equal(stats["hops_per_token"], info["hops_per_token"]))}
                lat = {"full_depth": timed(lambda: model(x, hard_halt=True, halt_rule="none"), device, a.warmup, a.repeats),
                       "reference_hard": timed(lambda: model(x, hard_halt=True), device, a.warmup, a.repeats),
                       "bucketed_hard": timed(lambda: bucketed_hard_forward(model, x), device, a.warmup, a.repeats)}
            key = f"b{batch}/t{seq}"
            r[key] = {**lat, "agreement": agree, "mean_hops": stats["mean_hops"],
                      "packed_rows": stats["packed_rows"], "active_rows": stats["active_rows"],
                      "capacities": [h["capacity"] for h in stats["hops"]],
                      "speedup_vs_full_depth": 1 - lat["bucketed_hard"]["median_ms"] / lat["full_depth"]["median_ms"]}
            print(f"{run_dir.name:28s} {key:9s} full={lat['full_depth']['median_ms']:.2f} ref_hard={lat['reference_hard']['median_ms']:.2f} "
                  f"bucketed={lat['bucketed_hard']['median_ms']:.2f} ms  saving={r[key]['speedup_vs_full_depth']:+.1%} "
                  f"hops={stats['mean_hops']:.2f} caps={r[key]['capacities']} maxdiff={agree['max_abs_logit_diff']:.3g}", flush=True)
        out["runs"][f"{run_dir.parent.name}/{run_dir.name}"] = r
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
