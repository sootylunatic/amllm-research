"""E02: routing / causality / skipping invariants on many random inputs, plus a route trace.

    python -m amllm.invariants --checkpoint results/e01_smoke/v01_topology_ts/checkpoint.pt --out results/e02_invariants
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import env
from .config import ModelConfig
from .model import AdaptiveModularLM, build_model, expert_banks, set_dispatch

ARCH_VARIANTS = {
    "amllm_survival": dict(),
    "amllm_step": dict(halt_rule="step", halt_threshold=0.65),
    "amllm_v01_topology": dict(attn_per_hop=False, halt_rule="step", halt_threshold=0.65),
    "amllm_fixed": dict(halt_rule="none"),
    "amllm_unshared_attention": dict(unshared_attention=True),
    "moe": dict(arch="moe"),
    "dense": dict(arch="dense"),
}


def small_cfg(**kw) -> ModelConfig:
    base = dict(max_seq_len=48, d_model=32, n_heads=4, n_modules=6, experts_per_module=4, module_topk=2,
                expert_topk=2, n_hops=4, expert_hidden=32, n_layers=3, ffn_hidden=64, n_experts=6, moe_topk=2)
    base.update(kw)
    return ModelConfig(**base)


@torch.no_grad()
def check_variant(name: str, cfg: ModelConfig, trials: int, device) -> dict:
    fails = {"causality": 0, "determinism": 0, "count_consistency": 0, "rows_equal_routes": 0, "finite": 0}
    for trial in range(trials):
        torch.manual_seed(trial)
        # CUDA checks run in fp64: bf16/fp32 GEMMs may round a row differently depending on how many
        # rows share an expert, which is batch-composition numerics, not information flow
        model = build_model(cfg).to(device, torch.float64 if device.type == "cuda" else torch.float32).eval()
        if isinstance(model, AdaptiveModularLM) and cfg.halt_rule != "none":
            model.halt_head.weight.normal_(0, 1.0)  # make halting token-dependent
            model.halt_head.bias.uniform_(-1, 1)
        b, t = 2, int(torch.randint(8, cfg.max_seq_len + 1, ()))
        x = torch.randint(0, 256, (b, t), device=device)
        cut = int(torch.randint(1, t, ()))
        x2 = x.clone()
        x2[:, cut:] = torch.randint(0, 256, (b, t - cut), device=device)
        for bank in expert_banks(model):
            bank.rows_computed = 0
        l1, i1 = model(x)
        rows = sum(bank.rows_computed for bank in expert_banks(model))
        l1b, i1b = model(x)
        l2, _ = model(x2)
        if not torch.allclose(l1[:, :cut], l2[:, :cut], atol=1e-5 if device.type == "cpu" else 1e-12):
            fails["causality"] += 1
        if not (torch.equal(l1, l1b) and torch.equal(i1["expert_counts"], i1b["expert_counts"])):
            fails["determinism"] += 1
        if not torch.isfinite(l1).all():
            fails["finite"] += 1
        ec = i1["expert_counts"].sum().item()
        if cfg.arch == "amllm":
            token_hops = i1["hops_per_token"].sum().item()
            ok = (i1["module_counts"].sum().item() == token_hops * cfg.module_topk
                  and ec == token_hops * cfg.module_topk * cfg.expert_topk)
        elif cfg.arch == "moe":
            ok = ec == b * t * cfg.n_layers * cfg.moe_topk
        else:
            ok = True
        fails["count_consistency"] += int(not ok)
        if cfg.arch != "dense" and rows != ec:
            fails["rows_equal_routes"] += 1
    return {"variant": name, "trials": trials, "failures": fails, "passed": all(v == 0 for v in fails.values())}


@torch.no_grad()
def skip_timing(device, repeats: int = 30) -> dict:
    """Does early exit really remove work? Same weights, halting forced to 1 hop vs full depth."""
    cfg = ModelConfig(max_seq_len=256, d_model=128, n_heads=4, n_modules=8, experts_per_module=4,
                      module_topk=2, expert_topk=2, expert_hidden=128, n_hops=4)
    torch.manual_seed(0)
    model = AdaptiveModularLM(cfg).to(device).eval()
    x = torch.randint(0, 256, (32, 256), device=device)
    out = {}
    for dispatch in ("loop", "bmm"):
        set_dispatch(model, dispatch)
        for label, bias in (("forced_1_hop", 20.0), ("forced_4_hops", -20.0)):
            model.halt_head.bias.fill_(bias)
            model.bank.experts.rows_computed = 0
            _, info = model(x)
            rows = model.bank.experts.rows_computed
            times = []
            for i in range(repeats + 3):
                torch.cuda.synchronize() if device.type == "cuda" else None
                s = time.perf_counter()
                model(x)
                torch.cuda.synchronize() if device.type == "cuda" else None
                if i >= 3:
                    times.append(1000 * (time.perf_counter() - s))
            times.sort()
            out[f"{dispatch}/{label}"] = {"mean_hops": info["mean_hops"].item(),
                                          "expert_ffn_rows_computed": rows,
                                          "routed_assignments": int(info["expert_counts"].sum()),
                                          "median_ms": times[len(times) // 2], "p95_ms": times[int(0.95 * len(times))]}
    return out


@torch.no_grad()
def route_trace(checkpoint: str, device, text: bytes) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    ids = torch.tensor(list(text[:cfg.max_seq_len]), device=device)[None]
    _, info = model(ids, trace=True)
    tokens = []
    for pos in range(ids.shape[1]):
        hops = []
        for h, hop in enumerate(info["trace"]):
            where = (hop["positions"] == pos).nonzero().flatten()
            if where.numel():
                j = where.item()
                hops.append({"hop": h, "modules": hop["modules"][j].tolist(), "experts": hop["experts"][j].flatten().tolist()})
        tokens.append({"pos": pos, "byte": chr(ids[0, pos].item()), "hops": hops})
    return {"checkpoint": checkpoint, "config": cfg.to_dict(), "mode": "hard (default inference)", "tokens": tokens}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="results/e02_invariants")
    p.add_argument("--trials", type=int, default=50)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
    results = [{**check_variant(n, small_cfg(**kw), a.trials, dev), "device": f"{dev.type} ({'fp64' if dev.type == 'cuda' else 'fp32'})"}
               for dev in devices for n, kw in ARCH_VARIANTS.items()]
    for r in results:
        print(f"{r['device']:12s} {r['variant']:26s} passed={r['passed']} {r['failures']}")
    timing = skip_timing(device)
    for k, v in timing.items():
        print(f"{k:24s} hops={v['mean_hops']:.2f} rows={v['expert_ffn_rows_computed']:7d} "
              f"routed={v['routed_assignments']:7d} median={v['median_ms']:.2f}ms p95={v['p95_ms']:.2f}ms")
    (out / "invariants.json").write_text(json.dumps({"env": env.collect(), "random_input_checks": results,
                                                     "skip_timing_batch32_ctx256": timing}, indent=2))
    sample = b"export function sum(xs: number[]): number {\n  return xs.reduce((a, b) => a + b, 0);\n}\n"
    (out / "route_trace.json").write_text(json.dumps(route_trace(a.checkpoint, device, sample), indent=1))
    print(f"wrote {out}/invariants.json and route_trace.json")


if __name__ == "__main__":
    main()
