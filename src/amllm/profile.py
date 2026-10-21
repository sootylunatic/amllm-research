"""Latency, VRAM, profiled FLOPs and active parameters of trained checkpoints (E04/E05 table).

    python -m amllm.profile --runs results/e07_seeds/dense_s42 results/e07_seeds/amllm_s42 --out results/profile.json

Measures a *full-sequence forward pass* (prefill-like), NOT autoregressive generated tok/s.
- latency: CUDA-synchronised wall clock, warmup + repeats, median and p95, with diagnostics on
  (as during evaluation) and off (inference fast path: no routing statistics, balance loss or FLOP report)
- host_syncs_per_forward: CUDA sync-debug count (lower bound)
- profiled_flops: torch.profiler FLOP counters (matmul/linear/bmm/SDPA ops only; elementwise ops
  are not counted) - an independent check of the analytic estimate
- cuda_kernels: kernels launched per forward; cpu/cuda time from the profiler
- active parameters per token: weights a token actually touches (unique) and parameter
  applications (each hop counts again), measured from the route trace on validation data
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from . import env
from .config import ModelConfig
from .data import eval_batches, load_split
from .model import build_model, count_parameters, expert_banks, ffn_flops, set_diagnostics, set_dispatch


def load_run(run_dir: Path, device):
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    cfg = ModelConfig.from_dict(ckpt["config"])
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    return cfg, model


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def latency(model, x, device, warmup: int, repeats: int, **mode) -> dict:
    for _ in range(warmup):
        model(x, **mode)
    times = []
    for _ in range(repeats):
        sync(device)
        s = time.perf_counter()
        model(x, **mode)
        sync(device)
        times.append(1000 * (time.perf_counter() - s))
    times.sort()
    return {"median_ms": times[len(times) // 2], "p95_ms": times[min(len(times) - 1, int(0.95 * len(times)))],
            "min_ms": times[0], "repeats": repeats}


@torch.no_grad()
def profiled(model, x, **mode) -> dict:
    model(x, **mode)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_flops=True) as prof:
        model(x, **mode)
    events = prof.events()
    flops = sum(e.flops for e in events if e.flops)
    kernels = [e for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    cuda_us = sum(e.device_time for e in kernels)
    return {"profiled_matmul_flops_per_token": flops / x.numel(), "cuda_kernels_per_forward": len(kernels),
            "cuda_kernel_time_ms": cuda_us / 1000}


@torch.no_grad()
def count_syncs(model, x, **mode) -> int:
    """Host<->device synchronisations in one forward, via CUDA sync-debug mode (a prototype PyTorch
    feature that may miss some synchronising ops: treat the count as a lower bound)."""
    import warnings
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model(x, **mode)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    return sum("synchroniz" in str(w.message).lower() for w in caught)


@torch.no_grad()
def executed_flops(model, cfg: ModelConfig, x, **mode) -> dict:
    """Routed vs. executed FLOPs: `flops_per_token` counts only the routed expert assignments; with the `bmm`
    dispatch the padding rows are computed too. Executed = routed + (computed rows - routed rows) x FFN cost."""
    banks = expert_banks(model)
    for bank in banks:
        bank.rows_computed = 0
    diag = getattr(model, "diagnostics", True)
    set_diagnostics(model, True)  # the FLOP report needs the routing statistics
    _, info = model(x, **mode)
    set_diagnostics(model, diag)
    routed_flops = float(info["flops_per_token"])
    if not banks:
        return {"routed_flops_per_token": routed_flops, "executed_flops_per_token": routed_flops, "padding_ratio": 1.0}
    rows = sum(bank.rows_computed for bank in banks)
    routed_rows = int(info["expert_counts"].sum())
    extra = (rows - routed_rows) * ffn_flops(cfg.d_model, cfg.expert_hidden) / x.numel()
    return {"routed_flops_per_token": routed_flops, "executed_flops_per_token": routed_flops + extra,
            "padding_ratio": rows / max(1, routed_rows)}


def param_sizes(cfg: ModelConfig) -> dict:
    d = cfg.d_model
    attn = 4 * d * d + 4 * d
    norm = 2 * d
    if cfg.arch == "amllm":
        expert = 2 * d * cfg.expert_hidden + cfg.expert_hidden + d
        router_m = d * cfg.n_modules + cfg.n_modules
        router_e = d * cfg.experts_per_module + cfg.experts_per_module
    else:
        expert = 2 * d * cfg.expert_hidden + cfg.expert_hidden + d
        router_m, router_e = 1 * d + 1, d * cfg.n_experts + cfg.n_experts
    return {"attn": attn, "norm": norm, "expert": expert, "router_m": router_m, "router_e": router_e,
            "head": d * cfg.vocab_size, "embed_row": 2 * d}


@torch.no_grad()
def active_params(cfg: ModelConfig, model, val, device) -> dict:
    """Unique weights touched per token and parameter applications per token (non-embedding + head)."""
    s = param_sizes(cfg)
    d = cfg.d_model
    if cfg.arch == "dense":
        per_layer = s["attn"] + 2 * s["norm"] + 2 * d * cfg.ffn_hidden + cfg.ffn_hidden + d
        n = cfg.n_layers * per_layer + s["head"] + s["norm"]
        return {"unique_active_params_per_token": n, "param_applications_per_token": n,
                "param_applications_per_token_ideal": n}
    x, _, _ = next(eval_batches(val, 64, 64, cfg.max_seq_len, device))
    _, info = model(x, trace=True)
    n_tok = x.numel()
    if cfg.arch == "moe":
        per_layer = s["attn"] + 2 * s["norm"] + s["router_e"] + cfg.moe_topk * s["expert"]
        n = cfg.n_layers * per_layer + s["head"] + s["norm"]
        return {"unique_active_params_per_token": n, "param_applications_per_token": n,
                "param_applications_per_token_ideal": n}
    # amllm: union of experts/modules over hops, per token
    experts = [set() for _ in range(n_tok)]
    modules = [set() for _ in range(n_tok)]
    for hop in info["trace"]:
        for p, ms, es in zip(hop["positions"].tolist(), hop["modules"].tolist(), hop["experts"].tolist()):
            modules[p].update(ms)
            experts[p].update(e for row in es for e in row)
    mean_unique_e = sum(len(e) for e in experts) / n_tok
    mean_unique_m = sum(len(m) for m in modules) / n_tok
    hops = info["hops_per_token"].float().mean().item()
    # Attention: the implementation runs every attention call for *all* tokens while any token is
    # active (halted tokens still serve as keys/values); "ideal" counts only active query tokens.
    attn_impl = info["attention_applications"]
    attn_ideal = info["attention_query_fraction"]
    shared = s["attn"] + 2 * s["norm"] + s["router_m"] + d + 1 + s["head"] + s["norm"]  # + halt head
    unique = shared + mean_unique_m * s["router_e"] + mean_unique_e * s["expert"]
    if cfg.unshared_attention:  # each executed hop beyond the first uses its own attention weights
        unique += (attn_impl - 1) * (s["attn"] + s["norm"])
    routed = hops * (s["router_m"] + cfg.module_topk * s["router_e"] + cfg.module_topk * cfg.expert_topk * s["expert"])
    return {"unique_active_params_per_token": unique,
            "param_applications_per_token": attn_impl * s["attn"] + routed + s["head"],
            "param_applications_per_token_ideal": attn_ideal * s["attn"] + routed + s["head"],
            "attention_applications": attn_impl, "attention_query_applications_ideal": attn_ideal,
            "mean_unique_experts_per_token": mean_unique_e, "mean_unique_modules_per_token": mean_unique_m,
            "mean_hops_on_sample": hops}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data/ts_corpus")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--seq-lens", default="", help="comma-separated context lengths (default: the model's max_seq_len)")
    p.add_argument("--diag", choices=["both", "off"], default="both", help="measure with diagnostics on and off, or off only")
    p.add_argument("--default-mode-only", action="store_true",
                   help="only the default inference mode (hard halting for adaptive AM-LLM, full depth otherwise)")
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = load_split(a.data, "val")
    results = {"env": env.collect(), "WARNING": "full-sequence forward latency, NOT generated tokens/s", "runs": {}}
    for run in a.runs:
        run_dir = Path(run)
        cfg, model = load_run(run_dir, device)
        weights_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 2**20
        modes = {"default": {}} if cfg.arch != "amllm" else (
            {"hard": {"hard_halt": True}, "soft": {"hard_halt": False}} if cfg.halt_rule != "none" else {"fixed": {"hard_halt": True}})
        if a.default_mode_only:
            modes = dict(list(modes.items())[:1])
        r = {"arch": cfg.arch, "config": cfg.to_dict(), "parameters": count_parameters(model),
             "weights_mb_fp32": round(weights_mb, 2), **active_params(cfg, model, val, device), "measurements": {}}
        for dispatch in (("loop", "bmm") if cfg.arch != "dense" else ("dense",)):
            if cfg.arch != "dense":
                set_dispatch(model, dispatch)
            seq_lens = [int(t) for t in a.seq_lens.split(",") if t] or [cfg.max_seq_len]
            diags = (True, False) if a.diag == "both" else (False,)
            for (mode_name, mode), diag, seq_len in ((m, dg, t) for m in modes.items() for dg in diags for t in seq_lens):
                set_diagnostics(model, diag)
                for batch in (1, 32):
                    # real validation text, not random bytes: halting depth depends on content
                    x = next(eval_batches(val, batch, batch, min(seq_len, cfg.max_seq_len), device))[0]
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        base = torch.cuda.memory_allocated()
                    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        lat = latency(model, x, device, a.warmup, a.repeats, **mode)
                        prof = profiled(model, x, **mode) if batch == 32 else {}
                        if batch == 32:
                            prof.update(executed_flops(model, cfg, x, **mode))
                        syncs = count_syncs(model, x, **mode) if device.type == "cuda" else None
                    peak = (torch.cuda.max_memory_allocated() - base) / 2**20 if device.type == "cuda" else None
                    key = f"{dispatch}/{mode_name}/b{batch}/t{x.shape[1]}/diag_{'on' if diag else 'off'}"
                    r["measurements"][key] = {**lat, **prof, "host_syncs_per_forward": syncs, "peak_activation_vram_mb": round(peak, 1) if peak is not None else None,
                                              "peak_total_vram_mb": round(peak + weights_mb, 1) if peak is not None else None}
                    print(f"{run_dir.name:22s} {key:36s} median={lat['median_ms']:.2f}ms p95={lat['p95_ms']:.2f}ms "
                          + (f"kernels={prof['cuda_kernels_per_forward']} prof_MFLOP/tok={prof['profiled_matmul_flops_per_token'] / 1e6:.3f}" if prof else ""), flush=True)
        set_diagnostics(model, True)
        results["runs"][f"{run_dir.parent.name}/{run_dir.name}"] = r
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
