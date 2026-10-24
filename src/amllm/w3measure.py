"""W3: practical measurement of trained checkpoints (validation split only; execution code, not part of train_code_sha256).

    python -m amllm.w3measure --runs results/round_v3_w2/FDC1/amllm_f5_s42 ... --out results/round_v3_w3/measure/<gpu>.json

Per checkpoint, in the model's default evaluation mode (fixed depth, or hard survival halting at its threshold):
  position_nll   NLL by position bucket inside the context window (does an advantage hold late in the window?)
  work           routed / executed FLOP per token and computed expert rows under each dispatch: `bmm` (training and
                 evaluation dispatch) pads every expert group to the largest one; `loop` and `triton` (CUDA, inference)
                 compute the routed rows only; for a halting model also the bucketed hard path's packed-row padding
  latency        full-sequence forward (batch 8 x 512): eager, and as a CUDA Graph where the path is capturable
  decode         KV-cache generation per batch size: prefill time (TTFT for batch 1), decode ms/token p50/p95 and tok/s
                 (eager; CUDA Graph where capturable), cache and peak memory
  vram           weights (fp32 / bf16), KV-cache bytes per batch and context, measured peak memory per case (CUDA)
  hotset         expert usage concentration per bank/layer and an LRU expert-cache simulation (hit rate, bytes moved)
  generation     free generation from validation prompts, greedy and sampled: repeated 4-gram rate, distinct-1/2 and
                 the share of sequences that end in a loop (a period <= 32 tokens repeated >= 3 times)
Times from different GPUs or code versions are never mixed in one output file.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.nn import functional as F

from . import env
from .data import eval_batches, load_split
from .fastpath import GraphRunner, bucketed_hard_forward, static_hard_forward
from .generate import GraphDecoder, Generator
from .model import AdaptiveModularLM, expert_banks, set_diagnostics, set_dispatch
from .profile import load_run

PROVENANCE = ("visible_device", "gpus", "driver", "torch", "cuda", "git_commit", "source_sha256", "train_code_sha256")


def speed_dispatch(device) -> str:
    """Padding-free dispatch used for every timing: the Triton grouped GEMM on CUDA (as in the F5 evaluation), the
    per-expert loop on CPU."""
    return "triton" if device.type == "cuda" else "loop"


def capturable(model) -> bool:
    """CUDA Graphs need static shapes: adaptive K (expert_topp) dispatches a data-dependent number of rows, so its
    paths are not captured at all (a failed capture can leave the CUDA context in an error state)."""
    return model.cfg.expert_topp == 0


def default_mode(model) -> dict:
    cfg = model.cfg
    return {"hard_halt": True} if isinstance(model, AdaptiveModularLM) and cfg.halt_rule != "none" else {}


def sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def autocast(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)


def pct(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


# ------------------------------------------------------------------------------------------------- quality by position
@torch.no_grad()
def position_nll(model, val, device, windows: int = 128, batch: int = 8, bucket: int = 64) -> dict:
    block = model.cfg.max_seq_len
    n_b = -(-block // bucket)
    sums, counts = torch.zeros(n_b, dtype=torch.float64), torch.zeros(n_b, dtype=torch.float64)
    for x, y, _ in eval_batches(val, windows, batch, block, device):
        with autocast(device):
            logits = model(x, **default_mode(model))[0]
        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").view(y.shape)
        per_pos = nll.sum(0).double().cpu()
        for i in range(n_b):
            sums[i] += per_pos[i * bucket:(i + 1) * bucket].sum()
            counts[i] += y.shape[0] * per_pos[i * bucket:(i + 1) * bucket].numel()
    return {"bucket_tokens": bucket, "windows": windows,
            "nll_by_bucket": [round(float(s / c), 4) for s, c in zip(sums, counts)],
            "nll_mean": round(float(sums.sum() / counts.sum()), 4)}


# --------------------------------------------------------------------------------------------------- executed work
@torch.no_grad()
def work(model, val, device, batch: int = 8) -> dict:
    x = next(eval_batches(val, batch, batch, model.cfg.max_seq_len, device))[0]
    set_diagnostics(model, True)
    out = {}
    modes = ["bmm", "loop", "triton"] if device.type == "cuda" else ["bmm", "loop"]
    for disp in modes if model.cfg.arch != "dense" else ["dense"]:
        if disp != "dense":
            set_dispatch(model, disp)
        with autocast(device):
            info = model(x, **default_mode(model))[1]
        out[disp] = {"flops_per_token_routed": float(info["flops_per_token"]),
                     "flops_per_token_executed": float(info["flops_per_token_executed"]),
                     "flops_per_token_ideal_attention": float(info["flops_per_token_ideal"]),
                     "expert_rows_per_token": float(info["expert_rows_per_token"]),
                     "attention_query_fraction": float(info["attention_query_fraction"]),
                     "mean_hops": float(info["mean_hops"])}
    if isinstance(model, AdaptiveModularLM) and model.cfg.halt_rule == "survival" and model.cfg.attn_per_hop:
        with autocast(device):
            _, st = bucketed_hard_forward(model, x)
        out["bucketed_hard"] = {"packed_rows": st["packed_rows"], "active_rows": st["active_rows"],
                                "padding_ratio": st["packed_rows"] / max(1, st["active_rows"]),
                                "mean_hops": st["mean_hops"]}
    set_diagnostics(model, False)
    return out


# ---------------------------------------------------------------------------------------------------------- latency
@torch.no_grad()
def latency(model, val, device, batch: int = 8, warmup: int = 5, repeats: int = 20) -> dict:
    x = next(eval_batches(val, batch, batch, model.cfg.max_seq_len, device))[0]

    static = isinstance(model, AdaptiveModularLM) and model.cfg.halt_rule in ("survival", "none")

    def fwd(ids):  # AM-LLM: the sync-free static path (hard halting or fixed depth); Dense/MoE: the ordinary forward
        with autocast(device):
            return static_hard_forward(model, ids)[0] if static else model(ids, **default_mode(model))[0]

    def timed(fn) -> dict:
        for _ in range(warmup):
            fn(x)
        times = []
        for _ in range(repeats):
            sync(device)
            s = time.perf_counter()
            fn(x)
            sync(device)
            times.append(1000 * (time.perf_counter() - s))
        return {"median_ms": statistics.median(times), "p95_ms": pct(times, 0.95)}

    out = {"batch": batch, "tokens": x.numel(), "eager": timed(fwd)}
    if device.type == "cuda" and not capturable(model):
        out["graph"] = {"not_capturable": "adaptive K: data-dependent number of expert rows"}
    elif device.type == "cuda":
        try:
            out["graph"] = timed(GraphRunner(fwd, x))
        except Exception as e:  # e.g. adaptive K uses a data-dependent number of rows: not capturable
            out["graph"] = {"not_capturable": type(e).__name__}
    return out


# ----------------------------------------------------------------------------------------------------------- decode
@torch.no_grad()
def decode(model, val, device, batches=(1, 8, 32), prompt: int = 128, new: int = 128, repeats: int = 3) -> dict:
    out = {}
    cache_dtype = torch.bfloat16 if device.type == "cuda" else None
    for b in batches:
        ids = next(eval_batches(val, b, b, prompt, device))[0]
        gen = Generator(model, batch=b, max_len=prompt + new, cache_dtype=cache_dtype)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        prefill_ms, step_ms = [], []
        for _ in range(repeats):
            sync(device)
            s = time.perf_counter()
            with autocast(device):
                logits = gen.prefill(ids)
            sync(device)
            prefill_ms.append(1000 * (time.perf_counter() - s))
            token = logits[:, -1].argmax(-1)
            for _ in range(new):
                sync(device)
                s = time.perf_counter()
                with autocast(device):
                    token = gen.decode_step(token).argmax(-1)
                sync(device)
                step_ms.append(1000 * (time.perf_counter() - s))
        r = {"prefill_ms_median": statistics.median(prefill_ms), "decode_ms_per_token_p50": pct(step_ms, 0.5),
             "decode_ms_per_token_p95": pct(step_ms, 0.95), "decode_tokens_per_s_eager": b * 1000 / statistics.median(step_ms),
             "cache_mb": gen.cache_bytes() / 2**20}
        if b == 1:
            r["ttft_ms"] = r["prefill_ms_median"]
        if device.type == "cuda" and not capturable(model):
            r["decode_tokens_per_s_graph"] = None
            r["graph_not_capturable"] = "adaptive K: data-dependent number of expert rows"
        elif device.type == "cuda":
            try:
                with autocast(device):
                    logits = gen.prefill(ids)
                token = logits[:, -1].argmax(-1)
                dec = GraphDecoder(gen, token)
                times = []
                for _ in range(new):
                    sync(device)
                    s = time.perf_counter()
                    token = dec(token).argmax(-1)
                    sync(device)
                    times.append(1000 * (time.perf_counter() - s))
                r["decode_tokens_per_s_graph"] = b * 1000 / statistics.median(times)
                r["decode_ms_per_token_p50_graph"] = pct(times, 0.5)
                r["decode_ms_per_token_p95_graph"] = pct(times, 0.95)
            except Exception as e:
                r["decode_tokens_per_s_graph"] = None
                r["graph_not_capturable"] = type(e).__name__
            r["peak_memory_mb"] = torch.cuda.max_memory_allocated() / 2**20
        out[f"b{b}"] = r
        del gen
    return out


# ------------------------------------------------------------------------------------------------------------- vram
def kv_cache_bytes(model, batch: int, context: int, bytes_per_value: int = 2) -> int:
    cfg = model.cfg
    units = cfg.n_hops if isinstance(model, AdaptiveModularLM) else cfg.n_layers
    return 2 * units * batch * context * cfg.d_model * bytes_per_value


@torch.no_grad()
def vram(model, val, device, batches=(1, 2, 4, 8), contexts=(128, 256, 512)) -> dict:
    n = sum(p.numel() for p in model.parameters())
    expert = sum(b.w1.numel() + b.b1.numel() + b.w2.numel() + b.b2.numel() for b in expert_banks(model))
    out = {"parameters": n, "expert_parameters": expert, "weights_mb_fp32": 4 * n / 2**20, "weights_mb_bf16": 2 * n / 2**20,
           "expert_weights_mb_bf16": 2 * expert / 2**20, "cases": {}}
    for b in batches:
        for c in contexts:
            case = {"kv_cache_mb_bf16": kv_cache_bytes(model, b, c) / 2**20}
            if device.type == "cuda":
                ids = next(eval_batches(val, b, b, c - 8, device))[0]
                torch.cuda.empty_cache()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                gen = Generator(model, batch=b, max_len=c, cache_dtype=torch.bfloat16)
                with autocast(device):
                    token = gen.prefill(ids)[:, -1].argmax(-1)
                    for _ in range(8):
                        token = gen.decode_step(token).argmax(-1)
                sync(device)
                case["peak_above_weights_mb"] = (torch.cuda.max_memory_allocated() - base) / 2**20
                del gen
            out["cases"][f"b{b}_c{c}"] = case
    return out


# ----------------------------------------------------------------------------------------------------------- hotset
def lru_hit_rate(stream: list[list[int]], capacity: int) -> tuple[float, int]:
    """stream: per token, the expert ids it uses. Returns (hit rate, misses) of an LRU cache with `capacity` experts."""
    cache: OrderedDict[int, None] = OrderedDict()
    hits = misses = 0
    for experts in stream:
        for e in experts:
            if e in cache:
                hits += 1
                cache.move_to_end(e)
            else:
                misses += 1
                cache[e] = None
                if len(cache) > capacity:
                    cache.popitem(last=False)
    total = hits + misses
    return (hits / total if total else 1.0), misses


def concentration(counts: torch.Tensor, fracs=(0.1, 0.25, 0.5)) -> dict:
    c = counts.double().sort(descending=True).values
    total = c.sum().clamp(min=1)
    return {f"top{int(100 * f)}pct_share": round(float(c[:max(1, round(f * len(c)))].sum() / total), 4) for f in fracs}


@torch.no_grad()
def hotset(model, val, device, windows: int = 32, batch: int = 8, cache_fracs=(0.25, 0.5, 0.75)) -> dict:
    cfg = model.cfg
    if cfg.arch == "dense":
        return {"skipped": "dense model has no experts"}
    is_am = isinstance(model, AdaptiveModularLM)
    n_units = cfg.n_hops if is_am else cfg.n_layers
    group_of = (lambda u: u * cfg.bank_groups // cfg.n_hops) if is_am else (lambda u: u)
    n_groups = cfg.bank_groups if is_am else cfg.n_layers
    per_group = cfg.n_modules * cfg.experts_per_module if is_am else cfg.n_experts
    streams = [[] for _ in range(n_groups)]   # per bank/layer: the experts of every (token, unit) visit, in token order
    counts = torch.zeros(n_groups, per_group, dtype=torch.long)
    for x, _, _ in eval_batches(val, windows, batch, cfg.max_seq_len, device):
        with autocast(device):  # the same bf16 path as every other measurement: routing is analysed as it runs
            info = model(x, trace=True, **default_mode(model))[1]
        visits = {}                           # (token position, unit) -> experts
        for u, rec in enumerate(info["trace"][:n_units]):
            ex = rec["experts"].reshape(len(rec["positions"]), -1)
            for p, row in zip(rec["positions"].tolist(), ex.tolist()):
                ids = [e for e in row if e >= 0]  # -1 = route dropped by adaptive K
                visits[(p, u)] = ids
                counts[group_of(u)] += torch.bincount(torch.tensor(ids, dtype=torch.long), minlength=per_group)
        for (p, u) in sorted(visits):         # token order, then unit order
            streams[group_of(u)].append(visits[(p, u)])
    expert_bytes_bf16 = 2 * (2 * cfg.d_model * cfg.expert_hidden + cfg.expert_hidden + cfg.d_model)
    n_tok = windows * cfg.max_seq_len
    groups = []
    for g in range(n_groups):
        sim = {}
        for f in cache_fracs:
            cap = max(1, round(f * per_group))
            hit, misses = lru_hit_rate(streams[g], cap)
            sim[f"cache_{int(100 * f)}pct"] = {"experts": cap, "hit_rate": round(hit, 4),
                                               "bytes_moved_per_token": misses * expert_bytes_bf16 / n_tok}
        groups.append({"experts": per_group, **concentration(counts[g]), "lru": sim})
    return {"windows": windows, "expert_bytes_bf16": expert_bytes_bf16, "groups": groups}


# ------------------------------------------------------------------------------------------------------- generation
def loop_metrics(ids: list[int], max_period: int = 32, repeats: int = 3) -> dict:
    def ngrams(n):
        return [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    g4, g1, g2 = ngrams(4), ngrams(1), ngrams(2)
    rep4 = 1 - len(set(g4)) / len(g4) if g4 else 0.0
    loop = False
    for p in range(1, max_period + 1):
        tail = ids[-p * repeats:]
        if len(tail) == p * repeats and all(tail[i] == tail[i % p] for i in range(len(tail))):
            loop = True
            break
    return {"repeated_4gram_rate": rep4, "distinct_1": len(set(g1)) / max(1, len(g1)),
            "distinct_2": len(set(g2)) / max(1, len(g2)), "ends_in_loop": loop}


@torch.no_grad()
def sample_continuations(model, prompts: torch.Tensor, new: int, device, temperature: float | None, top_p: float = 0.95,
                         seed: int = 0) -> torch.Tensor:
    gen = Generator(model, batch=prompts.shape[0], max_len=prompts.shape[1] + new,
                    cache_dtype=torch.bfloat16 if device.type == "cuda" else None)
    rng = torch.Generator(device=device).manual_seed(seed)

    def pick(logits):
        if temperature is None:
            return logits.argmax(-1)
        probs = (logits.float() / temperature).softmax(-1)
        sp, si = probs.sort(-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p
        sp = sp * keep
        choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=rng).squeeze(-1)
        return si.gather(-1, choice[:, None]).squeeze(-1)

    with autocast(device):
        token = pick(gen.prefill(prompts)[:, -1])
    out = []
    for _ in range(new):
        out.append(token)
        with autocast(device):
            token = pick(gen.decode_step(token))
    return torch.stack(out, 1)


@torch.no_grad()
def generation(model, val, device, n_prompts: int = 32, prompt: int = 128, new: int | None = None) -> dict:
    new = new or model.cfg.max_seq_len - prompt
    prompts = next(eval_batches(val, n_prompts, n_prompts, prompt, device))[0]
    out = {"prompts": n_prompts, "prompt_tokens": prompt, "new_tokens": new}
    for name, temp in (("greedy", None), ("sampled_t0.8_p0.95", 0.8)):
        seqs = sample_continuations(model, prompts, new, device, temp).tolist()
        ms = [loop_metrics(s) for s in seqs]
        out[name] = {"repeated_4gram_rate": round(statistics.fmean(m["repeated_4gram_rate"] for m in ms), 4),
                     "distinct_1": round(statistics.fmean(m["distinct_1"] for m in ms), 4),
                     "distinct_2": round(statistics.fmean(m["distinct_2"] for m in ms), 4),
                     "loop_share": round(sum(m["ends_in_loop"] for m in ms) / len(ms), 4)}
    return out


# ------------------------------------------------------------------------------------------------------------- main
PARTS = ("position_nll", "work", "latency", "decode", "vram", "hotset", "generation")


def measure(model, val, device, parts=PARTS) -> dict:
    set_diagnostics(model, False)
    if model.cfg.arch != "dense":
        set_dispatch(model, speed_dispatch(device))
    fns = {"position_nll": position_nll, "work": work, "latency": latency, "decode": decode, "vram": vram,
           "hotset": hotset, "generation": generation}
    r = {}
    for part in parts:
        if part in ("work", "hotset"):
            set_diagnostics(model, True)
        r[part] = fns[part](model, val, device)
        set_diagnostics(model, False)
        if model.cfg.arch != "dense":
            set_dispatch(model, speed_dispatch(device))
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data/code_corpus_v2")
    p.add_argument("--parts", nargs="*", default=list(PARTS))
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = load_split(a.data, "val")  # validation only: the test split and the holdout are never loaded
    out_path = Path(a.out)
    out = json.loads(out_path.read_text()) if out_path.exists() else {"runs": {}}
    here = env.collect()
    prov = {k: here.get(k) for k in PROVENANCE}
    if out["runs"] and out.get("provenance") != prov:
        raise SystemExit(f"{out_path} holds measurements from another environment; use a new --out")
    out["provenance"], out["env"] = prov, here
    for run in a.runs:
        run_dir = Path(run)
        key = f"{run_dir.parent.name}/{run_dir.name}"
        done = out["runs"].get(key, {})
        missing = [part for part in a.parts if part not in done]  # a resumed run measures only what is still missing
        if not missing:
            continue
        cfg, model = load_run(run_dir, device)
        start = time.perf_counter()
        r = {**done, "arch": cfg.arch, "parameters": sum(q.numel() for q in model.parameters()),
             **measure(model, val, device, missing)}
        r["seconds"] = done.get("seconds", 0.0) + time.perf_counter() - start
        out["runs"][key] = r
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=1))
        print(f"{key}: {r['seconds']:.0f} s", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
