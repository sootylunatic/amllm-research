"""Train one model on the byte corpus, evaluate it on the held-out split, write a run directory.

    python -m amllm.train --arch amllm --data data/ts_corpus --steps 3000 --run-dir results/e04/amllm_s42

Run directory: config.json, env.json, train.log, checkpoint.pt, report.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.nn import functional as F

from . import env
from .config import ModelConfig
from .data import eval_batches, load_categories, load_split, random_batch, split_id
from .model import build_model, count_parameters, set_dispatch, set_random_routing

AMLLM_EVAL_MODES = [
    {"name": "soft", "hard_halt": False},
    *({"name": f"hard_surv_{t}", "hard_halt": True, "halt_rule": "survival", "halt_threshold": t}
      for t in (0.3, 0.5, 0.65, 0.8)),
    *({"name": f"hard_step_{t}", "hard_halt": True, "halt_rule": "step", "halt_threshold": t}
      for t in (0.5, 0.65, 0.8)),
    # K1 diagnostic: same trained weights, route choices replaced by uniform random ones
    {"name": "soft_random_routes", "hard_halt": False, "random_routing": True},
]


def eval_modes(cfg: ModelConfig) -> list[dict]:
    if cfg.arch == "dense":
        return [{"name": "fixed"}]
    if cfg.arch == "moe":
        return [{"name": "fixed"}, {"name": "random_routes", "random_routing": True}]
    if cfg.halt_rule == "none":
        return [{"name": "fixed", "hard_halt": True, "halt_rule": "none"},
                {"name": "random_routes", "hard_halt": True, "halt_rule": "none", "random_routing": True}]
    return AMLLM_EVAL_MODES


@dataclass
class TrainConfig:
    data: str = "data/ts_corpus"
    steps: int = 2000
    batch: int = 32
    lr: float = 1e-3
    min_lr_frac: float = 0.1
    warmup: int = 100
    weight_decay: float = 0.05
    seed: int = 42
    eval_every: int = 500
    eval_windows: int = 512
    eval_batch: int = 64
    amp: bool = True
    dispatch: str = "loop"  # expert dispatch used for training/eval: loop (true skipping) or bmm
    # The test split is evaluated (once, default mode) only if asked for. Default off since round V3: a run must not
    # produce test numbers that can leak into selection; the independent holdout replaces them.
    eval_test: bool = False
    run_dir: str = "results/run"


@torch.no_grad()
def evaluate(model, data, cfg: ModelConfig, tc: TrainConfig, device, cats=None, cat_names=None,
             random_routing: bool = False, **mode) -> dict:
    """Validation loss, hop and routing statistics; per-category stats for the *predicted* byte."""
    model.eval()
    set_random_routing(model, random_routing)
    gen_state = torch.cuda.get_rng_state() if device.type == "cuda" else None
    torch.manual_seed(1234)  # random-route mode is reproducible
    losses, hops, cat_all, flops, flops_ideal, times = [], [], [], [], [], []
    flops_exec, rows_pt = [], []
    mc = ec = None
    for x, y, c in eval_batches(data, tc.eval_windows, tc.eval_batch, cfg.max_seq_len, device, cats):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=tc.amp and device.type == "cuda"):
            logits, info = model(x, **mode)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        losses.append(F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1), reduction="none").cpu())
        hops.append(info["hops_per_token"].flatten().float().cpu())
        if c is not None:
            cat_all.append(c.flatten())
        flops.append(info["flops_per_token"])
        flops_ideal.append(info["flops_per_token_ideal"])
        flops_exec.append(info["flops_per_token_executed"])
        rows_pt.append(info["expert_rows_per_token"])
        mc = info["module_counts"].cpu() if mc is None else mc + info["module_counts"].cpu()
        ec = info["expert_counts"].cpu() if ec is None else ec + info["expert_counts"].cpu()
    set_random_routing(model, False)
    if gen_state is not None:
        torch.cuda.set_rng_state(gen_state)
    model.train()
    h, l = torch.cat(hops), torch.cat(losses)
    loss = l.mean().item()
    usage = ec.float().flatten() / ec.sum().clamp(min=1)
    entropy = -(usage[usage > 0] * usage[usage > 0].log()).sum().item()
    per_cat = {}
    if cat_all:
        ca = torch.cat(cat_all)
        for k, name in (cat_names or {}).items():
            m = ca == k
            if m.any():
                per_cat[name] = {"bytes": int(m.sum()), "loss": l[m].mean().item(), "mean_hops": h[m].mean().item()}
    return {
        "val_loss": loss, "val_bpb": loss / math.log(2), "val_ppl": math.exp(loss),
        "mean_hops": h.mean().item(), "p50_hops": h.quantile(0.5).item(), "p95_hops": h.quantile(0.95).item(),
        "hop_histogram": torch.bincount(h.long()).tolist(),
        "flops_per_token": sum(flops) / len(flops), "flops_per_token_ideal": sum(flops_ideal) / len(flops_ideal),
        # work the dispatch really did, padding rows of the bmm dispatch included (equals flops_per_token for loop/triton)
        "flops_per_token_executed": sum(flops_exec) / len(flops_exec), "expert_rows_per_token": sum(rows_pt) / len(rows_pt),
        "median_batch_ms": 1000 * sorted(times[1:] or times)[len(times[1:] or times) // 2],
        "module_counts": mc.tolist(), "expert_counts": ec.tolist(),
        "expert_usage_entropy": entropy, "expert_usage_entropy_max": math.log(max(1, usage.numel())),
        "dead_experts": int((ec.flatten() == 0).sum()),
        "max_expert_share": usage.max().item() if usage.numel() else 0.0,
        "per_category": per_cat,
    }


def run(cfg: ModelConfig, tc: TrainConfig) -> dict:
    run_dir = Path(tc.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(tc.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, val_data = load_split(tc.data, "train"), load_split(tc.data, "val")
    val_cats, cat_names = load_categories(tc.data, "val", val_data)
    model = build_model(cfg).to(device)
    set_dispatch(model, tc.dispatch)
    params = count_parameters(model)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tc.lr, betas=(0.9, 0.95))

    def lr_at(step: int) -> float:
        if step < tc.warmup:
            return tc.lr * (step + 1) / tc.warmup
        progress = (step - tc.warmup) / max(1, tc.steps - tc.warmup)
        return tc.lr * (tc.min_lr_frac + (1 - tc.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))

    (run_dir / "config.json").write_text(json.dumps({"model": cfg.to_dict(), "train": asdict(tc),
                                                     "data_split_id": split_id(tc.data),
                                                     "train_code_sha256": env.train_code_hash()}, indent=2))
    (run_dir / "env.json").write_text(json.dumps(env.collect(), indent=2))
    log = (run_dir / "train.log").open("w", encoding="utf-8")

    def emit(msg: str):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    emit(f"device={device} arch={cfg.arch} params={params['total']:,} non_embedding={params['non_embedding']:,}")
    gen = torch.Generator().manual_seed(tc.seed)
    curve, train_curve, tokens, train_time, steps_done = [], [], 0, 0.0, 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    amp = tc.amp and device.type == "cuda"
    model.train()
    for step in range(tc.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = random_batch(train_data, tc.batch, cfg.max_seq_len, gen, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, info = model(x)
            lm_loss = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
            loss = lm_loss + info["aux_loss"]
            if cfg.anytime_weight > 0 and "hop_states" in info:  # AD02a: train the intermediate hops to predict, too
                inter = info["hop_states"][:-1]
                if inter:
                    any_loss = sum(F.cross_entropy(model._head(s).float().reshape(-1, cfg.vocab_size), y.reshape(-1))
                                   for s in inter) / len(inter)
                    loss = loss + cfg.anytime_weight * any_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_time += time.perf_counter() - start
        tokens += x.numel()
        if not (math.isfinite(lm_loss.item()) and math.isfinite(loss.item())):
            # a diverged run is a failed run: no checkpoint, no report.json, a marker and a non-zero exit upstream
            emit(f"step={step} non-finite loss, aborting")
            log.close()
            (run_dir / "FAILED.json").write_text(json.dumps({"reason": "non-finite loss", "step": step,
                                                             "steps_requested": tc.steps, "lm_loss": str(lm_loss.item())}))
            raise FloatingPointError(f"non-finite loss at step {step} of {tc.steps} ({run_dir})")
        steps_done = step + 1
        if step % 25 == 0 or step == tc.steps - 1:
            train_curve.append([step, round(lm_loss.item(), 4), round(info["expected_hops"].item(), 3)])
        if step % 100 == 0 or step == tc.steps - 1:
            emit(f"step={step:5d} lm_loss={lm_loss.item():.4f} aux={info['aux_loss'].item():.4f} "
                 f"exp_hops={info['expected_hops'].item():.2f} lr={lr_at(step):.2e} tok/s={tokens / train_time:,.0f}")
        if (step + 1) % tc.eval_every == 0 and step + 1 < tc.steps:
            r = evaluate(model, val_data, cfg, tc, device)
            curve.append({"step": step + 1, "train_loss": lm_loss.item(), "val_loss": r["val_loss"],
                          "mean_hops": r["mean_hops"], "expected_hops_train": info["expected_hops"].item()})
            emit(f"  eval step={step + 1} val_loss={r['val_loss']:.4f} mean_hops={r['mean_hops']:.2f}")

    if steps_done != tc.steps:
        raise RuntimeError(f"training ended after {steps_done} of {tc.steps} steps ({run_dir})")
    torch.save({"config": cfg.to_dict(), "state_dict": model.state_dict(), "seed": tc.seed}, run_dir / "checkpoint.pt")
    train_peak_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None
    modes = eval_modes(cfg)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    evals = {}
    for mode in modes:
        kwargs = {k: v for k, v in mode.items() if k != "name"}
        evals[mode["name"]] = evaluate(model, val_data, cfg, tc, device, val_cats, cat_names, **kwargs)
        e = evals[mode["name"]]
        emit(f"  final[{mode['name']:>15s}] val_loss={e['val_loss']:.4f} bpb={e['val_bpb']:.3f} "
             f"hops={e['mean_hops']:.2f} (p95 {e['p95_hops']:.0f}) MFLOP/tok={e['flops_per_token'] / 1e6:.3f} "
             f"ideal={e['flops_per_token_ideal'] / 1e6:.3f} ms/batch={e['median_batch_ms']:.1f}")
    # Held-out test split: only on request (tc.eval_test), once, in the model's default inference mode only.
    test_default = None
    if tc.eval_test:
        test_data = load_split(tc.data, "test")
        test_cats, _ = load_categories(tc.data, "test", test_data)
        default_mode = {} if cfg.arch != "amllm" else {"hard_halt": True}
        test_eval = evaluate(model, test_data, cfg, tc, device, test_cats, cat_names, **default_mode)
        emit(f"  test[default] loss={test_eval['val_loss']:.4f} hops={test_eval['mean_hops']:.2f}")
        test_default = {k: test_eval[k] for k in ("val_loss", "val_bpb", "mean_hops", "per_category")}
    bad = [m for m, e in evals.items() if not all(math.isfinite(e[k]) for k in ("val_loss", "mean_hops", "flops_per_token"))]
    if bad:
        (run_dir / "FAILED.json").write_text(json.dumps({"reason": "non-finite evaluation", "modes": bad}))
        raise FloatingPointError(f"non-finite evaluation result in mode(s) {bad} ({run_dir})")
    report = {
        "arch": cfg.arch, "parameters": params, "seed": tc.seed, "steps": tc.steps, "steps_completed": steps_done,
        "train_tokens": tokens, "train_tokens_per_s": tokens / max(train_time, 1e-9),
        "train_seconds": train_time, "train_dispatch": tc.dispatch, "train_peak_vram_mb": train_peak_mb,
        "train_loss_last": lm_loss.item(), "curve": curve, "train_curve": train_curve,
        "eval": evals, "test_default_mode": test_default,
        "eval_peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == "cuda" else None,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    log.close()
    return report


def add_model_args(p: argparse.ArgumentParser):
    d = ModelConfig()
    for f, v in d.to_dict().items():
        flag = "--" + f.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            p.add_argument(flag, type=type(v), default=None)


def main():
    p = argparse.ArgumentParser()
    add_model_args(p)
    for f, v in asdict(TrainConfig()).items():
        p.add_argument("--" + f.replace("_", "-"), type=(lambda s: s.lower() in ("1", "true", "yes")) if isinstance(v, bool) else type(v), default=v)
    a = vars(p.parse_args())
    model_kw = {k: v for k, v in a.items() if k in ModelConfig.__dataclass_fields__ and v is not None}
    model_kw.setdefault("max_seq_len", 256)
    tc = TrainConfig(**{k: a[k] for k in TrainConfig.__dataclass_fields__})
    run(ModelConfig(**model_kw), tc)


if __name__ == "__main__":
    main()
