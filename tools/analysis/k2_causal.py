"""K2 / FD06 (CPU): causal test of the learned routing. How much worse does a trained checkpoint get on validation when its
routing is broken at evaluation time, everything else unchanged?

    python tools/analysis/k2_causal.py <run_dir> ... --data data/code_corpus_v2 --windows 16 --out out.json

Interventions (each on a fresh copy of the checkpoint, default evaluation mode):
  learned    the model as trained (reference)
  random     route choices drawn uniformly at random per token (weights are the learned scores at the random picks)
  permuted   every router's output columns permuted by a fixed random permutation: consistent but wrong routes
  reinit     every router re-initialised (module and expert routers): consistent routes that ignore training
Validation split only; the same windows for every intervention. Reports NLL and the difference to `learned`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from amllm.data import eval_batches, load_split  # noqa: E402
from amllm.model import AdaptiveModularLM, set_diagnostics, set_random_routing  # noqa: E402
from amllm.profile import load_run  # noqa: E402
from amllm.routing import HierarchicalModuleBank  # noqa: E402

INTERVENTIONS = ("learned", "random", "permuted", "reinit")


def intervene(model, how: str, seed: int = 0) -> None:
    g = torch.Generator().manual_seed(seed)
    banks = [m for m in model.modules() if isinstance(m, HierarchicalModuleBank)]
    if how == "random":
        set_random_routing(model, True)
    with torch.no_grad():
        for b in banks:
            if how == "permuted":
                pm = torch.randperm(b.n_modules, generator=g)
                b.router.weight.copy_(b.router.weight[pm])
                b.router.bias.copy_(b.router.bias[pm])
                pe = torch.randperm(b.n_experts, generator=g)
                b.expert_router_w.copy_(b.expert_router_w[..., pe])
                b.expert_router_b.copy_(b.expert_router_b[..., pe])
            elif how == "reinit":
                torch.nn.init.normal_(b.router.weight, 0, 0.02, generator=g)
                b.router.bias.zero_()
                b.expert_router_w.normal_(0, 0.02, generator=g)
                b.expert_router_b.zero_()


@torch.no_grad()
def val_nll(model, val, windows: int, batch: int = 4) -> float:
    mode = {"hard_halt": True} if isinstance(model, AdaptiveModularLM) and model.cfg.halt_rule != "none" else {}
    tot = n = 0
    torch.manual_seed(0)  # the random-routing intervention draws from the global generator
    for x, y, _ in eval_batches(val, windows, batch, model.cfg.max_seq_len, torch.device("cpu")):
        logits = model(x, **mode)[0]
        tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1), reduction="sum").item()
        n += y.numel()
    return tot / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--data", default="data/code_corpus_v2")
    p.add_argument("--windows", type=int, default=16)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    val = load_split(a.data, "val")
    out = {"windows": a.windows, "split": "val", "runs": {}}
    for run in a.runs:
        run_dir = Path(run)
        key = f"{run_dir.parent.name}/{run_dir.name}"
        r = {}
        for how in INTERVENTIONS:
            cfg, model = load_run(run_dir, torch.device("cpu"))
            set_diagnostics(model, False)
            if cfg.arch == "dense":
                break
            intervene(model, how)
            r[how] = val_nll(model, val, a.windows)
        if r:
            r["delta"] = {h: r[h] - r["learned"] for h in INTERVENTIONS[1:]}
            out["runs"][key] = r
            print(key, {h: round(v, 4) for h, v in r.items() if h != "delta"}, flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
