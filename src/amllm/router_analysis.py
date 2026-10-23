"""Router specialisation analysis (K1, K5): usage balance and route/input dependence.

    python -m amllm.router_analysis --runs results/e07_seeds/amllm_s42 ... --out results/router_analysis.json

For every hop (AM-LLM) or layer (MoE) on validation text:
- module/expert usage entropy (normalised to [0,1]), max share, dead units
- mutual information I(selected module ; byte class of the *input* byte), normalised by H(class)
- the same for an *untrained* model with the same config and seed: a random router is already
  input dependent, so only the trained-minus-untrained difference is evidence of learning
- hop-to-hop route persistence and hard-halting depth per byte class
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .config import ModelConfig
from .data import CODE_BYTE_CLASSES, byte_classes, eval_batches, load_split
from .model import build_model


def entropy(counts: torch.Tensor) -> float:
    p = counts.float() / counts.sum().clamp(min=1)
    p = p[p > 0]
    return -(p * p.log()).sum().item()


def mutual_information(a: torch.Tensor, b: torch.Tensor, na: int, nb: int) -> tuple[float, float]:
    joint = torch.zeros(na, nb)
    joint.index_put_((a, b), torch.ones_like(a, dtype=torch.float), accumulate=True)
    joint /= joint.sum()
    pa, pb = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
    nz = joint > 0
    mi = (joint[nz] * (joint[nz] / (pa @ pb)[nz]).log()).sum().item()
    hb = -(pb[pb > 0] * pb[pb > 0].log()).sum().item()
    return mi / math.log(2), mi / hb if hb > 0 else 0.0


@torch.no_grad()
def analyse(cfg: ModelConfig, model, x: torch.Tensor, **mode) -> dict:
    cls = byte_classes(x.cpu()).flatten().long()
    n_cls = len(CODE_BYTE_CLASSES)
    _, info = model(x, trace=True, **mode)
    n_units = cfg.n_modules if cfg.arch == "amllm" else cfg.n_experts
    per_step, prev_sets = [], None
    for h, step in enumerate(info["trace"]):
        pos = step["positions"]
        # AM-LLM: route unit = module; MoE: route unit = expert of that layer
        # dropped routes of adaptive K (expert_topp) are -1 in the trace: they are no unit and are left out
        raw = step["modules"] if cfg.arch == "amllm" else step["experts"].reshape(len(pos), -1)
        k = raw.shape[1]
        flat_raw = raw.flatten()
        kept = flat_raw >= 0
        flat_units = (flat_raw[kept] % n_units) if cfg.arch != "amllm" else flat_raw[kept]
        flat_cls = cls[pos].repeat_interleave(k)[kept]
        cls_tokens = torch.bincount(cls[pos], minlength=n_cls).float()
        counts = torch.bincount(flat_units, minlength=n_units)
        mi_bits, mi_norm = mutual_information(flat_units, flat_cls, n_units, n_cls)
        cond = torch.zeros(n_cls, n_units)
        cond.index_put_((flat_cls, flat_units), torch.ones(len(flat_units)), accumulate=True)
        top_unit = {CODE_BYTE_CLASSES[c]: int(cond[c].argmax()) for c in range(n_cls) if cond[c].sum() > 0}
        # fraction of the class's tokens that selected its most frequent unit (each token picks k units)
        top_share = {CODE_BYTE_CLASSES[c]: round((cond[c].max() / cls_tokens[c]).item(), 3)
                     for c in range(n_cls) if cond[c].sum() > 0}
        r = {"step": h, "tokens": len(pos), "usage": counts.tolist(),
             "usage_entropy_norm": entropy(counts) / math.log(n_units) if n_units > 1 else 1.0,
             "max_share": (counts.max() / counts.sum()).item(), "dead_units": int((counts == 0).sum()),
             "mi_bits_unit_vs_byteclass": mi_bits, "mi_norm_by_H_class": mi_norm,
             "top_unit_per_byteclass": top_unit, "top_unit_share_per_byteclass": top_share}
        if cfg.arch == "amllm":
            ge = step["experts"].flatten()
            ec = torch.bincount(ge[ge >= 0], minlength=cfg.n_modules * cfg.experts_per_module)
            r["expert_usage_entropy_norm"] = entropy(ec) / math.log(len(ec))
            r["dead_experts"] = int((ec == 0).sum())
            sets = {p: frozenset(u) for p, u in zip(pos.tolist(), raw.tolist())}  # AM: modules (never -1)
            if prev_sets is not None:
                common = [p for p in sets if p in prev_sets]
                r["same_module_set_as_prev_hop"] = sum(sets[p] == prev_sets[p] for p in common) / max(1, len(common))
            prev_sets = sets
        per_step.append(r)
    out = {"steps": per_step}
    if cfg.arch == "amllm":
        hops = info["hops_per_token"].flatten().cpu().float()
        out["mean_hops_per_input_byteclass"] = {CODE_BYTE_CLASSES[c]: hops[cls == c].mean().item()
                                                for c in range(n_cls) if (cls == c).any()}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data/ts_corpus")
    p.add_argument("--windows", type=int, default=64)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val = load_split(a.data, "val")
    x, _, _ = next(eval_batches(val, a.windows, a.windows, 256, device))
    result = {"data": a.data, "tokens": x.numel(), "byte_classes": CODE_BYTE_CLASSES, "runs": {}}
    for run in a.runs:
        ckpt = torch.load(Path(run) / "checkpoint.pt", map_location="cpu", weights_only=True)
        cfg = ModelConfig.from_dict(ckpt["config"])
        if cfg.arch == "dense":
            continue
        xs = x[:, :cfg.max_seq_len]
        torch.manual_seed(ckpt["seed"])
        untrained = build_model(cfg).to(device).eval()
        trained = build_model(cfg).to(device).eval()
        trained.load_state_dict(ckpt["state_dict"])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            # trained: its real (default, hard-halting) inference; untrained control: every hop runs,
            # otherwise its untrained halting head would cut the trace short
            control_mode = {"hard_halt": True, "halt_rule": "none"} if cfg.arch == "amllm" else {}
            t, u = analyse(cfg, trained, xs), analyse(cfg, untrained, xs, **control_mode)
        key = f"{Path(run).parent.name}/{Path(run).name}"  # experiment/run: run names repeat across experiments
        result["runs"][key] = {"arch": cfg.arch, "trained": t, "untrained_control": u}
        for st, su in zip(t["steps"], u["steps"]):
            print(f"{key:40s} step{st['step']} MI_norm trained={st['mi_norm_by_H_class']:.3f} "
                  f"untrained={su['mi_norm_by_H_class']:.3f} usageH={st['usage_entropy_norm']:.3f} "
                  f"max_share={st['max_share']:.3f} dead={st['dead_units']}")
    Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
