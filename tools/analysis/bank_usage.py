"""Correct expert-usage statistics for AM-LLM checkpoints with several banks (bank_groups > 1), computed after the fact.

The training code of wave W1 sums the per-hop counters of different banks into one [modules, experts] table, so the
`dead_experts`, `expert_usage_entropy` and `max_expert_share` of such reports describe merged experts, not the distinct
ones. This tool re-derives them per bank from the route trace (validation split, CPU):

    python tools/analysis/bank_usage.py <checkpoint.pt> --windows 128 --out out.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from amllm.config import ModelConfig  # noqa: E402
from amllm.data import eval_batches, load_split  # noqa: E402
from amllm.model import build_model  # noqa: E402


def stats(counts: torch.Tensor) -> dict:
    total = counts.sum().clamp(min=1).float()
    p = counts.float() / total
    nz = p[p > 0]
    return {"experts": int(counts.numel()), "dead": int((counts == 0).sum()),
            "entropy_bits": round(float(-(nz * nz.log2()).sum()), 4), "max_entropy_bits": round(math.log2(counts.numel()), 4),
            "max_share": round(float(p.max()), 4)}


def bank_usage(model, batches) -> dict:
    cfg = model.cfg
    g_n, hops = cfg.bank_groups, cfg.n_hops
    per_bank = [torch.zeros(cfg.n_modules * cfg.experts_per_module, dtype=torch.long) for _ in range(g_n)]
    with torch.no_grad():
        for x in batches:
            _, info = model(x, hard_halt=True, halt_rule="none", trace=True)
            for hop, rec in enumerate(info["trace"]):
                ids = rec["experts"].reshape(-1)
                ids = ids[ids >= 0]  # -1 = route dropped by adaptive K (expert_topp)
                per_bank[hop * g_n // hops] += torch.bincount(ids, minlength=per_bank[0].numel())
    merged = sum(per_bank)
    out = {"per_bank": [stats(c) for c in per_bank], "all_distinct_experts": stats(torch.cat(per_bank)),
           "merged_like_the_w1_report": stats(merged)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/code_corpus_v2")
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu")
    cfg = ModelConfig.from_dict(ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    val = load_split(a.data, "val")
    res = {"checkpoint": a.checkpoint, "bank_groups": cfg.bank_groups,
           **bank_usage(model, (x for x, _, _ in eval_batches(val, a.windows, 8, cfg.max_seq_len, torch.device("cpu"))))}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
