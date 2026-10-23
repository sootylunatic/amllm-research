"""K2 / FD06 (CPU): what do the routers depend on - token identity, context or position?

For every hop (AM-LLM) or layer (MoE) the route label X is the first-choice global expert id (and, for AM-LLM, the
module id). It is related to
  T  the current input token id,
  P  the previous token id (context),
  B  a coarse position bucket (position // 64),
  S  a token-shape class (whitespace/newline, punctuation, digit, identifier-like, other) from the decoded token.
Plug-in mutual information is biased upwards for small samples, so every value is reported next to a permutation
baseline (X shuffled across tokens of the same hop, 5 draws) and as the excess over it. Tokens seen fewer than
--min-count times are grouped as 'rare' for T and P.

    python tools/analysis/k2_route_trace.py <checkpoint.pt> --data data/code_corpus_v2 --windows 256 --out out.json

Validation split only. CPU, fp32.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from amllm.config import ModelConfig  # noqa: E402
from amllm.data import eval_batches, load_split  # noqa: E402
from amllm.model import build_model  # noqa: E402


def entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in I(X;Y) in bits for integer label arrays."""
    xi, xn = np.unique(x, return_inverse=True)
    yi, yn = np.unique(y, return_inverse=True)
    joint = np.zeros((len(xi), len(yi)))
    np.add.at(joint, (xn, yn), 1)
    return entropy(joint.sum(1)) + entropy(joint.sum(0)) - entropy(joint.ravel())


def shape_class(text: str) -> int:
    t = text.replace("Ġ", " ").replace("Ċ", "\n")
    if t == "" or t.strip() == "":
        return 0 if "\n" not in t else 1
    if re.fullmatch(r"[0-9.]+", t.strip()):
        return 2
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", t.strip()):
        return 3
    if re.fullmatch(r"[^\w\s]+", t.strip()):
        return 4
    return 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/code_corpus_v2")
    ap.add_argument("--windows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--min-count", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    ck = torch.load(a.checkpoint, map_location="cpu")
    cfg = ModelConfig.from_dict(ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    val = load_split(a.data, "val")
    kw = {"hard_halt": True, "halt_rule": "none"} if cfg.arch == "amllm" else {}
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(Path(a.data) / "tokenizer.json"))
    shapes = np.array([shape_class(tok.id_to_token(i) or "") for i in range(cfg.vocab_size)])
    units = cfg.n_hops if cfg.arch == "amllm" else cfg.n_layers
    route_x = [[] for _ in range(units)]
    route_m = [[] for _ in range(units)]
    toks, prevs, poss = [], [], []
    with torch.no_grad():
        for x, y, _ in eval_batches(val, a.windows, a.batch, cfg.max_seq_len, torch.device("cpu")):
            _, info = model(x, trace=True, **kw)
            b, t = x.shape
            toks.append(x.reshape(-1).numpy())
            prevs.append(torch.cat([torch.zeros(b, 1, dtype=x.dtype), x[:, :-1]], 1).reshape(-1).numpy())
            poss.append(torch.arange(t).repeat(b).numpy() // 64)
            for u, rec in enumerate(info["trace"]):
                assert rec["modules"].shape[0] == b * t, "hard halting active: trace is not token-aligned"
                route_x[u].append(rec["experts"][:, 0, 0].numpy())
                route_m[u].append(rec["modules"][:, 0].numpy())
    T, P, B = (np.concatenate(v) for v in (toks, prevs, poss))
    counts = Counter(T.tolist())
    frequent = {k for k, c in counts.items() if c >= a.min_count}

    def group(v):
        return np.array([int(i) if i in frequent else -1 for i in v])

    Tg, Pg, S = group(T), group(P), shapes[T]
    res = {"checkpoint": a.checkpoint, "platform": platform.platform(), "torch": torch.__version__, "split": "val",
           "tokens": int(T.size), "arch": cfg.arch, "units": units, "min_count": a.min_count,
           "frequent_token_types": len(frequent), "frequent_token_share": round(float(np.isin(T, list(frequent)).mean()), 4),
           "per_unit": []}
    for u in range(units):
        X = np.concatenate(route_x[u])
        Mx = np.concatenate(route_m[u])
        row = {"unit": u, "route_entropy_bits": round(entropy(np.bincount(X)), 3), "route_entropy_max_bits": round(math.log2(len(np.unique(X))), 3)}
        for name, Z in (("token", Tg), ("prev_token", Pg), ("position_bucket", B), ("token_shape", S)):
            mi = mutual_info(X, Z)
            base = float(np.mean([mutual_info(rng.permutation(X), Z) for _ in range(5)]))
            row[f"MI_expert_{name}"] = round(mi, 4)
            row[f"MI_expert_{name}_permutation_baseline"] = round(base, 4)
            row[f"MI_expert_{name}_excess"] = round(mi - base, 4)
        if cfg.arch == "amllm":
            mi = mutual_info(Mx, Tg)
            base = float(np.mean([mutual_info(rng.permutation(Mx), Tg) for _ in range(5)]))
            row["MI_module_token_excess"] = round(mi - base, 4)
            row["module_entropy_bits"] = round(entropy(np.bincount(Mx)), 3)
        row["token_explained_share"] = round(row["MI_expert_token_excess"] / max(row["route_entropy_bits"], 1e-9), 3)
        row["prev_token_explained_share"] = round(row["MI_expert_prev_token_excess"] / max(row["route_entropy_bits"], 1e-9), 3)
        res["per_unit"].append(row)
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
