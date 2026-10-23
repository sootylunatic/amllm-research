"""AD01 gradient audit on a real checkpoint (CPU): which parts of a halting model receive how much gradient, and what
the stopping decision is pushed by. Training-split batches only (train mode, soft path = the training forward).

  grad_lm      gradient norm of the language-model cross-entropy alone, per component
  grad_aux     gradient norm of the auxiliary loss alone (ponder + balance, with the checkpoint's coefficients)
  halt_head    both, and their ratio: when |aux| >> |lm| on the halt head the ponder penalty, not the language model,
               decides when to stop (the collapse seen at high ponder weights)

    python tools/analysis/ad01_grad.py <checkpoint.pt> --batches 8 --out out.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from amllm.config import ModelConfig  # noqa: E402
from amllm.data import load_split, random_batch  # noqa: E402
from amllm.model import build_model  # noqa: E402


def component(name: str) -> str:
    if name.startswith("halt_head"):
        return "halt_head"
    if name.startswith("attention") or name.startswith("attn_norm"):
        return "attention_hop1"
    if name.startswith("hop_attentions"):
        return f"attention_hop{int(name.split('.')[1]) + 2}"
    if name.startswith("hop_attn_norms"):
        return f"attention_hop{int(name.split('.')[1]) + 2}"
    if name.startswith(("bank.router", "bank.expert_router")):
        return "routers"
    if name.startswith("bank.experts"):
        return "experts"
    if name.startswith(("embed", "positions")):
        return "embeddings"
    if name.startswith(("lm_head", "out_norm")):
        return "head"
    return "other"


def grad_norms(model, loss) -> dict[str, float]:
    model.zero_grad()
    loss.backward(retain_graph=True)
    sq: dict[str, float] = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            sq[component(n)] = sq.get(component(n), 0.0) + float(p.grad.double().norm() ** 2)
    return {k: v ** 0.5 for k, v in sq.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/code_corpus_v2")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cpu")
    cfg = ModelConfig.from_dict(ck["config"])
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    model.train()
    data = load_split(a.data, "train")
    gen = torch.Generator().manual_seed(0)
    lm_acc: dict[str, float] = {}
    aux_acc: dict[str, float] = {}
    for _ in range(a.batches):
        x, y = random_batch(data, a.batch, cfg.max_seq_len, gen, torch.device("cpu"))
        logits, info = model(x, hard_halt=False)
        lm = F.cross_entropy(logits.float().reshape(-1, cfg.vocab_size), y.reshape(-1))
        for acc, loss in ((lm_acc, lm), (aux_acc, info["aux_loss"])):
            for k, v in grad_norms(model, loss).items():
                acc[k] = acc.get(k, 0.0) + v / a.batches
    comps = sorted(set(lm_acc) | set(aux_acc))
    res = {"checkpoint": a.checkpoint, "platform": platform.platform(), "torch": torch.__version__, "split": "train",
           "batches": a.batches, "batch": a.batch, "halt_rule": cfg.halt_rule, "ponder_coefficient": cfg.ponder_coefficient,
           "grad_norm_lm": {k: round(lm_acc.get(k, 0.0), 6) for k in comps},
           "grad_norm_aux": {k: round(aux_acc.get(k, 0.0), 6) for k in comps},
           "halt_head_aux_over_lm": round(aux_acc.get("halt_head", 0.0) / max(lm_acc.get("halt_head", 1e-12), 1e-12), 3)}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
