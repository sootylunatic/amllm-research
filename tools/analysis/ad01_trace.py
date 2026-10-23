"""AD01 (CPU): per-hop trace of an AdaptiveModularLM checkpoint on validation windows.

Re-runs the hop loop of `AdaptiveModularLM.forward` and records the state after every hop, so that the same state can be
read out through the final norm + head ("logit lens"). The trace is checked against `model.forward` (soft and hard
mode): a mismatch aborts, so the numbers below describe the real model. Validation split only; no test data.

    python tools/analysis/ad01_trace.py <checkpoint.pt> --data data/code_corpus_v2 --windows 128 --out out.json

Reports (nat/token):
  nll_per_hop      NLL when the state after hop h is read out directly (soft and fixed-depth traces)
  soft / hard      final NLL of the soft path and of the hard path at each survival threshold, with mean hops
  oracle           per-token best hop of the soft trace (analytic bound, not a deployable policy)
  random exit      matched control: the multiset of the hard path's exit hops is shuffled over the tokens, then
                   - random_exit_matched_nll_hard: a real hard run that stops each token at its assigned hop (frozen
                     state, still a key/value for later tokens; the same survival weights) — the valid control;
                   - random_exit_matched_nll_from_soft_trace: the read-out of the *soft* trace at the assigned hop
                     (W0/W1 approximation; later tokens attended to unfrozen states, so it is not a stopped run)
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
from amllm.data import eval_batches, load_split  # noqa: E402
from amllm.model import AdaptiveModularLM, build_model  # noqa: E402


@torch.no_grad()
def trace(model: AdaptiveModularLM, ids: torch.Tensor, *, hard: bool, rule: str | None = None, threshold: float | None = None,
          exits: torch.Tensor | None = None):
    """Same loop as AdaptiveModularLM.forward; returns (states after each hop [H,B,T,D], hops_per_token, survival [H,B,T]).
    exits [B,T] (hard only): each token stops after exactly that many hops instead of by the halting rule."""
    cfg, d = model.cfg, model.cfg.d_model
    rule = rule or cfg.halt_rule
    threshold = cfg.halt_threshold if threshold is None else threshold
    b, t = ids.shape
    x = model._embed(ids)
    survival = x.new_ones(b, t)
    active = torch.ones(b, t, dtype=torch.bool)
    counts = torch.zeros(b, t, dtype=torch.long)
    states, surv = [], []
    for hop in range(cfg.n_hops):
        if exits is not None:
            active = hop < exits
        if hard:
            idx = active.flatten().nonzero().squeeze(1)
            if idx.numel() == 0:
                states += [x] * (cfg.n_hops - hop)
                surv += [survival] * (cfg.n_hops - hop)
                break
        weight = survival if rule == "survival" else torch.ones_like(survival)
        if hard:
            weight = weight * active
        a = model.attention_at(hop, x)  # None on an expert-only hop (FD04 attn_every)
        if a is not None:
            x = x + weight[..., None] * a
        flat, w_flat = x.reshape(-1, d), weight.reshape(-1, 1)
        if hard:
            delta, *_ = model.bank_for(hop)(model.normed(hop, flat.index_select(0, idx)))
            flat = flat.index_add(0, idx, delta * w_flat.index_select(0, idx))
            counts += active
        else:
            delta, *_ = model.bank_for(hop)(model.normed(hop, flat))
            flat = flat + w_flat * delta
            counts += 1
        x = flat.view(b, t, d)
        states.append(x)
        surv.append(survival)
        if hop < cfg.n_hops - 1 and rule != "none":
            p_stop = torch.sigmoid(model.halt_head(x)).squeeze(-1).float()
            if rule == "step" and hard and exits is None:
                active = active & (p_stop < threshold)
            survival = survival * (1.0 - p_stop)
            if rule == "survival" and hard and exits is None:
                active = active & (survival >= threshold)
    return torch.stack(states), counts, torch.stack(surv)


def nll(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model._head(x)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1), reduction="none").view(y.shape)


@torch.no_grad()  # the analysis must never build autograd graphs (they kept ~12 GiB of softmax tensors alive)
def analyse(model, cfg, batches, thresholds):
    n_hops = cfg.n_hops
    acc = {"hop_nll_soft": torch.zeros(n_hops), "soft_final": 0.0, "n": 0, "oracle_min": 0.0,
           "surv_mean": torch.zeros(n_hops)}
    hard_acc = {th: {"nll": 0.0, "hops": 0.0, "random_matched": 0.0, "random_matched_hard": 0.0} for th in thresholds}
    max_diff = 0.0
    for x, y in batches:
        states, _, surv = trace(model, x, hard=False)
        ref_logits, _ = model(x, hard_halt=False)
        diff = (model._head(states[-1]) - ref_logits).abs().max().item()
        max_diff = max(max_diff, diff)
        per_hop = torch.stack([nll(model, states[h], y) for h in range(n_hops)])  # [H,B,T]
        acc["hop_nll_soft"] += per_hop.mean((1, 2)) * y.numel()
        acc["soft_final"] += per_hop[-1].mean().item() * y.numel()
        acc["oracle_min"] += per_hop.min(0).values.mean().item() * y.numel()
        acc["surv_mean"] += surv.mean((1, 2)) * y.numel()
        acc["n"] += y.numel()
        for th in thresholds:
            hs, counts, _ = trace(model, x, hard=True, threshold=th)
            ref_hard, info = model(x, hard_halt=True, halt_threshold=th)
            hard_diff = (model._head(hs[-1]) - ref_hard).abs().max().item()
            max_diff = max(max_diff, hard_diff)
            n_h = nll(model, hs[-1], y)
            # matched random control: same multiset of exit hops, assigned to random tokens, read from the soft trace
            hops = counts.flatten()
            perm = hops[torch.randperm(hops.numel())].view_as(counts)
            rnd = per_hop.gather(0, (perm - 1).clamp(min=0)[None]).squeeze(0)
            rs, r_counts, _ = trace(model, x, hard=True, exits=perm)
            assert torch.equal(r_counts, perm), "forced exits were not applied"
            rnd_hard = nll(model, rs[-1], y)
            hard_acc[th]["nll"] += n_h.mean().item() * y.numel()
            hard_acc[th]["hops"] += counts.float().mean().item() * y.numel()
            hard_acc[th]["random_matched"] += rnd.mean().item() * y.numel()
            hard_acc[th]["random_matched_hard"] += rnd_hard.mean().item() * y.numel()
    return acc, hard_acc, max_diff


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--data", default="data/code_corpus_v2")
    ap.add_argument("--windows", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--thresholds", type=float, nargs="*", default=[0.1, 0.3, 0.5, 0.8])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.manual_seed(0)
    ck = torch.load(a.checkpoint, map_location="cpu")
    cfg = ModelConfig.from_dict(ck["config"])
    assert cfg.arch == "amllm", "AD01 trace is for AM-LLM checkpoints"
    model = build_model(cfg)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    model.diagnostics = False
    val = load_split(a.data, "val")
    acc, hard_acc, max_diff = analyse(model, cfg, ((x, y) for x, y, _ in eval_batches(val, a.windows, a.batch, cfg.max_seq_len, torch.device("cpu"))), a.thresholds)
    n_hops = cfg.n_hops
    if max_diff > 1e-3:
        print(f"TRACE MISMATCH vs model.forward: max |dlogit| = {max_diff:.2e}", file=sys.stderr)
        return 1
    n = acc["n"]
    res = {
        "checkpoint": a.checkpoint, "platform": platform.platform(), "torch": torch.__version__, "windows": a.windows,
        "tokens": n, "split": "val", "halt_rule": cfg.halt_rule, "n_hops": n_hops, "trace_vs_forward_max_abs_logit_diff": max_diff,
        "nll_per_hop_soft": [round(v / n, 4) for v in acc["hop_nll_soft"].tolist()],
        "soft_final": round(acc["soft_final"] / n, 4), "oracle_min_over_hops_soft": round(acc["oracle_min"] / n, 4),
        "mean_survival_before_hop": [round(v / n, 4) for v in acc["surv_mean"].tolist()],
        "hard": {str(th): {"val_nll": round(v["nll"] / n, 4), "mean_hops": round(v["hops"] / n, 3),
                           "random_exit_matched_nll_hard": round(v["random_matched_hard"] / n, 4),
                           "random_exit_matched_nll_from_soft_trace": round(v["random_matched"] / n, 4)}
                 for th, v in hard_acc.items()},
    }
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
