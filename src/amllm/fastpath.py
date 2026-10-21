"""F2-1: bucketed, static-shape execution of hard-halting AM-LLM inference (docs/PHASE2_PLAN.md §4, variant F2a).

Same mathematical model as `AdaptiveModularLM.forward(hard_halt=True)` with the survival rule: in the reference, a
halted token's attention and module updates are multiplied by weight 0, so computing them only for active tokens does
not change the result. Per hop:
  1. the active tokens of every sequence are packed (position order) and padded to the smallest bucket
     (BUCKETS x context length) that holds the longest sequence's active set -> only len(BUCKETS) distinct shapes;
  2. attention queries are computed only for the packed tokens; keys/values use all tokens (halted tokens keep
     serving as context); a causal mask is built from the original positions;
  3. the module bank runs on the packed rows (padding rows are computed and counted, their weight is 0);
  4. results are scattered back with the reference survival weights; halting is updated for all tokens.
This module is execution code, not training code: it is not part of `train_code_sha256`.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import AdaptiveModularLM, CausalAttention

BUCKETS = (0.25, 0.5, 0.75, 1.0)


def bucket_size(n_active: int, context: int, buckets=BUCKETS) -> int:
    for frac in buckets:
        cap = max(1, round(frac * context))
        if cap >= n_active:
            return cap
    return context


def _packed_attention(attn: CausalAttention, h: Tensor, idx: Tensor, valid: Tensor) -> Tensor:
    """Causal attention with queries only at positions idx [B,C] and keys/values at all T positions."""
    b, t, d = h.shape
    heads = attn.n_heads
    w_q, w_kv = attn.qkv.weight[:d], attn.qkv.weight[d:]
    b_q, b_kv = attn.qkv.bias[:d], attn.qkv.bias[d:]
    hq = h.gather(1, idx[..., None].expand(-1, -1, d))                       # [B,C,d]
    q = F.linear(hq, w_q, b_q)
    k, v = F.linear(h, w_kv, b_kv).chunk(2, dim=-1)                         # [B,T,d] each

    def split(y: Tensor) -> Tensor:
        return y.reshape(b, y.shape[1], heads, d // heads).transpose(1, 2)

    key_pos = torch.arange(t, device=h.device)
    allowed = key_pos[None, None, :] <= idx[:, :, None]                       # [B,C,T] causal by original position
    allowed = allowed | ~valid[:, :, None]                                    # padding rows: anything (discarded)
    out = F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=allowed[:, None])
    return attn.out(out.transpose(1, 2).reshape(b, -1, d))                    # [B,C,d]


@torch.no_grad()
def bucketed_hard_forward(model: AdaptiveModularLM, token_ids: Tensor, threshold: float | None = None,
                          buckets=BUCKETS) -> tuple[Tensor, dict]:
    cfg = model.cfg
    if not (cfg.arch == "amllm" and cfg.attn_per_hop and cfg.halt_rule == "survival"):
        raise ValueError("bucketed execution supports AM-LLM with attention in every hop and survival halting")
    threshold = cfg.halt_threshold if threshold is None else threshold
    b, t = token_ids.shape
    d = cfg.d_model
    x = model._embed(token_ids)
    survival = x.new_ones(b, t)
    active = torch.ones(b, t, dtype=torch.bool, device=x.device)
    counts = torch.zeros(b, t, dtype=torch.long, device=x.device)
    positions = torch.arange(t, device=x.device)
    hop_stats = []

    for hop in range(cfg.n_hops):
        per_seq = active.sum(1)
        n_max = int(per_seq.max())                                            # one host sync per hop
        if n_max == 0:
            break
        cap = bucket_size(n_max, t, buckets)
        # active tokens first, in position order (stable sort), then padding
        # int64 key: an int8 key overflows for t >= 128 and mixes halted tokens into the active block
        order = torch.sort((~active).long() * t + positions[None, :], dim=1).indices
        idx = order[:, :cap]                                                  # [B,C]
        valid = torch.arange(cap, device=x.device)[None, :] < per_seq[:, None]
        weight = (survival * active).gather(1, idx) * valid                   # reference update weights

        if cfg.unshared_attention and hop > 0:
            attn, norm = model.hop_attentions[hop - 1], model.hop_attn_norms[hop - 1]
        else:
            attn, norm = model.attention, model.attn_norm
        a = _packed_attention(attn, norm(x), idx, valid)
        x = x.scatter_add(1, idx[..., None].expand(-1, -1, d), weight[..., None] * a)

        rows = x.gather(1, idx[..., None].expand(-1, -1, d)).reshape(-1, d)
        delta, _, _, _ = model.bank(model.step_norm(rows))
        x = x.scatter_add(1, idx[..., None].expand(-1, -1, d), weight[..., None] * delta.view(b, cap, d))

        counts += active
        packed_active = int((valid & active.gather(1, idx)).sum())   # active tokens that are really packed
        hop_stats.append({"hop": hop, "capacity": cap, "max_active_per_sequence": n_max,
                          "packed_active_tokens": packed_active,
                          "active_tokens": int(per_seq.sum()), "packed_valid_tokens": int(valid.sum()),
                          "padding_ratio": (b * cap) / max(1, int(per_seq.sum()))})
        if hop < cfg.n_hops - 1:
            # identical to the reference (model.py): the stop probability is rounded to fp32
            p_stop = torch.sigmoid(model.halt_head(x)).squeeze(-1).float()
            survival = survival * (1.0 - p_stop)
            active = active & (survival >= threshold)

    return model._head(x), {"hops_per_token": counts, "hops": hop_stats,
                            "mean_hops": float(counts.float().mean()),
                            "packed_rows": sum(s["capacity"] * b for s in hop_stats),
                            "active_rows": sum(s["active_tokens"] for s in hop_stats)}


def full_depth_forward(model: AdaptiveModularLM, token_ids: Tensor) -> Tensor:
    """Reference for the F2 gate: the same checkpoint executed at full depth (all hops, no halting)."""
    with torch.no_grad():
        logits, _ = model(token_ids, hard_halt=True, halt_rule="none")
    return logits


@torch.no_grad()
def static_hard_forward(model: AdaptiveModularLM, token_ids: Tensor, threshold: float | None = None) -> tuple[Tensor, Tensor]:
    """F3/B2: hard survival halting computed for all tokens with the reference update weights (halted tokens get
    weight 0). Same mathematical model as the reference hard forward, but without the per-hop `nonzero` and without
    early exit, so every shape is static and the forward has no host synchronisation (with `grouped` dispatch)."""
    cfg = model.cfg
    if not (cfg.arch == "amllm" and cfg.attn_per_hop and cfg.halt_rule == "survival"):
        raise ValueError("static execution supports AM-LLM with attention in every hop and survival halting")
    threshold = cfg.halt_threshold if threshold is None else threshold
    b, t = token_ids.shape
    d = cfg.d_model
    x = model._embed(token_ids)
    survival = x.new_ones(b, t)
    active = torch.ones(b, t, dtype=torch.bool, device=x.device)
    counts = torch.zeros(b, t, dtype=torch.long, device=x.device)
    for hop in range(cfg.n_hops):
        weight = (survival * active)[..., None]
        if cfg.unshared_attention and hop > 0:
            a = model.hop_attentions[hop - 1](model.hop_attn_norms[hop - 1](x))
        else:
            a = model.attention(model.attn_norm(x))
        x = x + weight * a
        delta, _, _, _ = model.bank(model.step_norm(x).reshape(-1, d))
        x = x + weight * delta.view(b, t, d)
        counts += active
        if hop < cfg.n_hops - 1:
            p_stop = torch.sigmoid(model.halt_head(x)).squeeze(-1).float()  # same rounding as the reference
            survival = survival * (1.0 - p_stop)
            active = active & (survival >= threshold)
    return model._head(x), counts


def static_forward(model, token_ids: Tensor) -> Tensor:
    """Logits of the sync-free path: static hard AM-LLM, or the ordinary forward for Dense/MoE controls."""
    if isinstance(model, AdaptiveModularLM):
        return static_hard_forward(model, token_ids)[0]
    with torch.no_grad():
        return model(token_ids)[0]


class GraphRunner:
    """Capture `fn(x)` once in a CUDA Graph with a static input buffer; replay for new inputs of the same shape."""

    def __init__(self, fn, example: Tensor, warmup: int = 3):
        self.static_in = example.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                fn(self.static_in)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = fn(self.static_in)

    def __call__(self, x: Tensor) -> Tensor:
        self.static_in.copy_(x)
        self.graph.replay()
        return self.static_out
