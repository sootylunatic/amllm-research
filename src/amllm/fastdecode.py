"""Exploratory decode-speed variants for goal C (W3 side experiment). Execution code only, not part of train_code_sha256:
nothing here changes what a trained model computes, only how its inference is executed.

  to_bf16(model)       cast the weights once; the Triton expert kernel otherwise converts the whole expert stack to bf16
                       on every call (measured: ~150 copy kernels per decode step)
  use_gather(model)    every module bank computes its routed experts by gathering the chosen experts' weights and doing
                       one batched product per route: no sort, no per-expert counts, no scatter, static shapes. Adaptive K
                       (expert_topp) runs every top-k route with the dropped ones at weight 0, so it is CUDA-Graph
                       capturable (same output, no compute saving). Meant for small decode batches: the gathered weights
                       take rows x k x d x hidden values.
The reference path stays available (use_gather(model, False)); tests compare both.
"""
from __future__ import annotations

import types

import torch
from torch import Tensor
from torch.nn import functional as F

from .routing import HierarchicalModuleBank, count_ids


def to_bf16(model):
    return model.to(torch.bfloat16)


def gather_forward(bank: HierarchicalModuleBank, x: Tensor):
    """Same routing, output and usage counts as HierarchicalModuleBank.forward; the balance term is 0 (inference)."""
    n, M, E = x.shape[0], bank.n_modules, bank.n_experts
    m_scores = bank.router(x)
    m_idx, m_w = bank._select(m_scores, bank.module_k)                                    # [N,Km]
    e_scores = torch.einsum("nd,nkde->nke", x, bank.expert_router_w[m_idx]) + bank.expert_router_b[m_idx]
    if bank.expert_topp > 0:
        e_idx, e_w, keep = bank._select_topp(e_scores, bank.expert_k, bank.expert_topp)
        e_w = e_w * keep                                                                 # dropped routes: weight 0
    else:
        e_idx, e_w = bank._select(e_scores, bank.expert_k)                               # [N,Km,Ke]
        keep = torch.ones_like(e_idx, dtype=torch.bool)
    gid = (m_idx[..., None] * E + e_idx).reshape(n, -1)                                  # [N,R]
    ex = bank.experts
    h = F.silu(torch.einsum("nd,nrdh->nrh", x, ex.w1[gid]) + ex.b1[gid])                 # [N,R,H]
    y = torch.einsum("nrh,nrhd->nrd", h, ex.w2[gid]) + ex.b2[gid]                        # [N,R,D]
    weight = (m_w[..., None] * e_w).reshape(n, -1)
    out = (y * weight[..., None].to(y.dtype)).sum(1).to(x.dtype)
    # counts of the routes really used (static shape: dropped routes are counted into a discarded extra bin)
    used = torch.where(keep.reshape(n, -1), gid, torch.full_like(gid, M * E))
    expert_counts = count_ids(used.flatten(), M * E + 1)[:-1].view(M, E)
    return out, x.new_zeros(()), count_ids(m_idx.flatten(), M), expert_counts


def use_gather(model, enabled: bool = True):
    for m in model.modules():
        if isinstance(m, HierarchicalModuleBank):
            if enabled:
                m.forward = types.MethodType(gather_forward, m)
            elif "forward" in m.__dict__:
                del m.forward
    return model
