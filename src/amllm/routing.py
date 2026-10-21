"""Sparse, differentiably weighted top-k dispatch. Educational reference, not a fused kernel.

Only selected experts run. top-k *indices* have no gradient, while softmax weights of the
selected routes do. Avoid calling this fully differentiable routing.

All experts of all modules live in one stacked ExpertBank, so one hop needs a single host
sync (the per-expert token counts) instead of one `nonzero()` per module and per expert.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def count_ids(ids: Tensor, n: int) -> Tensor:
    """bincount without a host sync (torch.bincount must read the max id back to the CPU)."""
    return torch.zeros(n, dtype=torch.long, device=ids.device).scatter_add_(0, ids, torch.ones_like(ids))


class ExpertBank(nn.Module):
    """n independent SiLU FFN experts stored as stacked weight tensors.

    dispatch="loop": one matmul pair per *selected* expert; unselected experts do no work.
    dispatch="bmm":  pad every expert's tokens to the largest group and run one batched
                     matmul (a grouped-GEMM stand-in). Few kernels, but the padding rows and
                     experts with zero tokens are computed too, so FLOPs are NOT skipped.
    dispatch="grouped": torch._grouped_mm with device-side offsets; only routed rows, no padding. On
                     sm_89 PyTorch copies the offsets to the host and launches one kernel per group,
                     so it is NOT sync-free there (measured, phase 2 F3).
    dispatch="triton": own Triton grouped GEMM (triton_grouped.py): routed rows only, no host sync,
                     CUDA-Graph capturable; inference only.
    """

    def __init__(self, n: int, width: int, hidden: int):
        super().__init__()
        self.n = n
        self.w1 = nn.Parameter(torch.randn(n, width, hidden) / math.sqrt(width))
        self.b1 = nn.Parameter(torch.zeros(n, hidden))
        self.w2 = nn.Parameter(torch.randn(n, hidden, width) / math.sqrt(hidden) * 0.5)
        self.b2 = nn.Parameter(torch.zeros(n, width))
        self.dispatch = "loop"
        self.last_executed: list[int] = []  # expert ids actually evaluated in the last call
        self.rows_computed = 0              # FFN rows actually computed (incl. padding), cumulative

    def forward(self, x: Tensor, token_idx: Tensor, expert_idx: Tensor, weight: Tensor,
                routes_per_token: int | None = None) -> tuple[Tensor, Tensor]:
        """x [N,D]; token_idx/expert_idx/weight [A]: one entry per (token, route) assignment.

        routes_per_token=k declares that assignment a belongs to token a // k (assignments grouped by
        token). The k expert outputs of a token are then summed with a fixed-order reduction instead of an
        atomic index_add, which makes the CUDA forward bitwise deterministic."""
        counts = count_ids(expert_idx, self.n)
        order = expert_idx.argsort(stable=True)
        tok = token_idx[order]
        xs = x.index_select(0, tok)
        if self.dispatch == "bmm":
            y = self._expert_outputs_bmm(xs, expert_idx[order], counts)
        elif self.dispatch == "grouped":
            y = self._expert_outputs_grouped(xs, expert_idx[order], counts)
        elif self.dispatch == "triton":
            from .triton_grouped import grouped_ffn  # CUDA + Triton only; inference only (no autograd)
            dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else xs.dtype
            y = grouped_ffn(xs.to(dtype), self.w1, self.b1, self.w2, self.b2, counts)
            self.last_executed = []
            self.rows_computed += xs.shape[0]
        else:
            y = self._expert_outputs_loop(xs, counts)
        if y is None:
            return torch.zeros_like(x), counts
        y = (y * weight[order, None]).to(x.dtype)
        if routes_per_token is not None:
            per_assignment = torch.empty_like(y).index_copy_(0, order, y)  # back to token-grouped order
            return per_assignment.view(x.shape[0], routes_per_token, -1).sum(1), counts
        return torch.zeros_like(x).index_add(0, tok, y), counts

    def _expert_outputs_loop(self, xs: Tensor, counts: Tensor) -> Tensor | None:
        outs, executed, start = [], [], 0
        for g, c in enumerate(counts.tolist()):  # the only host sync of the dispatch
            if c == 0:
                continue
            h = F.silu(xs[start:start + c] @ self.w1[g] + self.b1[g])
            outs.append(h @ self.w2[g] + self.b2[g])
            executed.append(g)
            start += c
        self.last_executed = executed
        self.rows_computed += start
        return torch.cat(outs) if outs else None

    def _expert_outputs_grouped(self, xs: Tensor, eid: Tensor, counts: Tensor) -> Tensor:
        """Grouped GEMM with device-side group offsets (CUDA): only routed rows are computed, no padding and no
        host synchronisation, so the dispatch can be captured in a CUDA Graph (phase 2, F3)."""
        offs = torch.cumsum(counts, 0).to(torch.int32)
        dtype = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else xs.dtype
        xs = xs.to(dtype)
        h = torch._grouped_mm(xs, self.w1.to(dtype), offs=offs) + self.b1.to(dtype)[eid]
        y = torch._grouped_mm(F.silu(h), self.w2.to(dtype), offs=offs) + self.b2.to(dtype)[eid]
        self.last_executed = []  # not materialised: listing the experts would need a host sync
        self.rows_computed += xs.shape[0]
        return y

    def _expert_outputs_bmm(self, xs: Tensor, eid: Tensor, counts: Tensor) -> Tensor:
        cap = int(counts.max())  # the only host sync of the dispatch
        offsets = torch.cumsum(counts, 0) - counts
        pos = torch.arange(eid.numel(), device=xs.device) - offsets[eid]
        buf = xs.new_zeros(self.n, cap, xs.shape[1])
        buf[eid, pos] = xs
        h = F.silu(torch.baddbmm(self.b1[:, None, :].to(buf.dtype), buf, self.w1.to(buf.dtype)))
        out = torch.baddbmm(self.b2[:, None, :].to(buf.dtype), h, self.w2.to(buf.dtype))
        self.last_executed = list(range(self.n))
        self.rows_computed += self.n * cap
        return out[eid, pos]


class HierarchicalModuleBank(nn.Module):
    """Token -> top-k modules -> top-k experts inside each selected module.

    The expert router of a module is only evaluated for tokens that selected that module.
    A plain MoE layer is the special case n_modules=1, module_k=1.

    weighting="topk_softmax": route weights = softmax over the *selected* scores (renormalised).
        With k=1 the weight is identically 1, so that router gets no language-model gradient.
    weighting="full_softmax": route weights = softmax over *all* scores, gathered at the selected
        routes (Switch-Transformer style); the router gets gradient for every k.
    weighting="straight_through": forward value of topk_softmax, gradient additionally through the full softmax
        probability of the selected routes (straight-through), so k=1 routers learn without changing the forward.
    router_mode="random": route choices are uniform random and weights uniform (1/k): a training-time
        "no learned routing" control with identical parameters and compute.
    """

    def __init__(self, width: int, n_modules: int, module_k: int,
                 experts_per_module: int, expert_k: int, hidden: int,
                 weighting: str = "topk_softmax", router_mode: str = "learned"):
        super().__init__()
        assert weighting in ("topk_softmax", "full_softmax", "straight_through") and router_mode in ("learned", "random")
        self.n_modules, self.module_k = n_modules, module_k
        self.n_experts, self.expert_k = experts_per_module, expert_k
        self.weighting, self.router_mode = weighting, router_mode
        self.router = nn.Linear(width, n_modules)
        self.expert_router_w = nn.Parameter(torch.randn(n_modules, width, experts_per_module) * 0.02)
        self.expert_router_b = nn.Parameter(torch.zeros(n_modules, experts_per_module))
        self.experts = ExpertBank(n_modules * experts_per_module, width, hidden)
        # Eval diagnostic: replace learned route *choices* by uniform random ones
        # (mixing weights stay the learned scores at the random picks).
        self.random_routing = False
        # False = inference fast path: no balance loss and no module statistics (fewer kernels, no extra syncs).
        self.diagnostics = True
        self.last_route: tuple[Tensor, Tensor] | None = None  # (module idx [N,Km], global expert idx [N,Km,Ke])

    def _select(self, scores: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Returns (indices, weights) of the k chosen routes along the last dim."""
        if self.router_mode == "random":
            idx = torch.rand(scores.shape, device=scores.device).topk(k, dim=-1).indices
            return idx, torch.full(idx.shape, 1.0 / k, device=scores.device, dtype=scores.dtype)
        if self.random_routing:
            idx = torch.rand(scores.shape, device=scores.device).topk(k, dim=-1).indices
        else:
            idx = scores.topk(k, dim=-1).indices
        if self.weighting == "full_softmax":
            return idx, scores.softmax(-1).gather(-1, idx)
        weights = scores.gather(-1, idx).softmax(-1)
        if self.weighting == "straight_through":
            # forward value = renormalised top-k weight; backward also through the full softmax probability,
            # so a top-1 router receives LM gradient without changing the forward computation (F1b)
            full = scores.softmax(-1).gather(-1, idx)
            weights = weights + (full - full.detach())  # the bracket is exactly 0 in the forward pass
        return idx, weights

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        n, M, E = x.shape[0], self.n_modules, self.n_experts
        if n == 0:
            return (x, x.new_zeros(()), torch.zeros(M, dtype=torch.long, device=x.device),
                    torch.zeros(M, E, dtype=torch.long, device=x.device))
        m_scores = self.router(x)                                         # [N,M]
        m_idx, m_w = self._select(m_scores, self.module_k)                # [N,Km]
        e_scores = torch.einsum("nd,nkde->nke", x, self.expert_router_w[m_idx]) + self.expert_router_b[m_idx]
        e_idx, e_w = self._select(e_scores, self.expert_k)                # [N,Km,Ke]
        self.last_route = (m_idx.detach(), (m_idx[..., None] * E + e_idx).detach())

        weight = (m_w[..., None] * e_w).reshape(-1)
        global_id = (m_idx[..., None] * E + e_idx).reshape(-1)
        token = torch.arange(n, device=x.device)[:, None, None].expand_as(e_idx).reshape(-1)
        y, counts = self.experts(x, token, global_id, weight, routes_per_token=self.module_k * self.expert_k)
        expert_counts = counts.view(M, E)
        if not self.diagnostics:
            return y, x.new_zeros(()), torch.zeros(M, dtype=torch.long, device=x.device), expert_counts

        module_counts = count_ids(m_idx.flatten(), M)
        # Switch-style balance: differentiable importance x detached load.
        m_imp = m_scores.float().softmax(-1).mean(0)
        m_load = module_counts.float() / (n * self.module_k)
        module_balance = M * torch.sum(m_imp * m_load)
        e_prob = e_scores.float().softmax(-1).reshape(-1, E)
        sel = module_counts.clamp(min=1).float()[:, None]
        e_imp = torch.zeros(M, E, device=x.device).index_add(0, m_idx.flatten(), e_prob) / sel
        e_load = expert_counts.float() / (sel * self.expert_k)
        expert_balance = E * torch.sum(m_load[:, None] * e_imp * e_load)
        return y, module_balance + expert_balance, module_counts, expert_counts
