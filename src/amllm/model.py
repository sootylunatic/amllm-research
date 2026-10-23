"""Byte-level causal LMs for the AM-LLM experiments.

- AdaptiveModularLM: a shared module bank applied for up to n_hops recurrent hops with
  per-token halting. With attn_per_hop=True every hop also runs a *shared* causal attention
  (Universal-Transformer style); with False it is the v0.1 topology (one attention in front).
- LayeredLM: fixed-depth dense or MoE control models with per-layer attention.

Training uses soft survival weights over all hops; hard inference really skips the module
bank for halted tokens. Halted tokens keep their frozen state and still serve as attention
keys/values, so attention itself is not skipped in this prototype.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .routing import HierarchicalModuleBank


def attention_flops(d: int, t: int) -> float:
    """Forward FLOPs per query token: qkv/out projections + causal QK^T and AV (avg (t+1)/2 keys)."""
    return 8 * d * d + 2 * d * (t + 1)


def ffn_flops(d: int, hidden: int) -> float:
    return 4 * d * hidden


class CausalAttention(nn.Module):
    def __init__(self, width: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def split(y: Tensor) -> Tensor:
            return y.reshape(b, t, self.n_heads, d // self.n_heads).transpose(1, 2)
        attn = F.scaled_dot_product_attention(split(q), split(k), split(v), is_causal=True)
        return self.out(attn.transpose(1, 2).reshape(b, t, d))


class FeedForward(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Linear(hidden, width))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _ByteLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.positions = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # False = inference fast path: skip routing statistics, balance loss and the FLOP report
        # (which reads counts back to the host). Use set_diagnostics(); latency is measured both ways.
        self.diagnostics = True

    def _embed(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be int64 tensor of shape [batch, length]")
        t = token_ids.shape[1]
        if t > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds configured {self.cfg.max_seq_len}")
        return self.embed(token_ids) + self.positions(torch.arange(t, device=token_ids.device))[None]

    def _head(self, x: Tensor) -> Tensor:
        return self.lm_head(self.out_norm(x))


class LayeredLM(_ByteLM):
    """Fixed-depth control: dense FFN or top-k MoE FFN in each of n_layers blocks."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        d = cfg.d_model
        self.attn_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(cfg.n_layers))
        self.attns = nn.ModuleList(CausalAttention(d, cfg.n_heads) for _ in range(cfg.n_layers))
        self.ffn_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(cfg.n_layers))
        if cfg.arch == "dense":
            self.ffns = nn.ModuleList(FeedForward(d, cfg.ffn_hidden) for _ in range(cfg.n_layers))
        else:
            self.ffns = nn.ModuleList(
                HierarchicalModuleBank(d, 1, 1, cfg.n_experts, cfg.moe_topk, cfg.expert_hidden,
                                       cfg.route_weighting, cfg.router_mode, cfg.expert_topp)
                for _ in range(cfg.n_layers))

    def forward(self, token_ids: Tensor, *, trace: bool = False, **_) -> tuple[Tensor, dict]:
        cfg, d = self.cfg, self.cfg.d_model
        route_trace = []
        x = self._embed(token_ids)
        b, t = token_ids.shape
        balance = x.new_zeros(())
        rows0 = sum(f.experts.rows_computed for f in self.ffns) if cfg.arch != "dense" else 0
        ec = torch.zeros(cfg.n_layers, cfg.n_experts, dtype=torch.long, device=x.device)
        for i in range(cfg.n_layers):
            x = x + self.attns[i](self.attn_norms[i](x))
            h = self.ffn_norms[i](x)
            if cfg.arch == "dense":
                x = x + self.ffns[i](h)
            else:
                y, bal, _, counts = self.ffns[i](h.reshape(-1, d))
                x = x + y.view(b, t, d)
                balance = balance + bal
                ec[i] = counts.view(-1)
                if trace:
                    m_idx, g_idx = self.ffns[i].last_route
                    route_trace.append({"positions": torch.arange(b * t), "modules": m_idx.cpu(), "experts": g_idx.cpu()})
        if cfg.arch == "dense":
            per_layer = attention_flops(d, t) + ffn_flops(d, cfg.ffn_hidden)
        else:
            per_layer = (attention_flops(d, t) + cfg.moe_topk * ffn_flops(d, cfg.expert_hidden)
                         + 2 * d * (1 + cfg.n_experts))
        flops = cfg.n_layers * per_layer + 2 * d * cfg.vocab_size
        routed = cfg.n_layers * cfg.moe_topk * b * t
        if cfg.arch != "dense" and cfg.expert_topp > 0:  # AD03 adaptive K: count the routes really dispatched
            routed = int(ec.sum())
            flops += (routed - cfg.n_layers * cfg.moe_topk * b * t) * ffn_flops(d, cfg.expert_hidden) / (b * t)
        executed, rows_pt = float(flops), 0.0
        if cfg.arch != "dense":  # padding rows of the bmm dispatch are real work
            rows = sum(f.experts.rows_computed for f in self.ffns) - rows0
            executed += max(0.0, rows - routed) * ffn_flops(d, cfg.expert_hidden) / (b * t)
            rows_pt = rows / (b * t)
        hops = torch.full((b, t), cfg.n_layers, device=x.device)
        return self._head(x), {
            "aux_loss": cfg.balance_coefficient * balance / cfg.n_layers,
            "attention_applications": cfg.n_layers, "attention_query_fraction": 1.0,
            "mean_hops": hops.float().mean(), "expected_hops": hops.float().mean(),
            "hops_per_token": hops, "module_counts": torch.zeros(0, dtype=torch.long),
            "expert_counts": ec, "flops_per_token": float(flops), "flops_per_token_ideal": float(flops),
            "flops_per_token_executed": executed, "expert_rows_per_token": rows_pt,
            "trace": route_trace,
        }


class AdaptiveModularLM(_ByteLM):
    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        d = cfg.d_model
        self.attn_norm = nn.LayerNorm(d)
        self.attention = CausalAttention(d, cfg.n_heads)
        if cfg.unshared_attention:
            assert cfg.attn_per_hop
            self.hop_attn_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(cfg.n_hops - 1))
            self.hop_attentions = nn.ModuleList(CausalAttention(d, cfg.n_heads) for _ in range(cfg.n_hops - 1))
        self.step_norm = nn.LayerNorm(d)
        self.bank = HierarchicalModuleBank(d, cfg.n_modules, cfg.module_topk,
                                           cfg.experts_per_module, cfg.expert_topk, cfg.expert_hidden,
                                           cfg.route_weighting, cfg.router_mode, cfg.expert_topp)
        self.halt_head = nn.Linear(d, 1)
        nn.init.zeros_(self.halt_head.weight)
        nn.init.constant_(self.halt_head.bias, cfg.halt_bias_init)
        # FD01 / FD03 switches: created last so that default configurations keep their exact initialisation and state-dict keys
        if cfg.bank_groups > 1:
            self.extra_banks = nn.ModuleList(
                HierarchicalModuleBank(d, cfg.n_modules, cfg.module_topk, cfg.experts_per_module, cfg.expert_topk,
                                       cfg.expert_hidden, cfg.route_weighting, cfg.router_mode, cfg.expert_topp)
                for _ in range(cfg.bank_groups - 1))
        if cfg.bank_norm == "per_hop":
            self.extra_step_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(cfg.n_hops - 1))
        elif cfg.bank_norm == "hop_affine":
            self.hop_scale = nn.Parameter(torch.ones(cfg.n_hops - 1, d))
            self.hop_shift = nn.Parameter(torch.zeros(cfg.n_hops - 1, d))
        if cfg.attn_groups > 1:
            self.extra_attn_norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(cfg.attn_groups - 1))
            self.extra_attentions = nn.ModuleList(CausalAttention(d, cfg.n_heads) for _ in range(cfg.attn_groups - 1))
        if cfg.attn_adapter_rank > 0:
            r = cfg.attn_adapter_rank
            self.adapter_down = nn.ModuleList(nn.Linear(d, r, bias=False) for _ in range(cfg.n_hops - 1))
            self.adapter_up = nn.ModuleList(nn.Linear(r, d, bias=False) for _ in range(cfg.n_hops - 1))
            for up in self.adapter_up:
                nn.init.zeros_(up.weight)

    def attention_at(self, hop: int, x: Tensor) -> Tensor | None:
        """Attention update at `hop` (x: [b, t, d]); None on an expert-only hop (FD04 attn_every)."""
        cfg = self.cfg
        if hop % cfg.attn_every:
            return None
        if cfg.unshared_attention and hop > 0:
            return self.hop_attentions[hop - 1](self.hop_attn_norms[hop - 1](x))
        group = (hop // cfg.attn_every) * cfg.attn_groups // cfg.attention_hops
        if group == 0:
            a = self.attention(self.attn_norm(x))
        else:
            a = self.extra_attentions[group - 1](self.extra_attn_norms[group - 1](x))
        if cfg.attn_adapter_rank > 0 and hop > 0:
            a = a + self.adapter_up[hop - 1](self.adapter_down[hop - 1](a))
        return a

    def bank_for(self, hop: int) -> HierarchicalModuleBank:
        group = hop * self.cfg.bank_groups // self.cfg.n_hops
        return self.bank if group == 0 else self.extra_banks[group - 1]

    def normed(self, hop: int, x: Tensor) -> Tensor:
        """Normalisation in front of the bank at `hop` (x: [rows, d])."""
        mode = self.cfg.bank_norm
        if mode == "per_hop" and hop > 0:
            return self.extra_step_norms[hop - 1](x)
        y = self.step_norm(x)
        if mode == "hop_affine" and hop > 0:
            return y * self.hop_scale[hop - 1] + self.hop_shift[hop - 1]
        return y

    def forward(self, token_ids: Tensor, *, hard_halt: bool | None = None,
                halt_rule: str | None = None, halt_threshold: float | None = None,
                trace: bool = False) -> tuple[Tensor, dict]:
        cfg, d = self.cfg, self.cfg.d_model
        hard = (not self.training) if hard_halt is None else hard_halt
        rule = halt_rule or cfg.halt_rule
        threshold = cfg.halt_threshold if halt_threshold is None else halt_threshold
        b, t = token_ids.shape
        x = self._embed(token_ids)
        n_tok = b * t

        attn_apps, attn_query_tokens, adapter_apps = 0, 0.0, 0
        if not cfg.attn_per_hop:
            x = x + self.attention(self.attn_norm(x))
            attn_apps, attn_query_tokens = 1, float(n_tok)
        survival = x.new_ones(b, t)        # P(token still running before this hop)
        active = torch.ones(b, t, dtype=torch.bool, device=x.device)
        counts = torch.zeros(b, t, dtype=torch.long, device=x.device)
        # per-bank statistics: with bank_groups > 1 the banks are distinct experts, so their counts must not be summed
        mcs = [torch.zeros(cfg.n_modules, dtype=torch.long, device=x.device) for _ in range(cfg.bank_groups)]
        ecs = [torch.zeros(cfg.n_modules, cfg.experts_per_module, dtype=torch.long, device=x.device)
               for _ in range(cfg.bank_groups)]
        balance, ponder, hops_run = x.new_zeros(()), x.new_zeros(()), 0
        route_trace = []
        banks = [self.bank, *(self.extra_banks if cfg.bank_groups > 1 else [])]
        rows0 = sum(bk.experts.rows_computed for bk in banks)  # FFN rows computed so far (padding included)
        collect = self.training and cfg.anytime_weight > 0  # AD02a: keep the state after every hop for the exit loss
        hop_states = []

        for hop in range(cfg.n_hops):
            if hard:
                idx = active.flatten().nonzero().squeeze(1)
                n_live = idx.numel()
                if n_live == 0:
                    break
            weight = survival if rule == "survival" else torch.ones_like(survival)
            if hard:
                weight = weight * active
            ponder = ponder + survival.mean()
            a = self.attention_at(hop, x) if cfg.attn_per_hop else None
            if a is not None:
                x = x + weight[..., None] * a
                attn_apps += 1
                attn_query_tokens += n_live if hard else n_tok
                if cfg.attn_adapter_rank > 0 and hop > 0:
                    adapter_apps += 1
            flat = x.reshape(-1, d)
            w_flat = weight.reshape(-1, 1)
            if hard:
                bank = self.bank_for(hop)
                delta, bal, mc_step, ec_step = bank(self.normed(hop, flat.index_select(0, idx)))
                flat = flat.index_add(0, idx, delta * w_flat.index_select(0, idx))
                counts += active
            else:
                bank = self.bank_for(hop)
                delta, bal, mc_step, ec_step = bank(self.normed(hop, flat))
                flat = flat + w_flat * delta
                counts += 1
            x = flat.view(b, t, d)
            if collect:
                hop_states.append(x)
            if trace:
                m_idx, g_idx = bank.last_route
                pos = idx if hard else torch.arange(n_tok, device=x.device)
                route_trace.append({"positions": pos.cpu(), "modules": m_idx.cpu(), "experts": g_idx.cpu()})
            group = hop * cfg.bank_groups // cfg.n_hops
            mcs[group], ecs[group] = mcs[group] + mc_step, ecs[group] + ec_step
            balance, hops_run = balance + bal, hops_run + 1
            if hop < cfg.n_hops - 1 and rule != "none":
                p_stop = torch.sigmoid(self.halt_head(x)).squeeze(-1).float()
                if rule == "step" and hard:
                    active = active & (p_stop < threshold)
                survival = survival * (1.0 - p_stop)
                if rule == "survival" and hard:
                    active = active & (survival >= threshold)

        mc, ec = torch.cat(mcs), torch.cat(ecs)  # [G*M], [G*M, E]; identical to the old shapes when bank_groups == 1
        if hard:
            ponder = counts.float().mean()
        logits = self._head(x)
        if not self.diagnostics:
            out = {"aux_loss": x.new_zeros(()), "hops_per_token": counts, "trace": route_trace,
                   "attention_applications": attn_apps}
            if collect:
                out["hop_states"] = hop_states
            return logits, out
        expert_calls = float(ec.sum())
        router_flops = float(mc.sum()) / cfg.module_topk * 2 * d * cfg.n_modules + float(mc.sum()) * 2 * d * cfg.experts_per_module
        common = (expert_calls * ffn_flops(d, cfg.expert_hidden) + router_flops) / n_tok + 2 * d * cfg.vocab_size \
            + (2 * d * max(0, hops_run - 1) if rule != "none" else 0)
        common += adapter_apps * 4 * d * cfg.attn_adapter_rank  # FD04 adapter: down and up projection per query token
        flops = common + attn_apps * attention_flops(d, t)
        flops_ideal = common + attn_query_tokens / n_tok * attention_flops(d, t)
        # executed = routed work plus the padding / idle-expert rows the dispatch really computed (bmm pads every
        # expert's group to the largest one); equals `flops` for the loop and triton dispatch
        rows = sum(bk.experts.rows_computed for bk in banks) - rows0
        flops_executed = flops + max(0.0, rows - expert_calls) * ffn_flops(d, cfg.expert_hidden) / n_tok
        aux = cfg.ponder_coefficient * ponder + cfg.balance_coefficient * balance / max(1, hops_run)
        info = {
            "aux_loss": aux,
            "attention_applications": attn_apps,
            "attention_query_fraction": attn_query_tokens / n_tok,
            "mean_hops": counts.float().mean().detach(),
            "expected_hops": ponder.detach(),
            "hops_per_token": counts.detach(),
            "module_counts": mc.detach(),
            "expert_counts": ec.detach(),
            "flops_per_token": flops,
            "flops_per_token_ideal": flops_ideal,
            "flops_per_token_executed": flops_executed,
            "expert_rows_per_token": rows / n_tok,
            "trace": route_trace,
        }
        if collect:
            info["hop_states"] = hop_states
        return logits, info


def build_model(cfg: ModelConfig) -> nn.Module:
    return AdaptiveModularLM(cfg) if cfg.arch == "amllm" else LayeredLM(cfg)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    embed = sum(p.numel() for n, p in model.named_parameters()
                if n.startswith(("embed", "positions", "lm_head")))
    return {"total": total, "non_embedding": total - embed}


def expert_banks(model: nn.Module):
    from .routing import ExpertBank
    return [m for m in model.modules() if isinstance(m, ExpertBank)]


def set_dispatch(model: nn.Module, mode: str) -> None:
    assert mode in ("loop", "bmm", "grouped", "triton")
    for bank in expert_banks(model):
        bank.dispatch = mode


def set_diagnostics(model: nn.Module, enabled: bool) -> None:
    """Toggle routing statistics / balance loss / FLOP report (needed for training and evaluation)."""
    for m in model.modules():
        if isinstance(m, (HierarchicalModuleBank, _ByteLM)):
            m.diagnostics = enabled


def set_random_routing(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if isinstance(m, HierarchicalModuleBank):
            m.random_routing = enabled
