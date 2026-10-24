"""F4: autoregressive generation with a KV-cache (docs/PHASE2_PLAN.md §6, pre-registration F4).

AM-LLM uses the static masked hard semantics (identical to `forward(hard_halt=True)`, see fastpath.py): every new
token runs all hops with its own survival weight (halted: weight 0); its K/V at hop h are computed from its state
entering hop h, exactly as in the full forward, and appended to that hop's cache. Dense/MoE controls use one cache per
layer. Caches are preallocated and the position lives on the device, so `decode_step` has static shapes and can be
captured in a CUDA Graph (`GraphDecoder`). Execution code, not training code.
"""
from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .model import AdaptiveModularLM, CausalAttention


class Generator:
    def __init__(self, model, batch: int, max_len: int, cache_dtype: torch.dtype | None = None):
        self.model = model
        cfg = model.cfg
        self.is_amllm = isinstance(model, AdaptiveModularLM)
        if self.is_amllm:
            from .fastpath import require_supported
            require_supported(model)
        if self.is_amllm and not (cfg.attn_per_hop and cfg.halt_rule in ("survival", "none")):
            raise ValueError("KV-cache decoding supports AM-LLM with attention in every hop and survival/no halting")
        self.units = cfg.n_hops if self.is_amllm else cfg.n_layers
        param = next(model.parameters())
        dtype = cache_dtype or param.dtype
        heads, d = cfg.n_heads, cfg.d_model
        shape = (self.units, batch, heads, max_len, d // heads)
        self.k = torch.zeros(shape, device=param.device, dtype=dtype)
        self.v = torch.zeros(shape, device=param.device, dtype=dtype)
        self.pos = torch.zeros((), dtype=torch.long, device=param.device)  # next position to write
        self.hops_per_token = torch.zeros(batch, max_len, dtype=torch.long, device=param.device)
        self.max_len = max_len

    def cache_bytes(self) -> int:
        return 2 * self.k.numel() * self.k.element_size()

    def reset(self) -> None:
        self.pos.zero_()
        self.hops_per_token.zero_()

    def _attend(self, attn: CausalAttention, h: Tensor, positions: Tensor, unit: int) -> Tensor:
        """Write K/V of the new tokens at `positions` into the cache of `unit`, attend causally over the cache."""
        b, s, d = h.shape
        heads = attn.n_heads
        q, k, v = attn.qkv(h).chunk(3, dim=-1)

        def split(y: Tensor) -> Tensor:
            return y.reshape(b, s, heads, d // heads).transpose(1, 2).to(self.k.dtype)

        self.k[unit].index_copy_(2, positions, split(k))
        self.v[unit].index_copy_(2, positions, split(v))
        keys = torch.arange(self.max_len, device=h.device)
        mask = keys[None, :] <= positions[:, None]                                   # [s, max_len]
        out = F.scaled_dot_product_attention(split(q), self.k[unit], self.v[unit], attn_mask=mask[None, None])
        return attn.out(out.transpose(1, 2).reshape(b, s, d).to(h.dtype))

    def _run(self, ids: Tensor, positions: Tensor) -> Tensor:
        m, cfg = self.model, self.model.cfg
        d = cfg.d_model
        x = m.embed(ids) + m.positions(positions)[None]
        b, s = ids.shape
        if self.is_amllm:
            survival = x.new_ones(b, s)
            active = torch.ones(b, s, dtype=torch.bool, device=x.device)
            counts = torch.zeros(b, s, dtype=torch.long, device=x.device)
            for hop in range(cfg.n_hops):
                weight = (survival * active)[..., None].to(x.dtype)  # survival stays fp32; bf16 weights keep a bf16 state
                if cfg.unshared_attention and hop > 0:
                    attn, norm = m.hop_attentions[hop - 1], m.hop_attn_norms[hop - 1]
                else:
                    attn, norm = m.attention, m.attn_norm
                x = x + weight * self._attend(attn, norm(x), positions, hop)
                delta, _, _, _ = m.bank_for(hop)(m.normed(hop, x.reshape(-1, d)))
                x = x + weight * delta.view(b, s, d)
                counts += active
                if hop < cfg.n_hops - 1 and cfg.halt_rule == "survival":
                    p_stop = torch.sigmoid(m.halt_head(x)).squeeze(-1).float()  # same rounding as the reference
                    survival = survival * (1.0 - p_stop)
                    active = active & (survival >= cfg.halt_threshold)
            self.hops_per_token.index_copy_(1, positions, counts)
        else:
            for i in range(cfg.n_layers):
                x = x + self._attend(m.attns[i], m.attn_norms[i](x), positions, i)
                h = m.ffn_norms[i](x)
                if cfg.arch == "dense":
                    x = x + m.ffns[i](h)
                else:
                    y, _, _, _ = m.ffns[i](h.reshape(-1, d))
                    x = x + y.view(b, s, d)
        return m._head(x)

    @torch.no_grad()
    def prefill(self, ids: Tensor) -> Tensor:
        """Process a prompt [B, S] from position 0; returns logits for every prompt position [B, S, V]."""
        self.reset()
        positions = torch.arange(ids.shape[1], device=ids.device)
        logits = self._run(ids, positions)
        self.pos.fill_(ids.shape[1])
        return logits

    @torch.no_grad()
    def decode_step(self, token: Tensor) -> Tensor:
        """Append one token per sequence [B] at the current position; returns its next-token logits [B, V]."""
        logits = self._run(token[:, None], self.pos.view(1))[:, 0]
        self.pos += 1
        return logits


class GraphDecoder:
    """`decode_step` captured once in a CUDA Graph; `__call__(token)` replays it (static shapes, device position)."""

    def __init__(self, gen: Generator, example_token: Tensor, autocast_dtype: torch.dtype | None = torch.bfloat16):
        self.gen = gen
        self.token = example_token.clone()
        self.autocast_dtype = autocast_dtype
        saved_pos = gen.pos.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # warm-up (writes into the cache are rolled back below)
                gen.pos.copy_(saved_pos)
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        gen.pos.copy_(saved_pos)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._step()
        gen.pos.copy_(saved_pos)  # capture does not execute; restore the position for the first replay

    def _step(self) -> Tensor:
        with torch.autocast("cuda", dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None):
            return self.gen.decode_step(self.token)

    def __call__(self, token: Tensor) -> Tensor:
        self.token.copy_(token)
        self.graph.replay()
        return self.logits
