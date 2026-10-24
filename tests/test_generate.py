"""F4: incremental decoding with a KV-cache must reproduce the full prefix forward (written before generate.py)."""
import pytest
import torch

from amllm.config import ModelConfig
from amllm.model import AdaptiveModularLM, build_model

VARIANTS = {
    "amllm_shared_attention": dict(arch="amllm"),
    "amllm_per_hop_attention": dict(arch="amllm", unshared_attention=True),
    "amllm_ua_module_top1_st": dict(arch="amllm", unshared_attention=True, module_topk=1, route_weighting="straight_through"),
    "amllm_ua_expert_top1_fs": dict(arch="amllm", unshared_attention=True, expert_topk=1, route_weighting="full_softmax"),
    # W2/W3 switches: per-hop banks with flat routing (FDC1-like), per-hop bank norms, adaptive K (AD03)
    "amllm_fdc1_like": dict(arch="amllm", unshared_attention=True, bank_groups=2, n_modules=1, experts_per_module=6,
                            module_topk=1, route_weighting="straight_through"),
    "amllm_bank_norm_per_hop": dict(arch="amllm", unshared_attention=True, bank_groups=4, bank_norm="per_hop"),
    "amllm_hop_affine": dict(arch="amllm", bank_norm="hop_affine"),
    "amllm_adaptive_k": dict(arch="amllm", unshared_attention=True, expert_topp=0.6, route_weighting="straight_through"),
    "moe_adaptive_k": dict(arch="moe", expert_topp=0.6),
    "moe": dict(arch="moe"),
    "dense": dict(arch="dense"),
}


def make(name, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    base = dict(max_seq_len=40, d_model=16, n_heads=4, n_modules=4, experts_per_module=3, module_topk=2,
                expert_topk=2, n_hops=4, expert_hidden=24, n_layers=3, ffn_hidden=32, n_experts=4, moe_topk=2)
    model = build_model(ModelConfig(**{**base, **VARIANTS[name]})).to(dtype).eval()
    if isinstance(model, AdaptiveModularLM):
        with torch.no_grad():
            model.halt_head.weight.normal_(0, 2.0)  # tokens halt after different numbers of hops
            model.halt_head.bias.fill_(0.3)
            if hasattr(model, "hop_scale"):  # move the per-hop affine away from identity so it really matters
                model.hop_scale.normal_(1.0, 0.3)
                model.hop_shift.normal_(0.0, 0.3)
    if model.cfg.expert_topp > 0:  # a sharp router (AM-LLM and MoE), so that adaptive K really drops routes
        with torch.no_grad():
            for bank in (m for m in model.modules() if hasattr(m, "expert_router_w")):
                bank.expert_router_w.mul_(200.0)
    return model


def reference_logits(model, ids):
    with torch.no_grad():
        if isinstance(model, AdaptiveModularLM):
            logits, info = model(ids, hard_halt=True)
            return logits, info["hops_per_token"]
        return model(ids)[0], None


@pytest.mark.parametrize("name", list(VARIANTS))
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-10), (torch.float32, 1e-4)])
def test_incremental_decoding_equals_full_forward(name, dtype, tol):
    from amllm.generate import Generator
    model = make(name, dtype)
    ids = torch.randint(0, 256, (2, 40))
    ref, ref_hops = reference_logits(model, ids)
    gen = Generator(model, batch=2, max_len=40)
    with torch.no_grad():
        out = [gen.prefill(ids[:, :24])]                       # logits for positions 0..23
        for p in range(24, 40):
            out.append(gen.decode_step(ids[:, p])[:, None])    # logits for position p
    logits = torch.cat(out, dim=1)
    assert (logits - ref).abs().max() <= tol
    if ref_hops is not None and dtype == torch.float64:
        assert torch.equal(gen.hops_per_token[:, :40], ref_hops)


def test_cache_size_formula():
    from amllm.generate import Generator
    model = make("amllm_per_hop_attention", torch.float32)
    gen = Generator(model, batch=3, max_len=40)
    units = model.cfg.n_hops
    assert gen.cache_bytes() == units * 2 * 3 * 40 * model.cfg.d_model * 4


@pytest.mark.parametrize("name", ["amllm_adaptive_k", "moe_adaptive_k"])
def test_adaptive_k_variants_really_drop_routes(name):
    model = make(name)
    with torch.no_grad():
        info = model(torch.randint(0, 256, (2, 40)), **({"hard_halt": True} if isinstance(model, AdaptiveModularLM) else {}))[1]
    routed = int(info["expert_counts"].sum())
    if isinstance(model, AdaptiveModularLM):
        full = int(info["hops_per_token"].sum()) * model.cfg.module_topk * model.cfg.expert_topk
    else:
        full = 2 * 40 * model.cfg.n_layers * model.cfg.moe_topk
    assert 0 < routed < full
