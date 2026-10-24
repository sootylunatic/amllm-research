"""Exploratory decode variants (amllm.fastdecode) must compute the same model as the reference path."""
import pytest
import torch

from amllm import fastdecode
from amllm.config import ModelConfig
from amllm.generate import Generator
from amllm.model import AdaptiveModularLM, build_model

BASE = dict(max_seq_len=40, d_model=16, n_heads=4, n_modules=4, experts_per_module=3, module_topk=2, expert_topk=2,
            n_hops=4, expert_hidden=24, n_layers=3, ffn_hidden=32, n_experts=4, moe_topk=2)
KINDS = {
    "amllm_hier_top2": dict(arch="amllm", unshared_attention=True),
    "fdc1_like": dict(arch="amllm", halt_rule="none", unshared_attention=True, bank_groups=2, n_modules=1,
                      experts_per_module=6, module_topk=1, route_weighting="straight_through"),
    "adaptive_k": dict(arch="amllm", unshared_attention=True, expert_topp=0.6, route_weighting="straight_through"),
    "moe": dict(arch="moe"),
    "moe_adaptive_k": dict(arch="moe", expert_topp=0.6),
}


def make(kind, dtype=torch.float64):
    torch.manual_seed(0)
    model = build_model(ModelConfig(**{**BASE, **KINDS[kind]})).to(dtype).eval()
    with torch.no_grad():
        if isinstance(model, AdaptiveModularLM):
            model.halt_head.weight.normal_(0, 2.0)
            model.halt_head.bias.fill_(0.3)
        if model.cfg.expert_topp > 0:  # sharp router: adaptive K really drops routes
            for m in model.modules():
                if hasattr(m, "expert_router_w"):
                    m.expert_router_w.mul_(200.0)
    return model


def decode_logits(model, ids):
    gen = Generator(model, batch=ids.shape[0], max_len=ids.shape[1])
    with torch.no_grad():
        out = [gen.prefill(ids[:, :20])]
        for p in range(20, ids.shape[1]):
            out.append(gen.decode_step(ids[:, p])[:, None])
    return torch.cat(out, 1)


@pytest.mark.parametrize("kind", list(KINDS))
def test_gather_path_equals_the_reference_fp64(kind):
    model = make(kind)
    ids = torch.randint(0, 256, (2, 40))
    ref = decode_logits(model, ids)
    with torch.no_grad():
        ref_full = model(ids, **({"hard_halt": True} if isinstance(model, AdaptiveModularLM) else {}))[0]
    fastdecode.use_gather(model)
    out = decode_logits(model, ids)
    with torch.no_grad():
        out_full = model(ids, **({"hard_halt": True} if isinstance(model, AdaptiveModularLM) else {}))[0]
    fastdecode.use_gather(model, False)
    assert (out - ref).abs().max() <= 1e-10
    assert (out_full - ref_full).abs().max() <= 1e-10


def test_use_gather_can_be_switched_off():
    model = make("moe")
    fastdecode.use_gather(model)
    fastdecode.use_gather(model, False)
    assert not any("forward" in m.__dict__ for m in model.modules())


@pytest.mark.parametrize("kind", ["fdc1_like", "moe"])
def test_bf16_weights_stay_close_to_fp32(kind):
    model = make(kind, torch.float32)
    ids = torch.randint(0, 256, (2, 40))
    ref = decode_logits(model, ids)
    out = decode_logits(fastdecode.to_bf16(model), ids).float()
    assert (out - ref).abs().mean() < 0.02          # bf16 rounding; single logits of a random tiny model can move more
    assert (out.argmax(-1) == ref.argmax(-1)).float().mean() > 0.9


def test_halting_model_runs_in_pure_bf16_on_every_execution_path():
    """Regression: the fp32 survival weight promoted the residual state to fp32, which then met bf16 LayerNorm weights."""
    from amllm.fastpath import bucketed_hard_forward, static_hard_forward
    model = make("adaptive_k", torch.float32)
    ids = torch.randint(0, 256, (2, 40))
    ref = decode_logits(model, ids)
    fastdecode.to_bf16(model)
    out = decode_logits(model, ids).float()
    assert (out.argmax(-1) == ref.argmax(-1)).float().mean() > 0.9
    with torch.no_grad():
        assert static_hard_forward(model, ids)[0].dtype == torch.bfloat16
        assert bucketed_hard_forward(model, ids)[0].dtype == torch.bfloat16
    fastdecode.use_gather(model)
    decode_logits(model, ids)
