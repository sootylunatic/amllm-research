"""F2-1: the bucketed hard-halting execution must be the same mathematical model as the reference hard forward.

Written before the implementation (results/phase2/preregistration.md, F2-1)."""
import pytest
import torch

from amllm.config import ModelConfig
from amllm.fastpath import BUCKETS, bucketed_hard_forward
from amllm.model import AdaptiveModularLM

VARIANTS = {
    "shared_attention_top2": dict(),
    "per_hop_attention_top2": dict(unshared_attention=True),
    "per_hop_attention_module_top1_st": dict(unshared_attention=True, module_topk=1, route_weighting="straight_through"),
    "per_hop_attention_expert_top1_fs": dict(unshared_attention=True, expert_topk=1, route_weighting="full_softmax"),
    "fdc1_like": dict(unshared_attention=True, bank_groups=2, n_modules=1, experts_per_module=6, module_topk=1,
                      route_weighting="straight_through"),
    "bank_norm_per_hop": dict(unshared_attention=True, bank_groups=4, bank_norm="per_hop"),
    "hop_affine": dict(bank_norm="hop_affine"),
    "adaptive_k": dict(unshared_attention=True, expert_topp=0.6, route_weighting="straight_through"),
}


def model_with_varied_halting(seed=0, **kw):
    """A model whose tokens halt after different numbers of hops (random halting head)."""
    torch.manual_seed(seed)
    base = dict(max_seq_len=32, d_model=16, n_heads=4, n_modules=4, experts_per_module=3, module_topk=2,
                expert_topk=2, n_hops=4, expert_hidden=24)
    cfg = ModelConfig(**{**base, **kw})
    model = AdaptiveModularLM(cfg).double().eval()
    with torch.no_grad():
        model.halt_head.weight.normal_(0, 2.0)
        model.halt_head.bias.fill_(0.3)
        if model.cfg.expert_topp > 0:  # a sharp router, so that adaptive K really drops routes
            for bank in (m for m in model.modules() if hasattr(m, "expert_router_w")):
                bank.expert_router_w.mul_(200.0)
        if hasattr(model, "hop_scale"):  # move the per-hop affine away from identity so it really matters
            model.hop_scale.normal_(1.0, 0.3)
            model.hop_shift.normal_(0.0, 0.3)
    return model


@pytest.mark.parametrize("name", list(VARIANTS))
def test_bucketed_equals_reference_hard_forward_fp64(name):
    model = model_with_varied_halting(**VARIANTS[name])
    x = torch.randint(0, 256, (3, 32))
    with torch.no_grad():
        ref, ref_info = model(x, hard_halt=True)
        out, stats = bucketed_hard_forward(model, x)
    assert torch.allclose(out, ref, atol=1e-10, rtol=0)
    assert torch.equal(stats["hops_per_token"], ref_info["hops_per_token"])
    assert len(set(ref_info["hops_per_token"].flatten().tolist())) > 1  # halting really varies


@pytest.mark.parametrize("name", list(VARIANTS))
def test_bucketed_prefix_causality_fp64(name):
    model = model_with_varied_halting(seed=1, **VARIANTS[name])
    x = torch.randint(0, 256, (2, 32))
    x2 = x.clone()
    x2[:, 20:] = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        a, _ = bucketed_hard_forward(model, x)
        b, _ = bucketed_hard_forward(model, x2)
    assert torch.allclose(a[:, :20], b[:, :20], atol=1e-10, rtol=0)


def test_packed_size_is_a_bucket_and_no_token_is_lost():
    model = model_with_varied_halting(seed=2, unshared_attention=True)
    x = torch.randint(0, 256, (3, 32))
    with torch.no_grad():
        _, ref_info = model(x, hard_halt=True)
        _, stats = bucketed_hard_forward(model, x)
    allowed = {max(1, round(f * 32)) for f in BUCKETS}
    for hop in stats["hops"]:
        assert hop["capacity"] in allowed
        assert hop["capacity"] >= hop["max_active_per_sequence"]
        assert hop["packed_valid_tokens"] == hop["active_tokens"]  # every active token is processed once
        assert hop["packed_active_tokens"] == hop["active_tokens"]  # ... and the packed ones are the active ones
    assert sum(h["active_tokens"] for h in stats["hops"]) == int(ref_info["hops_per_token"].sum())


def test_all_tokens_halted_after_first_hop_runs_one_hop():
    model = model_with_varied_halting(seed=3)
    with torch.no_grad():
        model.halt_head.weight.zero_()
        model.halt_head.bias.fill_(20.0)
        _, stats = bucketed_hard_forward(model, torch.randint(0, 256, (2, 32)))
    assert len(stats["hops"]) == 1


@pytest.mark.parametrize("name", list(VARIANTS))
def test_bucketed_equals_reference_long_context(name):
    """Regression (F2 bench): the packing sort key overflowed int8 for contexts >= 128, mixing halted tokens into the
    active block. Contexts above 127 must give the same result as the reference."""
    torch.manual_seed(4)
    base = dict(max_seq_len=200, d_model=16, n_heads=4, n_modules=4, experts_per_module=3, module_topk=2,
                expert_topk=2, n_hops=4, expert_hidden=24)
    model = AdaptiveModularLM(ModelConfig(**{**base, **VARIANTS[name]})).double().eval()
    with torch.no_grad():
        model.halt_head.weight.normal_(0, 2.0)
        model.halt_head.bias.fill_(0.3)
        x = torch.randint(0, 256, (2, 200))
        ref, info = model(x, hard_halt=True)
        out, stats = bucketed_hard_forward(model, x)
    assert torch.allclose(out, ref, atol=1e-10, rtol=0)
    assert all(h["packed_active_tokens"] == h["active_tokens"] for h in stats["hops"])


@pytest.mark.parametrize("name", list(VARIANTS))
def test_static_masked_hard_equals_reference_fp64(name):
    """F3/B2: hard halting computed for all tokens with the reference weights (no nonzero, no early exit)."""
    from amllm.fastpath import static_hard_forward
    model = model_with_varied_halting(seed=5, **VARIANTS[name])
    x = torch.randint(0, 256, (3, 32))
    with torch.no_grad():
        ref, info = model(x, hard_halt=True)
        out, counts = static_hard_forward(model, x)
    assert torch.allclose(out, ref, atol=1e-10, rtol=0)
    assert torch.equal(counts, info["hops_per_token"])


def test_adaptive_k_variant_really_drops_routes():
    """Guards the equivalence tests above: with the sharp router some tokens use fewer than expert_topk experts."""
    model = model_with_varied_halting(**VARIANTS["adaptive_k"])
    with torch.no_grad():
        _, info = model(torch.randint(0, 256, (3, 32)), hard_halt=True)
    routed = int(info["expert_counts"].sum())
    full = int(info["hops_per_token"].sum()) * model.cfg.module_topk * model.cfg.expert_topk
    assert 0 < routed < full


def test_static_path_equals_reference_at_fixed_depth_fp64():
    """W3: the static path also runs fixed-depth (halt_rule none) models, e.g. FDC1, so they can be CUDA-Graph captured."""
    from amllm.fastpath import static_hard_forward
    model = model_with_varied_halting(seed=2, halt_rule="none", **VARIANTS["fdc1_like"])
    x = torch.randint(0, 256, (3, 32))
    with torch.no_grad():
        ref, info = model(x, hard_halt=True)
        out, counts = static_hard_forward(model, x)
    assert torch.allclose(out, ref, atol=1e-10, rtol=0)
    assert torch.equal(counts, info["hops_per_token"])
