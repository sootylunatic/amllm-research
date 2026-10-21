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
