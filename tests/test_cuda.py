"""Correctness checks in the actual training/evaluation setting: CUDA + bf16 autocast.

Skipped when no CUDA device is available (e.g. in CI); run locally before experiments."""
import pytest
import torch

from amllm.config import ModelConfig
from amllm.model import build_model, set_dispatch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

CFGS = {
    "amllm": dict(arch="amllm"),
    "amllm_unshared_attn": dict(arch="amllm", unshared_attention=True),
    "amllm_fixed": dict(arch="amllm", halt_rule="none"),
    "moe": dict(arch="moe"),
    "dense": dict(arch="dense"),
}


def cfg(**kw):
    base = dict(max_seq_len=64, d_model=64, n_heads=4, n_modules=6, experts_per_module=4, module_topk=2,
                expert_topk=2, n_hops=4, expert_hidden=64, n_layers=3, ffn_hidden=128, n_experts=6, moe_topk=2)
    base.update(kw)
    return ModelConfig(**base)


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def _causality_diff(name, hard, dtype):
    torch.manual_seed(1)
    model = build_model(cfg(**CFGS[name])).cuda().eval()
    if dtype == torch.float64:
        model = model.double()
    x = torch.randint(0, 256, (4, 64), device="cuda")
    x2 = x.clone()
    x2[:, 40:] = torch.randint(0, 256, (4, 24), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        a, _ = model(x, hard_halt=hard)
        b, _ = model(x2, hard_halt=hard)
    assert torch.isfinite(a).all()
    return (a[:, :40].double() - b[:, :40].double()).abs().max().item()


@pytest.mark.parametrize("name", list(CFGS))
@pytest.mark.parametrize("hard", [False, True])
def test_cuda_fp64_causality_exact(name, hard):
    """In fp64 the prefix must not depend on future tokens beyond rounding (~1e-15)."""
    assert _causality_diff(name, hard, torch.float64) < 1e-12


@pytest.mark.parametrize("name", list(CFGS))
@pytest.mark.parametrize("hard", [False, True])
def test_cuda_bf16_causality_within_rounding(name, hard):
    """In bf16 a GEMM over a different number of rows (the expert groups change when future tokens are
    routed differently) may round the *same* row differently by ~1 ulp. That is batch-composition
    numerics, not information flow (see the fp64 test); a real leak would change logits by O(1)."""
    assert _causality_diff(name, hard, torch.bfloat16) <= 2 ** -6


@pytest.mark.parametrize("name", ["amllm", "amllm_unshared_attn", "moe"])
def test_cuda_bf16_loop_and_bmm_agree_forward_and_all_gradients(name):
    results = []
    for dispatch in ("loop", "bmm"):
        torch.manual_seed(2)
        model = build_model(cfg(**CFGS[name])).cuda().train()
        set_dispatch(model, dispatch)
        torch.manual_seed(3)
        x = torch.randint(0, 256, (4, 64), device="cuda")
        with autocast():
            logits, info = model(x)
            loss = logits.float().square().mean() + info["aux_loss"]
        loss.backward()
        grads = {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}
        results.append((logits.float().detach(), grads, info["expert_counts"].clone()))
    (la, ga, ca), (lb, gb, cb) = results
    assert torch.equal(ca, cb)
    assert torch.allclose(la, lb, atol=2e-2, rtol=2e-2)
    assert ga.keys() == gb.keys()
    for n in ga:
        scale = ga[n].abs().max().clamp(min=1e-6)
        assert ((ga[n] - gb[n]).abs().max() / scale) < 5e-2, n


def test_cuda_hard_halting_counts_and_skipping():
    torch.manual_seed(4)
    c = cfg()
    model = build_model(c).cuda().eval()
    with torch.no_grad():
        model.halt_head.weight.normal_(0, 2.0)
        model.halt_head.bias.zero_()
        model.bank.experts.rows_computed = 0
        with autocast():
            _, info = model(torch.randint(0, 256, (4, 64), device="cuda"))
    hops = info["hops_per_token"].sum().item()
    assert info["expert_counts"].sum().item() == hops * c.module_topk * c.expert_topk
    assert model.bank.experts.rows_computed == info["expert_counts"].sum().item()
    assert info["hops_per_token"].min() >= 1


@pytest.mark.parametrize("name", list(CFGS))
@pytest.mark.parametrize("dispatch", ["loop", "bmm"])
def test_cuda_bf16_forward_is_bitwise_deterministic(name, dispatch):
    """Expert outputs are combined with a fixed-order reduction (not atomics), so repeated evaluation
    forwards on the same input must be bitwise identical on CUDA."""
    torch.manual_seed(5)
    model = build_model(cfg(**CFGS[name])).cuda().eval()
    if name != "dense":
        set_dispatch(model, dispatch)
    x = torch.randint(0, 256, (4, 64), device="cuda")
    with torch.no_grad(), autocast():
        outs = [model(x)[0] for _ in range(3)]
    assert all(torch.equal(outs[0], o) for o in outs[1:])
