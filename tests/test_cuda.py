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
    # phase 2 routing variants (audit of F1): per-hop attention with gradient-passing top-1 weightings
    "ua_mk1_full_softmax": dict(arch="amllm", unshared_attention=True, module_topk=1, route_weighting="full_softmax"),
    "ua_ek1_full_softmax": dict(arch="amllm", unshared_attention=True, expert_topk=1, route_weighting="full_softmax"),
    "ua_mk1_straight_through": dict(arch="amllm", unshared_attention=True, module_topk=1,
                                    route_weighting="straight_through"),
    "ua_top2_straight_through": dict(arch="amllm", unshared_attention=True, route_weighting="straight_through"),
}
ROUTING_VARIANTS = ["ua_mk1_full_softmax", "ua_ek1_full_softmax", "ua_mk1_straight_through", "ua_top2_straight_through"]


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


@pytest.mark.parametrize("name", ["amllm", "amllm_unshared_attn", "moe", *ROUTING_VARIANTS])
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


@pytest.mark.parametrize("name", ROUTING_VARIANTS)
def test_cuda_bf16_top1_routers_receive_gradient(name):
    """Both router levels get a finite, non-zero LM gradient in the real training setting (CUDA + bf16)."""
    torch.manual_seed(6)
    model = build_model(cfg(**CFGS[name], halt_rule="none")).cuda().train()
    with autocast():
        logits, _ = model(torch.randint(0, 256, (4, 64), device="cuda"))
        loss = logits.float().square().mean()  # LM-style loss only, no balance term
    loss.backward()
    for p in (model.bank.router.weight, model.bank.expert_router_w):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_cuda_bf16_straight_through_forward_is_bitwise_default():
    """straight_through must not change the forward value of topk_softmax on CUDA/bf16 either."""
    x = torch.randint(0, 256, (4, 64), device="cuda")
    outs = []
    for weighting in ("topk_softmax", "straight_through"):
        torch.manual_seed(7)
        model = build_model(cfg(arch="amllm", unshared_attention=True, module_topk=1,
                                route_weighting=weighting)).cuda().eval()
        with torch.no_grad(), autocast():
            outs.append(model(x)[0])
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("name", ["amllm_unshared_attn", "ua_mk1_straight_through"])
def test_cuda_bucketed_execution_matches_reference(name):
    """F2-1 on CUDA: fp64 equal to the reference hard forward; bf16 within rounding; same hops."""
    from amllm.fastpath import bucketed_hard_forward
    torch.manual_seed(8)
    model = build_model(cfg(**CFGS[name])).cuda().eval()
    with torch.no_grad():
        model.halt_head.weight.normal_(0, 2.0)
        model.halt_head.bias.fill_(0.3)
    x = torch.randint(0, 256, (4, 64), device="cuda")
    with torch.no_grad():
        m64 = model.double()
        ref, info = m64(x, hard_halt=True)
        out, stats = bucketed_hard_forward(m64, x)
        assert torch.allclose(out, ref, atol=1e-10, rtol=0)
        assert torch.equal(stats["hops_per_token"], info["hops_per_token"])
        m32 = model.float()
        with autocast():
            ref, info = m32(x, hard_halt=True)
            out, stats = bucketed_hard_forward(m32, x)
        assert (out.float() - ref.float()).abs().max() <= 2 ** -5


@pytest.mark.parametrize("name", ["amllm_unshared_attn", "ua_mk1_straight_through", "moe"])
@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_cuda_grouped_dispatch_matches_loop_forward_and_gradients(name, dtype):
    """F3/B1: grouped GEMM dispatch (device-side offsets, no padding) is the same computation as loop dispatch."""
    results = []
    for dispatch in ("loop", "grouped"):
        torch.manual_seed(9)
        model = build_model(cfg(**CFGS[name])).cuda().train()
        set_dispatch(model, dispatch)
        torch.manual_seed(10)
        x = torch.randint(0, 256, (4, 64), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "bf16"):
            logits, info = model(x)
            loss = logits.float().square().mean() + info["aux_loss"]
        loss.backward()
        grads = {n: p.grad.float().clone() for n, p in model.named_parameters() if p.grad is not None}
        results.append((logits.float().detach(), grads))
    (la, ga), (lb, gb) = results
    tol = 1e-4 if dtype == "fp32" else 2e-2
    assert torch.allclose(la, lb, atol=tol, rtol=tol)
    assert ga.keys() == gb.keys()
    for n in ga:
        scale = ga[n].abs().max().clamp(min=1e-6)
        assert ((ga[n] - gb[n]).abs().max() / scale) < (1e-3 if dtype == "fp32" else 5e-2), n


@pytest.mark.parametrize("name", ["ua_mk1_straight_through", "moe"])
def test_cuda_graph_replay_matches_eager(name):
    """F3/B2: a captured CUDA Graph gives bitwise the eager result of the same backend, also for new input data."""
    from amllm.fastpath import GraphRunner, static_forward
    from amllm.model import set_diagnostics
    torch.manual_seed(11)
    model = build_model(cfg(**CFGS[name])).cuda().eval()
    set_dispatch(model, "triton")
    set_diagnostics(model, False)
    x1 = torch.randint(0, 256, (4, 64), device="cuda")
    x2 = torch.randint(0, 256, (4, 64), device="cuda")

    def fn(x):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return static_forward(model, x)

    with torch.no_grad():
        eager1, eager2 = fn(x1).clone(), fn(x2).clone()
        runner = GraphRunner(fn, x1)
        assert torch.equal(runner(x1), eager1)
        assert torch.equal(runner(x2), eager2)


@pytest.mark.parametrize("name", ["amllm_unshared_attn", "ua_mk1_straight_through", "ua_ek1_full_softmax", "moe"])
def test_cuda_triton_dispatch_matches_loop_fp32(name):
    """F3: in fp32 (IEEE products) the Triton dispatch computes the loop-dispatch forward, routes included."""
    torch.manual_seed(12)
    model = build_model(cfg(**CFGS[name])).cuda().eval()
    x = torch.randint(0, 256, (4, 64), device="cuda")
    outs = {}
    for dispatch in ("loop", "triton"):
        set_dispatch(model, dispatch)
        with torch.no_grad():
            logits, info = model(x, hard_halt=False)
        outs[dispatch] = (logits, info["expert_counts"])
    assert torch.equal(outs["loop"][1], outs["triton"][1])
    assert torch.allclose(outs["loop"][0], outs["triton"][0], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name", ["amllm_unshared_attn", "ua_mk1_straight_through", "moe"])
def test_cuda_triton_dispatch_bf16_statistically_equal(name):
    """In bf16 the Triton path rounds intermediates differently from loop (fp32 bias/SiLU); discrete top-k routing
    can flip for a few tokens (see F2 report). Require near-identical predictions and routing statistics."""
    torch.manual_seed(13)
    model = build_model(cfg(**CFGS[name])).cuda().eval()
    x = torch.randint(0, 256, (8, 64), device="cuda")
    outs = {}
    for dispatch in ("loop", "triton"):
        set_dispatch(model, dispatch)
        with torch.no_grad(), autocast():
            logits, info = model(x, hard_halt=False)
        outs[dispatch] = (logits.float(), info["expert_counts"].float())
    agreement = (outs["loop"][0].argmax(-1) == outs["triton"][0].argmax(-1)).float().mean()
    count_shift = (outs["loop"][1] - outs["triton"][1]).abs().sum() / outs["loop"][1].sum()
    assert agreement >= 0.97 and count_shift <= 0.05


@pytest.mark.parametrize("name", ["ua_mk1_straight_through", "moe", "dense"])
def test_cuda_graph_decoding_matches_eager_decoding(name):
    """F4: CUDA-Graph decode steps (Triton dispatch, bf16) give bitwise the eager decode logits, step after step."""
    from amllm.generate import Generator, GraphDecoder
    from amllm.model import set_diagnostics
    torch.manual_seed(14)
    model = build_model(cfg(**CFGS[name])).cuda().eval()
    if name != "dense":
        set_dispatch(model, "triton")
    set_diagnostics(model, False)
    prompt = torch.randint(0, 256, (2, 32), device="cuda")
    steps = torch.randint(0, 256, (2, 16), device="cuda")
    logits = {}
    for mode in ("eager", "graph"):
        gen = Generator(model, batch=2, max_len=64, cache_dtype=torch.bfloat16)
        with torch.no_grad(), autocast():
            gen.prefill(prompt)
        dec = GraphDecoder(gen, steps[:, 0]) if mode == "graph" else None
        out = []
        for i in range(steps.shape[1]):
            if dec is None:
                with torch.no_grad(), autocast():
                    out.append(gen.decode_step(steps[:, i]).clone())
            else:
                out.append(dec(steps[:, i]).clone())
        logits[mode] = torch.stack(out)
    assert torch.equal(logits["eager"], logits["graph"])


def test_triton_dispatch_refuses_training():
    """The Triton grouped GEMM has no backward: a forward with gradients enabled must fail loudly, not train
    the expert weights silently without gradients (audit 2026-10-21)."""
    from amllm.experiment import PRESETS
    cfg = ModelConfig(**{**PRESETS["amllm_f5"], "d_model": 64, "n_heads": 4, "expert_hidden": 64, "vocab_size": 256})
    model = build_model(cfg).cuda()
    set_dispatch(model, "triton")
    x = torch.randint(0, 256, (2, 16), device="cuda")
    with pytest.raises(RuntimeError, match="inference only"):
        model(x)
    with torch.no_grad():
        model(x)  # inference works
