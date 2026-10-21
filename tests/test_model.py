import pytest
import torch
from amllm.config import ModelConfig
from amllm.model import AdaptiveModularLM, build_model
from amllm.routing import ExpertBank, HierarchicalModuleBank


def tiny_cfg(**kw):
    base = dict(max_seq_len=16, d_model=16, n_heads=4, n_modules=4, experts_per_module=3,
                module_topk=2, expert_topk=2, n_hops=3, expert_hidden=32,
                n_layers=2, ffn_hidden=32, n_experts=4, moe_topk=2)
    base.update(kw)
    return ModelConfig(**base)


ALL_CFGS = {
    "amllm": tiny_cfg(),
    "amllm_v01": tiny_cfg(attn_per_hop=False),
    "amllm_step": tiny_cfg(halt_rule="step", halt_threshold=0.65),
    "amllm_fixed": tiny_cfg(halt_rule="none"),
    "amllm_unshared_attn": tiny_cfg(unshared_attention=True),
    "dense": tiny_cfg(arch="dense"),
    "moe": tiny_cfg(arch="moe"),
}


@pytest.mark.parametrize("name", ["amllm", "amllm_v01", "amllm_step"])
def test_forward_backward_router_gradient(name):
    torch.manual_seed(4)
    model = AdaptiveModularLM(ALL_CFGS[name])
    x = torch.randint(0, 256, (2, 9))
    y = torch.randint(0, 256, (2, 9))
    logits, info = model(x, hard_halt=False)
    assert logits.shape == (2, 9, 256)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) + info['aux_loss']
    loss.backward()
    for p in (model.bank.router.weight, model.bank.expert_router_w, model.halt_head.weight, model.bank.experts.w1):
        assert p.grad is not None and p.grad.abs().sum().item() > 0
        assert torch.isfinite(p.grad).all()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("rule", ["survival", "step"])
def test_hard_halting_bounds_and_no_extra_expert_calls(rule):
    torch.manual_seed(3)
    cfg = tiny_cfg(halt_rule=rule)
    model = AdaptiveModularLM(cfg).eval()
    with torch.no_grad():
        model.halt_head.weight.zero_()
        model.halt_head.bias.fill_(20)  # every token exits after first hop
        logits, info = model(torch.randint(0, 256, (2, 7)))
    assert logits.shape == (2, 7, 256)
    assert info['hops_per_token'].min().item() == 1
    assert info['hops_per_token'].max().item() == 1
    assert info['module_counts'].sum().item() == 2 * 7 * cfg.module_topk
    assert info['expert_counts'].sum().item() == 2 * 7 * cfg.module_topk * cfg.expert_topk
    with torch.no_grad():
        model.halt_head.bias.fill_(-20)  # force full depth
        _, info = model(torch.randint(0, 256, (2, 7)))
    assert info['hops_per_token'].min().item() == cfg.n_hops
    assert info['expert_counts'].sum().item() == 2 * 7 * cfg.n_hops * cfg.module_topk * cfg.expert_topk


def test_counts_match_hops_per_token():
    torch.manual_seed(8)
    cfg = tiny_cfg()
    model = AdaptiveModularLM(cfg).eval()
    with torch.no_grad():
        model.halt_head.weight.normal_(0, 2.0)
        model.halt_head.bias.fill_(0.0)
        _, info = model(torch.randint(0, 256, (3, 16)))
    token_hops = info['hops_per_token'].sum().item()
    assert 3 * 16 <= token_hops <= 3 * 16 * cfg.n_hops
    assert info['module_counts'].sum().item() == token_hops * cfg.module_topk
    assert info['expert_counts'].sum().item() == token_hops * cfg.module_topk * cfg.expert_topk


def test_hard_with_tiny_threshold_equals_soft():
    """Survival rule: hard inference is soft inference truncated where survival < threshold."""
    torch.manual_seed(1)
    model = AdaptiveModularLM(tiny_cfg()).eval()
    x = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        soft, _ = model(x, hard_halt=False)
        hard, info = model(x, hard_halt=True, halt_threshold=1e-6)
    assert info['hops_per_token'].min().item() == tiny_cfg().n_hops
    assert torch.allclose(soft, hard, atol=1e-5)


def test_expert_bank_matches_naive_reference():
    torch.manual_seed(0)
    bank = ExpertBank(5, 8, 16)
    x = torch.randn(6, 8)
    tok = torch.tensor([0, 0, 1, 2, 3, 3, 5, 4])
    eid = torch.tensor([4, 1, 1, 0, 4, 2, 2, 4])
    w = torch.rand(8)
    y, counts = bank(x, tok, eid, w)
    ref = torch.zeros_like(x)
    for a in range(len(tok)):
        g = eid[a]
        h = torch.nn.functional.silu(x[tok[a]] @ bank.w1[g] + bank.b1[g])
        ref[tok[a]] += w[a] * (h @ bank.w2[g] + bank.b2[g])
    assert torch.allclose(y, ref, atol=1e-5)
    assert counts.tolist() == [1, 2, 2, 0, 3]
    assert bank.last_executed == [0, 1, 2, 4]  # expert 3 never ran


def test_expert_dispatch_skips_unselected():
    torch.manual_seed(0)
    bank = HierarchicalModuleBank(width=8, n_modules=4, module_k=2, experts_per_module=4, expert_k=2, hidden=16)
    y, balance, mc, ec = bank(torch.randn(5, 8))
    assert y.shape == (5, 8)
    assert mc.sum().item() == 10
    assert ec.sum().item() == 20
    assert (ec.sum(1) == mc * 2).all()  # each selection of a module runs exactly expert_k experts
    executed = set(bank.experts.last_executed)
    assert executed == set(torch.nonzero(ec.flatten()).flatten().tolist())
    assert torch.isfinite(balance)


@pytest.mark.parametrize("k", [1, 4])
def test_topk_edge_cases(k):
    torch.manual_seed(2)
    bank = HierarchicalModuleBank(width=8, n_modules=4, module_k=k, experts_per_module=3, expert_k=min(k, 3), hidden=8)
    y, _, mc, ec = bank(torch.randn(7, 8))
    assert mc.sum().item() == 7 * k and ec.sum().item() == 7 * k * min(k, 3)
    assert (mc <= 7).all()


@pytest.mark.parametrize("name", list(ALL_CFGS))
@pytest.mark.parametrize("hard", [False, True])
def test_causality_prefix_not_affected_by_future(name, hard):
    torch.manual_seed(5)
    model = build_model(ALL_CFGS[name]).eval()
    x = torch.randint(0, 256, (2, 12))
    x2 = x.clone()
    x2[:, 6:] = (x2[:, 6:] + 17) % 256
    with torch.no_grad():
        logits1, _ = model(x, hard_halt=hard)
        logits2, _ = model(x2, hard_halt=hard)
    assert torch.allclose(logits1[:, :6], logits2[:, :6], atol=1e-5)
    assert not torch.allclose(logits1[:, 6:], logits2[:, 6:], atol=1e-5)


@pytest.mark.parametrize("name", list(ALL_CFGS))
def test_eval_is_deterministic(name):
    torch.manual_seed(6)
    model = build_model(ALL_CFGS[name]).eval()
    x = torch.randint(0, 256, (2, 10))
    with torch.no_grad():
        a, ia = model(x)
        b, ib = model(x)
    assert torch.equal(a, b)
    assert torch.equal(ia['expert_counts'], ib['expert_counts'])


def test_flops_drop_with_early_exit():
    torch.manual_seed(7)
    model = AdaptiveModularLM(tiny_cfg()).eval()
    x = torch.randint(0, 256, (1, 16))
    with torch.no_grad():
        model.halt_head.bias.fill_(-20)
        _, full = model(x)
        model.halt_head.bias.fill_(20)
        _, early = model(x)
    assert early['flops_per_token_ideal'] < full['flops_per_token_ideal']
    assert early['flops_per_token'] < full['flops_per_token']


@pytest.mark.parametrize("name", ["amllm", "moe"])
def test_bmm_dispatch_matches_loop(name):
    from amllm.model import set_dispatch
    torch.manual_seed(9)
    model = build_model(ALL_CFGS[name]).eval()
    x = torch.randint(0, 256, (2, 14))
    with torch.no_grad():
        a, ia = model(x)
        set_dispatch(model, "bmm")
        b, ib = model(x)
    assert torch.allclose(a, b, atol=1e-5)
    assert torch.equal(ia['expert_counts'], ib['expert_counts'])


def test_bmm_dispatch_gradients_match_loop():
    from amllm.model import set_dispatch
    grads = []
    for mode in ("loop", "bmm"):
        torch.manual_seed(10)
        model = build_model(ALL_CFGS["amllm"])
        set_dispatch(model, mode)
        logits, info = model(torch.randint(0, 256, (2, 10)), hard_halt=False)
        (logits.square().mean() + info["aux_loss"]).backward()
        grads.append(model.bank.experts.w1.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_loop_dispatch_computes_only_routed_rows():
    """Real skipping: computed FFN rows == routed assignments; bmm pads and computes more."""
    from amllm.model import set_dispatch
    torch.manual_seed(11)
    model = AdaptiveModularLM(tiny_cfg()).eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        model.halt_head.bias.fill_(20)  # one hop
        _, info = model(x)
        assert model.bank.experts.rows_computed == info['expert_counts'].sum().item() == 2 * 16 * 4
        set_dispatch(model, "bmm")
        model.bank.experts.rows_computed = 0
        _, info = model(x)
        assert model.bank.experts.rows_computed >= info['expert_counts'].sum().item()


def test_random_routing_and_trace():
    from amllm.model import set_random_routing
    torch.manual_seed(12)
    cfg = tiny_cfg()
    model = AdaptiveModularLM(cfg).eval()
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        _, learned = model(x, trace=True)
        set_random_routing(model, True)
        _, rand = model(x, trace=True)
    for info in (learned, rand):
        hops = info['hops_per_token'].sum().item()
        assert sum(h['positions'].numel() for h in info['trace']) == hops
        assert info['expert_counts'].sum().item() == hops * cfg.module_topk * cfg.expert_topk
        for h in info['trace']:
            assert (h['experts'] // cfg.experts_per_module == h['modules'][..., None]).all()
    assert not torch.equal(learned['trace'][0]['modules'], rand['trace'][0]['modules'])


def _router_grads(**cfg_kw):
    torch.manual_seed(13)
    model = AdaptiveModularLM(tiny_cfg(halt_rule="none", **cfg_kw))
    logits, _ = model(torch.randint(0, 256, (2, 10)), hard_halt=False)
    logits.square().mean().backward()  # LM-style loss only, no aux/balance term
    return (model.bank.router.weight.grad.abs().sum().item(),
            model.bank.expert_router_w.grad.abs().sum().item())


@pytest.mark.parametrize("module_k,expert_k", [(1, 2), (2, 1), (1, 1), (2, 2)])
def test_topk_softmax_top1_router_gets_no_lm_gradient(module_k, expert_k):
    """Known defect of the default weighting (ADR-0013): with k=1 the single selected route has
    renormalised softmax weight exactly 1, so that router (module OR expert level) receives gradient
    only from the balance loss. Results of top-1 runs with this weighting do not measure top-1 routing."""
    g_module, g_expert = _router_grads(module_topk=module_k, expert_topk=expert_k)
    assert (g_module > 0) == (module_k > 1)
    assert (g_expert > 0) == (expert_k > 1)


@pytest.mark.parametrize("module_k,expert_k", [(1, 2), (2, 1), (1, 1)])
def test_full_softmax_weighting_gives_top1_routers_gradient(module_k, expert_k):
    g_module, g_expert = _router_grads(module_topk=module_k, expert_topk=expert_k, route_weighting="full_softmax")
    assert g_module > 0 and g_expert > 0


def test_random_router_mode_ignores_router():
    torch.manual_seed(14)
    cfg = tiny_cfg(halt_rule="none", router_mode="random")
    model = AdaptiveModularLM(cfg)
    logits, info = model(torch.randint(0, 256, (2, 10)), hard_halt=False)
    logits.square().mean().backward()
    assert model.bank.router.weight.grad is None or model.bank.router.weight.grad.abs().sum() == 0
    assert model.bank.experts.w1.grad.abs().sum() > 0
    assert info["expert_counts"].sum().item() == 2 * 10 * cfg.n_hops * cfg.module_topk * cfg.expert_topk


@pytest.mark.parametrize("name", ["amllm", "moe", "dense", "amllm_unshared_attn"])
def test_diagnostics_off_does_not_change_outputs(name):
    from amllm.model import set_diagnostics
    torch.manual_seed(15)
    model = build_model(ALL_CFGS[name]).eval()
    x = torch.randint(0, 256, (2, 12))
    with torch.no_grad():
        a, ia = model(x)
        set_diagnostics(model, False)
        b, ib = model(x)
    assert torch.equal(a, b)
    assert torch.equal(ia["hops_per_token"], ib["hops_per_token"])


def test_attention_applications_count_executed_calls():
    """The implementation runs attention for all tokens while any token is active; the ideal count only
    for active query tokens. Parameter-application metrics must use the executed count (audit P2)."""
    torch.manual_seed(16)
    cfg = tiny_cfg()
    model = AdaptiveModularLM(cfg).eval()
    with torch.no_grad():
        model.halt_head.weight.normal_(0, 3.0)
        model.halt_head.bias.fill_(0.5)
        _, info = model(torch.randint(0, 256, (3, 16)))
    hops = info["hops_per_token"].float()
    assert info["attention_applications"] == int(hops.max())
    assert abs(info["attention_query_fraction"] - hops.mean().item()) < 1e-6


@pytest.mark.parametrize("dispatch", ["loop", "bmm"])
def test_grouped_combine_matches_index_add(dispatch):
    """routes_per_token (fixed-order, deterministic) must equal the generic index_add combine."""
    torch.manual_seed(17)
    bank = ExpertBank(6, 8, 16)
    bank.dispatch = dispatch
    n, k = 5, 3
    x = torch.randn(n, 8)
    tok = torch.arange(n).repeat_interleave(k)
    eid = torch.randint(0, 6, (n * k,))
    w = torch.rand(n * k)
    a, ca = bank(x, tok, eid, w)
    b, cb = bank(x, tok, eid, w, routes_per_token=k)
    assert torch.allclose(a, b, atol=1e-6) and torch.equal(ca, cb)


def test_straight_through_forward_equals_default_weighting():
    """F1b: straight-through weighting has exactly the forward value of topk_softmax (only gradients differ)."""
    x = torch.randint(0, 256, (2, 12))
    outs = []
    for weighting in ("topk_softmax", "straight_through"):
        torch.manual_seed(18)
        model = AdaptiveModularLM(tiny_cfg(route_weighting=weighting, module_topk=1)).eval()
        with torch.no_grad():
            outs.append(model(x, hard_halt=False)[0])
    assert torch.equal(outs[0], outs[1])


@pytest.mark.parametrize("module_k,expert_k", [(1, 2), (2, 1), (1, 1), (2, 2)])
def test_straight_through_gives_top1_routers_gradient(module_k, expert_k):
    g_module, g_expert = _router_grads(module_topk=module_k, expert_topk=expert_k, route_weighting="straight_through")
    assert g_module > 0 and g_expert > 0
