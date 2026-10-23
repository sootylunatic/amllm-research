"""FD01 (bank_groups) and FD03 (bank_norm) switches: defaults are bitwise-neutral, variants behave as specified."""
import json
from pathlib import Path

import pytest
import torch

from amllm.config import ModelConfig
from amllm.model import AdaptiveModularLM, build_model
from test_model import tiny_cfg

GOLDEN = json.loads((Path(__file__).parent / "golden_default_behaviour.json").read_text())


def golden_cases():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_golden", Path(__file__).parents[1] / "tools" / "make_golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CASES, mod.stats


@pytest.mark.parametrize("name", list(GOLDEN))
def test_default_behaviour_is_unchanged(name):
    """The state recorded before the switches existed (tools/make_golden.py) is reproduced exactly."""
    cases, stats = golden_cases()
    cfg = cases[name]
    torch.manual_seed(7)
    model = build_model(cfg).eval()
    x = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(11))
    rec = GOLDEN[name]
    assert sum(p.numel() for p in model.parameters()) == rec["n_params"]
    with torch.no_grad():
        got = {"soft": stats(model(x, hard_halt=False)[0]), "hard": stats(model(x, hard_halt=True)[0])} if cfg.arch == "amllm" \
            else {"fixed": stats(model(x)[0])}
    for mode, ref in got.items():
        assert ref["first"] == pytest.approx(rec[mode]["first"], abs=1e-5), (name, mode)
        assert ref["sum"] == pytest.approx(rec[mode]["sum"], abs=1e-3), (name, mode)


def test_default_state_dict_has_no_new_keys():
    model = build_model(tiny_cfg())
    assert not any(k.startswith(("extra_banks", "extra_step_norms", "hop_scale", "hop_shift")) for k in model.state_dict())


def test_bank_group_mapping():
    model = build_model(tiny_cfg(n_hops=6, bank_groups=3))
    ids = [id(model.bank_for(h)) for h in range(6)]
    assert ids[0] == ids[1] and ids[2] == ids[3] and ids[4] == ids[5]
    assert len(set(ids)) == 3 and ids[0] == id(model.bank)
    per_hop = build_model(tiny_cfg(n_hops=3, bank_groups=3))
    assert len({id(per_hop.bank_for(h)) for h in range(3)}) == 3


def test_separate_banks_multiply_expert_parameters_only():
    base = build_model(tiny_cfg(n_hops=3, unshared_attention=True))
    sep = build_model(tiny_cfg(n_hops=3, unshared_attention=True, bank_groups=3))
    bank_params = sum(p.numel() for p in base.bank.parameters())
    assert sum(p.numel() for p in sep.parameters()) == sum(p.numel() for p in base.parameters()) + 2 * bank_params


@pytest.mark.parametrize("kw", [dict(bank_groups=3), dict(bank_norm="per_hop"), dict(bank_norm="hop_affine"),
                                dict(bank_groups=3, bank_norm="per_hop")])
def test_variants_train_and_every_new_part_gets_gradient(kw):
    torch.manual_seed(3)
    model = AdaptiveModularLM(tiny_cfg(n_hops=3, halt_rule="none", **kw))
    x = torch.randint(0, 256, (2, 9))
    y = torch.randint(0, 256, (2, 9))
    logits, info = model(x, hard_halt=False)
    (torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) + info["aux_loss"]).backward()
    new = {n: p for n, p in model.named_parameters() if n.startswith(("extra_banks", "extra_step_norms", "hop_scale", "hop_shift"))}
    assert new
    for n, p in new.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    assert any(p.grad.abs().sum() > 0 for n, p in new.items() if "experts.w1" in n or "weight" in n or "hop_" in n)


def test_hop_affine_starts_as_the_shared_norm():
    torch.manual_seed(5)
    shared = build_model(tiny_cfg(halt_rule="none")).eval()
    affine = build_model(tiny_cfg(halt_rule="none", bank_norm="hop_affine")).eval()
    affine.load_state_dict(shared.state_dict(), strict=False)
    x = torch.randint(0, 256, (2, 9))
    with torch.no_grad():
        assert torch.allclose(shared(x, hard_halt=True)[0], affine(x, hard_halt=True)[0], atol=1e-6)


def test_old_checkpoint_keys_load_into_default_model():
    model = build_model(tiny_cfg())
    state = model.state_dict()
    fresh = build_model(tiny_cfg())
    fresh.load_state_dict(state)  # strict: a new default key would fail here


def test_causality_with_separate_banks_and_norms():
    torch.manual_seed(2)
    model = build_model(tiny_cfg(halt_rule="none", n_hops=3, bank_groups=3, bank_norm="per_hop", unshared_attention=True)).eval()
    a = torch.randint(0, 256, (1, 10))
    b = a.clone()
    b[0, 7:] = (b[0, 7:] + 5) % 256
    with torch.no_grad():
        la, lb = model(a, hard_halt=True)[0], model(b, hard_halt=True)[0]
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-5)


def test_fast_paths_refuse_non_default_bank():
    from amllm.fastpath import static_hard_forward
    model = build_model(tiny_cfg(bank_groups=2)).eval()
    with pytest.raises(ValueError):
        static_hard_forward(model, torch.randint(0, 256, (1, 8)))
    with pytest.raises(ValueError):
        from amllm.generate import Generator
        Generator(model, batch=1, max_len=16)


def test_config_validation():
    with pytest.raises(AssertionError):
        tiny_cfg(bank_groups=0)
    with pytest.raises(AssertionError):
        tiny_cfg(n_hops=3, bank_groups=4)
    with pytest.raises(AssertionError):
        tiny_cfg(bank_norm="nope")
    assert ModelConfig.from_dict({"arch": "amllm"}).bank_groups == 1  # older configs without the new fields still load


# --- AD02a: anytime (multi-exit) loss switch -------------------------------------------------------------------------

def test_anytime_off_adds_nothing_to_the_output():
    model = build_model(tiny_cfg()).train()
    _, info = model(torch.randint(0, 256, (2, 9)), hard_halt=False)
    assert "hop_states" not in info


def test_anytime_on_returns_a_state_per_hop_and_last_equals_the_final_state():
    torch.manual_seed(1)
    model = build_model(tiny_cfg(n_hops=3, anytime_weight=0.5)).train()
    x = torch.randint(0, 256, (2, 9))
    logits, info = model(x, hard_halt=False)
    assert len(info["hop_states"]) == 3
    assert torch.allclose(model._head(info["hop_states"][-1]), logits, atol=1e-6)
    model.eval()
    assert "hop_states" not in model(x, hard_halt=True)[1]  # evaluation never collects


def test_anytime_loss_trains_the_early_hops_and_the_head():
    from amllm.train import F  # the same cross-entropy the training loop uses
    torch.manual_seed(2)
    model = build_model(tiny_cfg(n_hops=3, halt_rule="none", anytime_weight=1.0)).train()
    x, y = torch.randint(0, 256, (2, 9)), torch.randint(0, 256, (2, 9))

    def first_hop_grad(weight):
        model.zero_grad()
        logits, info = model(x, hard_halt=False)
        loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
        if weight > 0:
            inter = info["hop_states"][:-1]
            loss = loss + weight * sum(F.cross_entropy(model._head(s).reshape(-1, 256), y.reshape(-1)) for s in inter) / len(inter)
        loss.backward()
        return model.attention.qkv.weight.grad.clone()

    g_on, g_off = first_hop_grad(1.0), first_hop_grad(0.0)
    assert torch.isfinite(g_on).all() and not torch.allclose(g_on, g_off)


def test_anytime_weight_validation_and_training_step(tmp_path):
    with pytest.raises(AssertionError):
        tiny_cfg(anytime_weight=-0.1)
    from amllm.train import TrainConfig, run
    data = tmp_path / "toy.txt"
    data.write_bytes(bytes(range(32, 120)) * 80)
    cfg = tiny_cfg(max_seq_len=16, n_hops=3, anytime_weight=0.5, halt_rule="none")
    rep = run(cfg, TrainConfig(data=str(data), steps=3, batch=2, eval_every=100, eval_windows=4, eval_batch=2, amp=False,
                               run_dir=str(tmp_path / "run")))
    assert torch.isfinite(torch.tensor(rep["train_loss_last"]))


def test_separate_banks_keep_separate_usage_statistics():
    torch.manual_seed(0)
    cfg = tiny_cfg(n_hops=3, bank_groups=3, halt_rule="none", n_modules=2, experts_per_module=2, module_topk=1, expert_topk=1)
    model = build_model(cfg).eval()
    with torch.no_grad():
        _, info = model(torch.randint(0, 256, (4, 16)), hard_halt=True)
    assert info["expert_counts"].shape == (3 * 2, 2)           # 12 distinct experts are 12 counters, not 4
    assert info["module_counts"].shape == (3 * 2,)
    assert int(info["expert_counts"].sum()) == 3 * 4 * 16 * 1  # every token, hop and route counted exactly once
    for g in range(3):                                          # each bank counts only its own hop
        assert int(info["expert_counts"][g * 2:(g + 1) * 2].sum()) == 4 * 16


def test_shared_bank_statistics_keep_their_old_shape():
    model = build_model(tiny_cfg(n_hops=3, halt_rule="none")).eval()
    with torch.no_grad():
        _, info = model(torch.randint(0, 256, (2, 8)), hard_halt=True)
    assert info["expert_counts"].shape == (4, 3) and info["module_counts"].shape == (4,)


# --- FD04: attention sharing switches --------------------------------------------------------------------------------

FD04_NEW = ("extra_attentions", "extra_attn_norms", "adapter_down", "adapter_up")


def test_fd04_defaults_add_no_state_dict_keys():
    assert not any(k.startswith(FD04_NEW) for k in build_model(tiny_cfg()).state_dict())


def test_attention_group_mapping_and_expert_only_hops():
    model = build_model(tiny_cfg(n_hops=6, attn_groups=3, halt_rule="none"))
    x = torch.zeros(1, 4, model.cfg.d_model)
    calls = []
    for i, attn in enumerate([model.attention, *model.extra_attentions]):
        attn.register_forward_hook(lambda *_, i=i: calls.append(i))
    for hop in range(6):
        model.attention_at(hop, x)
    assert calls == [0, 0, 1, 1, 2, 2]
    every2 = build_model(tiny_cfg(n_hops=6, attn_every=2, attn_groups=3, halt_rule="none"))
    assert [every2.attention_at(h, x) is None for h in range(6)] == [False, True] * 3
    assert every2.cfg.attention_hops == 3 and len(every2.extra_attentions) == 2


def test_expert_only_hops_count_fewer_attention_applications_and_flops():
    torch.manual_seed(0)
    x = torch.randint(0, 256, (2, 9))
    full = build_model(tiny_cfg(n_hops=4, halt_rule="none")).eval()
    half = build_model(tiny_cfg(n_hops=4, halt_rule="none", attn_every=2)).eval()
    with torch.no_grad():
        i_full, i_half = full(x, hard_halt=True)[1], half(x, hard_halt=True)[1]
    assert (i_full["attention_applications"], i_half["attention_applications"]) == (4, 2)
    assert i_half["flops_per_token"] < i_full["flops_per_token"]


def test_adapter_starts_as_the_shared_attention_model_and_is_counted():
    torch.manual_seed(5)
    shared = build_model(tiny_cfg(halt_rule="none")).eval()
    adapted = build_model(tiny_cfg(halt_rule="none", attn_adapter_rank=4)).eval()
    adapted.load_state_dict(shared.state_dict(), strict=False)
    x = torch.randint(0, 256, (2, 9))
    with torch.no_grad():
        (ls, i_s), (la, i_a) = shared(x, hard_halt=True), adapted(x, hard_halt=True)
    assert torch.allclose(ls, la, atol=1e-6)
    d, hops = shared.cfg.d_model, shared.cfg.n_hops
    assert i_a["flops_per_token"] - i_s["flops_per_token"] == pytest.approx((hops - 1) * 4 * d * 4)
    extra = sum(p.numel() for p in adapted.parameters()) - sum(p.numel() for p in shared.parameters())
    assert extra == (hops - 1) * 2 * d * 4


@pytest.mark.parametrize("kw", [dict(attn_groups=2), dict(attn_adapter_rank=4), dict(attn_every=2, attn_groups=2)])
def test_fd04_variants_train_and_every_new_part_gets_gradient(kw):
    torch.manual_seed(3)
    model = AdaptiveModularLM(tiny_cfg(n_hops=4, halt_rule="none", **kw))
    x, y = torch.randint(0, 256, (2, 9)), torch.randint(0, 256, (2, 9))
    logits, info = model(x, hard_halt=False)
    (torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) + info["aux_loss"]).backward()
    new = {n: p for n, p in model.named_parameters() if n.startswith(FD04_NEW)}
    assert new
    for n, p in new.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), n
    if "attn_adapter_rank" in kw:  # the zero-initialised up projection is what gets the first gradient
        assert any(p.grad.abs().sum() > 0 for n, p in new.items() if n.startswith("adapter_up"))


def test_fd04_causality():
    torch.manual_seed(2)
    model = build_model(tiny_cfg(halt_rule="none", n_hops=4, attn_every=2, attn_groups=2, attn_adapter_rank=2)).eval()
    a = torch.randint(0, 256, (1, 10))
    b = a.clone()
    b[0, 7:] = (b[0, 7:] + 5) % 256
    with torch.no_grad():
        assert torch.allclose(model(a, hard_halt=True)[0][:, :7], model(b, hard_halt=True)[0][:, :7], atol=1e-5)


def test_fd04_validation_and_fast_path_refusal():
    with pytest.raises(AssertionError):
        tiny_cfg(n_hops=4, attn_groups=5)
    with pytest.raises(AssertionError):
        tiny_cfg(n_hops=4, attn_every=2, attn_groups=3)   # only 2 attention hops
    with pytest.raises(AssertionError):
        tiny_cfg(unshared_attention=True, attn_groups=2)  # FD04 replaces unshared_attention
    with pytest.raises(AssertionError):
        tiny_cfg(attn_per_hop=False, attn_every=2)
    from amllm.fastpath import static_hard_forward
    with pytest.raises(ValueError):
        static_hard_forward(build_model(tiny_cfg(attn_every=2)).eval(), torch.randint(0, 256, (1, 8)))


# --- AD03: adaptive K (expert_topp) ----------------------------------------------------------------------------------

def test_expert_topp_off_keeps_the_state_dict_and_p1_keeps_every_route():
    torch.manual_seed(4)
    base = build_model(tiny_cfg(halt_rule="none")).eval()
    full = build_model(tiny_cfg(halt_rule="none", expert_topp=1.0)).eval()
    full.load_state_dict(base.state_dict())  # strict: adaptive K adds no parameters
    x = torch.randint(0, 256, (2, 9))
    with torch.no_grad():
        (lb, ib), (lf, i_f) = base(x, hard_halt=True), full(x, hard_halt=True)
    # p = 1 keeps a route while the mass before it is < 1, i.e. all top-k routes: same output up to summation order
    assert torch.allclose(lb, lf, atol=1e-5)
    assert torch.equal(ib["expert_counts"], i_f["expert_counts"]) and ib["flops_per_token"] == i_f["flops_per_token"]


@pytest.mark.parametrize("arch", ["amllm", "moe"])
def test_small_topp_drops_routes_and_flops_follow(arch):
    torch.manual_seed(6)
    kw = dict(arch=arch, halt_rule="none")
    base, low = build_model(tiny_cfg(**kw)).eval(), build_model(tiny_cfg(**kw, expert_topp=1e-6)).eval()
    low.load_state_dict(base.state_dict())
    x = torch.randint(0, 256, (2, 9))
    with torch.no_grad():
        ib, il = base(x, hard_halt=True)[1], low(x, hard_halt=True)[1]
    assert int(il["expert_counts"].sum()) * 2 == int(ib["expert_counts"].sum())  # top-2 -> exactly one expert each
    assert il["flops_per_token"] < ib["flops_per_token"]
    assert il["flops_per_token_executed"] >= il["flops_per_token"]


@pytest.mark.parametrize("weighting", ["topk_softmax", "full_softmax", "straight_through"])
def test_topp_weights_and_gradients(weighting):
    from amllm.routing import HierarchicalModuleBank
    torch.manual_seed(1)
    bank = HierarchicalModuleBank(16, 2, 1, 4, 3, 8, weighting, expert_topp=0.5)
    scores = torch.randn(5, 1, 4, requires_grad=True)
    idx, w, keep = bank._select_topp(scores, 3, 0.5)
    assert keep[..., 0].all() and (w[~keep] == 0).all()
    probs = scores.softmax(-1).gather(-1, idx)
    assert ((probs.cumsum(-1) - probs)[keep] < 0.5).all() and ((probs.cumsum(-1) - probs)[~keep] >= 0.5).all()
    if weighting != "full_softmax":
        assert torch.allclose(w.sum(-1), torch.ones(5, 1))
    w.sum().backward() if weighting != "topk_softmax" else (w * torch.randn_like(w)).sum().backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.abs().sum() > 0


def test_topp_trains_and_is_refused_by_fast_paths_and_random_routing():
    torch.manual_seed(2)
    model = AdaptiveModularLM(tiny_cfg(n_hops=3, expert_topp=0.6, route_weighting="straight_through"))
    x, y = torch.randint(0, 256, (2, 9)), torch.randint(0, 256, (2, 9))
    logits, info = model(x, hard_halt=False)
    (torch.nn.functional.cross_entropy(logits.reshape(-1, 256), y.reshape(-1)) + info["aux_loss"]).backward()
    assert torch.isfinite(model.bank.expert_router_w.grad).all() and model.bank.expert_router_w.grad.abs().sum() > 0
    with pytest.raises(AssertionError):
        tiny_cfg(expert_topp=0.5, router_mode="random")
    with pytest.raises(AssertionError):
        tiny_cfg(expert_topp=1.5)
    from amllm.fastpath import static_hard_forward
    with pytest.raises(ValueError):
        static_hard_forward(build_model(tiny_cfg(expert_topp=0.5)).eval(), torch.randint(0, 256, (1, 8)))
