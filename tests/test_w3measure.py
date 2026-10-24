"""W3 measurement (amllm.w3measure): every part runs on CPU for the W3 model kinds, and the metrics mean what they say."""
import pytest
import torch
from torch.nn import functional as F

from amllm import w3measure as w3
from amllm.config import ModelConfig
from amllm.data import eval_batches
from amllm.model import build_model

BASE = dict(max_seq_len=32, d_model=16, n_heads=4, n_modules=4, experts_per_module=3, module_topk=2, expert_topk=2,
            n_hops=4, expert_hidden=24, n_layers=3, ffn_hidden=32, n_experts=4, moe_topk=2)
KINDS = {
    "fdc1_like": dict(arch="amllm", halt_rule="none", unshared_attention=True, bank_groups=2, n_modules=1,
                      experts_per_module=6, module_topk=1, route_weighting="straight_through"),
    "ad03_like": dict(arch="amllm", unshared_attention=True, expert_topp=0.7, route_weighting="straight_through"),
    "moe": dict(arch="moe"),
    "dense": dict(arch="dense"),
}
VAL = torch.randint(0, 256, (4000,), generator=torch.Generator().manual_seed(1))


def make(kind):
    torch.manual_seed(0)
    model = build_model(ModelConfig(**{**BASE, **KINDS[kind]})).eval()
    if kind == "ad03_like":
        with torch.no_grad():
            model.halt_head.weight.normal_(0, 2.0)
            model.halt_head.bias.fill_(0.3)
    return model


@pytest.mark.parametrize("kind", list(KINDS))
def test_every_part_runs_on_cpu(kind):
    model = make(kind)
    parts = {"position_nll": dict(windows=8, batch=4), "work": dict(batch=4), "latency": dict(batch=2, warmup=1, repeats=2),
             "decode": dict(batches=(1, 2), prompt=8, new=4, repeats=1), "vram": dict(batches=(1,), contexts=(16,)),
             "hotset": dict(windows=4, batch=2), "generation": dict(n_prompts=3, prompt=8, new=12)}
    cpu = torch.device("cpu")
    for name, kw in parts.items():
        out = getattr(w3, name)(model, VAL, cpu, **kw)
        assert isinstance(out, dict) and out, name


def test_position_nll_mean_equals_the_model_loss():
    model, cpu = make("moe"), torch.device("cpu")
    r = w3.position_nll(model, VAL, cpu, windows=8, batch=8, bucket=8)
    x, y, _ = next(eval_batches(VAL, 8, 8, 32, cpu))
    with torch.no_grad():
        ref = F.cross_entropy(model(x)[0].reshape(-1, 256), y.reshape(-1)).item()
    assert r["nll_mean"] == pytest.approx(ref, abs=1e-4) and len(r["nll_by_bucket"]) == 4


def test_padding_free_dispatch_computes_only_routed_rows():
    model, cpu = make("fdc1_like"), torch.device("cpu")
    r = w3.work(model, VAL, cpu, batch=4)
    routed_rows = model.cfg.n_hops * model.cfg.module_topk * model.cfg.expert_topk
    assert r["loop"]["expert_rows_per_token"] == pytest.approx(routed_rows)
    assert r["bmm"]["expert_rows_per_token"] >= routed_rows
    assert r["loop"]["flops_per_token_executed"] == pytest.approx(r["loop"]["flops_per_token_routed"])


def test_kv_cache_formula_matches_the_generator():
    from amllm.generate import Generator
    for kind in ("fdc1_like", "moe"):
        model = make(kind)
        gen = Generator(model, batch=3, max_len=20, cache_dtype=torch.bfloat16)
        assert gen.cache_bytes() == w3.kv_cache_bytes(model, 3, 20)


def test_lru_and_concentration():
    hit, misses = w3.lru_hit_rate([[0], [1], [0], [2], [0], [1]], capacity=2)
    assert misses == 4 and hit == pytest.approx(2 / 6)       # 0 m, 1 m, 0 h, 2 m (evicts 1), 0 h, 1 m
    c = w3.concentration(torch.tensor([90, 5, 3, 2]), fracs=(0.25,))
    assert c["top25pct_share"] == pytest.approx(0.9)


def test_hotset_counts_every_kept_route_once():
    model, cpu = make("fdc1_like"), torch.device("cpu")
    r = w3.hotset(model, VAL, cpu, windows=4, batch=2, cache_fracs=(1.0,))
    assert len(r["groups"]) == model.cfg.bank_groups
    for g in r["groups"]:  # a cache holding every expert of the bank misses each expert at most once
        assert g["lru"]["cache_100pct"]["hit_rate"] > 0.9


def test_loop_metrics():
    looped = [5, 6, 7] + [1, 2, 3, 4] * 10
    m = w3.loop_metrics(looped)
    assert m["ends_in_loop"] and m["repeated_4gram_rate"] > 0.7
    fresh = list(range(40))
    m = w3.loop_metrics(fresh)
    assert not m["ends_in_loop"] and m["repeated_4gram_rate"] == 0 and m["distinct_1"] == 1.0


def test_greedy_generation_is_deterministic_and_sampling_follows_the_seed():
    model, cpu = make("moe"), torch.device("cpu")
    prompts = next(eval_batches(VAL, 2, 2, 8, cpu))[0]
    g1 = w3.sample_continuations(model, prompts, 6, cpu, None)
    g2 = w3.sample_continuations(model, prompts, 6, cpu, None)
    s1 = w3.sample_continuations(model, prompts, 6, cpu, 0.8, seed=3)
    s2 = w3.sample_continuations(model, prompts, 6, cpu, 0.8, seed=3)
    assert torch.equal(g1, g2) and torch.equal(s1, s2)


def test_main_end_to_end_reads_only_validation(monkeypatch, tmp_path):
    import json
    model = make("fdc1_like")
    run = tmp_path / "FDC1" / "amllm_f5_s42"
    run.mkdir(parents=True)
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict()}, run / "checkpoint.pt")
    loaded = []
    monkeypatch.setattr(w3, "load_split", lambda data, split: loaded.append(split) or VAL)
    out = tmp_path / "m.json"
    monkeypatch.setattr("sys.argv", ["w3measure", "--runs", str(run), "--out", str(out), "--parts", "position_nll", "work"])
    w3.main()
    r = json.loads(out.read_text())
    assert loaded == ["val"] and set(r["runs"]["FDC1/amllm_f5_s42"]) >= {"position_nll", "work"}
    assert r["provenance"]["train_code_sha256"]


def test_hotset_routes_under_the_same_autocast_as_the_other_measurements(monkeypatch):
    calls = []
    real = w3.autocast
    monkeypatch.setattr(w3, "autocast", lambda device: calls.append(device) or real(device))
    w3.hotset(make("moe"), VAL, torch.device("cpu"), windows=2, batch=2)
    assert calls  # the traced forward runs inside autocast (bf16 on CUDA)


def test_a_resumed_run_measures_the_parts_still_missing(monkeypatch, tmp_path):
    import json
    model = make("moe")
    run = tmp_path / "B_moe" / "moe_f5_s42"
    run.mkdir(parents=True)
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict()}, run / "checkpoint.pt")
    monkeypatch.setattr(w3, "load_split", lambda data, split: VAL)
    out = tmp_path / "m.json"
    monkeypatch.setattr("sys.argv", ["w3measure", "--runs", str(run), "--out", str(out), "--parts", "work"])
    w3.main()
    first = json.loads(out.read_text())["runs"]["B_moe/moe_f5_s42"]
    monkeypatch.setattr("sys.argv", ["w3measure", "--runs", str(run), "--out", str(out), "--parts", "work", "position_nll"])
    w3.main()
    second = json.loads(out.read_text())["runs"]["B_moe/moe_f5_s42"]
    assert "position_nll" in second and second["work"] == first["work"]  # missing part added, finished part kept
