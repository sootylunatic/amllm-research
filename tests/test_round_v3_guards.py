"""Round V3 guards: the test split is never read unless asked; the FLOP report counts the dispatch's real work."""
import json

import pytest
import torch

from amllm import data as data_mod
from amllm import train as train_mod
from amllm.model import build_model, set_dispatch
from amllm.train import TrainConfig, run
from test_model import tiny_cfg


def toy(tmp_path):
    p = tmp_path / "toy.txt"
    p.write_bytes(bytes(range(32, 120)) * 80)
    return p


def test_default_run_never_reads_or_reports_the_test_split(tmp_path, monkeypatch):
    real = data_mod.load_split
    seen = []

    def spy(d, split):
        seen.append(split)
        return real(d, split)

    monkeypatch.setattr(train_mod, "load_split", spy)
    rep = run(tiny_cfg(max_seq_len=16, n_hops=3), TrainConfig(data=str(toy(tmp_path)), steps=3, batch=2, eval_every=100,
                                                              eval_windows=4, eval_batch=2, amp=False, run_dir=str(tmp_path / "r")))
    assert "test" not in seen
    assert rep["test_default_mode"] is None
    log = (tmp_path / "r" / "train.log").read_text()
    assert "test[" not in log and "test_default" not in log
    assert json.loads((tmp_path / "r" / "config.json").read_text())["train"]["eval_test"] is False


def test_test_split_only_on_explicit_request(tmp_path):
    rep = run(tiny_cfg(max_seq_len=16, n_hops=3), TrainConfig(data=str(toy(tmp_path)), steps=3, batch=2, eval_every=100,
                                                              eval_windows=4, eval_batch=2, amp=False, eval_test=True,
                                                              run_dir=str(tmp_path / "r")))
    assert rep["test_default_mode"]["val_loss"] > 0


@pytest.mark.parametrize("arch", ["amllm", "moe"])
def test_executed_flops_include_bmm_padding(arch):
    torch.manual_seed(0)
    cfg = tiny_cfg(arch=arch, d_model=16, expert_hidden=32)
    x = torch.randint(0, 256, (2, 12))
    out = {}
    for mode in ("loop", "bmm"):
        model = build_model(cfg).eval()
        torch.manual_seed(0)
        model = build_model(cfg).eval()
        set_dispatch(model, mode)
        with torch.no_grad():
            _, info = model(x, **({"hard_halt": True, "halt_rule": "none"} if arch == "amllm" else {}))
        out[mode] = info
    loop, bmm = out["loop"], out["bmm"]
    assert loop["flops_per_token_executed"] == pytest.approx(loop["flops_per_token"])  # no padding in the loop dispatch
    assert bmm["flops_per_token"] == pytest.approx(loop["flops_per_token"])           # routed work is dispatch-independent
    assert bmm["flops_per_token_executed"] > bmm["flops_per_token"]                   # bmm computes padding / idle experts
    assert bmm["expert_rows_per_token"] > loop["expert_rows_per_token"]


def test_executed_equals_routed_for_dense():
    model = build_model(tiny_cfg(arch="dense")).eval()
    with torch.no_grad():
        _, info = model(torch.randint(0, 256, (2, 8)))
    assert info["flops_per_token_executed"] == info["flops_per_token"]


# --- a numerically broken training is a failed job, not a finished one -------------------------------------------------

def _nan_after(model, calls):
    state = {"n": 0}
    original = model.forward

    def forward(*a, **kw):
        logits, info = original(*a, **kw)
        state["n"] += 1
        if state["n"] > calls:
            logits = logits * float("nan")
        return logits, info
    model.forward = forward
    return model


def test_non_finite_loss_aborts_with_an_error_and_leaves_no_report(tmp_path, monkeypatch):
    real = train_mod.build_model
    monkeypatch.setattr(train_mod, "build_model", lambda cfg: _nan_after(real(cfg), 1))
    run_dir = tmp_path / "r"
    with pytest.raises(FloatingPointError):
        run(tiny_cfg(max_seq_len=16, n_hops=3), TrainConfig(data=str(toy(tmp_path)), steps=5, batch=2, eval_every=100,
                                                            eval_windows=4, eval_batch=2, amp=False, run_dir=str(run_dir)))
    assert not (run_dir / "report.json").exists() and not (run_dir / "checkpoint.pt").exists()
    assert json.loads((run_dir / "FAILED.json").read_text())["step"] == 1


def test_complete_run_reports_the_steps_it_really_did(tmp_path):
    rep = run(tiny_cfg(max_seq_len=16, n_hops=3), TrainConfig(data=str(toy(tmp_path)), steps=4, batch=2, eval_every=100,
                                                              eval_windows=4, eval_batch=2, amp=False, run_dir=str(tmp_path / "r")))
    assert rep["steps"] == 4 == rep["steps_completed"]
