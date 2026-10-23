"""Phase 2 gate evaluator: decisions only on complete, consistent run sets (audit of F1)."""
import json

import pytest

from amllm import gates

CONTROLS = ("moe_e8h128", "moe_e4")


def write_stage(root, losses, ideal, code="c0de", split="s1", seeds=gates.PREREG_SEEDS, skip=()):
    for preset, loss in losses.items():
        for seed in seeds:
            if (preset, seed) in skip:
                continue
            d = root / "F1" / f"{preset}_s{seed}"
            d.mkdir(parents=True)
            mode = "fixed" if preset.startswith("moe") else "hard_surv_0.5"
            (d / "report.json").write_text(json.dumps({
                "parameters": {"total": 1}, "test_default_mode": {"val_loss": loss},
                "eval": {mode: {"val_loss": loss, "mean_hops": 4.0, "flops_per_token": ideal[preset],
                                "flops_per_token_ideal": ideal[preset]}}}))
            (d / "config.json").write_text(json.dumps({"train_code_sha256": code, "data_split_id": split}))


def passing_stage():
    losses = {"amllm_ua_mk1_fs": 1.40, "amllm_ua_ek1_fs": 1.40, "amllm_ua_both1_fs": 1.40, "amllm_ua_fs": 1.39,
              "moe_e8h128": 1.39, "moe_e4": 1.39}
    ideal = {k: (1.0 if "1_fs" in k else 2.0) for k in losses}
    return losses, ideal


def test_complete_consistent_stage_gives_full_success(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "PHASE2", tmp_path)
    write_stage(tmp_path, *passing_stage())
    assert gates.quality_compute_gate("f1")["verdict"] == "full success"


def test_incomplete_seed_set_gives_no_decision(tmp_path, monkeypatch):
    """Audit: with only four seeds the old evaluator still reported full success."""
    monkeypatch.setattr(gates, "PHASE2", tmp_path)
    write_stage(tmp_path, *passing_stage(), skip={("amllm_ua_mk1_fs", 46)})
    with pytest.raises(gates.GateError, match="seeds"):
        gates.quality_compute_gate("f1")


def test_mixed_training_code_gives_no_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "PHASE2", tmp_path)
    losses, ideal = passing_stage()
    write_stage(tmp_path, {k: v for k, v in losses.items() if k != "moe_e4"}, ideal)
    write_stage(tmp_path, {"moe_e4": losses["moe_e4"]}, ideal, code="other")
    with pytest.raises(gates.GateError, match="training-code"):
        gates.quality_compute_gate("f1")


@pytest.mark.parametrize("args,expected", [
    ((True, True, True, True), "full success"),
    ((False, True, True, True), "partial success"),
    ((False, True, False, True), "compute only"),
    ((False, False, True, False), "quality only"),
    ((False, False, False, False), "failure"),
])
def test_verdict_table_is_complete(args, expected):
    assert gates.classify(*args).startswith(expected)


# ---------------------------------------------------------------------------------------------- F5 (audit 2026-10-21)
def write_f5(root, loss, lat, rows, syn, p1, long=None, hard_rows=10.0):
    ev = {"runs": {}}
    for exp, seeds, losses in (("F5", gates.F5_SEEDS, loss), ("F5_2x", (42,), long or {})):
        for m, value in losses.items():
            for s in seeds:
                d = root / exp / f"{m}_s{s}"
                d.mkdir(parents=True)
                mode = "hard_surv_0.5" if m.startswith("amllm") else "fixed"
                (d / "report.json").write_text(json.dumps({"eval": {mode: {"val_loss": value}},
                                                          "steps": 9000 if exp == "F5" else 18000}))
                (d / "config.json").write_text(json.dumps({"train_code_sha256": "c", "data_split_id": "s",
                                                          "model": {"arch": m}}))
                if exp == "F5":
                    ev["runs"][f"F5/{m}_s{s}"] = {
                        "b2_latency_b8_t512": {"median_ms": lat[m], "expert_rows_per_token_executed": rows[m],
                                               "expert_rows_per_token_hard_routed": hard_rows},
                        "syntax": {"valid_continuation_share": syn[m]}, "tasks": {"pass_at_1": p1[m]},
                        "bpb": {"val": {"bits_per_byte": 1.0}, "test": {"bits_per_byte": 1.0}}, "mean_hops": 5.0,
                        "decode": {f"b{b}": {"decode_tokens_per_s_graph": 1.0, "peak_memory_mb": 1.0} for b in (1, 32)}}
    (root / "F5" / "f5eval.json").write_text(json.dumps(ev))


F5_OK = dict(loss={"amllm_f5": 1.00, "moe_f5": 1.00, "dense_f5": 1.1}, lat={"amllm_f5": 1.0, "moe_f5": 1.0, "dense_f5": 0.5},
             rows={"amllm_f5": 12.0, "moe_f5": 12.0, "dense_f5": 0.0}, syn={"amllm_f5": 0.5, "moe_f5": 0.5, "dense_f5": 0.5},
             p1={"amllm_f5": 0.0, "moe_f5": 0.0, "dense_f5": 0.0})


def test_f5_syntax_with_pass_at_1_floor_cannot_meet_c(tmp_path, monkeypatch):
    """Audit P1: C must not be met by syntax alone when no model solves a task."""
    monkeypatch.setattr(gates, "PHASE2", tmp_path)
    write_f5(tmp_path, **{**F5_OK, "lat": {"amllm_f5": 2.0, "moe_f5": 1.0, "dense_f5": 0.5}},
             long={"amllm_f5": 1.0, "moe_f5": 1.0})
    r = gates.f5()
    assert r["C"]["informative"] is False and r["C"]["met"] is False and r["verdict"] == "partial"


def test_f5_speed_needs_executed_work_not_above_moe(tmp_path, monkeypatch):
    monkeypatch.setattr(gates, "PHASE2", tmp_path)
    write_f5(tmp_path, **{**F5_OK, "rows": {"amllm_f5": 13.0, "moe_f5": 12.0, "dense_f5": 0.0}},
             long={"amllm_f5": 1.0, "moe_f5": 1.0})
    r = gates.f5()
    assert r["S"]["latency_met"] and not r["S"]["work_met"] and r["verdict"] == "partial"


def test_f5_stability_detects_ranking_reversal_below_threshold(tmp_path, monkeypatch):
    """Audit P2: 0.99 -> 1.01 reverses the ranking without crossing 1.02."""
    monkeypatch.setattr(gates, "PHASE2", tmp_path)
    write_f5(tmp_path, **{**F5_OK, "loss": {"amllm_f5": 0.99, "moe_f5": 1.0, "dense_f5": 1.1}},
             long={"amllm_f5": 1.01, "moe_f5": 1.0})
    r = gates.f5()
    assert r["verdict"] == "met (unstable)" and r["stability_2x"]["ranking_reversed"]


def test_f5_dense_success_does_not_make_c_informative(tmp_path, monkeypatch):
    """Second audit: AM-LLM 0%, MoE 0%, Dense 10% pass@1 must not count as an informative coding comparison."""
    monkeypatch.setattr(gates, "PHASE2", tmp_path)
    write_f5(tmp_path, **{**F5_OK, "lat": {"amllm_f5": 2.0, "moe_f5": 1.0, "dense_f5": 0.5},
                          "p1": {"amllm_f5": 0.0, "moe_f5": 0.0, "dense_f5": 0.1}},
             long={"amllm_f5": 1.0, "moe_f5": 1.0})
    r = gates.f5()
    assert r["C"]["informative"] is False and r["C"]["met"] is False and r["verdict"] == "partial"


@pytest.mark.parametrize("field, value", [("train_code_sha256", "other"), ("data_split_id", "other"),
                                          ("model", {"arch": "x"}), ("steps", 9000)])
def test_f5_2x_runs_must_be_comparable(tmp_path, monkeypatch, field, value):
    """Review: a 2x run with other code, split, model or step count must not yield a stability statement."""
    monkeypatch.setattr(gates, "PHASE2", tmp_path)
    write_f5(tmp_path, **F5_OK, long={"amllm_f5": 1.0, "moe_f5": 1.0})
    d = tmp_path / "F5_2x" / "moe_f5_s42"
    target = d / ("report.json" if field == "steps" else "config.json")
    data = json.loads(target.read_text())
    data[field] = value
    target.write_text(json.dumps(data))
    with pytest.raises(gates.GateError):
        gates.f5()


def test_invalid_gate_exits_non_zero(tmp_path, monkeypatch):
    """Review: an invalid gate (no decision) must not look like success to a caller."""
    monkeypatch.setattr(gates, "PHASE2", tmp_path)
    monkeypatch.setattr("sys.argv", ["gates", "f5"])
    with pytest.raises(SystemExit) as e:
        gates.main()
    assert e.value.code == 3 and "invalid" in json.loads((tmp_path / "gates.json").read_text())["F5"]["verdict"]
