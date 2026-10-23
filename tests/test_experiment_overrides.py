"""--set overrides must reach the configuration of every *_f5 preset (regression: moe_f5 silently ignored them)."""
import sys

import pytest

from amllm import experiment as ex


def test_moe_f5_receives_the_requested_router_rule():
    cfg, applied = ex.effective_config("moe_f5", {"route_weighting": "straight_through"})
    assert applied and cfg.route_weighting == "straight_through" and cfg.arch == "moe"


def test_dense_f5_and_amllm_f5_receive_overrides_too():
    assert ex.effective_config("dense_f5", {"ffn_hidden": 512})[0].ffn_hidden == 512
    cfg, applied = ex.effective_config("amllm_f5", {"halt_rule": "none", "bank_groups": 6, "route_weighting": "full_softmax"})
    assert applied and (cfg.halt_rule, cfg.bank_groups, cfg.route_weighting) == ("none", 6, "full_softmax")


def test_legacy_non_halting_presets_keep_ignoring_overrides():
    cfg, applied = ex.effective_config("amllm_fixed", {"n_hops": 8})
    assert not applied and cfg.n_hops == ex.PRESETS["amllm_fixed"]["n_hops"]


def test_command_line_path_builds_the_requested_moe_without_training(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(ex, "run", lambda cfg, tc: seen.append(cfg) or {"eval": {}, "arch": cfg.arch, "parameters": {"total": 1, "non_embedding": 1},
                                                                         "train_tokens_per_s": 1.0, "train_seconds": 1.0})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["experiment", "--name", "t", "--presets", "moe_f5", "--seeds", "42", "--steps", "1",
                                      "--data", "data/x", "--set", "route_weighting=straight_through"])
    ex.main()
    assert [c.route_weighting for c in seen] == ["straight_through"]
