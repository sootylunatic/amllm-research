"""The cloud benchmark must time the same training step as train.run: identical losses for the same seed/schedule."""
import json
import shutil
from pathlib import Path

import pytest
import torch

from amllm.cloudbench import step_stats, train_steps
from amllm.config import ModelConfig
from amllm.train import TrainConfig, run

TOY = Path(__file__).resolve().parent.parent / "data" / "toy_typescript.txt"
SMALL = dict(max_seq_len=32, d_model=32, n_heads=2, n_layers=2, ffn_hidden=64, n_experts=4, moe_topk=2,
             n_modules=3, experts_per_module=2, module_topk=1, expert_topk=1, expert_hidden=32, n_hops=2)


@pytest.mark.parametrize("arch", ["amllm", "moe", "dense"])
def test_benchmark_step_matches_train_run(tmp_path, arch):
    data = tmp_path / "corpus.txt"
    shutil.copy(TOY, data)
    cfg = ModelConfig(arch=arch, **SMALL)
    tc = TrainConfig(data=str(data), steps=3, batch=4, eval_every=1000, eval_windows=4, eval_batch=4,
                     dispatch="bmm", run_dir=str(tmp_path / "run"))
    run(cfg, tc)
    curve = json.loads((tmp_path / "run" / "report.json").read_text())["train_curve"]  # steps 0 and 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # the device train.run chose
    _, times, losses, _ = train_steps(cfg, tc, 3, device, schedule_steps=3)
    assert len(times) == 3
    # train.run logs losses rounded to 4 decimals; on CUDA, bf16 kernels may differ in the last logged digit
    tol = 1e-4 if device.type == "cuda" else 0
    assert [losses[s] for s, _, _ in curve] == pytest.approx([loss for _, loss, _ in curve], abs=5e-5 + tol)


def test_step_stats():
    r = step_stats([0.1] * 19 + [1.0], tokens_per_step=1000)
    assert r["median_step_s"] == 0.1 and r["p95_step_s"] == 1.0 and r["steps"] == 20
    assert r["tokens_per_s"] == pytest.approx(1000 / r["mean_step_s"])


def test_parse_set_accepts_strings():
    from amllm.experiment import parse_set
    assert parse_set(["halt_rule=none", "module_topk=2", "ponder_coefficient=0.005", "unshared_attention=true",
                      "halt_bias_init=-2.0"]) == {"halt_rule": "none", "module_topk": 2, "ponder_coefficient": 0.005,
                                                  "unshared_attention": True, "halt_bias_init": -2.0}
