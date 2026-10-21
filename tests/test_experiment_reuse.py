"""--skip-existing must reuse a finished run only if configuration, data split and training code are identical."""
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import pytest

from amllm import env
from amllm.config import ModelConfig
from amllm.data import split_id
from amllm.experiment import check_reusable
from amllm.train import TrainConfig

TOY = Path(__file__).resolve().parent.parent / "data" / "toy_typescript.txt"


def stored_run(tmp_path, cfg, tc, **override):
    run = tmp_path / "run_s42"
    run.mkdir()
    record = {"model": cfg.to_dict(), "train": asdict(tc), "data_split_id": split_id(tc.data),
              "train_code_sha256": env.train_code_hash()}
    record.update(override)
    record = {k: v for k, v in record.items() if v is not None}
    (run / "config.json").write_text(json.dumps(record))
    (run / "report.json").write_text("{}")
    return run


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "corpus.txt"
    shutil.copy(TOY, data)
    return ModelConfig(), TrainConfig(data=str(data), steps=10)


def test_identical_run_is_reusable(tmp_path, setup):
    cfg, tc = setup
    check_reusable(stored_run(tmp_path, cfg, tc), cfg, tc)


def test_changed_config_is_rejected(tmp_path, setup):
    cfg, tc = setup
    run = stored_run(tmp_path, cfg, tc)
    with pytest.raises(SystemExit, match="model configuration"):
        check_reusable(run, ModelConfig(n_hops=2), tc)


def test_changed_corpus_at_same_path_is_rejected(tmp_path, setup):
    cfg, tc = setup
    run = stored_run(tmp_path, cfg, tc)
    Path(tc.data).write_text(Path(tc.data).read_text() + "\n// changed\n")
    with pytest.raises(SystemExit, match="data split"):
        check_reusable(run, cfg, tc)


def test_changed_training_code_is_rejected(tmp_path, setup):
    cfg, tc = setup
    run = stored_run(tmp_path, cfg, tc, train_code_sha256="0000000000000000")
    with pytest.raises(SystemExit, match="training code"):
        check_reusable(run, cfg, tc)


def test_legacy_run_needs_explicit_permission(tmp_path, setup):
    cfg, tc = setup
    run = stored_run(tmp_path, cfg, tc, train_code_sha256=None)
    with pytest.raises(SystemExit, match="legacy"):
        check_reusable(run, cfg, tc)
    check_reusable(run, cfg, tc, allow_legacy=True)
