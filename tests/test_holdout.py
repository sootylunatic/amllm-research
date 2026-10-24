"""Holdout evaluator (amllm.holdout): refusals, hash-chained log, package attribution and paired package-level statistics."""
import hashlib
import json

import numpy as np
import pytest
import torch

from amllm import holdout as ho
from amllm.model import build_model
from test_model import tiny_cfg


def make_holdout(tmp_path):
    d = tmp_path / "holdout"
    d.mkdir()
    rng = np.random.default_rng(0)
    ids, docs = [], []
    for i in range(12):
        n = int(rng.integers(20, 60))
        docs.append({"package": f"pkg{i % 5}", "start": len(ids), "end": len(ids) + n, "sha256": "x", "bytes": 3 * n})
        ids += rng.integers(1, 256, n).tolist() + [0]
    arr = np.asarray(ids, dtype=np.uint16)
    arr.tofile(d / "H1.bin")
    (d / "H1_documents.json").write_text(json.dumps(docs))
    (d / "manifest.json").write_text(json.dumps({"splits": {"H1": {"sha256": hashlib.sha256(arr.tobytes()).hexdigest()}}}))
    (d / "SHA256SUMS").write_text("".join(f"{ho.sha256_file(f)}  {f.name}\n" for f in sorted(d.iterdir())))
    return d


def make_run(tmp_path, name, seed):
    torch.manual_seed(seed)
    model = build_model(tiny_cfg(halt_rule="none"))
    run = tmp_path / name / f"amllm_f5_s{seed}"
    run.mkdir(parents=True)
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict()}, run / "checkpoint.pt")
    return run


def frozen(tmp_path, runs, ok=True):
    f = tmp_path / "candidates.json"
    f.write_text(json.dumps({"frozen": ok, "runs": [{"run": str(r), "checkpoint_sha256": ho.sha256_file(r / "checkpoint.pt")}
                                                    for r in runs]}))
    return f


def test_attribution_covers_every_document_token(tmp_path):
    ids, owner, names, pbytes = ho.load_holdout(make_holdout(tmp_path), "H1")
    assert (owner >= 0).all() and len(names) == 5 and pbytes.sum() > 0


def test_refuses_a_changed_holdout_an_unfrozen_list_and_a_changed_checkpoint(tmp_path):
    d = make_holdout(tmp_path)
    run = make_run(tmp_path, "A", 42)
    with pytest.raises(SystemExit):
        ho.frozen_runs(frozen(tmp_path, [run], ok=False))
    f = frozen(tmp_path, [run])
    torch.save({"config": {}, "state_dict": {}}, run / "checkpoint.pt")
    with pytest.raises(SystemExit):
        ho.frozen_runs(f)
    (d / "H1_documents.json").chmod(0o644)
    (d / "H1_documents.json").write_text("[]")
    with pytest.raises(SystemExit):
        ho.load_holdout(d, "H1")


def test_evaluate_logs_a_hash_chain_and_compare_is_paired(tmp_path):
    d = make_holdout(tmp_path)
    runs = [make_run(tmp_path, "A", s) for s in (42, 43)] + [make_run(tmp_path, "B", s) for s in (42, 43)]
    log, out = tmp_path / "calls.log", tmp_path / "res.json"
    ho.evaluate(d, "H1", frozen(tmp_path, runs), out, log)
    ho.evaluate(d, "H1", frozen(tmp_path, runs), out, log)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert lines[1]["prev"] == lines[0]["hash"] and len(lines) == 2
    res = json.loads(out.read_text())
    same = ho.paired(res, "A", "A")
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0]
    ab = ho.paired(res, "A", "B")                                   # same seeds, same init: A and B are identical models
    assert abs(ab["delta"]) < 1e-9
    assert res["runs"]["A/amllm_f5_s42"]["nll_mean"] > 0


def test_paired_ci_brackets_the_point_estimate_for_different_models(tmp_path):
    d = make_holdout(tmp_path)
    runs = [make_run(tmp_path, "A", 42), make_run(tmp_path, "B", 7)]
    out = tmp_path / "res.json"
    res = ho.evaluate(d, "H1", frozen(tmp_path, runs), out, tmp_path / "calls.log")
    r = ho.paired(res, "A", "B")
    assert r["delta"] != 0 and r["ci95"][0] <= r["delta"] <= r["ci95"][1] and r["packages"] == 5
