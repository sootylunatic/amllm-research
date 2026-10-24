"""Evaluation on the sealed holdout (docs/03-research/HOLDOUT_PLAN.md). Execution code, not part of train_code_sha256.

    python -m amllm.holdout evaluate --holdout <dir> --split H1 --candidates frozen.json --out results.json --log calls.log
    python -m amllm.holdout compare --results results.json --pairs pairs.json --out comparison.json

* The holdout directory must match its SHA256SUMS and manifest (any change is refused).
* The candidate list must be frozen: {"frozen": true, "runs": [{"run": <run_dir>, "checkpoint_sha256": <hex>}, ...]};
  a checkpoint whose hash differs is refused.
* Every evaluate call appends a line to a hash-chained log (time, split, candidate-list hash, previous line hash).
* Output: per run and package the summed NLL, token count and byte count (aggregates; the text is never written out).
* Every model is evaluated in its default mode on the same non-overlapping windows; each target token is attributed to
  the package of the document it belongs to. Comparisons are paired over packages (seed means per package), with a
  package-level bootstrap CI: the package, not the token, is the independent unit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .model import AdaptiveModularLM, set_diagnostics
from .profile import load_run


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_holdout(holdout: Path, split: str):
    """(token ids, per-token package index, package names, per-package bytes) after verifying every file's hash."""
    sums = {}
    for line in (holdout / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        sums[name.strip()] = digest
    for name, digest in sums.items():
        if sha256_file(holdout / name) != digest:
            raise SystemExit(f"holdout file {name} does not match SHA256SUMS: refusing")
    manifest = json.loads((holdout / "manifest.json").read_text())
    ids = np.fromfile(holdout / f"{split}.bin", dtype=np.uint16)
    if hashlib.sha256(ids.tobytes()).hexdigest() != manifest["splits"][split]["sha256"]:
        raise SystemExit(f"{split}.bin does not match the manifest: refusing")
    docs = json.loads((holdout / f"{split}_documents.json").read_text())
    names = sorted({d["package"] for d in docs})
    pid = {n: i for i, n in enumerate(names)}
    owner = np.full(len(ids), -1, dtype=np.int32)          # the separator after a document belongs to that document
    pbytes = np.zeros(len(names))
    for d in docs:
        owner[d["start"]:d["end"] + 1] = pid[d["package"]]
        pbytes[pid[d["package"]]] += d["bytes"] + 1
    return ids, owner, names, pbytes


def frozen_runs(candidates: Path) -> list[Path]:
    c = json.loads(candidates.read_text())
    if c.get("frozen") is not True:
        raise SystemExit("the candidate list is not frozen: refusing")
    runs = []
    for r in c["runs"]:
        run = Path(r["run"])
        if sha256_file(run / "checkpoint.pt") != r["checkpoint_sha256"]:
            raise SystemExit(f"{run}: checkpoint hash differs from the frozen list: refusing")
        runs.append(run)
    return runs


def append_log(log: Path, entry: dict) -> None:
    prev = "0" * 64
    if log.exists() and log.read_text().strip():
        prev = json.loads(log.read_text().strip().splitlines()[-1])["hash"]
    entry = {**entry, "prev": prev}
    entry["hash"] = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
    with log.open("a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


@torch.no_grad()
def per_package_nll(model, ids: np.ndarray, owner: np.ndarray, n_packages: int, device, batch: int = 16):
    block = model.cfg.max_seq_len
    mode = {"hard_halt": True} if isinstance(model, AdaptiveModularLM) and model.cfg.halt_rule != "none" else {}
    n_win = (len(ids) - 1) // block
    nll_sum, count = np.zeros(n_packages), np.zeros(n_packages)
    for s in range(0, n_win, batch):
        starts = np.arange(s, min(s + batch, n_win)) * block
        x = torch.from_numpy(np.stack([ids[a:a + block] for a in starts]).astype(np.int64)).to(device)
        y = torch.from_numpy(np.stack([ids[a + 1:a + block + 1] for a in starts]).astype(np.int64)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x, **mode)[0]
        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none").cpu().numpy()
        own = np.concatenate([owner[a + 1:a + block + 1] for a in starts])
        ok = own >= 0
        np.add.at(nll_sum, own[ok], nll[ok])
        np.add.at(count, own[ok], 1)
    return nll_sum, count


def evaluate(holdout: Path, split: str, candidates: Path, out: Path, log: Path) -> dict:
    ids, owner, names, pbytes = load_holdout(holdout, split)
    runs = frozen_runs(candidates)
    append_log(log, {"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "split": split,
                     "candidates_sha256": sha256_file(candidates), "runs": len(runs)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = {"split": split, "packages": names, "package_bytes": pbytes.tolist(), "runs": {}}
    for run in runs:
        cfg, model = load_run(run, device)
        set_diagnostics(model, False)
        s, c = per_package_nll(model, ids, owner, len(names), device)
        res["runs"][f"{run.parent.name}/{run.name}"] = {"nll_sum": s.tolist(), "tokens": c.tolist(),
                                                        "nll_mean": float(s.sum() / c.sum())}
        del model
        out.write_text(json.dumps(res))
    return res


def config_package_means(res: dict, config: str) -> tuple[np.ndarray, np.ndarray]:
    """Per package: the seed-mean NLL per token and the token count, for the runs of one configuration."""
    runs = [r for k, r in res["runs"].items() if k.split("/")[0] == config]
    if not runs:
        raise KeyError(config)
    s = np.mean([np.asarray(r["nll_sum"]) for r in runs], axis=0)
    c = np.asarray(runs[0]["tokens"])
    return s, c


def paired(res: dict, a: str, b: str, n_boot: int = 10000, seed: int = 0) -> dict:
    """Token-weighted mean of (a − b) over packages, with a package-level bootstrap 95% CI."""
    sa, c = config_package_means(res, a)
    sb, _ = config_package_means(res, b)
    keep = c > 0
    sa, sb, c = sa[keep], sb[keep], c[keep]
    point = float((sa - sb).sum() / c.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(c), (n_boot, len(c)))
    boots = (sa[idx] - sb[idx]).sum(1) / c[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    share = float(((sa - sb) / c < 0).mean())
    return {"delta": point, "ci95": [float(lo), float(hi)], "packages": int(len(c)), "share_packages_a_better": share}


def compare(results: Path, pairs: Path, out: Path) -> dict:
    res = json.loads(results.read_text())
    spec = json.loads(pairs.read_text())
    cmp = {"pairs": {f"{a} - {b}": paired(res, a, b) for a, b in spec["pairs"]}}
    configs = sorted({k.split("/")[0] for k in res["runs"]})
    cmp["config_means"] = {c: float(config_package_means(res, c)[0].sum() / config_package_means(res, c)[1].sum())
                           for c in configs}
    out.write_text(json.dumps(cmp, indent=1))
    return cmp


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--holdout", required=True)
    e.add_argument("--split", default="H1")
    e.add_argument("--candidates", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--log", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--results", required=True)
    c.add_argument("--pairs", required=True)
    c.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "evaluate":
        evaluate(Path(a.holdout), a.split, Path(a.candidates), Path(a.out), Path(a.log))
    else:
        compare(Path(a.results), Path(a.pairs), Path(a.out))


if __name__ == "__main__":
    main()
