"""F5 corpus v2: licence-checked, deduplicated TS/JS code with a package hold-out and a byte-level BPE tokenizer.

    python -m amllm.corpus_v2 build --src <node_modules dirs / package caches> --out data/code_corpus_v2

Rules (results/phase2/preregistration.md, F5):
- third-party packages only (the caller passes node_modules folders and package caches, never own project sources);
- .ts .tsx .js .mjs .cjs; no .d.ts, *.min.js, files > 256 KB, minified files (mean line length > 300), non-UTF-8;
- the nearest package.json must declare a permissive licence; files without one are dropped;
- dedup by content SHA-256; at most `--max-package-mb` per package (files in sorted path order);
- split by the hash of the package *name* (all versions/copies of a package land in the same split);
- byte-level BPE trained on the training split only; splits are written as uint16 token ids with `<|endoftext|>`
  (id 0) between documents. The manifest records bytes and tokens per split for bits-per-byte conversion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .data import SPLITS, looks_minified, split_of

EXTENSIONS = (".ts", ".tsx", ".js", ".mjs", ".cjs")
PERMISSIVE = {"MIT", "ISC", "APACHE-2.0", "BSD-2-CLAUSE", "BSD-3-CLAUSE", "0BSD", "UNLICENSE", "CC0-1.0", "BLUEOAK-1.0.0"}
EOT = "<|endoftext|>"


def licence_ok(meta: dict) -> bool:
    lic = meta.get("license", meta.get("licenses"))
    if isinstance(lic, list):
        lic = [x.get("type") if isinstance(x, dict) else x for x in lic]
        return bool(lic) and all(isinstance(x, str) and x.strip("()").upper() in PERMISSIVE for x in lic)
    if isinstance(lic, dict):
        lic = lic.get("type")
    if not isinstance(lic, str):
        return False
    # SPDX expressions: "MIT OR Apache-2.0" is fine if any alternative is permissive; AND needs all
    expr = lic.strip("()").upper()
    if " AND " in expr:
        return all(p.strip("() ") in PERMISSIVE for p in expr.split(" AND "))
    return any(p.strip("() ") in PERMISSIVE for p in expr.split(" OR "))


class PackageIndex:
    """Nearest package.json with a name for a file (cached per directory)."""

    def __init__(self):
        self.cache: dict[Path, dict | None] = {}

    def find(self, directory: Path, stop: Path) -> dict | None:
        if directory in self.cache:
            return self.cache[directory]
        meta = None
        pj = directory / "package.json"
        if pj.is_file():
            try:
                m = json.loads(pj.read_text(encoding="utf-8"))
                if isinstance(m, dict) and isinstance(m.get("name"), str) and m["name"]:
                    meta = m
            except (OSError, ValueError):
                pass
        if meta is None and directory != stop and stop in directory.parents:
            meta = self.find(directory.parent, stop)
        self.cache[directory] = meta
        return meta


def collect(srcs: list[Path], max_file_kb: int, max_package_bytes: int) -> tuple[dict[str, list[bytes]], dict]:
    seen: set[str] = set()
    per_package: dict[str, list[bytes]] = {}
    package_bytes: dict[str, int] = {}
    stats = {"candidates": 0, "kept": 0, "duplicate": 0, "too_large": 0, "minified": 0, "non_utf8": 0,
             "no_package": 0, "licence": 0, "package_cap": 0}
    index = PackageIndex()
    for src in srcs:
        for root, dirs, files in os.walk(src, followlinks=False):
            dirs.sort()
            for name in sorted(files):
                if not name.endswith(EXTENSIONS) or name.endswith(".d.ts") or name.endswith(".min.js"):
                    continue
                stats["candidates"] += 1
                path = Path(root) / name
                try:
                    if path.is_symlink() or path.stat().st_size > max_file_kb * 1024:
                        stats["too_large"] += 1
                        continue
                    text = path.read_bytes()
                except OSError:
                    continue
                try:
                    text.decode("utf-8")
                except UnicodeDecodeError:
                    stats["non_utf8"] += 1
                    continue
                if looks_minified(text):
                    stats["minified"] += 1
                    continue
                meta = index.find(path.parent, src)
                if meta is None:
                    stats["no_package"] += 1
                    continue
                if not licence_ok(meta):
                    stats["licence"] += 1
                    continue
                digest = hashlib.sha256(text).hexdigest()
                if digest in seen:
                    stats["duplicate"] += 1
                    continue
                pkg = meta["name"]
                if package_bytes.get(pkg, 0) + len(text) > max_package_bytes:
                    stats["package_cap"] += 1
                    continue
                seen.add(digest)
                per_package.setdefault(pkg, []).append(text.replace(b"\x00", b""))
                package_bytes[pkg] = package_bytes.get(pkg, 0) + len(text)
                stats["kept"] += 1
    return per_package, stats


def build(srcs: list[Path], out: Path, val_frac: float, test_frac: float, vocab: int, max_file_kb: int,
          max_package_mb: float, max_train_mb: float) -> dict:
    from tokenizers import ByteLevelBPETokenizer

    per_package, stats = collect(srcs, max_file_kb, int(max_package_mb * 2**20))
    docs: dict[str, list[str]] = {s: [] for s in SPLITS}
    packages: dict[str, list[str]] = {s: [] for s in SPLITS}
    train_bytes = 0
    for pkg in sorted(per_package):
        split = split_of(pkg, val_frac, test_frac)
        size = sum(len(t) for t in per_package[pkg])
        if split == "train":
            if train_bytes + size > max_train_mb * 2**20:
                stats["train_cap_packages"] = stats.get("train_cap_packages", 0) + 1
                continue
            train_bytes += size
        docs[split].extend(t.decode("utf-8") for t in per_package[pkg])
        packages[split].append(pkg)
    out.mkdir(parents=True, exist_ok=True)
    tok = ByteLevelBPETokenizer(add_prefix_space=False)
    tok.train_from_iterator(iter(docs["train"]), vocab_size=vocab, min_frequency=2, special_tokens=[EOT],
                            show_progress=False)
    tok.save(str(out / "tokenizer.json"))
    assert tok.token_to_id(EOT) == 0
    manifest = {"format": "tokens-uint16", "tokenizer": "byte-level BPE", "vocab_size": tok.get_vocab_size(),
                "eot_id": 0, "tokenizer_sha256": hashlib.sha256((out / "tokenizer.json").read_bytes()).hexdigest(),
                "n_sources": len(srcs), "val_frac": val_frac, "test_frac": test_frac, "max_file_kb": max_file_kb,
                "max_package_mb": max_package_mb, "max_train_mb": max_train_mb, "filter": stats, "splits": {}}
    for split in SPLITS:
        ids: list[int] = []
        n_bytes = 0
        for enc, text in zip(tok.encode_batch(docs[split]), docs[split]):
            ids.extend(enc.ids)
            ids.append(0)
            n_bytes += len(text.encode("utf-8")) + 1  # the separator counts as one byte (phase-1 used 3)
        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(out / f"{split}.bin")
        manifest["splits"][split] = {"bytes": n_bytes, "tokens": int(arr.size), "files": len(docs[split]),
                                     "packages": len(packages[split]), "bytes_per_token": n_bytes / max(1, arr.size),
                                     "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
                                     "package_names": packages[split]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", nargs="+", required=True)
    b.add_argument("--out", default="data/code_corpus_v2")
    b.add_argument("--val-frac", type=float, default=0.03)
    b.add_argument("--test-frac", type=float, default=0.03)
    b.add_argument("--vocab", type=int, default=8192)
    b.add_argument("--max-file-kb", type=int, default=256)
    b.add_argument("--max-package-mb", type=float, default=4.0)
    b.add_argument("--max-train-mb", type=float, default=400.0)
    a = p.parse_args()
    m = build([Path(s).expanduser().resolve() for s in a.src], Path(a.out), a.val_frac, a.test_frac, a.vocab,
              a.max_file_kb, a.max_package_mb, a.max_train_mb)
    for split, info in m["splits"].items():
        print(f"{split:5s} {info['bytes'] / 2**20:8.2f} MB  {info['tokens'] / 1e6:7.2f} M tokens  "
              f"bytes/token={info['bytes_per_token']:.2f}  files={info['files']:6d}  packages={info['packages']}")
    print("filter:", m["filter"])


if __name__ == "__main__":
    main()
