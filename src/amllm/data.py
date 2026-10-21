"""Byte-level code corpus: build a package-disjoint train/val/test split and sample batches.

The split is decided per *package* (hash of the package name), not per file, so files of
one library never leak between train and evaluation. Content is deduplicated by SHA-256.

    python -m amllm.data build --src ~/project/node_modules --out data/ts_corpus --max-mb 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

EXTENSIONS = (".ts", ".tsx")
DOC_SEPARATOR = b"\n\x00\n"
SPLITS = ("train", "val", "test")


def package_key(path: Path) -> str:
    parts = path.parts
    if "node_modules" not in parts:
        return str(path.parent)
    i = len(parts) - 1 - parts[::-1].index("node_modules")
    name = parts[i + 1] if i + 1 < len(parts) else "?"
    if name.startswith("@") and i + 2 < len(parts):
        name += "/" + parts[i + 2]
    return name


def split_of(package: str, val_frac: float, test_frac: float) -> str:
    u = int(hashlib.sha256(package.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def looks_minified(text: bytes) -> bool:
    lines = text.count(b"\n") + 1
    return len(text) / lines > 300


def iter_sources(src: Path, include_dts: bool):
    for root, dirs, files in os.walk(src, followlinks=False):
        dirs.sort()
        for name in sorted(files):
            if not name.endswith(EXTENSIONS):
                continue
            if name.endswith(".d.ts") and not include_dts:
                continue
            yield Path(root) / name


def build(srcs: list[Path], out: Path, max_bytes: int, val_frac: float, test_frac: float,
          include_dts: bool, max_file_kb: int) -> dict:
    seen: set[str] = set()
    buckets: dict[str, list[bytes]] = {s: [] for s in SPLITS}
    packages: dict[str, set[str]] = {s: set() for s in SPLITS}
    total = 0
    skipped = {"duplicate": 0, "too_large": 0, "minified": 0, "non_utf8": 0}
    for src in srcs:
        for path in iter_sources(src, include_dts):
            if total >= max_bytes:
                break
            try:
                if path.stat().st_size > max_file_kb * 1024:
                    skipped["too_large"] += 1
                    continue
                text = path.read_bytes()
            except OSError:
                continue
            try:
                text.decode("utf-8")
            except UnicodeDecodeError:
                skipped["non_utf8"] += 1
                continue
            if looks_minified(text):
                skipped["minified"] += 1
                continue
            digest = hashlib.sha256(text).hexdigest()
            if digest in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(digest)
            pkg = package_key(path)
            split = split_of(pkg, val_frac, test_frac)
            buckets[split].append(text.replace(b"\x00", b""))
            packages[split].add(pkg)
            total += len(text)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": [str(s) for s in srcs], "include_dts": include_dts,
                "val_frac": val_frac, "test_frac": test_frac, "skipped": skipped, "splits": {}}
    for split in SPLITS:
        blob = DOC_SEPARATOR.join(buckets[split])
        (out / f"{split}.bin").write_bytes(blob)
        manifest["splits"][split] = {
            "bytes": len(blob), "files": len(buckets[split]), "packages": len(packages[split]),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "package_names": sorted(packages[split]),
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split(data_dir: str | Path, split: str) -> torch.Tensor:
    path = Path(data_dir)
    if path.is_file():  # legacy single text file: use it for every split (smoke tests only)
        return torch.from_numpy(np.frombuffer(path.read_bytes(), dtype=np.uint8).copy())
    return torch.from_numpy(np.fromfile(path / f"{split}.bin", dtype=np.uint8))


def split_id(data_dir: str | Path) -> str:
    manifest = Path(data_dir) / "manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text())
        return hashlib.sha256("".join(m["splits"][s]["sha256"] for s in SPLITS).encode()).hexdigest()[:16]
    return hashlib.sha256(Path(data_dir).read_bytes()).hexdigest()[:16]


def random_batch(data: torch.Tensor, batch: int, block: int, gen: torch.Generator, device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(data) < block + 2:
        raise ValueError("dataset is smaller than block size")
    offsets = torch.randint(0, len(data) - block - 1, (batch,), generator=gen)
    windows = data[offsets[:, None] + torch.arange(block + 1)].long().to(device, non_blocking=True)
    return windows[:, :-1], windows[:, 1:]


CODE_BYTE_CLASSES = {0: "space/tab", 1: "newline", 2: "letter/_/$", 3: "digit", 4: "punct/operator", 5: "other/non-ascii"}


def byte_classes(data: torch.Tensor) -> torch.Tensor:
    lut = torch.full((256,), 5, dtype=torch.uint8)
    for c in b" \t\r":
        lut[c] = 0
    lut[ord("\n")] = 1
    for c in b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_$":
        lut[c] = 2
    for c in b"0123456789":
        lut[c] = 3
    for c in range(33, 127):
        if lut[c] == 5:
            lut[c] = 4
    return lut[data.long()]


def load_categories(data_dir: str | Path, split: str, data: torch.Tensor) -> tuple[torch.Tensor, dict[int, str]]:
    """Per-byte evaluation categories: task labels if the split ships a .cat.bin, else byte classes."""
    path = Path(data_dir)
    cat_file = path / f"{split}.cat.bin"
    if path.is_dir() and cat_file.exists():
        names = json.loads((path / "manifest.json").read_text())["categories"]
        return torch.from_numpy(np.fromfile(cat_file, dtype=np.uint8)), {int(k): v for k, v in names.items()}
    return byte_classes(data), CODE_BYTE_CLASSES


def eval_batches(data: torch.Tensor, n_windows: int, batch: int, block: int, device, cats: torch.Tensor | None = None):
    """Deterministic, evenly spaced windows: identical for every model evaluated on this split.

    Yields (x, y, target_categories) where the category belongs to the predicted byte y."""
    offsets = torch.linspace(0, len(data) - block - 2, n_windows).long()
    for i in range(0, n_windows, batch):
        o = offsets[i:i + batch]
        index = o[:, None] + torch.arange(block + 1)
        windows = data[index].long().to(device)
        c = cats[index[:, 1:]].long() if cats is not None else None
        yield windows[:, :-1], windows[:, 1:], c


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--src", nargs="+", required=True)
    b.add_argument("--out", default="data/ts_corpus")
    b.add_argument("--max-mb", type=float, default=24)
    b.add_argument("--val-frac", type=float, default=0.05)
    b.add_argument("--test-frac", type=float, default=0.05)
    b.add_argument("--include-dts", action="store_true")
    b.add_argument("--max-file-kb", type=int, default=256)
    a = p.parse_args()
    m = build([Path(s).expanduser() for s in a.src], Path(a.out), int(a.max_mb * 2**20),
              a.val_frac, a.test_frac, a.include_dts, a.max_file_kb)
    for split, info in m["splits"].items():
        print(f"{split:5s} {info['bytes'] / 2**20:7.2f} MB  files={info['files']:6d}  packages={info['packages']}")
    print("skipped:", m["skipped"])


if __name__ == "__main__":
    main()
