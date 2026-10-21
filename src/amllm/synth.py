"""Synthetic byte tasks of graded difficulty, with a per-byte category file for evaluation.

Each line is one task. Only the *answer* bytes carry a task category; everything else is 0.
  R:aaaaaaaa      repetition  - trivial after the first answer byte        (category 1)
  C:qwzrt>qwzrt   copy        - needs attention retrieval of the prompt    (category 2)
  A:347+589=639   addition    - sum written least-significant digit first  (category 3)
                                so it is learnable left-to-right with carries
Validation/test copy/add lines that also occur in train are dropped (exact-duplicate filter);
repeat lines are exempt because only 26 x 9 distinct ones exist.

    python -m amllm.synth --out data/synthetic
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

CATEGORIES = {0: "prompt/other", 1: "repeat", 2: "copy", 3: "add"}
LETTERS = "abcdefghijklmnopqrstuvwxyz"


def task(rng: random.Random) -> tuple[str, int, int]:
    """Returns (line, answer_start, category)."""
    kind = rng.choice("RCA")
    if kind == "R":
        n = rng.randint(4, 12)
        return "R:" + rng.choice(LETTERS) * n, 3, 1  # first repeated byte is not yet predictable
    if kind == "C":
        s = "".join(rng.choice(LETTERS) for _ in range(rng.randint(4, 10)))
        prompt = f"C:{s}>"
        return prompt + s, len(prompt), 2
    a, b = (rng.randint(0, 10 ** rng.randint(2, 4) - 1) for _ in range(2))
    prompt = f"A:{a}+{b}="
    return prompt + str(a + b)[::-1], len(prompt), 3


def generate(n_bytes: int, seed: int, exclude: set[str] | None = None) -> tuple[bytes, bytes, set[str]]:
    rng = random.Random(seed)
    text, cats, lines = bytearray(), bytearray(), set()
    while len(text) < n_bytes:
        line, start, cat = task(rng)
        if exclude and cat != 1 and line in exclude:  # repeat lines have only 26x9 variants
            continue
        lines.add(line)
        b = (line + "\n").encode()
        text += b
        cats += bytes([0] * start + [cat] * (len(line) - start) + [0])
    return bytes(text), bytes(cats), lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--train-mb", type=float, default=4)
    p.add_argument("--eval-mb", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=2026)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"categories": CATEGORIES, "seed": a.seed, "splits": {}}
    train, train_cats, train_lines = generate(int(a.train_mb * 2**20), a.seed)
    splits = {"train": (train, train_cats)}
    for i, split in enumerate(("val", "test"), start=1):
        t, c, _ = generate(int(a.eval_mb * 2**20), a.seed + i, exclude=train_lines)
        splits[split] = (t, c)
    for split, (t, c) in splits.items():
        (out / f"{split}.bin").write_bytes(t)
        (out / f"{split}.cat.bin").write_bytes(c)
        manifest["splits"][split] = {"bytes": len(t), "sha256": hashlib.sha256(t).hexdigest(),
                                     "category_bytes": {CATEGORIES[k]: c.count(k) for k in CATEGORIES}}
        print(split, manifest["splits"][split]["bytes"], manifest["splits"][split]["category_bytes"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
