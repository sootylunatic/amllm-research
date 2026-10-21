import json

import numpy as np
import pytest

from amllm.corpus_v2 import build, licence_ok
from amllm.data import eval_batches, is_token_corpus, load_categories, load_split, split_id


@pytest.mark.parametrize("meta,ok", [
    ({"license": "MIT"}, True), ({"license": "(MIT OR GPL-3.0)"}, True), ({"license": "Apache-2.0 AND MIT"}, True),
    ({"license": "MIT AND GPL-3.0"}, False), ({"license": "GPL-3.0"}, False), ({}, False),
    ({"license": {"type": "ISC"}}, True), ({"licenses": [{"type": "MIT"}, {"type": "BSD-3-Clause"}]}, True),
    ({"license": "UNLICENSED"}, False), ({"license": "SEE LICENSE IN LICENSE"}, False),
])
def test_licence_filter(meta, ok):
    assert licence_ok(meta) is ok


def _package(root, name, licence, files):
    d = root / "node_modules" / name
    d.mkdir(parents=True)
    (d / "package.json").write_text(json.dumps({"name": name, "license": licence}))
    for fname, text in files.items():
        (d / fname).write_text(text)


def test_build_filters_splits_and_loads_tokens(tmp_path):
    src = tmp_path / "src"
    body = "export function add(a: number, b: number): number {\n  return a + b;\n}\n" * 20
    for i in range(40):
        _package(src, f"pkg{i}", "MIT", {"index.ts": body + f"// {i}\n", "copy.js": body + f"// {i}\n",
                                          "types.d.ts": "declare const x: number;\n"})
    _package(src, "gpl-pkg", "GPL-3.0", {"index.ts": "const secret = 1;\n"})
    _package(src, "dup-pkg", "MIT", {"index.ts": body + "// 0\n"})  # duplicate content of pkg0
    out = tmp_path / "corpus"
    m = build([src / "node_modules"], out, val_frac=0.1, test_frac=0.1, vocab=300, max_file_kb=256,
              max_package_mb=8, max_train_mb=100)
    assert m["filter"]["licence"] == 1 and m["filter"]["duplicate"] == 41  # copy.js of every package + dup-pkg
    names = {s: set(m["splits"][s]["package_names"]) for s in ("train", "val", "test")}
    assert "gpl-pkg" not in set().union(*names.values())
    assert not (names["train"] & names["val"]) and not (names["train"] & names["test"])
    assert is_token_corpus(out)
    train = load_split(out, "train")
    assert train.dtype.is_floating_point is False and int(train.max()) < m["vocab_size"]
    assert len(train) == m["splits"]["train"]["tokens"] and int(train[-1]) == 0  # <|endoftext|> after each file
    assert load_categories(out, "train", train) == (None, {})
    x, y, c = next(eval_batches(train, 2, 2, 16, "cpu"))
    assert x.shape == (2, 16) and c is None and np.array_equal(x[:, 1:].numpy(), y[:, :-1].numpy())
    assert len(split_id(out)) == 16
