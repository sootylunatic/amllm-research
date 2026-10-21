**English** | [Magyar](README.hu.md)

# Data

The repository contains **no training data**; the corpora are rebuilt locally and deterministically.

| Folder | Source | How to build |
|---|---|---|
| `toy_typescript.txt` | hand-written 986-byte synthetic sample (smoke tests only) | part of the repository |
| `ts_corpus/` | `.ts/.tsx` files of third-party open-source npm packages from a local `node_modules` folder | `python -m amllm.data build --src <node_modules ...> --out data/ts_corpus --max-mb 32 --val-frac 0.1 --test-frac 0.1` |
| `synthetic/` | generated repeat / copy / add tasks | `python -m amllm.synth --out data/synthetic` |

The code corpus is split **by package** (SHA-256 of the package name), so files of one library cannot end up in both
the training and the validation/test part; files are deduplicated by content. The `manifest.json` contains local
paths and is not published. Package licences differ: the corpus is built for local research only — do not
redistribute it.

Do **not** add code from your own (non-open-source) projects to the corpus: it may contain secrets or personal data.
