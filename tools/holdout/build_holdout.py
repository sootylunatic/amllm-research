"""Build the independent holdout H-A (docs/03-research/HOLDOUT_PLAN.md): fresh TS/JS packages from the npm registry that
appear in no split of code_corpus_v2 by name, family, repository, exact content or near-duplicate content.

    python tools/holdout/build_holdout.py --v2 data/code_corpus_v2 --work /srv/holdout_build --out /srv/holdout

Runs on the relay (CPU + network). Steps:
  1. v2 index: package names and families of all three splits (name, scope, name stem, repository URL from the registry);
     every v2 document decoded from the token files -> exact SHA-256 and MinHash signature (5-token shingles).
  2. candidates: npm search over fixed topic queries (the list below; recorded), deterministic order by name hash;
     family / repository exclusion against v2.
  3. download the latest tarball of each candidate, extract, and apply the v2 file filters (amllm.corpus_v2.collect:
     licence, size, minified, UTF-8, exact dedup, 1 MB per package).
  4. leakage filter against v2: exact SHA-256 match -> file out; MinHash Jaccard >= 0.8 with any v2 document -> file out;
     a package with > 20% of its files out is excluded entirely.
  5. split into H1 (primary) and H2 (sealed reserve) by family hash, so a family never straddles; tokens with the v2
     tokenizer (unchanged), uint16 with <|endoftext|> between documents; per-document package index for package-level
     statistics; stop adding packages to a split once it holds `--target-tokens`.
  6. write H1.bin / H2.bin, documents.json (per document: package, token range, sha256), manifest.json, audit.json,
     SHA256SUMS; the output directory is made read-only.
No model is evaluated here. The content of the holdout is never published (package names and hashes may be).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

REGISTRY = "https://registry.npmjs.org"
QUERIES = ("keywords:typescript", "keywords:javascript", "keywords:cli", "keywords:parser", "keywords:react",
           "keywords:node", "keywords:util", "keywords:http", "keywords:testing", "keywords:math", "keywords:date",
           "keywords:string", "keywords:color", "keywords:markdown", "keywords:crypto", "keywords:validation",
           "keywords:async", "keywords:stream", "keywords:logger", "keywords:config", "keywords:graph",
           "keywords:database", "keywords:image", "keywords:game", "keywords:compiler", "keywords:vue",
           "keywords:svelte", "keywords:websocket", "keywords:queue", "keywords:cache")
SHINGLE = 5
N_PERM = 64
BANDS = 16
NEAR_DUP = 0.8
PACKAGE_DROP = 0.2
EOT = 0


# ------------------------------------------------------------------------------------------------- families
def family_keys(name: str, repo: str | None = None) -> set[str]:
    """Keys under which two packages count as one family: the scope, the bare name, its stem before the first
    '-' / '.' / '_', and the normalised repository URL."""
    name = name.lower()
    keys = set()
    if name.startswith("@") and "/" in name:
        scope, base = name.split("/", 1)
        keys.add(f"scope:{scope}")
    else:
        base = name
    keys.add(f"name:{base}")
    stem = re.split(r"[-._]", base)[0]
    if stem:
        keys.add(f"stem:{stem}")
    if repo:
        r = normalise_repo(repo)
        if r:
            keys.add(f"repo:{r}")
    return keys


def normalise_repo(url: str) -> str | None:
    url = url.lower().strip()
    url = re.sub(r"^git\+", "", url)
    url = re.sub(r"\.git$", "", url)
    url = re.sub(r"^(https?|git|ssh)://(git@)?", "", url)
    url = url.replace("git@github.com:", "github.com/").replace("github:", "github.com/")
    m = re.match(r"(github\.com|gitlab\.com|bitbucket\.org)/([^/]+)/([^/#]+)", url)
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}" if m else None


def repo_of(meta: dict) -> str | None:
    r = meta.get("repository")
    if isinstance(r, dict):
        r = r.get("url")
    return r if isinstance(r, str) else None


# ------------------------------------------------------------------------------------------------- minhash
_RNG = np.random.default_rng(20261024)
_A = _RNG.integers(1, (1 << 61) - 1, N_PERM, dtype=np.uint64)
_B = _RNG.integers(0, (1 << 61) - 1, N_PERM, dtype=np.uint64)


def shingles(tokens: list[str], k: int = SHINGLE) -> np.ndarray:
    if len(tokens) < k:
        tokens = tokens + [""] * (k - len(tokens))
    hs = {int.from_bytes(hashlib.blake2b(" ".join(tokens[i:i + k]).encode(), digest_size=8).digest(), "little")
          for i in range(len(tokens) - k + 1)}
    return np.fromiter(hs, dtype=np.uint64)


def minhash(sh: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        v = (sh[None, :] * _A[:, None] + _B[:, None]) & np.uint64((1 << 61) - 1)
    return v.min(axis=1)


def code_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_$][A-Za-z0-9_$]*|\d+|\S", text)


class LshIndex:
    def __init__(self, bands: int = BANDS):
        self.bands, self.rows = bands, N_PERM // bands
        self.tables: list[dict[bytes, list[int]]] = [dict() for _ in range(bands)]
        self.sigs: list[np.ndarray] = []

    def add(self, sig: np.ndarray) -> None:
        i = len(self.sigs)
        self.sigs.append(sig)
        for b in range(self.bands):
            self.tables[b].setdefault(sig[b * self.rows:(b + 1) * self.rows].tobytes(), []).append(i)

    def max_similarity(self, sig: np.ndarray) -> float:
        cands = set()
        for b in range(self.bands):
            cands.update(self.tables[b].get(sig[b * self.rows:(b + 1) * self.rows].tobytes(), ()))
        return max((float((self.sigs[c] == sig).mean()) for c in cands), default=0.0)


# ------------------------------------------------------------------------------------------------- npm
def get_json(url: str, tries: int = 4) -> dict:
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(2 * (t + 1))
    return {}


def packument(name: str) -> dict:
    return get_json(f"{REGISTRY}/{urllib.parse.quote(name, safe='@')}")


def search(query: str, size: int = 250, pages: int = 4) -> list[str]:
    out = []
    for page in range(pages):
        d = get_json(f"{REGISTRY}/-/v1/search?text={urllib.parse.quote(query)}&size={size}&from={page * size}")
        out += [o["package"]["name"] for o in d.get("objects", [])]
    return out


def download(url: str, dest: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
            safe = [m for m in tf.getmembers() if (m.isfile() or m.isdir()) and not m.name.startswith(("/", ".."))
                    and ".." not in Path(m.name).parts]
            tf.extractall(dest, members=safe, filter="data")
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------- build
def v2_documents(v2: Path, tok) -> tuple[set[str], LshIndex, list[str]]:
    hashes, index, names = set(), LshIndex(), []
    manifest = json.loads((v2 / "manifest.json").read_text())
    for split in ("train", "val", "test"):
        names += manifest["splits"][split]["package_names"]
        ids = np.fromfile(v2 / f"{split}.bin", dtype=np.uint16)
        cuts = np.flatnonzero(ids == EOT)
        start = 0
        batch = []
        for end in list(cuts) + [len(ids)]:
            if end > start:
                batch.append(ids[start:end].tolist())
            start = end + 1
            if len(batch) == 512 or (end == len(ids) and batch):
                for text in tok.decode_batch(batch, skip_special_tokens=False):
                    hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
                    index.add(minhash(shingles(code_tokens(text))))
                batch = []
    return hashes, index, names


def split_of_family(keys: set[str]) -> str:
    anchor = sorted(k for k in keys if k.startswith(("scope:", "stem:")))[0]
    return "H1" if int(hashlib.sha256(anchor.encode()).hexdigest()[:8], 16) % 2 == 0 else "H2"


def main():
    from tokenizers import Tokenizer
    from amllm.corpus_v2 import collect

    p = argparse.ArgumentParser()
    p.add_argument("--v2", default="data/code_corpus_v2")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--target-tokens", type=int, default=3_000_000)
    p.add_argument("--max-package-mb", type=float, default=1.0)
    a = p.parse_args()
    v2, work, out = Path(a.v2), Path(a.work), Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty: a holdout is built once")
    tok = Tokenizer.from_file(str(v2 / "tokenizer.json"))
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)

    log("v2 index: documents")
    v2_hashes, v2_index, v2_names = v2_documents(v2, tok)
    log(f"v2: {len(v2_names)} packages, {len(v2_hashes)} documents")
    v2_keys = set()
    repo_cache = work / "v2_repos.json"
    repos = json.loads(repo_cache.read_text()) if repo_cache.exists() else {}
    for n in v2_names:
        if n not in repos:
            try:
                d = packument(n)
                latest = d.get("dist-tags", {}).get("latest")
                repos[n] = repo_of(d.get("versions", {}).get(latest, d)) or repo_of(d)
            except Exception:
                repos[n] = None
        v2_keys |= {k for k in family_keys(n, repos[n]) if not k.startswith("stem:") or len(k) > 7}
    repo_cache.write_text(json.dumps(repos))
    log(f"v2 family keys: {len(v2_keys)}")

    names = sorted({n for q in QUERIES for n in search(q)}, key=lambda n: hashlib.sha256(n.encode()).hexdigest())
    log(f"candidates from {len(QUERIES)} queries: {len(names)}")
    audit = {"queries": QUERIES, "candidates": len(names), "excluded_family": [], "excluded_meta": [],
             "excluded_download": [], "excluded_leakage_package": [], "files_exact_dup": 0, "files_near_dup": 0,
             "params": {"shingle": SHINGLE, "minhash_perm": N_PERM, "bands": BANDS, "near_dup_jaccard": NEAR_DUP,
                        "package_drop_share": PACKAGE_DROP, "max_package_mb": a.max_package_mb,
                        "target_tokens_per_split": a.target_tokens}}
    splits = {"H1": {"docs": [], "tokens": 0, "packages": []}, "H2": {"docs": [], "tokens": 0, "packages": []}}
    seen_families: set[str] = set()
    for name in names:
        if all(s["tokens"] >= a.target_tokens for s in splits.values()):
            break
        try:
            d = packument(name)
            latest = d["dist-tags"]["latest"]
            meta = d["versions"][latest]
        except Exception:
            audit["excluded_meta"].append(name)
            continue
        keys = family_keys(name, repo_of(meta) or repo_of(d))
        if keys & v2_keys or keys & seen_families:
            audit["excluded_family"].append(name)
            continue
        split = split_of_family(keys)
        if splits[split]["tokens"] >= a.target_tokens:
            continue
        dest = work / "raw" / hashlib.sha256(name.encode()).hexdigest()[:16]
        if not dest.exists() and not download(meta["dist"]["tarball"], dest):
            audit["excluded_download"].append(name)
            continue
        per_package, _ = collect([dest], 256, int(a.max_package_mb * 2**20))
        files, total = [], 0
        for t in (t for pkg in sorted(per_package) for t in per_package[pkg]):  # one cap for the whole tarball:
            if total + len(t) > a.max_package_mb * 2**20:                       # a monorepo tarball holds several
                break                                                           # package.json names
            files.append(t)
            total += len(t)
        if not files:
            audit["excluded_meta"].append(name)
            continue
        kept, dropped = [], 0
        for text in files:
            s = text.decode("utf-8")
            if hashlib.sha256(text).hexdigest() in v2_hashes or hashlib.sha256(s.encode()).hexdigest() in v2_hashes:
                audit["files_exact_dup"] += 1
                dropped += 1
            elif v2_index.max_similarity(minhash(shingles(code_tokens(s)))) >= NEAR_DUP:
                audit["files_near_dup"] += 1
                dropped += 1
            else:
                kept.append(s)
        if dropped > PACKAGE_DROP * len(files) or not kept:
            audit["excluded_leakage_package"].append(name)
            continue
        seen_families |= keys
        enc = tok.encode_batch(kept)
        for text, e in zip(kept, enc):
            splits[split]["docs"].append({"package": name, "version": latest, "ids": e.ids,
                                          "sha256": hashlib.sha256(text.encode()).hexdigest(),
                                          "bytes": len(text.encode())})
            splits[split]["tokens"] += len(e.ids) + 1
        splits[split]["packages"].append({"name": name, "version": latest, "licence": meta.get("license"),
                                          "repository": repo_of(meta), "files": len(kept)})
        log(f"{split} +{name}@{latest}: {len(kept)} files; H1 {splits['H1']['tokens']:,} H2 {splits['H2']['tokens']:,}")

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "npm registry, latest versions, built " + time.strftime("%Y-%m-%d"),
                "tokenizer_sha256": hashlib.sha256((v2 / "tokenizer.json").read_bytes()).hexdigest(),
                "v2_manifest_sha256": hashlib.sha256((v2 / "manifest.json").read_bytes()).hexdigest(), "splits": {}}
    for name, s in splits.items():
        ids, docs = [], []
        for d in s["docs"]:
            docs.append({"package": d["package"], "start": len(ids), "end": len(ids) + len(d["ids"]),
                         "sha256": d["sha256"], "bytes": d["bytes"]})
            ids += d["ids"] + [EOT]
        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(out / f"{name}.bin")
        (out / f"{name}_documents.json").write_text(json.dumps(docs))
        manifest["splits"][name] = {"tokens": int(arr.size), "documents": len(docs), "packages": s["packages"],
                                    "bytes": sum(d["bytes"] + 1 for d in docs), "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    (out / "audit.json").write_text(json.dumps(audit, indent=1))
    sums = "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n" for f in sorted(out.iterdir()))
    (out / "SHA256SUMS").write_text(sums)
    for f in out.iterdir():
        f.chmod(0o444)
    out.chmod(0o555)
    log("done", {k: (v["tokens"], len(v["packages"])) for k, v in manifest["splits"].items()})


if __name__ == "__main__":
    main()
