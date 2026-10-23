"""F5 evaluation (results/phase2/preregistration.md, F5 + addenda): per checkpoint, after training.

    AMLLM_TYPESCRIPT=<typescript package dir> python -m amllm.f5eval --runs <run dirs> --out results/phase2/F5/f5eval.json

All in bf16, Triton dispatch where the model has experts, diagnostics off; AM-LLM with hard halting semantics.
  bpb       exact bits per byte of the training report's validation/test loss: the byte length of every predicted token
            of the same evaluation windows (byte-level BPE: one vocabulary character = one byte; <|endoftext|> = 1)
  syntax    200 prompts (test split); greedy continuation of 128 tokens cut at the last complete line. Main measure
            (revised after the audit, before any F5 checkpoint was evaluated): over the prompts that parse without
            errors alone, the share whose continuation contains at least one non-blank, non-comment line AND whose
            prompt + continuation has no TypeScript parse error before the end of the text. An empty continuation
            scores 0. The file's real continuation scored the same way is the reference (ceiling).
  tasks     20 write + 10 bug-fix tasks (f5tasks.py), greedy 256 tokens, cut after the first line "}"; tsc over
            solution + runner (type errors fail); the runner executes in a bubblewrap sandbox (no network, no host
            file system or environment, read-only work dir, time limit) and prints results; the expected outputs are
            compared in Python. No sandbox -> error (no unsandboxed fallback).
  latency   B2 (Triton + static masked execution + CUDA Graph) forward, batch 8 x 512, median of 50; executed expert
            rows per token on the same path and on the reference hard path
  decode    generated tokens/s (KV cache, CUDA Graph decode step) at batch 1 and 32, cache size and peak memory
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import torch

from . import env
from .data import eval_batches, load_split
from .f3bench import timed
from .f4bench import greedy as greedy_batch
from .f5tasks import TASKS, UNDEFINED
from .fastpath import GraphRunner, static_forward
from .generate import Generator
from .model import AdaptiveModularLM, expert_banks, set_diagnostics, set_dispatch
from .profile import load_run

N_PROMPTS, PROMPT_TOKENS, NEW_TOKENS, TASK_TOKENS = 200, 256, 128, 256
JS_DIR = Path(__file__).parent / "js"


def typescript_dir(arg: str | None) -> Path:
    path = Path(arg or os.environ.get("AMLLM_TYPESCRIPT", "")).expanduser()
    if not (path / "lib" / "tsc.js").is_file():
        raise SystemExit("set --typescript or AMLLM_TYPESCRIPT to an installed typescript package directory")
    return path


# ----------------------------------------------------------------------------------------------- bits per byte
def token_bytes(tok) -> torch.Tensor:
    """Byte length of every token id (byte-level BPE: each vocabulary character stands for exactly one byte)."""
    vocab = tok.get_vocab()
    lut = torch.zeros(tok.get_vocab_size(), dtype=torch.long)
    for s, i in vocab.items():
        lut[i] = len(s)
    lut[0] = 1  # <|endoftext|> counts as one separator byte, as in the corpus manifest
    return lut


def window_bytes(data: torch.Tensor, lut: torch.Tensor, n_windows: int, block: int) -> tuple[int, int]:
    n_tok = n_bytes = 0
    for _, y, _ in eval_batches(data, n_windows, 64, block, "cpu"):
        n_tok += y.numel()
        n_bytes += int(lut[y].sum())
    return n_tok, n_bytes


# ----------------------------------------------------------------------------------------------- syntax
def parse_errors(texts: list[str], ts_dir: Path) -> list[dict]:
    out = subprocess.run(["node", str(JS_DIR / "ts_parse.js"), str(ts_dir)], input=json.dumps(texts),
                         capture_output=True, text=True, check=True, timeout=600)
    return json.loads(out.stdout)


def complete_lines(text: str) -> str:
    return text[:text.rfind("\n") + 1] if "\n" in text else ""


def code_mask(text: str) -> list[bool]:
    """Per character: True if it is code (not inside a // or /* */ comment). A small scanner that keeps the comment
    state across lines and skips string and template literals; an unclosed comment runs to the end of the text."""
    mask, i, n, quote = [True] * len(text), 0, len(text), None
    while i < n:
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote or (c == "\n" and quote != "`"):
                quote = None
            i += 1
        elif c in "'\"`":
            quote = c
            i += 1
        elif text.startswith("//", i) or text.startswith("/*", i):
            end = text.find("\n", i) if text[i + 1] == "/" else text.find("*/", i + 2)
            end = n if end < 0 else end + (0 if text[i + 1] == "/" else 2)
            for k in range(i, end):
                mask[k] = False
            i = end
        else:
            i += 1
    return mask


def has_code_line(continuation: str, prompt: str = "") -> bool:
    """The continuation contains code outside comments: at least one non-whitespace code character after the prompt.
    The scan runs over prompt + continuation, so a comment or string opened in the prompt is respected."""
    text = prompt + continuation
    mask = code_mask(text)
    return any(mask[k] and not text[k].isspace() for k in range(len(prompt), len(text)))


def test_prompts(test: torch.Tensor, tok) -> list[tuple[list[int], list[int]]]:
    """(prompt, real continuation) for evenly spaced test files with >= PROMPT_TOKENS + NEW_TOKENS tokens; the prompt
    is the first PROMPT_TOKENS tokens cut back to the last complete line."""
    ends = (test == 0).nonzero().flatten().tolist()
    starts = [0] + [e + 1 for e in ends[:-1]]
    docs = [(s, e) for s, e in zip(starts, ends) if e - s >= PROMPT_TOKENS + NEW_TOKENS]
    pick = [docs[round(i * (len(docs) - 1) / (N_PROMPTS - 1))] for i in range(min(N_PROMPTS, len(docs)))]
    out = []
    for s, _ in pick:
        ids = test[s:s + PROMPT_TOKENS].tolist()
        while ids and "\n" not in tok.decode(ids[-1:]):
            ids.pop()
        out.append((ids, test[s + len(ids):s + len(ids) + NEW_TOKENS].tolist()))
    return out


def score_continuations(prompts: list[str], continuations: list[str], ts_dir: Path) -> dict:
    """Main syntax measure (see module doc). `continuations` are already cut at the last complete line."""
    alone = parse_errors(prompts, ts_dir)
    full = parse_errors([p + c for p, c in zip(prompts, continuations)], ts_dir)
    valid_prompt = [a["inner"] == 0 for a in alone]
    nonempty = [has_code_line(c, p) for p, c in zip(prompts, continuations)]
    ok = [v and n and f["inner"] == 0 for v, n, f in zip(valid_prompt, nonempty, full)]
    n_valid = sum(valid_prompt)
    return {"n_prompts": len(prompts), "n_valid_prompts": n_valid,
            "valid_continuation_share": sum(ok) / max(1, n_valid),
            "nonempty_share": sum(n for n, v in zip(nonempty, valid_prompt) if v) / max(1, n_valid),
            "mean_complete_lines": statistics.fmean(c.count("\n") for c in continuations)}


@torch.no_grad()
def greedy(model, ids: list[int], steps: int, device) -> list[int]:
    gen = Generator(model, batch=1, max_len=len(ids) + steps, cache_dtype=torch.bfloat16)
    out = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        token = gen.prefill(torch.tensor([ids], device=device))[:, -1].argmax(-1)
        for _ in range(steps):
            if int(token) == 0:  # <|endoftext|>
                break
            out.append(int(token))
            token = gen.decode_step(token).argmax(-1)
    return out


def syntax_eval(model, tok, prompts, ts_dir: Path, device) -> dict:
    texts = [tok.decode(p) for p, _ in prompts]
    conts = [complete_lines(tok.decode(greedy(model, p, NEW_TOKENS, device))) for p, _ in prompts]
    r = score_continuations(texts, conts, ts_dir)
    r["examples"] = [t[-300:] + "<<<CONTINUATION>>>" + c for t, c in zip(texts[:3], conts[:3])]
    return r


# ----------------------------------------------------------------------------------------------- tasks
def sandbox_cmd(work: Path, argv: list[str]) -> list[str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("bubblewrap (bwrap) is required to execute generated code; refusing to run unsandboxed")
    node = shutil.which("node")
    return [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
            "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", str(work), "/work", "--chdir", "/work", node, "--max-old-space-size=256", *argv]


def run_task(solution: str, task: dict, ts_dir: Path, timeout: float = 10.0) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "solution.ts").write_text(solution)
        (d / "runner.ts").write_text(task["runner"])
        try:
            tsc = subprocess.run(["node", str(ts_dir / "lib" / "tsc.js"), "--target", "es2020", "--module", "commonjs",
                                  "--lib", "es2020", "--outDir", "out", "solution.ts", "runner.ts"],
                                 cwd=d, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:  # one pathological solution fails its task, the evaluation goes on
            return False, "tsc timeout"
        if tsc.returncode != 0:
            return False, "tsc: " + tsc.stdout.strip()[:300]
        try:
            r = subprocess.run(sandbox_cmd(d / "out", ["runner.js"]), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    lines = r.stdout.strip().split("\n")
    try:
        results = json.loads(lines[-1])
    except (json.JSONDecodeError, IndexError):
        return False, f"no result line (exit {r.returncode}): {r.stderr.strip()[-200:]}"
    if not isinstance(results, list) or len(results) != len(task["cases"]):
        return False, "wrong number of results"
    for (args, expected), got in zip(task["cases"], results):
        if not (isinstance(got, dict) and got.get("ok") is True and "v" in got and got["v"] == expected
                and type(got["v"]) is type(expected)):
            return False, f"case {args}: expected {expected!r}, got {got!r}"
    return True, "ok"


def cut_function(completion: str) -> str | None:
    """The completion up to and including the first line that is exactly '}' (None if there is none)."""
    lines = completion.split("\n")
    for i, line in enumerate(lines):
        if line.rstrip() == "}":
            return "\n".join(lines[:i + 1]) + "\n"
    return None


def generate_tasks(model, tok, device) -> dict:
    """GPU part of the task evaluation: greedy completion of every task prompt, cut after the first line '}'."""
    out = {}
    for t in TASKS:
        completion = tok.decode(greedy(model, tok.encode(t["prompt"]).ids, TASK_TOKENS, device))
        out[f"{t['set']}/{t['name']}"] = {"body": cut_function(completion), "completion": completion}
    return out


def score_tasks(generated: dict, ts_dir: Path) -> dict:
    """Sandbox part: compile and run every generated solution (run_task: bubblewrap); pass@1 overall and per set."""
    results = {}
    for t in TASKS:
        key = f"{t['set']}/{t['name']}"
        g = generated[key]
        ok, msg = (False, "no closing brace") if g["body"] is None else run_task(t["prompt"] + g["body"], t, ts_dir)
        results[key] = {"pass": ok, "message": msg, "completion": g["completion"][:400]}
    by_set = {s: [r["pass"] for k, r in results.items() if k.startswith(s + "/")] for s in ("write", "bugfix")}
    return {"pass_at_1": sum(r["pass"] for r in results.values()) / len(results),
            **{f"pass_at_1_{s}": sum(v) / len(v) for s, v in by_set.items()}, "tasks": results}


def task_eval(model, tok, ts_dir: Path, device) -> dict:
    return score_tasks(generate_tasks(model, tok, device), ts_dir)


def score_deferred(path: Path, ts_dir: Path):
    """Score the task completions a GPU host saved with --defer-tasks (on a host where bubblewrap works)."""
    sandbox_cmd(Path("/"), [])  # fail early without bubblewrap
    out = json.loads(path.read_text())
    for key, r in out["runs"].items():
        if "tasks_generated" in r and "tasks" not in r:
            r["tasks"] = score_tasks(r["tasks_generated"], ts_dir)
            r["tasks"]["scored_on"] = {k: v for k, v in env.collect().items() if k in ("os", "cpu", "python")}
            path.write_text(json.dumps(out, indent=2))
            print(f"{key:28s} pass@1 {r['tasks']['pass_at_1']:.2f} (write {r['tasks']['pass_at_1_write']:.2f}, "
                  f"bugfix {r['tasks']['pass_at_1_bugfix']:.2f})", flush=True)


# ----------------------------------------------------------------------------------------------- speed / work
def expert_rows(model, fn, x) -> float:
    for bank in expert_banks(model):
        bank.rows_computed = 0
    fn(x)
    return sum(bank.rows_computed for bank in expert_banks(model)) / x.numel()


@torch.no_grad()
def b2_latency(model, val: torch.Tensor, device) -> dict:
    x = next(eval_batches(val, 8, 8, 512, device))[0]
    is_amllm = isinstance(model, AdaptiveModularLM)

    def static(ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return static_forward(model, ids)

    def reference(ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(ids, hard_halt=True)[0] if is_amllm else model(ids)[0]

    r = timed(GraphRunner(static, x), x, warmup=10, repeats=50)
    r["expert_rows_per_token_executed"] = expert_rows(model, static, x)
    r["expert_rows_per_token_hard_routed"] = expert_rows(model, reference, x)
    return r


@torch.no_grad()
def decode_speed(model, val: torch.Tensor, device, repeats: int = 3) -> dict:
    out = {}
    for batch in (1, 32):
        prompt = next(eval_batches(val, batch, batch, 128, device))[0]
        gen = Generator(model, batch=batch, max_len=128 + 128, cache_dtype=torch.bfloat16)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = [greedy_batch(gen, prompt, 128, graph=True)[1] for _ in range(repeats + 1)][1:]
        out[f"b{batch}"] = {"decode_tokens_per_s_graph": batch * 128 / statistics.median(times),
                            "cache_mb": gen.cache_bytes() / 2**20,
                            "peak_memory_mb": torch.cuda.max_memory_allocated() / 2**20}
        del gen
    return out


PROVENANCE = ("visible_device", "gpus", "driver", "torch", "cuda", "git_commit", "source_sha256", "train_code_sha256")


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_runs(runs: list[Path], data_dir: Path) -> dict[str, str]:
    """Every checkpoint must belong to this corpus and tokenizer and to its own run: the split files and the tokenizer
    must hash to the manifest's values (measured, not trusted), the run's data split id must be the corpus', the
    checkpoint's model config and seed must equal its run's config.json and directory name. Returns the SHA-256 of
    every checkpoint, recorded with its results."""
    from .config import ModelConfig
    from .data import split_id
    manifest = json.loads((data_dir / "manifest.json").read_text())
    problems = []
    import hashlib
    if hashlib.sha256((data_dir / "tokenizer.json").read_bytes()).hexdigest() != manifest.get("tokenizer_sha256"):
        problems.append(f"{data_dir}/tokenizer.json does not match the manifest's tokenizer_sha256")
    for split, meta in manifest.get("splits", {}).items():
        f = data_dir / f"{split}.bin"
        if not f.exists() or file_sha256(f) != meta.get("sha256"):
            problems.append(f"{f} does not match the manifest's sha256")
    sid = split_id(data_dir)
    hashes = {}
    for run in runs:
        cfg = json.loads((run / "config.json").read_text())
        if cfg.get("data_split_id") != sid:
            problems.append(f"{run}: data split {cfg.get('data_split_id')} != corpus {sid}")
        ckpt = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=True)
        if ModelConfig.from_dict(ckpt["config"]) != ModelConfig.from_dict(cfg["model"]):
            problems.append(f"{run}: checkpoint model config differs from config.json")
        dir_seed = int(run.name.rsplit("_s", 1)[1]) if "_s" in run.name else None
        seeds = {ckpt.get("seed"), cfg.get("train", {}).get("seed"), dir_seed}
        if len(seeds) != 1 or None in seeds:
            problems.append(f"{run}: seeds differ (checkpoint {ckpt.get('seed')}, config {cfg.get('train', {}).get('seed')}, "
                            f"directory {dir_seed})")
        if not (run / "report.json").exists():
            problems.append(f"{run}: no report.json")
        hashes[str(run)] = file_sha256(run / "checkpoint.pt")
    if problems:
        raise SystemExit("refusing to evaluate: " + "; ".join(problems))
    return hashes


def main():
    from tokenizers import Tokenizer

    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data/code_corpus_v2")
    p.add_argument("--typescript", default=None)
    p.add_argument("--defer-tasks", action="store_true",
                   help="GPU host without a sandbox: save the task completions; score them with --score-tasks")
    p.add_argument("--score-tasks", action="store_true",
                   help="sandbox host: score the completions saved by --defer-tasks into --out")
    a = p.parse_args()
    ts_dir = typescript_dir(a.typescript)
    if a.score_tasks:
        return score_deferred(Path(a.out), ts_dir)
    if not a.runs:
        p.error("--runs is required")
    ckpt_sha = check_runs([Path(r) for r in a.runs], Path(a.data))
    if not a.defer_tasks:
        sandbox_cmd(Path("/"), [])  # fail early without bubblewrap
    device = torch.device("cuda")
    tok = Tokenizer.from_file(str(Path(a.data) / "tokenizer.json"))
    lut = token_bytes(tok)
    val, test = load_split(a.data, "val"), load_split(a.data, "test")
    prompts = test_prompts(test, tok)
    out_path = Path(a.out)
    out = json.loads(out_path.read_text()) if out_path.exists() else {"runs": {}}
    here = env.collect()
    prov = {k: here.get(k) for k in PROVENANCE}
    old = {json.dumps(r.get("provenance"), sort_keys=True) for r in out["runs"].values()}
    if out["runs"] and old != {json.dumps(prov, sort_keys=True)}:  # latencies of different GPUs/code never mix
        raise SystemExit(f"{out_path} holds measurements from another environment; use a new --out")
    out["env"] = here
    ref = score_continuations([tok.decode(p) for p, _ in prompts], [complete_lines(tok.decode(c)) for _, c in prompts],
                              ts_dir)
    out["protocol"] = {"prompts": len(prompts), "prompt_tokens": PROMPT_TOKENS, "new_tokens": NEW_TOKENS,
                       "task_tokens": TASK_TOKENS, "tasks": len(TASKS), "decoding": "greedy, bf16, KV cache",
                       "typescript": json.loads((ts_dir / "package.json").read_text())["version"],
                       "syntax_reference_real_continuation": ref}
    for run in a.runs:
        run_dir = Path(run)
        cfg, model = load_run(run_dir, device)
        if cfg.arch != "dense":
            set_dispatch(model, "triton")
        set_diagnostics(model, False)
        report = json.loads((run_dir / "report.json").read_text())
        train_cfg = json.loads((run_dir / "config.json").read_text())["train"]
        default = "fixed" if cfg.arch != "amllm" else "hard_surv_0.5"
        bpb = {}
        for split, data, loss in (("val", val, report["eval"][default]["val_loss"]),
                                  ("test", test, report["test_default_mode"]["val_loss"])):
            n_tok, n_bytes = window_bytes(data, lut, train_cfg["eval_windows"], cfg.max_seq_len)
            bpb[split] = {"loss_nat_per_token": loss, "bits_per_byte": loss * n_tok / n_bytes / math.log(2),
                          "tokens": n_tok, "bytes": n_bytes}
        r = {"arch": cfg.arch, "steps": report["steps"], "bpb": bpb, "provenance": prov,
             "checkpoint_sha256": ckpt_sha[str(run_dir)],
             "mean_hops": report["eval"][default]["mean_hops"],
             "flops_per_token_ideal": report["eval"][default]["flops_per_token_ideal"]}
        start = time.perf_counter()
        r["b2_latency_b8_t512"] = b2_latency(model, val, device)
        r["decode"] = decode_speed(model, val, device)
        r["syntax"] = syntax_eval(model, tok, prompts, ts_dir, device)
        if a.defer_tasks:
            r["tasks_generated"] = generate_tasks(model, tok, device)
        else:
            r["tasks"] = task_eval(model, tok, ts_dir, device)
        r["eval_seconds"] = time.perf_counter() - start
        key = f"{run_dir.parent.name}/{run_dir.name}"
        out["runs"][key] = r
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        s, l = r["syntax"], r["b2_latency_b8_t512"]
        p1 = f"{r['tasks']['pass_at_1']:.2f}" if "tasks" in r else "deferred"
        print(f"{key:28s} val {bpb['val']['loss_nat_per_token']:.4f} nat/tok = {bpb['val']['bits_per_byte']:.4f} bpb, "
              f"hops {r['mean_hops']:.2f}, B2 {l['median_ms']:.2f} ms, rows {l['expert_rows_per_token_executed']:.2f}"
              f"/{l['expert_rows_per_token_hard_routed']:.2f}, decode b1 {r['decode']['b1']['decode_tokens_per_s_graph']:,.0f} tok/s, "
              f"syntax {s['valid_continuation_share']:.3f} (ref {ref['valid_continuation_share']:.3f}), "
              f"pass@1 {p1}", flush=True)


if __name__ == "__main__":
    main()
