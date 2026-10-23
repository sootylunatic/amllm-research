"""F5 coding harness: reference solutions pass, wrong or adversarial ones fail, generated code is sandboxed, and the
syntax measure gives no credit to an empty continuation (needs node + typescript + bubblewrap)."""
import os
import shutil
from pathlib import Path

import pytest

from amllm.f5eval import complete_lines, cut_function, has_code_line, run_task, score_continuations
from amllm.f5tasks import TASKS

TS = Path(os.environ.get("AMLLM_TYPESCRIPT", "")).expanduser()
needs_ts = pytest.mark.skipif(not (TS / "lib" / "tsc.js").is_file() or not shutil.which("node") or not shutil.which("bwrap"),
                              reason="set AMLLM_TYPESCRIPT to a typescript package; needs node and bwrap")

REFERENCE = {
    "clamp": "return Math.min(max, Math.max(min, value));",
    "sum": "return values.reduce((a, b) => a + b, 0);",
    "isPalindrome": 'return str === str.split("").reverse().join("");',
    "reverseString": 'return str.split("").reverse().join("");',
    "factorial": "return n <= 1 ? 1 : n * factorial(n - 1);",
    "fibonacci": "let a = 0, b = 1;\n  for (let i = 0; i < n; i++) [a, b] = [b, a + b];\n  return a;",
    "max": "return Math.max(...values);",
    "countVowels": "return (str.match(/[aeiou]/gi) || []).length;",
    "capitalize": "return str.charAt(0).toUpperCase() + str.slice(1);",
    "unique": "return Array.from(new Set(values));",
    "chunk": "const out: T[][] = [];\n  for (let i = 0; i < values.length; i += size) out.push(values.slice(i, i + size));\n  return out;",
    "flatten": "return ([] as T[]).concat(...values);",
    "range": "const out: number[] = [];\n  for (let i = start; i < end; i++) out.push(i);\n  return out;",
    "gcd": "return b === 0 ? a : gcd(b, a % b);",
    "isPrime": "if (n < 2) return false;\n  for (let i = 2; i * i <= n; i++) if (n % i === 0) return false;\n  return true;",
    "sumDigits": 'return String(n).split("").reduce((a, c) => a + Number(c), 0);',
    "average": "return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;",
    "last": "return values[values.length - 1];",
    "repeat": "let s = \"\";\n  for (let i = 0; i < n; i++) s += str;\n  return s;",
    "isEmpty": "return Object.keys(obj).length === 0;",
}
ids = lambda t: f"{t['set']}/{t['name']}"  # noqa: E731


def solve(t, body):
    return t["prompt"] + "  " + body + "\n}\n"


def test_task_sets():
    assert sum(t["set"] == "write" for t in TASKS) == 20 and sum(t["set"] == "bugfix" for t in TASKS) == 10
    assert all(t["name"] in REFERENCE for t in TASKS)
    for t in TASKS:  # expected outputs never reach the sandbox
        for _, expected in t["cases"]:
            if isinstance(expected, str) and len(expected) > 1:
                assert f'"{expected}"' not in t["runner"].split("const inputs")[1].split("\n")[0] or \
                    any(expected in str(a) for a, _ in t["cases"])


def test_cut_function():
    assert cut_function("  return 1;\n}\n\nexport const x = 2;\n") == "  return 1;\n}\n"
    assert cut_function("  return 1;\n  }\n") is None


@needs_ts
@pytest.mark.parametrize("t", TASKS, ids=ids)
def test_reference_passes_wrong_fails(t):
    ok, msg = run_task(solve(t, REFERENCE[t["name"]]), t, TS)
    assert ok, msg
    assert not run_task(solve(t, "throw new Error('no');"), t, TS)[0]


@needs_ts
@pytest.mark.parametrize("t", [t for t in TASKS if t["set"] == "bugfix"], ids=ids)
def test_buggy_version_fails(t):
    buggy = t["prompt"].split("{\n", 1)[1].split("\n}\n", 1)[0]
    assert not run_task(t["prompt"] + buggy + "\n}\n", t, TS)[0]


ATTACKS = {
    "early exit": "(globalThis as any).process.exit(0);\n  return 0 as any;",
    # fakes "all ok" without knowing the expected outputs (knowing them would amount to solving the task)
    "patched stringify": "JSON.stringify = () => '[{\"ok\":true},{\"ok\":true},{\"ok\":true}]';\n  return 0;",
    "patched console": "(globalThis as any).console.log = () => undefined;\n  return 0;",
    "read host file": "const fs = (globalThis as any).require ? null : null;\n"
                      "  return Number(eval(\"require('fs').readFileSync('/etc/hostname', 'utf8').length\"));",
}


@needs_ts
@pytest.mark.parametrize("name", ATTACKS)
def test_attacks_fail(name):
    t = TASKS[0]  # clamp
    ok, msg = run_task(solve(t, ATTACKS[name]), t, TS)
    assert not ok, msg


@needs_ts
def test_sandbox_blocks_host_and_network(tmp_path):
    from amllm.f5eval import sandbox_cmd
    import subprocess
    (tmp_path / "a.js").write_text(
        "const r = {};\ntry { require('fs').readFileSync(process.argv[2]); r.home = 'READ'; } catch (e) { r.home = 'blocked'; }\n"
        "try { require('fs').writeFileSync('/work/x', '1'); r.write = 'ok'; } catch (e) { r.write = 'blocked'; }\n"
        "r.env = Object.keys(process.env).filter((k) => k !== 'PWD').length;\n"
        "require('net').connect(80, '1.1.1.1').on('error', (e) => { r.net = e.code; console.log(JSON.stringify(r)); })"
        ".on('connect', () => { r.net = 'CONNECTED'; console.log(JSON.stringify(r)); process.exit(); });\n")
    secret = Path.home() / ".bashrc"
    out = subprocess.run(sandbox_cmd(tmp_path, ["a.js", str(secret)]), capture_output=True, text=True, timeout=30,
                         env={**os.environ, "SECRET_TOKEN": "x"})
    assert '"home":"blocked"' in out.stdout and '"write":"blocked"' in out.stdout and '"env":0' in out.stdout
    assert "CONNECTED" not in out.stdout


@needs_ts
def test_type_error_and_timeout_fail():
    t = TASKS[0]
    assert run_task(solve(t, 'return "x";'), t, TS)[1].startswith("tsc")
    assert run_task(solve(t, "while (true) {}"), t, TS, timeout=2)[1] == "timeout"


def test_code_line_detection():
    assert not has_code_line("\n   \n// comment\n/* block */\n")
    # second audit: a multi-line block comment is not code
    assert not has_code_line("/*\nonly comment text\n*/\n")
    # a comment opened in the prompt continues into the continuation
    assert not has_code_line(" * more\n */\n", prompt="/**\n * doc\n")
    assert has_code_line(" */\nconst a = 1;\n", prompt="/**\n")
    assert has_code_line('const u = "http://x";\n') and has_code_line("/* x */ const a = 1;\n")
    assert has_code_line("// c\nconst x = 1;\n")
    assert complete_lines("a\nb") == "a\n" and complete_lines("ab") == ""


@needs_ts
def test_empty_continuation_scores_zero():
    prompts = ["const a = 1;\n", "function f() {\n  return 1;\n"]
    r = score_continuations(prompts, ["", ""], TS)
    assert r["n_valid_prompts"] == 2 and r["valid_continuation_share"] == 0.0
    r = score_continuations(prompts, ["// only a comment\n", "  return 2;\n"], TS)
    assert r["valid_continuation_share"] == 0.5
    r = score_continuations(prompts, ["let = ;\n", "  return 2;\n}\n"], TS)
    assert r["valid_continuation_share"] == 0.5
    r = score_continuations(prompts, ["/*\nonly comment text\n*/\n", "/* a */\n// b\n"], TS)
    assert r["valid_continuation_share"] == 0.0


def test_deferred_scoring_equals_direct(monkeypatch):
    """Split evaluation (generate on the GPU host, score on the sandbox host) must give task_eval's result."""
    import amllm.f5eval as fe
    completions = {f"{t['set']}/{t['name']}": ("  return 0;\n}\nextra\n" if i % 2 else "  no brace\n")
                   for i, t in enumerate(fe.TASKS)}
    order = iter([completions[f"{t['set']}/{t['name']}"] for t in fe.TASKS] * 2)
    monkeypatch.setattr(fe, "greedy", lambda model, ids, steps, device: [0])
    class Tok:
        def encode(self, text):
            return type("E", (), {"ids": [0]})()
        def decode(self, ids):
            return next(order)
    scored = []
    monkeypatch.setattr(fe, "run_task", lambda sol, task, ts: (scored.append(sol) or (sol.endswith("}\n"), "x")))
    direct = fe.task_eval(None, Tok(), TS, None)
    deferred = fe.score_tasks(fe.generate_tasks(None, Tok(), None), TS)
    assert direct == deferred
    assert all(r["message"] == "no closing brace" for k, r in direct["tasks"].items()
               if completions[k] == "  no brace\n")


def test_tsc_timeout_fails_only_that_task(monkeypatch, tmp_path):
    import subprocess as sp
    import amllm.f5eval as fe
    def fake_run(cmd, **kw):
        if "tsc.js" in " ".join(map(str, cmd)):
            raise sp.TimeoutExpired(cmd, 120)
        raise AssertionError("the sandbox must not run after a failed compile")
    monkeypatch.setattr(fe.subprocess, "run", fake_run)
    assert fe.run_task("x", fe.TASKS[0], tmp_path) == (False, "tsc timeout")


def test_evaluation_refuses_a_foreign_corpus_or_mixed_environment(tmp_path, monkeypatch):
    import json as js
    import torch as th
    import amllm.f5eval as fe
    from amllm.config import ModelConfig
    data = tmp_path / "corpus"
    data.mkdir()
    (data / "tokenizer.json").write_text("{}")
    import hashlib
    for name in ("train", "val", "test"):
        (data / f"{name}.bin").write_bytes(name.encode())
    (data / "manifest.json").write_text(js.dumps({"tokenizer_sha256": hashlib.sha256(b"{}").hexdigest(),
                                                  "splits": {n: {"sha256": hashlib.sha256(n.encode()).hexdigest()}
                                                             for n in ("train", "val", "test")}}))
    run = tmp_path / "run_s42"
    run.mkdir()
    cfg = ModelConfig(arch="dense", d_model=16, n_heads=2, n_layers=1, ffn_hidden=16, max_seq_len=8)
    (run / "config.json").write_text(js.dumps({"model": cfg.to_dict(), "data_split_id": "other", "train": {"seed": 42}}))
    (run / "report.json").write_text("{}")
    th.save({"config": cfg.to_dict(), "state_dict": {}, "seed": 42}, run / "checkpoint.pt")
    with pytest.raises(SystemExit, match="data split"):
        fe.check_runs([run], data)
    from amllm.data import split_id
    cfgd = js.loads((run / "config.json").read_text())
    cfgd["data_split_id"] = split_id(data)
    (run / "config.json").write_text(js.dumps(cfgd))
    hashes = fe.check_runs([run], data)                               # consistent: accepted, checkpoint hashed
    assert len(hashes[str(run)]) == 64
    th.save({"config": cfg.to_dict(), "state_dict": {}, "seed": 43}, run / "checkpoint.pt")
    with pytest.raises(SystemExit, match="seeds differ"):            # seed-43 weights in the seed-42 run
        fe.check_runs([run], data)
    th.save({"config": cfg.to_dict(), "state_dict": {}, "seed": 42}, run / "checkpoint.pt")
    (data / "val.bin").write_bytes(b"x")                              # a split file changed, manifest unchanged
    js_m = js.loads((data / "manifest.json").read_text())
    js_m["splits"]["val"]["sha256"] = "0" * 64
    (data / "manifest.json").write_text(js.dumps(js_m))
    with pytest.raises(SystemExit, match="val.bin"):
        fe.check_runs([run], data)
    (data / "tokenizer.json").write_text('{"changed": 1}')
    with pytest.raises(SystemExit, match="tokenizer"):
        fe.check_runs([run], data)
