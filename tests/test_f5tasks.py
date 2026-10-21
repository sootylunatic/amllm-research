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
    assert not has_code_line("\n   \n// comment\n/* block */\n * doc\n")
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
