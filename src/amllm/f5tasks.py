"""F5 coding tasks: 20 function-writing tasks and 10 bug-fix tasks (TypeScript).

The model continues the prompt; the completion is cut after the first line that is exactly "}" (the function's closing
brace at column 0). The sandbox only receives the solution and the *inputs* of the cases: a generated runner calls the
function on every input and prints the results as JSON. The expected outputs never enter the sandbox; they are compared
in Python. An early `process.exit(0)`, a patched `JSON.stringify` or missing output therefore cannot pass.
"""
from __future__ import annotations

import json

UNDEFINED = "__undefined__"  # how the runner reports an `undefined` result


def _prompt(name: str, doc: str, signature: str) -> str:
    return f"/**\n * {doc}\n */\nexport function {signature} {{\n"


def _bugfix_prompt(name: str, doc: str, signature: str, buggy_body: str) -> str:
    buggy = signature.replace(f"{name}", f"{name}Buggy", 1)
    return (f"// This implementation has a bug.\nfunction {buggy} {{\n{buggy_body.rstrip()}\n}}\n\n"
            f"/**\n * {doc}\n * Fixed version of {name}Buggy above.\n */\nexport function {signature} {{\n")


# (name, doc, signature, [(args, expected), ...])
WRITE = [
    ("clamp", "Clamps a value to the inclusive range [min, max].", "clamp(value: number, min: number, max: number): number",
     [([5, 0, 10], 5), ([-1, 0, 10], 0), ([11, 0, 10], 10)]),
    ("sum", "Returns the sum of all numbers in the array.", "sum(values: number[]): number",
     [([[]], 0), ([[1, 2, 3]], 6), ([[-1, 1]], 0)]),
    ("isPalindrome", "Returns true if the string reads the same forwards and backwards.", "isPalindrome(str: string): boolean",
     [(["abba"], True), (["abc"], False), ([""], True)]),
    ("reverseString", "Returns the characters of the string in reverse order.", "reverseString(str: string): string",
     [(["abc"], "cba"), ([""], "")]),
    ("factorial", "Returns n! for a non-negative integer n.", "factorial(n: number): number",
     [([0], 1), ([5], 120), ([1], 1)]),
    ("fibonacci", "Returns the n-th Fibonacci number (fibonacci(0) === 0, fibonacci(1) === 1).", "fibonacci(n: number): number",
     [([0], 0), ([1], 1), ([10], 55)]),
    ("max", "Returns the largest number in a non-empty array.", "max(values: number[]): number",
     [([[1, 5, 3]], 5), ([[-3, -1, -2]], -1)]),
    ("countVowels", "Counts the vowels (a, e, i, o, u, case-insensitive) in the string.", "countVowels(str: string): number",
     [(["Hello World"], 3), (["xyz"], 0), (["AEIOU"], 5)]),
    ("capitalize", "Returns the string with its first character in upper case.", "capitalize(str: string): string",
     [(["hello"], "Hello"), ([""], ""), (["a b"], "A b")]),
    ("unique", "Returns the distinct values of the array in order of first occurrence.", "unique<T>(values: T[]): T[]",
     [([[1, 2, 1, 3, 2]], [1, 2, 3]), ([[]], [])]),
    ("chunk", "Splits the array into chunks of the given size; the last chunk may be shorter.",
     "chunk<T>(values: T[], size: number): T[][]", [([[1, 2, 3, 4, 5], 2], [[1, 2], [3, 4], [5]]), ([[], 3], [])]),
    ("flatten", "Flattens an array of arrays by one level.", "flatten<T>(values: T[][]): T[]",
     [([[[1, 2], [3], []]], [1, 2, 3]), ([[]], [])]),
    ("range", "Returns the integers from start (inclusive) to end (exclusive).", "range(start: number, end: number): number[]",
     [([0, 4], [0, 1, 2, 3]), ([3, 3], []), ([-2, 1], [-2, -1, 0])]),
    ("gcd", "Returns the greatest common divisor of two non-negative integers.", "gcd(a: number, b: number): number",
     [([12, 18], 6), ([7, 0], 7), ([17, 5], 1)]),
    ("isPrime", "Returns true if n is a prime number.", "isPrime(n: number): boolean",
     [([2], True), ([1], False), ([15], False), ([97], True)]),
    ("sumDigits", "Returns the sum of the decimal digits of a non-negative integer.", "sumDigits(n: number): number",
     [([0], 0), ([1234], 10)]),
    ("average", "Returns the arithmetic mean of the numbers, or 0 for an empty array.", "average(values: number[]): number",
     [([[]], 0), ([[1, 2, 3, 4]], 2.5)]),
    ("last", "Returns the last element of the array, or undefined if it is empty.", "last<T>(values: T[]): T | undefined",
     [([[1, 2, 3]], 3), ([[]], UNDEFINED)]),
    ("repeat", "Returns the string repeated n times.", "repeat(str: string, n: number): string",
     [(["ab", 3], "ababab"), (["x", 0], "")]),
    ("isEmpty", "Returns true if the object has no own enumerable keys.", "isEmpty(obj: object): boolean",
     [([{}], True), ([{"a": 1}], False)]),
]

# (name, doc, signature, buggy body, [(args, expected), ...])
BUGFIX = [
    ("sum", "Returns the sum of all numbers in the array.", "sum(values: number[]): number",
     "  let total = 1;\n  for (const v of values) total += v;\n  return total;", [([[]], 0), ([[1, 2, 3]], 6)]),
    ("max", "Returns the largest number in a non-empty array.", "max(values: number[]): number",
     "  let best = 0;\n  for (const v of values) if (v > best) best = v;\n  return best;", [([[1, 5, 3]], 5), ([[-3, -1, -2]], -1)]),
    ("range", "Returns the integers from start (inclusive) to end (exclusive).", "range(start: number, end: number): number[]",
     "  const out: number[] = [];\n  for (let i = start; i <= end; i++) out.push(i);\n  return out;",
     [([0, 4], [0, 1, 2, 3]), ([3, 3], [])]),
    ("isPrime", "Returns true if n is a prime number.", "isPrime(n: number): boolean",
     "  if (n < 1) return false;\n  for (let i = 2; i * i <= n; i++) if (n % i === 0) return false;\n  return true;",
     [([1], False), ([2], True), ([9], False), ([13], True)]),
    ("countVowels", "Counts the vowels (a, e, i, o, u, case-insensitive) in the string.", "countVowels(str: string): number",
     "  let n = 0;\n  for (const c of str) if (\"aeiou\".includes(c)) n++;\n  return n;", [(["Hello World"], 3), (["AEIOU"], 5)]),
    ("average", "Returns the arithmetic mean of the numbers, or 0 for an empty array.", "average(values: number[]): number",
     "  if (values.length === 0) return 0;\n  return values.reduce((a, b) => a + b, 0) / (values.length - 1);",
     [([[]], 0), ([[1, 2, 3, 4]], 2.5)]),
    ("last", "Returns the last element of the array, or undefined if it is empty.", "last<T>(values: T[]): T | undefined",
     "  return values[values.length];", [([[1, 2, 3]], 3), ([[]], UNDEFINED)]),
    ("reverseString", "Returns the characters of the string in reverse order.", "reverseString(str: string): string",
     "  let out = \"\";\n  for (let i = str.length - 1; i > 0; i--) out += str[i];\n  return out;", [(["abc"], "cba"), ([""], "")]),
    ("factorial", "Returns n! for a non-negative integer n.", "factorial(n: number): number",
     "  if (n <= 0) return 0;\n  return n * factorial(n - 1);", [([0], 1), ([5], 120)]),
    ("capitalize", "Returns the string with its first character in upper case.", "capitalize(str: string): string",
     "  return str.charAt(0).toUpperCase() + str.slice(1).toLowerCase();", [(["hello"], "Hello"), (["aBC"], "ABC")]),
]


def runner(name: str, cases: list) -> str:
    """TypeScript runner: calls `name` on every case input, prints one JSON line of {ok, v} results."""
    inputs = json.dumps([args for args, _ in cases])
    return (f'import {{ {name} }} from "./solution";\n'
            "declare const console: { log(message: string): void };\n"
            f"const inputs: unknown[][] = {inputs};\n"
            f"const fn = {name} as unknown as (...args: unknown[]) => unknown;\n"
            "const results = inputs.map((args) => {\n"
            "  try {\n"
            "    const v = fn(...args);\n"
            f'    return {{ ok: true, v: v === undefined ? "{UNDEFINED}" : v }};\n'
            "  } catch {\n    return { ok: false };\n  }\n});\n"
            'console.log("\\n" + JSON.stringify(results));\n')


TASKS: list[dict] = (
    [{"set": "write", "name": n, "prompt": _prompt(n, d, s), "cases": c} for n, d, s, c in WRITE]
    + [{"set": "bugfix", "name": n, "prompt": _bugfix_prompt(n, d, s, b), "cases": c} for n, d, s, b, c in BUGFIX]
)
for _t in TASKS:
    _t["runner"] = runner(_t["name"], _t["cases"])
