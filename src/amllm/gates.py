"""Evaluate pre-registered phase 2 gates from raw run reports (results/phase2/preregistration.md).

    python -m amllm.gates f1a     # halting calibration: choose lambda
    python -m amllm.gates f1      # F1 quality/compute gate -> results/phase2/gates.json
    python -m amllm.gates f1b     # the same gate for the straight-through weighting (F1b)
    python -m amllm.gates f2 | f3 | f4 | f5

The rules are implemented exactly as written in the pre-registration; nothing here is tuned after the runs.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PHASE2 = Path("results/phase2")


def default_mode(report: dict) -> str:
    return "hard_surv_0.5" if "hard_surv_0.5" in report["eval"] else "fixed"


PREREG_SEEDS = (42, 43, 44, 45, 46)


class GateError(Exception):
    """The stored runs do not allow a valid gate decision (incomplete seeds, mixed code or data)."""


def load(exp: str) -> dict[str, dict[int, dict]]:
    runs: dict[str, dict[int, dict]] = {}
    for rep in sorted((PHASE2 / exp).glob("*_s*/report.json")):
        preset, seed = rep.parent.name.rsplit("_s", 1)
        report = json.loads(rep.read_text())
        config = rep.parent / "config.json"
        report["_config"] = json.loads(config.read_text()) if config.exists() else {}
        runs.setdefault(preset, {})[int(seed)] = report
    return runs


def check_stage(runs: dict[str, dict[int, dict]], models: list[str], seeds=PREREG_SEEDS) -> None:
    """Every model needs exactly the pre-registered seed set, and all runs of the stage must share one
    training-code hash and one data split; otherwise no gate decision is made."""
    problems = []
    for m in models:
        have = tuple(sorted(runs.get(m, {})))
        if have != tuple(seeds):
            problems.append(f"{m}: seeds {list(have)} instead of {list(seeds)}")
    configs = [r["_config"] for m in models for r in runs.get(m, {}).values()]
    codes = {c.get("train_code_sha256") for c in configs}
    splits = {c.get("data_split_id") for c in configs}
    if None in codes or len(codes) != 1:
        problems.append(f"training-code hashes not identical/recorded: {sorted(map(str, codes))}")
    if None in splits or len(splits) != 1:
        problems.append(f"data split ids not identical/recorded: {sorted(map(str, splits))}")
    if problems:
        raise GateError("; ".join(problems))


def default_eval(report: dict) -> dict:
    return report["eval"][default_mode(report)]


def f1a() -> dict:
    runs = load("F1a")
    ref = runs["amllm_ua_fs_fixed"][42]["eval"]["fixed"]["val_loss"]
    limit = 1.02 * ref
    rows = []
    for tag, lam in (("002", 0.002), ("01", 0.01), ("03", 0.03), ("10", 0.1)):
        e = runs[f"amllm_ua_fs_p{tag}"][42]["eval"]["hard_surv_0.5"]
        rows.append({"lambda": lam, "val_loss": e["val_loss"], "mean_hops": e["mean_hops"],
                     "ideal_mflops": e["flops_per_token_ideal"] / 1e6, "qualifies": e["val_loss"] <= limit})
    qualifying = [r for r in rows if r["qualifies"]]
    chosen = min(qualifying, key=lambda r: r["mean_hops"])["lambda"] if qualifying else 0.002
    return {"rule": "fewest hard_surv_0.5 hops with val_loss <= 1.02 x amllm_ua_fs_fixed (seed 42); else 0.002",
            "reference_val_loss": ref, "limit": limit, "candidates": rows, "chosen_lambda": chosen}


STAGES = {
    # stage: (experiment folder, top-2 FLOP reference, {variant: role})
    "f1": ("F1", "amllm_ua_fs", {"amllm_ua_mk1_fs": "primary", "amllm_ua_ek1_fs": "secondary",
                                 "amllm_ua_both1_fs": "exploratory"}),
    "f1b": ("F1b", "amllm_ua_st", {"amllm_ua_mk1_st": "primary", "amllm_ua_ek1_st": "secondary"}),
}


def classify(primary_quality: bool, primary_compute: bool, any_quality: bool, any_compute: bool) -> str:
    """Complete verdict table of the pre-registration (amendment of 2026-10-21, before the F1b evaluation)."""
    if primary_quality and primary_compute:
        return "full success"
    if any_quality and any_compute:
        return "partial success"
    if any_compute:
        return "compute only: quality condition not met by any variant (decision rule 1: no kernel work)"
    if any_quality:
        return "quality only: no variant reaches the compute saving (F2 not started)"
    return "failure"


def quality_compute_gate(stage: str) -> dict:
    """F1-type gate (pre-registration): quality <= 1.02 x both matched MoE controls on >= 4/5 paired seeds,
    and ideal FLOPs >= 25% below the top-2 reference."""
    exp, reference, variants = STAGES[stage]
    runs = load(exp)
    primary_name = next(v for v, r in variants.items() if r == "primary")
    check_stage(runs, [*variants, reference, "moe_e8h128", "moe_e4"])
    seeds = list(PREREG_SEEDS)

    def loss(p, s):
        return default_eval(runs[p][s])["val_loss"]

    def ideal(p):  # always over the pre-registered seed set
        return statistics.fmean(default_eval(runs[p][s])["flops_per_token_ideal"] for s in seeds)

    def quality(candidate):
        out = {}
        for control in ("moe_e8h128", "moe_e4"):
            ok = [s for s in seeds if loss(candidate, s) <= 1.02 * loss(control, s)]
            out[control] = {"seeds_within_2pct": len(ok), "paired_seeds": len(seeds),
                            "per_seed_ratio": {s: loss(candidate, s) / loss(control, s) for s in seeds},
                            "met": len(ok) >= 4}
        out["met"] = all(v["met"] for k, v in out.items() if k != "met")
        return out

    def compute(candidate):
        saving = 1 - ideal(candidate) / ideal(reference)
        return {"ideal_mflops": ideal(candidate) / 1e6, "reference": reference,
                "reference_ideal_mflops": ideal(reference) / 1e6, "saving": saving, "met": saving >= 0.25}

    per_variant = {v: {"role": role, "quality": quality(v), "compute": compute(v)} for v, role in variants.items()}
    primary = per_variant[primary_name]
    any_quality = any(r["quality"]["met"] for r in per_variant.values())
    any_compute = any(r["compute"]["met"] for r in per_variant.values())
    verdict = classify(primary["quality"]["met"], primary["compute"]["met"], any_quality, any_compute)
    summary = {p: {"seeds": sorted(runs[p]),
                   "val_loss_mean": statistics.fmean(loss(p, s) for s in runs[p]),
                   "val_loss_std": statistics.stdev([loss(p, s) for s in runs[p]]) if len(runs[p]) > 1 else 0.0,
                   "val_loss_per_seed": {s: loss(p, s) for s in sorted(runs[p])},
                   "train_code_sha256": next(iter(runs[p].values()))["_config"].get("train_code_sha256"),
                   "test_loss_mean": (statistics.fmean(runs[p][s]["test_default_mode"]["val_loss"] for s in runs[p])
                                      if all(runs[p][s].get("test_default_mode") for s in runs[p]) else None),
                   "mean_hops": statistics.fmean(default_eval(runs[p][s])["mean_hops"] for s in runs[p]),
                   "ideal_mflops": ideal(p) / 1e6,
                   "impl_mflops": statistics.fmean(default_eval(runs[p][s])["flops_per_token"] for s in runs[p]) / 1e6,
                   "params_total": next(iter(runs[p].values()))["parameters"]["total"]}
               for p in sorted(runs)}
    return {"verdict": verdict, "variants": per_variant, "models": summary}


def f2() -> dict:
    """F2-2: largest lambda in {0.01, 0.02, 0.03} meeting the F1 quality condition (controls from F1b, same training
    code); F2 gate: bucketed hard execution >= 15% faster than full depth (bmm, diagnostics off, batch 32 x 256,
    median over the 5 seeds' medians), from results/phase2/F2/f2bench.json."""
    f1b, f2runs = load("F1b"), load("F2")
    candidates = {0.01: f1b["amllm_ua_mk1_st"], 0.02: f2runs.get("amllm_ua_mk1_st_p02", {}),
                  0.03: f2runs.get("amllm_ua_mk1_st_p03", {})}
    controls = {c: f1b[c] for c in ("moe_e8h128", "moe_e4")}
    combined = {**{f"lambda_{k}": v for k, v in candidates.items()}, **controls}
    check_stage(combined, list(combined))
    seeds = list(PREREG_SEEDS)
    rows = {}
    for lam, runs in candidates.items():
        q = {}
        for c, cr in controls.items():
            ratios = {s: default_eval(runs[s])["val_loss"] / default_eval(cr[s])["val_loss"] for s in seeds}
            q[c] = {"per_seed_ratio": ratios, "seeds_within_2pct": sum(r <= 1.02 for r in ratios.values())}
        rows[lam] = {"quality": q, "quality_met": all(v["seeds_within_2pct"] >= 4 for v in q.values()),
                     "val_loss_mean": statistics.fmean(default_eval(runs[s])["val_loss"] for s in seeds),
                     "mean_hops": statistics.fmean(default_eval(runs[s])["mean_hops"] for s in seeds),
                     "ideal_mflops": statistics.fmean(default_eval(runs[s])["flops_per_token_ideal"] for s in seeds) / 1e6}
    passing = [lam for lam, r in rows.items() if r["quality_met"]]
    chosen = max(passing) if passing else None
    result = {"operating_points": rows, "chosen_lambda": chosen}
    bench_file = PHASE2 / "F2" / "f2bench.json"
    if chosen is not None and bench_file.exists():
        bench = json.loads(bench_file.read_text())["runs"]
        exp, preset = ("F1b", "amllm_ua_mk1_st") if chosen == 0.01 else ("F2", f"amllm_ua_mk1_st_p{str(chosen)[2:]}")
        per_seed = {}
        for s in seeds:
            m = bench.get(f"{exp}/{preset}_s{s}", {}).get("b32/t256")
            if m is None:
                raise GateError(f"f2bench missing {exp}/{preset}_s{s}")
            per_seed[s] = {"full_depth_ms": m["full_depth"]["median_ms"], "bucketed_ms": m["bucketed_hard"]["median_ms"]}
        full = statistics.median(v["full_depth_ms"] for v in per_seed.values())
        fast = statistics.median(v["bucketed_ms"] for v in per_seed.values())
        saving = 1 - fast / full
        result["gate"] = {"per_seed": per_seed, "full_depth_median_ms": full, "bucketed_median_ms": fast,
                          "saving": saving, "met": saving >= 0.15}
    return result


def f3() -> dict:
    """F3 gate: on the best backend where both run, AM-LLM (F2 operating point) median latency <= 1.10 x the matched
    MoE (batch 32 x 256, median over the 5 seeds' medians). Executed vs. routed expert rows are reported under both
    readings of the pre-registered FLOP condition (no padding / no extra work vs. the hard path)."""
    bench = json.loads((PHASE2 / "F3" / "f3bench.json").read_text())["runs"]
    am, moe = "F2/amllm_ua_mk1_st_p02", "F1b/moe_e8h128"
    for name in (am, moe):
        have = sorted(int(k.rsplit("_s", 1)[1]) for k in bench if k.startswith(name + "_s"))
        if have != list(PREREG_SEEDS):
            raise GateError(f"{name}: seeds {have}")
    backends = list(bench[f"{am}_s42"]["backends"])

    def med(name, backend, load="b32/t256"):
        return statistics.median(bench[f"{name}_s{s}"]["backends"][backend]["loads"][load]["median_ms"] for s in PREREG_SEEDS)

    def rows(name, backend):
        return statistics.fmean(bench[f"{name}_s{s}"]["backends"][backend]["expert_rows_per_token"] for s in PREREG_SEEDS)

    per_backend = {b: {"amllm_ms": med(am, b), "moe_ms": med(moe, b), "ratio": med(am, b) / med(moe, b),
                       "amllm_b1_ms": med(am, b, "b1/t256"), "moe_b1_ms": med(moe, b, "b1/t256"),
                       "amllm_t64_ms": med(am, b, "b32/t64"), "moe_t64_ms": med(moe, b, "b32/t64"),
                       "amllm_expert_rows_per_token": rows(am, b), "moe_expert_rows_per_token": rows(moe, b)}
                   for b in backends}
    best = min(backends, key=lambda b: per_backend[b]["amllm_ms"])
    hard_routed_rows = rows(am, "B1_triton")  # triton + reference hard path: routed rows only
    best_rows = per_backend[best]["amllm_expert_rows_per_token"]
    latency_met = per_backend[best]["ratio"] <= 1.10
    strict_work_met = best_rows <= hard_routed_rows * 1.0001
    # the pre-registration requires both conditions (audit 2026-10-21: latency alone is "partially met")
    verdict = ("met" if latency_met and strict_work_met else "partially met (latency only)" if latency_met
               else "partially met (work only)" if strict_work_met else "not met")
    return {"verdict": verdict, "best_backend": best, "per_backend": per_backend,
            "latency_condition_met": latency_met,
            "flop_condition": {"no_padding_reading": "met" if "triton" in best else "not met",
                               "strict_reading_vs_hard_path": {"executed_rows": best_rows, "hard_routed_rows": hard_routed_rows,
                                                               "extra": best_rows / hard_routed_rows - 1,
                                                               "met": best_rows <= hard_routed_rows * 1.0001}},
            "val_loss_by_backend": {b: statistics.fmean(bench[f"{am}_s{s}"]["backends"][b]["val_loss"] for s in PREREG_SEEDS)
                                    for b in backends}}


def f4() -> dict:
    """F4 gate (correctness): the KV-cache tests (tests/test_generate.py, CUDA graph-decode test) pass, and on real
    checkpoints the bf16 greedy continuation agrees with the full prefix forward on >= 97% of predictions. Tokens/s
    are reported without a target."""
    bench = json.loads((PHASE2 / "F4" / "f4bench.json").read_text())["runs"]
    models = {"amllm": "F2/amllm_ua_mk1_st_p02", "moe_e8h128": "F1b/moe_e8h128", "moe_e4": "F1b/moe_e4", "dense": "F1/dense"}
    out = {"agreement_threshold": 0.97, "models": {}}
    for label, name in models.items():
        runs = [bench[f"{name}_s{s}"] for s in PREREG_SEEDS if f"{name}_s{s}" in bench]
        if len(runs) != len(PREREG_SEEDS):
            raise GateError(f"{name}: {len(runs)} of {len(PREREG_SEEDS)} seeds measured")
        agree = [r["greedy_agreement_vs_full_forward"] for r in runs]

        def med(batch, key):
            return statistics.median(r[f"b{batch}"][key] for r in runs)

        out["models"][label] = {"agreement_min": min(agree), "agreement_mean": statistics.fmean(agree),
                                "agreement_met": min(agree) >= 0.97,
                                **{f"b{b}_{k}": med(b, k) for b in (1, 32)
                                   for k in ("prefill_tokens_per_s", "decode_tokens_per_s_eager", "decode_tokens_per_s_graph",
                                             "decode_ms_per_step_graph")},
                                "cache_mb_batch32": runs[0]["cache_mb_batch32"]}
    out["met"] = all(m["agreement_met"] for m in out["models"].values())
    return out


F5_SEEDS = (42, 43, 44)
F5_MODELS = ("amllm_f5", "moe_f5", "dense_f5")


def check_long_runs(short: dict, long: dict, models, seed: int = 42) -> None:
    """The 2x stability runs must be the 1x runs with twice the steps: same training code, data split and model
    configuration; otherwise no stability statement is made."""
    problems = []
    for m in models:
        a, b = short[m][seed], long[m][seed]
        ca, cb = a["_config"], b["_config"]
        for key in ("train_code_sha256", "data_split_id", "model"):
            if ca.get(key) != cb.get(key):
                problems.append(f"{m}: {key} differs between the 1x and 2x run")
        if b.get("steps") != 2 * a.get("steps", -1):
            problems.append(f"{m}: 2x run has {b.get('steps')} steps, 1x has {a.get('steps')}")
    if problems:
        raise GateError("; ".join(problems))


def f5() -> dict:
    """F5 gate (preregistration F5, addenda and the audit amendment of 2026-10-21, all fixed before any F5 checkpoint
    was evaluated).
    Q (required): AM-LLM default-mode validation loss <= 1.02 x moe_f5 on >= 2 of 3 paired seeds.
    S: B2 latency (batch 8 x 512, median over seeds) <= 1.10 x moe_f5 AND executed expert rows/token on that path
       <= moe_f5's (seed mean); AM-LLM's executed vs. hard-routed rows are reported (extra work of static execution).
    C: syntax measure (valid continuation share) >= moe_f5 - 5 points (seed mean) AND pass@1 >= moe_f5's AND pass@1 is
       informative (> 0 for AM-LLM or moe_f5; Dense does not count); otherwise C is "not informative" = not met.
    met = Q and (S or C); partial = Q only; not met = not Q. The 2x run (seed 42) marks the verdict unstable if the
    loss ratio crosses 1.0 (the ranking reverses) or crosses the 1.02 threshold."""
    runs = load("F5")
    check_stage(runs, list(F5_MODELS), F5_SEEDS)
    ev = json.loads((PHASE2 / "F5" / "f5eval.json").read_text())["runs"]

    def e(model, seed):
        key = f"F5/{model}_s{seed}"
        if key not in ev:
            raise GateError(f"{key} not evaluated")
        return ev[key]

    def mean(f, model):
        return statistics.fmean(f(e(model, s)) for s in F5_SEEDS)

    loss = {m: {s: default_eval(runs[m][s])["val_loss"] for s in F5_SEEDS} for m in F5_MODELS}
    ratios = {s: loss["amllm_f5"][s] / loss["moe_f5"][s] for s in F5_SEEDS}
    q = sum(r <= 1.02 for r in ratios.values()) >= 2
    lat = {m: statistics.median(e(m, s)["b2_latency_b8_t512"]["median_ms"] for s in F5_SEEDS) for m in F5_MODELS}
    rows = {m: mean(lambda r: r["b2_latency_b8_t512"]["expert_rows_per_token_executed"], m) for m in F5_MODELS}
    am_hard_rows = mean(lambda r: r["b2_latency_b8_t512"]["expert_rows_per_token_hard_routed"], "amllm_f5")
    s_lat = lat["amllm_f5"] <= 1.10 * lat["moe_f5"]
    s_work = rows["amllm_f5"] <= rows["moe_f5"]
    syn = {m: mean(lambda r: r["syntax"]["valid_continuation_share"], m) for m in F5_MODELS}
    p1 = {m: mean(lambda r: r["tasks"]["pass_at_1"], m) for m in F5_MODELS}
    # informative only if AM-LLM or the MoE solves something: Dense's successes do not make a 0 vs. 0 comparison
    # informative (second audit, 2026-10-21)
    informative = p1["amllm_f5"] > 0 or p1["moe_f5"] > 0
    c_met = informative and syn["amllm_f5"] >= syn["moe_f5"] - 0.05 and p1["amllm_f5"] >= p1["moe_f5"]
    verdict = "met" if q and ((s_lat and s_work) or c_met) else "partial" if q else "not met"
    long = load("F5_2x")
    stability = None
    if all(42 in long.get(m, {}) for m in ("amllm_f5", "moe_f5")):
        check_long_runs(runs, long, ("amllm_f5", "moe_f5"))
        r1 = ratios[42]
        r2 = default_eval(long["amllm_f5"][42])["val_loss"] / default_eval(long["moe_f5"][42])["val_loss"]
        stability = {"ratio_1x_seed42": r1, "ratio_2x_seed42": r2,
                     "ranking_reversed": (r1 < 1.0) != (r2 < 1.0),
                     "threshold_crossed": (r1 <= 1.02) != (r2 <= 1.02)}
        if stability["ranking_reversed"] or stability["threshold_crossed"]:
            verdict += " (unstable)"
    else:
        verdict += " (2x runs missing)"
    descriptive = {m: {"val_bits_per_byte": mean(lambda r: r["bpb"]["val"]["bits_per_byte"], m),
                       "test_bits_per_byte": mean(lambda r: r["bpb"]["test"]["bits_per_byte"], m),
                       "decode_b1_tokens_per_s": statistics.median(e(m, s)["decode"]["b1"]["decode_tokens_per_s_graph"]
                                                                   for s in F5_SEEDS),
                       "decode_b32_tokens_per_s": statistics.median(e(m, s)["decode"]["b32"]["decode_tokens_per_s_graph"]
                                                                    for s in F5_SEEDS),
                       "peak_memory_mb_b32": max(e(m, s)["decode"]["b32"]["peak_memory_mb"] for s in F5_SEEDS),
                       "mean_hops": mean(lambda r: r["mean_hops"], m)} for m in F5_MODELS}
    return {"verdict": verdict,
            "Q": {"met": q, "val_loss": loss, "ratio_amllm_to_moe": ratios},
            "S": {"met": s_lat and s_work, "latency_met": s_lat, "work_met": s_work, "b2_median_ms": lat,
                  "latency_ratio": lat["amllm_f5"] / lat["moe_f5"], "executed_expert_rows_per_token": rows,
                  "amllm_hard_routed_rows_per_token": am_hard_rows,
                  "amllm_static_extra_work": rows["amllm_f5"] / am_hard_rows - 1},
            "C": {"met": c_met, "informative": informative, "syntax_valid_continuation_share": syn, "pass_at_1": p1},
            "stability_2x": stability, "descriptive": descriptive}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["f1a", *STAGES, "f2", "f3", "f4", "f5"])
    p.add_argument("--out", type=Path, default=PHASE2 / "gates.json",
                   help="gate results file (default: the versioned results/phase2/gates.json; automated runs write elsewhere)")
    a = p.parse_args()
    try:
        result = (f1a() if a.stage == "f1a" else f2() if a.stage == "f2" else f3() if a.stage == "f3"
                  else f4() if a.stage == "f4" else f5() if a.stage == "f5" else quality_compute_gate(a.stage))
    except GateError as e:
        result = {"verdict": "invalid: no gate decision", "reason": str(e)}
    gates_file = a.out
    gates = json.loads(gates_file.read_text()) if gates_file.exists() else {}
    gates[a.stage.upper()] = result
    gates_file.write_text(json.dumps(gates, indent=2))
    print(json.dumps(result, indent=2)[:4000])
    if result["verdict"].startswith("invalid"):  # no decision is a failure for callers (exit status 3)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
