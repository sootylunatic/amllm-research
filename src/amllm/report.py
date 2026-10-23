"""Aggregate all run directories into summary.csv, architecture_comparison.json and figures.

    python -m amllm.report

Reads results/e*/<preset>_s<seed>/report.json, results/profile.json and results/router_analysis.json.
Figures are rendered in English (results/figures/en) and Hungarian (results/figures/hu), see ADR-0015.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = Path("results")
FIG = RESULTS / "figures"
LANGS = ("en", "hu")

# Fixed entity -> colour (reference categorical palette, light mode); never re-assigned by rank.
COLOR = {"dense": "#2a78d6", "moe": "#eb6834", "amllm": "#1baf7a", "ut": "#eda100", "amllm_fixed": "#e87ba4",
         "amllm_m16": "#008300", "dense_wide": "#6250d6", "dense_l2": "#2a78d6", "moe_l2": "#eb6834",
         "amllm_p03": "#1baf7a", "amllm_ua": "#e34948"}
HOLLOW = {"dense_l2", "moe_l2", "amllm_p03"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# manual direct-label offsets (points) where labels would otherwise collide
SCATTER_OFFSET = {"amllm": (-10, 4), "amllm_fixed": (8, 3), "amllm_m16": (8, -1), "moe": (8, -10), "amllm_ua": (8, 6),
                  "dense": (8, -11), "amllm_p03": (-10, 4), "moe_l2": (8, -10), "dense_l2": (8, 4)}
CURVE_OFFSET = {"amllm": 5, "amllm_fixed": -5, "moe": -6, "amllm_ua": 5}

LABEL = {
    "en": {"dense": "Dense (4 layers)", "moe": "MoE (4 layers, 8e top-2)", "amllm": "AM-LLM (adaptive, λ=0.01)",
           "ut": "UT (shared, no routing)", "amllm_fixed": "AM-LLM fixed 4 hops", "amllm_m16": "AM-LLM 16 modules",
           "dense_wide": "Wide Dense (FFN 2048)", "dense_l2": "Dense 2 layers", "moe_l2": "MoE 2 layers",
           "amllm_p03": "AM-LLM λ=0.03 (strong halting)", "amllm_ua": "AM-LLM per-hop attention (exploratory)"},
    "hu": {"dense": "Dense (4 réteg)", "moe": "MoE (4 réteg, 8e top-2)", "amllm": "AM-LLM (adaptív, λ=0,01)",
           "ut": "UT (megosztott, routing nélkül)", "amllm_fixed": "AM-LLM fix 4 hop", "amllm_m16": "AM-LLM 16 modul",
           "dense_wide": "Dense széles (FFN 2048)", "dense_l2": "Dense 2 réteg", "moe_l2": "MoE 2 réteg",
           "amllm_p03": "AM-LLM λ=0,03 (erős halting)", "amllm_ua": "AM-LLM hoponkénti attention (feltáró)"},
}
TEXT = {
    "en": {
        "curves_title": "Validation loss during training (mean of 3 seeds)", "steps": "training step (×8192 bytes)",
        "val_loss": "val loss (nat/byte)",
        "scatter_title": "Quality vs. estimated active compute (3 seeds: mean, min–max)",
        "mflops": "estimated MFLOP / token (ideal, forward)", "val_loss_lower": "val loss (lower = better)",
        "sweep": "AM-LLM λ=0.03 threshold sweep (seed 42)",
        "halt_title": "Dynamic depth: hops vs. quality (seed 42; ◇ = soft training mode)",
        "hops_axis": "mean executed hops / token", "hard_range": "hard (threshold 0.3–0.8)",
        "fixed_ref": "fixed 4-hop reference", "val": "val loss",
        "class_title": "Executed hops by class of the predicted byte (hard, threshold 0.5)",
        "class_y": "mean hops / token", "s42_p03": "λ=0.03 (seed 42)", "s42_main": "λ=0.01 main configuration (seed 42)",
        "mi_title": "Route–input dependence: I(selected unit ; byte class) / H(byte class), seed 42",
        "mi_y": "normalised MI", "untrained": "untrained control", "trained": "trained", "layer": "layer", "hop": "hop",
        "lat_title": "GPU forward latency, batch 32 × 256 bytes\n(diagnostics off)",
        "lat_x": "ms / full-sequence forward (NOT generated tokens/s); median over seeds",
        "median": "median", "p95": "p95 (worst seed)",
    },
    "hu": {
        "curves_title": "Validációs veszteség a tanítás során (3 seed átlaga)", "steps": "tanítási lépés (×8192 bájt)",
        "val_loss": "val loss (nat/bájt)",
        "scatter_title": "Minőség vs. becsült aktív számítás (3 seed: átlag, min–max)",
        "mflops": "becsült MFLOP / token (ideális, forward)", "val_loss_lower": "val loss (alacsonyabb = jobb)",
        "sweep": "AM-LLM λ=0,03 küszöb-sweep (seed 42)",
        "halt_title": "Dinamikus mélység: hop-szám vs. minőség (seed 42; ◇ = soft tanítási mód)",
        "hops_axis": "átlagos végrehajtott hop / token", "hard_range": "hard (küszöb 0,3–0,8)",
        "fixed_ref": "fix 4 hop referencia", "val": "val loss",
        "class_title": "Végrehajtott hop a megjósolandó bájt osztálya szerint (hard, küszöb 0,5)",
        "class_y": "átlagos hop / token", "s42_p03": "λ=0,03 (seed 42)", "s42_main": "λ=0,01 fő konfiguráció (seed 42)",
        "mi_title": "Routing–bemenet függés: I(választott egység ; bájtosztály) / H(bájtosztály), seed 42",
        "mi_y": "normált MI", "untrained": "tanítatlan kontroll", "trained": "tanított", "layer": "réteg", "hop": "hop",
        "lat_title": "GPU forward késleltetés, batch 32 × 256 bájt\n(diagnosztika nélkül)",
        "lat_x": "ms / teljes szekvenciás forward (NEM generált token/s); seedek mediánja",
        "median": "medián", "p95": "p95 (legrosszabb seed)",
    },
}
CLASS_HU = {"space/tab": "szóköz/tab", "newline": "újsor", "letter/_/$": "betű/_/$", "digit": "számjegy",
            "punct/operator": "írásjel/operátor", "other/non-ascii": "egyéb/nem-ASCII"}

# Runs whose numbers must not be read as what their name suggests (see ADR-0013).
VALIDITY = {
    ("e06_ablations", "amllm_mk1"): "invalid as a top-1 measurement: module router gets no LM gradient (ADR-0013)",
    ("e06_ablations", "amllm_ek1"): "invalid as a top-1 measurement: expert router gets no LM gradient (ADR-0013)",
}
CODE_VERSION = {"e09_audit_controls": "v0.2.1 (deterministic combine, ADR-0014)"}


def num(x: float, lang: str, digits: int = 1) -> str:
    s = f"{x:.{digits}f}"
    return s.replace(".", ",") if lang == "hu" else s


def default_mode(report: dict) -> str:
    ev = report["eval"]
    return "hard_surv_0.5" if "hard_surv_0.5" in ev else "fixed"


def load_experiment(name: str) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for rep in sorted((RESULTS / name).glob("*_s*/report.json")):
        preset = rep.parent.name.rsplit("_s", 1)[0]
        r = json.loads(rep.read_text())
        r["_dir"] = str(rep.parent)
        runs.setdefault(preset, []).append(r)
    return runs


def mstd(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs), "max": max(xs), "n": len(xs), "values": xs}


def write_summary(experiments: list[str]) -> list[dict]:
    rows = []
    for exp in experiments:
        for preset, reps in load_experiment(exp).items():
            for r in reps:
                for mode, e in r["eval"].items():
                    rows.append({"experiment": exp, "code_version": CODE_VERSION.get(exp, "v0.2.0"),
                                 "preset": preset, "seed": r["seed"], "arch": r["arch"],
                                 "eval_mode": mode, "is_default_mode": mode == default_mode(r),
                                 "validity_note": VALIDITY.get((exp, preset), ""),
                                 "params_total": r["parameters"]["total"],
                                 "params_non_embedding": r["parameters"]["non_embedding"],
                                 "val_loss": round(e["val_loss"], 5), "val_ppl": round(math.exp(e["val_loss"]), 4),
                                 "val_bpb": round(e["val_bpb"], 4), "mean_hops": round(e["mean_hops"], 3),
                                 "p95_hops": e["p95_hops"],
                                 "est_flops_per_token_impl": round(e["flops_per_token"]),
                                 "est_flops_per_token_ideal": round(e["flops_per_token_ideal"]),
                                 "eval_batch_ms_with_diagnostics": round(e["median_batch_ms"], 2),
                                 "expert_usage_entropy_norm": round(e["expert_usage_entropy"] / e["expert_usage_entropy_max"], 4)
                                 if e["expert_usage_entropy_max"] > 0 else None,
                                 "dead_experts": e["dead_experts"] if r["arch"] != "dense" else None,
                                 "max_expert_share": round(e.get("max_expert_share", 0), 4) if r["arch"] != "dense" else None,
                                 "test_loss_default_mode": (round(r["test_default_mode"]["val_loss"], 5)
                                                            if mode == default_mode(r) and r.get("test_default_mode") else None),
                                 "train_tokens": r["train_tokens"], "train_seconds": round(r["train_seconds"], 1),
                                 "train_tokens_per_s": round(r["train_tokens_per_s"]),
                                 "train_peak_vram_mb": r.get("train_peak_vram_mb")})
    with (RESULTS / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def profile_summary(profile: dict, exp: str, preset: str) -> dict:
    """Per execution path (dispatch/mode/batch/diagnostics): median over seeds of the per-seed medians,
    worst p95, and the matching VRAM / profiled FLOP / kernel / sync values — one coherent row per path."""
    runs = [r for k, r in profile.get("runs", {}).items() if k.startswith(f"{exp}/{preset}_s")]
    if not runs:
        return {}
    out = {"seeds_profiled": len(runs),
           "unique_active_params_per_token": statistics.fmean(r["unique_active_params_per_token"] for r in runs),
           "param_applications_per_token": statistics.fmean(r["param_applications_per_token"] for r in runs),
           "param_applications_per_token_ideal": statistics.fmean(r["param_applications_per_token_ideal"] for r in runs),
           "weights_mb_fp32": runs[0]["weights_mb_fp32"], "paths": {}}
    for key in runs[0]["measurements"]:
        ms = [r["measurements"][key] for r in runs if key in r["measurements"]]
        row = {"median_ms": statistics.median(m["median_ms"] for m in ms), "p95_ms_worst_seed": max(m["p95_ms"] for m in ms),
               "median_ms_per_seed": [round(m["median_ms"], 3) for m in ms],
               "peak_total_vram_mb": statistics.median(m["peak_total_vram_mb"] for m in ms)}
        for k in ("profiled_matmul_flops_per_token", "cuda_kernels_per_forward", "cuda_kernel_time_ms", "host_syncs_per_forward"):
            vals = [m[k] for m in ms if m.get(k) is not None]
            if vals:
                row[k] = statistics.median(vals)
        out["paths"][key] = row
    return out


GROUPS = {"e07": ("e07_seeds", "e07b_lowcompute"), "e09": ("e09_audit_controls",)}


def comparison(profile: dict, router: dict) -> dict:
    out = {"NOTE": ("val/test loss: mean ± std over seeds, default inference mode (dense/moe/fixed: full depth; "
                    "adaptive AM-LLM: hard survival halting, threshold 0.5). FLOPs: analytic estimate (forward, "
                    "per token, multiply-add = 2); 'ideal' excludes attention for halted query tokens. "
                    "profile: per execution path (dispatch/mode/batch/diagnostics) with matching latency, VRAM and "
                    "profiled FLOPs; full-sequence forward, NOT generated tok/s. E07 and E09 used different code "
                    "versions (ADR-0014): compare models only within the same group."),
           "groups": {"e07/equal_active_flop_full_depth": ["dense", "moe", "ut", "amllm_fixed", "amllm", "amllm_m16"],
                      "e07/equal_total_params": ["dense_wide", "moe", "amllm_m16"],
                      "e07/reduced_compute": ["dense_l2", "moe_l2", "amllm_p03"],
                      "e07/exploratory_after_e07": ["amllm_ua"],
                      "e09/attention_vs_capacity": ["amllm", "amllm_ua", "amllm_e5"],
                      "e09/learned_vs_random_routing": ["amllm_fixed", "amllm_fixed_rand"],
                      "e09/depth_only": ["ut_adaptive"],
                      "e09/top1_full_softmax_seed42": ["amllm_fs", "amllm_mk1_fs", "amllm_ek1_fs"],
                      "e09/experts_per_module_seed42": ["amllm_e8"]},
           "models": {}}
    for group, exps in GROUPS.items():
        for exp in exps:
            for preset, reps in load_experiment(exp).items():
                modes = [default_mode(r) for r in reps]
                evals = [r["eval"][m] for r, m in zip(reps, modes)]
                entry = {
                    "experiment": exp, "arch": reps[0]["arch"], "default_eval_mode": modes[0],
                    "seeds": [r["seed"] for r in reps],
                    "params_total": reps[0]["parameters"]["total"],
                    "params_non_embedding": reps[0]["parameters"]["non_embedding"],
                    "val_loss": mstd([e["val_loss"] for e in evals]),
                    "val_ppl": mstd([math.exp(e["val_loss"]) for e in evals]),
                    "test_loss": (mstd([r["test_default_mode"]["val_loss"] for r in reps])
                                  if all(r.get("test_default_mode") for r in reps) else None),
                    "mean_hops": mstd([e["mean_hops"] for e in evals]),
                    "est_mflops_per_token_impl": mstd([e["flops_per_token"] / 1e6 for e in evals]),
                    "est_mflops_per_token_ideal": mstd([e["flops_per_token_ideal"] / 1e6 for e in evals]),
                    "train_seconds": mstd([r["train_seconds"] for r in reps]),
                    "train_tokens_per_s": mstd([r["train_tokens_per_s"] for r in reps]),
                    "train_dispatch": reps[0].get("train_dispatch"),
                    "train_peak_vram_mb": reps[0].get("train_peak_vram_mb"),
                }
                rr = [r["eval"].get("random_routes") or r["eval"].get("soft_random_routes") for r in reps]
                if all(rr):
                    ref = [r["eval"]["soft"]["val_loss"] if "soft_random_routes" in r["eval"] else e["val_loss"]
                           for r, e in zip(reps, evals)]
                    entry["random_routing_loss_increase"] = mstd([x["val_loss"] - y for x, y in zip(rr, ref)])
                prof = profile_summary(profile, exp, preset)
                if prof:
                    entry["profile"] = prof
                analysed = [router["runs"][k] for k in sorted(router.get("runs", {}))
                            if k.startswith(f"{exp}/{preset}_s")]
                if analysed:
                    def per_run(ra):
                        t, u = ra["trained"]["steps"], ra["untrained_control"]["steps"]
                        return {"mi_norm_trained_per_step": [s["mi_norm_by_H_class"] for s in t],
                                "mi_norm_untrained_per_step": [s["mi_norm_by_H_class"] for s in u],
                                "usage_entropy_norm_per_step": [s["usage_entropy_norm"] for s in t],
                                "max_share_per_step": [s["max_share"] for s in t],
                                "dead_units_per_step": [s["dead_units"] for s in t]}
                    runs_ = [per_run(ra) for ra in analysed]
                    entry["router"] = {
                        "seeds_analysed": len(runs_),
                        "mi_norm_trained_mean": mstd([statistics.fmean(r["mi_norm_trained_per_step"]) for r in runs_]),
                        "mi_norm_untrained_mean": mstd([statistics.fmean(r["mi_norm_untrained_per_step"]) for r in runs_]),
                        "max_share_max": max(max(r["max_share_per_step"]) for r in runs_),
                        "dead_units_max": max(max(r["dead_units_per_step"]) for r in runs_),
                        "per_seed": runs_}
                out["models"][f"{group}/{preset}"] = entry
    (RESULTS / "architecture_comparison.json").write_text(json.dumps(out, indent=2))
    return out


# ------------------------------------------------------------------------------------- figures
def style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_facecolor(SURFACE)


def new_fig(w=7.2, h=4.2):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    return fig, ax


def fig_learning_curves(main: dict, lang: str):
    T, L = TEXT[lang], LABEL[lang]
    fig, ax = new_fig()
    for preset in ["dense", "moe", "ut", "amllm_fixed", "amllm", "amllm_ua"]:
        reps = main.get(preset)
        if not reps:
            continue
        steps = [c["step"] for c in reps[0]["curve"]] + [reps[0]["steps"]]
        ys = []
        for i in range(len(steps)):
            vals = [(r["curve"][i]["val_loss"] if i < len(r["curve"]) else r["eval"][default_mode(r)]["val_loss"]) for r in reps]
            ys.append(statistics.fmean(vals))
        ax.plot(steps, ys, color=COLOR[preset], linewidth=2, marker="o", markersize=4, label=L[preset])
        ax.annotate(L[preset].split(" (")[0], (steps[-1], ys[-1]), xytext=(6, CURVE_OFFSET.get(preset, 0)),
                    textcoords="offset points", fontsize=7, color=INK2, va="center")
    style(ax, T["curves_title"], T["steps"], T["val_loss"])
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / lang / "learning_curves.png")
    plt.close(fig)


def fig_loss_vs_flops(cmp: dict, e03: dict, lang: str):
    T, L = TEXT[lang], LABEL[lang]
    fig, ax = new_fig()
    for key, m in cmp["models"].items():
        group, preset = key.split("/", 1)
        if group != "e07":
            continue
        x, y = m["est_mflops_per_token_ideal"]["mean"], m["val_loss"]
        hollow = preset in HOLLOW
        ax.errorbar(x, y["mean"], yerr=[[y["mean"] - y["min"]], [y["max"] - y["mean"]]], fmt="o",
                    color=COLOR.get(preset, INK2), markersize=8, markerfacecolor=SURFACE if hollow else COLOR.get(preset, INK2),
                    markeredgewidth=2, elinewidth=1.2, capsize=0, label=L.get(preset, preset), zorder=3)
        off = SCATTER_OFFSET.get(preset, (8, 4))
        ax.annotate(L.get(preset, preset).split(" (")[0], (x, y["mean"]), xytext=off, textcoords="offset points",
                    fontsize=7, color=INK2, ha="right" if off[0] < 0 else "left")
    rep = e03.get("amllm_p03", [None])[0]
    if rep:
        pts = sorted(((e["flops_per_token_ideal"] / 1e6, e["val_loss"]) for k, e in rep["eval"].items()
                      if k.startswith("hard_surv")))
        ax.plot(*zip(*pts), color=COLOR["amllm"], linewidth=1.2, linestyle="--", alpha=0.8, label=T["sweep"])
    style(ax, T["scatter_title"], T["mflops"], T["val_loss_lower"])
    ax.legend(fontsize=6.5, frameon=False, loc="upper right", ncol=2)
    ax.set_ylim(top=ax.get_ylim()[1] + 0.12)
    ax.set_xlim(left=0.2)
    fig.tight_layout()
    fig.savefig(FIG / lang / "loss_vs_flops.png")
    plt.close(fig)


def fig_halting_tradeoff(e03: dict, lang: str):
    T = TEXT[lang]
    fig, ax = new_fig()
    for preset, lam, color in (("amllm_p002", 0.002, "#2a78d6"), ("amllm_p01", 0.01, "#1baf7a"),
                               ("amllm_p03", 0.03, "#eb6834"), ("amllm_p10", 0.1, "#6250d6")):
        rep = e03.get(preset, [None])[0]
        if not rep:
            continue
        pts = sorted(((e["mean_hops"], e["val_loss"]) for k, e in rep["eval"].items() if k.startswith("hard_surv")))
        label = f"λ={num(lam, lang, 3 if lam < 0.01 else 2)} {T['hard_range']}"
        ax.plot(*zip(*pts), color=color, linewidth=2, marker="o", markersize=5, label=label)
        soft = rep["eval"]["soft"]
        ax.plot([soft["mean_hops"]], [soft["val_loss"]], marker="D", markersize=7, color=color,
                markerfacecolor=SURFACE, markeredgewidth=2, linestyle="none")
    fixed = e03.get("amllm_fixed", [None])[0]
    if fixed:
        ax.axhline(fixed["eval"]["fixed"]["val_loss"], color=INK2, linewidth=1, linestyle=":")
        ax.annotate(T["fixed_ref"], (1.0, fixed["eval"]["fixed"]["val_loss"]), xytext=(0, 4),
                    textcoords="offset points", fontsize=7, color=INK2)
    style(ax, T["halt_title"], T["hops_axis"], T["val"])
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / lang / "halting_tradeoff.png")
    plt.close(fig)


def fig_hops_per_class(e03: dict, main: dict, lang: str):
    T = TEXT[lang]
    fig, ax = new_fig()
    series = [(T["s42_p03"], e03.get("amllm_p03", [None])[0], COLOR["amllm_p03"], SURFACE),
              (T["s42_main"], (main.get("amllm") or [None])[0], COLOR["amllm"], COLOR["amllm"])]
    classes = None
    width = 0.38
    for i, (label, rep, color, face) in enumerate(series):
        if not rep:
            continue
        pc = rep["eval"]["hard_surv_0.5"]["per_category"]
        classes = classes or list(pc)
        xs = [j + (i - 0.5) * width for j in range(len(classes))]
        ax.bar(xs, [pc[c]["mean_hops"] for c in classes], width=width - 0.04, color=face, edgecolor=color,
               linewidth=1.5, label=label)
    if classes:
        names = [CLASS_HU.get(c, c) if lang == "hu" else c for c in classes]
        ax.set_xticks(range(len(classes)), names, fontsize=8)
    style(ax, T["class_title"], "", T["class_y"])
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / lang / "hops_per_byteclass.png")
    plt.close(fig)


def fig_router_mi(router: dict, lang: str):
    T = TEXT[lang]
    panels = [("e07_seeds/moe_s42", "MoE"), ("e07_seeds/amllm_s42", "AM-LLM"),
              ("e07b_lowcompute/amllm_ua_s42", "AM-LLM per-hop attn" if lang == "en" else "AM-LLM hoponkénti attn."),
              ("e09_audit_controls/amllm_fixed_rand_s42", "random routes (trained)" if lang == "en" else "véletlen útvonal (tanítva)"),
              ("e06_ablations/amllm_mk1_s42", "module top-1, topk_softmax" if lang == "en" else "modul Top-1, topk_softmax"),
              ("e09_audit_controls/amllm_mk1_fs_s42", "module top-1, full_softmax" if lang == "en" else "modul Top-1, full_softmax")]
    runs = {title: router["runs"][key] for key, title in panels if key in router.get("runs", {})}
    if not runs:
        return
    n = len(runs)
    cols = 3
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.6, 2.3 * rows + 0.6), dpi=150, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, r) in zip(axes.flat, runs.items()):
        t = [st["mi_norm_by_H_class"] for st in r["trained"]["steps"]]
        u = [su["mi_norm_by_H_class"] for su in r["untrained_control"]["steps"]][:len(t)]
        xs = list(range(1, len(t) + 1))
        ax.bar([x - 0.19 for x in xs[:len(u)]], u, width=0.36, color=SURFACE, edgecolor=INK2, linewidth=1.1,
               label=T["untrained"])
        ax.bar([x + 0.19 for x in xs], t, width=0.36, color=COLOR["amllm"], label=T["trained"])
        style(ax, name, T["layer"] if r["arch"] == "moe" else T["hop"], "")
        ax.title.set_fontsize(9)
        ax.set_xticks(xs)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    axes[0][0].set_ylabel(T["mi_y"], color=INK2, fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(T["mi_title"], x=0.01, ha="left", fontsize=10, color=INK)
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.95), fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(FIG / lang / "router_mutual_information.png")
    plt.close(fig)


def fig_latency(cmp: dict, lang: str):
    T, L = TEXT[lang], LABEL[lang]
    names, med, p95, colors = [], [], [], []
    for key, m in cmp["models"].items():
        group, preset = key.split("/", 1)
        if group != "e07" or "profile" not in m:
            continue
        for path, row in m["profile"]["paths"].items():
            parts = path.split("/")  # dispatch/mode/batch[/t<len>]/diag
            dispatch, mode, batch, diag = parts[0], parts[1], parts[2], parts[-1]
            if batch != "b32" or mode == "soft" or diag != "diag_off" or (len(parts) == 5 and parts[3] != "t256"):
                continue
            names.append(f"{L.get(preset, preset).split(' (')[0]} · {dispatch}")
            med.append(row["median_ms"])
            p95.append(row["p95_ms_worst_seed"])
            colors.append(COLOR.get(preset, INK2))
    if not names:
        return
    fig, ax = new_fig(7.2, 0.28 * len(names) + 1.2)
    ys = list(range(len(names)))[::-1]
    ax.barh(ys, med, color=colors, height=0.62, label=T["median"])
    ax.scatter(p95, ys, color=INK, marker="|", s=90, zorder=3, label=T["p95"])
    for y, v, w in zip(ys, med, p95):
        ax.annotate(f"{num(v, lang)} ms", (max(v, w), y), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=7, color=INK2)
    ax.set_yticks(ys, names, fontsize=7)
    style(ax, T["lat_title"], T["lat_x"], "")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(right=max(p95) * 1.18)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / lang / "latency.png")
    plt.close(fig)


def main():
    for lang in LANGS:
        (FIG / lang).mkdir(parents=True, exist_ok=True)
    # e04_dense / e05_moe only hold symlinks into the E07 matrix: skip them to avoid duplicate rows
    experiments = sorted(p.name for p in RESULTS.glob("e0*") if p.is_dir()
                         and any(r.parent.is_dir() and not r.parent.is_symlink() for r in p.glob("*_s*/report.json")))
    rows = write_summary(experiments)
    profile = json.loads((RESULTS / "profile.json").read_text()) if (RESULTS / "profile.json").exists() else {}
    router = json.loads((RESULTS / "router_analysis.json").read_text()) if (RESULTS / "router_analysis.json").exists() else {}
    cmp = comparison(profile, router)
    e03, main_runs = load_experiment("e03_halting"), load_experiment("e07_seeds")
    for lang in LANGS:
        fig_learning_curves({**main_runs, **load_experiment("e07b_lowcompute")}, lang)
        fig_loss_vs_flops(cmp, e03, lang)
        fig_halting_tradeoff(e03, lang)
        fig_hops_per_class(e03, main_runs, lang)
        fig_router_mi(router, lang)
        fig_latency(cmp, lang)
    print(f"summary.csv rows={len(rows)} experiments={experiments}; figures in {FIG}/{{en,hu}}")


if __name__ == "__main__":
    main()
