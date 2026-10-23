"""Run a matrix of presets x seeds with identical data, token budget and eval; write summary.csv.

    python -m amllm.experiment --name e07_seeds --presets dense,moe,amllm --seeds 42,43,44

All presets share d_model=128, context 256, data, optimizer and token budget. Most are matched
to ~1.9 MFLOP/token at full depth (attention + 4 x 262k expert/FFN FLOPs); dense_wide,
moe and amllm_m16 form the ~equal *total* parameter group (~2.2-2.4M non-embedding).
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

from .config import ModelConfig
from . import env
from .data import is_token_corpus, split_id
from .train import TrainConfig, run

BASE = dict(max_seq_len=256, d_model=128, n_heads=4)
AMLLM = dict(BASE, arch="amllm", n_modules=8, experts_per_module=4, module_topk=2, expert_topk=2,
             expert_hidden=128, n_hops=4, halt_rule="survival", halt_threshold=0.5)
F5_BASE = dict(max_seq_len=512, d_model=384, n_heads=6, vocab_size=8192)

PRESETS: dict[str, dict] = {
    # --- controls --------------------------------------------------------------------------
    "dense": dict(BASE, arch="dense", n_layers=4, ffn_hidden=512),          # ~equal active FLOP
    "dense_wide": dict(BASE, arch="dense", n_layers=4, ffn_hidden=2048),    # ~equal total params to moe/amllm_m16
    "moe": dict(BASE, arch="moe", n_layers=4, n_experts=8, moe_topk=2, expert_hidden=256),
    # reduced-compute controls (~0.98 MFLOP/token) for comparison with strongly halting AM-LLM
    "dense_l2": dict(BASE, arch="dense", n_layers=2, ffn_hidden=512),
    "moe_l2": dict(BASE, arch="moe", n_layers=2, n_experts=8, moe_topk=2, expert_hidden=256),
    # weight-tied recurrence without routing (Universal Transformer): isolates sharing from modularity
    "ut": dict(BASE, arch="amllm", n_modules=1, experts_per_module=1, module_topk=1, expert_topk=1,
               expert_hidden=512, n_hops=4, halt_rule="none"),
    # --- AM-LLM ----------------------------------------------------------------------------
    "amllm_fixed": dict(AMLLM, halt_rule="none"),                           # routing only, fixed depth
    "amllm": dict(AMLLM),                                                    # routing + adaptive depth
    "amllm_m16": dict(AMLLM, n_modules=16),                                  # ~equal total params to moe
    "amllm_m16_fixed": dict(AMLLM, n_modules=16, halt_rule="none"),
    # E03 ponder sweep
    **{f"amllm_p{tag}": dict(AMLLM, ponder_coefficient=v)
       for tag, v in (("002", 0.002), ("01", 0.01), ("03", 0.03), ("10", 0.1))},
    # E06 one-factor ablations around "amllm"
    "amllm_h2": dict(AMLLM, n_hops=2),
    "amllm_h8": dict(AMLLM, n_hops=8),
    "amllm_mk1": dict(AMLLM, module_topk=1),
    "amllm_ek1": dict(AMLLM, expert_topk=1),
    "amllm_ek4": dict(AMLLM, expert_topk=4),
    "amllm_m4": dict(AMLLM, n_modules=4),
    "amllm_v01": dict(AMLLM, attn_per_hop=False),                           # v0.1 topology: one attention
    # exploratory (after E07): per-hop attention weights, shared module bank
    "amllm_ua": dict(AMLLM, unshared_attention=True),
    # --- E09: controls added after the audit of E00-E08 -------------------------------------
    # per-hop attention adds 198,912 parameters; this shared-attention model gets +265,224 instead
    # (5 experts/module, same top-2 => same active FLOP): separates "unsharing" from "more capacity"
    "amllm_e5": dict(AMLLM, experts_per_module=5),
    "amllm_e8": dict(AMLLM, experts_per_module=8),                          # E06: 4 vs 8 experts/module
    # learned vs. random (not learned) routing at identical parameters and FLOP, fixed depth
    "amllm_fixed_rand": dict(AMLLM, halt_rule="none", router_mode="random"),
    # dynamic depth only: weight-tied block + halting, no module router
    "ut_adaptive": dict(BASE, arch="amllm", n_modules=1, experts_per_module=1, module_topk=1, expert_topk=1,
                        expert_hidden=512, n_hops=4, halt_rule="survival", halt_threshold=0.5),
    # Switch-style full-softmax route weights: the top-1 routers receive LM gradient
    "amllm_fs": dict(AMLLM, route_weighting="full_softmax"),
    "amllm_mk1_fs": dict(AMLLM, module_topk=1, route_weighting="full_softmax"),
    "amllm_ek1_fs": dict(AMLLM, expert_topk=1, route_weighting="full_softmax"),
    # --- Phase 2 / F1 (docs/PHASE2_PLAN.md §3) ---------------------------------------------
    # MoE controls matched in parameters AND FLOPs to per-hop-attention AM-LLM (top-2 / top-1)
    "moe_e4": dict(BASE, arch="moe", n_layers=4, n_experts=4, moe_topk=2, expert_hidden=256),
    "moe_e8h128": dict(BASE, arch="moe", n_layers=4, n_experts=8, moe_topk=2, expert_hidden=128),
    "amllm_ua_fs": dict(AMLLM, unshared_attention=True, route_weighting="full_softmax"),
    "amllm_ua_mk1_fs": dict(AMLLM, unshared_attention=True, module_topk=1, route_weighting="full_softmax"),
    "amllm_ua_ek1_fs": dict(AMLLM, unshared_attention=True, expert_topk=1, route_weighting="full_softmax"),
    "amllm_ua_both1_fs": dict(AMLLM, unshared_attention=True, module_topk=1, expert_topk=1,
                              route_weighting="full_softmax"),
    # F1b: straight-through weighting (forward of topk_softmax, router gradient for every k)
    "amllm_ua_st": dict(AMLLM, unshared_attention=True, route_weighting="straight_through"),
    "amllm_ua_mk1_st": dict(AMLLM, unshared_attention=True, module_topk=1, route_weighting="straight_through"),
    "amllm_ua_ek1_st": dict(AMLLM, unshared_attention=True, expert_topk=1, route_weighting="straight_through"),
    # F2-2 operating point of the F1b primary (same training code as F1b)
    **{f"amllm_ua_mk1_st_p{tag}": dict(AMLLM, unshared_attention=True, module_topk=1,
                                      route_weighting="straight_through", ponder_coefficient=v)
       for tag, v in (("02", 0.02), ("03", 0.03))},
    # F1a halting calibration on the amllm_ua_fs family (fixed-depth reference + ponder sweep)
    "amllm_ua_fs_fixed": dict(AMLLM, unshared_attention=True, route_weighting="full_softmax", halt_rule="none"),
    **{f"amllm_ua_fs_p{tag}": dict(AMLLM, unshared_attention=True, route_weighting="full_softmax",
                                   ponder_coefficient=v)
       for tag, v in (("002", 0.002), ("01", 0.01), ("03", 0.03), ("10", 0.1))},
    # --- Phase 2 / F5: BPE corpus v2, d_model 384, context 512 (preregistration F5) ------------
    # moe_f5 and amllm_f5 have the same expert pool (6 x 4 = 24 experts of width 384) and the same active expert
    # width per layer/hop (2 x 384); dense_f5 is matched to their active FLOPs (FFN 768)
    "dense_f5": dict(F5_BASE, arch="dense", n_layers=6, ffn_hidden=768),
    "moe_f5": dict(F5_BASE, arch="moe", n_layers=6, n_experts=4, moe_topk=2, expert_hidden=384),
    "amllm_f5": dict(F5_BASE, arch="amllm", n_modules=6, experts_per_module=4, module_topk=1, expert_topk=2,
                     expert_hidden=384, n_hops=6, halt_rule="survival", halt_threshold=0.5, unshared_attention=True,
                     route_weighting="straight_through", ponder_coefficient=0.02),
}


def parse_set(items: list[str]) -> dict:
    """key=value overrides: true/false, int, float, otherwise the string (e.g. halt_rule=none)."""
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
            continue
        for cast in (int, float):
            try:
                out[k] = cast(v)
                break
            except ValueError:
                pass
        else:
            out[k] = v
    return out


# val_bpb: bits per byte on the byte corpus. On a BPE corpus the training report's "val_bpb" is bits per *token*
# (loss / ln 2); it is written as val_bits_per_token and val_bpb is left empty (exact bits/byte: amllm.f5eval).
FIELDS = ["preset", "seed", "arch", "eval_mode", "params_total", "params_non_embedding", "val_loss", "val_bpb",
          "val_bits_per_token", "val_ppl",
          "mean_hops", "p95_hops", "flops_per_token", "flops_per_token_ideal", "median_batch_ms",
          "train_tokens_per_s", "train_seconds", "expert_usage_entropy", "expert_usage_entropy_max", "dead_experts", "max_expert_share"]


def check_reusable(run_dir: Path, cfg: ModelConfig, tc: TrainConfig, allow_legacy: bool = False) -> None:
    """--skip-existing may reuse a finished run only if model/train configuration, data split and training code
    are identical. Runs stored before the code hash was recorded need an explicit allow_legacy."""
    saved = json.loads((run_dir / "config.json").read_text())
    problems = []
    if ModelConfig.from_dict(saved["model"]) != cfg:  # newly added fields at their default compare equal
        problems.append("model configuration")
    stored_train = {k: v for k, v in saved["train"].items() if k in asdict(tc)}
    if stored_train != {k: v for k, v in asdict(tc).items() if k in saved["train"]}:
        problems.append("training configuration")
    if saved.get("data_split_id") != split_id(tc.data):
        problems.append("data split (corpus content changed)")
    stored_code = saved.get("train_code_sha256")
    if stored_code is None:
        if not allow_legacy:
            problems.append("training-code hash not recorded (legacy run; pass --allow-legacy-reuse to accept)")
    elif stored_code != env.train_code_hash():
        problems.append("training code")
    if problems:
        raise SystemExit(f"{run_dir}: cannot reuse with --skip-existing, differs in: {', '.join(problems)}; "
                         "use a new --name or remove the run")


def write_summary(path: Path, rows: list[dict]) -> None:
    """Merge with rows of other (preset, seed) pairs already in the file, so several invocations
    with different presets/seeds can share one experiment directory."""
    done = {(r["preset"], str(r["seed"])) for r in rows}
    kept = []
    if path.exists():
        with path.open(newline="") as f:
            kept = [r for r in csv.DictReader(f) if (r["preset"], r["seed"]) not in done]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(kept + rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--presets", required=True)
    p.add_argument("--seeds", default="42")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data", default="data/ts_corpus")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--skip-existing", action="store_true",
                   help="reuse finished runs whose configuration, data split and training code are identical")
    p.add_argument("--allow-legacy-reuse", action="store_true",
                   help="with --skip-existing: also reuse runs stored before the training-code hash was recorded")
    p.add_argument("--dispatch", default="bmm", choices=["loop", "bmm", "grouped"],
                   help="training dispatch; bmm is numerically equivalent to loop (tested); triton is inference only")
    p.add_argument("--set", nargs="*", default=[], help="key=value overrides applied to every amllm-arch preset")
    a = p.parse_args()
    overrides = parse_set(a.set)
    out = Path("results") / a.name
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for preset in a.presets.split(","):
        for seed in (int(s) for s in a.seeds.split(",")):
            run_dir = out / f"{preset}_s{seed}"
            kw = dict(PRESETS[preset])
            if kw["arch"] == "amllm" and kw.get("halt_rule", "survival") != "none":
                # the E03 sweep presets keep their own ponder coefficient
                kw.update({k: v for k, v in overrides.items()
                           if not (preset.startswith("amllm_p") and k == "ponder_coefficient")})
            cfg = ModelConfig(**kw)
            tc = TrainConfig(data=a.data, steps=a.steps, batch=a.batch, lr=a.lr, seed=seed,
                             eval_every=a.eval_every, dispatch=a.dispatch, run_dir=str(run_dir))
            if a.skip_existing and (run_dir / "report.json").exists():
                check_reusable(run_dir, cfg, tc, a.allow_legacy_reuse)
                report = json.loads((run_dir / "report.json").read_text())
            else:
                print(f"\n=== {a.name}: {preset} seed={seed} ===", flush=True)
                report = run(cfg, tc)
            for mode, e in report["eval"].items():
                rows.append({"preset": preset, "seed": seed, "arch": report["arch"], "eval_mode": mode,
                             "params_total": report["parameters"]["total"],
                             "params_non_embedding": report["parameters"]["non_embedding"],
                             "train_tokens_per_s": round(report["train_tokens_per_s"]),
                             "train_seconds": round(report["train_seconds"], 1),
                             **{k: e[k] for k in FIELDS if k in e}})
                if is_token_corpus(a.data):
                    rows[-1]["val_bits_per_token"], rows[-1]["val_bpb"] = rows[-1]["val_bpb"], ""
            write_summary(out / "summary.csv", rows)
    print(f"\nwrote {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
