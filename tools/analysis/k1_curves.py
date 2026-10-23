"""K1 (CPU): existing 2x (18,000-step) F5 curves - was the MoE degradation overfitting or a training instability?

Reads results/phase2/F5_2x/*_s42/report.json only. Overfitting predicts rising validation loss while the training loss
keeps falling; an instability event predicts both rising together. Training losses are single-batch samples every 25
steps, so they are averaged over 500-step windows.
"""
import json
import statistics as st
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2] / "results" / "phase2" / "F5_2x"
out = {}
for model in ("moe_f5", "dense_f5", "amllm_f5"):
    r = json.loads((root / f"{model}_s42" / "report.json").read_text())
    tc = r["train_curve"]
    windows = {f"{a}-{a + 500}": round(st.mean(l for s, l, _ in tc if a <= s < a + 500), 4) for a in range(10000, 18000, 500)}
    out[model] = {"val_curve": {x["step"]: round(x["val_loss"], 4) for x in r["curve"]},
                  "train_loss_500step_mean": windows,
                  "final_val_default": round(r["eval"].get("fixed", r["eval"].get("hard_surv_0.5"))["val_loss"], 4),
                  "epochs": round(r["steps"] * 32 * 512 / 53.1e6, 2)}
json.dump(out, sys.stdout, indent=1)
