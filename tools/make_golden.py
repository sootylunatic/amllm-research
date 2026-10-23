"""Record golden outputs of the current model code (run BEFORE changing model code).

Writes tests/golden_default_behaviour.json: for several default-configuration models (CPU, fp32, fixed seeds) the logits
statistics in soft and hard mode. tests/test_fd_switches.py asserts that new configuration switches leave these
unchanged when they are at their defaults.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from amllm.model import build_model  # noqa: E402
from test_model import ALL_CFGS, tiny_cfg  # noqa: E402

CASES = {**ALL_CFGS, "amllm_unshared_fixed": tiny_cfg(unshared_attention=True, halt_rule="none"),
         "amllm_straight_through": tiny_cfg(unshared_attention=True, route_weighting="straight_through", module_topk=1),
         "amllm_halting_active": tiny_cfg(unshared_attention=True, halt_bias_init=2.0)}  # hard path really truncates


def stats(t: torch.Tensor) -> dict:
    return {"sum": round(t.double().sum().item(), 5), "abs_mean": round(t.abs().double().mean().item(), 6),
            "first": [round(v, 6) for v in t.flatten()[:6].tolist()]}


def main():
    out = {}
    for name, cfg in CASES.items():
        torch.manual_seed(7)
        model = build_model(cfg).eval()
        model.diagnostics = True
        x = torch.randint(0, 256, (2, 12), generator=torch.Generator().manual_seed(11))
        rec = {}
        with torch.no_grad():
            if cfg.arch == "amllm":
                rec["soft"] = stats(model(x, hard_halt=False)[0])
                rec["hard"] = stats(model(x, hard_halt=True)[0])
            else:
                rec["fixed"] = stats(model(x)[0])
        rec["n_params"] = sum(p.numel() for p in model.parameters())
        out[name] = rec
    Path("tests/golden_default_behaviour.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} cases")


if __name__ == "__main__":
    main()
