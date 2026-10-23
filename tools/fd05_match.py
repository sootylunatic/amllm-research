"""FD05 matching table: routed FLOP per token (relative to moe_f5) and total parameters of fixed-depth candidates.

    python tools/fd05_match.py                       # the default FD05 candidate list
    python tools/fd05_match.py "amllm_f5:halt_rule=none n_hops=8 expert_hidden=192" "moe_f5:expert_hidden=320"

All candidates are evaluated with a real forward pass (CPU, one 512-token sequence, hard path, fixed depth), so the
numbers are the model's own `flops_per_token` and `count_parameters`, not a separate formula. Executed FLOP (bmm padding)
is a GPU measurement and is not reported here.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from amllm.experiment import effective_config, parse_set  # noqa: E402
from amllm.model import build_model, count_parameters  # noqa: E402

FIXED = "halt_rule=none"
# Matched designs found with this tool on 2026-10-23 (routed FLOP x MoE / params x MoE in the comments)
CANDIDATES = {
    "B_fixed (6 hops, ref)": FIXED,                                                          # 1.001 / 1.000
    "H05 8 hops 1.00x": f"{FIXED} n_hops=8 expert_hidden=160 experts_per_module=8",          # 1.003 / 1.001
    "H05 4 hops 1.00x": f"{FIXED} n_hops=4 expert_hidden=832 experts_per_module=2",          # 1.000 / 0.965 (no expert choice)
    "H03 top2/top2 1.00x": f"{FIXED} module_topk=2 expert_hidden=192",                       # 1.002 / 0.793
    "H03 top2/top1 1.00x": f"{FIXED} module_topk=2 expert_topk=1",                           # 1.002 / 1.000
    "H04 finer experts 0.85x": f"{FIXED} experts_per_module=8 expert_hidden=192",            # 0.847 / 1.001
    "H04 smaller experts 0.86x": f"{FIXED} expert_hidden=208",                               # 0.859 / 0.810
}
# matched controls (pass as arguments): "moe_f5:n_experts=8 expert_hidden=192" 0.846 / 1.001 (pairs with H04 finer);
# "moe_f5:expert_hidden=208" 0.858 / 0.811 and "dense_f5:ffn_hidden=416" 0.857 / 0.698 (pair with H04 smaller)


def measure(preset: str, sets: str) -> dict:
    cfg, _ = effective_config(preset, parse_set(sets.split()) if sets else {})
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    with torch.no_grad():
        _, info = model(torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len)), hard_halt=True)
    return {"flops": float(info["flops_per_token"]), "params": count_parameters(model)["total"]}


def main():
    ref = measure("moe_f5", "")
    rows = [("moe_f5 (ref)", "moe_f5", "")] + [(k, "amllm_f5", v) for k, v in CANDIDATES.items()] \
        + [(a, *a.split(":", 1)) for a in sys.argv[1:]]
    print(f"{'candidate':34s} {'FLOP/tok':>12s} {'x MoE':>7s} {'params':>12s} {'x MoE':>7s}")
    for name, preset, sets in rows:
        m = ref if (preset, sets) == ("moe_f5", "") else measure(preset, sets)
        print(f"{name:34s} {m['flops']:12.4g} {m['flops'] / ref['flops']:7.3f} {m['params']:12,d} {m['params'] / ref['params']:7.3f}")


if __name__ == "__main__":
    main()
