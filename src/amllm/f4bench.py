"""F4 measurement: KV-cache generation correctness on real checkpoints (bf16) and generated tokens/s.

    python -m amllm.f4bench --runs <run dirs> --out results/phase2/F4/f4bench.json

Per checkpoint (Triton dispatch where the model has experts, diagnostics off, bf16, prompts from the validation split):
  agreement   greedy continuation (128 tokens after a 128-byte prompt, batch 32) decoded incrementally with a
              CUDA-Graph decode step, vs. the next-token argmax of the full prefix forward (reference path) on the
              same generated sequence; pre-registered threshold >= 97%
  prefill     tokens/s of the prompt pass (eager), batch 1 and 32
  decode      generated tokens/s for 128 greedy steps, eager and CUDA Graph, batch 1 and 32
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from . import env
from .data import eval_batches, load_split
from .generate import Generator, GraphDecoder
from .model import AdaptiveModularLM, set_diagnostics, set_dispatch
from .profile import load_run

PROMPT, NEW = 128, 128


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def greedy(gen: Generator, prompt: torch.Tensor, steps: int, graph: bool) -> tuple[torch.Tensor, float]:
    """Returns generated tokens [B, steps] and the decode time in seconds (prefill excluded)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = gen.prefill(prompt)
    token = logits[:, -1].argmax(-1)
    dec = GraphDecoder(gen, token) if graph else None
    out = []
    sync()
    start = time.perf_counter()
    for _ in range(steps):
        out.append(token)
        if dec is None:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = gen.decode_step(token)
        else:
            logits = dec(token)
        token = logits.argmax(-1)
    sync()
    return torch.stack(out, 1), time.perf_counter() - start


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data", default="data/ts_corpus")
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args()
    device = torch.device("cuda")
    val = load_split(a.data, "val")
    out = {"env": env.collect(), "protocol": {"prompt": PROMPT, "new_tokens": NEW, "decoding": "greedy", "dtype": "bf16"},
           "runs": {}}
    for run in a.runs:
        run_dir = Path(run)
        cfg, model = load_run(run_dir, device)
        if cfg.arch != "dense":
            set_dispatch(model, "triton")
        set_diagnostics(model, False)
        r = {"arch": cfg.arch}
        # --- correctness on real data (batch 32) ---
        prompt = next(eval_batches(val, 32, 32, PROMPT, device))[0]
        gen = Generator(model, batch=32, max_len=PROMPT + NEW, cache_dtype=torch.bfloat16)
        generated, _ = greedy(gen, prompt, NEW, graph=True)
        seq = torch.cat([prompt, generated], 1)[:, :PROMPT + NEW]
        set_dispatch(model, "bmm") if cfg.arch != "dense" else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            full = model(seq, hard_halt=True)[0] if isinstance(model, AdaptiveModularLM) else model(seq)[0]
        if cfg.arch != "dense":
            set_dispatch(model, "triton")
        full_pred = full[:, PROMPT - 1:PROMPT + NEW - 1].argmax(-1)       # predictions for the generated positions
        r["greedy_agreement_vs_full_forward"] = float((full_pred == generated).float().mean())
        r["cache_mb_batch32"] = gen.cache_bytes() / 2**20
        # --- throughput ---
        for batch in (1, 32):
            prompt_b = next(eval_batches(val, batch, batch, PROMPT, device))[0]
            gen = Generator(model, batch=batch, max_len=PROMPT + NEW, cache_dtype=torch.bfloat16)
            pre, dec_e, dec_g = [], [], []
            for _ in range(a.repeats + 1):
                sync()
                s = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gen.prefill(prompt_b)
                sync()
                pre.append(time.perf_counter() - s)
                dec_e.append(greedy(gen, prompt_b, NEW, graph=False)[1])
                dec_g.append(greedy(gen, prompt_b, NEW, graph=True)[1])
            pre, dec_e, dec_g = (statistics.median(v[1:]) for v in (pre, dec_e, dec_g))  # first repeat = warm-up
            r[f"b{batch}"] = {"prefill_tokens_per_s": batch * PROMPT / pre, "prefill_ms": 1000 * pre,
                              "decode_tokens_per_s_eager": batch * NEW / dec_e,
                              "decode_tokens_per_s_graph": batch * NEW / dec_g,
                              "decode_ms_per_step_graph": 1000 * dec_g / NEW}
        out["runs"][f"{run_dir.parent.name}/{run_dir.name}"] = r
        print(f"{run_dir.parent.name}/{run_dir.name:24s} agree={r['greedy_agreement_vs_full_forward']:.3f} "
              f"b1: prefill {r['b1']['prefill_tokens_per_s']:,.0f} tok/s, decode eager {r['b1']['decode_tokens_per_s_eager']:,.0f} "
              f"graph {r['b1']['decode_tokens_per_s_graph']:,.0f} tok/s | b32: decode graph {r['b32']['decode_tokens_per_s_graph']:,.0f} tok/s",
              flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
