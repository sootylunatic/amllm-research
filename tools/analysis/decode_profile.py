"""Goal C exploration: profile one KV-cache decode step per model (torch.profiler; batch 1, prompt 128, 10 eager steps).
Run on a GPU pod from the repository root: python tools/analysis/decode_profile.py. Prints per model the GPU time, kernel
count and the top kernels (output recorded in results/round_v3_w3/c_decode/profile_w3a_console.txt).
"""
import torch, json
from torch.profiler import profile, ProfilerActivity
from amllm.profile import load_run
from amllm.model import set_dispatch, set_diagnostics
from amllm.generate import Generator, GraphDecoder
from amllm.data import load_split, eval_batches
from pathlib import Path
dev = torch.device("cuda"); val = load_split("data/code_corpus_v2", "val")
RUNS = {"B_moe": "results/round_v3_w2/B_moe/moe_f5_s42", "FDC1": "results/round_v3_w2/FDC1/amllm_f5_s42",
        "B_fixed": "results/round_v3_w2/B_fixed/amllm_f5_s42", "B_dense": "results/round_v3_w2/B_dense/dense_f5_s42"}
for name, run in RUNS.items():
    cfg, m = load_run(Path(run), dev); set_diagnostics(m, False)
    if cfg.arch != "dense": set_dispatch(m, "triton")
    ids = next(eval_batches(val, 1, 1, 128, dev))[0]
    gen = Generator(m, batch=1, max_len=256, cache_dtype=torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        tok = gen.prefill(ids)[:, -1].argmax(-1)
        for _ in range(5): tok = gen.decode_step(tok).argmax(-1)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(10): tok = gen.decode_step(tok).argmax(-1)
            torch.cuda.synchronize()
    ev = prof.key_averages()
    cuda_total = sum(e.self_device_time_total for e in ev) / 10 / 1000
    kernels = sum(e.count for e in ev if e.device_type is not None and str(e.device_type).endswith("CUDA") and e.self_device_time_total > 0) / 10
    syncs = sum(e.count for e in ev if "cudaStreamSynchronize" in e.key or "cudaMemcpy" in e.key or "item" in e.key) / 10
    print(f"== {name}: GPU time/step {cuda_total:.3f} ms, kernels/step {kernels:.0f}, sync-ish calls/step {syncs:.0f}")
    top = sorted([e for e in ev if e.self_device_time_total > 0], key=lambda e: -e.self_device_time_total)[:8]
    for e in top: print(f"   {e.self_device_time_total/10/1000:7.3f} ms  x{e.count/10:5.1f}  {e.key[:80]}")
