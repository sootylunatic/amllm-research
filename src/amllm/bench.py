"""Prototype microbenchmark only. Not a realistic language decoding throughput test."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from .config import ModelConfig
from .model import build_model, count_parameters


def sync(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=20)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--mode', choices=['soft', 'hard'], default='hard')
    p.add_argument('--output', default=None)
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Only load local, trusted checkpoints; weights_only never unpickles arbitrary objects.
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    cfg = ModelConfig.from_dict(ckpt['config'])
    model = build_model(cfg).to(device).eval()
    model.load_state_dict(ckpt['state_dict'])
    torch.manual_seed(2026)
    ids = torch.randint(0, 256, (a.batch, cfg.max_seq_len), device=device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    durations = []
    with torch.no_grad():
        for i in range(a.warmup + a.repeats):
            sync(device)
            start = time.perf_counter()
            _, info = model(ids, hard_halt=(a.mode == 'hard'))
            sync(device)
            if i >= a.warmup:
                durations.append(time.perf_counter() - start)
    times = sorted(durations)
    report = {
        'WARNING': 'Microbenchmark of a full sequence forward pass, NOT autoregressive generated tokens per second.',
        'device': str(device), 'arch': cfg.arch, 'mode': a.mode, 'batch': a.batch, 'context_length': cfg.max_seq_len,
        'median_full_forward_ms': 1000 * times[len(times) // 2],
        'p95_full_forward_ms': 1000 * times[min(len(times) - 1, int(0.95 * len(times)))],
        'mean_hops': float(info['mean_hops']),
        'flops_per_token': info['flops_per_token'],
        'parameters': count_parameters(model),
        'cuda_peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 1) if device.type == 'cuda' else None,
    }
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        Path(a.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
