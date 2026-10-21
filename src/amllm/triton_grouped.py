"""Triton grouped GEMM for the expert bank (phase 2, F3): sync-free, padding-free, CUDA-Graph capturable.

Rows are sorted by expert (counts on the device). Each program computes one BM x BN tile of one expert:
- the tile -> expert assignment is found inside the kernel from the cumulative tile counts (no host sync);
- the grid size depends only on the input shape: cdiv(rows, BM) + n_experts tiles (upper bound), extra tiles exit;
- the epilogue adds the expert bias and optionally applies SiLU.
Inference only (no autograd); `grouped_ffn` is the two-layer expert FFN on sorted rows.
"""
from __future__ import annotations

import glob
import os
import sys
import sysconfig
from pathlib import Path


def _ensure_python_headers() -> None:
    """Triton compiles a small C helper against Python.h. If the interpreter's include dir lacks it (e.g. a system
    Python without the -dev package), use the headers of a uv-managed CPython of the same minor version."""
    if Path(sysconfig.get_paths()["include"], "Python.h").exists() or os.environ.get("C_INCLUDE_PATH"):
        return
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    hits = sorted(glob.glob(str(Path.home() / f".local/share/uv/python/cpython-{ver}*/include/python{ver}/Python.h")))
    if hits:
        os.environ["C_INCLUDE_PATH"] = str(Path(hits[-1]).parent)


_ensure_python_headers()

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _grouped_linear_kernel(x_ptr, w_ptr, b_ptr, y_ptr, row_off_ptr, counts_ptr, tile_cum_ptr,
                           K, N, stride_xm, stride_wg, stride_wk, stride_wn, stride_bg, stride_ym,
                           G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                           SILU: tl.constexpr, IEEE: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # expert of this tile: number of experts whose cumulative tile count is <= pid_m
    g = 0
    for i in tl.static_range(G):
        g += (tl.load(tile_cum_ptr + i) <= pid_m).to(tl.int32)
    valid_tile = g < G
    g = tl.minimum(g, G - 1)
    tiles_before = tl.where(g > 0, tl.load(tile_cum_ptr + tl.maximum(g - 1, 0)), 0)
    row_start = tl.load(row_off_ptr + g) + (pid_m - tiles_before) * BM
    row_end = tl.load(row_off_ptr + g) + tl.load(counts_ptr + g)

    rows = row_start + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    row_mask = (rows < row_end) & valid_tile
    col_mask = cols < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        x = tl.load(x_ptr + rows[:, None] * stride_xm + ks[None, :], mask=row_mask[:, None] & (ks[None, :] < K), other=0.0)
        w = tl.load(w_ptr + g * stride_wg + ks[:, None] * stride_wk + cols[None, :] * stride_wn,
                    mask=(ks[:, None] < K) & col_mask[None, :], other=0.0)
        if IEEE:
            acc += tl.dot(x, w, input_precision="ieee")  # fp32 inputs: exact fp32 products (no TF32)
        else:
            acc += tl.dot(x, w)
    acc += tl.load(b_ptr + g * stride_bg + cols, mask=col_mask, other=0.0).to(tl.float32)[None, :]
    if SILU:
        acc = acc * tl.sigmoid(acc)
    tl.store(y_ptr + rows[:, None] * stride_ym + cols[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=row_mask[:, None] & col_mask[None, :])


def grouped_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, counts: torch.Tensor, silu: bool,
                   bm: int = 64, bn: int = 64, bk: int = 32) -> torch.Tensor:
    """y[rows of expert g] = act(x[rows] @ w[g] + b[g]); x [A,K] sorted by expert, w [G,K,N], b [G,N], counts [G]."""
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, w, b)):
        # the kernel has no backward: training through it would leave the expert weights without gradients
        raise RuntimeError("triton dispatch is inference only (no autograd); train with dispatch loop, bmm or grouped")
    a_rows, k = x.shape
    g_count, _, n = w.shape
    w = w.to(x.dtype).contiguous()
    b = b.to(x.dtype).contiguous()
    x = x.contiguous()
    counts = counts.to(torch.int32)
    row_off = (torch.cumsum(counts, 0) - counts).to(torch.int32)
    tile_cum = torch.cumsum((counts + bm - 1) // bm, 0).to(torch.int32)
    y = torch.empty(a_rows, n, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(a_rows, bm) + g_count, triton.cdiv(n, bn))
    _grouped_linear_kernel[grid](x, w, b, y, row_off, counts, tile_cum, k, n,
                                 x.stride(0), w.stride(0), w.stride(1), w.stride(2), b.stride(0), y.stride(0),
                                 G=g_count, BM=bm, BN=bn, BK=bk, SILU=silu, IEEE=x.dtype == torch.float32)
    return y


def grouped_ffn(xs: torch.Tensor, w1, b1, w2, b2, counts: torch.Tensor) -> torch.Tensor:
    """Expert FFN on expert-sorted rows: SiLU(xs @ w1[g] + b1[g]) @ w2[g] + b2[g]."""
    h = grouped_linear(xs, w1, b1, counts, silu=True)
    return grouped_linear(h, w2, b2, counts, silu=False)
