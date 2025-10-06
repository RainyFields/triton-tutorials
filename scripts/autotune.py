#!/usr/bin/env python3
"""
Autotuned vector addition in Triton with a built-in benchmark.

Usage:
  python autotune.py --print --plot --save /work/lei/triton-tutorials/results --check

Notes:
  - Works on CUDA if available; otherwise falls back to CPU (Torch path only).
  - Autotuning varies BLOCK_SIZE, num_warps, num_stages to find fastest config.
"""

import os
import argparse
import math
import torch
import triton
import triton.language as tl

# ---------- Device selection ----------
os.environ.setdefault("TRITON_DEBUG", "0")
if os.getenv("TRITON_INTERPRET") == "1":
    DEVICE = torch.device("cpu")
elif torch.cuda.is_available():
    DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}")  # e.g., cuda:0
else:
    DEVICE = torch.device("cpu")


# ---------- Autotuned Triton kernel ----------
# We try several configurations; Triton will pick the fastest for each problem size.
autotune_configs = [
    triton.Config({"BLOCK_SIZE": 128},  num_warps=2, num_stages=1),
    triton.Config({"BLOCK_SIZE": 256},  num_warps=4, num_stages=1),
    triton.Config({"BLOCK_SIZE": 512},  num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=3),
]

@triton.autotune(configs=autotune_configs, key=['n_elements'])
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)  # 1D launch grid
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = x + y
    tl.store(out_ptr + offsets, z, mask=mask)


# ---------- Public API ----------
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Elementwise sum of x and y using Triton (if on CUDA) or Torch fallback.
    """
    # Basic checks
    if x.shape != y.shape:
        raise ValueError(f"Shapes must match: {x.shape} vs {y.shape}")
    if x.dtype != y.dtype:
        raise ValueError(f"dtype must match: {x.dtype} vs {y.dtype}")
    if x.device != y.device:
        raise ValueError(f"device must match: {x.device} vs {y.device}")

    # Torch fallback when not on CUDA
    if x.device.type != "cuda":
        return x + y

    # Triton path
    out = torch.empty_like(x)
    n_elements = out.numel()

    # Require contiguous for simplest addressing (optional)
    if not x.is_contiguous():
        x = x.contiguous()
    if not y.is_contiguous():
        y = y.contiguous()
    if not out.is_contiguous():
        out = out.contiguous()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, n_elements)
    # No explicit synchronize needed unless you read on CPU immediately.
    return out


# ---------- Benchmark ----------
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(12, 28, 1)],  # 4 KB .. 256 MB (float32)
        x_log=True,
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'Torch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance-autotune',
        args={}
    )
)
def benchmark(size, provider: str):
    # Allocate test tensors
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)

    # Measure time
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    else:  # 'triton'
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)

    # Effective BW: read x, read y, write out => 3 * N * sizeof(float)
    to_gbps = lambda t_ms: 3 * x.numel() * x.element_size() * 1e-9 / (t_ms * 1e-3)
    return to_gbps(ms), to_gbps(max_ms), to_gbps(min_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--print', dest='print_data', action='store_true', help='Print raw benchmark rows')
    parser.add_argument('--plot', dest='show_plots', action='store_true', help='Display plots interactively')
    parser.add_argument('--save', dest='save_path', type=str, default='', help='Directory to save plot + CSV')
    parser.add_argument('--check', action='store_true', help='Run a quick correctness check')
    args = parser.parse_args()

    print(f'Using device: {DEVICE}')

    if args.check:
        torch.manual_seed(0)
        size = 98_432
        x = torch.rand(size, device=DEVICE)
        y = torch.rand(size, device=DEVICE)
        z_torch = x + y
        z_triton = add(x, y)
        diff = torch.max(torch.abs(z_torch - z_triton)).item()
        print(f'[check] max |Δ| = {diff:.3e}')

    benchmark.run(
        print_data=args.print_data,
        show_plots=args.show_plots,
        save_path=(args.save_path if args.save_path else None)
    )


if __name__ == '__main__':
    main()
