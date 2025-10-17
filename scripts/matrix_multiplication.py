# PID reordering for improved SRAM sharing between PIDs
# multi-dimensional pointer arithmetic
# data types => high precision accumulation
import os
import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else torch.device('cpu')
print(DEVICE)

# os.environ['TRITON_INTERPRET'] = '1' # interpret mode, easier to debug
# it suppose to support use print() in the kernel, but it does not work sometimes

# when encoutering new keys, profiling will be triggered to find the best config
# the best config will be cached for future use
autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
] # usually we can use for loops as in fla. 

# everytime M/N/K changes, autotune will be triggered to find the best config
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])

@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # manually defined meta parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):
    """
    an example:
    M = N = K = 8
    BLOCK_SIZE_M/N/K = 2
    [0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9, 10, 11]
    [12, 13, 14, 15]
    => 0 is a 2*2 chunk 
    put mutliple pid on the same SM, so they have access to the same SRAM
    row major ordering vs grouped ordering 
    [0,  2, |  4,  6]
    [1,  3, |  5,  7]
    --------|--------
    [8, 10, | 12, 14]
    [9, 11, | 13, 15] 
    what we need to do is to map the group of pids to the same SM!
    """
    GRID_M = tl.cdiv(M, BLOCK_SIZE_M)
    GRID_N = tl.cdiv(N, BLOCK_SIZE_N)

    GROUP_ROWS = GROUP_SIZE                   # number of rows per group
    PIDS_PER_GROUP = GROUP_ROWS * GRID_N      # number of pids per group

    pid = tl.program_id(0)
    group_id = pid // PIDS_PER_GROUP # which group
    pid_in_group = pid % PIDS_PER_GROUP # which pid inside the group

    # Base row index for this group
    m_base = group_id * GROUP_ROWS

    # Actual rows left for the last (possibly short) group, similar to mask
    rows_left = tl.maximum(0, GRID_M - m_base)
    rows_in_this_group = tl.minimum(GROUP_ROWS, rows_left)

    # Map flat pid_in_group to (PID_M, PID_n) inside the group
    PID_M = m_base + (pid_in_group % rows_in_this_group)
    PID_N = pid_in_group // rows_in_this_group  # columns first inside the group

    # suppose PID_M = 1, PID_N = 2， then 
    # offsets_M = 1*2 + [0,1] = [2,3]
    # offsets_N = 2*2 + [0,1] = [4,5]
    # offsets_K = [0,1]
    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)

    # create ptrs
    # a_offsets = [[2],[3]] * stride_am + [[0,1]] * stride_ak
    # a_offsets = [[2],[3]] * 8 + [[0,1]] * 1
    # a_offsets = [[16],[24]]  + [[0,1]] = [[16,17],[24,25]]
    a_offsets = offsets_M[:, None] * stride_am + offsets_K[None, :] * stride_ak
    b_offsets = offsets_K[:, None] * stride_bk + offsets_N[None, :] * stride_bn
    c_offsets = offsets_M[:, None] * stride_cm + offsets_N[None, :] * stride_cn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offsets_K < K - k * BLOCK_SIZE_K # 1-D mask
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0) # mask columns
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0) # mask rows
        # tl.dot actually do matmul
        # accumulator += tl.dot(a, b) # a: BLOCK_SIZE_M x BLOCK_SIZE_K, b: BLOCK_SIZE_K x BLOCK_SIZE_N => accumulator: BLOCK_SIZE_M x BLOCK_SIZE_N
        accumulator = tl.dot(a, b, acc = accumulator) # in place operation to save SRAM

        a_offsets += BLOCK_SIZE_K * stride_ak
        b_offsets += BLOCK_SIZE_K * stride_bk
    
    accumulator = accumulator.to(tl.float16) # cast back to float16; ?why save in float16?
    c_offsets = offsets_M[:,None] * stride_cm + offsets_N[None, :] * stride_cn
    # 2-D mask
    c_mask = (offsets_M[:, None] < M) & ( offsets_N[None, :] < N)
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)


def matmul(a, b):
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0]
    assert a.is_contiguous() and b.is_contiguous()

    (M, K), (_, N) = a.shape, b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # here we consider 1D grid for simplicity
    # each program computes a BLOCK_SIZE_M x BLOCK_SIZE_N output tile
    """
    an example:
    M = N = K = 8
    BLOCK_SIZE_M/N/K = 2
    [0, 1, 2, 3]
    [4, 5, 6, 7]
    [8, 9, 10, 11]
    [12, 13, 14, 15]
    """
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )
    # cdiv(x, y) = (x + y - 1) // y, ceiling division

    _matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c



def test_matmul_kernel(size: tuple, atol = 1e-2, rtol = 1e-1, device=DEVICE):
    # |a - b| < atol + rtol * |b|
    # atol : absolute tolerance
    # rtol : relative tolerance/percentage error; bigger numbers can have higher abs error
    # rule of thumb: atol = 1e-2 for fp16, 1e-4 for fp32; rtol = 1e-1 for fp16, 1e-3 for fp32
    assert type(size) == tuple and len(size) == 2
    torch.manual_seed(0)

    # during accumulation, it will be fp32 for higher accuracy
    x = torch.randn(size[0], size[1], device=device, dtype=torch.float16)
    y = torch.randn(size[1], size[0], device=device, dtype=torch.float16)
    z_tri = matmul(x, y)
    z_ref = torch.matmul(x, y)
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")

configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"],
        x_vals = [128 * i for i in range(2,16)],
        line_arg = 'provider',
        line_vals = ['torch', 'triton'],
        line_names = ['torch', 'triton'],
        styles = [("green", "-"), ("blue", "-")],
        ylabel = "GB/s",
        plot_name = "matmul_performance",
        args = {}
    )
]


@triton.testing.perf_report(configs)
def benchmark(M,N,K, provider):
    a = torch.randn(M, K, device = DEVICE, dtype = torch.float16)
    b = torch.randn(K, N, device = DEVICE, dtype = torch.float16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 3 * M * N * K * 1e-12 / (ms*1e-3)
    return perf(ms), perf(min_ms), perf(max_ms)

if __name__ == "__main__":
    test_matmul_kernel((512, 512))

    # only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark.run(save_path = '/work/lei/triton-tutorials/results/', print_data = False)
    