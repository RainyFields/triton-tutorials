# code online softmax
# reduce the numer of passes to 2
import time
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def naive_softmax(x: torch.Tensor)->torch.Tensor:
    """
    eager mode softmax, assume input of shape (M,N)
    """
    x_max = x.max(dim=1)[0] # pass 1： read MN elements and write M elements
    safe_x = x - x_max[:, None] # pass 2: read MN + M, subtraction is MN flops then write MN elements 
    numerator = torch.exp(safe_x) # pass 3: read MN elements, and write MN elements
    denominator = numerator.sum(dim = 1) # pass 4: read MN elements, sum MN flops then write M elements 
    sm_out = numerator / denominator[:,None] # pass 5: read MN + M, division is MN flops then write MN elements
    return sm_out 

    # in total: 8MN + 4M memory memory operations 


# fetch properties of the current device
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"] # number of streaming multiprocessors
NUM_REGS = properties["max_num_regs"] # each program share those registers
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] # how many kernel we can put on each SM
WARP_SIZE = properties["warp_size"] # usually 32 for NVIDIA GPU; the smallest group of GPU cores

# wrapper
def softmax(x):
    assert x.ndim == 2
    n_rows, n_cols = x.shape

    # assume each row fits with in SRAM
    BLOCK_SIZE = triton.next_power_of_2(n_cols) 

    # how many warps do we want to use to a single SM
    num_warps = 4
    # if rows are really big, we can use more warps
    if BLOCK_SIZE > 2048:
        num_warps = 8
    if BLOCK_SIZE > 4096:
        num_warps = 16 # 

    # how many stages to use in the pipeline
    num_stages = 4 if TOTAL_SRAM_PER_SM >= 16384 else 2

    y = torch.empty_like(x)

    # warmup
    kernel = _softmax_kernel.warmup(
        x, y, n_rows, n_cols, 
        BLOCK_SIZE = BLOCK_SIZE,
        num_stages = num_stages,
        num_warps = num_warps,
        grid=(1, ))
    kernel._init_handles()
    n_regs_per_program = kernel.n_regs # tells us how many registers are used by each program
    sram_needed_per_program = kernel.metadata.shared 

    # each program might use 32 registers per program, WARP_SIZE = 32
    # and num_warps = 8
    # so each program need 32*32*8 registers total
    # 65536 // (32*32*8) = 8 programs can run on each SM
    reg_occupancy = NUM_REGS // (n_regs_per_program * WARP_SIZE * num_warps)
    

    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    programs_per_sm = min(reg_occupancy, sram_occupancy) # one of the factors will be limitting
    # how many programs can we launch in total
    num_programs = min(NUM_SAM * programs_per_sm, n_rows)

    grid = (num_programs, 1, 1 ) # we did a warmup, so we need to launch 3-axis in total
    # assert x.is_contiguous()

    kernel[grid](
        x, y,
        x.stride(0), y.stride(0), # relate to how x,y are stored in memory
        n_rows, n_cols,
    )
    return y
    # x.stride()
    # x.stride() would be (N, 1)
    # x.stride(0) would be N
    # x.stride(1) would be 1
    # z.stride() z shape (B,N,D)
    # z.stride() would be (N*D, D, 1) # those tensors are contiguous in memory
    # if we use torch.reshape to change the shape of the tensor, the stride might be different

@triton.jit
def _softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # shape (M,N)
    # BLOCK_SIZE = next power of 2 bigger than N
    # we are gonna fit the entire row in SRAM
    row_start = tl.program_id(0) # each program handles one row
    row_step = tl.num_programs(0) # how many programs are launched in total

    # if 4 programs, then row_step = 4
    # if n_rows = 6,
    # pid 0 => row 0, 4
    # pid 1 => row 1, 5
    # pid 2 => row 2

    for row_idx in tl.range(row_start, n_rows, row_step, num_stages = num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE) # vector of size BLOCK_SIZE
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask = mask, other = float("-inf")) # shape (BLOCK_SIZE) which is roughly shape n_cols
        #......
        tl.store()







# def online_softmax(x: torch.Tensor) -> torch.Tensor:
#     """
#     Online (streaming) softmax: two passes per row.
#     Keeps a running max m and a running normalized sum s.
#     """
#     assert x.dim() == 2
#     R, C = x.shape
#     out = torch.empty_like(x)

#     for r in range(R):
#         # init as proper tensors on the right device/dtype
#         m = torch.tensor(float("-inf"), device=x.device, dtype=x.dtype)
#         s = torch.tensor(0.0, device=x.device, dtype=torch.float32)  # accumulate in fp32

#         row = x[r]
#         # pass 1: build (m, s)
#         for c in range(C):
#             v = row[c]
#             m_new = torch.maximum(m, v)  # use tensor op, not Python max
#             # rescale old sum to new max, then add current term
#             s = s * torch.exp((m - m_new).to(torch.float32)) + torch.exp((v - m_new).to(torch.float32))
#             m = m_new
#         # pass 2: normalize
#         out[r] = torch.exp(row - m) / s.to(row.dtype)
#     return out

# @triton.jit
# def _softmax_pass1_kernel(
#     x_ptr, stride_x_row,
#     m_ptr, s_ptr,
#     num_cols,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     row = tl.program_id(axis=0)
#     x_row_ptr = x_ptr + row * stride_x_row
#     cols = tl.arange(0, BLOCK_SIZE)
#     mask = cols < num_cols

#     x = tl.load(x_row_ptr + cols, mask=mask, other=float("-inf"))
#     m = tl.max(x, axis=0)
#     s = tl.sum(tl.exp(x - m), axis=0)

#     tl.store(m_ptr + row, m)
#     tl.store(s_ptr + row, s)

# @triton.jit
# def _softmax_pass2_kernel(
#     x_ptr, stride_x_row,
#     m_ptr, s_ptr,
#     y_ptr, stride_y_row,
#     num_cols,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     row = tl.program_id(axis=0)
#     x_row_ptr = x_ptr + row * stride_x_row
#     y_row_ptr = y_ptr + row * stride_y_row

#     cols = tl.arange(0, BLOCK_SIZE)
#     mask = cols < num_cols

#     x = tl.load(x_row_ptr + cols, mask=mask, other=float("-inf"))
#     m = tl.load(m_ptr + row)
#     s = tl.load(s_ptr + row).to(x.dtype)

#     y = tl.exp(x - m) / s
#     tl.store(y_row_ptr + cols, y, mask=mask)




# def softmax_online_triton(x: torch.Tensor) -> torch.Tensor:
#     assert x.dim() == 2, "only 2D tensors supported"
#     rows, cols = x.shape
#     y = torch.empty_like(x)

#     BLOCK_SIZE = triton.next_power_of_2(cols)
#     num_warps = 4
#     if BLOCK_SIZE > 2048:
#         num_warps = 8
#     if BLOCK_SIZE > 4096:
#         num_warps = 16

#     m_buf = torch.empty((rows,), device=x.device, dtype=x.dtype)
#     s_buf = torch.empty((rows,), device=x.device, dtype=torch.float32)

#     grid = (rows,)

#     _softmax_pass1_kernel[grid](
#         x, x.stride(0),
#         m_buf, s_buf,
#         cols,
#         BLOCK_SIZE=BLOCK_SIZE,
#         num_warps=num_warps,
#     )
#     _softmax_pass2_kernel[grid](
#         x, x.stride(0),
#         m_buf, s_buf,
#         y, y.stride(0),
#         cols,
#         BLOCK_SIZE=BLOCK_SIZE,
#         num_warps=num_warps,
#     )
#     return y



# def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device = DEVICE):
#     assert type(size) is tuple and len(size) == 2
#     torch.manual_seed(0)
#     x = torch.randn(size[0], size[1], device = DEVICE)
#     z_tri = softmax(x)
#     z_ref = torch.softmax(x, dim=1)

# import contextlib

# def time_ms(fn, iters=100, warmup=10):
#     for _ in range(warmup):
#         fn()
#         torch.cuda.synchronize()
#     t0 = time.perf_counter()
#     for _ in range(iters):
#         fn()
#         torch.cuda.synchronize()
#     t1 = time.perf_counter()
#     return (t1 - t0) * 1e3 / iters

# if __name__ == "__main__":
#     # do a small warmup to JIT-compile the Triton kernels
#     sample = torch.tensor([[1,2,3,4,5], [5,4,3,2,1]], dtype=torch.float32, device='cuda')
#     _ = softmax_online_triton(sample)  # compile
#     torch.cuda.synchronize()

#     # now measure fairly on something non-trivial
#     M, N = 1028, 1028
#     x = torch.randn(M, N, device='cuda', dtype=torch.float32)

#     # correctness (spot-check)
#     ref = F.softmax(x, dim=1)
#     tri = softmax_online_triton(x)
#     print("max abs diff (triton vs torch):", (ref - tri).abs().max().item())

#     # perf
#     ms_torch  = time_ms(lambda: F.softmax(x, dim=1))
#     ms_naive  = time_ms(lambda: naive_softmax(x))
#     # Python online is algorithmically fine but slow—do fewer iters to avoid waiting forever.
#     ms_online = time_ms(lambda: online_softmax(x), iters=5, warmup=2)
#     ms_triton = time_ms(lambda: softmax_online_triton(x))

#     # effective bandwidth (conservative ~3 passes over HBM)
#     def gbps(ms): return 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)

#     print(f"torch:  {ms_torch:.3f} ms  | {gbps(ms_torch):.1f} GB/s")
#     print(f"naive:  {ms_naive:.3f} ms  | {gbps(ms_naive):.1f} GB/s")
#     print(f"online: {ms_online:.3f} ms  | {gbps(ms_online):.1f} GB/s   (Python loop, for reference)")
#     print(f"triton: {ms_triton:.3f} ms  | {gbps(ms_triton):.1f} GB/s")