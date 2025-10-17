# code online softmax
# reduce the numer of passes to 2
import time
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else torch.device('cpu')


# fetch properties of the current device
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"] # number of SMs on the GPU
NUM_REGS = properties["max_num_regs"] 
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] 
WARP_SIZE = properties["warpSize"] # usually 32 for NVIDIA GPU; the smallest group of GPU cores


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

    # in total: 8MN + 4M memory operations 



# wrapper
def softmax(x):
    assert x.ndim == 2 # only with 2D tensors
    assert x.is_contiguous() # only support contiguous tensors
    n_rows, n_cols = x.shape

    # assume each row fits with in SRAM
    BLOCK_SIZE = triton.next_power_of_2(n_cols) 

    num_warps = 4
    # if rows are really big, we can use more warps
    if BLOCK_SIZE > 2048:
        num_warps = 8
    if BLOCK_SIZE > 4096:
        num_warps = 16 # 

    # how many stages to use in the pipeline
    # bigger SRAM => bigger block size => more stages
    num_stages = 4 if TOTAL_SRAM_PER_SM >= 16384 else 2

    y = torch.empty_like(x) # output tensor

    # warmup: run a dummy compilation with the kernel; convert to TritonIR => PTX => CUDA
    kernel = _softmax_kernel.warmup(
        x, y, x.stride(0), y.stride(0),
        n_rows, n_cols, 
        BLOCK_SIZE = BLOCK_SIZE,
        num_stages = num_stages,
        num_warps = num_warps,
        grid=(1, ))
    # the kernel function now has a handle to the compiled CUDA function
    kernel._init_handles() 
    
    # how many registers are used by each thread
    n_regs_per_register = kernel.n_regs 
    # how much SRAM is used by each program
    sram_needed_per_program = kernel.metadata.shared 
    # print(f"SRAM needed per program: {sram_needed_per_program}")
    # print(f"Number of registers per thread: {n_regs_per_register}")

    # let's calculate how many programs we can launch per SM
    # each thread might use 32 registers, 
    # WARP_SIZE = 32 (32 threads run in parallel)
    # and num_warps = 8
    # so each program need 32*32*8 registers total
    # 65536 // (32*32*8) = 8 programs can run on each SM
    reg_occupancy = NUM_REGS // (n_regs_per_register * WARP_SIZE * num_warps)

    # how many programs can fit in SRAM per SM
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program
    # print(f"reg occupancy: {reg_occupancy}, sram occupancy: {sram_occupancy}")

    # identify the bottleneck 
    programs_per_sm = min(reg_occupancy, sram_occupancy) # one of the factors will be limitting
    # how many programs can we launch in total
    # NUM_SM * programs_per_sm = total number of programs we can launch 
    num_programs = min(NUM_SM * programs_per_sm, n_rows) # we don't consider if there are more rows

    grid = (num_programs, 1, 1) # !!!!warmup does necessarily require launch 3-axis in total

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
    PID = tl.program_id(0) # each program handles one row
    row_step = tl.num_programs(0) # how many programs are launched in total; => the grid being launched
    # if n_rows is smaller than row_step, then some programs will be idle
    # if n_rows is larger than row_step, then some programs will handle multiple rows

    # if there are 4 programs, then row_step = 4
    # if n_rows = 6,
    # pid 0 => row 0, 4 (pid 0 += row_step)
    # pid 1 => row 1, 5 (pid 1 += row_step)
    # pid 2 => row 2, 6 (pid 2 += row_step)
    # pid 3 => row 3

    # each program handles every row_step rows; 
    # num_stages tells the compiler to pipeline memory operations across iterations 
    for row_idx in tl.range(PID, n_rows, row_step, num_stages = num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE) # vector of size BLOCK_SIZE, a bit bigger than n_cols
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols

        # fuse mutliple operations within one pass over the data
        # the only time we read from memory
        row = tl.load(input_ptrs, mask = mask, other = float("-inf")) # shape (BLOCK_SIZE) which is roughly shape n_cols
        row_minus_max = row - tl.max(row, axis=0) # shape (BLOCK_SIZE)
        numerator = tl.exp(row_minus_max) # shape (BLOCK_SIZE)
        denominator = tl.sum(numerator, axis=0) # shape ()
        sm_row = numerator / denominator # shape (BLOCK_SIZE)

        # the only time we write to memory
        output_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_start_ptr + col_offsets
        tl.store(output_ptrs, sm_row, mask=mask)


# # test when there is no pipelining
# @triton.jit
# def _softmax_kernel(
#     input_ptr, output_ptr,
#     input_row_stride, output_row_stride,
#     n_rows, n_cols,
#     BLOCK_SIZE: tl.constexpr,
#     num_stages: tl.constexpr,
# ):
   
#     PID = tl.program_id(0) 
#     if PID < n_rows:
#         row_idx = PID
#         row_start_ptr = input_ptr + row_idx * input_row_stride
#         col_offsets = tl.arange(0, BLOCK_SIZE) # vector of size BLOCK_SIZE, a bit bigger than n_cols
#         input_ptrs = row_start_ptr + col_offsets

#         mask = col_offsets < n_cols

#         row = tl.load(input_ptrs, mask = mask, other = float("-inf")) # shape (BLOCK_SIZE) which is roughly shape n_cols
#         row_minus_max = row - tl.max(row, axis=0) # shape (BLOCK_SIZE)
#         numerator = tl.exp(row_minus_max) # shape (BLOCK_SIZE)
#         denominator = tl.sum(numerator, axis=0) # shape ()
#         sm_row = numerator / denominator # shape (BLOCK_SIZE)

#         # the only time we write to memory
#         output_start_ptr = output_ptr + row_idx * output_row_stride
#         output_ptrs = output_start_ptr + col_offsets
#         tl.store(output_ptrs, sm_row, mask=mask)




def test_softmax_kernel(size: tuple, atol=1e-3, rtol=1e-3, device = DEVICE):
    assert type(size) == tuple and len(size) == 2
    torch.manual_seed(0)
    x = torch.randn(size[0], size[1], device = DEVICE)
    z_tri = softmax(x)
    z_ref = torch.softmax(x, dim=-1)
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print("PASSED")


def check_equal(x, atol=1e-3, rtol=1e-3):
    y_torch = torch.softmax(x, dim=-1)
    y_tri   = softmax(x)
    torch.testing.assert_close(y_tri, y_torch, atol=atol, rtol=rtol)

@torch.no_grad()
def error_metrics(x):
    y_torch = torch.softmax(x, dim=-1)
    y_tri   = softmax(x)
    abs_err = (y_tri - y_torch).abs()
    rel_err = abs_err / y_torch.abs().clamp_min(1e-12)
    return {
        "max_abs": abs_err.max().item(),
        "mean_abs": abs_err.mean().item(),
        "max_rel": rel_err.max().item(),
        "mean_rel": rel_err.mean().item(),
        # sanity checks:
        "row_sum_max_abs_diff": (y_tri.sum(-1) - 1.0).abs().max().item(),
        "has_nan_triton": torch.isnan(y_tri).any().item(),
        "has_nan_torch": torch.isnan(y_torch).any().item(),
    }

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # Argument names to use as an x-axis
        x_vals=[128*i for i in range(2, 100)],  # Different possible values for `x_name`.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel=['GB/s'],  # Label name for the y-axis.
        plot_name='softmax-performance',  # Name for the plot. Used also as a
        args={'M': 4096},  # Values for function arguments not in `x_names` and `y_name`.
        # M is the number of rows, N is the number of columns
    ))
 
def benchmark(M: int, N: int, provider: str, do_check: bool=False, atol=1e-6, rtol=1e-6):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)

    if do_check and provider == "triton":
        # compare Triton vs Torch
        try:
            torch.testing.assert_close(softmax(x), torch.softmax(x, dim=-1), atol=atol, rtol=rtol)
        except AssertionError as e:
            metrics = error_metrics(x)
            print("[WARN] correctness drift:", metrics)

    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=-1))
    elif provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    # only read, write each once 
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms*1e-3)
    return gbps(ms)



if __name__ == "__main__":
    # always run unit-tests
    test_softmax_kernel((1823, 781))

    # only run benchmarks when explicitly intended
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        print("running benchmark...")
        benchmark.run(save_path = "/work/lei/triton-tutorials/results/", print_data = False)

