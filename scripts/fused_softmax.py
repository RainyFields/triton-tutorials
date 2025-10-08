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
NUM_REGS = properties["max_num_regs"] # each program share those registers
TOTAL_SRAM_PER_SM = properties["max_shared_mem"] # how many kernel we can put on each SM
WARP_SIZE = properties["warp_size"] # usually 32 for NVIDIA GPU; the smallest group of GPU cores


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

    # how many warps do we want to use to a single SM
    num_warps = 4
    # if rows are really big, we can use more warps
    if BLOCK_SIZE > 2048:
        num_warps = 8
    if BLOCK_SIZE > 4096:
        num_warps = 16 # 

    # how many stages to use in the pipeline
    num_stages = 4 if TOTAL_SRAM_PER_SM >= 16384 else 2

    y = torch.empty_like(x) # output tensor

    # warmup: run a dummy compilation with the kernel; convert to TritonIR => PTX => CUDA
    kernel = _softmax_kernel.warmup(
        x, y, n_rows, n_cols, 
        BLOCK_SIZE = BLOCK_SIZE,
        num_stages = num_stages,
        num_warps = num_warps,
        grid=(1, ))
    kernel._init_handles() # the kernel function now has a handle to the compiled CUDA function
    
    n_regs_per_register = kernel.n_regs # tells us how many registers are used by each thread ! I believe this should be for each thread, not for each program
    sram_needed_per_program = kernel.metadata.shared # tell us how much SRAM is used by each program

    # let's calculate how many programs we can launch per SM
    # each thread might use 32 registers, WARP_SIZE = 32 (32 threads run in parallel)
    # and num_warps = 8
    # so each program need 32*32*8 registers total
    # 65536 // (32*32*8) = 8 programs can run on each SM
    reg_occupancy = NUM_REGS // (n_regs_per_register * WARP_SIZE * num_warps)

    # how many programs can fit in SRAM per SM
    sram_occupancy = TOTAL_SRAM_PER_SM // sram_needed_per_program

    # identify the bottleneck 
    programs_per_sm = min(reg_occupancy, sram_occupancy) # one of the factors will be limitting
    # how many programs can we launch in total
    # NUM_SM * programs_per_sm = total number of programs we can launch 
    num_programs = min(NUM_SM * programs_per_sm, n_rows) # we don't consider if there are more rows

    grid = (num_programs,) # warmup does not necessarily require launch 3-axis in total
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
    PID = tl.program_id(0) # each program handles one row
    row_step = tl.num_programs(0) # how many programs are launched in total; if row_step is smaller than n_rows, we are all good; if row_step is bigger than n_rows, then we have to jump over some rows
    
    # if there are 4 programs, then row_step = 4
    # if n_rows = 6,
    # pid 0 => row 0, 4 (pid 0 += row_step)
    # pid 1 => row 1, 5 (pid 1 += row_step)
    # pid 2 => row 2, 6 (pid 2 += row_step)
    # pid 3 => row 3

    # each program handles every row_step rows; num_stages tells the compiler to pipeline memory operations across iterations 
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
        tl.store(output_ptr + row_idx * output_row_stride, sm_row, mask=mask)





