# code eager softmax in PyTorch, Triton

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

def naive_softmax(x: torch.Tensor)->torch.Tensor:
    """
    eager mode softmax
    """
    x_max = x.max(dim=1)[0] # pass 1
    safe_x = x - x_max[:, None] # pass 2 
    numerator = torch.exp(safe_x) # pass 3 
    denominator = numerator.sum(dim = 1) # pass 4
    sm_out = numerator / denominator[:,None] # pass 5
    return sm_out 

@triton.jit
def _softmax_fwd_kernel(
    output_ptr, 
    stride_output_row,
    input_ptr,
    stride_input_row,
    num_cols,
    block_size: tl.constexpr,):
    # setup input ptrs
    row_index = tl.program_id(0)

    row_start_ptr = input_ptr + row_index * stride_input_row
    col_offsets = tl.arange(0,block_size)
    input_pointers = row_start_ptr + col_offsets
    row_mask = col_offsets < num_cols

    # move to SRAM
    row = tl.load(input_pointers, mask = row_mask, other = float("-inf")) 
    # why -inf? => exp(-inf) don't contribute

    # softmax itself
    row_max = tl.max(row, axis=0)
    safe_row = row - row_max # numerical stability, to avoid overflow
    numerator = tl.exp(safe_row)
    denominator = tl.sum(numerator, axis=0)
    sm_row = numerator / denominator

    # write back to HBM
    output_row_start_ptr = output_ptr + row_index * stride_output_row
    output_pointers = output_row_start_ptr + col_offsets
    tl.store(output_pointers, sm_row, mask = row_mask)


def softmax(x: torch.Tensor)->torch.Tensor:
    """
    triton  impl of softmax, fwd pass only
    """
    rows, cols = x.shape
    assert x.dim() == 2, "only 2d tensor supported"

    # each program handles one row => row_index = tl.program_id(0) in kernel
    # within each row, we process up to block_size elements in parallel
    block_size = triton.next_power_of_2(cols) # block size is always larger than cols
    # block size in triton is different from CUDA block size
    # in triton, block size is a logical vector length inside the program
    # these vector lanes are mapped onto the actual CUDA threads by Triton
    # using num_warps to decide how many threads per program run in parallel. 

    num_warps = 4 # *32 thread each warp;
    # each warp has 32 threads, so num_warps=4 means 128 threads per block
    # with block size = 1024, each thread executes 1024/128 lanes
    # each thread can execute multiple lanes sequentially, in this case, 8;
    if block_size > 2047: #2048
        num_warps = 8
    if block_size > 4095: #4096
        num_warps = 16
    
    grid = (rows,)

    # allocate our output buffer
    sm_out = torch.empty_like(x)
    _softmax_fwd_kernel[grid](
        sm_out,
        sm_out.stride(0), # how many elements you jump in memory to move one row forward
        x, 
        x.stride(0),
        cols,
        block_size=block_size,
        num_warps=num_warps, # triton reserved keywords (num_warps, num_stages, num_ctas, enable_warp_specilization)
    )
    return sm_out
    

if __name__ == "__main__":
    
    sample = torch.tensor([[1,2,3,4,5], [5,4,3,2,1]], dtype=torch.float32, device='cuda')
    ref_out = F.softmax(sample, dim=1) # rowwise softmax
    print(f"{ref_out=}")

    eager_out = naive_softmax(sample)
    print(f"{eager_out=}")

    triton_out = softmax(sample)
    print(f"{triton_out=}")