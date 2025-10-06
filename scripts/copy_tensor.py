import os
# set environment variable to enable CPU debugging, need to be set *before* trition is imported
os.environ["TRITON_DEBUG"] = "1"

import torch
import triton
import triton.language as tl

from utils import check_tensors_gpu_ready, print_if

def cdiv(n, d):
    """
    Ceiling division between two numbers.
    Args:
        n (int): Numerator
        d (int): Denominator
    Returns:
        ceiling divison result
    """
    return (n + d - 1) // d

def copy(x, bs, kernel_fn):
    """
    Launch a Triton kernel to copy data from one GPU tensor to another

    Args:
        x (torch.Tensor): Input tensor to copy from
        bs (int): Block size (number of elements processed per GPU thread block)
        kernel_fn (Callable): Triton kernel function to execute
    
    Returns:
        z: New tensor containing copied data
    """
    # create output tensor with same properties as input
    z = torch.zeros_like(x)

    # verify tensors are GPU ready
    check_tensors_gpu_ready(x, z)

    # calculate grid dimensions for GPU execution
    n = x.numel()
    n_blocks = cdiv(n, bs)
    grid = (n_blocks, )

    # launch the kernel on GPU 
    kernel_fn[grid](x, z, n, BLOCK_SIZE=bs)
    return z 

@triton.jit # decorator converts the Python function into GPU code
def copy_kernel(x_ptr, z_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    Important notes:
        only a limited set of operations are allowed inside GPU kernels
        - basic arithmetic and logical operations
        - python print() and debugging tools like breakpoints are not allowed
        - use specialized Triton functions for GPU operations 

    Triton kernel to copy data from one tensor to another

    Args:
        x_ptr (tl.pointer): Pointer to input tensor (Triton automatically converts tensor to pointer)
        z_ptr (tl.pointer): Pointer to output tensor
        n_elements (int): Number of elements in the tensors
        BLOCK_SIZE (int): Number of elements processed per GPU thread block. marked as compile time constant with tl.constexpr
    """
    
    # block_id = tl.program_id(0) # get the current block ID
    # offs = tl.arange(0, BLOCK_SIZE) # Create offsets [0,...BLOCK_SIZE-1]
    # mask = offs < n_elements # Create a mask to handle out-of-bounds accesses

    # x = tl.load(x_ptr + offs, mask) # load input values
    # tl.store(z_ptr + offs, x, mask) # store values to output
    # print_if(f'pid = {pid} | offs = {offs}, mask = {mask}, x = {x}', '')
    
    pid = tl.program_id(0)  # Get current block ID
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # Calculate correct offsets for this block
    mask = offs < n_elements  # Prevent out-of-bounds access
    x = tl.load(x_ptr + offs, mask)  # Load input values
    tl.store(z_ptr + offs, x, mask)  # Store to output

    

x = torch.tensor([1,2,3,4,5,6], device='cuda', dtype=torch.float32)
y = torch.tensor([0,1,0,1,0,1], device='cuda', dtype=torch.float32)
z = copy(x , bs = 2, kernel_fn = copy_kernel)

# Sanity check
assert torch.allclose(z, x), f"copy failed: got {z}, expected {x}"
print("OK:", z)


    


