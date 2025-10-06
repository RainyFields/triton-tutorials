import os
import torch
import triton
import triton.language as tl

# ===== Helper Functions for Computing Memory Offsets and Masks =====

@triton.jit
def get_1d_offset(size, n_prev_chunks):
    """
    Calculate 1D memory offsets for a given chunk size and position.

    Args:
        size: Size of the current chunk
        n_prev_chunks: Number of previous chunks (used for position)

    Returns:
        Array of offsets for the current chunk
    """
    return n_prev_chunks * size + tl.arange(0, size)

@triton.jit
def get_2d_offset(offs_0, offs_1, stride_0, stride_1=1):
    """
    Calculate 2D memory offsets for matrix operations.

    Args:
        offs_0, offs_1: Offsets in first and second dimensions
        stride_0, stride_1: Stride values for memory layout

    Returns:
        2D array of memory offsets
    """
    return tl.expand_dims(offs_0, 1)*stride_0 + tl.expand_dims(offs_1, 0)*stride_1

@triton.jit
def get_1d_mask(offs, max):
    """
    Create a mask for boundary checking in 1D.

    Args:
        offs: Current offsets
        max: Maximum valid offset

    Returns:
        Boolean mask indicating valid positions
    """
    return offs < max

@triton.jit
def get_2d_mask(offs_0, offs_1, max_0, max_1):
    """
    Create a mask for boundary checking in 2D.

    Args:
        offs_0, offs_1: Current offsets in both dimensions
        max_0, max_1: Maximum valid offsets

    Returns:
        Boolean mask indicating valid positions in 2D
    """
    return (tl.expand_dims(offs_0, 1) < max_0) & (tl.expand_dims(offs_1, 0) < max_1)