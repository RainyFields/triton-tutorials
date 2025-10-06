import os
# set environment variable to enable CPU debugging, need to be set *before* trition is imported
os.environ["TRITON_DEBUG"] = "1"

import triton
import triton.language as tl
from IPython.core.debugger import set_trace


def test_pid_conds(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """
    Test if conditions on the program IDs (PIDs) are fullfilled.

    Args:
        conds (Str): String containing conditions to test. Multiple conditions are separated by commas. Each condition consists of an operator and a number.
        pid_0 (List): 1st program ID value in a single element list. Default is [0].
        pid_1 (List): 2nd program ID value in a single element list. Default is [0].
        pid_2 (List): 3rd program ID value in a single element list. Default is [0].
    
    Examples:
        '=0' -> check if pid_0 equals 0
        ',>1' -> check if pid_0 is greater than 0
        '>1,=0' -> check if pid_0 is greater than 1 and pid_1 equals 0
    
    Returns:
        bool: True if all conditions are fullfilled, False otherwise    
    """
    # extract PID values from lists:
    pids = pid_0[0], pid_1[0], pid_2[0]

    # remove spaces and split conditions by comma
    conds = conds.replace(" ", "").split(",")

    # valid operator
    valid_ops = ['<', '>', '>=', '<=', '=', '!=']

    # check each condition against corresponding PID
    for i, (cond, pid) in enumerate(zip(conds, pids)):
        if cond == '':
            continue # skip empty conditions
        
        # split condition into operator and number
        op, thresh = cond[0], cond[1:]
        if op not in valid_ops:
            raise ValueError(f"Invalid operator: {op}. Must be one of {valid_ops}.")

        # convert "=" to "==" for python evaluation
        op == "==" if op == "=" else op

        # evaluate condition
        if not eval(f'{pid} {op} {thresh}'):
            return False # condition not met
        
    return True # all conditions met


def check_tensors_gpu_ready(*tensors):
    """
    Check if all tensors are on the GPU and are contiguous.

    Args:
        *tensors (torch.Tensor): Variable number of torch tensors to check.
    
    Raises:
        Assertion error if any tensor is not on the GPU or not contiguous.
    """
    for tensor in tensors:
        assert tensor.is_contiguous(), "a tensor is not contiguous"
        # check GPU if not in simulator mode
        if not os.environ.get("TRITON_INTERPRET") == "1":
            assert tensor.is_cuda, "a tensor is not on GPU"


def print_if(txt, conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """
    Print a message if conditions on the program IDs (PIDs) are fullfilled.
    useful for debugging specific threads in GPU kernels

    Args:
        tex (Str): Text message to print.
        conds (Str): String containing conditions to test. Multiple conditions are separated by commas. Each condition consists of an operator and a number.
        pid_0 (List): 1st program ID value in a single element list. Default is [0].
        pid_1 (List): 2nd program ID value in a single element list. Default is [0].
        pid_2 (List): 3rd program ID value in a single element list. Default is [0].
    """
    if test_pid_conds(conds, pid_0, pid_1, pid_2):
        print(txt)


def breakpoint_if(conds, pid_0=[0], pid_1=[0], pid_2=[0]):
    """
    Trigger a breakpoint if conditions on the program IDs (PIDs) are fullfilled.
    useful for debugging specific threads in GPU kernels

    Args:
        conds (Str): String containing conditions to test. Multiple conditions are separated by commas. Each condition consists of an operator and a number.
        pid_0 (List): 1st program ID value in a single element list. Default is [0].
        pid_1 (List): 2nd program ID value in a single element list. Default is [0].
        pid_2 (List): 3rd program ID value in a single element list. Default is [0].
    """
    if test_pid_conds(conds, pid_0, pid_1, pid_2):
        set_trace()


