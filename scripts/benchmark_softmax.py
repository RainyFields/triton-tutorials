import torch
import triton
import triton.language as tl

from softmax import softmax, naive_softmax

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # Argument names to use as an x-axis
        x_vals=[2**i for i in range(2, 16)],  # Different possible values for `x_name`.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch-native','torch-jit'],  # Possible values for `line_arg`.
        line_names=['triton', 'torch-native','torch-jit'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-'), ('green','--')],  # Line styles.
        ylabel=['GB/s'],  # Label name for the y-axis.
        plot_name='softmax-performance',  # Name for the plot. Used also as a
        args={'M': 4096},  # Values for function arguments not in `x_names` and `y_name`.
        # M is the number of rows, N is the number of columns
    ))


def benchmark(M, N, provider):
    x = torch.randn((M, N), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8] # median, 20%(fast) and 80%(slow)
    if provider == 'torch-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=1), quantiles=quantiles)
    if provider == 'torch-jit':
        x_jit = torch.jit.trace(torch.nn.functional.softmax, x)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x_jit(x), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: softmax(x), quantiles=quantiles)
    
    # report effective memory bandwidth 
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


from pathlib import Path
benchmark.run(print_data=True, show_plots=True, save_path=Path('../results/'))