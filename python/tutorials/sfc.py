# %%
# Kernels
# -------

import torch

import triton
import triton.language as tl
import os
import re

from gilbert_d2xy import gilbert_d2xy

DTYPE = os.getenv("DTYPE", "bfloat16")
in_dtype = getattr(torch, DTYPE)
out_dtype = torch.float32

# For AMX
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 32

# Matmul kernel using the space curve filling approach in https://arxiv.org/abs/2601.16294v1,
# based on the generalized hilbert curve implementation from https://github.com/jakubcerveny/gilbert
#
# Each program computes a single output tile with the 2D coordinates derived from the precomputed SFC mapping.
# If `BLOCKING_FACTOR_K == 1`, then program handles all `BLOCKS_K = K // BLOCK_SIZE_K` blocks along the common dimension,
# otherwise the program performs a partial accumulation of the K blocks in the half-open interval:
#    [ ik * (BLOCKS_K // BLOCKING_FACTOR_K), (ik + 1) * (BLOCKS_K // BLOCKING_FACTOR_K) )
#
# Support for transposition and packing is the same as in the `matmul_kernel` above.
@triton.jit
def sfc_kernel(a_ptr, b_ptr, c_ptr, sfc_map_m_ptr, sfc_map_n_ptr, M, N, K, ik,
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
               BLOCK_SIZE_K: tl.constexpr,
               OUT_DTYPE: tl.constexpr,
               BLOCKING_FACTOR_K: tl.constexpr):
    BLOCKS_M = M // BLOCK_SIZE_M
    BLOCKS_N = N // BLOCK_SIZE_N
    BLOCKS_K = K // BLOCK_SIZE_K
    BLOCKS_K_PER_PROG = BLOCKS_K // BLOCKING_FACTOR_K

    pid = tl.program_id(axis=0)
    block_m = tl.load(sfc_map_m_ptr + pid)
    block_n = tl.load(sfc_map_n_ptr + pid)
    block_k = ik * BLOCKS_K_PER_PROG

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(BLOCKS_M, BLOCKS_K, BLOCK_SIZE_K, BLOCK_SIZE_M),
                                    strides=(BLOCK_SIZE_M * K, BLOCK_SIZE_M * BLOCK_SIZE_K, BLOCK_SIZE_M, 1),
                                    offsets=(block_m, block_k, 0, 0), block_shape=(1, 1, BLOCK_SIZE_K, BLOCK_SIZE_M),
                                    order=(3, 2, 1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(BLOCKS_N, BLOCKS_K, BLOCK_SIZE_N, BLOCK_SIZE_K),
                                    strides=(BLOCK_SIZE_N * K, BLOCK_SIZE_N * BLOCK_SIZE_K, BLOCK_SIZE_K, 1),
                                    offsets=(block_n, block_k, 0, 0), block_shape=(1, 1, BLOCK_SIZE_N, BLOCK_SIZE_K),
                                    order=(3, 2, 1, 0))
    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(BLOCKS_N, BLOCKS_M, BLOCK_SIZE_N, BLOCK_SIZE_M),
                                    strides=(BLOCK_SIZE_N * M, BLOCK_SIZE_N * BLOCK_SIZE_M, BLOCK_SIZE_M, 1),
                                    offsets=(block_n, block_m, 0, 0),
                                    block_shape=(1, 1, BLOCK_SIZE_N, BLOCK_SIZE_M), order=(3, 2, 1, 0))

    if ik == 0:
        c0 = tl.zeros((1, 1, BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=OUT_DTYPE)
        tl.store(c_block_ptr, c0)
    
    c = tl.load(c_block_ptr).reshape((BLOCK_SIZE_N, BLOCK_SIZE_M))

    for _ in range(BLOCKS_K_PER_PROG):
        a = tl.load(a_block_ptr).reshape((BLOCK_SIZE_K, BLOCK_SIZE_M))
        b = tl.load(b_block_ptr).reshape((BLOCK_SIZE_N, BLOCK_SIZE_K))

        c += tl.dot(b, a, out_dtype=OUT_DTYPE)

        a_block_ptr = tl.advance(a_block_ptr, (0, 1, 0, 0))
        b_block_ptr = tl.advance(b_block_ptr, (0, 1, 0, 0))

    tl.store(c_block_ptr, c.reshape((1, 1, BLOCK_SIZE_N, BLOCK_SIZE_M)))


def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
           sfc_map_m: torch.Tensor, sfc_map_n: torch.Tensor, M, N, K, blocking_factor_k=1):
    #TODO: Currently masked load is not supported yet.
    assert (M % BLOCK_SIZE_M == 0) and (N % BLOCK_SIZE_N == 0) and (
        K % BLOCK_SIZE_K == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    # 1D launch kernel where each block gets its own program.
    grid = ((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N), )
    assert sfc_map_m is not None and sfc_map_n is not None
    for ik in range(blocking_factor_k):
        sfc_kernel[grid](
            a, b, c,  #
            sfc_map_m, sfc_map_n, #
            M, N, K,  #
            ik,  #
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
            OUT_DTYPE=tl.float32,  #
            BLOCKING_FACTOR_K=blocking_factor_k)
    return c

def make_sfc_tensors(x, y, dtype=torch.int32, device='cpu'):
    gilbert = [gilbert_d2xy(i, x, y) for i in range(x * y)]
    return (torch.tensor([x for (x, _) in gilbert], dtype=dtype, device=device),
            torch.tensor([y for (_, y) in gilbert], dtype=dtype, device=device))


# %%
# Unit Test
# ---------

torch.manual_seed(0)

triton.runtime.driver.set_active_to_cpu()

M, N, K = 512, 256, 1024

a = torch.rand((M, K), device='cpu', dtype=in_dtype)
b = torch.rand((K, N), device='cpu', dtype=in_dtype)
# a = torch.tril(torch.ones((M, K), device='cpu', dtype=in_dtype))
# b = torch.triu(torch.ones((K, N), device='cpu', dtype=in_dtype))

torch_output = torch.matmul(a.to(out_dtype), b.to(out_dtype))
rtol = 0
sfc_map_m, sfc_map_n = make_sfc_tensors(M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)

ap = a.reshape((M // BLOCK_SIZE_M, BLOCK_SIZE_M, K // BLOCK_SIZE_K, BLOCK_SIZE_K)).permute((0, 2, 3, 1)).contiguous()  # M, K, k, m
bp = b.reshape((K // BLOCK_SIZE_K, BLOCK_SIZE_K, N // BLOCK_SIZE_N, BLOCK_SIZE_N)).permute((2, 0, 3, 1)).contiguous()  # N, K, n, k
cp = torch.empty((N // BLOCK_SIZE_N, M // BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_M), device='cpu', dtype=out_dtype)    # N, M, n, m
triton_output_p = matmul(ap, bp, cp, sfc_map_m, sfc_map_n, M=M, N=N, K=K, blocking_factor_k=2)
triton_output = triton_output_p.permute((1, 3, 0, 2)).reshape(M, N)

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ TritonCPU pre-packed SFC and TorchCPU match")
else:
    print("❌ TritonCPU pre-packed SFC and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')
    assert False


# %%
# Benchmark
# ---------

def calculate_layers(M, N, K, dtype: torch.dtype, low_limit_in_gb=5.0):
    size_total = lambda n_layers: dtype.itemsize * n_layers * (M * K + K * N + M * N) / (1024.0 * 1024.0 * 1024.0)

    n_layers = 1
    while size_total(n_layers) < low_limit_in_gb:
        n_layers += 1
    
    return n_layers

def encode_triton_provider(sfc_bfk, dtype):
    assert dtype == 'float32' or dtype == 'bfloat16' or dtype == 'float16'
    return f"triton-cpu{f'-sfc{sfc_bfk}' if sfc_bfk > 0 else ''}-{dtype}"


def encode_torch_provider(dtype):
    assert dtype == 'float32' or dtype == 'bfloat16' or dtype == 'float16'
    return f"torch-cpu-native-{dtype}"


def decode_provider(provider):
    if '-bfloat16' in provider:
        dtype = torch.bfloat16
    if '-float16' in provider:
        dtype = torch.float16
    elif '-float32' in provider:
        dtype = torch.float32

    if 'triton-cpu' in provider:
        backend = 'triton-cpu'
    elif 'torch-cpu-native' in provider:
        backend = 'torch-cpu-native'

    sfc_bfk = 0
    if m := re.search(r'-sfc(\d+)', provider):
        sfc_bfk = int(m.group(1))

    return backend, sfc_bfk, dtype

SFC_BFK_OPTS = [1, 2, 4, 8]
DTYPE_OPTS = [str(in_dtype)[len('torch.'):]]
LINE_VALS = [
    encode_triton_provider(sfc_bfk, dtype)
    for dtype in DTYPE_OPTS
    for sfc_bfk in SFC_BFK_OPTS
] + [encode_torch_provider(dtype) for dtype in DTYPE_OPTS]
LINE_NAMES = LINE_VALS
LINE_STYLES = None

default_num_threads = torch.get_num_threads()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[2048],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel='GFLOPS',  # Label name for the y-axis.
        plot_name=
        # Name for the plot. Used also as a file name for saving the plot.
        f'matmul-performance-{DTYPE_OPTS[0]} (BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, BLOCK_SIZE_K={BLOCK_SIZE_K}',
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(M, N, K, provider):
    device = 'cpu'
    backend, sfc_bfk, dtype = decode_provider(provider)
    assert dtype.is_floating_point

    # Make sure we have enough independent matmuls so that tensors are always cold.
    n_layers = calculate_layers(M, N, K, dtype)
    a = torch.randn((n_layers, M, K), device=device, dtype=dtype)
    b = torch.randn((n_layers, K, N), device=device, dtype=dtype)

    torch.set_num_threads(default_num_threads)

    if backend == 'torch-cpu-native':
        c = torch.zeros((n_layers, M, N), device=device, dtype=dtype)
    elif backend == 'triton-cpu':
        sfc_map_m, sfc_map_n = make_sfc_tensors(M // BLOCK_SIZE_M, N // BLOCK_SIZE_N) if sfc_bfk > 0 else (None, None)
        # NB: Packing is done outside of the benchmark loop
        apack = a.reshape((n_layers, M // BLOCK_SIZE_M, BLOCK_SIZE_M, K // BLOCK_SIZE_K, BLOCK_SIZE_K)).permute((0, 1, 3, 4, 2)).contiguous()  # layer, M, K, k, m
        bpack = b.reshape((n_layers, K // BLOCK_SIZE_K, BLOCK_SIZE_K, N // BLOCK_SIZE_N, BLOCK_SIZE_N)).permute((0, 3, 1, 4, 2)).contiguous()  # layer, N, K, n, k
        c = torch.zeros((n_layers, M, N), device=device, dtype=out_dtype)

    quantiles = [0.5, 0.2, 0.8]
    if backend == 'torch-cpu-native':
        # NB: Take with a grain of salt: 1) Input and output element types are not the same, 2) batch matmul might do something smarter than a sequential loop
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(a, b, out=c), quantiles=quantiles, rep=10)
    elif backend == 'triton-cpu':
        def doit():
            for l in range(n_layers):
                matmul(apack[l], bpack[l], c[l], sfc_map_m, sfc_map_n, M, N, K, blocking_factor_k=sfc_bfk)
        ms, min_ms, max_ms = triton.testing.do_bench(doit, quantiles=quantiles, measure_time_with_hooks=True, rep=10)
    perf = lambda ms: 2 * n_layers * M * N * K * 1e-9 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(print_data=True, show_plots=True)
