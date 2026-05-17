import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_input = input_ptr + row_idx * input_row_stride
    row_start_output = output_ptr + row_idx * output_row_stride

    # Load row in chunks of BLOCK_SIZE and compute max
    m = tl.full([], float('-inf'), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start_input + cols, mask=mask, other=float('-inf'))
        m = tl.maximum(m, tl.max(x, axis=0))

    # Compute sum of exp(x - max)
    s = tl.full([], 0.0, dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start_input + cols, mask=mask, other=float('-inf'))
        s += tl.sum(tl.exp(x.to(tl.float32) - m), axis=0)

    # Write output
    for off in range(0, n_cols, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(row_start_input + cols, mask=mask, other=float('-inf'))
        out = tl.exp(x.to(tl.float32) - m) / s
        tl.store(row_start_output + cols, out.to(x.dtype), mask=mask)


@triton.jit
def softmax_kernel_single(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride, output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel when n_cols <= BLOCK_SIZE (single pass per phase)."""
    row_idx = tl.program_id(0)
    row_start_input = input_ptr + row_idx * input_row_stride
    row_start_output = output_ptr + row_idx * output_row_stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(row_start_input + cols, mask=mask, other=float('-inf'))
    x_f32 = x.to(tl.float32)
    m = tl.max(x_f32, axis=0)
    numerator = tl.exp(x_f32 - m)
    denominator = tl.sum(numerator, axis=0)
    out = numerator / denominator
    tl.store(row_start_output + cols, out.to(x.dtype), mask=mask)


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    output = torch.empty_like(x)

    # Pick BLOCK_SIZE as next power of 2 >= n_cols, capped reasonably
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 65536:
        BLOCK_SIZE = 8192

    num_warps = 8 if BLOCK_SIZE >= 2048 else (4 if BLOCK_SIZE >= 1024 else 2)

    if BLOCK_SIZE >= n_cols:
        softmax_kernel_single[(n_rows,)](
            x, output, n_cols,
            x.stride(0), output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=1,
        )
    else:
        softmax_kernel[(n_rows,)](
            x, output, n_cols,
            x.stride(0), output.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=1,
        )
    return output


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return softmax_triton(x)


batch_size = 256
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
