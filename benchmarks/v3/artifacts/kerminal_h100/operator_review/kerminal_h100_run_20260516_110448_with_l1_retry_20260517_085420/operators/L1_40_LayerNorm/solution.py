import torch
import torch.nn as nn
import triton
import triton.language as tl

# LayerNorm: normalize over last 3 dims (64*256*256 = 4194304 elements per instance)
# With only 16 instances, use two-kernel approach to maximize parallelism:
#   1) Partial reduction: each block computes partial sum and sum_sq for a chunk
#   2) Normalize: each block normalizes its chunk using the global mean/var

@triton.jit
def partial_reduce_kernel(
    X_ptr,
    partial_sum_ptr,
    partial_sum_sq_ptr,
    N,
    stride,
    BLOCKS_PER_INSTANCE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    instance_id = tl.program_id(0)
    block_id = tl.program_id(1)
    
    X_ptr = X_ptr + instance_id * stride
    
    start = block_id * BLOCK_SIZE
    cols = start + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    s = tl.sum(x, axis=0)
    sq = tl.sum(x * x, axis=0)
    
    out_idx = instance_id * BLOCKS_PER_INSTANCE + block_id
    tl.store(partial_sum_ptr + out_idx, s)
    tl.store(partial_sum_sq_ptr + out_idx, sq)


@triton.jit
def normalize_kernel(
    X_ptr, Y_ptr, W_ptr, B_ptr,
    partial_sum_ptr,
    partial_sum_sq_ptr,
    N,
    stride,
    BLOCKS_PER_INSTANCE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    REDUCE_BLOCK: tl.constexpr,
):
    instance_id = tl.program_id(0)
    block_id = tl.program_id(1)
    
    base = instance_id * BLOCKS_PER_INSTANCE
    reduce_offsets = tl.arange(0, REDUCE_BLOCK)
    reduce_mask = reduce_offsets < BLOCKS_PER_INSTANCE
    
    ps = tl.load(partial_sum_ptr + base + reduce_offsets, mask=reduce_mask, other=0.0)
    psq = tl.load(partial_sum_sq_ptr + base + reduce_offsets, mask=reduce_mask, other=0.0)
    
    total_sum = tl.sum(ps, axis=0)
    total_sum_sq = tl.sum(psq, axis=0)
    
    n_float = N.to(tl.float32)
    mean = total_sum / n_float
    var = total_sum_sq / n_float - mean * mean
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    
    X_ptr = X_ptr + instance_id * stride
    Y_ptr = Y_ptr + instance_id * stride
    
    start = block_id * BLOCK_SIZE
    cols = start + tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    y = (x - mean) * rstd * w + b
    tl.store(Y_ptr + cols, y.to(x.dtype), mask=mask)


class Model(nn.Module):
    def __init__(self, normalized_shape: tuple):
        super().__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_shape = self.ln.normalized_shape
        N = 1
        for s in normalized_shape:
            N *= s
        
        num_instances = x.numel() // N
        x_flat = x.contiguous()
        y = torch.empty_like(x_flat)
        
        weight = self.ln.weight
        bias = self.ln.bias
        
        BLOCK_SIZE = 8192
        blocks_per_instance = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        REDUCE_BLOCK = triton.next_power_of_2(blocks_per_instance)
        
        partial_sum = torch.empty(num_instances, blocks_per_instance, dtype=torch.float32, device=x.device)
        partial_sum_sq = torch.empty(num_instances, blocks_per_instance, dtype=torch.float32, device=x.device)
        
        grid1 = (num_instances, blocks_per_instance)
        partial_reduce_kernel[grid1](
            x_flat, partial_sum, partial_sum_sq,
            N, N,
            BLOCKS_PER_INSTANCE=blocks_per_instance,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        grid2 = (num_instances, blocks_per_instance)
        normalize_kernel[grid2](
            x_flat, y, weight, bias,
            partial_sum, partial_sum_sq,
            N, N,
            BLOCKS_PER_INSTANCE=blocks_per_instance,
            BLOCK_SIZE=BLOCK_SIZE,
            REDUCE_BLOCK=REDUCE_BLOCK,
        )
        
        return y.view_as(x)


batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]
