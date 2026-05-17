import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemv_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    K: tl.constexpr, N: tl.constexpr,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x_vals = tl.load(x_ptr + k_offs, mask=k_mask, other=0.0)
        # W is (N, K) stored row-major (nn.Linear weight)
        w_vals = tl.load(
            w_ptr + n_offs[:, None] * K + k_offs[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    bias = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0)
    acc += bias
    tl.store(out_ptr + n_offs, acc, mask=n_mask)


@triton.jit
def softmax_kernel(
    inp_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(inp_ptr + offs, mask=mask, other=float('-inf'))
    x = x - tl.max(x, axis=0)
    ex = tl.exp(x)
    ex = tl.where(mask, ex, 0.0)
    s = tl.sum(ex, axis=0)
    out = ex / s
    tl.store(out_ptr + offs, out, mask=mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # In eval mode, dropout is identity
        # mean(x @ W^T + b, dim=0) = mean(x, dim=0) @ W^T + b
        if not self.training:
            x_mean = x.mean(dim=0).contiguous()  # (in_features,)
            K = self.in_features
            N = self.out_features
            out = torch.empty(N, device=x.device, dtype=torch.float32)
            
            BLOCK_N = 64
            BLOCK_K = 128
            grid = ((N + BLOCK_N - 1) // BLOCK_N,)
            gemv_bias_kernel[grid](
                x_mean, self.matmul.weight, self.matmul.bias, out,
                K=K, N=N,
                BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            )
            
            # Softmax on 1D vector
            result = torch.empty(N, device=x.device, dtype=torch.float32)
            BLOCK = triton.next_power_of_2(N)
            softmax_kernel[(1,)](out, result, N=N, BLOCK=BLOCK)
            return result.unsqueeze(0)
        else:
            # Training path with dropout
            x = torch.addmm(self.matmul.bias, x, self.matmul.weight.t())
            x = self.dropout(x)
            x = torch.mean(x, dim=0, keepdim=True)
            x = torch.softmax(x, dim=1)
            return x


batch_size = 128
in_features = 4096
out_features = 4096
dropout_p = 0.2

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, dropout_p]
