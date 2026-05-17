import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemv_bf16_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
        x_mask = (rm[:, None] < M) & (rk[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk
        w_mask = (rn[:, None] < N) & (rk[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, tl.trans(w))

    acc = acc.to(tl.bfloat16)

    y_ptrs = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    y_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


class Model(nn.Module):
    def __init__(self, in_features: int = 4096, out_features: int = 14336):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.bfloat16) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_bf16 = x.to(dtype=torch.bfloat16)
        M, K = x_bf16.shape
        N = self.weight.shape[0]
        y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

        gemv_bf16_kernel[grid](
            x_bf16, self.weight, y,
            M, N, K,
            x_bf16.stride(0), x_bf16.stride(1),
            self.weight.stride(0), self.weight.stride(1),
            y.stride(0), y.stride(1),
        )
        return y


def get_inputs():
    return [torch.randn(32, 4096, dtype=torch.bfloat16)]


def get_init_inputs():
    return []
