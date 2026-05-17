import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemv_kernel(
    X_ptr, WT_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
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
        x = tl.load(x_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)

        wt_ptrs = WT_ptr + rk[:, None] * stride_wtk + rn[None, :] * stride_wtn
        wt = tl.load(wt_ptrs, mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)

        acc += tl.dot(x, wt)

    y_ptrs = Y_ptr + rm[:, None] * stride_ym + rn[None, :] * stride_yn
    y_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float16), mask=y_mask)


class Model(nn.Module):
    def __init__(self, in_features: int = 4096, out_features: int = 14336):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float16) * 0.02)
        self.weight_t = None

    def _ensure_weight_t(self):
        if self.weight_t is None:
            self.weight_t = self.weight.data.t().contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(dtype=torch.float16)
        self._ensure_weight_t()
        M, K = x_fp16.shape
        N = self.weight_t.shape[1]
        y = torch.empty((M, N), device=x.device, dtype=torch.float16)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

        gemv_kernel[grid](
            x_fp16, self.weight_t, y,
            M, N, K,
            x_fp16.stride(0), x_fp16.stride(1),
            self.weight_t.stride(0), self.weight_t.stride(1),
            y.stride(0), y.stride(1),
        )
        return y


OP_TYPE = "gemv"
SUPPORTED_PRECISIONS = ["fp16"]


def get_inputs():
    return [torch.randn(32, 4096, dtype=torch.float16)]


def get_init_inputs():
    return []
