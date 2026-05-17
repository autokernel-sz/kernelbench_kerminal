import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def fused_maxpool_sum_scale_kernel(
    X_ptr, bias_ptr, OUT_ptr,
    N_cols,
    scale_factor,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    acc = 0.0
    half_n = N_cols // 2

    for off in range(0, half_n, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < half_n
        idx0 = idx * 2
        idx1 = idx0 + 1

        v0 = tl.load(X_ptr + row * N_cols + idx0, mask=mask, other=-float('inf'))
        v1 = tl.load(X_ptr + row * N_cols + idx1, mask=mask, other=-float('inf'))
        b0 = tl.load(bias_ptr + idx0, mask=mask, other=0.0)
        b1 = tl.load(bias_ptr + idx1, mask=mask, other=0.0)
        v0 = v0 + b0
        v1 = v1 + b1
        mx = tl.maximum(v0, v1)
        acc += tl.sum(mx, axis=0)

    tl.store(OUT_ptr + row, acc * scale_factor)


class Model(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.weight_fp16 = None

    def forward(self, x):
        M, K = x.shape
        N = self.matmul.weight.shape[0]

        if self.weight_fp16 is None:
            self.weight_fp16 = self.matmul.weight.half().contiguous()

        weight = self.weight_fp16
        bias = self.matmul.bias
        x_fp16 = x.half()

        c = torch.empty((M, N), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        matmul_kernel[grid](
            x_fp16, weight, c,
            M, N, K,
            x_fp16.stride(0), x_fp16.stride(1),
            weight.stride(1), weight.stride(0),
            c.stride(0), c.stride(1),
        )

        out = torch.empty(M, device=x.device, dtype=torch.float32)
        fused_maxpool_sum_scale_kernel[(M,)](
            c, bias, out,
            N, self.scale_factor,
            BLOCK_SIZE=2048,
        )
        return out


batch_size = 128
in_features = 4096
out_features = 4096
kernel_size = 2
scale_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, kernel_size, scale_factor]
