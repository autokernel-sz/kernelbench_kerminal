import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_swish_scale_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    scaling_factor,
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_bn)
    acc = acc + bias[None, :]

    # Fused swish + scaling
    acc = acc * tl.sigmoid(acc) * scaling_factor

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)


@triton.jit
def swish_scale_kernel(
    X_ptr, Out_ptr, N, scaling_factor,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask)
    tl.store(Out_ptr + offs, x * tl.sigmoid(x) * scaling_factor, mask=mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self._weight_t = None
        self._bias_fp32 = None

    def _prepare(self):
        if self._weight_t is None:
            self._weight_t = self.matmul.weight.half().t().contiguous()
            self._bias_fp32 = self.matmul.bias.float()

    def forward(self, x):
        self._prepare()
        x_fp16 = x.half()
        weight_t = self._weight_t
        bias = self._bias_fp32

        M, K = x_fp16.shape
        N = weight_t.shape[1]

        C = torch.empty((M, N), device=x.device, dtype=torch.float16)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

        fused_matmul_swish_scale_kernel[grid](
            x_fp16, weight_t, bias, C,
            M, N, K,
            x_fp16.stride(0), x_fp16.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            C.stride(0), C.stride(1),
            self.scaling_factor,
        )
        return C.float()


batch_size = 128
in_features = 4096
out_features = 4096
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]
