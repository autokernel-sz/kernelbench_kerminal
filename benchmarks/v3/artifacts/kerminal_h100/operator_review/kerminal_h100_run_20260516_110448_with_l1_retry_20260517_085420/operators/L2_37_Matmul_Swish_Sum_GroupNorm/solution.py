import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_swish_bias_kernel(
    A_ptr, B_ptr, linear_bias_ptr, add_bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    lin_bias = tl.load(linear_bias_ptr + offs_bn).to(tl.float32)
    acc = acc + lin_bias[None, :]

    sigmoid_val = tl.sigmoid(acc)
    acc = acc * sigmoid_val

    add_bias = tl.load(add_bias_ptr + offs_bn).to(tl.float32)
    acc = acc + add_bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


@triton.jit
def group_norm_kernel(
    X_ptr, Y_ptr, gamma_ptr, beta_ptr,
    M, N, num_groups,
    stride_xm, stride_xn,
    eps: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    offs = tl.arange(0, CHANNELS_PER_GROUP)
    col_start = group * CHANNELS_PER_GROUP
    cols = col_start + offs

    x_ptrs = X_ptr + row * stride_xm + cols * stride_xn
    x = tl.load(x_ptrs).to(tl.float32)

    mean = tl.sum(x, axis=0) / CHANNELS_PER_GROUP
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / CHANNELS_PER_GROUP
    inv_std = 1.0 / tl.sqrt(var + eps)

    x_norm = x_centered * inv_std

    gamma = tl.load(gamma_ptr + cols).to(tl.float32)
    beta = tl.load(beta_ptr + cols).to(tl.float32)
    y = x_norm * gamma + beta

    y_ptrs = Y_ptr + row * stride_xm + cols * stride_xn
    tl.store(y_ptrs, y.to(tl.float32))


class Model(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.num_groups = num_groups
        self.out_features = out_features
        self.in_features = in_features
        self._weight_t_cache = None

    def _get_weight_t(self):
        if self._weight_t_cache is None:
            self._weight_t_cache = self.matmul.weight.t().contiguous().to(torch.float16)
        return self._weight_t_cache

    def forward(self, x):
        M, K = x.shape
        N = self.out_features

        weight_t = self._get_weight_t()
        x_fp16 = x.to(torch.float16)

        mid = torch.empty((M, N), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        matmul_swish_bias_kernel[grid](
            x_fp16, weight_t, self.matmul.bias, self.bias, mid,
            M, N, K,
            x_fp16.stride(0), x_fp16.stride(1),
            K, 1,
            mid.stride(0), mid.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            GROUP_SIZE_M=8,
        )

        out = torch.empty_like(mid)
        channels_per_group = N // self.num_groups
        group_norm_kernel[(M, self.num_groups)](
            mid, out,
            self.group_norm.weight, self.group_norm.bias,
            M, N, self.num_groups,
            mid.stride(0), mid.stride(1),
            eps=self.group_norm.eps,
            CHANNELS_PER_GROUP=channels_per_group,
        )
        return out


batch_size = 128
in_features = 4096
out_features = 4096
num_groups = 32
bias_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]
