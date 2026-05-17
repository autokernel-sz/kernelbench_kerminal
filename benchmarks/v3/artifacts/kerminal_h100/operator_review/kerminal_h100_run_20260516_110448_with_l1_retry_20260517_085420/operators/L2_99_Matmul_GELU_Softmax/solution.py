import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_gelu_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
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

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_bn)
    acc = acc + bias[None, :]

    x = acc
    cdf = 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    acc = x * cdf

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * stride_m
    out_start = output_ptr + row_idx * stride_m

    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N
    row = tl.load(row_start + col_offsets * stride_n, mask=mask, other=-float('inf'))

    row_max = tl.max(row, axis=0)
    row = row - row_max
    numerator = tl.math.exp(row)
    denominator = tl.sum(numerator, axis=0)
    result = numerator / denominator

    tl.store(out_start + col_offsets * stride_n, result, mask=mask)


class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self._weight_t_fp16 = None

    def _prepare_weights(self):
        if self._weight_t_fp16 is None:
            self._weight_t_fp16 = self.linear.weight.data.t().contiguous().half()

    def forward(self, x):
        self._prepare_weights()
        weight_t = self._weight_t_fp16
        bias = self.linear.bias

        M, K = x.shape
        N = weight_t.shape[1]

        x_fp16 = x.half()
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        matmul_gelu_kernel[grid](
            x_fp16, weight_t, bias, out,
            M, N, K,
            x_fp16.stride(0), x_fp16.stride(1),
            weight_t.stride(0), weight_t.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            GROUP_M=8,
        )

        BLOCK_N_soft = triton.next_power_of_2(N)
        softmax_kernel[(M,)](
            out, out, M, N,
            out.stride(0), out.stride(1),
            BLOCK_N=BLOCK_N_soft,
        )

        return out


batch_size = 128
in_features = 4096
out_features = 4096

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]
