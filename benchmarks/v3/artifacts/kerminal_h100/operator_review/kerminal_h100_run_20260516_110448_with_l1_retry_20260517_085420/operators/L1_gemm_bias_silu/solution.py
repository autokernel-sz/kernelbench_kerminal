import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def matmul_bias_silu_kernel(
    A, B, C, bias_ptr,
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Fused bias + SiLU
    bias = tl.load(bias_ptr + offs_bn).to(tl.float32)
    acc = acc + bias[None, :]
    # SiLU: x * sigmoid(x)
    silu = acc * tl.sigmoid(acc)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, silu.to(tl.float16), mask=mask)


def matmul_bias_silu(a, b, bias):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    matmul_bias_silu_kernel[grid](
        a, b, c, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,
        GROUP_M=8,
        num_warps=8,
        num_stages=3,
    )
    return c


class Model(nn.Module):
    def __init__(self, n: int = 4096):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(n, dtype=torch.float16) * 0.02)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return matmul_bias_silu(a.to(torch.float16), b.to(torch.float16), self.bias)


OP_TYPE = "gemm_epilogue"
SUPPORTED_PRECISIONS = ["fp16", "bf16"]
HARDWARE_REQUIRED = ["H100", "B200"]
SPECIALIZED_LEVEL = 2


def get_inputs():
    m = 2048
    n = 4096
    k = 2048
    return [torch.randn(m, k, dtype=torch.float16), torch.randn(k, n, dtype=torch.float16)]


def get_init_inputs():
    return []
