import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_residual_add_kernel(
    a_ptr, b_ptr, res_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_rm, stride_rn,
    stride_om, stride_on,
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

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    res_ptrs = res_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
    residual = tl.load(res_ptrs, mask=mask, other=0.0)
    out = acc.to(tl.float16) + residual

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask)


def gemm_residual_add(a, b, residual):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    K, N = b.shape
    out = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    gemm_residual_add_kernel[grid](
        a, b, residual, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        residual.stride(0), residual.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=128, BLOCK_N=256, BLOCK_K=64,
        GROUP_SIZE_M=8,
        num_warps=8,
        num_stages=3,
    )
    return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, residual):
        return gemm_residual_add(
            a.to(torch.float16),
            b.to(torch.float16),
            residual.to(torch.float16),
        )


def get_inputs():
    m = 2048
    n = 4096
    k = 2048
    a = torch.randn(m, k, dtype=torch.float16)
    b = torch.randn(k, n, dtype=torch.float16)
    residual = torch.randn(m, n, dtype=torch.float16)
    return [a, b, residual]


def get_init_inputs():
    return []
