import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def int8_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    scale_ab,
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

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_bn[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0)
        accumulator = tl.dot(a, b, accumulator, input_precision="ieee")
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c = accumulator.to(tl.float32) * scale_ab
    c = c.to(tl.float16)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


class Model(nn.Module):
    def __init__(self, m: int = 2048, n: int = 2048, k: int = 2048):
        super().__init__()
        self.m = m
        self.n = n
        self.k = k

    def forward(self, a_q, b_q, scale_a, scale_b):
        M, K = a_q.shape
        K2, N = b_q.shape
        assert K == K2

        c = torch.empty((M, N), device=a_q.device, dtype=torch.float16)
        scale_ab = (scale_a * scale_b).item()

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

        int8_gemm_kernel[grid](
            a_q, b_q, c,
            scale_ab,
            M, N, K,
            a_q.stride(0), a_q.stride(1),
            b_q.stride(0), b_q.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            GROUP_M=8,
        )
        return c


def get_inputs():
    m = 2048
    n = 2048
    k = 2048
    a_q = torch.randint(-8, 8, (m, k), dtype=torch.int8)
    b_q = torch.randint(-8, 8, (k, n), dtype=torch.int8)
    scale_a = torch.tensor(0.08, dtype=torch.float32)
    scale_b = torch.tensor(0.08, dtype=torch.float32)
    return [a_q, b_q, scale_a, scale_b]


def get_init_inputs():
    return []
