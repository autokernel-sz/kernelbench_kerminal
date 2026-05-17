import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_dequant_gemm_kernel(
    a_ptr, b_ptr, c_ptr, scale_a_ptr,
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

    scale_a = tl.load(scale_a_ptr).to(tl.float32)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_a = (offs_am[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_bn[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        a_fp16 = (a.to(tl.float32) * scale_a).to(tl.float16)
        b_fp16 = b.to(tl.float16)

        acc += tl.dot(a_fp16, b_fp16)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c = acc.to(tl.float16)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a_q: torch.Tensor, b_fp16: torch.Tensor, scale_a: torch.Tensor) -> torch.Tensor:
        M, K = a_q.shape
        K2, N = b_fp16.shape
        assert K == K2

        c = torch.empty((M, N), dtype=torch.float16, device=a_q.device)

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)

        fused_dequant_gemm_kernel[grid](
            a_q, b_fp16, c, scale_a,
            M, N, K,
            a_q.stride(0), a_q.stride(1),
            b_fp16.stride(0), b_fp16.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=8,
        )

        return c


def get_inputs():
    m = 2048
    n = 2048
    k = 2048
    a_q = torch.randint(-127, 127, (m, k), dtype=torch.int8)
    b_fp16 = torch.randn(k, n, dtype=torch.float16)
    scale_a = torch.tensor(0.01, dtype=torch.float32)
    return [a_q, b_fp16, scale_a]


def get_init_inputs():
    return []
