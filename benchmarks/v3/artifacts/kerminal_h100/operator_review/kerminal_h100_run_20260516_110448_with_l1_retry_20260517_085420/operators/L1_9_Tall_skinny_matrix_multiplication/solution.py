import torch
import torch.nn as nn
import triton
import triton.language as tl

M_SIZE = 16384
N_SIZE = 16

@triton.jit
def matmul_tall_skinny_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    mask_a = (rm[:, None] < M) & (rk[None, :] < K)
    mask_b = (rk[:, None] < K) & (rn[None, :] < N)

    a = tl.load(A, mask=mask_a, other=0.0).to(tl.float16)
    b = tl.load(B, mask=mask_b, other=0.0).to(tl.float16)

    acc = tl.dot(a, b)

    C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C, acc, mask=mask_c)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B):
        M, K = A.shape
        K2, N = B.shape
        assert K == K2
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 16

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        matmul_tall_skinny_kernel[grid](
            A, B, C,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return C


def get_inputs():
    A = torch.randn(M_SIZE, N_SIZE)
    B = torch.randn(N_SIZE, M_SIZE)
    return [A, B]

def get_init_inputs():
    return []
