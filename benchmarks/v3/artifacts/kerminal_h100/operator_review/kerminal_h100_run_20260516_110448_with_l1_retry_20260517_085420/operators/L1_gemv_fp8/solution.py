import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def int8_gemv_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk, mask=mask_x, other=0)

        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk, mask=mask_w, other=0)

        acc += tl.dot(x, tl.trans(w), input_precision="ieee")

    result = acc.to(tl.float32) * scale
    result_fp16 = result.to(tl.float16)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, result_fp16, mask=mask_out)


class Model(nn.Module):
    def __init__(self, in_features: int = 4096, out_features: int = 14336):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight_q", torch.randint(-127, 127, (out_features, in_features), dtype=torch.int8))
        self.register_buffer("weight_scale", torch.tensor(0.02, dtype=torch.float32))

    def forward(self, x_q: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
        M, K = x_q.shape
        N = self.out_features
        scale = (x_scale * self.weight_scale).item()

        out = torch.empty((M, N), dtype=torch.float16, device=x_q.device)

        BLOCK_M = 32
        BLOCK_N = 128
        BLOCK_K = 128

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        int8_gemv_kernel[grid](
            x_q, self.weight_q, out,
            M, N, K,
            x_q.stride(0), x_q.stride(1),
            self.weight_q.stride(0), self.weight_q.stride(1),
            out.stride(0), out.stride(1),
            scale=scale,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        return out


def get_inputs():
    x_q = torch.randint(-127, 127, (32, 4096), dtype=torch.int8)
    x_scale = torch.tensor(0.02, dtype=torch.float32)
    return [x_q, x_scale]


def get_init_inputs():
    return []
