import torch
import time
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(X_ptr, Y_ptr, num_features, eps, stride_batch, stride_feat, N_spatial, BLOCK_S: tl.constexpr):
    pid = tl.program_id(0)
    num_spatial_blocks = tl.cdiv(N_spatial, BLOCK_S)
    batch_idx = pid // num_spatial_blocks
    block_idx = pid % num_spatial_blocks
    spatial_off = block_idx * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = spatial_off < N_spatial
    base = batch_idx * stride_batch
    sum_sq = tl.zeros([BLOCK_S], dtype=tl.float32)
    for f in range(num_features):
        ptr = X_ptr + base + f * stride_feat + spatial_off
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        sum_sq += x * x
    rms = tl.sqrt(sum_sq / num_features + eps)
    for f in range(num_features):
        ptr = X_ptr + base + f * stride_feat + spatial_off
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        y = x / rms
        tl.store(Y_ptr + base + f * stride_feat + spatial_off, y, mask=mask)

x = torch.randn(16, 64, 256, 256, device='cuda')
N_spatial = 256*256
stride_batch = x.stride(0)
stride_feat = x.stride(1)

for BS in [256, 512, 1024, 2048, 4096]:
    for nw in [2, 4, 8]:
        y = torch.empty_like(x)
        nsb = (N_spatial + BS - 1) // BS
        grid = (16 * nsb,)
        for _ in range(5):
            rmsnorm_kernel[grid](x, y, 64, 1e-5, stride_batch, stride_feat, N_spatial, BLOCK_S=BS, num_warps=nw)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            rmsnorm_kernel[grid](x, y, 64, 1e-5, stride_batch, stride_feat, N_spatial, BLOCK_S=BS, num_warps=nw)
        torch.cuda.synchronize()
        t = (time.time() - start) / 100
        print(f'BLOCK_S={BS}, num_warps={nw}: {t*1000:.3f} ms')
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    X_ptr, Y_ptr,
    num_features,
    eps,
    stride_batch,
    stride_feat,
    N_spatial,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    num_spatial_blocks = tl.cdiv(N_spatial, BLOCK_S)
    batch_idx = pid // num_spatial_blocks
    block_idx = pid % num_spatial_blocks

    spatial_off = block_idx * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = spatial_off < N_spatial

    base = batch_idx * stride_batch

    # Accumulate sum of squares over features - contiguous loads!
    sum_sq = tl.zeros([BLOCK_S], dtype=tl.float32)
    for f in range(num_features):
        ptr = X_ptr + base + f * stride_feat + spatial_off
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        sum_sq += x * x

    inv_rms = 1.0 / tl.sqrt(sum_sq / num_features + eps)

    # Normalize
    for f in range(num_features):
        ptr = X_ptr + base + f * stride_feat + spatial_off
        x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms
        out_ptr = Y_ptr + base + f * stride_feat + spatial_off
        tl.store(out_ptr, y, mask=mask)


class Model(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        N_spatial = 1
        for s in x.shape[2:]:
            N_spatial *= s

        y = torch.empty_like(x)

        stride_batch = x.stride(0)
        stride_feat = x.stride(1)

        BLOCK_S = 1024
        num_spatial_blocks = (N_spatial + BLOCK_S - 1) // BLOCK_S
        grid = (batch_size * num_spatial_blocks,)

        rmsnorm_kernel[grid](
            x, y,
            self.num_features, self.eps,
            stride_batch, stride_feat, N_spatial,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )

        return y


def get_inputs():
    x = torch.randn(16, 64, 256, 256, device='cuda')
    return [x]

def get_init_inputs():
    return [64]
