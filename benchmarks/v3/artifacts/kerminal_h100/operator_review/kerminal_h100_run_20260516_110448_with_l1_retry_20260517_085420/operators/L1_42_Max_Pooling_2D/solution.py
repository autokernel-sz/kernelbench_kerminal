import torch
import torch.nn as nn
import triton
import triton.language as tl

batch_size = 16
channels = 32
height = 512
width = 512
kernel_size = 2
stride = 2
padding = 1
dilation = 3

@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    batch_channels,  # batch_size * channels
    H: tl.constexpr, W: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
    padding: tl.constexpr,
    dilation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles BLOCK_SIZE output elements
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_out = batch_channels * OH * OW
    mask = offs < total_out

    # Decompose linear index into (bc, oh, ow)
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    bc = tmp // OH

    # Input base offset for this batch*channel
    in_base = bc * (H * W)

    # Compute max over kernel window
    val = tl.full([BLOCK_SIZE], value=float('-inf'), dtype=tl.float32)

    for kh in tl.static_range(kernel_size):
        for kw in tl.static_range(kernel_size):
            ih = oh * stride - padding + kh * dilation
            iw = ow * stride - padding + kw * dilation
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
            idx = in_base + ih * W + iw
            x_val = tl.load(X_ptr + idx, mask=valid, other=float('-inf'))
            val = tl.maximum(val, x_val)

    # Store output
    tl.store(Y_ptr + offs, val, mask=mask)


class Model(nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        OH = (H + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        OW = (W + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1

        y = torch.empty(B, C, OH, OW, device=x.device, dtype=x.dtype)

        bc = B * C
        total = bc * OH * OW
        BLOCK_SIZE = 1024
        grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        maxpool2d_kernel[grid](
            x, y,
            bc,
            H, W,
            OH, OW,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y


def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]
