import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def instance_norm_div_kernel(
    X_ptr, Y_ptr,
    N_instances, HW,
    divide_by: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N_instances:
        return

    base = pid * HW
    
    # Compute mean
    mean_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        mean_acc += x
    mean = tl.sum(mean_acc) / HW

    # Compute variance
    var_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean
        var_acc += diff * diff
    var = tl.sum(var_acc) / HW
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    inv_div = inv_std / divide_by

    # Normalize and divide
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x = tl.load(X_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * inv_div
        tl.store(Y_ptr + base + idx, y, mask=mask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        HW = H * W
        N_instances = B * C
        y = torch.empty_like(x)
        
        BLOCK_SIZE = 8192
        
        grid = (N_instances,)
        instance_norm_div_kernel[grid](
            x, y,
            N_instances, HW,
            divide_by=self.divide_by,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return y


batch_size = 16
in_channels = 3
out_channels = 16
height, width = 256, 256
kernel_size = 3
divide_by = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divide_by]
