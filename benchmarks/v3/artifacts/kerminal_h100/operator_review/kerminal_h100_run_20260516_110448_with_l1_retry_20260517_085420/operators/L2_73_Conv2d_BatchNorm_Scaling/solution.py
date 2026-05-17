import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['out_h', 'out_w'],
)
@triton.jit
def conv2d_fused_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch, in_channels: tl.constexpr, out_channels,
    in_h, in_w, out_h, out_w,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    total_spatial = out_h * out_w
    n_spatial_blocks = tl.cdiv(total_spatial, BLOCK_SIZE)

    oc = pid % out_channels
    remaining = pid // out_channels
    n = remaining % batch
    spatial_block = remaining // batch

    spatial_offset = spatial_block * BLOCK_SIZE
    spatial_idx = spatial_offset + tl.arange(0, BLOCK_SIZE)
    mask = spatial_idx < total_spatial

    oh = spatial_idx // out_w
    ow = spatial_idx % out_w

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    input_base = n * in_channels * in_h * in_w
    weight_base = oc * in_channels * KH * KW

    for ic in tl.static_range(in_channels):
        ic_input_base = input_base + ic * in_h * in_w
        ic_weight_base = weight_base + ic * KH * KW
        for kh in tl.static_range(KH):
            ih = oh + kh
            row_base = ic_input_base + ih * in_w
            for kw in tl.static_range(KW):
                iw = ow + kw
                inp_val = tl.load(input_ptr + row_base + iw, mask=mask, other=0.0)
                w_val = tl.load(weight_ptr + ic_weight_base + kh * KW + kw)
                acc += inp_val * w_val

    b = tl.load(bias_ptr + oc)
    acc = acc + b

    out_base = n * out_channels * out_h * out_w + oc * out_h * out_w
    tl.store(output_ptr + out_base + spatial_idx, acc, mask=mask)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels)
        self.scaling_factor = scaling_factor
        self.fused_weight = None
        self.fused_bias = None

    def _fuse_params(self):
        conv_w = self.conv.weight  # (out_c, in_c, kh, kw)
        conv_b = self.conv.bias    # (out_c,)

        bn_weight = self.bn.weight        # (out_c,)
        bn_bias = self.bn.bias            # (out_c,)
        bn_mean = self.bn.running_mean    # (out_c,)
        bn_var = self.bn.running_var      # (out_c,)
        bn_eps = self.bn.eps

        gamma = bn_weight / torch.sqrt(bn_var + bn_eps)  # (out_c,)
        beta = bn_bias - bn_mean * gamma                  # (out_c,)

        scale = gamma * self.scaling_factor
        new_weight = conv_w * scale.view(-1, 1, 1, 1)
        new_bias = conv_b * scale + beta * self.scaling_factor

        self.fused_weight = new_weight.contiguous()
        self.fused_bias = new_bias.contiguous()

    def forward(self, x):
        if self.fused_weight is None:
            self._fuse_params()

        batch, in_c, in_h, in_w = x.shape
        out_c = self.fused_weight.shape[0]
        kh, kw = self.fused_weight.shape[2], self.fused_weight.shape[3]
        out_h = in_h - kh + 1
        out_w = in_w - kw + 1

        output = torch.empty(batch, out_c, out_h, out_w, device=x.device, dtype=x.dtype)

        total_spatial = out_h * out_w
        BLOCK_SIZE = 1024
        n_spatial_blocks = (total_spatial + BLOCK_SIZE - 1) // BLOCK_SIZE
        grid = lambda meta: (out_c * batch * ((total_spatial + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE']),)

        conv2d_fused_kernel[grid](
            x, self.fused_weight, self.fused_bias, output,
            batch, in_c, out_c,
            in_h, in_w, out_h, out_w,
            kh, kw,
        )
        return output


batch_size = 16
in_channels = 3
out_channels = 16
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor]
