import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_tanh_scale_bias_maxpool_kernel(
    const float* __restrict__ input,
    const float* __restrict__ bias,
    float* __restrict__ output,
    float scaling_factor,
    int N, int C, int H, int W,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int tmp = idx / W_out;
    int h_out = tmp % H_out;
    tmp = tmp / H_out;
    int c = tmp % C;
    int n = tmp / C;

    float b = __ldg(&bias[c]);

    int h_start = h_out * 2;
    int w_start = w_out * 2;
    int base = (n * C + c) * H * W;

    float v00 = __ldg(&input[base + h_start * W + w_start]);
    float v01 = __ldg(&input[base + h_start * W + w_start + 1]);
    float v10 = __ldg(&input[base + (h_start + 1) * W + w_start]);
    float v11 = __ldg(&input[base + (h_start + 1) * W + w_start + 1]);

    v00 = __fmaf_rn(tanhf(v00), scaling_factor, b);
    v01 = __fmaf_rn(tanhf(v01), scaling_factor, b);
    v10 = __fmaf_rn(tanhf(v10), scaling_factor, b);
    v11 = __fmaf_rn(tanhf(v11), scaling_factor, b);

    float max_val = fmaxf(fmaxf(v00, v01), fmaxf(v10, v11));
    output[idx] = max_val;
}

torch::Tensor fused_tanh_scale_bias_maxpool(
    torch::Tensor input, torch::Tensor bias,
    float scaling_factor, int pool_k)
{
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int H_out = H / pool_k;
    int W_out = W / pool_k;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    int total = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fused_tanh_scale_bias_maxpool_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), bias.data_ptr<float>(),
        output.data_ptr<float>(), scaling_factor,
        N, C, H, W, H_out, W_out);

    return output;
}
'''

cpp_source = '''
torch::Tensor fused_tanh_scale_bias_maxpool(
    torch::Tensor input, torch::Tensor bias,
    float scaling_factor, int pool_k);
'''

ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_tanh_scale_bias_maxpool'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = scaling_factor
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = ext.fused_tanh_scale_bias_maxpool(
            x, self.bias.view(-1), self.scaling_factor, self.pool_kernel_size)
        return x

batch_size = 16
in_channels = 3
out_channels = 16
height, width = 256, 256
kernel_size = 3
scaling_factor = 2.0
bias_shape = (out_channels, 1, 1)
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size]
