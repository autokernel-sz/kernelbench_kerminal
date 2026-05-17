import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_sub_tanh_sub_avgpool_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float subtract1,
    const float subtract2,
    const int N, const int C,
    const int H, const int W,
    const int pool_size,
    const int outH, const int outW)
{
    const int total = N * C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= total) return;
    
    const int ow = idx % outW;
    int tmp = idx / outW;
    const int oh = tmp % outH;
    tmp = tmp / outH;
    const int c = tmp % C;
    const int n = tmp / C;
    
    const int h_start = oh * pool_size;
    const int w_start = ow * pool_size;
    
    const int base = (n * C + c) * H * W;
    
    float sum = 0.0f;
    const float inv_pool = 1.0f / (float)(pool_size * pool_size);
    
    for (int ph = 0; ph < pool_size; ph++) {
        const int h_idx = h_start + ph;
        const int row_offset = base + h_idx * W + w_start;
        for (int pw = 0; pw < pool_size; pw++) {
            float val = input[row_offset + pw];
            val = tanhf(val - subtract1) - subtract2;
            sum += val;
        }
    }
    
    output[idx] = sum * inv_pool;
}

torch::Tensor fused_sub_tanh_sub_avgpool(
    torch::Tensor input,
    float subtract1,
    float subtract2,
    int pool_size)
{
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    const int outH = H / pool_size;
    const int outW = W / pool_size;
    
    auto output = torch::empty({N, C, outH, outW}, input.options());
    
    const int total = N * C * outH * outW;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    fused_sub_tanh_sub_avgpool_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        subtract1, subtract2,
        N, C, H, W,
        pool_size, outH, outW);
    
    return output;
}
'''

cpp_source = r'''
torch::Tensor fused_sub_tanh_sub_avgpool(torch::Tensor input, float subtract1, float subtract2, int pool_size);
'''

ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_sub_tanh_sub_avgpool'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.kernel_size_pool = kernel_size_pool

    def forward(self, x):
        x = self.conv(x)
        x = ext.fused_sub_tanh_sub_avgpool(x, self.subtract1_value, self.subtract2_value, self.kernel_size_pool)
        return x

batch_size = 16
in_channels = 3
out_channels = 16
height, width = 256, 256
kernel_size = 3
subtract1_value = 0.5
subtract2_value = 0.2
kernel_size_pool = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool]
