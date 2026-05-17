import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <math.h>

// Fused Mish activation + BatchNorm (eval mode) kernel
// Mish(x) = x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
// BN_eval(y) = (y - running_mean) / sqrt(running_var + eps) * weight + bias
//            = y * scale + shift  where scale = weight/sqrt(var+eps), shift = bias - mean*scale

__global__ void fused_mish_bn_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ scale,   // weight / sqrt(var + eps)
    const float* __restrict__ shift,   // bias - mean * scale
    int N, int C, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * HW;
    
    for (int i = idx; i < total; i += blockDim.x * gridDim.x) {
        float x = input[i];
        
        // Mish: x * tanh(softplus(x))
        // softplus(x) = log(1 + exp(x)), use stable version
        float sp;
        if (x > 20.0f) {
            sp = x;
        } else if (x < -20.0f) {
            sp = expf(x);
        } else {
            sp = log1pf(expf(x));
        }
        float mish = x * tanhf(sp);
        
        // BN eval: mish * scale[c] + shift[c]
        int c = (i / HW) % C;
        output[i] = mish * scale[c] + shift[c];
    }
}

// Vectorized float4 version
__global__ void fused_mish_bn_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    int N, int C, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vec = N * C * HW / 4;
    
    for (int i = idx; i < total_vec; i += blockDim.x * gridDim.x) {
        float4 in4 = input[i];
        int base = i * 4;
        
        float vals[4] = {in4.x, in4.y, in4.z, in4.w};
        float4 out4;
        float* outp = (float*)&out4;
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float x = vals[j];
            float sp;
            if (x > 20.0f) {
                sp = x;
            } else if (x < -20.0f) {
                sp = expf(x);
            } else {
                sp = log1pf(expf(x));
            }
            float mish = x * tanhf(sp);
            int c = ((base + j) / HW) % C;
            outp[j] = mish * scale[c] + shift[c];
        }
        
        output[i] = out4;
    }
}

torch::Tensor fused_mish_bn(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                              torch::Tensor running_mean, torch::Tensor running_var, float eps) {
    auto output = torch::empty_like(input);
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int HW = H * W;
    int total = N * C * HW;
    
    // Precompute scale and shift on GPU
    auto inv_std = torch::rsqrt(running_var + eps);
    auto scale = weight * inv_std;
    auto shift = bias - running_mean * scale;
    
    if (total % 4 == 0) {
        int total_vec = total / 4;
        int threads = 256;
        int blocks = min((total_vec + threads - 1) / threads, 65535);
        fused_mish_bn_kernel_vec4<<<blocks, threads>>>(
            (const float4*)input.data_ptr<float>(),
            (float4*)output.data_ptr<float>(),
            scale.data_ptr<float>(),
            shift.data_ptr<float>(),
            N, C, HW);
    } else {
        int threads = 256;
        int blocks = min((total + threads - 1) / threads, 65535);
        fused_mish_bn_kernel<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            scale.data_ptr<float>(),
            shift.data_ptr<float>(),
            N, C, HW);
    }
    
    return output;
}
'''

cpp_source = r'''
torch::Tensor fused_mish_bn(torch::Tensor input, torch::Tensor weight, torch::Tensor bias,
                              torch::Tensor running_mean, torch::Tensor running_var, float eps);
'''

ext = load_inline(
    name='fused_mish_bn_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_mish_bn'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        x = self.conv(x)
        if self.bn.training:
            # Fallback for training mode
            x = torch.multiply(torch.tanh(torch.nn.functional.softplus(x)), x)
            x = self.bn(x)
        else:
            x = ext.fused_mish_bn(x, self.bn.weight, self.bn.bias,
                                   self.bn.running_mean, self.bn.running_var, self.eps)
        return x

batch_size = 16
in_channels = 3
out_channels = 16
height, width = 256, 256
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
