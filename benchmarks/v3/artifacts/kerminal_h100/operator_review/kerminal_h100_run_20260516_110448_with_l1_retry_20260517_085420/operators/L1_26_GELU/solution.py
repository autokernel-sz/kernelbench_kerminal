import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

// Vectorized GELU kernel using float4 (processes 4 floats per thread)
__global__ void gelu_vec4_kernel(const float4* __restrict__ x, float4* __restrict__ y, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 val = x[idx];
        const float k = 0.7978845608028654f; // sqrt(2/pi)
        const float c = 0.044715f;
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            float v = reinterpret_cast<float*>(&val)[i];
            float t = k * (v + c * v * v * v);
            reinterpret_cast<float*>(&val)[i] = 0.5f * v * (1.0f + tanhf(t));
        }
        y[idx] = val;
    }
}

// Scalar fallback for remainder
__global__ void gelu_scalar_kernel(const float* __restrict__ x, float* __restrict__ y, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < n) {
        float v = x[idx];
        const float k = 0.7978845608028654f;
        const float c = 0.044715f;
        float t = k * (v + c * v * v * v);
        y[idx] = 0.5f * v * (1.0f + tanhf(t));
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32);
    auto y = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;
    int rem = n - n4 * 4;
    
    const int threads = 256;
    if (n4 > 0) {
        int blocks = (n4 + threads - 1) / threads;
        gelu_vec4_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(y.data_ptr<float>()),
            n4);
    }
    if (rem > 0) {
        int blocks = (rem + threads - 1) / threads;
        gelu_scalar_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(), y.data_ptr<float>(), n4 * 4, n);
    }
    return y;
}
'''

ext = load_inline(
    name='gelu_ext',
    cpp_sources='torch::Tensor gelu_forward(torch::Tensor);',
    cuda_sources=cuda_source,
    functions=['gelu_forward'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_90'],
)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return ext.gelu_forward(x)

batch_size = 256
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
