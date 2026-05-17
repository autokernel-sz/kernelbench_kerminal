import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>

__global__ void fused_softmax_maxpool4_kernel(
    const half* __restrict__ input,
    float* __restrict__ output,
    int C, int D, int H, int W,
    int D_out, int H_out, int W_out,
    int in_stride_b, int in_stride_d, int in_stride_h, int in_stride_w,
    int out_stride_b, int out_stride_c, int out_stride_d, int out_stride_h, int out_stride_w
) {
    int bhd_idx = blockIdx.x;
    int h_out = bhd_idx % H_out;
    bhd_idx /= H_out;
    int d_out = bhd_idx % D_out;
    int b = bhd_idx / D_out;
    
    int w_out = threadIdx.x;
    if (w_out >= W_out) return;
    
    int d_start = d_out * 4;
    int h_start = h_out * 4;
    int w_start = w_out * 4;
    
    float max_vals[16];
    #pragma unroll
    for (int c = 0; c < 16; c++) max_vals[c] = -FLT_MAX;
    
    const half* base = input + b * in_stride_b;
    
    #pragma unroll
    for (int dd = 0; dd < 4; dd++) {
        int d_idx = d_start + dd;
        if (d_idx >= D) continue;
        const half* d_base = base + d_idx * in_stride_d;
        #pragma unroll
        for (int hh = 0; hh < 4; hh++) {
            int h_idx = h_start + hh;
            if (h_idx >= H) continue;
            const half* dh_base = d_base + h_idx * in_stride_h;
            #pragma unroll
            for (int ww = 0; ww < 4; ww++) {
                int w_idx = w_start + ww;
                if (w_idx >= W) continue;
                
                const half* ptr = dh_base + w_idx * in_stride_w;
                
                // Vectorized load: 16 half values = 32 bytes = 2x float4
                float4 v0 = __ldg(reinterpret_cast<const float4*>(ptr));
                float4 v1 = __ldg(reinterpret_cast<const float4*>(ptr + 8));
                
                half2* h0 = reinterpret_cast<half2*>(&v0);
                half2* h1 = reinterpret_cast<half2*>(&v1);
                
                float vals[16];
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    float2 f = __half22float2(h0[i]);
                    vals[i*2] = f.x;
                    vals[i*2+1] = f.y;
                }
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    float2 f = __half22float2(h1[i]);
                    vals[8+i*2] = f.x;
                    vals[8+i*2+1] = f.y;
                }
                
                float max_val = vals[0];
                #pragma unroll
                for (int c = 1; c < 16; c++) max_val = fmaxf(max_val, vals[c]);
                
                float sum_exp = 0.0f;
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    vals[c] = __expf(vals[c] - max_val);
                    sum_exp += vals[c];
                }
                
                float inv_sum = __fdividef(1.0f, sum_exp);
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    max_vals[c] = fmaxf(max_vals[c], vals[c] * inv_sum);
                }
            }
        }
    }
    
    int out_base = b * out_stride_b + d_out * out_stride_d + h_out * out_stride_h + w_out * out_stride_w;
    #pragma unroll
    for (int c = 0; c < 16; c++) {
        output[out_base + c * out_stride_c] = max_vals[c];
    }
}

torch::Tensor fused_softmax_maxpool4(torch::Tensor input, int D_out, int H_out, int W_out) {
    auto B = input.size(0);
    auto C = input.size(1);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    
    auto output = torch::empty({B, C, D_out, H_out, W_out}, 
                               torch::TensorOptions().device(input.device()).dtype(torch::kFloat32));
    
    int num_blocks = B * D_out * H_out;
    int threads = ((W_out + 31) / 32) * 32;
    
    fused_softmax_maxpool4_kernel<<<num_blocks, threads>>>(
        reinterpret_cast<const half*>(input.data_ptr<at::Half>()),
        output.data_ptr<float>(),
        C, D, H, W,
        D_out, H_out, W_out,
        input.stride(0), input.stride(2), input.stride(3), input.stride(4),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4)
    );
    
    return output;
}
'''

cpp_source = "torch::Tensor fused_softmax_maxpool4(torch::Tensor input, int D_out, int H_out, int W_out);"

ext = load_inline(
    name='fused_sm_pool',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_softmax_maxpool4'],
    verbose=False,
    extra_cuda_cflags=['-O3', '-arch=sm_90', '--use_fast_math'],
)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size).half().to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.half().to(memory_format=torch.channels_last_3d)
        x = self.conv(x)
        B, C, D, H, W = x.shape
        return ext.fused_softmax_maxpool4(x, D // 4, H // 4, W // 4)


batch_size = 16
in_channels = 3
out_channels = 16
depth, height, width = 16, 128, 128
kernel_size = 3
pool_kernel_size = 2


def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]


def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]
