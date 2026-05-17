import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>

// Warp reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Kernel 1: Compute per-group mean and inverse std
// Grid: (num_groups, batch_size)
__global__ void group_norm_stats_kernel(
    const float* __restrict__ input,
    float* __restrict__ group_mean,
    float* __restrict__ group_inv_std,
    int C, int H, int W,
    int num_groups, float eps
) {
    int group_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    
    int channels_per_group = C / num_groups;
    int group_size = channels_per_group * H * W;
    int c_start = group_idx * channels_per_group;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    int base = batch_idx * C * H * W + c_start * H * W;
    
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        float val = input[base + i];
        sum += val;
        sum_sq += val * val;
    }
    
    // Warp-level reduction first
    sum = warp_reduce_sum(sum);
    sum_sq = warp_reduce_sum(sum_sq);
    
    __shared__ float s_sum[32];
    __shared__ float s_sum_sq[32];
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    
    if (lane_id == 0) {
        s_sum[warp_id] = sum;
        s_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();
    
    int num_warps = blockDim.x / 32;
    if (warp_id == 0) {
        sum = (lane_id < num_warps) ? s_sum[lane_id] : 0.0f;
        sum_sq = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
        sum = warp_reduce_sum(sum);
        sum_sq = warp_reduce_sum(sum_sq);
    }
    
    if (threadIdx.x == 0) {
        float m = sum / group_size;
        float v = sum_sq / group_size - m * m;
        int out_idx = batch_idx * num_groups + group_idx;
        group_mean[out_idx] = m;
        group_inv_std[out_idx] = rsqrtf(v + eps);
    }
}

// Kernel 2: Fused GroupNorm-apply + Scale + MaxPool2d + Clamp
// Each thread computes one output element
__global__ void fused_gn_scale_maxpool_clamp_kernel(
    const float* __restrict__ input,
    const float* __restrict__ group_mean,
    const float* __restrict__ group_inv_std,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ scale,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int num_groups, int pool_size,
    float clamp_min, float clamp_max
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_H * out_W;
    
    if (idx >= total) return;
    
    int ow = idx % out_W;
    int tmp = idx / out_W;
    int oh = tmp % out_H;
    tmp = tmp / out_H;
    int c = tmp % C;
    int n = tmp / C;
    
    int channels_per_group = C / num_groups;
    int group = c / channels_per_group;
    
    float m = group_mean[n * num_groups + group];
    float inv_s = group_inv_std[n * num_groups + group];
    float g = gamma[c];
    float b = beta[c];
    float sc = scale[c];
    
    int h_start = oh * pool_size;
    int w_start = ow * pool_size;
    int base = (n * C + c) * H * W;
    
    float max_val = -FLT_MAX;
    
    for (int ph = 0; ph < pool_size; ph++) {
        int h = h_start + ph;
        int row_base = base + h * W + w_start;
        for (int pw = 0; pw < pool_size; pw++) {
            float val = input[row_base + pw];
            val = (val - m) * inv_s * g + b;
            val = val * sc;
            if (val > max_val) max_val = val;
        }
    }
    
    max_val = fminf(fmaxf(max_val, clamp_min), clamp_max);
    output[idx] = max_val;
}

torch::Tensor fused_gn_scale_maxpool_clamp(
    torch::Tensor input,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor scale,
    int num_groups,
    int pool_size,
    float clamp_min,
    float clamp_max
) {
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    float eps = 1e-5f;
    
    auto opts = input.options();
    auto group_mean = torch::empty({N, num_groups}, opts);
    auto group_inv_std = torch::empty({N, num_groups}, opts);
    
    // Kernel 1: stats
    int threads1 = 512;
    dim3 grid1(num_groups, N);
    group_norm_stats_kernel<<<grid1, threads1>>>(
        input.data_ptr<float>(),
        group_mean.data_ptr<float>(),
        group_inv_std.data_ptr<float>(),
        C, H, W, num_groups, eps
    );
    
    // Kernel 2: fused apply
    int out_H = H / pool_size;
    int out_W = W / pool_size;
    auto output = torch::empty({N, C, out_H, out_W}, opts);
    
    int total = N * C * out_H * out_W;
    int threads2 = 256;
    int blocks2 = (total + threads2 - 1) / threads2;
    
    auto scale_flat = scale.contiguous().view({C});
    
    fused_gn_scale_maxpool_clamp_kernel<<<blocks2, threads2>>>(
        input.data_ptr<float>(),
        group_mean.data_ptr<float>(),
        group_inv_std.data_ptr<float>(),
        gn_weight.data_ptr<float>(),
        gn_bias.data_ptr<float>(),
        scale_flat.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, out_H, out_W,
        num_groups, pool_size,
        clamp_min, clamp_max
    );
    
    return output;
}
'''

cpp_source = r'''
torch::Tensor fused_gn_scale_maxpool_clamp(
    torch::Tensor input,
    torch::Tensor gn_weight,
    torch::Tensor gn_bias,
    torch::Tensor scale,
    int num_groups,
    int pool_size,
    float clamp_min,
    float clamp_max
);
'''

ext = load_inline(
    name='fused_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['fused_gn_scale_maxpool_clamp'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math'],
)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.num_groups = num_groups
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        x = ext.fused_gn_scale_maxpool_clamp(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.scale,
            self.num_groups,
            self.maxpool_kernel_size,
            self.clamp_min,
            self.clamp_max
        )
        return x


batch_size = 16
in_channels = 3
out_channels = 16
height, width = 256, 256
kernel_size = 3
num_groups = 8
scale_shape = (out_channels, 1, 1)
maxpool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, num_groups, scale_shape, maxpool_kernel_size, clamp_min, clamp_max]
