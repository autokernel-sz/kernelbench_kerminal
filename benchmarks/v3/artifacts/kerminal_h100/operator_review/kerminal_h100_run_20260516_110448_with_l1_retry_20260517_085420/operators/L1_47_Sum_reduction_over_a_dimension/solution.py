import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

template <int WARPS>
__global__ void sum_reduce_warp_coop_f32(
    const float* __restrict__ x,
    float* __restrict__ out,
    int S1, int S2)
{
    __shared__ float smem[32 * WARPS];
    
    int batch = blockIdx.x;
    int col = blockIdx.y * 32 + threadIdx.x;
    int warp_id = threadIdx.y;
    
    if (col >= S2) return;
    
    const float* base = x + (size_t)batch * S1 * S2 + col;
    float acc = 0.0f;
    
    for (int k = warp_id; k < S1; k += WARPS) {
        acc += base[k * S2];
    }
    
    smem[warp_id * 32 + threadIdx.x] = acc;
    __syncthreads();
    
    if (warp_id == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < WARPS; w++) {
            sum += smem[w * 32 + threadIdx.x];
        }
        out[(size_t)batch * S2 + col] = sum;
    }
}

// Instantiate multiple
template __global__ void sum_reduce_warp_coop_f32<4>(const float*, float*, int, int);
template __global__ void sum_reduce_warp_coop_f32<8>(const float*, float*, int, int);
template __global__ void sum_reduce_warp_coop_f32<16>(const float*, float*, int, int);
template __global__ void sum_reduce_warp_coop_f32<32>(const float*, float*, int, int);

torch::Tensor sum_reduce_dim1(torch::Tensor x, torch::Tensor out, int dim) {
    TORCH_CHECK(x.dim() == 3 && dim == 1);
    int S0 = x.size(0), S1 = x.size(1), S2 = x.size(2);
    
    dim3 grid(S0, (S2 + 31) / 32);
    
    // Choose WARPS based on S1 to balance work per warp
    if (S1 <= 32) {
        dim3 block(32, 4);
        sum_reduce_warp_coop_f32<4><<<grid, block>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), S1, S2);
    } else if (S1 <= 128) {
        dim3 block(32, 8);
        sum_reduce_warp_coop_f32<8><<<grid, block>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), S1, S2);
    } else if (S1 <= 512) {
        dim3 block(32, 16);
        sum_reduce_warp_coop_f32<16><<<grid, block>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), S1, S2);
    } else {
        dim3 block(32, 32);
        sum_reduce_warp_coop_f32<32><<<grid, block>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), S1, S2);
    }
    
    return out;
}
'''

cpp_source = "torch::Tensor sum_reduce_dim1(torch::Tensor x, torch::Tensor out, int dim);"

ext = load_inline(
    name='sum_reduce_ext4',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['sum_reduce_dim1'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_90'],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self._out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S0, S1, S2 = x.shape
        out_shape = (S0, 1, S2)
        if self._out is None or self._out.shape != out_shape or self._out.dtype != x.dtype or self._out.device != x.device:
            self._out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
        return ext.sum_reduce_dim1(x, self._out, self.dim)

batch_size = 64
dim1 = 256
dim2 = 256
reduce_dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduce_dim]
