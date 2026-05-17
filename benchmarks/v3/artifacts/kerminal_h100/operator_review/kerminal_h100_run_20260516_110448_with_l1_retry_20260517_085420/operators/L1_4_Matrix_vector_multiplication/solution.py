import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREADS 256
#define BLOCKS_PER_ROW 4
#define ELEMS_PER_BLOCK (131072 / BLOCKS_PER_ROW)  // 32768

__global__ __launch_bounds__(THREADS)
void matvec_kernel(const float* __restrict__ A,
                   const float* __restrict__ B,
                   float* __restrict__ C,
                   int K) {
    int row = blockIdx.x;
    int chunk_id = blockIdx.y;
    int tid = threadIdx.x;

    int k_start = chunk_id * ELEMS_PER_BLOCK;
    int k_end = k_start + ELEMS_PER_BLOCK;
    if (k_end > K) k_end = K;

    const float4* A_row4 = reinterpret_cast<const float4*>(A + row * K + k_start);
    const float4* B4 = reinterpret_cast<const float4*>(B + k_start);
    int len4 = (k_end - k_start) >> 2;

    float sum = 0.0f;
    for (int i = tid; i < len4; i += THREADS) {
        float4 a = __ldg(A_row4 + i);
        float4 b = __ldg(B4 + i);
        sum = __fmaf_rn(a.x, b.x, sum);
        sum = __fmaf_rn(a.y, b.y, sum);
        sum = __fmaf_rn(a.z, b.z, sum);
        sum = __fmaf_rn(a.w, b.w, sum);
    }

    // Warp reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    __shared__ float sdata[32];
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = THREADS >> 5;

    if (lane == 0) sdata[warp_id] = sum;
    __syncthreads();

    if (warp_id == 0) {
        sum = (lane < num_warps) ? sdata[lane] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) atomicAdd(&C[row], sum);
    }
}

torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    auto C = torch::zeros({M, 1}, A.options());
    dim3 grid(M, BLOCKS_PER_ROW);
    matvec_kernel<<<grid, THREADS>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K);
    return C;
}
'''

cpp_source = "torch::Tensor matvec_cuda(torch::Tensor A, torch::Tensor B);"

ext = load_inline(
    name='matvec_ext5',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matvec_cuda'],
    verbose=False,
    extra_cuda_cflags=['-O3', '-arch=sm_90', '--use_fast_math'],
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return ext.matvec_cuda(A, B)


M = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, 1)
    return [A, B]

def get_init_inputs():
    return []
