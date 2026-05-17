import torch
import torch.nn as nn
import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0'
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

static cublasLtHandle_t ltHandle = nullptr;
static void* workspace = nullptr;
static const size_t workspaceSize = 32 * 1024 * 1024;
static cublasLtMatmulDesc_t operationDesc = nullptr;
static cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr;
static cublasLtMatmulAlgo_t cachedAlgo;
static bool initialized = false;

void init_once(int M, int K, int N) {
    if (initialized) return;
    
    cublasLtCreate(&ltHandle);
    cudaMalloc(&workspace, workspaceSize);
    
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    
    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB));
    
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, N, K, N);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, K, M, K);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, N, M, N);
    
    cublasLtMatmulPreference_t preference;
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
    
    cublasLtMatmulHeuristicResult_t heuristicResult[8];
    int returnedResults = 0;
    cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 8, heuristicResult, &returnedResults);
    
    if (returnedResults > 0) {
        cachedAlgo = heuristicResult[0].algo;
    }
    
    cublasLtMatmulPreferenceDestroy(preference);
    initialized = true;
}

torch::Tensor matmul_cublaslt(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::empty({M, N}, A.options());
    
    init_once(M, K, N);
    
    auto stream = c10::cuda::getCurrentCUDAStream();
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    cublasLtMatmul(ltHandle, operationDesc,
        &alpha,
        B.data_ptr<float>(), Adesc,
        A.data_ptr<float>(), Bdesc,
        &beta,
        C.data_ptr<float>(), Cdesc,
        C.data_ptr<float>(), Cdesc,
        &cachedAlgo,
        workspace, workspaceSize,
        stream);
    
    return C;
}
'''

cpp_source = r'''
torch::Tensor matmul_cublaslt(torch::Tensor A, torch::Tensor B);
'''

ext = load_inline(
    name='matmul_ext_v6',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['matmul_cublaslt'],
    extra_ldflags=['-lcublasLt', '-lcublas'],
    verbose=False,
)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return ext.matmul_cublaslt(A, B)

M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []
