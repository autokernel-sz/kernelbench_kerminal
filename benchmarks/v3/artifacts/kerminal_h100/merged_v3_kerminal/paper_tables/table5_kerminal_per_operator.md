| Level | Problem | Category | Correct | Speedup | Ref ms | Sol ms | Error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L1 | Square matrix multiplication | GEMM | Y | 0.69x | 0.359 | 0.524 |  |
| L1 | Softmax | Softmax | Y | 0.78x | 0.024 | 0.031 |  |
| L1 | GELU | Elementwise | Y | 0.95x | 0.019 | 0.021 |  |
| L1 | Standard matrix multiplication | GEMM | Y | 0.76x | 0.405 | 0.530 |  |
| L1 | RMSNorm | Norm | Y | 1.81x | 0.542 | 0.299 |  |
| L1 | Batched matrix multiplication | GEMM | Y | 0.70x | 0.110 | 0.157 |  |
| L1 | LayerNorm | Norm | Y | 2.18x | 0.697 | 0.319 |  |
| L1 | Max Pooling 2D | Reduction | Y | 1.73x | 0.414 | 0.240 |  |
| L1 | Sum reduction over a dimension | Reduction | Y | 1.22x | 0.019 | 0.015 |  |
| L1 | Matrix vector multiplication | Other | N | -- | -- | -- |  |
| L1 | conv standard 2D  square input  square kernel | Conv | Y | 0.46x | 0.248 | 0.534 |  |
| L1 | conv depthwise 2D square input square kernel | Conv | Y | 1.06x | 0.045 | 0.043 |  |
| L1 | Matmul with irregular shapes | Other | N | -- | -- | -- |  |
| L1 | CrossEntropyLoss | Reduction | Y | 2.12x | 0.087 | 0.041 |  |
| L1 | Tall skinny matrix multiplication | GEMM | Y | 1.36x | 0.542 | 0.397 |  |
| L1 | gemm bf16 | GEMM | Y | 1.02x | 0.055 | 0.054 |  |
| L1 | gemm bias gelu | GEMM | Y | 1.16x | 0.124 | 0.107 |  |
| L1 | gemm bias relu | GEMM | Y | 1.34x | 0.134 | 0.100 |  |
| L1 | gemm bias silu | GEMM | Y | 1.53x | 0.135 | 0.089 |  |
| L1 | gemm fp4 | GEMM | Y | 2.89x | 0.447 | 0.155 |  |
| L1 | gemm fp8 | GEMM | Y | 3.41x | 0.444 | 0.130 |  |
| L1 | gemm mixed fp8 fp16 | GEMM | Y | 1.43x | 0.106 | 0.074 |  |
| L1 | gemm residual add | GEMM | Y | 1.09x | 0.097 | 0.089 |  |
| L1 | gemv bf16 | Other | Y | 0.95x | 0.074 | 0.078 |  |
| L1 | gemv fp16 | Other | Y | 1.01x | 0.076 | 0.075 |  |
| L1 | gemv fp4 | Other | Y | 4.50x | 0.385 | 0.086 |  |
| L1 | gemv fp8 | Other | Y | 4.56x | 0.384 | 0.084 |  |
| L1 | moe grouped gemm | GEMM | Y | 4.14x | 1.901 | 0.459 |  |
| L2 | Conv2d InstanceNorm Divide | Fused | Y | 0.64x | 0.248 | 0.386 |  |
| L2 | Matmul Swish Sum GroupNorm | Fused | Y | 2.05x | 0.175 | 0.086 |  |
| L2 | Matmul Scaling ResidualAdd | Fused | Y | 1.56x | 0.172 | 0.110 |  |
| L2 | Conv2d Subtract Tanh Subtract AvgPool | Fused | Y | 0.95x | 0.250 | 0.262 |  |
| L2 | Conv2d Activation BatchNorm | Fused | Y | 0.70x | 0.208 | 0.297 |  |
| L2 | Matmul MaxPool Sum Scale | Fused | Y | 2.79x | 0.221 | 0.079 |  |
| L2 | Matmul Swish Scaling | Fused | Y | 2.11x | 0.170 | 0.081 |  |
| L2 | Matmul Dropout Mean Softmax | Fused | Y | 2.48x | 0.214 | 0.086 |  |
| L2 | Conv3d Softmax MaxPool MaxPool | Fused | Y | 2.89x | 2.064 | 0.713 |  |
| L2 | Conv2d BatchNorm Scaling | Fused | Y | 1.02x | 0.204 | 0.200 |  |
| L2 | Conv2d Tanh Scaling BiasAdd Max | Fused | Y | 0.96x | 0.245 | 0.255 |  |
| L2 | Conv2d GroupNorm Scale MaxPool Clamp | Fused | Y | 0.77x | 0.293 | 0.383 |  |
| L2 | Matmul Divide GELU | Fused | Y | 2.49x | 0.172 | 0.069 |  |
| L2 | Matmul AvgPool GELU Scale Max | Fused | Y | 2.04x | 0.224 | 0.110 |  |
| L2 | Matmul GELU Softmax | Fused | Y | 2.07x | 0.175 | 0.085 |  |
| L3 | VisionAttention | Attention | Y | 10.02x | 14.517 | 1.448 |  |
| L3 | MinGPTCausalAttention | Attention | Y | 1.23x | 5.291 | 4.301 |  |
| L3 | MiniGPTBlock | Other | Y | 8.02x | 24.346 | 3.035 |  |
| L4 | DeepSeek MLA | Other | N | -- | -- | -- |  |
| L4 | DeepSeek MoE | Other | N | -- | -- | -- |  |
| L4 | GroupedQueryAttention | Other | N | -- | -- | -- |  |
| L4 | FP8 Matmul | GEMM | Y | 1.13x | 0.780 | 0.687 |  |
| L4 | MoE GatedGEMM | Fused | Y | 8.12x | 123.065 | 15.155 |  |
| L4 | INT4 Quantized GEMM | GEMM | Y | 1.04x | 1.157 | 1.109 |  |
| L4 | GatedDeltaNet | Other | N | -- | -- | -- |  |
| L4 | KimiDeltaAttention | Other | N | -- | -- | -- |  |
