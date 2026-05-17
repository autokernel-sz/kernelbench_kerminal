import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

// Optimized depthwise conv2d kernel for small kernels
// Each thread computes one output pixel
template<int KERNEL_SIZE>
__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int stride, int padding,
    bool has_bias)
{
    // Each thread: one output element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int c = tmp % channels;
    int b = tmp / channels;

    float sum = 0.0f;
    const float* w_ptr = weight + c * KERNEL_SIZE * KERNEL_SIZE;

    #pragma unroll
    for (int kh = 0; kh < KERNEL_SIZE; kh++) {
        int ih = oh * stride - padding + kh;
        if (ih >= 0 && ih < in_h) {
            const float* in_row = input + ((b * channels + c) * in_h + ih) * in_w;
            #pragma unroll
            for (int kw = 0; kw < KERNEL_SIZE; kw++) {
                int iw = ow * stride - padding + kw;
                if (iw >= 0 && iw < in_w) {
                    sum += in_row[iw] * w_ptr[kh * KERNEL_SIZE + kw];
                }
            }
        }
    }

    if (has_bias) sum += bias[c];
    output[idx] = sum;
}

// Generic kernel size version
__global__ void depthwise_conv2d_kernel_generic(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w,
    int kernel_size, int stride, int padding,
    bool has_bias)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * channels * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int tmp = idx / out_w;
    int oh = tmp % out_h;
    tmp = tmp / out_h;
    int c = tmp % channels;
    int b = tmp / channels;

    float sum = 0.0f;
    const float* w_ptr = weight + c * kernel_size * kernel_size;

    for (int kh = 0; kh < kernel_size; kh++) {
        int ih = oh * stride - padding + kh;
        if (ih >= 0 && ih < in_h) {
            const float* in_row = input + ((b * channels + c) * in_h + ih) * in_w;
            for (int kw = 0; kw < kernel_size; kw++) {
                int iw = ow * stride - padding + kw;
                if (iw >= 0 && iw < in_w) {
                    sum += in_row[iw] * w_ptr[kh * kernel_size + kw];
                }
            }
        }
    }

    if (has_bias) sum += bias[c];
    output[idx] = sum;
}

torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int stride,
    int padding)
{
    int batch_size = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int kernel_size = weight.size(2);

    int out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    int out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({batch_size, channels, out_h, out_w}, input.options());

    int total = batch_size * channels * out_h * out_w;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    bool has_bias = bias_opt.has_value() && bias_opt.value().defined();
    const float* bias_ptr = has_bias ? bias_opt.value().data_ptr<float>() : nullptr;

    if (kernel_size == 3) {
        depthwise_conv2d_kernel<3><<<blocks, threads>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(),
            batch_size, channels, in_h, in_w, out_h, out_w,
            stride, padding, has_bias);
    } else if (kernel_size == 5) {
        depthwise_conv2d_kernel<5><<<blocks, threads>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(),
            batch_size, channels, in_h, in_w, out_h, out_w,
            stride, padding, has_bias);
    } else if (kernel_size == 7) {
        depthwise_conv2d_kernel<7><<<blocks, threads>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(),
            batch_size, channels, in_h, in_w, out_h, out_w,
            stride, padding, has_bias);
    } else {
        depthwise_conv2d_kernel_generic<<<blocks, threads>>>(
            input.data_ptr<float>(), weight.data_ptr<float>(), bias_ptr,
            output.data_ptr<float>(),
            batch_size, channels, in_h, in_w, out_h, out_w,
            kernel_size, stride, padding, has_bias);
    }

    return output;
}
'''

cpp_source = r'''
torch::Tensor depthwise_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int stride,
    int padding);
'''

ext = load_inline(
    name='depthwise_conv2d_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['depthwise_conv2d_forward'],
    verbose=False,
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_90'],
)

class Model(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size,
                                stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ext.depthwise_conv2d_forward(
            x, self.conv2d.weight, self.conv2d.bias, self.stride, self.padding)

batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]
