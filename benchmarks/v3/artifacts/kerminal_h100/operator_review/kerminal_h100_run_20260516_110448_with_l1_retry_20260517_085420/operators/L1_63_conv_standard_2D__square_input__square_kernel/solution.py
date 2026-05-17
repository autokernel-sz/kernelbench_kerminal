import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <ATen/cudnn/Descriptors.h>
#include <ATen/native/ConvUtils.h>

torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight,
                              int64_t stride, int64_t padding, 
                              int64_t dilation, int64_t groups) {
    // Convert to half for tensor core acceleration
    auto input_half = input.to(torch::kHalf);
    auto weight_half = weight.to(torch::kHalf);
    
    std::vector<int64_t> stride_vec = {stride, stride};
    std::vector<int64_t> padding_vec = {padding, padding};
    std::vector<int64_t> dilation_vec = {dilation, dilation};
    
    auto output = at::cudnn_convolution(input_half, weight_half,
                                         padding_vec, stride_vec, 
                                         dilation_vec, groups,
                                         true,   // benchmark
                                         false,  // deterministic
                                         true);  // allow_tf32
    
    return output.to(torch::kFloat);
}
'''

cpp_source = r'''
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight,
                              int64_t stride, int64_t padding,
                              int64_t dilation, int64_t groups);
'''

ext = load_inline(
    name='conv2d_fp16_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['conv2d_forward'],
    verbose=False,
    extra_cuda_cflags=['-O3', '-arch=sm_90'],
)


class Model(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1, 
                 groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size),
                                stride=stride, padding=padding, dilation=dilation,
                                groups=groups, bias=bias)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ext.conv2d_forward(x, self.conv2d.weight,
                                   self.stride, self.padding,
                                   self.dilation, self.groups)


batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]
