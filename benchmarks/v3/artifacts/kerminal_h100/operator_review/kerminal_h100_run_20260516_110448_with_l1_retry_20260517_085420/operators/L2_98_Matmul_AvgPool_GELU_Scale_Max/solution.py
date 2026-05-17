import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_pool_gelu_scale_max_kernel(
    X, OUT,
    N_cols,
    pool_k,
    scale_factor,
    stride_x,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pooled_size = N_cols // pool_k

    max_val = tl.full([], value=-float('inf'), dtype=tl.float32)

    for start in range(0, pooled_size, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < pooled_size

        acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        base_ptr = X + pid_m * stride_x
        for p in range(pool_k):
            idx = offs * pool_k + p
            val = tl.load(base_ptr + idx, mask=mask & (idx < N_cols), other=0.0).to(tl.float32)
            acc += val
        acc = acc / pool_k

        inv_sqrt2: tl.constexpr = 0.7071067811865476
        x = acc
        gelu_val = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))

        scaled = gelu_val * scale_factor

        scaled = tl.where(mask, scaled, -float('inf'))
        block_max = tl.max(scaled, axis=0)
        max_val = tl.where(block_max > max_val, block_max, max_val)

    tl.store(OUT + pid_m, max_val)


class Model(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor
        self.in_features = in_features
        self.out_features = out_features
        self._weight_h = None
        self._bias_h = None

    def forward(self, x):
        M, K = x.shape
        N = self.out_features

        if self._weight_h is None:
            self._weight_h = self.matmul.weight.half()
            self._bias_h = self.matmul.bias.half()

        x_h = x.half()
        c = torch.addmm(self._bias_h.unsqueeze(0), x_h, self._weight_h.t())

        out = torch.empty(M, device=x.device, dtype=torch.float32)
        BLOCK_SIZE = 1024

        fused_pool_gelu_scale_max_kernel[(M,)](
            c, out,
            N, self.pool_kernel_size, self.scale_factor,
            c.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return out


batch_size = 128
in_features = 4096
out_features = 4096
pool_kernel_size = 4
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, pool_kernel_size, scale_factor]
