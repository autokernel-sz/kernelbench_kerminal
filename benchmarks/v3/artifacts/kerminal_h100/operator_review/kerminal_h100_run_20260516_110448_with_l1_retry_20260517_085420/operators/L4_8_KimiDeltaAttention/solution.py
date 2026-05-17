    beta: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    B, H, T, K = q.shape
    V = v.shape[-1]
    q = q * scale
    o = torch.zeros(B, H, T, V, device=q.device, dtype=q.dtype)
    S = torch.zeros(B, H, K, V, device=q.device, dtype=q.dtype)
    for t in range(T):
        k_t = k[:, :, t, :]
        v_t = v[:, :, t, :]
        q_t = q[:, :, t, :]
        a_t = a[:, :, t, :]
        b_t = beta[:, :, t].unsqueeze(-1)
        S = S * a_t.unsqueeze(-1)
        Sk = (k_t.unsqueeze(-1) * S).sum(-2)
        S = S + torch.einsum('bhk,bhv->bhkv', b_t * k_t, v_t - Sk)
        o[:, :, t, :] = torch.einsum('bhk,bhkv->bhv', q_t, S)
    return o


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_source = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>

#define D 128

__global__ void kda_forward_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ a,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float scale,
    int T
) {
    const int bh = blockIdx.x;
    const int col = threadIdx.x;

    float S[D];
    #pragma unroll
    for (int j = 0; j < D; j++) S[j] = 0.0f;

    __shared__ float k_sh[D];
    __shared__ float a_sh[D];
    __shared__ float q_sh[D];

    const long long base = (long long)bh * T * D;
    const float* q_ptr = q + base;
    const float* k_ptr = k + base;
    const float* v_ptr = v + base;
    const float* a_ptr = a + base;
    const float* beta_ptr = beta + (long long)bh * T;
    float* out_ptr = out + base;

    for (int t = 0; t < T; t++) {
        const int tD = t * D;
        a_sh[col] = a_ptr[tD + col];
        k_sh[col] = k_ptr[tD + col];
        q_sh[col] = q_ptr[tD + col] * scale;
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < D; kk++) {
            S[kk] *= a_sh[kk];
        }

        float sk = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < D; kk++) {
            sk += k_sh[kk] * S[kk];
        }

        float v_val = v_ptr[tD + col];
        float delta = beta_ptr[t] * (v_val - sk);

        #pragma unroll
        for (int kk = 0; kk < D; kk++) {
            S[kk] += delta * k_sh[kk];
        }

        float o_val = 0.0f;
        #pragma unroll
        for (int kk = 0; kk < D; kk++) {
            o_val += q_sh[kk] * S[kk];
        }

        out_ptr[tD + col] = o_val;
        __syncthreads();
    }
}

torch::Tensor kda_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor beta, float scale
) {
    const auto B = q.size(0);
    const auto H = q.size(1);
    const auto T = q.size(2);
    const auto K = q.size(3);
    const auto V = v.size(3);

    auto q_flat = q.reshape({B*H, T, K}).contiguous();
    auto k_flat = k.reshape({B*H, T, K}).contiguous();
    auto v_flat = v.reshape({B*H, T, V}).contiguous();
    auto a_flat = a.reshape({B*H, T, K}).contiguous();
    auto beta_flat = beta.reshape({B*H, T}).contiguous();

    auto out = torch::empty({B*H, T, V}, q.options());

    kda_forward_kernel<<<B*H, V, 3*K*sizeof(float)>>>(
        q_flat.data_ptr<float>(),
        k_flat.data_ptr<float>(),
        v_flat.data_ptr<float>(),
        a_flat.data_ptr<float>(),
        beta_flat.data_ptr<float>(),
        out.data_ptr<float>(),
        scale, T
    );

    return out.reshape({B, H, T, V});
}
''';

cpp_source = r'''
torch::Tensor kda_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor a, torch::Tensor beta, float scale
);
'''

ext = load_inline(
    name='kda_ext',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['kda_forward'],
    extra_cuda_cflags=['-O3', '--use_fast_math', '-arch=sm_90'],
    verbose=False,
)


class Model(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
        use_dplr: bool = False,
        dplr_rank: int = 4,
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim_qk = head_dim_qk
        self.head_dim_v = head_dim_v
        self.use_short_conv = use_short_conv

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim_qk, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.a_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=True)
        self.b_proj = nn.Linear(hidden_size, num_heads, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim_v, hidden_size, bias=False)

        if use_short_conv:
            self.q_conv = nn.Conv1d(
                num_heads * head_dim_qk, num_heads * head_dim_qk,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1
            )
            self.k_conv = nn.Conv1d(
                num_heads * head_dim_qk, num_heads * head_dim_qk,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_qk,
                padding=conv_kernel_size - 1
            )
            self.v_conv = nn.Conv1d(
                num_heads * head_dim_v, num_heads * head_dim_v,
                kernel_size=conv_kernel_size, groups=num_heads * head_dim_v,
                padding=conv_kernel_size - 1
            )

        self.g_proj = nn.Linear(hidden_size, num_heads * head_dim_v, bias=False)
        self.o_norm = nn.LayerNorm(head_dim_v)
        self.scale = head_dim_qk ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        if self.use_short_conv:
            q = self.q_conv(q.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            k = self.k_conv(k.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            v = self.v_conv(v.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
            q = F.silu(q)
            k = F.silu(k)
            v = F.silu(v)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim_qk).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim_v).transpose(1, 2)

        a = torch.sigmoid(self.a_proj(x))
        a = a.view(batch_size, seq_len, self.num_heads, self.head_dim_v).transpose(1, 2)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)

        o = ext.kda_forward(q, k, v, a, beta, self.scale)

        o = o.transpose(1, 2)
        o = self.o_norm(o)

        g = torch.sigmoid(self.g_proj(x))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim_v)
        o = o * g

        o = o.reshape(batch_size, seq_len, self.num_heads * self.head_dim_v)
        o = self.o_proj(o)
        return o


batch_size = 4
seq_len = 2048
hidden_size = 2048
num_heads = 16
head_dim_qk = 128
head_dim_v = 128

def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]

def get_init_inputs():
    return [hidden_size, num_heads, head_dim_qk, head_dim_v]
