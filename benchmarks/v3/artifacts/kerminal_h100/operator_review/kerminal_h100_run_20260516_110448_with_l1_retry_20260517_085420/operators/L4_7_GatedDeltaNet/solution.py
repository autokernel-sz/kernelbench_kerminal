import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gated_delta_rule_kernel(
    q_ptr, k_ptr, v_ptr, g_ptr, beta_ptr, o_ptr,
    B: tl.constexpr, T: tl.constexpr, H: tl.constexpr,
    DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr,
    scale,
):
    i_bh = tl.program_id(0)
    i_v = tl.program_id(1)

    i_b = i_bh // H
    i_h = i_bh % H

    o_k = tl.arange(0, DK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_v = o_v < DV

    b_h = tl.zeros([DK, BV], dtype=tl.float32)

    p_q = q_ptr + (i_b * T * H + i_h) * DK + o_k
    p_k = k_ptr + (i_b * T * H + i_h) * DK + o_k
    p_v = v_ptr + (i_b * T * H + i_h) * DV + o_v
    p_g = g_ptr + i_b * T * H + i_h
    p_beta = beta_ptr + i_b * T * H + i_h
    p_o = o_ptr + (i_b * T * H + i_h) * DV + o_v

    for t in range(T):
        b_q = tl.load(p_q).to(tl.float32) * scale
        b_k = tl.load(p_k).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_bt = tl.load(p_beta).to(tl.float32)

        b_h *= tl.exp(b_g)
        b_v = b_bt * (b_v - tl.sum(b_h * b_k[:, None], 0))
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)

        tl.store(p_o, b_o.to(o_ptr.dtype.element_ty), mask=mask_v)

        p_q += H * DK
        p_k += H * DK
        p_v += H * DV
        p_g += H
        p_beta += H
        p_o += H * DV


def gated_delta_attention(q, k, v, alpha, beta, scale):
    g = alpha.clamp(min=1e-6).log()

    q_c = q.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    g_c = g.contiguous()
    beta_c = beta.contiguous()

    B, T_eff, H_eff, DK = q_c.shape
    DV = v_c.shape[3]

    BV = min(8, triton.next_power_of_2(DV))
    NV = triton.cdiv(DV, BV)

    output = torch.empty_like(v_c)

    grid = (B * H_eff, NV)

    gated_delta_rule_kernel[grid](
        q_c, k_c, v_c, g_c, beta_c, output,
        B, T_eff, H_eff, DK, DV, BV,
        scale,
        num_warps=1,
        num_stages=3,
    )

    return output


class Model(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim_qk: int,
        head_dim_v: int,
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

        self.a_proj = nn.Linear(hidden_size, num_heads, bias=True)
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

        alpha = torch.sigmoid(self.a_proj(x)).transpose(1, 2)
        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)

        o = gated_delta_attention(q, k, v, alpha, beta, scale=self.scale)

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
