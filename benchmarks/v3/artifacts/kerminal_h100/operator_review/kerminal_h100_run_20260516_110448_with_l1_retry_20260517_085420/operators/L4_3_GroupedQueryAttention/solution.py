import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def rope_kernel(
    X_ptr, OUT_ptr, COS_ptr, SIN_ptr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    num_heads, seq_len,
    HALF_DIM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % seq_len
    tmp = pid // seq_len
    h = tmp % num_heads
    b = tmp // num_heads

    x_base = b * stride_xb + h * stride_xh + s * stride_xs

    out_base = (b * num_heads * seq_len + h * seq_len + s) * HEAD_DIM

    d = tl.arange(0, HALF_DIM)

    x1 = tl.load(X_ptr + x_base + d * stride_xd)
    x2 = tl.load(X_ptr + x_base + (d + HALF_DIM) * stride_xd)

    cos_val = tl.load(COS_ptr + s * HEAD_DIM + d)
    sin_val = tl.load(SIN_ptr + s * HEAD_DIM + d)

    out1 = x1 * cos_val - x2 * sin_val
    out2 = x2 * cos_val + x1 * sin_val

    tl.store(OUT_ptr + out_base + d, out1)
    tl.store(OUT_ptr + out_base + d + HALF_DIM, out2)


def apply_rope_triton(x, cos, sin, num_heads, seq_len, head_dim):
    batch = x.shape[0]
    out = torch.empty(batch, num_heads, seq_len, head_dim, device=x.device, dtype=x.dtype)
    total = batch * num_heads * seq_len
    rope_kernel[(total,)](
        x, out, cos, sin,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        num_heads, seq_len,
        HALF_DIM=head_dim // 2,
        HEAD_DIM=head_dim,
    )
    return out


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    @torch.no_grad()
    def forward(self, seq_len, device):
        if seq_len <= self._seq_len_cached and self._cos_cached is not None:
            return self._cos_cached[:seq_len], self._sin_cached[:seq_len]
        self._seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_cached = emb.cos().contiguous()
        self._sin_cached = emb.sin().contiguous()
        return self._cos_cached, self._sin_cached


class Model(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.attention_dropout = attention_dropout
        self.softmax_scale = head_dim ** (-0.5)

        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )
        self._gqa_supported = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(q_len, hidden_states.device)

        query_states = apply_rope_triton(query_states, cos, sin, self.num_heads, q_len, self.head_dim)
        key_states = apply_rope_triton(key_states, cos, sin, self.num_kv_heads, q_len, self.head_dim)

        query_states = query_states.to(torch.bfloat16)
        key_states = key_states.to(torch.bfloat16)
        value_states = value_states.to(torch.bfloat16).contiguous()

        if self._gqa_supported is None:
            try:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    is_causal=True, scale=self.softmax_scale, enable_gqa=True,
                )
                self._gqa_supported = True
            except TypeError:
                self._gqa_supported = False
                key_states = key_states[:, :, None, :, :].expand(
                    bsz, self.num_kv_heads, self.num_key_value_groups, q_len, self.head_dim
                ).reshape(bsz, self.num_heads, q_len, self.head_dim)
                value_states = value_states[:, :, None, :, :].expand(
                    bsz, self.num_kv_heads, self.num_key_value_groups, q_len, self.head_dim
                ).reshape(bsz, self.num_heads, q_len, self.head_dim)
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    is_causal=True, scale=self.softmax_scale,
                )
        elif self._gqa_supported:
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                is_causal=True, scale=self.softmax_scale, enable_gqa=True,
            )
        else:
            key_states = key_states[:, :, None, :, :].expand(
                bsz, self.num_kv_heads, self.num_key_value_groups, q_len, self.head_dim
            ).reshape(bsz, self.num_heads, q_len, self.head_dim)
            value_states = value_states[:, :, None, :, :].expand(
                bsz, self.num_kv_heads, self.num_key_value_groups, q_len, self.head_dim
            ).reshape(bsz, self.num_heads, q_len, self.head_dim)
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                is_causal=True, scale=self.softmax_scale,
            )

        attn_output = attn_output.to(hidden_states.dtype)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output


batch_size = 4
seq_len = 2048
hidden_size = 4096
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
max_position_embeddings = 4096


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        max_position_embeddings,
    ]
