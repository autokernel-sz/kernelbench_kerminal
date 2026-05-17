import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl


@triton.jit
def fused_qkv_reshape_kernel(
    qkv_ptr, q_ptr, k_ptr, v_ptr,
    B, T, C: tl.constexpr, n_head: tl.constexpr, head_dim: tl.constexpr,
    stride_qkv_b, stride_qkv_t, stride_qkv_c,
    stride_out_b, stride_out_h, stride_out_t, stride_out_d,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * n_head * T * head_dim
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    d = offsets % head_dim
    tmp = offsets // head_dim
    t = tmp % T
    tmp2 = tmp // T
    h = tmp2 % n_head
    b = tmp2 // n_head

    qkv_offset_q = b * stride_qkv_b + t * stride_qkv_t + (h * head_dim + d)
    qkv_offset_k = b * stride_qkv_b + t * stride_qkv_t + (C + h * head_dim + d)
    qkv_offset_v = b * stride_qkv_b + t * stride_qkv_t + (2 * C + h * head_dim + d)

    out_offset = b * stride_out_b + h * stride_out_h + t * stride_out_t + d * stride_out_d

    q_val = tl.load(qkv_ptr + qkv_offset_q, mask=mask)
    k_val = tl.load(qkv_ptr + qkv_offset_k, mask=mask)
    v_val = tl.load(qkv_ptr + qkv_offset_v, mask=mask)

    tl.store(q_ptr + out_offset, q_val, mask=mask)
    tl.store(k_ptr + out_offset, k_val, mask=mask)
    tl.store(v_ptr + out_offset, v_val, mask=mask)


def fused_qkv_reshape(qkv, B, T, n_head, head_dim):
    C = n_head * head_dim
    q = torch.empty(B, n_head, T, head_dim, device=qkv.device, dtype=qkv.dtype)
    k = torch.empty(B, n_head, T, head_dim, device=qkv.device, dtype=qkv.dtype)
    v = torch.empty(B, n_head, T, head_dim, device=qkv.device, dtype=qkv.dtype)

    total = B * n_head * T * head_dim
    BLOCK_SIZE = 1024
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    fused_qkv_reshape_kernel[grid](
        qkv, q, k, v,
        B, T, C, n_head, head_dim,
        qkv.stride(0), qkv.stride(1), qkv.stride(2),
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return q, k, v


class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        assert n_embd % n_head == 0
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        self.register_buffer("bias", torch.tril(torch.ones(max_seqlen, max_seqlen))
                                     .view(1, 1, max_seqlen, max_seqlen))
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_pdrop = attn_pdrop

    def forward(self, x):
        B, T, C = x.size()
        head_dim = C // self.n_head

        qkv = self.c_attn(x)
        q, k, v = fused_qkv_reshape(qkv, B, T, self.n_head, head_dim)

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.attn_pdrop if self.training else 0.0,
            is_causal=True,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


batch_size = 64
max_seqlen = 1024
seq_len = 512
n_embd = 768
n_head = 8
attn_pdrop = 0.0
resid_pdrop = 0.0

def get_inputs():
    return [torch.randn(batch_size, seq_len, n_embd)]

def get_init_inputs():
    return [n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen]
