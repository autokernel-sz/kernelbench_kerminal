import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_residual_layernorm_kernel(
    OUT, ATTN, RESIDUAL, WEIGHT, BIAS,
    N,
    stride_out, stride_attn, stride_res,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    attn_ptr = ATTN + row * stride_attn
    res_ptr = RESIDUAL + row * stride_res
    out_ptr = OUT + row * stride_out

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    a = tl.load(attn_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(res_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    x = a + r

    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = xc * rstd

    w = tl.load(WEIGHT + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)
    out = xn * w + b

    tl.store(out_ptr + cols, out, mask=mask)


class Model(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        seq_len = H * W

        # (B, C, H, W) -> (seq_len*B, embed_dim)
        x_seq = x.view(B, C, seq_len).permute(2, 0, 1).contiguous()
        x_flat = x_seq.view(seq_len * B, self.embed_dim)

        # QKV projection: (seq_len*B, embed_dim) @ (embed_dim, 3*embed_dim) + bias
        qkv = torch.addmm(self.attn.in_proj_bias, x_flat, self.attn.in_proj_weight.t())
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (B, num_heads, seq_len, head_dim)
        q = q.view(seq_len, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        k = k.view(seq_len, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        v = v.view(seq_len, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        # Flash attention in fp16
        with torch.amp.autocast('cuda', dtype=torch.float16):
            attn_out = F.scaled_dot_product_attention(q, k, v)

        attn_out = attn_out.float()

        # (B, num_heads, seq_len, head_dim) -> (seq_len*B, embed_dim)
        attn_out = attn_out.permute(2, 0, 1, 3).reshape(seq_len * B, self.embed_dim)

        # Output projection
        attn_out = torch.addmm(self.attn.out_proj.bias, attn_out, self.attn.out_proj.weight.t())

        # Fused residual + LayerNorm via Triton
        out = torch.empty_like(x_flat)
        n_rows = seq_len * B
        BLOCK_SIZE = triton.next_power_of_2(self.embed_dim)

        fused_residual_layernorm_kernel[(n_rows,)](
            out, attn_out, x_flat,
            self.norm.weight, self.norm.bias,
            self.embed_dim,
            out.stride(0), attn_out.stride(0), x_flat.stride(0),
            eps=self.norm.eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # (seq_len*B, embed_dim) -> (B, C, H, W)
        out = out.view(seq_len, B, C).permute(1, 2, 0).view(B, C, H, W)
        return out


embed_dim = 128
num_heads = 4
batch_size = 2
num_channels = embed_dim
image_height = 128
image_width = 128


def get_inputs():
    return [torch.randn(batch_size, num_channels, image_height, image_width)]


def get_init_inputs():
    return [embed_dim, num_heads]
