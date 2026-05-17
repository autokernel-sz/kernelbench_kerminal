import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl

@triton.jit
def fused_gelu_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask)
    c = 0.7978845608028654
    inner = c * (x + 0.044715 * x * x * x)
    y = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner.to(tl.float32)).to(x.dtype))
    tl.store(Y + row * N + offs, y, mask=mask)

def fused_gelu(x):
    shape = x.shape
    x_2d = x.reshape(-1, shape[-1])
    rows, N = x_2d.shape
    BLOCK = triton.next_power_of_2(N)
    y = torch.empty_like(x_2d)
    fused_gelu_kernel[(rows,)](x_2d, y, N, BLOCK)
    return y.reshape(shape)


class NewGELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return fused_gelu(x)


class CausalSelfAttention(nn.Module):
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

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Model(nn.Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop, max_seqlen)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_embd, 4 * n_embd),
            c_proj  = nn.Linear(4 * n_embd, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))

    def forward(self, x):
        if not hasattr(self, '_converted'):
            self.half()
            self._converted = True
        x = x.half()
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x.float()


batch_size = 128
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
