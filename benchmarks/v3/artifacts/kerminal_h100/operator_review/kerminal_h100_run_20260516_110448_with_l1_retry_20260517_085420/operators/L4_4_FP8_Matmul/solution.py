import torch
import torch.nn as nn
import triton
import triton.language as tl

FP8_MAX_E4M3 = 448.0
FP8_MAX_E5M2 = 57344.0


@triton.jit
def _amax_kernel(X, Out, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    amax = tl.max(tl.abs(x))
    tl.atomic_max(Out, amax)


@triton.jit
def _quantize_fp8_kernel(
    X, Out, Scale,
    N: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    scale = tl.load(Scale)
    x = tl.load(X + offs, mask=mask, other=0.0)
    x_scaled = x * scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    tl.store(Out + offs, x_clamped.to(tl.float8e4nv), mask=mask)


def quantize_to_fp8_triton(x: torch.Tensor, fp8_max: float = FP8_MAX_E4M3):
    x_flat = x.reshape(-1)
    N = x_flat.numel()
    BLOCK_SIZE = 4096

    amax = x.abs().max()
    scale = fp8_max / amax.clamp(min=1e-12)

    out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    out_flat = out.reshape(-1)
    grid_quant = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _quantize_fp8_kernel[grid_quant](x_flat, out_flat, scale, N, fp8_max, BLOCK_SIZE)

    scale_inv = (1.0 / scale).to(torch.float32)
    return out, scale_inv


class Model(nn.Module):
    def __init__(self, M: int, K: int, N: int, use_e4m3: bool = True):
        super().__init__()
        self.M = M
        self.K = K
        self.N = N
        self.use_e4m3 = use_e4m3

        if use_e4m3:
            self.fp8_dtype = torch.float8_e4m3fn
            self.fp8_max = FP8_MAX_E4M3
        else:
            self.fp8_dtype = torch.float8_e5m2
            self.fp8_max = FP8_MAX_E5M2

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(1337)
        self.weight = nn.Parameter(torch.randn(K, N) * 0.02)
        torch.random.set_rng_state(rng_state)

        self._w_fp8 = None
        self._w_scale_inv = None

    def _prepare_weight(self):
        w_t = self.weight.t().contiguous()
        amax = w_t.abs().max()
        w_scale = self.fp8_max / amax.clamp(min=1e-12)
        w_scaled = w_t * w_scale
        w_clamped = w_scaled.clamp(-self.fp8_max, self.fp8_max)
        self._w_fp8 = w_clamped.to(self.fp8_dtype)
        self._w_scale_inv = (1.0 / w_scale).to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        if self._w_fp8 is None or self.training:
            self._prepare_weight()

        x_2d = x.reshape(-1, self.K)

        x_fp8, x_scale_inv = quantize_to_fp8_triton(x_2d, self.fp8_max)

        out = torch._scaled_mm(
            x_fp8,
            self._w_fp8.t(),
            scale_a=x_scale_inv,
            scale_b=self._w_scale_inv,
            out_dtype=input_dtype,
        )

        return out.view(batch_size, seq_len, self.N)


batch_size = 8
seq_len = 2048
M = batch_size * seq_len
K = 4096
N = 4096
use_e4m3 = True


def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]


def get_init_inputs():
    return [M, K, N, use_e4m3]
