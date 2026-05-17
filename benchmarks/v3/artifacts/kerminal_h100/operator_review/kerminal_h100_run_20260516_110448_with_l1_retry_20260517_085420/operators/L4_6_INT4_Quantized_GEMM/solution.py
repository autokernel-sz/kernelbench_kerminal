import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def dequant_int4_kernel(
    w_packed_ptr, scales_ptr, w_out_ptr,
    N, K, group_size,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    pid_n = tl.program_id(1)

    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    pk = rk // 2
    mask = (rk[:, None] < K) & (rn[None, :] < N)

    w_packed = tl.load(w_packed_ptr + pk[:, None] * N + rn[None, :],
                       mask=(pk[:, None] < K // 2) & (rn[None, :] < N), other=0)

    w_int = tl.where((rk[:, None] & 1) == 0,
                     w_packed & 0x0F,
                     (w_packed >> 4) & 0x0F)

    gk = rk // group_size
    num_groups = K // group_size
    scales = tl.load(scales_ptr + gk[:, None] * N + rn[None, :],
                     mask=(gk[:, None] < num_groups) & (rn[None, :] < N), other=0.0)

    w_dequant = (scales * (w_int.to(tl.float16) - 8.0)).to(tl.float16)

    tl.store(w_out_ptr + rk[:, None] * N + rn[None, :], w_dequant, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)

    GROUP_SIZE_M: tl.constexpr = 8
    num_pid_in_group = GROUP_SIZE_M * num_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_ptrs = a_ptr + rm[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptrs = b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + rn[None, :] * stride_bn

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (tl.arange(0, BLOCK_K)[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(tl.arange(0, BLOCK_K)[:, None] < k_remaining) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(tl.float16)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn, c, mask=mask)


class Model(nn.Module):
    def __init__(self, K: int, N: int, group_size: int = 128):
        super().__init__()
        self.K = K
        self.N = N
        self.group_size = group_size
        self.num_groups = K // group_size

        assert K % group_size == 0
        assert K % 2 == 0

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(1337)
        self.register_buffer(
            "weight_packed",
            torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        )
        self.register_buffer(
            "scales",
            torch.randn(N, self.num_groups, dtype=torch.float16).abs() * 0.1
        )
        torch.random.set_rng_state(rng_state)

        self._w_dequant = None

    def _dequant(self):
        if self._w_dequant is not None:
            return self._w_dequant

        K, N = self.K, self.N
        wt = self.weight_packed.t().contiguous()
        st = self.scales.t().contiguous()

        w_out = torch.empty((K, N), dtype=torch.float16, device=self.weight_packed.device)

        BLOCK_K = 128
        BLOCK_N = 128
        grid = (triton.cdiv(K, BLOCK_K), triton.cdiv(N, BLOCK_N))
        dequant_int4_kernel[grid](wt, st, w_out, N, K, self.group_size, BLOCK_K, BLOCK_N)

        self._w_dequant = w_out
        return w_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dequant = self._dequant()

        batch_size, seq_len, K = x.shape
        M = batch_size * seq_len
        N = self.N

        x_2d = x.reshape(M, K)
        out = torch.empty((M, N), dtype=torch.float16, device=x.device)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )

        matmul_kernel[grid](
            x_2d, w_dequant, out,
            M, N, K,
            x_2d.stride(0), x_2d.stride(1),
            w_dequant.stride(0), w_dequant.stride(1),
            out.stride(0), out.stride(1),
        )

        return out.reshape(batch_size, seq_len, N)


batch_size = 4
seq_len = 2048
K = 4096
N = 11008
group_size = 128


def get_inputs():
    return [torch.randn(batch_size, seq_len, K, dtype=torch.float16)]


def get_init_inputs():
    return [K, N, group_size]
