import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_extract_weight_scatter_kernel(
    down_ptr, out_ptr,
    sorted_expert_ids_ptr, sorted_orig_tokens_ptr, sorted_weights_ptr,
    expert_offsets_ptr,
    hidden, max_count, num_expanded,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    n_block = tl.program_id(1)

    if pid >= num_expanded:
        return

    expert_id = tl.load(sorted_expert_ids_ptr + pid).to(tl.int64)
    expert_offset = tl.load(expert_offsets_ptr + expert_id)
    pos = pid - expert_offset
    weight = tl.load(sorted_weights_ptr + pid)
    dst_idx = tl.load(sorted_orig_tokens_ptr + pid)

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs_n < hidden

    val = tl.load(
        down_ptr + expert_id * max_count * hidden + pos * hidden + offs_n,
        mask=mask, other=0.0,
    )
    weighted_val = (val * weight).to(tl.float32)
    tl.atomic_add(out_ptr + dst_idx * hidden + offs_n, weighted_val, mask=mask)


class Model(nn.Module):
    def __init__(self, num_experts: int = 8, hidden_dim: int = 1024, expert_dim: int = 3072):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim
        self.expert_up = nn.Parameter(torch.randn(num_experts, hidden_dim, expert_dim) * 0.02)
        self.expert_down = nn.Parameter(torch.randn(num_experts, expert_dim, hidden_dim) * 0.02)
        self._up_half = None
        self._down_half = None

    def _ensure_half(self):
        if self._up_half is None:
            self._up_half = self.expert_up.data.half()
            self._down_half = self.expert_down.data.half()

    def forward(self, x, expert_indices, expert_weights):
        batch, seq, hidden = x.shape
        top_k = expert_indices.shape[-1]
        total_tokens = batch * seq
        num_expanded = total_tokens * top_k
        device = x.device

        self._ensure_half()

        x_flat = x.reshape(total_tokens, hidden)
        idx_flat = expert_indices.reshape(-1)
        w_flat = expert_weights.reshape(-1)

        sorted_expert_ids, sort_idx = idx_flat.sort(stable=True)
        sorted_orig_tokens = (sort_idx // top_k).long()
        sorted_weights = w_flat[sort_idx]

        expert_counts = torch.bincount(sorted_expert_ids.int(), minlength=self.num_experts)
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=device)
        expert_offsets[1:] = torch.cumsum(expert_counts, 0)

        max_count = expert_counts.max().item()

        token_range = torch.arange(max_count, device=device)
        src_indices = (expert_offsets[:-1].unsqueeze(1) + token_range.unsqueeze(0)).clamp(max=num_expanded - 1)

        padded_orig = sorted_orig_tokens[src_indices.reshape(-1)]
        padded_x = x_flat[padded_orig].reshape(self.num_experts, max_count, hidden).half()

        up_result = torch.bmm(padded_x, self._up_half)
        down_result = torch.bmm(up_result, self._down_half)

        out_flat = torch.zeros(total_tokens, self.hidden_dim, device=device, dtype=x.dtype)
        BLOCK_N = 256
        n_blocks = (self.hidden_dim + BLOCK_N - 1) // BLOCK_N
        fused_extract_weight_scatter_kernel[(num_expanded, n_blocks)](
            down_result, out_flat,
            sorted_expert_ids.int(), sorted_orig_tokens, sorted_weights,
            expert_offsets,
            self.hidden_dim, max_count, num_expanded,
            BLOCK_N=BLOCK_N,
        )

        return out_flat.reshape(batch, seq, hidden)


def get_inputs():
    batch, seq, hidden = 4, 256, 1024
    top_k = 2
    num_experts = 8
    x = torch.randn(batch, seq, hidden)
    expert_indices = torch.randint(0, num_experts, (batch, seq, top_k))
    expert_weights = torch.softmax(torch.randn(batch, seq, top_k), dim=-1)
    return [x, expert_indices, expert_weights]


def get_init_inputs():
    return []
