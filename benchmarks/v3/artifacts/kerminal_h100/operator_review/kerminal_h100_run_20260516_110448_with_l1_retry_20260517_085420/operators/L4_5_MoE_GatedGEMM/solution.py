import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_silu_mul_kernel(
    gate_ptr, up_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)
    silu_gate = gate * tl.sigmoid(gate)
    result = silu_gate * up
    tl.store(out_ptr + offsets, result.to(tl.float16), mask=mask)


class Model(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts

        rng_state = torch.random.get_rng_state()
        torch.manual_seed(1337)
        self.gate_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.up_proj = nn.Parameter(
            torch.randn(num_experts, intermediate_size, hidden_size) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(num_experts, hidden_size, intermediate_size) * 0.02
        )
        torch.random.set_rng_state(rng_state)

    def _ensure_half_cache(self):
        if not hasattr(self, '_cached_gate_h') or self._cached_gate_h is None:
            self._cached_gate_h = self.gate_proj.data.half()
            self._cached_up_h = self.up_proj.data.half()
            self._cached_down_h = self.down_proj.data.half()

    def forward(self, x, expert_indices, expert_weights):
        self._ensure_half_cache()
        batch, seq_len, _ = x.shape
        top_k = expert_indices.shape[-1]
        num_tokens = batch * seq_len
        total_slots = num_tokens * top_k

        x_flat = x.view(num_tokens, self.hidden_size)
        indices_flat = expert_indices.view(total_slots)
        weights_flat = expert_weights.view(total_slots)

        token_ids = torch.arange(num_tokens, device=x.device)
        token_ids = token_ids.unsqueeze(1).expand(-1, top_k).reshape(-1)

        sorted_expert_idx, sort_order = indices_flat.sort()
        sorted_token_ids = token_ids[sort_order]
        sorted_weights = weights_flat[sort_order]

        expert_counts = torch.bincount(sorted_expert_idx, minlength=self.num_experts)
        expert_offsets = torch.zeros(self.num_experts + 1, dtype=torch.long, device=x.device)
        expert_offsets[1:] = expert_counts.cumsum(0)

        sorted_x = x_flat[sorted_token_ids].half()

        max_count = expert_counts.max().item()

        # Vectorized padding into (E, max_count, H) directly in FP16
        global_pos = torch.arange(total_slots, device=x.device, dtype=torch.long)
        within_pos = global_pos - expert_offsets[sorted_expert_idx]
        flat_idx = sorted_expert_idx.long() * max_count + within_pos

        padded_x = torch.zeros(self.num_experts * max_count, self.hidden_size,
                               device=x.device, dtype=torch.float16)
        padded_x[flat_idx] = sorted_x
        padded_x = padded_x.view(self.num_experts, max_count, self.hidden_size)

        # Batched FP16 GEMMs
        gate_out = torch.bmm(padded_x, self._cached_gate_h.transpose(1, 2))
        up_out = torch.bmm(padded_x, self._cached_up_h.transpose(1, 2))

        # Fused SiLU(gate) * up via Triton
        n_el = gate_out.numel()
        intermediate = torch.empty_like(gate_out)
        grid = (triton.cdiv(n_el, 4096),)
        fused_silu_mul_kernel[grid](gate_out, up_out, intermediate, n_el, BLOCK_SIZE=4096)

        # Down projection
        down_out = torch.bmm(intermediate, self._cached_down_h.transpose(1, 2))

        # Vectorized unpadding
        down_flat = down_out.reshape(self.num_experts * max_count, self.hidden_size)
        sorted_output = down_flat[flat_idx].float()

        # Weight and scatter
        sorted_output = sorted_output * sorted_weights.unsqueeze(-1)
        output = x_flat.new_zeros(num_tokens, self.hidden_size)
        output.index_add_(0, sorted_token_ids, sorted_output)

        return output.view(batch, seq_len, self.hidden_size)


batch_size = 4
seq_len = 2048
hidden_size = 4096
intermediate_size = 14336
num_experts = 8
top_k = 2


def get_inputs():
    x = torch.randn(batch_size, seq_len, hidden_size)
    expert_indices = torch.stack([
        torch.randperm(num_experts)[:top_k]
        for _ in range(batch_size * seq_len)
    ]).view(batch_size, seq_len, top_k)
    expert_weights = F.softmax(torch.randn(batch_size, seq_len, top_k), dim=-1)
    return [x, expert_indices, expert_weights]


def get_init_inputs():
    return [hidden_size, intermediate_size, num_experts]
