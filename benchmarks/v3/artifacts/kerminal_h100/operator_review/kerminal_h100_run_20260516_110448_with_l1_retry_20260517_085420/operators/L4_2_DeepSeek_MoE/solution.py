import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import triton
import triton.language as tl


@triton.jit
def fused_silu_mul_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    g = tl.load(gate_ptr + offs, mask=mask).to(tl.float32)
    u = tl.load(up_ptr + offs, mask=mask).to(tl.float32)
    result = (g * tl.sigmoid(g)) * u
    tl.store(out_ptr + offs, result.to(out_ptr.dtype.element_ty), mask=mask)


def fused_silu_mul(gate, up):
    out = torch.empty_like(gate)
    n = gate.numel()
    BLOCK = 1024
    fused_silu_mul_kernel[(triton.cdiv(n, BLOCK),)](gate, up, out, n, BLOCK=BLOCK)
    return out


def _mm(a, b):
    return torch.einsum('ij,jk->ik', a, b)


class MoEGate(nn.Module):
    def __init__(self, hidden_size, n_routed_experts, num_experts_per_tok,
                 n_group, topk_group, routed_scaling_factor=1.0, norm_topk_prob=True):
        super().__init__()
        self.top_k = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_group = n_group
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.weight = nn.Parameter(torch.empty(n_routed_experts, hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(n_routed_experts))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = torch.einsum('ij,kj->ik', hidden_states.float(), self.weight.float())
        scores = logits.sigmoid()
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(bsz * seq_len, self.n_group, -1)
            .topk(2, dim=-1)[0].sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(bsz * seq_len, -1)
        )
        tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        _, topk_idx = torch.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = scores.gather(1, topk_idx)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class Model(nn.Module):
    def __init__(self, hidden_size, intermediate_size, n_routed_experts,
                 num_experts_per_tok, n_group, topk_group,
                 n_shared_experts=0, routed_scaling_factor=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts

        self.gate_proj = nn.Parameter(
            torch.randn(n_routed_experts, intermediate_size, hidden_size) * 0.02)
        self.up_proj = nn.Parameter(
            torch.randn(n_routed_experts, intermediate_size, hidden_size) * 0.02)
        self.down_proj = nn.Parameter(
            torch.randn(n_routed_experts, hidden_size, intermediate_size) * 0.02)

        self.gate = MoEGate(
            hidden_size=hidden_size, n_routed_experts=n_routed_experts,
            num_experts_per_tok=num_experts_per_tok, n_group=n_group,
            topk_group=topk_group, routed_scaling_factor=routed_scaling_factor)

        if n_shared_experts > 0:
            shared_intermediate = intermediate_size * n_shared_experts
            self.shared_gate_proj = nn.Linear(hidden_size, shared_intermediate, bias=False)
            self.shared_up_proj = nn.Linear(hidden_size, shared_intermediate, bias=False)
            self.shared_down_proj = nn.Linear(shared_intermediate, hidden_size, bias=False)
        else:
            self.shared_gate_proj = None

        self._weights_cached = False

    def _cache_weights(self):
        gate_t = self.gate_proj.data.transpose(1, 2).contiguous()
        up_t = self.up_proj.data.transpose(1, 2).contiguous()
        self._gate_up_t = torch.cat([gate_t, up_t], dim=2)
        self._down_proj_t = self.down_proj.data.transpose(1, 2).contiguous()
        if self.shared_gate_proj is not None:
            sh_gate_t = self.shared_gate_proj.weight.data.t().contiguous()
            sh_up_t = self.shared_up_proj.weight.data.t().contiguous()
            self._sh_gate_up_w = torch.cat([sh_gate_t, sh_up_t], dim=1)
            self._sh_down_w = self.shared_down_proj.weight.data.t().contiguous()
        self._weights_cached = True

    def forward(self, hidden_states):
        assert not self.training
        if not self._weights_cached:
            self._cache_weights()

        orig_shape = hidden_states.shape
        bsz, seq_len, _ = orig_shape
        inter = self.intermediate_size

        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_flat = hidden_states.view(-1, self.hidden_size)
        num_tokens = hidden_flat.shape[0]

        flat_topk_idx = topk_idx.view(-1)
        sorted_expert_ids, sorted_order = flat_topk_idx.sort()

        expert_counts = torch.bincount(flat_topk_idx, minlength=self.n_routed_experts)
        expert_boundaries = torch.zeros(self.n_routed_experts + 1, dtype=torch.int64,
                                        device=hidden_flat.device)
        torch.cumsum(expert_counts, dim=0, out=expert_boundaries[1:])

        token_indices = sorted_order // self.num_experts_per_tok
        slot_indices = sorted_order % self.num_experts_per_tok

        output = torch.zeros(num_tokens, self.hidden_size, dtype=hidden_flat.dtype,
                             device=hidden_flat.device)

        for e in range(self.n_routed_experts):
            start = expert_boundaries[e].item()
            end = expert_boundaries[e + 1].item()
            if start == end:
                continue

            tok_ids = token_indices[start:end]
            sl_ids = slot_indices[start:end]
            inp = hidden_flat[tok_ids]

            gate_up = _mm(inp, self._gate_up_t[e])
            intermediate = fused_silu_mul(gate_up[:, :inter].contiguous(),
                                          gate_up[:, inter:].contiguous())
            expert_out = _mm(intermediate, self._down_proj_t[e])

            weights = topk_weight[tok_ids, sl_ids].unsqueeze(-1)
            output.scatter_add_(0, tok_ids.unsqueeze(-1).expand_as(expert_out),
                                expert_out * weights)

        y = output.view(*orig_shape)

        if self.shared_gate_proj is not None:
            identity_flat = hidden_states.view(-1, self.hidden_size)
            sh_gate_up = _mm(identity_flat, self._sh_gate_up_w)
            sh_inter = fused_silu_mul(sh_gate_up[:, :inter * self.n_shared_experts],
                                      sh_gate_up[:, inter * self.n_shared_experts:])
            sh_out = _mm(sh_inter, self._sh_down_w)
            y = y + sh_out.view(*orig_shape)

        return y


batch_size = 4
seq_len = 2048
hidden_size = 2048
intermediate_size = 1408
n_routed_experts = 64
num_experts_per_tok = 8
n_group = 8
topk_group = 4
n_shared_experts = 2
routed_scaling_factor = 2.5


def get_inputs():
    return [torch.randn(batch_size, seq_len, hidden_size)]


def get_init_inputs():
    return [
        hidden_size, intermediate_size, n_routed_experts,
        num_experts_per_tok, n_group, topk_group,
        n_shared_experts, routed_scaling_factor,
    ]
