import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr, targets_ptr, loss_ptr,
    num_classes: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    logits_row = logits_ptr + row * num_classes
    target = tl.load(targets_ptr + row)

    # Load logits in chunks of BLOCK_SIZE
    # First pass: find max for numerical stability
    max_val = float('-inf')
    for off in tl.static_range(0, num_classes, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < num_classes
        vals = tl.load(logits_row + cols, mask=mask, other=float('-inf'))
        max_val = tl.maximum(max_val, tl.max(vals, axis=0))

    # Second pass: compute sum of exp(x - max) and get target logit
    sum_exp = 0.0
    target_logit = 0.0
    for off in tl.static_range(0, num_classes, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < num_classes
        vals = tl.load(logits_row + cols, mask=mask, other=float('-inf'))
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)
        target_logit += tl.sum(tl.where(cols == target, vals, 0.0), axis=0)

    log_sum_exp = tl.log(sum_exp) + max_val
    loss = log_sum_exp - target_logit
    tl.store(loss_ptr + row, loss)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predictions, targets):
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]
        losses = torch.empty(batch_size, device=predictions.device, dtype=predictions.dtype)

        BLOCK_SIZE = 1024  # num_classes is 1024, so one iteration per loop

        cross_entropy_kernel[(batch_size,)](
            predictions, targets, losses,
            num_classes=num_classes,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )

        return losses.mean()


def get_inputs():
    return [torch.randn(4096, 1024, device='cuda'), torch.randint(0, 1024, (4096,), device='cuda')]


def get_init_inputs():
    return []
