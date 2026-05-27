"""Bi-directional InfoNCE loss with optional cross-GPU negative gathering."""

import torch
import torch.distributed as dist
import torch.nn.functional as F


class AllGatherWithGrad(torch.autograd.Function):
    """All-gather embeddings across DDP ranks while preserving gradients.

    The default `dist.all_gather` detaches gradients. CLIP-style contrastive
    training requires gradients to flow back to the producing rank's own slice,
    which we accomplish by splicing the local tensor (with grad) back in.
    """

    @staticmethod
    def forward(ctx, tensor):
        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor.detach())
        # Keep gradient path on this rank's local tensor
        gathered[ctx.rank] = tensor
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        bs = grad_output.shape[0] // ctx.world_size
        return grad_output[ctx.rank * bs:(ctx.rank + 1) * bs]


def all_gather_grad(t: torch.Tensor) -> torch.Tensor:
    """Gather a (B, D) tensor across ranks → (B*world_size, D), gradients flow."""
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        return AllGatherWithGrad.apply(t)
    return t


def bidirectional_infonce(pred: torch.Tensor, target: torch.Tensor,
                          temperature: float = 0.07,
                          gather_across_ranks: bool = True) -> torch.Tensor:
    """Bi-directional InfoNCE loss with optional cross-GPU all-gather negatives.

    Args:
        pred:    (B, D) predicted embeddings from predictor
        target:  (B, D) target embeddings from y_encoder
        temperature: softmax temperature
        gather_across_ranks: if True and DDP active, gather across all ranks so
                             effective negatives are bs * world_size − 1.
    Returns:
        scalar loss
    """
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)

    if gather_across_ranks:
        # gather both, keeping local slice's gradient
        pred_global = all_gather_grad(pred_n)
        tgt_global = all_gather_grad(tgt_n)
    else:
        pred_global, tgt_global = pred_n, tgt_n

    # Each rank computes the full (B_local, B_global) sim matrix.
    # Positive index for rank r is rank * B_local + i_local.
    if dist.is_available() and dist.is_initialized() and gather_across_ranks and dist.get_world_size() > 1:
        rank = dist.get_rank()
        bs_local = pred_n.shape[0]
        # logits_p2t: predictor → target, shape (B_local, B_global)
        logits_p2t = (pred_n @ tgt_global.T) / temperature
        # logits_t2p: target → predictor, shape (B_local, B_global)
        logits_t2p = (tgt_n @ pred_global.T) / temperature
        labels = torch.arange(bs_local, device=pred.device) + rank * bs_local
    else:
        logits_p2t = (pred_n @ tgt_n.T) / temperature
        logits_t2p = logits_p2t.T
        labels = torch.arange(logits_p2t.shape[0], device=pred.device)

    loss_p2t = F.cross_entropy(logits_p2t, labels)
    loss_t2p = F.cross_entropy(logits_t2p, labels)
    return (loss_p2t + loss_t2p) / 2.0
