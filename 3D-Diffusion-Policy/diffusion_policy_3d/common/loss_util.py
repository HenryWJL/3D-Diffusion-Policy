import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal
from torch_scatter import scatter_logsumexp


def info_nce_loss(
    query: Tensor,
    pos_key: Tensor,
    neg_key: Tensor,
    batch_idx: Tensor,
    temp: Optional[float] = 0.1,
    reduction: Literal['none', 'mean', 'sum'] = 'none'
) -> Tensor:
    """
    Args:
        query (torch.Tensor): query (N, D).
        pos_key (torch.Tensor): positive keys (N, D).
        neg_key (torch.Tensor): negative keys (N, K, D). 
        temp (float, optional): temperature coefficient.
    """
    assert reduction in ['none', 'mean', 'sum']
    B = int(batch_idx.max().item()) + 1

    query = F.normalize(query, dim=-1)
    pos_key = F.normalize(pos_key, dim=-1)
    neg_key = F.normalize(neg_key, dim=-1)

    pos_logits = (pos_key * query).sum(dim=1)  # (N,)
    neg_logits = (neg_key * query.unsqueeze(1)).sum(dim=2).flatten()  # (N * K,)
    batch_idx_repeat = batch_idx.repeat_interleave(neg_key.shape[1])  # (N * K,)
    pos_logits /= temp
    neg_logits /= temp

    # log Σ exp(pos)
    log_pos = scatter_logsumexp(
        pos_logits,
        batch_idx,
        dim=0,
        dim_size=B
    )
    # log Σ exp(pos + neg)
    all_logits = torch.cat([pos_logits, neg_logits], dim=0)
    all_batch_idx = torch.cat([batch_idx, batch_idx_repeat], dim=0)
    log_all = scatter_logsumexp(
        all_logits,
        all_batch_idx,
        dim=0,
        dim_size=B
    )
    # InfoNCE loss per batch
    loss = -(log_pos - log_all)
    # Final loss
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()