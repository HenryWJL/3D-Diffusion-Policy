import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Literal
from torch_scatter import scatter_logsumexp


# def info_nce_loss(
#     query: Tensor,
#     pos_key: Tensor,
#     neg_key: Tensor,
#     batch_idx: Tensor,
#     temp: Optional[float] = 0.1,
#     reduction: Literal['none', 'mean', 'sum'] = 'none'
# ) -> Tensor:
#     """
#     Args:
#         query (torch.Tensor): query (N, D).
#         pos_key (torch.Tensor): positive keys (N, D).
#         neg_key (torch.Tensor): negative keys (N, K, D). 
#         temp (float, optional): temperature coefficient.
#     """
#     assert reduction in ['none', 'mean', 'sum']
#     B = int(batch_idx.max().item()) + 1

#     query = F.normalize(query, dim=-1)
#     pos_key = F.normalize(pos_key, dim=-1)
#     neg_key = F.normalize(neg_key, dim=-1)

#     pos_logits = (pos_key * query).sum(dim=1)  # (N,)
#     neg_logits = (neg_key * query.unsqueeze(1)).sum(dim=2).flatten()  # (N * K,)
#     batch_idx_repeat = batch_idx.repeat_interleave(neg_key.shape[1])  # (N * K,)
#     pos_logits /= temp
#     neg_logits /= temp

#     # log Σ exp(pos)
#     log_pos = scatter_logsumexp(
#         pos_logits,
#         batch_idx,
#         dim=0,
#         dim_size=B
#     )
#     # log Σ exp(pos + neg)
#     all_logits = torch.cat([pos_logits, neg_logits], dim=0)
#     all_batch_idx = torch.cat([batch_idx, batch_idx_repeat], dim=0)
#     log_all = scatter_logsumexp(
#         all_logits,
#         all_batch_idx,
#         dim=0,
#         dim_size=B
#     )
#     # InfoNCE loss per batch
#     loss = -(log_pos - log_all)
#     # Final loss
#     if reduction == 'none':
#         return loss
#     elif reduction == 'mean':
#         return loss.mean()
#     elif reduction == 'sum':
#         return loss.sum()


def info_nce_loss(
    query: Tensor,          # (N, D)
    pos_key: Tensor,        # (N, M, D)
    neg_key: Tensor,        # (N, K, D)
    batch_idx: Tensor,      # (N,)
    temp: float = 0.1,
    reduction: Literal['none', 'mean', 'sum'] = 'none'
) -> Tensor:
    """
    Multi-positive InfoNCE loss.

    For each query, computes InfoNCE against each positive key independently,
    then averages the loss over positives.

    Returns loss per batch element.
    """
    assert reduction in ['none', 'mean', 'sum']
    assert pos_key.dim() == 3

    N, M, D = pos_key.shape
    K = neg_key.shape[1]
    B = int(batch_idx.max().item()) + 1

    # Normalize
    query = F.normalize(query, dim=-1)                 # (N, D)
    pos_key = F.normalize(pos_key, dim=-1)             # (N, M, D)
    neg_key = F.normalize(neg_key, dim=-1)             # (N, K, D)

    # Expand query for positives
    query_pos = query.unsqueeze(1)                     # (N, 1, D)

    # Positive logits: (N, M)
    pos_logits = (pos_key * query_pos).sum(dim=-1) / temp

    # Negative logits: (N, K)
    neg_logits = (neg_key * query.unsqueeze(1)).sum(dim=-1) / temp

    # Repeat negatives for each positive
    neg_logits = neg_logits.unsqueeze(1).expand(N, M, K)   # (N, M, K)

    # Flatten over (N * M)
    pos_logits = pos_logits.reshape(-1)                 # (N*M,)
    neg_logits = neg_logits.reshape(-1)                 # (N*M*K,)

    # Batch indices
    batch_idx_pos = batch_idx.repeat_interleave(M)      # (N*M,)
    batch_idx_neg = batch_idx_pos.repeat_interleave(K)  # (N*M*K,)

    # log Σ exp(pos)
    log_pos = scatter_logsumexp(
        pos_logits,
        batch_idx_pos,
        dim=0,
        dim_size=B
    )

    # log Σ exp(pos + neg)
    all_logits = torch.cat([pos_logits, neg_logits], dim=0)
    all_batch_idx = torch.cat([batch_idx_pos, batch_idx_neg], dim=0)

    log_all = scatter_logsumexp(
        all_logits,
        all_batch_idx,
        dim=0,
        dim_size=B
    )

    # Loss per batch
    loss = -(log_pos - log_all)

    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()