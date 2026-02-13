"""
Pruning helpers: select top-k visual token embeddings given importance scores.
"""

from typing import List, Optional, Tuple

import torch


def prune_embeddings(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    """
    Keep the top-*ratio* fraction of visual token embeddings ranked by score.

    Args:
        embeddings: Visual token embeddings ``[num_visual, dim]``.
        scores: Importance scores ``[num_visual]`` (higher = more important).
        ratio: Fraction of tokens to **retain** (e.g. ``0.5`` keeps 50%).

    Returns:
        Pruned embeddings ``[k, dim]`` where ``k = max(1, round(num_visual * ratio))``.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    num_tokens = embeddings.shape[0]
    k = max(1, round(num_tokens * ratio))
    k = min(k, num_tokens)

    _, topk_indices = torch.topk(scores, k, sorted=False)
    # Sort indices to preserve positional order
    topk_indices = topk_indices.sort().values
    return embeddings[topk_indices]


def prune_embeddings_with_indices(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Like :func:`prune_embeddings` but also returns the selected indices.

    Returns:
        ``(pruned_embeddings, selected_indices)`` — both sorted by position.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    num_tokens = embeddings.shape[0]
    k = max(1, round(num_tokens * ratio))
    k = min(k, num_tokens)

    _, topk_indices = torch.topk(scores, k, sorted=False)
    topk_indices = topk_indices.sort().values
    return embeddings[topk_indices], topk_indices


def prune_embeddings_batch(
    embeddings_list: List[torch.Tensor],
    scores_list: List[torch.Tensor],
    ratio: float,
) -> List[torch.Tensor]:
    """
    Apply :func:`prune_embeddings` to a list of documents.

    Args:
        embeddings_list: Per-document embeddings, each ``[num_visual_i, dim]``.
        scores_list: Per-document scores, each ``[num_visual_i]``.
        ratio: Retention ratio.

    Returns:
        List of pruned embedding tensors.
    """
    return [
        prune_embeddings(emb, sc, ratio)
        for emb, sc in zip(embeddings_list, scores_list)
    ]
