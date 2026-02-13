"""
Visual In-Degree Centrality computation (Eq. 2-4).

Computes how much attention each visual token receives from other visual tokens,
which serves as a proxy for structural importance in the document layout.
"""

import torch


def compute_visual_centrality(
    layer_attn: torch.Tensor,
    visual_indices: torch.Tensor,
    agg_mode: str = "mean",
) -> torch.Tensor:
    """
    Compute visual in-degree centrality from a single attention layer.

    For each head, extracts the visual-to-visual attention sub-matrix and
    computes column-wise sums (in-degree centrality). Then aggregates across
    heads using mean (Eq. 3) or max (Eq. 4).

    Args:
        layer_attn: Attention weights of shape ``[batch, num_heads, seq_len, seq_len]``.
            Only the first batch element is used.
        visual_indices: 1-D tensor of integer indices identifying visual token
            positions in the sequence.
        agg_mode: Head aggregation strategy — ``"mean"`` (SAP-Mean, Eq. 3) or
            ``"max"`` (SAP-Max, Eq. 4).

    Returns:
        Centrality scores of shape ``[num_visual_tokens]``.
    """
    if agg_mode not in ("mean", "max"):
        raise ValueError(f"agg_mode must be 'mean' or 'max', got '{agg_mode}'")

    # Extract visual-to-visual attention for all heads at once
    # layer_attn[0]: [num_heads, seq_len, seq_len]
    attn = layer_attn[0]  # [H, S, S]
    # Index rows and columns corresponding to visual tokens
    visual_attn = attn[:, visual_indices][:, :, visual_indices]  # [H, V, V]

    # In-degree centrality: sum over rows (source tokens) for each column (target)
    # Eq. 2: c_j = sum_i A_{ij}
    head_centralities = visual_attn.sum(dim=-2)  # [H, V]

    if agg_mode == "mean":
        return head_centralities.mean(dim=0)  # [V]
    else:
        return head_centralities.max(dim=0).values  # [V]


def compute_visual_centrality_batch(
    layer_attn: torch.Tensor,
    visual_indices: torch.Tensor,
    agg_mode: str = "mean",
) -> torch.Tensor:
    """
    Batch variant of :func:`compute_visual_centrality`.

    Computes centrality for every element in the batch dimension.

    Args:
        layer_attn: Attention weights ``[batch, num_heads, seq_len, seq_len]``.
        visual_indices: 1-D tensor of visual token indices (shared across batch).
        agg_mode: ``"mean"`` or ``"max"``.

    Returns:
        Centrality scores ``[batch, num_visual_tokens]``.
    """
    if agg_mode not in ("mean", "max"):
        raise ValueError(f"agg_mode must be 'mean' or 'max', got '{agg_mode}'")

    visual_attn = layer_attn[:, :, visual_indices][:, :, :, visual_indices]  # [B, H, V, V]
    head_centralities = visual_attn.sum(dim=-2)  # [B, H, V]

    if agg_mode == "mean":
        return head_centralities.mean(dim=1)  # [B, V]
    else:
        return head_centralities.max(dim=1).values  # [B, V]
