"""
Score Retention (SR) — per-layer compression diagnostic used to drive
SR-guided window selection.

SR at layer :math:`l` is the MaxSim score of a query against the document
pruned using *only* layer :math:`l`'s centrality, divided by the MaxSim
score against the unpruned document. Averaging SR across a calibration
set of (query, document) pairs yields a per-layer SR curve whose shape is
used by :mod:`sap.window_selection` to locate the Structural Plateau.

This module is the library counterpart of the
``experiments/per_layer_osr.py`` script — it implements the same algorithm
as a pure tensor/numpy API and contains no model loading.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .centrality import compute_visual_centrality
from .evaluation import compute_osr, maxsim_score
from .pruning import prune_embeddings_with_indices


def compute_per_layer_sr(
    attentions: Sequence[torch.Tensor],
    visual_indices: torch.Tensor,
    visual_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    ratio: float,
    agg_mode: str = "mean",
) -> np.ndarray:
    """
    Per-layer Score Retention for a single (query, document) pair.

    For each layer ``l``, score the document's visual tokens using
    only layer ``l``'s in-degree centrality, keep the top *ratio* fraction,
    then compute SR against the full unpruned document.

    Args:
        attentions: Sequence of attention tensors, one per layer.
            Each tensor has shape ``[batch, num_heads, seq_len, seq_len]``.
        visual_indices: 1-D tensor of visual token positions.
        visual_embeddings: Document visual token embeddings
            ``[num_visual, dim]`` — typically ``model_output[0, visual_indices]``.
        query_embeddings: Query token embeddings ``[num_query, dim]``.
        ratio: Fraction of visual tokens to keep (e.g. ``0.10`` keeps 10%).
        agg_mode: Head aggregation — ``"mean"`` or ``"max"``.

    Returns:
        Per-layer SR values, shape ``[num_layers]``.
    """
    num_layers = len(attentions)
    sr = np.zeros(num_layers, dtype=np.float32)
    for l in range(num_layers):
        layer_scores = compute_visual_centrality(
            attentions[l], visual_indices, agg_mode=agg_mode
        )
        pruned, _ = prune_embeddings_with_indices(
            visual_embeddings, layer_scores, ratio
        )
        sr[l] = compute_osr(query_embeddings, visual_embeddings, pruned)
    return sr


def compute_random_sr_baseline(
    visual_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    ratio: float,
    num_seeds: int = 5,
    generator: Optional[torch.Generator] = None,
) -> float:
    """
    Random-pruning SR baseline used to anchor SR curves.

    Args:
        visual_embeddings: ``[num_visual, dim]``.
        query_embeddings: ``[num_query, dim]``.
        ratio: Retention ratio.
        num_seeds: Number of random selections to average.
        generator: Optional torch generator for reproducibility.

    Returns:
        Mean SR over *num_seeds* random selections.
    """
    n = visual_embeddings.shape[0]
    k = max(1, round(n * ratio))
    k = min(k, n)
    device = visual_embeddings.device
    accum = 0.0
    for _ in range(num_seeds):
        perm = torch.randperm(n, generator=generator, device=device)[:k]
        perm, _ = perm.sort()
        accum += compute_osr(
            query_embeddings, visual_embeddings, visual_embeddings[perm]
        )
    return accum / num_seeds


def aggregate_calibration_sr(
    per_pair_sr: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Aggregate per-pair SR curves across a calibration set.

    Args:
        per_pair_sr: List of per-layer SR arrays, one per calibration pair.
            All arrays must share the same length.

    Returns:
        Dict with keys ``"mean"`` (per-layer mean SR) and ``"std"`` (per-layer std).
    """
    stacked = np.stack(per_pair_sr, axis=0)  # [N, L]
    return {
        "mean": stacked.mean(axis=0).astype(np.float32),
        "std": stacked.std(axis=0).astype(np.float32),
    }
