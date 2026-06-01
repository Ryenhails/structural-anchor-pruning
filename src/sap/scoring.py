"""
SAP scoring pipeline (Eq. 5-6) and default layer selection (Table 4).

Integrates per-layer centrality scores across the target layer window
by averaging, producing the final importance score for each visual token.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from .centrality import compute_visual_centrality

# Default target layers per model family. From v2 of the paper, these are the
# windows selected by the SR-guided procedure on the ColPali training corpus,
# validated against a 9-window brute-force search (Section 4.3).
# The window position depends on the backbone: ColPali / ColQwen2 land in the
# 60-80% relative-depth range, Jina v4 in the 70-90% range. For unknown
# backbones, run the SR-guided window selection procedure from the paper's
# experiments directory (`per_layer_osr.py` + `analyze_osr_window_rules.py`).
DEFAULT_LAYERS: Dict[str, List[int]] = {
    "paligemma": [11, 12, 13, 14],                       # 18 layers, 60-80% window
    "colpali":   [11, 12, 13, 14],
    "qwen2":     [18, 19, 20, 21, 22, 23],               # 28 layers, 60-80% window
    "colqwen2":  [18, 19, 20, 21, 22, 23],
    "jina":      [26, 27, 28, 29, 30, 31, 32, 33],       # 36 layers, 70-90% window
}


def get_default_layers(model_name_or_type: str) -> List[int]:
    """
    Return default target layers for a known model family.

    Accepts either a short key (``"colpali"``, ``"colqwen2"``, ``"jina"``) or
    a HuggingFace model identifier that contains one of those keywords.

    Args:
        model_name_or_type: Model family key or full model name.

    Returns:
        List of 0-indexed layer indices.

    Raises:
        ValueError: If the model type cannot be inferred.
    """
    key = model_name_or_type.lower()
    # Direct key lookup
    if key in DEFAULT_LAYERS:
        return list(DEFAULT_LAYERS[key])
    # Heuristic from HF model name
    if "jina" in key:
        return list(DEFAULT_LAYERS["jina"])
    if "qwen2" in key or "colqwen" in key:
        return list(DEFAULT_LAYERS["qwen2"])
    if "pali" in key or "colpali" in key:
        return list(DEFAULT_LAYERS["colpali"])
    raise ValueError(
        f"Cannot infer default layers for '{model_name_or_type}'. "
        f"Known types: {list(DEFAULT_LAYERS.keys())}. "
        "Pass explicit layer indices instead."
    )


def get_default_layers_by_depth(
    num_layers: int,
    start_pct: float = 0.60,
    end_pct: float = 0.80,
) -> List[int]:
    """
    Compute a target layer window from model depth using a relative-depth range.

    The default range (60--80%) corresponds to the SR-guided window selected for
    ColPali and ColQwen2 in v2 of the paper. Jina v4 uses 70--90%. For unknown
    backbones, the SR-guided procedure (see ``experiments/per_layer_osr.py``)
    should be used to determine the appropriate range.

    Args:
        num_layers: Total number of transformer layers.
        start_pct: Start of the window as a fraction of depth (inclusive).
        end_pct: End of the window as a fraction of depth (inclusive).

    Returns:
        List of 0-indexed layer indices.
    """
    start = int(num_layers * start_pct)
    end = int(num_layers * end_pct)
    return list(range(start, end + 1))


def compute_sap_scores(
    attentions: Sequence[torch.Tensor],
    visual_indices: torch.Tensor,
    target_layers: Optional[List[int]] = None,
    model_name: Optional[str] = None,
    agg_mode: str = "mean",
) -> torch.Tensor:
    """
    Full SAP scoring pipeline: integrate centrality across target layers (Eq. 5-6).

    For each target layer, computes visual in-degree centrality and averages
    the scores across all target layers to produce a single importance score
    per visual token.

    Args:
        attentions: Sequence of attention tensors, one per layer. Each tensor
            has shape ``[batch, num_heads, seq_len, seq_len]``.
        visual_indices: 1-D tensor of visual token positions.
        target_layers: 0-indexed layer indices to use. If *None*, inferred
            from *model_name* via :func:`get_default_layers`.
        model_name: Used to look up default layers when *target_layers* is None.
        agg_mode: Head aggregation — ``"mean"`` or ``"max"``.

    Returns:
        Importance scores of shape ``[num_visual_tokens]`` (float32).

    Raises:
        ValueError: If neither *target_layers* nor *model_name* is provided.
    """
    if target_layers is None:
        if model_name is None:
            raise ValueError("Provide either target_layers or model_name.")
        target_layers = get_default_layers(model_name)

    accum: Optional[torch.Tensor] = None
    for l_idx in target_layers:
        layer_attn = attentions[l_idx]
        centrality = compute_visual_centrality(layer_attn, visual_indices, agg_mode=agg_mode)
        if accum is None:
            accum = centrality.float()
        else:
            accum = accum + centrality.float()

    return accum / len(target_layers)
