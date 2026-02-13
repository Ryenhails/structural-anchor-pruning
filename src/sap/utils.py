"""
Utility helpers for SAP: attention mode switching and visual index detection.
"""

from typing import List

import torch
import torch.nn as nn


def ensure_eager_attention(model: nn.Module) -> None:
    """
    Switch a model from SDPA to eager attention so that ``output_attentions=True`` works.

    SDPA (Scaled Dot-Product Attention) does not support returning attention
    weights. This function recursively patches the model config so that the
    eager (loop-based) attention implementation is used instead.

    Args:
        model: A HuggingFace model (e.g. ColPali, ColQwen2, Jina).
    """
    def _patch_config(cfg):
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(cfg, attr) and getattr(cfg, attr) == "sdpa":
                setattr(cfg, attr, "eager")

    # Top-level config
    if hasattr(model, "config"):
        _patch_config(model.config)

    # Wrapped model (e.g. ColPali wraps PaliGemma)
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "config"):
            _patch_config(inner.config)
        # Language model inside PaliGemma
        if hasattr(inner, "language_model") and hasattr(inner.language_model, "config"):
            _patch_config(inner.language_model.config)

    # PeftModel wrapper (e.g. Jina)
    if hasattr(model, "base_model"):
        base = model.base_model
        if hasattr(base, "model") and hasattr(base.model, "config"):
            _patch_config(base.model.config)
        if hasattr(base, "model") and hasattr(base.model, "model"):
            inner2 = base.model.model
            if hasattr(inner2, "config"):
                _patch_config(inner2.config)


def detect_visual_indices_by_token_id(
    input_ids: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """
    Detect visual token positions using a dedicated image token ID.

    Works for PaliGemma/ColPali where visual tokens share a single token ID.

    Args:
        input_ids: Token IDs ``[seq_len]`` (1-D) or ``[1, seq_len]`` (2-D).
        image_token_id: The integer token ID assigned to image/visual tokens.

    Returns:
        1-D tensor of indices where ``input_ids == image_token_id``.
    """
    ids = input_ids.squeeze()
    return torch.where(ids == image_token_id)[0]


def detect_visual_indices_by_range(
    input_ids: torch.Tensor,
    start_token_id: int,
    end_token_id: int,
) -> torch.Tensor:
    """
    Detect visual token positions between start/end marker tokens.

    Works for Qwen2-VL / ColQwen2 and similar models that bracket visual
    tokens with ``<|vision_start|>`` / ``<|vision_end|>`` markers.

    Args:
        input_ids: Token IDs ``[seq_len]`` (1-D) or ``[1, seq_len]``.
        start_token_id: Token ID for ``<|vision_start|>``.
        end_token_id: Token ID for ``<|vision_end|>``.

    Returns:
        1-D tensor of indices of the visual tokens (excluding markers).
    """
    ids = input_ids.squeeze()
    start_positions = torch.where(ids == start_token_id)[0]
    end_positions = torch.where(ids == end_token_id)[0]

    all_visual: List[int] = []
    for s, e in zip(start_positions, end_positions):
        # Tokens between the markers (exclusive)
        all_visual.extend(range(s.item() + 1, e.item()))

    return torch.tensor(all_visual, dtype=torch.long, device=input_ids.device)
