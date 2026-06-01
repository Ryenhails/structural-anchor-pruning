"""
Attention extractor for ColPali (PaliGemma backbone).

Requires ``colpali-engine`` — install with ``pip install sap[colpali]``.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..utils import detect_visual_indices_by_token_id


class ColPaliAttentionExtractor:
    """
    Extract attention weights and embeddings from a ColPali model.

    Usage::

        extractor = ColPaliAttentionExtractor(model, processor)
        attentions, visual_indices = extractor.extract(batch)
        attentions, visual_indices, embeddings = extractor.extract_with_embeddings(batch)
    """

    def __init__(self, model: nn.Module, processor=None):
        """
        Args:
            model: A ColPali model instance.
            processor: The corresponding ColPaliProcessor (used to resolve
                the image token ID). If *None*, the default PaliGemma
                image token ID (257152) is used.
        """
        self.model = model
        if processor is not None and hasattr(processor, "tokenizer"):
            self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
                processor.tokenizer.additional_special_tokens.index("<image>")
            ]
        else:
            # Read from model config (consistent with experiment code),
            # falling back to PaliGemma's default image_token_index.
            inner = model.model if hasattr(model, "model") else model
            cfg = inner.config if hasattr(inner, "config") else getattr(model, "config", None)
            self.image_token_id = getattr(cfg, "image_token_index", 256000) if cfg else 256000

    def extract(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[tuple, torch.Tensor]:
        """
        Run a forward pass and return attention weights + visual indices.

        Args:
            batch: Tokenized batch dict (as produced by ColPaliProcessor).

        Returns:
            ``(attentions, visual_indices)`` where *attentions* is a tuple of
            ``[batch, heads, seq, seq]`` tensors (one per layer) and
            *visual_indices* is a 1-D index tensor.
        """
        pali_model = self.model.model if hasattr(self.model, "model") else self.model
        with torch.inference_mode():
            outputs = pali_model(
                **batch,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
        visual_indices = detect_visual_indices_by_token_id(
            batch["input_ids"][0], self.image_token_id
        )
        return outputs.attentions, visual_indices

    def extract_with_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[tuple, torch.Tensor, torch.Tensor]:
        """
        Extract attentions, visual indices, **and** multi-vector embeddings.

        Returns:
            ``(attentions, visual_indices, embeddings)`` where *embeddings*
            is the full model output (all token embeddings including visual).
        """
        attentions, visual_indices = self.extract(batch)
        with torch.inference_mode():
            embeddings = self.model(**batch)  # ColPali forward -> multi-vector
        return attentions, visual_indices, embeddings
