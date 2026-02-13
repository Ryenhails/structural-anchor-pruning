"""
Attention extractor for ColQwen2 (Qwen2-VL backbone).

Requires ``colpali-engine`` — install with ``pip install sap[colpali]``.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..utils import detect_visual_indices_by_range


class ColQwen2AttentionExtractor:
    """
    Extract attention weights and embeddings from a ColQwen2 model.

    Usage::

        extractor = ColQwen2AttentionExtractor(model, processor)
        attentions, visual_indices = extractor.extract(batch)
    """

    def __init__(self, model: nn.Module, processor=None):
        """
        Args:
            model: A ColQwen2 model instance.
            processor: The corresponding ColQwen2Processor (used to resolve
                vision start/end token IDs). If *None*, defaults for
                Qwen2-VL are used (151652 / 151653).
        """
        self.model = model
        if processor is not None and hasattr(processor, "tokenizer"):
            vocab = processor.tokenizer.get_vocab()
            self.vision_start_id = vocab.get("<|vision_start|>", 151652)
            self.vision_end_id = vocab.get("<|vision_end|>", 151653)
        else:
            self.vision_start_id = 151652
            self.vision_end_id = 151653

    @staticmethod
    def _unpad_pixel_values(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Unpad pixel_values to match ColQwen2's forward pass."""
        kwargs = {k: v for k, v in batch.items()}
        if "pixel_values" in kwargs and "image_grid_thw" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pix[:off] for pix, off in zip(kwargs["pixel_values"], offsets)], dim=0
            )
        return kwargs

    def extract(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[tuple, torch.Tensor]:
        """
        Run a forward pass and return attention weights + visual indices.

        Args:
            batch: Tokenized batch dict.

        Returns:
            ``(attentions, visual_indices)``
        """
        kwargs = self._unpad_pixel_values(batch)
        kwargs.update({
            "output_attentions": True,
            "output_hidden_states": True,
            "return_dict": True,
            "use_cache": False,
        })
        kwargs.pop("token_type_ids", None)

        # Import at call time to avoid hard dependency at module import
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

        with torch.inference_mode():
            outputs = Qwen2VLModel.forward(self.model, **kwargs)

        visual_indices = detect_visual_indices_by_range(
            batch["input_ids"][0], self.vision_start_id, self.vision_end_id
        )
        return outputs.attentions, visual_indices

    def extract_with_embeddings(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[tuple, torch.Tensor, torch.Tensor]:
        """
        Extract attentions, visual indices, **and** multi-vector embeddings.

        Returns:
            ``(attentions, visual_indices, embeddings)``
        """
        attentions, visual_indices = self.extract(batch)
        with torch.inference_mode():
            embeddings = self.model(**batch)
        return attentions, visual_indices, embeddings
