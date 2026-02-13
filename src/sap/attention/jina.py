"""
Attention extractor for Jina Embeddings V4 (Qwen2.5-VL backbone).

Requires ``transformers`` — install with ``pip install sap[jina]``.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from ..utils import detect_visual_indices_by_range


class JinaAttentionExtractor:
    """
    Extract attention weights and embeddings from a Jina Embeddings V4 model.

    Jina V4 wraps a Qwen2.5-VL backbone, potentially behind a PeftModel LoRA adapter.
    This extractor handles the unwrapping and produces both attention weights and
    the projected multi-vector embeddings.

    Usage::

        extractor = JinaAttentionExtractor(model, processor)
        attentions, visual_indices = extractor.extract(batch)
        attentions, visual_indices, embeddings = extractor.extract_with_embeddings(batch)
    """

    def __init__(self, model: nn.Module, processor=None):
        """
        Args:
            model: A Jina Embeddings V4 model instance.
            processor: The corresponding processor/tokenizer.
        """
        self.model = model
        if processor is not None and hasattr(processor, "tokenizer"):
            vocab = processor.tokenizer.get_vocab()
            self.vision_start_id = vocab.get("<|vision_start|>", 151652)
            self.vision_end_id = vocab.get("<|vision_end|>", 151653)
        else:
            self.vision_start_id = 151652
            self.vision_end_id = 151653

    def _get_jina_model(self):
        """Unwrap PeftModel / model wrappers to get the JinaEmbeddingsV4Model."""
        if hasattr(self.model, "base_model"):
            return self.model.base_model.model
        if hasattr(self.model, "model"):
            return self.model.model
        return self.model

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
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )

        kwargs = {k: v for k, v in batch.items()}
        # Unpad pixel_values
        if "pixel_values" in kwargs and "image_grid_thw" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pix[:off] for pix, off in zip(kwargs["pixel_values"], offsets)], dim=0
            )

        jina_model = self._get_jina_model()
        qwen_model = jina_model.model

        position_ids, rope_deltas = qwen_model.get_rope_index(
            input_ids=kwargs["input_ids"],
            image_grid_thw=kwargs.get("image_grid_thw", None),
            attention_mask=kwargs["attention_mask"],
        )

        kwargs.update({
            "output_attentions": True,
            "output_hidden_states": True,
            "return_dict": True,
            "use_cache": False,
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
        })
        kwargs.pop("token_type_ids", None)

        with torch.inference_mode():
            outputs = Qwen2_5_VLForConditionalGeneration.forward(
                jina_model, task_label="retrieval", **kwargs
            )

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

        The embeddings are obtained by projecting the last hidden state through
        Jina's ``multi_vector_projector`` and L2-normalizing.

        Returns:
            ``(attentions, visual_indices, visual_embeddings)`` where
            *visual_embeddings* has shape ``[num_visual_tokens, proj_dim]``.
        """
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
            Qwen2_5_VLForConditionalGeneration,
        )

        kwargs = {k: v for k, v in batch.items()}
        if "pixel_values" in kwargs and "image_grid_thw" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pix[:off] for pix, off in zip(kwargs["pixel_values"], offsets)], dim=0
            )

        jina_model = self._get_jina_model()
        qwen_model = jina_model.model

        position_ids, rope_deltas = qwen_model.get_rope_index(
            input_ids=kwargs["input_ids"],
            image_grid_thw=kwargs.get("image_grid_thw", None),
            attention_mask=kwargs["attention_mask"],
        )

        kwargs.update({
            "output_attentions": True,
            "output_hidden_states": True,
            "return_dict": True,
            "use_cache": False,
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
        })
        kwargs.pop("token_type_ids", None)

        with torch.inference_mode():
            outputs = Qwen2_5_VLForConditionalGeneration.forward(
                jina_model, task_label="retrieval", **kwargs
            )

        visual_indices = detect_visual_indices_by_range(
            batch["input_ids"][0], self.vision_start_id, self.vision_end_id
        )

        # Project visual tokens through Jina's multi-vector head
        last_hidden = outputs.hidden_states[-1]
        visual_hidden = last_hidden[0, visual_indices]
        visual_emb = jina_model.multi_vector_projector(visual_hidden, task_label="retrieval")
        visual_emb = torch.nn.functional.normalize(visual_emb, p=2, dim=-1)

        return outputs.attentions, visual_indices, visual_emb
