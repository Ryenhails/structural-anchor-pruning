"""
Model-specific attention extraction backends.

Each extractor wraps the forward pass of a VLM backbone so that raw
attention weights and (optionally) visual token embeddings can be
obtained in a single call.

Available extractors (imported lazily to avoid hard dependencies):

* :class:`ColPaliAttentionExtractor` — PaliGemma / ColPali
* :class:`ColQwen2AttentionExtractor` — Qwen2-VL / ColQwen2
* :class:`JinaAttentionExtractor` — Jina Embeddings V4 (Qwen2.5-VL backbone)
"""
