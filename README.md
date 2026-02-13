# Structural Anchor Pruning (SAP)

A training-free method for pruning visual token embeddings in VLM-based document retrieval systems.

**Paper:** [Look in the Middle: Structural Anchor Pruning for Scalable Visual RAG Indexing](https://arxiv.org/abs/2601.20107)

<p align="center">
  <img src="assets/sap_overview.png" alt="SAP Overview" width="100%">
</p>

## Overview

Multi-vector Visual RAG models (ColPali, ColQwen2, Jina V4) produce hundreds of visual token embeddings per document page, leading to large index sizes. SAP identifies structurally important tokens using **visual in-degree centrality** from the model's own attention maps — no training, no external data, no task-specific tuning.

Key insight: middle-layer attention (~40–60% depth) captures **layout structure** (headers, tables, regions), while final layers shift to sparse **alignment** patterns. By scoring tokens from these "structural" middle layers, SAP retains the embeddings that anchor document structure, achieving high score retention even at aggressive compression ratios.

## Installation

```bash
# Core (torch + numpy only)
pip install git+https://github.com/Ryenhails/structural-anchor-pruning.git

# With ColPali/ColQwen2 support
pip install "sap[colpali] @ git+https://github.com/Ryenhails/structural-anchor-pruning.git"

# With Jina V4 support
pip install "sap[jina] @ git+https://github.com/Ryenhails/structural-anchor-pruning.git"

# Development
git clone https://github.com/Ryenhails/structural-anchor-pruning.git
cd structural-anchor-pruning
pip install -e ".[dev]"
```

## Quick Start

### Pure tensor API (no model dependency)

```python
import torch
import sap

# Given: attention weights from a VLM forward pass
# attentions: tuple of [batch, heads, seq, seq] tensors (one per layer)
# visual_indices: 1-D tensor of visual token positions

# 1. Compute SAP importance scores
scores = sap.compute_sap_scores(
    attentions, visual_indices,
    target_layers=[8, 9, 10, 11],  # or model_name="colpali"
    agg_mode="mean",               # SAP-Mean (Eq. 3) or "max" (Eq. 4)
)

# 2. Prune to keep top 50% of visual tokens
pruned_embs = sap.prune_embeddings(visual_embeddings, scores, ratio=0.5)

# 3. Evaluate quality via Oracle Score Retention
osr = sap.compute_osr(query_embs, full_doc_embs, pruned_embs)
print(f"OSR: {osr:.4f}")  # 1.0 = perfect retention
```

### With ColPali model

```python
from colpali_engine.models import ColPali, ColPaliProcessor
from sap.attention.colpali import ColPaliAttentionExtractor
import sap

model = ColPali.from_pretrained("vidore/colpali-v1.2", device_map="cuda")
processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.2")
sap.ensure_eager_attention(model)  # Required for attention extraction

extractor = ColPaliAttentionExtractor(model, processor)

# Process an image
batch = processor.process_images([image]).to(model.device)
attentions, visual_indices = extractor.extract(batch)
embeddings = model(**batch)  # [1, seq_len, dim]

# Compute SAP scores and prune
scores = sap.compute_sap_scores(attentions, visual_indices, model_name="colpali")
visual_embs = embeddings[0, visual_indices]
pruned = sap.prune_embeddings(visual_embs, scores, ratio=0.5)
```

## API Reference

### Centrality (Eq. 2–4)

| Function | Description |
|----------|-------------|
| `compute_visual_centrality(layer_attn, visual_indices, agg_mode)` | In-degree centrality from a single layer |
| `compute_visual_centrality_batch(layer_attn, visual_indices, agg_mode)` | Batch variant |

### Scoring (Eq. 5–6)

| Function | Description |
|----------|-------------|
| `compute_sap_scores(attentions, visual_indices, target_layers, ...)` | Full SAP pipeline — average centrality across target layers |
| `get_default_layers(model_name)` | Default layer window for known models |
| `get_default_layers_by_depth(num_layers, start_pct, end_pct)` | Compute window from model depth |

### Pruning

| Function | Description |
|----------|-------------|
| `prune_embeddings(embeddings, scores, ratio)` | Keep top-*ratio* tokens by score |
| `prune_embeddings_with_indices(embeddings, scores, ratio)` | Also returns selected indices |
| `prune_embeddings_batch(embeddings_list, scores_list, ratio)` | Batch variant |

### Evaluation (Eq. 1, 8)

| Function | Description |
|----------|-------------|
| `maxsim_score(query_embs, doc_embs)` | MaxSim late-interaction score |
| `maxsim_score_batched(query_embs, doc_embs)` | Batched MaxSim |
| `maxsim_score_matrix(query_list, doc_list, device)` | Full Q×D score matrix |
| `compute_osr(query, full_doc, pruned_doc)` | Oracle Score Retention |
| `compute_osr_batch(...)` | Batch OSR |

### Utilities

| Function | Description |
|----------|-------------|
| `ensure_eager_attention(model)` | Switch SDPA → eager for attention extraction |
| `detect_visual_indices_by_token_id(input_ids, token_id)` | Find visual tokens (PaliGemma) |
| `detect_visual_indices_by_range(input_ids, start_id, end_id)` | Find visual tokens (Qwen2-VL) |

## Default Layer Windows (Table 4)

| Model | Backbone | Layers | Window |
|-------|----------|--------|--------|
| ColPali | PaliGemma-3B | 18 | `[8, 9, 10, 11]` |
| ColQwen2 | Qwen2-VL-2B | 28 | `[11, 12, 13, 14, 15, 16]` |
| Jina V4 | Qwen2.5-VL-3B | 36 | `[14, 15, 16, 17, 18, 19, 20, 21]` |

## Experiments

The `experiments/` directory contains the original benchmark scripts used in the paper. See [`experiments/README.md`](experiments/README.md) for usage instructions.

## Citation

```bibtex
@article{liu2025sap,
  title={Look in the Middle: Structural Anchor Pruning for Scalable Visual RAG Indexing},
  author={Liu, Zhuchenyang and Hu, Ziyu and Zhang, Yao and Xiao, Yu},
  journal={arXiv preprint arXiv:2601.20107},
  year={2025}
}
```

## License

MIT
