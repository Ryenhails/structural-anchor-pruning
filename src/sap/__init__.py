"""
Structural Anchor Pruning (SAP) — training-free visual token pruning for VLM-based retrieval.

Paper: "Look in the Middle: Structural Anchor Pruning for Scalable Visual RAG Indexing"
       arXiv:2601.20107
"""

from .centrality import compute_visual_centrality, compute_visual_centrality_batch
from .scoring import (
    DEFAULT_LAYERS,
    compute_sap_scores,
    get_default_layers,
    get_default_layers_by_depth,
)
from .pruning import (
    prune_embeddings,
    prune_embeddings_batch,
    prune_embeddings_with_indices,
)
from .evaluation import (
    compute_osr,
    compute_osr_batch,
    maxsim_score,
    maxsim_score_batched,
    maxsim_score_matrix,
)
from .utils import (
    detect_visual_indices_by_range,
    detect_visual_indices_by_token_id,
    ensure_eager_attention,
)

__all__ = [
    # Centrality (Eq. 2-4)
    "compute_visual_centrality",
    "compute_visual_centrality_batch",
    # Scoring pipeline (Eq. 5-6)
    "compute_sap_scores",
    "get_default_layers",
    "get_default_layers_by_depth",
    "DEFAULT_LAYERS",
    # Pruning
    "prune_embeddings",
    "prune_embeddings_with_indices",
    "prune_embeddings_batch",
    # Evaluation (Eq. 1, 8)
    "maxsim_score",
    "maxsim_score_batched",
    "maxsim_score_matrix",
    "compute_osr",
    "compute_osr_batch",
    # Utilities
    "ensure_eager_attention",
    "detect_visual_indices_by_token_id",
    "detect_visual_indices_by_range",
]
