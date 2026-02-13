#!/usr/bin/env python3
"""
Fixed Ratio Benchmark v3: Comprehensive multi-baseline evaluation
Evaluates across Vidore v1 and v2 datasets:

For ColPali/ColQwen2/JinaEmbeddingsV4 (multi-vector):
  - Random pruning
  - SAP-Mean (middle 4 layers for PaliGemma, middle 6 layers for Qwen2, middle 8 layers for Jina)
  - SAP-Max (same layer selection as SAP-Mean)
  - Cluster-Merge (hierarchical clustering-based merging)
  - Adaptive-EOS (EOS attention-based pruning with calibration)
  - Upper bound (no pruning)

Additionally for JinaEmbeddingsV4:
  - Single-vector baseline (dense 2048-dim embeddings for comparison)

Reports: Score Retention, NDCG@5, and NDCG Retention
"""

import argparse
import json
import csv
import hashlib
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
from datasets import load_dataset
from sklearn.metrics import ndcg_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize as sklearn_normalize
from PIL import Image

from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel

try:
    from colpali_engine.models import ColQwen2, ColQwen2Processor
    COLQWEN2_AVAILABLE = True
except ImportError:
    COLQWEN2_AVAILABLE = False
    print("Warning: ColQwen2 not available")

try:
    from transformers import AutoModel, AutoConfig
    JINA_AVAILABLE = True
except ImportError:
    JINA_AVAILABLE = False
    print("Warning: Jina models not available")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
# from core.centrality import ensure_eager_attention  # Removed, we handle it at load time

# Import Jina's custom Qwen2_5_VL implementation
try:
    from experiments.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    QWEN2_5_VL_AVAILABLE = True
except ImportError:
    QWEN2_5_VL_AVAILABLE = False
    print("Warning: Qwen2_5_VL not available")


# ============================================================================
# Utilities
# ============================================================================

def get_image_hash(image: Image.Image) -> str:
    """Get hash of image for deduplication"""
    return hashlib.md5(image.tobytes()).hexdigest()


def load_vidore_v2_data(dataset_path: str):
    """
    Load Vidore v2 format data and build indices for efficient lookup.

    Vidore v2 datasets have separate configs for queries, corpus, and qrels.
    We load them separately and build dictionaries for fast lookup.

    Returns: {
        'corpus': {corpus-id: image},
        'queries': {query-id: text},
        'qrels': [{query-id, corpus-id, score}]
    }
    """
    # Load different configs separately
    print("Loading Queries config...")
    queries_ds = load_dataset(dataset_path, 'queries', split='test')

    print("Loading Corpus config...")
    corpus_ds = load_dataset(dataset_path, 'corpus', split='test')

    print("Loading Qrels config...")
    qrels_ds = load_dataset(dataset_path, 'qrels', split='test')

    # 1. Build Query index: query-id -> query text
    print("Indexing Queries...")
    queries = {
        row['query-id']: row['query']
        for row in tqdm(queries_ds, desc="Index Queries")
    }

    # 2. Build Corpus index: corpus-id -> image
    print("Indexing Corpus...")
    corpus = {
        row['corpus-id']: row['image']
        for row in tqdm(corpus_ds, desc="Index Corpus")
    }

    # 3. Extract Qrels (Ground Truth)
    # Only keep items with score > 0
    print("Extracting Qrels...")
    qrels = [
        row for row in tqdm(qrels_ds, desc="Extract Qrels") if row['score'] > 0
    ]

    return {
        'queries': queries,
        'corpus': corpus,
        'qrels': qrels
    }


# ============================================================================
# Model Loading
# ============================================================================

def detect_model_type(model_name: str) -> str:
    """Detect model type from model name"""
    name_lower = model_name.lower()
    if 'jina' in name_lower:
        return 'jina'
    elif 'qwen' in name_lower:
        return 'qwen2'
    else:
        return 'paligemma'

def ensure_eager_attention(model):
    """Helper to ensure standard models use eager attention if possible"""
    if hasattr(model, 'config') and hasattr(model.config, '_attn_implementation'):
        model.config._attn_implementation = 'eager'
    return model

def load_model_and_processor(model_name: str, device: str):
    """Load model and processor based on model type"""
    model_type = detect_model_type(model_name)
    print(f"Model type: {model_type}")

    if model_type == 'jina':
        if not JINA_AVAILABLE:
            raise ImportError("Jina models require transformers library")

        print("Loading Jina model with forced EAGER attention...")

        # 1. Load Config first and force eager attention
        # This prevents Flash Attention modules from ever being instantiated
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.attn_implementation = "eager"

        # Ensure nested configs also respect this (critical for VLM)
        if hasattr(config, 'vision_config'):
            config.vision_config.attn_implementation = "eager"
            config.vision_config._attn_implementation = "eager"
        if hasattr(config, 'text_config'):
            config.text_config.attn_implementation = "eager"
            config.text_config._attn_implementation = "eager"

        # 2. Load Model with the modified config
        model = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager"  # Double insurance
        ).eval()

        print("✅ Jina model loaded with Eager Attention (no patching required)")

        # Jina doesn't have a separate processor, the model handles it
        processor = None

    elif model_type == 'qwen2':
        if not COLQWEN2_AVAILABLE:
            raise ImportError("ColQwen2 not available")
        processor = ColQwen2Processor.from_pretrained(model_name)
        model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager"
        ).eval()
        ensure_eager_attention(model)

    else:  # paligemma
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        ).eval()
        ensure_eager_attention(model)

    return model, processor, model_type

def get_image_token_id(model, model_type: str) -> Optional[int]:
    """Get image token ID for multi-vector models"""
    if model_type == 'jina':
        return None  # Jina uses single-vector, no image tokens
    elif model_type == 'qwen2':
        return getattr(model.config, 'image_token_id', 151655)
    else:
        # For PaliGemma: safely access model.config, handling potential PeftModel wrapping
        if hasattr(model, 'model') and hasattr(model.model, 'config'):
            return getattr(model.model.config, 'image_token_index', 256000)
        else:
            return getattr(model.config, 'image_token_index', 256000)


# ============================================================================
# Encoding Functions (for Jina)
# ============================================================================

def encode_jina_queries_multivec(model, queries: List[str], device: str, batch_size: int = 8):
    """Encode queries using Jina model with multi-vector mode"""
    all_embeddings = []

    for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Jina queries (multi-vec)"):
        batch_texts = queries[i:i+batch_size]

        with torch.inference_mode():
            # Use retrieval task with query prefix and multi-vector mode
            embeddings = model.encode_text(
                texts=batch_texts,
                task='retrieval',
                prompt_name='query',
                max_length=8192,
                batch_size=len(batch_texts),
                return_multivector=True,
                return_numpy=False
            )

            # embeddings is List[torch.Tensor], each tensor is [num_tokens, 128]
            for emb in embeddings:
                all_embeddings.append(emb.cpu())

    return all_embeddings

def encode_jina_images_multivec(model, images: List[Image.Image], device: str, batch_size: int = 4):
    """Encode images using Jina model with multi-vector mode"""
    all_embeddings = []

    for i in tqdm(range(0, len(images), batch_size), desc="Encoding Jina images (multi-vec)"):
        batch_images = images[i:i+batch_size]

        with torch.inference_mode():
            # Use retrieval task for images with multi-vector mode
            embeddings = model.encode_image(
                images=batch_images,
                task='retrieval',
                batch_size=len(batch_images),
                return_multivector=True,
                return_numpy=False
            )

            # embeddings is List[torch.Tensor], each tensor is [num_tokens, 128]
            for emb in embeddings:
                all_embeddings.append(emb.cpu())

    return all_embeddings

def encode_jina_queries_singlevec(model, queries: List[str], device: str, batch_size: int = 8):
    """Encode queries using Jina model with single-vector mode (for comparison baseline)"""
    all_embeddings = []

    for i in tqdm(range(0, len(queries), batch_size), desc="Encoding Jina queries (single-vec)"):
        batch_texts = queries[i:i+batch_size]

        with torch.inference_mode():
            embeddings = model.encode_text(
                texts=batch_texts,
                task='retrieval',
                prompt_name='query',
                max_length=8192,
                batch_size=len(batch_texts),
                return_numpy=False
            )

            if isinstance(embeddings, torch.Tensor):
                all_embeddings.append(embeddings.cpu())
            else:
                for emb in embeddings:
                    all_embeddings.append(emb.cpu() if isinstance(emb, torch.Tensor) else torch.tensor(emb))

    if len(all_embeddings) > 0 and all_embeddings[0].dim() == 2:
        result = []
        for batch in all_embeddings:
            for i in range(batch.shape[0]):
                result.append(batch[i])
        return result
    else:
        return all_embeddings

def encode_jina_images_singlevec(model, images: List[Image.Image], device: str, batch_size: int = 4):
    """Encode images using Jina model with single-vector mode (for comparison baseline)"""
    all_embeddings = []

    for i in tqdm(range(0, len(images), batch_size), desc="Encoding Jina images (single-vec)"):
        batch_images = images[i:i+batch_size]

        with torch.inference_mode():
            embeddings = model.encode_image(
                images=batch_images,
                task='retrieval',
                batch_size=len(batch_images),
                return_numpy=False
            )

            if isinstance(embeddings, torch.Tensor):
                all_embeddings.append(embeddings.cpu())
            else:
                for emb in embeddings:
                    all_embeddings.append(emb.cpu() if isinstance(emb, torch.Tensor) else torch.tensor(emb))

    if len(all_embeddings) > 0 and all_embeddings[0].dim() == 2:
        result = []
        for batch in all_embeddings:
            for i in range(batch.shape[0]):
                result.append(batch[i])
        return result
    else:
        return all_embeddings


# ============================================================================
# Native Forward Calls (for ColPali/ColQwen2)
# ============================================================================

def get_embeddings(model, batch):
    """Native ColPali/ColQwen2 forward for embeddings"""
    return model(**batch)

def get_attentions(model, batch, model_type: str):
    """Native attention extraction matching forward pass"""
    with torch.inference_mode():
        if model_type == 'qwen2':
            # Match ColQwen2.forward pixel_values unpadding
            kwargs = {k: v for k, v in batch.items()}
            if "pixel_values" in kwargs and "image_grid_thw" in kwargs:
                offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
                kwargs["pixel_values"] = torch.cat(
                    [pix[:off] for pix, off in zip(kwargs["pixel_values"], offsets)], dim=0
                )
            kwargs.update({
                "output_attentions": True,
                "output_hidden_states": True,
                "return_dict": True,
                "use_cache": False
            })
            kwargs.pop("token_type_ids", None)
            outputs = Qwen2VLModel.forward(model, **kwargs)
        else:
            pali_model = model.model if hasattr(model, 'model') else model
            outputs = pali_model(**batch, output_attentions=True, output_hidden_states=True, return_dict=True)

        return outputs.attentions

def get_jina_outputs(model, batch):
    """
    Extract attentions AND hidden states for Jina model.
    Returns: (attentions, last_hidden_state)
    """
    with torch.inference_mode():
        # Unpad pixel_values first (same as Jina's get_last_hidden_states)
        kwargs = {k: v for k, v in batch.items()}
        if "pixel_values" in kwargs and "image_grid_thw" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pix[:off] for pix, off in zip(kwargs["pixel_values"], offsets)], dim=0
            )

        # Access the JinaEmbeddingsV4Model (bypass PeftModel wrapper if present)
        if hasattr(model, 'base_model'):
            jina_model = model.base_model.model
        elif hasattr(model, 'model'):
            jina_model = model.model
        else:
            jina_model = model

        # Get position_ids and rope_deltas from the underlying Qwen2_5_VLModel
        qwen_model = jina_model.model
        position_ids, rope_deltas = qwen_model.get_rope_index(
            input_ids=kwargs["input_ids"],
            image_grid_thw=kwargs.get("image_grid_thw", None),
            attention_mask=kwargs["attention_mask"],
        )

        # Prepare kwargs for forward call
        kwargs.update({
            "output_attentions": True,
            "output_hidden_states": True,  # Request hidden states
            "return_dict": True,
            "use_cache": False,
            "position_ids": position_ids,
            "rope_deltas": rope_deltas,
        })
        kwargs.pop("token_type_ids", None)

        # Forward pass
        outputs = Qwen2_5_VLForConditionalGeneration.forward(
            jina_model,
            task_label='retrieval',
            **kwargs
        )

        return outputs.attentions, outputs.hidden_states[-1]


# ============================================================================
# Centrality Computation (for SAP methods)
# ============================================================================

def compute_visual_centrality(layer_attn: torch.Tensor, visual_indices: torch.Tensor, agg_mode='mean'):
    """
    Compute centrality from visual token attention.
    layer_attn: [batch_size, num_heads, seq_len, seq_len]
    agg_mode: 'mean' to average over heads, 'max' to take max over heads
    Returns: [num_visual_tokens]
    """
    batch_size, num_heads = layer_attn.shape[:2]
    head_centralities = []

    for h in range(num_heads):
        attn_h = layer_attn[0, h]
        visual_attn = attn_h[visual_indices][:, visual_indices]
        centrality = visual_attn.sum(dim=-2)  # In-degree centrality
        head_centralities.append(centrality)

    stacked = torch.stack(head_centralities)
    return stacked.mean(dim=0) if agg_mode == 'mean' else stacked.max(dim=0).values


# ============================================================================
# Cluster-Based Merging
# ============================================================================

def cluster_merge(embeddings: torch.Tensor, merging_factor: float) -> torch.Tensor:
    """
    Merge patch embeddings using hierarchical clustering.

    Args:
        embeddings: [num_patches, dim] tensor of patch embeddings
        merging_factor: ratio by which to reduce the number of patches
                       num_clusters = num_patches / merging_factor

    Returns:
        merged_embeddings: [num_clusters, dim] tensor of cluster centroids
    """
    num_patches = embeddings.shape[0]
    dim = embeddings.shape[1]

    # Calculate target number of clusters
    num_clusters = max(1, int(num_patches / merging_factor))

    # If num_clusters >= num_patches, no merging needed
    if num_clusters >= num_patches:
        return embeddings

    # Step 1: Normalize embeddings
    # Convert to float32 to avoid BFloat16 issues with sklearn, then to numpy
    embeddings_fp32 = embeddings.float().cpu().numpy()
    embeddings_normalized = sklearn_normalize(embeddings_fp32, norm='l2', axis=1)

    # Step 2: Compute pairwise distance matrix
    # Distance = 1 - cosine_similarity
    # For normalized vectors: cosine_sim = dot product
    cosine_sim = embeddings_normalized @ embeddings_normalized.T
    distance_matrix = 1 - cosine_sim

    # Ensure distance matrix is symmetric, has zeros on diagonal, and is float64
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = distance_matrix.astype(np.float64)

    # Step 3: Hierarchical agglomerative clustering with average linkage
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric='precomputed',
        linkage='average'
    )

    cluster_labels = clustering.fit_predict(distance_matrix)

    # Step 4: Compute cluster centroids by averaging embeddings in each cluster
    merged_embeddings = torch.zeros((num_clusters, dim), dtype=embeddings.dtype)

    for cluster_id in range(num_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_members = embeddings[cluster_mask]

        if len(cluster_members) > 0:
            # Average embeddings in the cluster
            merged_embeddings[cluster_id] = cluster_members.mean(dim=0)

    return merged_embeddings


# ============================================================================
# Adaptive-EOS Attention Extraction
# ============================================================================

def extract_eos_attention(
    model,
    batch: Dict[str, torch.Tensor],
    model_type: str,
    visual_indices: torch.Tensor  # CHANGED: Accept indices directly
) -> Tuple[torch.Tensor, int]:
    """
    Extract EOS attention scores from the final layer.

    Args:
        model: The model
        batch: Processed image batch
        model_type: Type of model ('paligemma' or 'qwen2')
        visual_indices: Pre-computed indices of visual tokens

    Returns:
        scores: [N_visual] attention scores from EOS to visual patches
        num_visual: Number of visual tokens
    """
    # Get all layer attentions
    all_attentions = get_attentions(model, batch, model_type)

    # Get final layer attention
    final_layer_attn = all_attentions[-1]  # [batch, num_heads, seq_len, seq_len]

    # Get EOS position
    input_ids = batch['input_ids'][0]
    attention_mask = batch.get('attention_mask', None)

    if attention_mask is not None:
        valid_positions = torch.where(attention_mask[0] == 1)[0]
        eos_position = valid_positions[-1].item()
    else:
        eos_position = len(input_ids) - 1

    # Extract EOS-to-Visual attention, average over heads
    # Ensure visual_indices are used directly (pre-calculated in main loop)
    eos_to_visual = final_layer_attn[0, :, eos_position, visual_indices]  # [num_heads, N_visual]
    eos_scores = eos_to_visual.mean(dim=0)  # [N_visual]

    return eos_scores, len(visual_indices)

def extract_eos_attention_jina(
    model,
    batch: Dict[str, torch.Tensor],
    visual_indices: torch.Tensor
) -> torch.Tensor:
    """
    Extract EOS attention scores from the final layer for Jina model.

    Args:
        model: Jina model
        batch: Processed image batch
        visual_indices: Indices of visual tokens

    Returns:
        scores: [N_visual] attention scores from EOS to visual patches
    """
    # Get all layer attentions
    all_attentions = get_jina_attentions(model, batch)

    # Get final layer attention
    final_layer_attn = all_attentions[-1]  # [batch, num_heads, seq_len, seq_len]

    # Get EOS position
    attention_mask = batch.get('attention_mask', None)
    input_ids = batch['input_ids'][0]

    if attention_mask is not None:
        valid_positions = torch.where(attention_mask[0] == 1)[0]
        eos_position = valid_positions[-1].item()
    else:
        eos_position = len(input_ids) - 1

    # Extract EOS-to-Visual attention, average over heads
    eos_to_visual = final_layer_attn[0, :, eos_position, visual_indices]  # [num_heads, N_visual]
    eos_scores = eos_to_visual.mean(dim=0)  # [N_visual]

    return eos_scores


def calibrate_k_by_quantile(all_scores_list: List[torch.Tensor], target_ratio: float) -> float:
    """
    Calculate k based on Z-score distribution of the calibration set.

    Strategy:
    1. For each document, compute Z-scores: z = (score - μ) / σ
    2. Collect all Z-scores from calibration set
    3. Find the (1 - target_ratio) quantile
    4. This quantile value is our k

    Args:
        all_scores_list: List of score tensors, one per document
        target_ratio: Target retention ratio (e.g., 0.1 for 10%)

    Returns:
        k: The calibrated k value for τ = μ + k·σ
    """
    all_z_scores = []

    for scores in all_scores_list:
        # Per-document statistics
        mu = scores.mean()
        sigma = scores.std() + 1e-8  # Add epsilon to avoid division by zero

        # Compute Z-scores
        z_scores = (scores - mu) / sigma
        all_z_scores.append(z_scores)

    # Concatenate all Z-scores
    flat_z = torch.cat(all_z_scores)

    # If target retention is 10%, we need the 90th percentile threshold
    # (We keep tokens with z > threshold, so higher threshold = less tokens)
    k = torch.quantile(flat_z.float(), 1.0 - target_ratio)

    return k.item()


def apply_adaptive_pruning(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    k: float
) -> Tuple[torch.Tensor, int]:
    """
    Prune a single document using μ + k·σ threshold.

    Args:
        embeddings: [N, dim] visual token embeddings
        scores: [N] EOS attention scores
        k: Calibrated k value

    Returns:
        pruned_embeddings: [K, dim] where K <= N
        num_kept: Number of tokens kept
    """
    # Compute threshold
    mu = scores.mean()
    sigma = scores.std() + 1e-8
    threshold = mu + k * sigma

    # Create mask
    mask = scores > threshold

    # Safety check: Ensure at least 1 token is kept
    if mask.sum() == 0:
        # Keep the token with maximum score
        max_idx = scores.argmax()
        mask[max_idx] = True

    # Apply mask
    pruned_embeddings = embeddings[mask]
    num_kept = mask.sum().item()

    return pruned_embeddings, num_kept


# ============================================================================
# Scoring
# ============================================================================

def compute_score_matrix_maxsim(query_embs: List[torch.Tensor], doc_embs: List[torch.Tensor], device: str):
    """MaxSim scoring for multi-vector embeddings (ColPali/ColQwen2)"""
    num_queries = len(query_embs)
    num_docs = len(doc_embs)
    if num_queries == 0 or num_docs == 0:
        return np.zeros((num_queries, num_docs))

    max_q_len = max(q.shape[0] for q in query_embs)
    dim = query_embs[0].shape[-1]
    dtype = query_embs[0].dtype

    Q_batch = torch.zeros((num_queries, max_q_len, dim), dtype=dtype, device=device)
    Q_mask = torch.zeros((num_queries, max_q_len), dtype=torch.bool, device=device)

    for i, q in enumerate(query_embs):
        l = q.shape[0]
        Q_batch[i, :l] = q.to(device)
        Q_mask[i, :l] = True

    score_matrix = np.zeros((num_queries, num_docs), dtype=np.float32)

    with torch.inference_mode():
        for j, doc in enumerate(doc_embs):
            d = doc.to(device)
            if d.dtype != Q_batch.dtype:
                d = d.to(Q_batch.dtype)

            sim = torch.matmul(Q_batch, d.t())
            max_sim = sim.max(dim=-1).values * Q_mask
            score_matrix[:, j] = max_sim.sum(dim=-1).float().cpu().numpy()

    return score_matrix


def compute_score_matrix_singlevec(query_embs: List[torch.Tensor], doc_embs: List[torch.Tensor], device: str):
    """Single-vector cosine similarity scoring (for Jina)"""
    num_queries = len(query_embs)
    num_docs = len(doc_embs)
    if num_queries == 0 or num_docs == 0:
        return np.zeros((num_queries, num_docs))

    # Stack into matrices
    Q = torch.stack([q.to(device) for q in query_embs])  # [num_queries, dim]
    D = torch.stack([d.to(device) for d in doc_embs])    # [num_docs, dim]

    with torch.inference_mode():
        # Cosine similarity (vectors are already normalized by Jina)
        scores = torch.matmul(Q, D.t())  # [num_queries, num_docs]
        score_matrix = scores.cpu().numpy()

    return score_matrix


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_fixed_ratios(
    model, processor, dataset, dataset_name, target_layers: List[int],
    ratios, device, model_type, image_token_id, num_random_trials=5,
    calibration_size=128, version='v1'
):
    """
    Evaluate model on dataset with different pruning strategies.

    For Jina models: Only compute Upper bound (single-vector similarity)
    For ColPali/ColQwen2: Compute all pruning/merging methods
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {dataset_name} (Version: {version}, Model: {model_type})")
    print(f"{'='*80}")
    print(f"Step 1/5: Encoding and building unique corpus...")

    is_jina = (model_type == 'jina')

    # Track unique images and queries
    image_hash_to_idx = {}
    query_to_idx = {}

    unique_image_embs = []
    unique_query_embs = []
    # Initialize attention scores storage for all models (including Jina multi-vector mode)
    unique_sap_mean_scores = []
    unique_sap_max_scores = []
    unique_eos_scores = []

    # Map: sample_idx -> (unique_query_idx, unique_image_idx)
    sample_to_unique = []

    if version == 'v2':
        # ============================================================
        # V2: Encode ALL queries and ALL corpus, then build GT from qrels
        # ============================================================

        if is_jina:
            # Jina: Use multi-vector mode for pruning experiments
            print("Encoding all queries (Jina multi-vector)...")
            query_texts = list(dataset['queries'].values())
            query_ids = list(dataset['queries'].keys())

            # Use Jina's native multi-vector API
            unique_query_embs = model.encode_text(
                texts=query_texts,
                task='retrieval',
                prompt_name='query',
                return_multivector=True,
                return_numpy=False
            )
            query_id_to_idx = {q_id: idx for idx, q_id in enumerate(query_ids)}

            print("Encoding all corpus and extracting scores (Jina)...")
            corpus_images = [img.convert('RGB') for img in dataset['corpus'].values()]
            corpus_ids = list(dataset['corpus'].keys())

            # Process each image to extract both embeddings and attention scores
            corpus_id_to_idx = {}
            for i, (c_id, image) in enumerate(tqdm(zip(corpus_ids, corpus_images),
                                                    desc="Jina Encode & Attn",
                                                    total=len(corpus_ids))):

                # --- UNIFIED PROCESSING: Single forward pass for both embeddings and attentions ---
                # 1. Process image manually ONCE
                batch_images = model.processor.process_images([image])
                batch_images = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch_images.items()}

                # 2. Get both attentions AND hidden states from the same forward pass
                all_attentions, last_hidden_state = get_jina_outputs(model, batch_images)

                # Find visual token indices
                input_ids = batch_images['input_ids'][0]
                vision_start_id = model.config.vision_start_token_id
                vision_end_id = model.config.vision_end_token_id
                vision_start_pos = torch.where(input_ids == vision_start_id)[0]
                vision_end_pos = torch.where(input_ids == vision_end_id)[0]

                if len(vision_start_pos) > 0 and len(vision_end_pos) > 0:
                    visual_indices = torch.arange(
                        vision_start_pos[0] + 1, vision_end_pos[0],
                        device=input_ids.device
                    )
                else:
                    attention_mask = batch_images.get('attention_mask', None)
                    if attention_mask is not None:
                        valid_tokens = attention_mask[0].bool()
                        visual_indices = torch.where(valid_tokens)[0]
                    else:
                        visual_indices = torch.arange(len(input_ids), device=input_ids.device)

                # 3. Extract multi-vector embeddings from hidden state using model's projector
                with torch.inference_mode():
                    # Access the base JinaEmbeddingsV4Model (bypass PeftModel wrapper if present)
                    if hasattr(model, 'base_model'):
                        jina_model = model.base_model.model
                    elif hasattr(model, 'model'):
                        jina_model = model.model
                    else:
                        jina_model = model

                    # Extract visual tokens from hidden state
                    visual_hidden = last_hidden_state[0, visual_indices]  # [num_visual, hidden_dim]

                    # Project using the model's multi_vector_projector and normalize
                    # This matches what encode_image does internally (see get_multi_vector_embeddings)
                    visual_emb = jina_model.multi_vector_projector(visual_hidden, task_label='retrieval')
                    visual_emb = torch.nn.functional.normalize(visual_emb, p=2, dim=-1)

                    unique_image_embs.append(visual_emb.cpu())

                # 4. Compute SAP scores
                accum_sap_mean = None
                accum_sap_max = None
                for l_idx in target_layers:
                    layer_attn = all_attentions[l_idx]
                    s_mean = compute_visual_centrality(layer_attn, visual_indices, agg_mode='mean')
                    s_max = compute_visual_centrality(layer_attn, visual_indices, agg_mode='max')

                    if accum_sap_mean is None:
                        accum_sap_mean = s_mean.float()
                    else:
                        accum_sap_mean += s_mean.float()

                    if accum_sap_max is None:
                        accum_sap_max = s_max.float()
                    else:
                        accum_sap_max += s_max.float()

                num_layers = len(target_layers)
                unique_sap_mean_scores.append((accum_sap_mean / num_layers).cpu())
                unique_sap_max_scores.append((accum_sap_max / num_layers).cpu())

                # 5. Extract EOS attention (inline to avoid re-computation)
                final_layer_attn = all_attentions[-1]
                attention_mask = batch_images.get('attention_mask', None)
                if attention_mask is not None:
                    valid_positions = torch.where(attention_mask[0] == 1)[0]
                    eos_position = valid_positions[-1].item()
                else:
                    eos_position = len(input_ids) - 1

                eos_to_visual = final_layer_attn[0, :, eos_position, visual_indices]
                eos_scores = eos_to_visual.mean(dim=0)
                unique_eos_scores.append(eos_scores.cpu())

                corpus_id_to_idx[c_id] = i

        else:
            # ColPali/ColQwen2: Encode one by one with attention extraction
            print("Encoding all queries...")
            query_id_to_idx = {}
            for q_id, query_text in tqdm(dataset['queries'].items(), desc="Encode Queries"):
                batch_queries = processor.process_texts([query_text])
                batch_queries = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                               for k, v in batch_queries.items()}

                with torch.inference_mode():
                    query_emb = get_embeddings(model, batch_queries)

                if not isinstance(query_emb, torch.Tensor):
                    query_emb = query_emb[0]

                unique_query_embs.append(query_emb[0].cpu())
                query_id_to_idx[q_id] = len(unique_query_embs) - 1

            # 2. Encode all corpus
            print("Encoding all corpus...")
            corpus_id_to_idx = {}
            for c_id, image in tqdm(dataset['corpus'].items(), desc="Encode Corpus"):
                image = image.convert('RGB')

                batch_images = processor.process_images([image])
                batch_images = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in batch_images.items()}

                with torch.inference_mode():
                    image_emb_full = get_embeddings(model, batch_images)

                if not isinstance(image_emb_full, torch.Tensor):
                    image_emb_full = image_emb_full[0]

                # Extract visual tokens
                input_ids = batch_images['input_ids'][0]
                visual_mask = (input_ids == image_token_id).unsqueeze(-1)
                visual_indices = torch.where(visual_mask)[0]
                image_emb = image_emb_full[0, visual_indices]

                unique_image_embs.append(image_emb.cpu())

                # Compute centrality from middle 4 layers using both SAP-Mean and SAP-Max
                all_attentions = get_attentions(model, batch_images, model_type)

                accum_sap_mean = None
                accum_sap_max = None

                for l_idx in target_layers:
                    layer_attn = all_attentions[l_idx]

                    # SAP-Mean: mean over attention heads
                    s_mean = compute_visual_centrality(layer_attn, visual_indices, agg_mode='mean')
                    if accum_sap_mean is None:
                        accum_sap_mean = s_mean.float()
                    else:
                        accum_sap_mean += s_mean.float()

                    # SAP-Max: max over attention heads
                    s_max = compute_visual_centrality(layer_attn, visual_indices, agg_mode='max')
                    if accum_sap_max is None:
                        accum_sap_max = s_max.float()
                    else:
                        accum_sap_max += s_max.float()

                num_layers = len(target_layers)
                unique_sap_mean_scores.append((accum_sap_mean / num_layers).cpu())
                unique_sap_max_scores.append((accum_sap_max / num_layers).cpu())

                # Extract EOS attention scores
                # CHANGED: Pass visual_indices instead of image_token_id
                eos_scores, _ = extract_eos_attention(model, batch_images, model_type, visual_indices)
                unique_eos_scores.append(eos_scores.cpu())

                corpus_id_to_idx[c_id] = len(unique_image_embs) - 1

        # 3. Build ground truth from qrels
        print("Building ground truth from qrels...")
        for rel in tqdm(dataset['qrels'], desc="Process Qrels"):
            q_id = rel['query-id']
            c_id = rel['corpus-id']

            if q_id in query_id_to_idx and c_id in corpus_id_to_idx:
                q_idx = query_id_to_idx[q_id]
                c_idx = corpus_id_to_idx[c_id]
                sample_to_unique.append((q_idx, c_idx))

    else:
        # ============================================================
        # V1: Original logic - iterate over (query, image) pairs
        # ============================================================
        total_len = len(dataset)

        for idx in tqdm(range(total_len), desc="Encoding"):
            try:
                sample = dataset[idx]
                if 'image' not in sample or 'query' not in sample:
                    continue

                image = sample['image'].convert('RGB')
                query = sample['query']

                # Process image
                img_hash = get_image_hash(image)
                if img_hash in image_hash_to_idx:
                    unique_img_idx = image_hash_to_idx[img_hash]
                else:
                    if is_jina:
                        # --- UNIFIED PROCESSING: Single forward pass for both embeddings and attentions ---
                        # 1. Process image manually ONCE
                        batch_images = model.processor.process_images([image])
                        batch_images = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                      for k, v in batch_images.items()}

                        # 2. Get both attentions AND hidden states from the same forward pass
                        all_attentions, last_hidden_state = get_jina_outputs(model, batch_images)

                        # 3. Find visual token indices
                        input_ids = batch_images['input_ids'][0]
                        vision_start_id = model.config.vision_start_token_id
                        vision_end_id = model.config.vision_end_token_id
                        vision_start_pos = torch.where(input_ids == vision_start_id)[0]
                        vision_end_pos = torch.where(input_ids == vision_end_id)[0]

                        if len(vision_start_pos) > 0 and len(vision_end_pos) > 0:
                            visual_indices = torch.arange(
                                vision_start_pos[0] + 1, vision_end_pos[0],
                                device=input_ids.device
                            )
                        else:
                            attention_mask = batch_images.get('attention_mask', None)
                            if attention_mask is not None:
                                valid_tokens = attention_mask[0].bool()
                                visual_indices = torch.where(valid_tokens)[0]
                            else:
                                visual_indices = torch.arange(len(input_ids), device=input_ids.device)

                        # 4. Extract multi-vector embeddings from hidden state using model's projector
                        with torch.inference_mode():
                            # Access the base JinaEmbeddingsV4Model (bypass PeftModel wrapper if present)
                            if hasattr(model, 'base_model'):
                                jina_model = model.base_model.model
                            elif hasattr(model, 'model'):
                                jina_model = model.model
                            else:
                                jina_model = model

                            # Extract visual tokens from hidden state
                            visual_hidden = last_hidden_state[0, visual_indices]  # [num_visual, hidden_dim]

                            # Project using the model's multi_vector_projector and normalize
                            # This matches what encode_image does internally (see get_multi_vector_embeddings)
                            visual_emb = jina_model.multi_vector_projector(visual_hidden, task_label='retrieval')
                            visual_emb = torch.nn.functional.normalize(visual_emb, p=2, dim=-1)

                            unique_image_embs.append(visual_emb.cpu())

                        # 5. Compute SAP scores
                        accum_sap_mean = None
                        accum_sap_max = None
                        for l_idx in target_layers:
                            layer_attn = all_attentions[l_idx]
                            s_mean = compute_visual_centrality(layer_attn, visual_indices, agg_mode='mean')
                            s_max = compute_visual_centrality(layer_attn, visual_indices, agg_mode='max')
                            if accum_sap_mean is None:
                                accum_sap_mean = s_mean.float()
                            else:
                                accum_sap_mean += s_mean.float()
                            if accum_sap_max is None:
                                accum_sap_max = s_max.float()
                            else:
                                accum_sap_max += s_max.float()
                        num_layers = len(target_layers)
                        unique_sap_mean_scores.append((accum_sap_mean / num_layers).cpu())
                        unique_sap_max_scores.append((accum_sap_max / num_layers).cpu())

                        # 6. Extract EOS scores
                        final_layer_attn = all_attentions[-1]
                        attention_mask = batch_images.get('attention_mask', None)
                        if attention_mask is not None:
                            valid_positions = torch.where(attention_mask[0] == 1)[0]
                            eos_position = valid_positions[-1].item()
                        else:
                            eos_position = len(input_ids) - 1
                        eos_to_visual = final_layer_attn[0, :, eos_position, visual_indices]
                        unique_eos_scores.append(eos_to_visual.mean(dim=0).cpu())
                    else:
                        # ColPali/ColQwen2 image encoding
                        batch_images = processor.process_images([image])
                        batch_images = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                      for k, v in batch_images.items()}

                        with torch.inference_mode():
                            image_emb_full = get_embeddings(model, batch_images)

                        if not isinstance(image_emb_full, torch.Tensor):
                            image_emb_full = image_emb_full[0]

                        # Extract visual tokens
                        input_ids = batch_images['input_ids'][0]
                        visual_mask = (input_ids == image_token_id).unsqueeze(-1)
                        visual_indices = torch.where(visual_mask)[0]
                        image_emb = image_emb_full[0, visual_indices]

                        unique_image_embs.append(image_emb.cpu())

                        # Compute centrality from middle 4 layers
                        all_attentions = get_attentions(model, batch_images, model_type)

                        accum_sap_mean = None
                        accum_sap_max = None

                        for l_idx in target_layers:
                            layer_attn = all_attentions[l_idx]

                            s_mean = compute_visual_centrality(layer_attn, visual_indices, agg_mode='mean')
                            if accum_sap_mean is None:
                                accum_sap_mean = s_mean.float()
                            else:
                                accum_sap_mean += s_mean.float()

                            s_max = compute_visual_centrality(layer_attn, visual_indices, agg_mode='max')
                            if accum_sap_max is None:
                                accum_sap_max = s_max.float()
                            else:
                                accum_sap_max += s_max.float()

                        num_layers = len(target_layers)
                        unique_sap_mean_scores.append((accum_sap_mean / num_layers).cpu())
                        unique_sap_max_scores.append((accum_sap_max / num_layers).cpu())

                        # Extract EOS attention scores
                        # CHANGED: Pass visual_indices instead of image_token_id
                        eos_scores, _ = extract_eos_attention(model, batch_images, model_type, visual_indices)
                        unique_eos_scores.append(eos_scores.cpu())

                    unique_img_idx = len(unique_image_embs) - 1
                    image_hash_to_idx[img_hash] = unique_img_idx

                # Process query
                if query in query_to_idx:
                    unique_q_idx = query_to_idx[query]
                else:
                    if is_jina:
                        # Jina query encoding (multi-vector mode)
                        with torch.inference_mode():
                            q_emb = model.encode_text(
                                texts=[query],
                                task='retrieval',
                                prompt_name='query',
                                max_length=8192,
                                batch_size=1,
                                return_multivector=True,
                                return_numpy=False
                            )
                        # q_emb is a list with one element [num_tokens, 128]
                        unique_query_embs.append(q_emb[0].cpu())
                    else:
                        # ColPali/ColQwen2 query encoding
                        batch_queries = processor.process_texts([query])
                        batch_queries = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                       for k, v in batch_queries.items()}

                        with torch.inference_mode():
                            query_emb = get_embeddings(model, batch_queries)

                        if not isinstance(query_emb, torch.Tensor):
                            query_emb = query_emb[0]

                        unique_query_embs.append(query_emb[0].cpu())

                    unique_q_idx = len(unique_query_embs) - 1
                    query_to_idx[query] = unique_q_idx

                sample_to_unique.append((unique_q_idx, unique_img_idx))

            except Exception as e:
                print(f"Sample {idx} error: {e}")
                continue

    num_unique_queries = len(unique_query_embs)
    num_unique_images = len(unique_image_embs)
    num_samples = len(sample_to_unique)

    print(f"✅ Dataset: {num_samples} samples")
    print(f"   Unique queries: {num_unique_queries}")
    print(f"   Unique images: {num_unique_images}")

    if num_unique_queries == 0 or num_unique_images == 0:
        return {}

    # Build ground truth matrix
    print("Step 2/5: Building ground truth matrix...")
    y_true = np.zeros((num_unique_queries, num_unique_images), dtype=np.float32)
    for q_idx, img_idx in sample_to_unique:
        y_true[q_idx, img_idx] = 1.0

    # Choose scoring function
    # All models (including Jina in multi-vector mode) use MaxSim scoring
    score_fn = compute_score_matrix_maxsim

    # Compute baseline (Full/Upper bound) scores
    print("Computing baseline scores (Upper bound - no pruning)...")
    score_matrix_full = score_fn(unique_query_embs, unique_image_embs, device)

    # Compute full retention (for matched pairs)
    full_score_pairs = []
    for q_idx, img_idx in sample_to_unique:
        full_score_pairs.append(score_matrix_full[q_idx, img_idx])
    mean_full_score = np.mean(full_score_pairs)

    ndcg_full = ndcg_score(y_true, score_matrix_full, k=5)

    results = {
        'dataset_info': {
            'num_samples': num_samples,
            'num_unique_queries': num_unique_queries,
            'num_unique_images': num_unique_images,
            'model_type': model_type
        },
        'Upper_bound': {
            'score_retention': 1.0,
            'ndcg5': float(ndcg_full),
            'ndcg_retention': 1.0
        }
    }

    # For multi-vector models (including Jina): continue with pruning experiments
    results['SAP-Mean'] = {}
    results['SAP-Max'] = {}
    results['Cluster-Merge'] = {}
    results['Adaptive-EOS'] = {}
    results['Random'] = {}

    # ============================================================
    # Step 3: Calibration for Adaptive-EOS
    # ============================================================
    print(f"\nStep 3/5: Calibration Phase for Adaptive-EOS")
    print(f"Using first {min(calibration_size, num_unique_images)} images for calibration...")

    # Use a subset for calibration
    calibration_scores = unique_eos_scores[:calibration_size]

    # Calibrate k for each target ratio
    k_map = {}
    for ratio in ratios:
        k_val = calibrate_k_by_quantile(calibration_scores, ratio)
        k_map[ratio] = k_val
        print(f"  Target ratio {ratio:.1%} -> k = {k_val:.4f}")

    results['calibrated_k_values'] = k_map

    # Pre-compute sort indices for SAP-Mean and SAP-Max
    print("\nStep 4/5: Pre-computing sort indices...")
    sorted_indices_sap_mean = [torch.argsort(scores, descending=True)
                               for scores in tqdm(unique_sap_mean_scores, desc="SAP-Mean sorting")]
    sorted_indices_sap_max = [torch.argsort(scores, descending=True)
                              for scores in tqdm(unique_sap_max_scores, desc="SAP-Max sorting")]

    # Pre-compute random permutations
    random_permutations = [[torch.randperm(emb.shape[0]) for _ in range(num_random_trials)]
                          for emb in unique_image_embs]

    # Ratio sweep
    print("\nStep 5/5: Evaluating fixed ratios across all methods...")

    for ratio in tqdm(ratios, desc=f"Processing ratios"):
        # ============= SAP-Mean (middle 4 layers, mean over heads) =============
        pruned_docs_sap_mean = []
        for i in range(num_unique_images):
            img = unique_image_embs[i]
            N = img.shape[0]
            k = max(1, min(N, int(N * ratio)))
            idx = sorted_indices_sap_mean[i][:k]
            pruned_docs_sap_mean.append(img[idx])

        score_mat_sap_mean = compute_score_matrix_maxsim(unique_query_embs, pruned_docs_sap_mean, device)
        ndcg_sap_mean = ndcg_score(y_true, score_mat_sap_mean, k=5)

        # Compute score retention for SAP-Mean
        sap_mean_score_pairs = []
        for q_idx, img_idx in sample_to_unique:
            sap_mean_score_pairs.append(score_mat_sap_mean[q_idx, img_idx])

        score_retention_sap_mean = np.mean(sap_mean_score_pairs) / mean_full_score if mean_full_score != 0 else 0.0
        ndcg_retention_sap_mean = ndcg_sap_mean / ndcg_full if ndcg_full != 0 else 0.0

        results['SAP-Mean'][ratio] = {
            'score_retention': float(score_retention_sap_mean),
            'ndcg5': float(ndcg_sap_mean),
            'ndcg_retention': float(ndcg_retention_sap_mean)
        }

        # ============= SAP-Max (middle 4 layers, max over heads) =============
        pruned_docs_sap_max = []
        for i in range(num_unique_images):
            img = unique_image_embs[i]
            N = img.shape[0]
            k = max(1, min(N, int(N * ratio)))
            idx = sorted_indices_sap_max[i][:k]
            pruned_docs_sap_max.append(img[idx])

        score_mat_sap_max = compute_score_matrix_maxsim(unique_query_embs, pruned_docs_sap_max, device)
        ndcg_sap_max = ndcg_score(y_true, score_mat_sap_max, k=5)

        # Compute score retention for SAP-Max
        sap_max_score_pairs = []
        for q_idx, img_idx in sample_to_unique:
            sap_max_score_pairs.append(score_mat_sap_max[q_idx, img_idx])

        score_retention_sap_max = np.mean(sap_max_score_pairs) / mean_full_score if mean_full_score != 0 else 0.0
        ndcg_retention_sap_max = ndcg_sap_max / ndcg_full if ndcg_full != 0 else 0.0

        results['SAP-Max'][ratio] = {
            'score_retention': float(score_retention_sap_max),
            'ndcg5': float(ndcg_sap_max),
            'ndcg_retention': float(ndcg_retention_sap_max)
        }

        # ============= Cluster-Merge =============
        merging_factor = 1.0 / ratio
        merged_docs_cluster = []
        for i in range(num_unique_images):
            img = unique_image_embs[i]
            merged_img = cluster_merge(img, merging_factor)
            merged_docs_cluster.append(merged_img)

        score_mat_cluster = compute_score_matrix_maxsim(unique_query_embs, merged_docs_cluster, device)
        ndcg_cluster = ndcg_score(y_true, score_mat_cluster, k=5)

        # Compute score retention for Cluster-Merge
        cluster_score_pairs = []
        for q_idx, img_idx in sample_to_unique:
            cluster_score_pairs.append(score_mat_cluster[q_idx, img_idx])

        score_retention_cluster = np.mean(cluster_score_pairs) / mean_full_score if mean_full_score != 0 else 0.0
        ndcg_retention_cluster = ndcg_cluster / ndcg_full if ndcg_full != 0 else 0.0

        results['Cluster-Merge'][ratio] = {
            'score_retention': float(score_retention_cluster),
            'ndcg5': float(ndcg_cluster),
            'ndcg_retention': float(ndcg_retention_cluster)
        }

        # ============= Adaptive-EOS =============
        k_val = k_map[ratio]

        # Apply pruning to all documents
        pruned_docs_adaptive = []
        total_tokens = 0
        kept_tokens = 0

        for i in range(num_unique_images):
            img_emb = unique_image_embs[i]
            eos_scores = unique_eos_scores[i]

            # Prune
            pruned_emb, num_kept = apply_adaptive_pruning(img_emb, eos_scores, k_val)
            pruned_docs_adaptive.append(pruned_emb)

            # Track statistics
            total_tokens += img_emb.shape[0]
            kept_tokens += num_kept

        # Compute actual retention ratio
        actual_ratio = kept_tokens / total_tokens if total_tokens > 0 else 0.0

        # Compute scores
        score_mat_adaptive = compute_score_matrix_maxsim(unique_query_embs, pruned_docs_adaptive, device)
        ndcg_adaptive = ndcg_score(y_true, score_mat_adaptive, k=5)

        # Compute score retention
        adaptive_score_pairs = []
        for q_idx, img_idx in sample_to_unique:
            adaptive_score_pairs.append(score_mat_adaptive[q_idx, img_idx])

        score_retention_adaptive = np.mean(adaptive_score_pairs) / mean_full_score if mean_full_score != 0 else 0.0
        ndcg_retention_adaptive = ndcg_adaptive / ndcg_full if ndcg_full != 0 else 0.0

        results['Adaptive-EOS'][ratio] = {
            'target_ratio': float(ratio),
            'calibrated_k': float(k_val),
            'actual_ratio': float(actual_ratio),
            'score_retention': float(score_retention_adaptive),
            'ndcg5': float(ndcg_adaptive),
            'ndcg_retention': float(ndcg_retention_adaptive),
            'total_tokens': int(total_tokens),
            'kept_tokens': int(kept_tokens)
        }

        # ============= Random (with multiple trials) =============
        trial_score_retentions = []
        trial_ndcgs = []
        trial_ndcg_retentions = []

        for t in range(num_random_trials):
            pruned_docs_random = []
            for i in range(num_unique_images):
                img = unique_image_embs[i]
                N = img.shape[0]
                k = max(1, min(N, int(N * ratio)))
                idx = random_permutations[i][t][:k]
                pruned_docs_random.append(img[idx])

            score_mat_random = compute_score_matrix_maxsim(unique_query_embs, pruned_docs_random, device)
            ndcg_random = ndcg_score(y_true, score_mat_random, k=5)
            trial_ndcgs.append(ndcg_random)
            trial_ndcg_retentions.append(ndcg_random / ndcg_full if ndcg_full != 0 else 0.0)

            # Compute score retention for this trial
            random_score_pairs = []
            for q_idx, img_idx in sample_to_unique:
                random_score_pairs.append(score_mat_random[q_idx, img_idx])

            trial_score_retentions.append(np.mean(random_score_pairs) / mean_full_score if mean_full_score != 0 else 0.0)

        results['Random'][ratio] = {
            'score_retention': float(np.mean(trial_score_retentions)),
            'score_retention_std': float(np.std(trial_score_retentions)),
            'ndcg5': float(np.mean(trial_ndcgs)),
            'ndcg5_std': float(np.std(trial_ndcgs)),
            'ndcg_retention': float(np.mean(trial_ndcg_retentions)),
            'ndcg_retention_std': float(np.std(trial_ndcg_retentions))
        }

    return results


# ============================================================================
# Dataset Configuration
# ============================================================================

def get_all_datasets():
    """Return all Vidore v1 and v2 datasets"""
    vidore_v1 = [
        ("arxivqa", "vidore/arxivqa_test_subsampled"),
        ("docvqa", "vidore/docvqa_test_subsampled"),
        ("infovqa", "vidore/infovqa_test_subsampled"),
        ("tabfquad", "vidore/tabfquad_test_subsampled"),
        ("tatdqa", "vidore/tatdqa_test"),
        ("shiftproject", "vidore/shiftproject_test"),
        ("syntheticDocQA_ai", "vidore/syntheticDocQA_artificial_intelligence_test"),
        ("syntheticDocQA_energy", "vidore/syntheticDocQA_energy_test"),
        ("syntheticDocQA_govt", "vidore/syntheticDocQA_government_reports_test"),
        ("syntheticDocQA_health", "vidore/syntheticDocQA_healthcare_industry_test"),
    ]

    vidore_v2 = [
        ("esg_reports_v2", "vidore/esg_reports_v2"),
        ("biomedical_lectures_v2", "vidore/biomedical_lectures_v2"),
        ("economics_reports_v2", "vidore/economics_reports_v2"),
        ("esg_reports_human_v2", "vidore/esg_reports_human_labeled_v2"),
    ]

    return vidore_v1, vidore_v2


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fixed Ratio Benchmark v3: Comprehensive multi-baseline evaluation')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name or path (e.g., vidore/colpali-v1.2, jinaai/jina-embeddings-v4)')
    parser.add_argument('--layers', type=str, default="8,9,10,11",
                       help='Comma-separated layer indices for SAP (ignored for Jina, default: "8,9,10,11")')
    parser.add_argument('--calibration_size', type=int, default=128,
                       help='Number of samples for Adaptive-EOS calibration (default: 128)')
    parser.add_argument('--output_dir', type=str, default='./experiments/outputs',
                       help='Output directory for results')
    parser.add_argument('--skip_v1', action='store_true',
                       help='Skip Vidore v1 datasets')
    parser.add_argument('--skip_v2', action='store_true',
                       help='Skip Vidore v2 datasets')
    args = parser.parse_args()

    try:
        target_layers = [int(x.strip()) for x in args.layers.split(',')]
    except ValueError:
        print("Error: --layers must be comma-separated integers")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = args.model_name.split('/')[-1]
    output_dir = Path(args.output_dir) / f"fixed_ratio_benchmark_v3_{model_short_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect model type
    model_type_detected = detect_model_type(args.model_name)

    print(f"\n{'='*80}")
    print("FIXED RATIO BENCHMARK V3 - COMPREHENSIVE MULTI-BASELINE EVALUATION")
    print(f"{'='*80}")
    print(f"Model:  {args.model_name}")
    print(f"Type:   {model_type_detected}")

    if model_type_detected != 'jina':
        print(f"Layers: {target_layers}")
        print(f"Fixed Ratios: [0.2, 0.1, 0.05]")
        print(f"Calibration Size: {args.calibration_size}")
        print(f"Methods:")
        print(f"  - Random")
        print(f"  - SAP-Mean (middle 4 layers, mean over attention heads)")
        print(f"  - SAP-Max (middle 4 layers, max over attention heads)")
        print(f"  - Cluster-Merge (hierarchical clustering)")
        print(f"  - Adaptive-EOS (EOS attention with calibration)")
        print(f"  - Upper bound (no pruning)")
    else:
        print(f"Methods: Upper bound only (single-vector similarity)")

    print(f"Metrics: Score Retention, NDCG@5, NDCG Retention")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    device = get_torch_device("auto")
    model, processor, model_type = load_model_and_processor(args.model_name, device)
    image_token_id = get_image_token_id(model, model_type)
    print(f"✅ Model loaded on {device}, image_token_id={image_token_id}\n")

    vidore_v1, vidore_v2 = get_all_datasets()

    # Combine datasets with version information
    datasets_to_run = []
    if not args.skip_v1:
        datasets_to_run.extend([(name, path, 'v1') for name, path in vidore_v1])
        print(f"📊 Vidore v1: {len(vidore_v1)} datasets")
    if not args.skip_v2:
        datasets_to_run.extend([(name, path, 'v2') for name, path in vidore_v2])
        print(f"📊 Vidore v2: {len(vidore_v2)} datasets")

    print(f"📊 Total datasets to evaluate: {len(datasets_to_run)}\n")

    # Fixed ratios (only relevant for multi-vector models)
    ratios = [0.2, 0.1, 0.05]
    print(f"Fixed Ratios: {ratios}\n")

    all_results = {}

    for dataset_name, dataset_path, version in datasets_to_run:
        print(f"\n{'#'*80}")
        print(f"# {dataset_name.upper()} [{version}]")
        print(f"# Path: {dataset_path}")
        print(f"{'#'*80}")

        try:
            # Load dataset based on version
            if version == 'v2':
                dataset_obj = load_vidore_v2_data(dataset_path)
                print(f"✅ Loaded Vidore V2 Data. Qrels pairs: {len(dataset_obj['qrels'])}")
            else:
                dataset_obj = load_dataset(dataset_path, split="test")
                print(f"✅ Loaded {len(dataset_obj)} samples")

            results = evaluate_fixed_ratios(
                model, processor, dataset_obj, dataset_name,
                target_layers=target_layers,
                ratios=ratios,
                device=device,
                model_type=model_type,
                image_token_id=image_token_id,
                num_random_trials=5,
                calibration_size=args.calibration_size,
                version=version
            )

            # Save per-dataset results
            per_dataset_file = output_dir / f"results_{dataset_name}.json"
            with open(per_dataset_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"✅ Saved: {per_dataset_file}")

            all_results[dataset_name] = results

        except Exception as e:
            print(f"❌ Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save comprehensive summary
    print(f"\n{'='*80}")
    print("CREATING COMPREHENSIVE SUMMARY")
    print(f"{'='*80}\n")

    # JSON summary
    summary_json = output_dir / "benchmark_summary.json"
    with open(summary_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ JSON Summary: {summary_json}")

    # CSV summary
    csv_file = output_dir / "benchmark_summary.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dataset', 'Method', 'Ratio', 'Actual_Ratio', 'Score_Retention', 'Score_Retention_Std',
                        'NDCG5', 'NDCG5_Std', 'NDCG_Retention', 'NDCG_Retention_Std'])

        for dataset_name, results in all_results.items():
            # Upper bound (no pruning)
            ub = results.get('Upper_bound', {})
            writer.writerow([
                dataset_name, 'Upper_bound', 1.0, 1.0,
                ub.get('score_retention', 1.0), 0.0,
                ub.get('ndcg5', 0.0), 0.0,
                ub.get('ndcg_retention', 1.0), 0.0
            ])

            # Only add pruning results if available (not Jina)
            if 'SAP-Mean' in results:
                for ratio in ratios:
                    # SAP-Mean
                    if ratio in results.get('SAP-Mean', {}):
                        sap_mean = results['SAP-Mean'][ratio]
                        writer.writerow([
                            dataset_name, 'SAP-Mean', ratio, ratio,
                            sap_mean.get('score_retention', 0.0), 0.0,
                            sap_mean.get('ndcg5', 0.0), 0.0,
                            sap_mean.get('ndcg_retention', 0.0), 0.0
                        ])

                    # SAP-Max
                    if ratio in results.get('SAP-Max', {}):
                        sap_max = results['SAP-Max'][ratio]
                        writer.writerow([
                            dataset_name, 'SAP-Max', ratio, ratio,
                            sap_max.get('score_retention', 0.0), 0.0,
                            sap_max.get('ndcg5', 0.0), 0.0,
                            sap_max.get('ndcg_retention', 0.0), 0.0
                        ])

                    # Cluster-Merge
                    if ratio in results.get('Cluster-Merge', {}):
                        cluster = results['Cluster-Merge'][ratio]
                        writer.writerow([
                            dataset_name, 'Cluster-Merge', ratio, ratio,
                            cluster.get('score_retention', 0.0), 0.0,
                            cluster.get('ndcg5', 0.0), 0.0,
                            cluster.get('ndcg_retention', 0.0), 0.0
                        ])

                    # Adaptive-EOS
                    if ratio in results.get('Adaptive-EOS', {}):
                        aeos = results['Adaptive-EOS'][ratio]
                        writer.writerow([
                            dataset_name, 'Adaptive-EOS', ratio, aeos.get('actual_ratio', ratio),
                            aeos.get('score_retention', 0.0), 0.0,
                            aeos.get('ndcg5', 0.0), 0.0,
                            aeos.get('ndcg_retention', 0.0), 0.0
                        ])

                    # Random
                    if ratio in results.get('Random', {}):
                        rand = results['Random'][ratio]
                        writer.writerow([
                            dataset_name, 'Random', ratio, ratio,
                            rand.get('score_retention', 0.0), rand.get('score_retention_std', 0.0),
                            rand.get('ndcg5', 0.0), rand.get('ndcg5_std', 0.0),
                            rand.get('ndcg_retention', 0.0), rand.get('ndcg_retention_std', 0.0)
                        ])

    print(f"✅ CSV Summary: {csv_file}")

    # Create a human-readable summary table
    summary_txt = output_dir / "benchmark_summary.txt"
    with open(summary_txt, 'w') as f:
        f.write("="*100 + "\n")
        f.write("FIXED RATIO BENCHMARK V3 SUMMARY (COMPREHENSIVE MULTI-BASELINE)\n")
        f.write("="*100 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Type:  {model_type_detected}\n")

        if model_type_detected != 'jina':
            f.write(f"Layers: {target_layers}\n")
            f.write(f"Ratios: {ratios}\n")
            f.write(f"Calibration Size: {args.calibration_size}\n")
            f.write(f"Methods:\n")
            f.write(f"  - Random\n")
            f.write(f"  - SAP-Mean (middle 4 layers, mean over attention heads)\n")
            f.write(f"  - SAP-Max (middle 4 layers, max over attention heads)\n")
            f.write(f"  - Cluster-Merge (hierarchical clustering)\n")
            f.write(f"  - Adaptive-EOS (EOS attention with calibration)\n")
            f.write(f"  - Upper bound (no pruning)\n")
        else:
            f.write(f"Methods: Upper bound only (single-vector similarity)\n")

        f.write(f"Metrics: Score Retention, NDCG@5, NDCG Retention\n\n")

        for dataset_name, results in all_results.items():
            f.write(f"\n{'-'*100}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"{'-'*100}\n")

            info = results.get('dataset_info', {})
            f.write(f"Samples: {info.get('num_samples', 0)}, ")
            f.write(f"Unique Queries: {info.get('num_unique_queries', 0)}, ")
            f.write(f"Unique Images: {info.get('num_unique_images', 0)}\n")
            f.write(f"Model Type: {info.get('model_type', 'unknown')}\n\n")

            ub = results.get('Upper_bound', {})
            f.write(f"Upper bound (no pruning):\n")
            f.write(f"  NDCG@5: {ub.get('ndcg5', 0.0):.4f}\n\n")

            # Show calibrated k values if available
            k_vals = results.get('calibrated_k_values', {})
            if k_vals:
                f.write(f"Calibrated k values (Adaptive-EOS):\n")
                for ratio, k_val in k_vals.items():
                    f.write(f"  Ratio {ratio:.1%}: k = {k_val:.4f}\n")
                f.write("\n")

            # Only show pruning results if available
            if 'SAP-Mean' in results:
                for ratio in ratios:
                    f.write(f"Ratio {ratio:.1%}:\n")

                    if ratio in results.get('SAP-Mean', {}):
                        sap_mean = results['SAP-Mean'][ratio]
                        f.write(f"  SAP-Mean:\n")
                        f.write(f"    Score Retention: {sap_mean.get('score_retention', 0.0):.4f}\n")
                        f.write(f"    NDCG@5: {sap_mean.get('ndcg5', 0.0):.4f}\n")
                        f.write(f"    NDCG Retention: {sap_mean.get('ndcg_retention', 0.0):.4f}\n")

                    if ratio in results.get('SAP-Max', {}):
                        sap_max = results['SAP-Max'][ratio]
                        f.write(f"  SAP-Max:\n")
                        f.write(f"    Score Retention: {sap_max.get('score_retention', 0.0):.4f}\n")
                        f.write(f"    NDCG@5: {sap_max.get('ndcg5', 0.0):.4f}\n")
                        f.write(f"    NDCG Retention: {sap_max.get('ndcg_retention', 0.0):.4f}\n")

                    if ratio in results.get('Cluster-Merge', {}):
                        cluster = results['Cluster-Merge'][ratio]
                        f.write(f"  Cluster-Merge:\n")
                        f.write(f"    Score Retention: {cluster.get('score_retention', 0.0):.4f}\n")
                        f.write(f"    NDCG@5: {cluster.get('ndcg5', 0.0):.4f}\n")
                        f.write(f"    NDCG Retention: {cluster.get('ndcg_retention', 0.0):.4f}\n")

                    if ratio in results.get('Adaptive-EOS', {}):
                        aeos = results['Adaptive-EOS'][ratio]
                        f.write(f"  Adaptive-EOS:\n")
                        f.write(f"    Calibrated k: {aeos.get('calibrated_k', 0.0):.4f}\n")
                        f.write(f"    Actual Ratio: {aeos.get('actual_ratio', 0.0):.1%}\n")
                        f.write(f"    Score Retention: {aeos.get('score_retention', 0.0):.4f}\n")
                        f.write(f"    NDCG@5: {aeos.get('ndcg5', 0.0):.4f}\n")
                        f.write(f"    NDCG Retention: {aeos.get('ndcg_retention', 0.0):.4f}\n")

                    if ratio in results.get('Random', {}):
                        rand = results['Random'][ratio]
                        f.write(f"  Random:\n")
                        f.write(f"    Score Retention: {rand.get('score_retention', 0.0):.4f} ± {rand.get('score_retention_std', 0.0):.4f}\n")
                        f.write(f"    NDCG@5: {rand.get('ndcg5', 0.0):.4f} ± {rand.get('ndcg5_std', 0.0):.4f}\n")
                        f.write(f"    NDCG Retention: {rand.get('ndcg_retention', 0.0):.4f} ± {rand.get('ndcg_retention_std', 0.0):.4f}\n")
                    f.write("\n")

    print(f"✅ Text Summary: {summary_txt}")

    print(f"\n{'='*80}")
    print(f"✅ BENCHMARK COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Files created:")
    print(f"  - benchmark_summary.json (complete results)")
    print(f"  - benchmark_summary.csv (for analysis)")
    print(f"  - benchmark_summary.txt (human-readable)")
    print(f"  - results_<dataset>.json (per-dataset details)")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
