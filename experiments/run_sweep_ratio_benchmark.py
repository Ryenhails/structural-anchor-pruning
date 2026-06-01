#!/usr/bin/env python3
"""
Sweep Ratio Benchmark: Evaluate all baselines across ratio range [0.1, 0.2, ..., 0.9]
Only uses Vidore v2 datasets and reports NDCG@5.

Supports: ColPali, ColQwen2, JinaEmbeddingsV4
Methods: Random, SAP-Mean, SAP-Max, Cluster-Merge, Adaptive-EOS, Upper bound
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
from typing import Dict, List, Optional, Tuple
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
from core.centrality import ensure_eager_attention

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

def load_vidore_v2_data(dataset_path: str):
    """Load Vidore v2 format data and build indices for efficient lookup."""
    print("Loading Queries config...")
    queries_ds = load_dataset(dataset_path, 'queries', split='test')

    print("Loading Corpus config...")
    corpus_ds = load_dataset(dataset_path, 'corpus', split='test')

    print("Loading Qrels config...")
    qrels_ds = load_dataset(dataset_path, 'qrels', split='test')

    print("Indexing Queries...")
    queries = {
        row['query-id']: row['query']
        for row in tqdm(queries_ds, desc="Index Queries")
    }

    print("Indexing Corpus...")
    corpus = {
        row['corpus-id']: row['image']
        for row in tqdm(corpus_ds, desc="Index Corpus")
    }

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

def load_model_and_processor(model_name: str, device: str):
    """Load model and processor based on model type"""
    model_type = detect_model_type(model_name)
    print(f"Model type: {model_type}")

    if model_type == 'jina':
        if not JINA_AVAILABLE:
            raise ImportError("Jina models require transformers library")

        print("Loading Jina model with forced EAGER attention...")

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.attn_implementation = "eager"

        if hasattr(config, 'vision_config'):
            config.vision_config.attn_implementation = "eager"
            config.vision_config._attn_implementation = "eager"
        if hasattr(config, 'text_config'):
            config.text_config.attn_implementation = "eager"
            config.text_config._attn_implementation = "eager"

        model = AutoModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager"
        ).eval()

        print("✅ Jina model loaded with Eager Attention")
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

    else:  # paligemma
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        ).eval()

    # Apply ensure_eager_attention for non-Jina models
    if model_type != 'jina':
        ensure_eager_attention(model)

    return model, processor, model_type

def get_image_token_id(model, model_type: str) -> Optional[int]:
    """Get image token ID for multi-vector models"""
    if model_type == 'jina':
        return None
    elif model_type == 'qwen2':
        return getattr(model.config, 'image_token_id', 151655)
    else:  # paligemma
        return getattr(model.model.config, 'image_token_index', 256000)


# ============================================================================
# Native Forward Calls
# ============================================================================

def get_embeddings(model, batch):
    """Native ColPali/ColQwen2 forward for embeddings"""
    return model(**batch)

def get_attentions(model, batch, model_type: str):
    """Native attention extraction matching forward pass"""
    with torch.inference_mode():
        if model_type == 'qwen2':
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
    """Extract attentions AND hidden states for Jina model."""
    with torch.inference_mode():
        kwargs = {k: v for k, v in batch.items()}
        if "pixel_values" in kwargs and "image_grid_thw" in kwargs:
            offsets = kwargs["image_grid_thw"][:, 1] * kwargs["image_grid_thw"][:, 2]
            kwargs["pixel_values"] = torch.cat(
                [pix[:off] for pix, off in zip(kwargs["pixel_values"], offsets)], dim=0
            )

        if hasattr(model, 'base_model'):
            jina_model = model.base_model.model
        elif hasattr(model, 'model'):
            jina_model = model.model
        else:
            jina_model = model

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
    """Compute centrality from visual token attention."""
    batch_size, num_heads = layer_attn.shape[:2]
    head_centralities = []

    for h in range(num_heads):
        attn_h = layer_attn[0, h]
        visual_attn = attn_h[visual_indices][:, visual_indices]
        centrality = visual_attn.sum(dim=-2)
        head_centralities.append(centrality)

    stacked = torch.stack(head_centralities)
    return stacked.mean(dim=0) if agg_mode == 'mean' else stacked.max(dim=0).values


# ============================================================================
# Cluster-Based Merging
# ============================================================================

def cluster_merge(embeddings: torch.Tensor, merging_factor: float) -> torch.Tensor:
    """Merge patch embeddings using hierarchical clustering."""
    num_patches = embeddings.shape[0]
    dim = embeddings.shape[1]

    num_clusters = max(1, int(num_patches / merging_factor))

    if num_clusters >= num_patches:
        return embeddings

    embeddings_fp32 = embeddings.float().cpu().numpy()
    embeddings_normalized = sklearn_normalize(embeddings_fp32, norm='l2', axis=1)

    cosine_sim = embeddings_normalized @ embeddings_normalized.T
    distance_matrix = 1 - cosine_sim

    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = distance_matrix.astype(np.float64)

    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric='precomputed',
        linkage='average'
    )

    cluster_labels = clustering.fit_predict(distance_matrix)

    merged_embeddings = torch.zeros((num_clusters, dim), dtype=embeddings.dtype)

    for cluster_id in range(num_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_members = embeddings[cluster_mask]

        if len(cluster_members) > 0:
            merged_embeddings[cluster_id] = cluster_members.mean(dim=0)

    return merged_embeddings


# ============================================================================
# Adaptive-EOS Attention Extraction
# ============================================================================

def extract_eos_attention(
    model,
    batch: Dict[str, torch.Tensor],
    model_type: str,
    visual_indices: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    """Extract EOS attention scores from the final layer."""
    all_attentions = get_attentions(model, batch, model_type)
    final_layer_attn = all_attentions[-1]

    input_ids = batch['input_ids'][0]
    attention_mask = batch.get('attention_mask', None)

    if attention_mask is not None:
        valid_positions = torch.where(attention_mask[0] == 1)[0]
        eos_position = valid_positions[-1].item()
    else:
        eos_position = len(input_ids) - 1

    eos_to_visual = final_layer_attn[0, :, eos_position, visual_indices]
    eos_scores = eos_to_visual.mean(dim=0)

    return eos_scores, len(visual_indices)


def calibrate_k_by_quantile(all_scores_list: List[torch.Tensor], target_ratio: float) -> float:
    """Calculate k based on Z-score distribution of the calibration set."""
    all_z_scores = []

    for scores in all_scores_list:
        mu = scores.mean()
        sigma = scores.std() + 1e-8

        z_scores = (scores - mu) / sigma
        all_z_scores.append(z_scores)

    flat_z = torch.cat(all_z_scores)
    k = torch.quantile(flat_z.float(), 1.0 - target_ratio)

    return k.item()


def apply_adaptive_pruning(
    embeddings: torch.Tensor,
    scores: torch.Tensor,
    k: float
) -> torch.Tensor:
    """Prune a single document using μ + k·σ threshold."""
    mu = scores.mean()
    sigma = scores.std() + 1e-8
    threshold = mu + k * sigma

    mask = scores > threshold

    if mask.sum() == 0:
        max_idx = scores.argmax()
        mask[max_idx] = True

    pruned_embeddings = embeddings[mask]
    return pruned_embeddings


# ============================================================================
# Scoring
# ============================================================================

def compute_score_matrix_maxsim(query_embs: List[torch.Tensor], doc_embs: List[torch.Tensor], device: str):
    """MaxSim scoring for multi-vector embeddings"""
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


# ============================================================================
# Main Evaluation
# ============================================================================

def evaluate_sweep_ratios(
    model, processor, dataset, dataset_name, target_layers: List[int],
    ratios, device, model_type, image_token_id, num_random_trials=5,
    calibration_size=128
):
    """Evaluate model on dataset with different pruning strategies (NDCG@5 only)."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {dataset_name} (Model: {model_type})")
    print(f"{'='*80}")
    print(f"Step 1/5: Encoding corpus and queries...")

    is_jina = (model_type == 'jina')

    unique_image_embs = []
    unique_query_embs = []
    unique_sap_mean_scores = []
    unique_sap_max_scores = []
    unique_eos_scores = []

    sample_to_unique = []

    # Encode queries
    if is_jina:
        print("Encoding all queries (Jina multi-vector)...")
        query_texts = list(dataset['queries'].values())
        query_ids = list(dataset['queries'].keys())

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

        corpus_id_to_idx = {}
        for i, (c_id, image) in enumerate(tqdm(zip(corpus_ids, corpus_images),
                                                desc="Jina Encode & Attn",
                                                total=len(corpus_ids))):

            batch_images = model.processor.process_images([image])
            batch_images = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch_images.items()}

            all_attentions, last_hidden_state = get_jina_outputs(model, batch_images)

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

            with torch.inference_mode():
                if hasattr(model, 'base_model'):
                    jina_model = model.base_model.model
                elif hasattr(model, 'model'):
                    jina_model = model.model
                else:
                    jina_model = model

                visual_hidden = last_hidden_state[0, visual_indices]
                visual_emb = jina_model.multi_vector_projector(visual_hidden, task_label='retrieval')
                visual_emb = torch.nn.functional.normalize(visual_emb, p=2, dim=-1)

                unique_image_embs.append(visual_emb.cpu())

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
        # ColPali/ColQwen2
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

            input_ids = batch_images['input_ids'][0]
            visual_mask = (input_ids == image_token_id).unsqueeze(-1)
            visual_indices = torch.where(visual_mask)[0]
            image_emb = image_emb_full[0, visual_indices]

            unique_image_embs.append(image_emb.cpu())

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

            eos_scores, _ = extract_eos_attention(model, batch_images, model_type, visual_indices)
            unique_eos_scores.append(eos_scores.cpu())

            corpus_id_to_idx[c_id] = len(unique_image_embs) - 1

    # Build ground truth from qrels
    print("Building ground truth from qrels...")
    for rel in tqdm(dataset['qrels'], desc="Process Qrels"):
        q_id = rel['query-id']
        c_id = rel['corpus-id']

        if q_id in query_id_to_idx and c_id in corpus_id_to_idx:
            q_idx = query_id_to_idx[q_id]
            c_idx = corpus_id_to_idx[c_id]
            sample_to_unique.append((q_idx, c_idx))

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

    score_fn = compute_score_matrix_maxsim

    # Compute baseline (Upper bound) scores
    print("Computing baseline scores (Upper bound)...")
    score_matrix_full = score_fn(unique_query_embs, unique_image_embs, device)
    ndcg_full = ndcg_score(y_true, score_matrix_full, k=5)

    results = {
        'dataset_info': {
            'num_samples': num_samples,
            'num_unique_queries': num_unique_queries,
            'num_unique_images': num_unique_images,
            'model_type': model_type
        },
        'Upper_bound': {
            'ndcg5': float(ndcg_full)
        },
        'SAP-Mean': {},
        'SAP-Max': {},
        'Cluster-Merge': {},
        'Adaptive-EOS': {},
        'Random': {}
    }

    # ============================================================
    # Step 3: Calibration for Adaptive-EOS
    # ============================================================
    print(f"\nStep 3/5: Calibration Phase for Adaptive-EOS")
    print(f"Using first {min(calibration_size, num_unique_images)} images for calibration...")

    calibration_scores = unique_eos_scores[:calibration_size]

    k_map = {}
    for ratio in ratios:
        k_val = calibrate_k_by_quantile(calibration_scores, ratio)
        k_map[ratio] = k_val

    results['calibrated_k_values'] = k_map

    # Pre-compute sort indices
    print("\nStep 4/5: Pre-computing sort indices...")
    sorted_indices_sap_mean = [torch.argsort(scores, descending=True)
                               for scores in tqdm(unique_sap_mean_scores, desc="SAP-Mean sorting")]
    sorted_indices_sap_max = [torch.argsort(scores, descending=True)
                              for scores in tqdm(unique_sap_max_scores, desc="SAP-Max sorting")]

    random_permutations = [[torch.randperm(emb.shape[0]) for _ in range(num_random_trials)]
                          for emb in unique_image_embs]

    # Ratio sweep
    print("\nStep 5/5: Evaluating sweep ratios across all methods...")

    for ratio in tqdm(ratios, desc=f"Processing ratios"):
        # ============= SAP-Mean =============
        pruned_docs_sap_mean = []
        for i in range(num_unique_images):
            img = unique_image_embs[i]
            N = img.shape[0]
            k = max(1, min(N, int(N * ratio)))
            idx = sorted_indices_sap_mean[i][:k]
            pruned_docs_sap_mean.append(img[idx])

        score_mat_sap_mean = compute_score_matrix_maxsim(unique_query_embs, pruned_docs_sap_mean, device)
        ndcg_sap_mean = ndcg_score(y_true, score_mat_sap_mean, k=5)

        results['SAP-Mean'][ratio] = {
            'ndcg5': float(ndcg_sap_mean)
        }

        # ============= SAP-Max =============
        pruned_docs_sap_max = []
        for i in range(num_unique_images):
            img = unique_image_embs[i]
            N = img.shape[0]
            k = max(1, min(N, int(N * ratio)))
            idx = sorted_indices_sap_max[i][:k]
            pruned_docs_sap_max.append(img[idx])

        score_mat_sap_max = compute_score_matrix_maxsim(unique_query_embs, pruned_docs_sap_max, device)
        ndcg_sap_max = ndcg_score(y_true, score_mat_sap_max, k=5)

        results['SAP-Max'][ratio] = {
            'ndcg5': float(ndcg_sap_max)
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

        results['Cluster-Merge'][ratio] = {
            'ndcg5': float(ndcg_cluster)
        }

        # ============= Adaptive-EOS =============
        k_val = k_map[ratio]

        pruned_docs_adaptive = []
        total_tokens = 0
        kept_tokens = 0

        for i in range(num_unique_images):
            img_emb = unique_image_embs[i]
            eos_scores = unique_eos_scores[i]

            pruned_emb = apply_adaptive_pruning(img_emb, eos_scores, k_val)
            pruned_docs_adaptive.append(pruned_emb)

            total_tokens += img_emb.shape[0]
            kept_tokens += pruned_emb.shape[0]

        actual_ratio = kept_tokens / total_tokens if total_tokens > 0 else 0.0

        score_mat_adaptive = compute_score_matrix_maxsim(unique_query_embs, pruned_docs_adaptive, device)
        ndcg_adaptive = ndcg_score(y_true, score_mat_adaptive, k=5)

        results['Adaptive-EOS'][ratio] = {
            'target_ratio': float(ratio),
            'actual_ratio': float(actual_ratio),
            'ndcg5': float(ndcg_adaptive)
        }

        # ============= Random =============
        trial_ndcgs = []

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

        results['Random'][ratio] = {
            'ndcg5': float(np.mean(trial_ndcgs)),
            'ndcg5_std': float(np.std(trial_ndcgs))
        }

    return results


# ============================================================================
# Dataset Configuration
# ============================================================================

def get_vidore_v2_datasets():
    """Return all Vidore v2 datasets"""
    return [
        ("esg_reports_v2", "vidore/esg_reports_v2"),
        ("biomedical_lectures_v2", "vidore/biomedical_lectures_v2"),
        ("economics_reports_v2", "vidore/economics_reports_v2"),
        ("esg_reports_human_v2", "vidore/esg_reports_human_labeled_v2"),
    ]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sweep Ratio Benchmark: Evaluate baselines across [0.1-0.9]')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--layers', type=str, default="auto",
                       help='Comma-separated layer indices for SAP (default: "auto" - model-specific defaults)')
    parser.add_argument('--calibration_size', type=int, default=128,
                       help='Number of samples for Adaptive-EOS calibration (default: 128)')
    parser.add_argument('--output_dir', type=str, default='./experiments/outputs',
                       help='Output directory for results')
    args = parser.parse_args()

    # Detect model type and set default layers
    model_type_detected = detect_model_type(args.model_name)

    if args.layers == "auto":
        # Set default layers based on model type
        if model_type_detected == 'jina':
            target_layers = [14, 15, 16, 17, 18, 19, 20, 21]  # middle 8 layers for Jina
            print(f"Using auto-detected layers for Jina: {target_layers}")
        elif model_type_detected == 'qwen2':
            target_layers = [11, 12, 13, 14, 15, 16]  # middle 6 layers for Qwen2
            print(f"Using auto-detected layers for Qwen2: {target_layers}")
        else:
            target_layers = [8, 9, 10, 11]  # middle 4 layers for PaliGemma
            print(f"Using auto-detected layers for PaliGemma: {target_layers}")
    else:
        try:
            target_layers = [int(x.strip()) for x in args.layers.split(',')]
        except ValueError:
            print("Error: --layers must be comma-separated integers or 'auto'")
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = args.model_name.split('/')[-1]
    output_dir = Path(args.output_dir) / f"sweep_ratio_benchmark_{model_short_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("SWEEP RATIO BENCHMARK - VIDORE V2 ONLY")
    print(f"{'='*80}")
    print(f"Model:  {args.model_name}")
    print(f"Type:   {model_type_detected}")
    print(f"Layers: {target_layers}")
    print(f"Sweep Ratios: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]")
    print(f"Calibration Size: {args.calibration_size}")
    print(f"Methods:")
    print(f"  - Random")
    print(f"  - SAP-Mean")
    print(f"  - SAP-Max")
    print(f"  - Cluster-Merge")
    print(f"  - Adaptive-EOS")
    print(f"  - Upper bound")
    print(f"Metric: NDCG@5")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    device = get_torch_device("auto")
    model, processor, model_type = load_model_and_processor(args.model_name, device)
    image_token_id = get_image_token_id(model, model_type)
    print(f"✅ Model loaded on {device}, image_token_id={image_token_id}\n")

    vidore_v2_datasets = get_vidore_v2_datasets()
    print(f"📊 Vidore v2: {len(vidore_v2_datasets)} datasets\n")

    # Sweep ratios from 0.1 to 0.9
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"Sweep Ratios: {ratios}\n")

    all_results = {}

    for dataset_name, dataset_path in vidore_v2_datasets:
        print(f"\n{'#'*80}")
        print(f"# {dataset_name.upper()}")
        print(f"# Path: {dataset_path}")
        print(f"{'#'*80}")

        try:
            dataset_obj = load_vidore_v2_data(dataset_path)
            print(f"✅ Loaded Vidore V2 Data. Qrels pairs: {len(dataset_obj['qrels'])}")

            results = evaluate_sweep_ratios(
                model, processor, dataset_obj, dataset_name,
                target_layers=target_layers,
                ratios=ratios,
                device=device,
                model_type=model_type,
                image_token_id=image_token_id,
                num_random_trials=5,
                calibration_size=args.calibration_size
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
        writer.writerow(['Dataset', 'Method', 'Ratio', 'NDCG5', 'NDCG5_Std'])

        for dataset_name, results in all_results.items():
            # Upper bound
            ub = results.get('Upper_bound', {})
            writer.writerow([
                dataset_name, 'Upper_bound', 1.0,
                ub.get('ndcg5', 0.0), 0.0
            ])

            for ratio in ratios:
                # SAP-Mean
                if ratio in results.get('SAP-Mean', {}):
                    sap_mean = results['SAP-Mean'][ratio]
                    writer.writerow([
                        dataset_name, 'SAP-Mean', ratio,
                        sap_mean.get('ndcg5', 0.0), 0.0
                    ])

                # SAP-Max
                if ratio in results.get('SAP-Max', {}):
                    sap_max = results['SAP-Max'][ratio]
                    writer.writerow([
                        dataset_name, 'SAP-Max', ratio,
                        sap_max.get('ndcg5', 0.0), 0.0
                    ])

                # Cluster-Merge
                if ratio in results.get('Cluster-Merge', {}):
                    cluster = results['Cluster-Merge'][ratio]
                    writer.writerow([
                        dataset_name, 'Cluster-Merge', ratio,
                        cluster.get('ndcg5', 0.0), 0.0
                    ])

                # Adaptive-EOS
                if ratio in results.get('Adaptive-EOS', {}):
                    aeos = results['Adaptive-EOS'][ratio]
                    writer.writerow([
                        dataset_name, 'Adaptive-EOS', ratio,
                        aeos.get('ndcg5', 0.0), 0.0
                    ])

                # Random
                if ratio in results.get('Random', {}):
                    rand = results['Random'][ratio]
                    writer.writerow([
                        dataset_name, 'Random', ratio,
                        rand.get('ndcg5', 0.0), rand.get('ndcg5_std', 0.0)
                    ])

    print(f"✅ CSV Summary: {csv_file}")

    # Create a human-readable summary table
    summary_txt = output_dir / "benchmark_summary.txt"
    with open(summary_txt, 'w') as f:
        f.write("="*100 + "\n")
        f.write("SWEEP RATIO BENCHMARK SUMMARY (VIDORE V2 ONLY, NDCG@5)\n")
        f.write("="*100 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Type:  {model_type_detected}\n")
        f.write(f"Layers: {target_layers}\n")
        f.write(f"Ratios: {ratios}\n")
        f.write(f"Calibration Size: {args.calibration_size}\n")
        f.write(f"Metric: NDCG@5\n\n")

        for dataset_name, results in all_results.items():
            f.write(f"\n{'-'*100}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"{'-'*100}\n")

            info = results.get('dataset_info', {})
            f.write(f"Samples: {info.get('num_samples', 0)}, ")
            f.write(f"Unique Queries: {info.get('num_unique_queries', 0)}, ")
            f.write(f"Unique Images: {info.get('num_unique_images', 0)}\n\n")

            ub = results.get('Upper_bound', {})
            f.write(f"Upper bound: NDCG@5 = {ub.get('ndcg5', 0.0):.4f}\n\n")

            # Show calibrated k values if available
            k_vals = results.get('calibrated_k_values', {})
            if k_vals:
                f.write(f"Calibrated k values (Adaptive-EOS):\n")
                for ratio, k_val in k_vals.items():
                    f.write(f"  Ratio {ratio:.1%}: k = {k_val:.4f}\n")
                f.write("\n")

            # Create table
            f.write(f"{'Ratio':<8} {'Random':<12} {'SAP-Mean':<12} {'SAP-Max':<12} {'Cluster':<12} {'Adaptive-EOS':<12}\n")
            f.write("-" * 80 + "\n")

            for ratio in ratios:
                row = [f"{ratio:.1f}"]

                # Random
                if ratio in results.get('Random', {}):
                    rand = results['Random'][ratio]
                    row.append(f"{rand.get('ndcg5', 0.0):.4f}±{rand.get('ndcg5_std', 0.0):.4f}")
                else:
                    row.append("N/A")

                # SAP-Mean
                if ratio in results.get('SAP-Mean', {}):
                    row.append(f"{results['SAP-Mean'][ratio].get('ndcg5', 0.0):.4f}")
                else:
                    row.append("N/A")

                # SAP-Max
                if ratio in results.get('SAP-Max', {}):
                    row.append(f"{results['SAP-Max'][ratio].get('ndcg5', 0.0):.4f}")
                else:
                    row.append("N/A")

                # Cluster-Merge
                if ratio in results.get('Cluster-Merge', {}):
                    row.append(f"{results['Cluster-Merge'][ratio].get('ndcg5', 0.0):.4f}")
                else:
                    row.append("N/A")

                # Adaptive-EOS
                if ratio in results.get('Adaptive-EOS', {}):
                    aeos = results['Adaptive-EOS'][ratio]
                    row.append(f"{aeos.get('ndcg5', 0.0):.4f}")
                else:
                    row.append("N/A")

                f.write(f"{row[0]:<8} {row[1]:<12} {row[2]:<12} {row[3]:<12} {row[4]:<12} {row[5]:<12}\n")

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
