#!/usr/bin/env python3
"""
Fixed Ratio Benchmark v2 - LightColPali version
Evaluates SAP-Mean and SAP-Max on selected Vidore v1 datasets
with specific pruning ratios: [0.25, 0.1111111111, 0.04, 0.02041]
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
from typing import Dict, List
from datasets import load_dataset
from sklearn.metrics import ndcg_score
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.centrality import ensure_eager_attention


# ============================================================================
# Utilities
# ============================================================================

def get_image_hash(image: Image.Image) -> str:
    """Get hash of image for deduplication"""
    return hashlib.md5(image.tobytes()).hexdigest()


# ============================================================================
# Model Loading
# ============================================================================

def detect_model_type(model_name: str) -> str:
    return 'qwen2' if 'qwen' in model_name.lower() else 'paligemma'

def load_model_and_processor(model_name: str, device: str):
    model_type = detect_model_type(model_name)
    print(f"Model type: {model_type}")

    if model_type == 'qwen2':
        if not COLQWEN2_AVAILABLE:
            raise ImportError("ColQwen2 not available")
        processor = ColQwen2Processor.from_pretrained(model_name)
        model = ColQwen2.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager"
        ).eval()
    else:
        processor = ColPaliProcessor.from_pretrained(model_name)
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device
        ).eval()

    ensure_eager_attention(model)
    return model, processor, model_type

def get_image_token_id(model, model_type: str) -> int:
    if model_type == 'qwen2':
        return getattr(model.config, 'image_token_id', 151655)
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


# ============================================================================
# Centrality Computation
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
# Scoring
# ============================================================================

def compute_score_matrix(query_embs: List[torch.Tensor], doc_embs: List[torch.Tensor], device: str):
    """Standard MaxSim scoring"""
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

def evaluate_fixed_ratios(
    model, processor, dataset, dataset_name, target_layers: List[int],
    ratios, device, model_type, image_token_id
):
    print(f"\n{'='*80}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*80}")
    print(f"Step 1/4: Encoding and building unique corpus...")

    # Track unique images and queries
    image_hash_to_idx = {}
    query_to_idx = {}

    unique_image_embs = []
    unique_query_embs = []
    unique_sap_mean_scores = []  # SAP-Mean scores (mean over attention heads)
    unique_sap_max_scores = []   # SAP-Max scores (max over attention heads)

    # Map: sample_idx -> (unique_query_idx, unique_image_idx)
    sample_to_unique = []

    total_len = len(dataset)

    for idx in tqdm(range(total_len), desc="Encoding"):
        try:
            sample = dataset[idx]
            if 'image' not in sample or 'query' not in sample:
                continue

            image = sample['image'].convert('RGB')
            query = sample['query']

            # Check if image is already processed
            img_hash = get_image_hash(image)
            if img_hash in image_hash_to_idx:
                unique_img_idx = image_hash_to_idx[img_hash]
            else:
                # Process new image
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

                unique_img_idx = len(unique_image_embs) - 1
                image_hash_to_idx[img_hash] = unique_img_idx

            # Check if query is already processed
            if query in query_to_idx:
                unique_q_idx = query_to_idx[query]
            else:
                # Process new query
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
    print("Step 2/4: Building ground truth matrix...")
    y_true = np.zeros((num_unique_queries, num_unique_images), dtype=np.float32)
    for q_idx, img_idx in sample_to_unique:
        y_true[q_idx, img_idx] = 1.0

    # Pre-compute sort indices for SAP-Mean and SAP-Max
    print("Step 3/4: Pre-computing sort indices...")
    sorted_indices_sap_mean = [torch.argsort(scores, descending=True)
                               for scores in tqdm(unique_sap_mean_scores, desc="SAP-Mean sorting")]
    sorted_indices_sap_max = [torch.argsort(scores, descending=True)
                              for scores in tqdm(unique_sap_max_scores, desc="SAP-Max sorting")]

    # Compute baseline (Full/Upper bound) scores
    print("Computing baseline scores (Upper bound - no pruning)...")
    score_matrix_full = compute_score_matrix(unique_query_embs, unique_image_embs, device)

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
            'num_unique_images': num_unique_images
        },
        'Upper_bound': {
            'score_retention': 1.0,  # By definition
            'ndcg5': float(ndcg_full),
            'ndcg_retention': 1.0  # By definition
        },
        'SAP-Mean': {},
        'SAP-Max': {}
    }

    # Ratio sweep
    print("Step 4/4: Evaluating fixed ratios...")

    for ratio in tqdm(ratios, desc=f"Processing ratios"):
        # ============= SAP-Mean (middle 4 layers, mean over heads) =============
        pruned_docs_sap_mean = []
        for i in range(num_unique_images):
            img = unique_image_embs[i]
            N = img.shape[0]
            k = max(1, min(N, int(N * ratio)))
            idx = sorted_indices_sap_mean[i][:k]
            pruned_docs_sap_mean.append(img[idx])

        score_mat_sap_mean = compute_score_matrix(unique_query_embs, pruned_docs_sap_mean, device)
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

        score_mat_sap_max = compute_score_matrix(unique_query_embs, pruned_docs_sap_max, device)
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

    return results


# ============================================================================
# Dataset Configuration
# ============================================================================

def get_selected_vidore_v1_datasets():
    """Return selected Vidore v1 datasets"""
    return [
        ("infovqa", "vidore/infovqa_test_subsampled"),
        ("docvqa", "vidore/docvqa_test_subsampled"),
        ("arxivqa", "vidore/arxivqa_test_subsampled"),
        ("tabfquad", "vidore/tabfquad_test_subsampled"),
        ("tatdqa", "vidore/tatdqa_test"),
        ("shiftproject", "vidore/shiftproject_test"),
    ]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fixed Ratio Benchmark v2 - LightColPali')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name or path (e.g., vidore/colpali-v1.2)')
    parser.add_argument('--layers', type=str, default="8,9,10,11",
                       help='Comma-separated layer indices for SAP (default: middle 4 layers "8,9,10,11")')
    parser.add_argument('--output_dir', type=str, default='./experiments/outputs',
                       help='Output directory for results')
    args = parser.parse_args()

    try:
        target_layers = [int(x.strip()) for x in args.layers.split(',')]
    except ValueError:
        print("Error: --layers must be comma-separated integers")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short_name = args.model_name.split('/')[-1]
    output_dir = Path(args.output_dir) / f"fixed_ratio_benchmark_v2_lightcolpali_{model_short_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print("FIXED RATIO BENCHMARK V2 - LIGHTCOLPALI VERSION")
    print(f"{'='*80}")
    print(f"Model:  {args.model_name}")
    print(f"Layers: {target_layers}")
    print(f"Fixed Ratios: [0.25, 0.1111111111, 0.04, 0.02041]")
    print(f"Methods:")
    print(f"  - SAP-Mean (middle 4 layers, mean over attention heads)")
    print(f"  - SAP-Max (middle 4 layers, max over attention heads)")
    print(f"  - Upper bound (no pruning)")
    print(f"Metrics: Score Retention, NDCG@5, NDCG Retention")
    print(f"Datasets: InfoVQA, DocVQA, ArxivQA, TabFQuAD, TATDQA, ShiftProject")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    device = get_torch_device("auto")
    model, processor, model_type = load_model_and_processor(args.model_name, device)
    image_token_id = get_image_token_id(model, model_type)
    print(f"✅ Model loaded on {device}, image_token_id={image_token_id}\n")

    vidore_v1_datasets = get_selected_vidore_v1_datasets()
    print(f"📊 Total datasets to evaluate: {len(vidore_v1_datasets)}\n")

    # Fixed ratios for LightColPali
    ratios = [0.25, 0.1111111111, 0.04, 0.02041]
    print(f"Fixed Ratios: {ratios}\n")

    all_results = {}

    for dataset_name, dataset_path in vidore_v1_datasets:
        print(f"\n{'#'*80}")
        print(f"# {dataset_name.upper()}")
        print(f"# Path: {dataset_path}")
        print(f"{'#'*80}")

        try:
            dataset_obj = load_dataset(dataset_path, split="test")
            print(f"✅ Loaded {len(dataset_obj)} samples")

            results = evaluate_fixed_ratios(
                model, processor, dataset_obj, dataset_name,
                target_layers=target_layers,
                ratios=ratios,
                device=device,
                model_type=model_type,
                image_token_id=image_token_id
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
        writer.writerow(['Dataset', 'Method', 'Ratio', 'Score_Retention',
                        'NDCG5', 'NDCG_Retention'])

        for dataset_name, results in all_results.items():
            # Upper bound (no pruning)
            ub = results.get('Upper_bound', {})
            writer.writerow([
                dataset_name, 'Upper_bound', 1.0,
                ub.get('score_retention', 1.0),
                ub.get('ndcg5', 0.0),
                ub.get('ndcg_retention', 1.0)
            ])

            # Results for each ratio and method
            for ratio in ratios:
                # SAP-Mean
                if ratio in results.get('SAP-Mean', {}):
                    sap_mean = results['SAP-Mean'][ratio]
                    writer.writerow([
                        dataset_name, 'SAP-Mean', ratio,
                        sap_mean.get('score_retention', 0.0),
                        sap_mean.get('ndcg5', 0.0),
                        sap_mean.get('ndcg_retention', 0.0)
                    ])

                # SAP-Max
                if ratio in results.get('SAP-Max', {}):
                    sap_max = results['SAP-Max'][ratio]
                    writer.writerow([
                        dataset_name, 'SAP-Max', ratio,
                        sap_max.get('score_retention', 0.0),
                        sap_max.get('ndcg5', 0.0),
                        sap_max.get('ndcg_retention', 0.0)
                    ])

    print(f"✅ CSV Summary: {csv_file}")

    # Create a human-readable summary table
    summary_txt = output_dir / "benchmark_summary.txt"
    with open(summary_txt, 'w') as f:
        f.write("="*100 + "\n")
        f.write("FIXED RATIO BENCHMARK V2 - LIGHTCOLPALI SUMMARY\n")
        f.write("="*100 + "\n\n")
        f.write(f"Model: {args.model_name}\n")
        f.write(f"Layers: {target_layers}\n")
        f.write(f"Ratios: {ratios}\n")
        f.write(f"Methods:\n")
        f.write(f"  - SAP-Mean (middle 4 layers, mean over attention heads)\n")
        f.write(f"  - SAP-Max (middle 4 layers, max over attention heads)\n")
        f.write(f"  - Upper bound (no pruning)\n")
        f.write(f"Metrics: Score Retention, NDCG@5, NDCG Retention\n")
        f.write(f"Datasets: InfoVQA, DocVQA, ArxivQA, TabFQuAD, TATDQA, ShiftProject\n\n")

        for dataset_name, results in all_results.items():
            f.write(f"\n{'-'*100}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"{'-'*100}\n")

            info = results.get('dataset_info', {})
            f.write(f"Samples: {info.get('num_samples', 0)}, ")
            f.write(f"Unique Queries: {info.get('num_unique_queries', 0)}, ")
            f.write(f"Unique Images: {info.get('num_unique_images', 0)}\n\n")

            ub = results.get('Upper_bound', {})
            f.write(f"Upper bound (no pruning):\n")
            f.write(f"  NDCG@5: {ub.get('ndcg5', 0.0):.4f}\n\n")

            for ratio in ratios:
                f.write(f"Ratio {ratio}:\n")

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
