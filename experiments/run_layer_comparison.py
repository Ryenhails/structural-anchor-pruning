#!/usr/bin/env python3
"""
Section 4.1: Full Layer-wise Analysis Across ViDoRE Benchmark v3 (OPTIMIZED)
==============================================================================

Optimization: Single-pass inference per image.
- Run model ONCE per image with output_attentions=True and output_hidden_states=True
- Cache all layer outputs (attentions, hidden_states)
- Iterate through cached layers using pure tensor operations
- ~128x speedup (32 layers × 4 metrics → 1 inference per image)

This is a performance-optimized version of run_layer_comparison_v3.py.
Output JSON structure remains identical for compatibility.
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from datasets import load_dataset
import sys

from colpali_engine.models import ColPali, ColPaliProcessor

# Import core modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.centrality import ensure_eager_attention


# ============================================================================
# Optimized Metric Functions - NO MODEL INFERENCE
# ============================================================================

def extract_visual_attention_all_heads_cached(
    layer_attn: torch.Tensor,
    visual_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Extract Visual-to-Visual attention from cached layer attention.

    Args:
        layer_attn: [batch, num_heads, seq_len, seq_len] from cached attentions
        visual_indices: [N_visual] indices of visual tokens

    Returns:
        [batch, num_heads, N_visual, N_visual]
    """
    batch_size = layer_attn.shape[0]
    all_visual_attn = []

    for b in range(batch_size):
        head_visual_attn = []
        for h in range(layer_attn.shape[1]):
            head_attn = layer_attn[b, h]
            visual_attn = head_attn[visual_indices][:, visual_indices]
            head_visual_attn.append(visual_attn)

        head_visual_attn = torch.stack(head_visual_attn)
        all_visual_attn.append(head_visual_attn)

    return torch.stack(all_visual_attn)


def get_mean_centrality_cached(
    layer_attn: torch.Tensor,
    visual_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-Centrality from cached attention.

    Args:
        layer_attn: [batch, num_heads, seq_len, seq_len]
        visual_indices: [N_visual]

    Returns:
        [batch, N_visual]
    """
    attn = extract_visual_attention_all_heads_cached(layer_attn, visual_indices)
    # attn: [batch, num_heads, n_visual, n_visual]
    centrality_per_head = attn.sum(dim=-2)  # [batch, num_heads, n_visual]
    mean_centrality = centrality_per_head.mean(dim=1)  # [batch, n_visual]
    return mean_centrality


def get_max_centrality_cached(
    layer_attn: torch.Tensor,
    visual_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Max-Centrality from cached attention.

    Args:
        layer_attn: [batch, num_heads, seq_len, seq_len]
        visual_indices: [N_visual]

    Returns:
        [batch, N_visual]
    """
    attn = extract_visual_attention_all_heads_cached(layer_attn, visual_indices)
    centrality_per_head = attn.sum(dim=-2)  # [batch, num_heads, n_visual]
    max_centrality = centrality_per_head.max(dim=1).values  # [batch, n_visual]
    return max_centrality


def get_eos_attention_cached(
    layer_attn: torch.Tensor,
    visual_indices: torch.Tensor,
    eos_position: int,
) -> torch.Tensor:
    """
    EOS-Attention from cached attention.

    Args:
        layer_attn: [batch, num_heads, seq_len, seq_len]
        visual_indices: [N_visual]
        eos_position: int, position of EOS token

    Returns:
        [batch, N_visual]
    """
    batch_size = layer_attn.shape[0]
    all_eos_attn = []

    for b in range(batch_size):
        # Extract EOS-to-Visual attention, average over heads
        eos_to_visual = layer_attn[b, :, eos_position, visual_indices]
        eos_scores = eos_to_visual.mean(dim=0)
        all_eos_attn.append(eos_scores)

    return torch.stack(all_eos_attn)


def get_spectral_leverage_cached(
    layer_hidden: torch.Tensor,
    visual_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Spectral-Leverage from cached hidden states.

    Args:
        layer_hidden: [batch, seq_len, hidden_dim]
        visual_indices: [N_visual]

    Returns:
        [batch, N_visual]
    """
    batch_size = layer_hidden.shape[0]
    all_leverage_scores = []

    for b in range(batch_size):
        # Extract visual hidden states
        visual_hidden = layer_hidden[b, visual_indices, :]  # [n_visual, hidden_dim]

        # Convert to float32 (SVD not supported for bfloat16)
        visual_hidden_float = visual_hidden.float()

        # Center the embeddings
        visual_centered = visual_hidden_float - visual_hidden_float.mean(dim=0, keepdim=True)

        # Compute SVD
        U, S, V = torch.linalg.svd(visual_centered, full_matrices=False)

        # Leverage scores = ||U[i, :]||²
        leverage_scores = (U ** 2).sum(dim=1)

        all_leverage_scores.append(leverage_scores)

    return torch.stack(all_leverage_scores)


def compute_maxsim(query_embeddings: torch.Tensor, doc_embeddings: torch.Tensor) -> float:
    """Compute MaxSim score."""
    similarity = torch.einsum('qd,pd->qp', query_embeddings, doc_embeddings)
    max_sim = similarity.max(dim=1).values.sum().item()
    return max_sim


# ============================================================================
# Single-Pass Inference and Caching
# ============================================================================

def run_single_pass_inference(
    model,
    batch: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run model ONCE and cache all layer outputs.

    Args:
        model: ColPali model
        batch: Processed batch

    Returns:
        attentions: Tuple of [batch, num_heads, seq_len, seq_len] for each layer
        hidden_states: Tuple of [batch, seq_len, hidden_dim] for each layer
    """
    pali_model = model.model

    with torch.inference_mode():
        outputs = pali_model(
            **batch,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True
        )

    return outputs.attentions, outputs.hidden_states


# ============================================================================
# Optimized Evaluation: Single Dataset Layer Sweep
# ============================================================================

def evaluate_dataset_layer_sweep_optimized(
    model,
    processor,
    dataset_name: str,
    num_layers: int,
    ratio: float,
    device: str,
    num_samples: int = None,
    num_random_trials: int = 5,
) -> Dict:
    """
    Optimized evaluation with single-pass inference per image.

    Key optimization: Run model ONCE per image, cache all layer outputs,
    then iterate through layers using pure tensor operations.
    """
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")

    # Load dataset
    dataset = load_dataset(dataset_name, split="test")

    if num_samples is None:
        num_samples = len(dataset)
    else:
        num_samples = min(num_samples, len(dataset))

    print(f"Evaluating {num_samples} samples across {num_layers} layers...")
    print(f"Ratio: {ratio:.1%}")
    print(f"Optimization: Single-pass inference (1 forward pass per image)")
    print()

    # Initialize results structure (per layer, per method)
    results = {
        'Mean-Centrality': {layer: [] for layer in range(num_layers)},
        'Max-Centrality': {layer: [] for layer in range(num_layers)},
        'EOS-Attention': {layer: [] for layer in range(num_layers)},
        'Spectral-Leverage': {layer: [] for layer in range(num_layers)},
        'Random': [],  # Layer-independent
    }

    for idx in tqdm(range(num_samples), desc=f"Processing {dataset_name}"):
        try:
            sample = dataset[idx]

            # Validate sample has required fields
            if 'image' not in sample or sample['image'] is None:
                print(f"\n⚠️  Skipping sample {idx}: Missing or null image")
                continue

            if 'query' not in sample or sample['query'] is None:
                print(f"\n⚠️  Skipping sample {idx}: Missing or null query")
                continue

            image = sample['image'].convert('RGB')
            query = sample['query']

            # Process
            batch_images = processor.process_images([image])
            batch_queries = processor.process_queries([query])

            batch_images = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in batch_images.items()}
            batch_queries = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch_queries.items()}

            # Encode query (standard forward pass)
            with torch.inference_mode():
                query_emb = model(**batch_queries)

            # ================================================================
            # OPTIMIZATION: Single-pass inference with all outputs cached
            # ================================================================
            attentions, hidden_states = run_single_pass_inference(model, batch_images)

            # Encode image embeddings (for final embedding extraction)
            with torch.inference_mode():
                image_emb_full = model(**batch_images)

            # Extract visual token information (ONCE per image)
            image_token_id = model.model.config.image_token_index
            input_ids = batch_images['input_ids'][0]
            image_token_mask = (input_ids == image_token_id)
            visual_indices = torch.where(image_token_mask)[0]

            # Get visual-only embeddings
            image_emb = image_emb_full[0, visual_indices, :]  # [N_visual, dim]
            N = image_emb.shape[0]

            # Compute full score (reference)
            full_score = compute_maxsim(query_emb[0], image_emb)

            # Dynamic K calculation
            k = max(1, int(N * ratio))

            # Get EOS position (ONCE per image)
            attention_mask = batch_images.get('attention_mask', None)
            if attention_mask is not None:
                valid_positions = torch.where(attention_mask[0] == 1)[0]
                eos_position = valid_positions[-1].item()
            else:
                eos_position = len(input_ids) - 1

            # ================================================================
            # Iterate through CACHED layers (no model inference in loop!)
            # ================================================================
            for layer_idx in range(num_layers):
                # Get cached layer outputs
                layer_attn = attentions[layer_idx]  # [batch, num_heads, seq_len, seq_len]
                layer_hidden = hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]

                # Mean-Centrality (cached)
                mean_scores = get_mean_centrality_cached(layer_attn, visual_indices)[0]
                mean_indices = torch.topk(mean_scores, k=min(k, len(mean_scores))).indices
                mean_emb = image_emb[mean_indices]
                mean_score = compute_maxsim(query_emb[0], mean_emb)
                mean_retention = mean_score / full_score if full_score > 0 else 0.0
                results['Mean-Centrality'][layer_idx].append(float(mean_retention))

                # Max-Centrality (cached)
                max_scores = get_max_centrality_cached(layer_attn, visual_indices)[0]
                max_indices = torch.topk(max_scores, k=min(k, len(max_scores))).indices
                max_emb = image_emb[max_indices]
                max_score = compute_maxsim(query_emb[0], max_emb)
                max_retention = max_score / full_score if full_score > 0 else 0.0
                results['Max-Centrality'][layer_idx].append(float(max_retention))

                # EOS-Attention (cached)
                eos_scores = get_eos_attention_cached(layer_attn, visual_indices, eos_position)[0]
                eos_indices = torch.topk(eos_scores, k=min(k, len(eos_scores))).indices
                eos_emb = image_emb[eos_indices]
                eos_score = compute_maxsim(query_emb[0], eos_emb)
                eos_retention = eos_score / full_score if full_score > 0 else 0.0
                results['EOS-Attention'][layer_idx].append(float(eos_retention))

                # Spectral-Leverage (cached)
                spectral_scores = get_spectral_leverage_cached(layer_hidden, visual_indices)[0]
                spectral_indices = torch.topk(spectral_scores, k=min(k, len(spectral_scores))).indices
                spectral_emb = image_emb[spectral_indices]
                spectral_score = compute_maxsim(query_emb[0], spectral_emb)
                spectral_retention = spectral_score / full_score if full_score > 0 else 0.0
                results['Spectral-Leverage'][layer_idx].append(float(spectral_retention))

            # Random baseline (layer-independent, averaged over trials)
            random_retentions = []
            for _ in range(num_random_trials):
                random_indices = torch.randperm(N, device=device)[:k]
                random_emb = image_emb[random_indices]
                random_score = compute_maxsim(query_emb[0], random_emb)
                random_retention = random_score / full_score if full_score > 0 else 0.0
                random_retentions.append(float(random_retention))

            results['Random'].append(float(np.mean(random_retentions)))

        except Exception as e:
            print(f"\n❌ Error at sample {idx}: {e}")
            print(f"   Sample keys: {list(sample.keys()) if 'sample' in locals() else 'N/A'}")
            print(f"   Query type: {type(sample.get('query', None)) if 'sample' in locals() else 'N/A'}")
            print(f"   Image type: {type(sample.get('image', None)) if 'sample' in locals() else 'N/A'}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results (per layer)
    aggregated = {}

    # Per-layer metrics
    for metric in ['Mean-Centrality', 'Max-Centrality', 'EOS-Attention', 'Spectral-Leverage']:
        aggregated[metric] = {}
        for layer_idx in range(num_layers):
            values = results[metric][layer_idx]
            if len(values) > 0:
                aggregated[metric][layer_idx] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'count': len(values),
                }
            else:
                aggregated[metric][layer_idx] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'count': 0,
                }

    # Random baseline (layer-independent)
    random_values = results['Random']
    if len(random_values) > 0:
        aggregated['Random'] = {
            'mean': float(np.mean(random_values)),
            'std': float(np.std(random_values)),
            'count': len(random_values),
        }
    else:
        aggregated['Random'] = {
            'mean': 0.0,
            'std': 0.0,
            'count': 0,
        }

    return aggregated


# ============================================================================
# Global Aggregation (identical to v3)
# ============================================================================

def compute_global_average(
    per_dataset_results: Dict[str, Dict]
) -> Dict:
    """
    Compute macro-average across datasets for each layer.
    """
    # Get layer indices and methods from first dataset
    first_dataset = next(iter(per_dataset_results.values()))
    layer_indices = sorted([k for k in first_dataset['Mean-Centrality'].keys()])
    methods = ['Mean-Centrality', 'Max-Centrality', 'EOS-Attention', 'Spectral-Leverage']

    global_results = {}

    for method in methods:
        global_results[method] = {}

        for layer_idx in layer_indices:
            # Collect means from all datasets
            dataset_means = []

            for dataset_name, dataset_results in per_dataset_results.items():
                mean_value = dataset_results[method][layer_idx]['mean']
                dataset_means.append(mean_value)

            # Macro-average
            global_results[method][layer_idx] = {
                'mean': float(np.mean(dataset_means)),
                'std': float(np.std(dataset_means)),
            }

    # Random baseline (global average)
    random_means = []
    for dataset_name, dataset_results in per_dataset_results.items():
        random_means.append(dataset_results['Random']['mean'])

    global_results['Random'] = {
        'mean': float(np.mean(random_means)),
        'std': float(np.std(random_means)),
    }

    return global_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full Layer-wise Analysis v3 (OPTIMIZED)")
    parser.add_argument("--model_name", type=str, default="vidore/colpali-v1.2")
    parser.add_argument("--ratio", type=float, default=0.05, help="Retention ratio (default: 0.05 = 5%)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples per dataset (None = all)")
    parser.add_argument("--num_random_trials", type=int, default=5, help="Number of random trials for baseline")
    parser.add_argument("--output_dir", type=str, default="./experiments/outputs")

    args = parser.parse_args()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*70}")
    print(f"Section 4.1: Full Layer-wise Analysis v3 (OPTIMIZED)")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Ratio: {args.ratio} ({args.ratio*100:.1f}%)")
    print(f"Samples per dataset: {args.num_samples if args.num_samples else 'All'}")
    print()
    print("🚀 OPTIMIZATION: Single-pass inference")
    print("   - 1 forward pass per image (vs ~128 in original)")
    print("   - All layer outputs cached in memory")
    print("   - Pure tensor operations in layer loop")
    print()

    # Load model
    print("Loading model...")
    model = ColPali.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    ).eval()

    processor = ColPaliProcessor.from_pretrained(args.model_name)

    # CRITICAL: Enable eager attention (required for output_attentions=True)
    ensure_eager_attention(model)
    print(f"✅ Model loaded on {device}")

    # Get number of layers (compatible with PaliGemma)
    if hasattr(model.model, 'language_model'):
        num_layers = model.model.language_model.config.num_hidden_layers
    elif hasattr(model.model, 'config'):
        num_layers = model.model.config.num_hidden_layers
    else:
        raise ValueError("Cannot determine number of layers from model config")

    print(f"Model has {num_layers} layers (0 to {num_layers-1})")
    print()

    # Define ViDoRE datasets
    # Note: shiftproject_test excluded (only 100 valid queries, rest are None)
    datasets = [
        "vidore/arxivqa_test_subsampled",
        "vidore/docvqa_test_subsampled",
        "vidore/infovqa_test_subsampled",
        "vidore/tabfquad_test_subsampled",
        "vidore/tatdqa_test",
    ]

    print(f"Evaluating across {len(datasets)} datasets:")
    for ds_name in datasets:
        print(f"  • {ds_name}")
    print()

    # Evaluate each dataset
    per_dataset_results = {}

    for dataset_name in datasets:
        print(f"\n{'#'*70}")
        print(f"# DATASET: {dataset_name}")
        print(f"{'#'*70}")

        dataset_results = evaluate_dataset_layer_sweep_optimized(
            model, processor,
            dataset_name,
            num_layers,
            args.ratio,
            device,
            args.num_samples,
            args.num_random_trials
        )

        per_dataset_results[dataset_name] = dataset_results
        print(f"\n✅ Completed: {dataset_name}")

    # Compute global average
    print(f"\n{'='*70}")
    print("Computing Global Macro-Average...")
    print(f"{'='*70}")

    global_results = compute_global_average(per_dataset_results)

    # Print global summary
    print("\nGlobal Summary (Retention - Macro-Average):")
    print(f"{'─'*70}")
    print(f"{'Layer':<8} {'Mean-Cent':<12} {'Spectral':<12} {'Max-Cent':<12} {'EOS':<12} {'Random':<12}")
    print(f"{'─'*70}")

    for layer_idx in sorted(global_results['Mean-Centrality'].keys()):
        mean_val = global_results['Mean-Centrality'][layer_idx]['mean']
        spectral_val = global_results['Spectral-Leverage'][layer_idx]['mean']
        max_val = global_results['Max-Centrality'][layer_idx]['mean']
        eos_val = global_results['EOS-Attention'][layer_idx]['mean']
        random_val = global_results['Random']['mean']

        print(f"{layer_idx:<8} {mean_val:<12.4f} {spectral_val:<12.4f} {max_val:<12.4f} {eos_val:<12.4f} {random_val:<12.4f}")

    print(f"{'─'*70}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"layer_sweep_v3_quick_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save per-dataset results
    per_dataset_file = output_dir / "layer_sweep_v3_per_dataset.json"
    with open(per_dataset_file, 'w') as f:
        json.dump(per_dataset_results, f, indent=2)

    print(f"\n✅ Per-dataset results saved to: {per_dataset_file}")

    # Save global results
    global_file = output_dir / "layer_sweep_v3_global.json"
    with open(global_file, 'w') as f:
        json.dump(global_results, f, indent=2)

    print(f"✅ Global results saved to: {global_file}")

    # Save metadata
    metadata = {
        'model': args.model_name,
        'ratio': args.ratio,
        'num_layers': num_layers,
        'datasets': datasets,
        'num_samples_per_dataset': args.num_samples,
        'timestamp': timestamp,
        'optimized': True,
        'optimization_note': 'Single-pass inference with cached layer outputs'
    }

    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Metadata saved to: {metadata_file}")
    print()
    print(f"{'='*70}")
    print("Full ViDoRE Benchmark Analysis Complete! (OPTIMIZED)")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print()


if __name__ == "__main__":
    main()
