# Multi-Vector Retrieval Compression Benchmark

This repository contains the experimental code for evaluating various token pruning and compression methods for multi-vector document retrieval models.

## Environment Setup

### Requirements

- Python 3.10+
- CUDA-compatible GPU (recommended: 24GB+ VRAM)
- PyTorch 2.0+

### Installation

```bash
# Install core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install required packages
pip install transformers datasets scikit-learn pillow tqdm numpy

# Install ColPali engine
pip install colpali-engine
```

### Additional Dependencies

For Jina models support:
```bash
pip install -U transformers  # Ensure latest version
```

---

## Experimental Scripts

### 1. Fixed Ratio Benchmark (Comprehensive)

Evaluates multiple pruning/compression methods at fixed compression ratios.

**Methods evaluated:**
- Random pruning (baseline)
- SAP-Mean (Self-Attention Pruning with mean aggregation)
- SAP-Max (Self-Attention Pruning with max aggregation)
- Cluster-Merge (hierarchical clustering)
- Adaptive-EOS (EOS attention-based adaptive pruning)

**Usage:**
```bash
python run_fixed_ratio_benchmark.py \
    --model_name vidore/colpali-v1.2 \
    --layers 8,9,10,11 \
    --calibration_size 128 \
    --output_dir ./outputs
```

**Key arguments:**
- `--model_name`: HuggingFace model path (supports ColPali, ColQwen2, Jina models)
- `--layers`: Comma-separated layer indices for SAP methods (default: middle 4 layers)
- `--calibration_size`: Number of samples for Adaptive-EOS calibration
- `--skip_v1` / `--skip_v2`: Skip Vidore v1 or v2 datasets

**Output:**
- `benchmark_summary.json`: Complete results in JSON format
- `benchmark_summary.csv`: Results table for analysis
- `benchmark_summary.txt`: Human-readable summary
- `results_<dataset>.json`: Per-dataset detailed results

---

### 2. Sweep Ratio Benchmark

Evaluates all methods across a wide range of compression ratios [0.1, 0.2, ..., 0.9].

**Usage:**
```bash
python run_sweep_ratio_benchmark.py \
    --model_name vidore/colpali-v1.2 \
    --layers auto \
    --output_dir ./outputs
```

**Key arguments:**
- `--layers auto`: Automatically selects layers based on model type
  - PaliGemma: middle 4 layers (8-11)
  - Qwen2: middle 6 layers (11-16)
  - Jina: middle 8 layers (20-27)

**Note:** This script only evaluates Vidore v2 datasets and reports NDCG@5 metric.

---

### 3. LightColPali Comparison

Evaluates SAP methods with specific ratios used in LightColPali paper.

**Usage:**
```bash
python run_fixed_ratio_benchmark_compared_train.py \
    --model_name vidore/colpali-v1.2 \
    --layers 8,9,10,11 \
    --output_dir ./outputs
```

**Compression ratios:** [0.25, 0.1111, 0.04, 0.02041]

**Datasets:** InfoVQA, DocVQA, ArxivQA, TabFQuAD, TATDQA, ShiftProject

---

### 4. Layer-wise Analysis

Analyzes the effectiveness of different pruning methods across all model layers.

**Usage:**
```bash
python run_layer_comparison.py \
    --model_name vidore/colpali-v1.2 \
    --ratio 0.05 \
    --num_samples 500 \
    --output_dir ./outputs
```

**Key arguments:**
- `--ratio`: Token retention ratio (e.g., 0.05 = keep 5% of tokens)
- `--num_samples`: Number of samples per dataset (None = use all)

**Methods compared:**
- Mean-Centrality
- Max-Centrality
- EOS-Attention
- Spectral-Leverage
- Random (baseline)

**Optimization:** Uses single-pass inference with cached layer outputs (~128x speedup).

---

## Usage Tips

### 1. Model Selection

**Supported models:**
- **PaliGemma-based:** `vidore/colpali-v1.2`, `vidore/colpali-v1.3`
- **Qwen2-based:** `vidore/colqwen2-v0.1`, `vidore/colqwen2-v1.0`
- **Jina-based:** `jinaai/jina-embeddings-v4`

The script automatically detects model type and adjusts layer indices accordingly.

### 2. Memory Optimization

For large-scale experiments:
```bash
# Reduce batch size for images/queries (hardcoded in scripts)
# Current defaults: image_batch=4, query_batch=8

# Use gradient checkpointing (if OOM occurs)
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### 3. Layer Selection Guidelines

**For SAP methods**, choose middle layers:
- **PaliGemma (18 layers):** layers 8-11 (default)
- **Qwen2 (28 layers):** layers 11-16
- **Jina (48 layers):** layers 20-27

Use `--layers auto` for automatic selection.

### 4. Dataset Information

**Vidore v1:** Uses (query, image) pairs directly from HuggingFace datasets.

**Vidore v2:** Uses separate configs for queries, corpus, and qrels. Automatically handled by the scripts.

**Excluded dataset:** `shiftproject` has limited valid queries (~100 out of 1000), may be skipped in some experiments.

### 5. Output Interpretation

**Metrics reported:**
- **Score Retention:** Ratio of compressed MaxSim score to full MaxSim score
- **NDCG@5:** Normalized Discounted Cumulative Gain at rank 5
- **NDCG Retention:** Ratio of compressed NDCG@5 to full NDCG@5

**Higher values = better compression quality**

---

## Expected Runtime

Approximate runtime on a single A100 GPU:

| Script | Model | Dataset | Samples | Time |
|--------|-------|---------|---------|------|
| `run_fixed_ratio_benchmark.py` | ColPali | Vidore v1 (10 datasets) | All | ~6-8 hours |
| `run_sweep_ratio_benchmark.py` | ColPali | Vidore v2 (4 datasets) | All | ~4-6 hours |
| `run_layer_comparison.py` | ColPali | Vidore v1 (5 datasets) | 500/dataset | ~2-3 hours |

**Note:** Jina models require more memory and may run slower due to larger architecture.

---

## Troubleshooting

### 1. CUDA Out of Memory

```bash
# Reduce number of samples
python run_layer_comparison.py --num_samples 100

# Or use CPU (slower)
export CUDA_VISIBLE_DEVICES=""
```

### 2. Flash Attention Errors

The scripts automatically force eager attention mode. If you still encounter errors, ensure:
```bash
pip install transformers>=4.40.0
```

### 3. Dataset Loading Issues

If HuggingFace datasets fail to load:
```bash
# Set cache directory
export HF_HOME=/path/to/cache

# Or download datasets manually
huggingface-cli download vidore/arxivqa_test_subsampled
```

---

## Citation

If you use this code, please cite the original ColPali paper and related work:

```bibtex
@article{colpali2024,
  title={ColPali: Efficient Document Retrieval with Vision Language Models},
  author={...},
  journal={arXiv preprint arXiv:...},
  year={2024}
}
```

---

## License

This code is provided for research purposes. Please refer to the respective model licenses:
- ColPali models: MIT License
- Transformers library: Apache 2.0
- Datasets: See individual dataset licenses on HuggingFace
