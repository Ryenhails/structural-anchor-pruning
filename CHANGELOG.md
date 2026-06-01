# Changelog

## v2 (Paper revision; arXiv v2)

Title changed from *Look in the Middle: Structural Anchor Pruning for Scalable Visual RAG Indexing* to *Structural Anchor Pruning: Training-Free Multi-Vector Compression for Visual Document Retrieval*.

### Methodology

- **SR-guided window selection** replaces the previous fixed 40--60% middle window. The structural window is now automatically located per backbone using an unlabeled calibration set and a single relative-width hyperparameter $\rho$.
- **Score Retention (SR)** repositioned from a proxy for NDCG to a method-design diagnostic that drives window selection.
- **Calibration data** now drawn from the ColPali training corpus, disjoint from all ViDoRe evaluation splits.
- **Head aggregation** consolidated to the mean variant only (SAP-Max removed from main results).

### Default Layer Windows

| Model | v1 (40--60%) | v2 (SR-guided) |
|-------|-------------|-----------------|
| ColPali (18L) | `[8, 9, 10, 11]` | `[11, 12, 13, 14]` (60--80%) |
| ColQwen2 (28L) | `[11, 12, 13, 14, 15, 16]` | `[18, 19, 20, 21, 22, 23]` (60--80%) |
| Jina v4 (36L) | `[14, 15, 16, 17, 18, 19, 20, 21]` | `[26, 27, 28, 29, 30, 31, 32, 33]` (70--90%) |

`get_default_layers_by_depth` defaults changed from `start_pct=0.40, end_pct=0.60` to `start_pct=0.60, end_pct=0.80` to reflect the SR-guided window position. Unknown backbones should run the SR-guided procedure to determine their own window.

### New Experiments

- 9-window brute-force window ablation across all three backbones and both ViDoRe v1 and v2.
- Calibration-size robustness ($N \in \{50, 100, 200, 500, 1000\}$, 5 random seeds per setting).
- Window-width robustness ($k$ swept over $\{1, \dots, L_{\text{total}}\}$).
- End-to-end storage and retrieval-latency benchmark on the ViDoRe v2 union (3,006 documents, 1,152 queries): at $\gamma=0.10$, a $10\times$ index-size reduction and a $7.9\times$ MaxSim retrieval speedup; at $\gamma=0.05$, $20\times$ and $13\times$ respectively.
- Cross-cutoff stability analysis: NDCG@$\{1, 5, 10, 50, 100\}$ retention on both benchmarks.

### Backbones

- Jina v4 (Qwen2.5-VL, 36 layers) added as a third backbone, alongside ColPali (18 layers) and ColQwen2 (28 layers).

### Library

- New `sap.score_retention` module: `compute_per_layer_sr`, `compute_random_sr_baseline`, `aggregate_calibration_sr`.
- New `sap.window_selection` module: `select_sr_window` and a `WindowSelection` result dataclass. Algorithm 1 from the paper as a pure-numpy entry point usable on unknown backbones.
- Tests for both new modules; existing `test_scoring.py` updated to the v2 default-window expectations.
- Overview figure (`assets/sap_overview.png`) refreshed to the v2 method diagram.


## v1 (Initial release)

Initial release accompanying the original *Look in the Middle* preprint. Centrality-based pruning with a fixed 40--60% middle-layer window.
