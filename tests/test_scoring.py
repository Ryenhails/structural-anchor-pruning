"""Tests for SAP scoring pipeline."""

import pytest
import torch
from sap.scoring import compute_sap_scores, get_default_layers, get_default_layers_by_depth


class TestGetDefaultLayers:
    def test_colpali(self):
        assert get_default_layers("colpali") == [8, 9, 10, 11]

    def test_colqwen2(self):
        assert get_default_layers("colqwen2") == [11, 12, 13, 14, 15, 16]

    def test_jina(self):
        assert get_default_layers("jina") == [14, 15, 16, 17, 18, 19, 20, 21]

    def test_hf_model_name(self):
        layers = get_default_layers("vidore/colpali-v1.2")
        assert layers == [8, 9, 10, 11]

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot infer"):
            get_default_layers("unknown-model-xyz")


class TestGetDefaultLayersByDepth:
    def test_18_layers(self):
        layers = get_default_layers_by_depth(18, 0.40, 0.60)
        assert layers == [7, 8, 9, 10]

    def test_28_layers(self):
        layers = get_default_layers_by_depth(28, 0.40, 0.60)
        assert layers == [11, 12, 13, 14, 15, 16]


class TestComputeSapScores:
    def test_output_shape(self, multi_layer_attentions, visual_indices):
        scores = compute_sap_scores(
            multi_layer_attentions, visual_indices,
            target_layers=[8, 9, 10, 11], agg_mode="mean"
        )
        assert scores.shape == (len(visual_indices),)

    def test_float32_output(self, multi_layer_attentions, visual_indices):
        scores = compute_sap_scores(
            multi_layer_attentions, visual_indices,
            target_layers=[8, 9, 10, 11]
        )
        assert scores.dtype == torch.float32

    def test_model_name_auto(self, multi_layer_attentions, visual_indices):
        scores = compute_sap_scores(
            multi_layer_attentions, visual_indices,
            model_name="colpali"
        )
        assert scores.shape == (len(visual_indices),)

    def test_no_layers_no_model_raises(self, multi_layer_attentions, visual_indices):
        with pytest.raises(ValueError):
            compute_sap_scores(multi_layer_attentions, visual_indices)

    def test_single_layer(self, multi_layer_attentions, visual_indices):
        """Single-layer SAP should equal that layer's centrality."""
        from sap.centrality import compute_visual_centrality
        scores = compute_sap_scores(
            multi_layer_attentions, visual_indices,
            target_layers=[10], agg_mode="mean"
        )
        expected = compute_visual_centrality(
            multi_layer_attentions[10], visual_indices, agg_mode="mean"
        ).float()
        assert torch.allclose(scores, expected, atol=1e-5)
