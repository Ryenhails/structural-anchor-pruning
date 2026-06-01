"""Tests for the SR (Score Retention) diagnostic."""

import numpy as np
import torch

from sap.score_retention import (
    aggregate_calibration_sr,
    compute_per_layer_sr,
    compute_random_sr_baseline,
)


class TestComputePerLayerSr:
    def test_output_shape(self, multi_layer_attentions, visual_indices, doc_embs, query_embs):
        sr = compute_per_layer_sr(
            multi_layer_attentions, visual_indices, doc_embs, query_embs, ratio=0.5
        )
        assert sr.shape == (len(multi_layer_attentions),)

    def test_values_finite(self, multi_layer_attentions, visual_indices, doc_embs, query_embs):
        sr = compute_per_layer_sr(
            multi_layer_attentions, visual_indices, doc_embs, query_embs, ratio=0.3
        )
        assert np.all(np.isfinite(sr))

    def test_ratio_one_gives_unit_sr(
        self, multi_layer_attentions, visual_indices, doc_embs, query_embs,
    ):
        """At ratio=1.0 every token is kept, so SR must be exactly 1.0."""
        sr = compute_per_layer_sr(
            multi_layer_attentions, visual_indices, doc_embs, query_embs, ratio=1.0
        )
        assert np.allclose(sr, 1.0, atol=1e-5)

    def test_dtype_float32(self, multi_layer_attentions, visual_indices, doc_embs, query_embs):
        sr = compute_per_layer_sr(
            multi_layer_attentions, visual_indices, doc_embs, query_embs, ratio=0.5
        )
        assert sr.dtype == np.float32


class TestComputeRandomSrBaseline:
    def test_returns_scalar(self, doc_embs, query_embs):
        gen = torch.Generator().manual_seed(0)
        v = compute_random_sr_baseline(doc_embs, query_embs, ratio=0.3, num_seeds=3, generator=gen)
        assert isinstance(v, float)

    def test_ratio_one_gives_unit(self, doc_embs, query_embs):
        v = compute_random_sr_baseline(doc_embs, query_embs, ratio=1.0, num_seeds=2)
        assert abs(v - 1.0) < 1e-5


class TestAggregateCalibrationSr:
    def test_mean_and_std_shapes(self):
        per_pair = [np.array([0.5, 0.7, 0.9, 0.6]), np.array([0.55, 0.72, 0.88, 0.62])]
        agg = aggregate_calibration_sr(per_pair)
        assert agg["mean"].shape == (4,)
        assert agg["std"].shape == (4,)

    def test_mean_correctness(self):
        per_pair = [np.array([0.2, 0.4]), np.array([0.4, 0.8])]
        agg = aggregate_calibration_sr(per_pair)
        assert np.allclose(agg["mean"], [0.3, 0.6])

    def test_dtype(self):
        per_pair = [np.array([0.5, 0.7]), np.array([0.6, 0.8])]
        agg = aggregate_calibration_sr(per_pair)
        assert agg["mean"].dtype == np.float32


class TestEndToEndSrWindowFlow:
    """Composition test: SR curve -> window selection -> selected layers usable in SAP."""

    def test_full_flow(self, multi_layer_attentions, visual_indices, doc_embs, query_embs):
        from sap import compute_sap_scores, select_sr_window

        sr = compute_per_layer_sr(
            multi_layer_attentions, visual_indices, doc_embs, query_embs, ratio=0.5
        )
        window = select_sr_window(sr.tolist(), rho=0.2)

        scores = compute_sap_scores(
            multi_layer_attentions, visual_indices,
            target_layers=window.layer_indices, agg_mode="mean",
        )
        assert scores.shape == (len(visual_indices),)
        assert torch.all(torch.isfinite(scores))
