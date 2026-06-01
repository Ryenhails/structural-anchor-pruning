"""Tests for centrality computation."""

import torch
from sap.centrality import compute_visual_centrality, compute_visual_centrality_batch


class TestComputeVisualCentrality:
    def test_output_shape(self, layer_attn, visual_indices):
        scores = compute_visual_centrality(layer_attn, visual_indices, agg_mode="mean")
        assert scores.shape == (len(visual_indices),)

    def test_max_mode_shape(self, layer_attn, visual_indices):
        scores = compute_visual_centrality(layer_attn, visual_indices, agg_mode="max")
        assert scores.shape == (len(visual_indices),)

    def test_scores_are_positive(self, layer_attn, visual_indices):
        scores = compute_visual_centrality(layer_attn, visual_indices, agg_mode="mean")
        assert (scores >= 0).all()

    def test_mean_leq_max(self, layer_attn, visual_indices):
        mean_scores = compute_visual_centrality(layer_attn, visual_indices, agg_mode="mean")
        max_scores = compute_visual_centrality(layer_attn, visual_indices, agg_mode="max")
        # Mean over heads <= max over heads (element-wise)
        assert (mean_scores <= max_scores + 1e-6).all()

    def test_invalid_agg_mode(self, layer_attn, visual_indices):
        import pytest
        with pytest.raises(ValueError, match="agg_mode"):
            compute_visual_centrality(layer_attn, visual_indices, agg_mode="bad")

    def test_uniform_attention_gives_uniform_centrality(self, num_heads, seq_len, visual_indices):
        """If attention is uniform, all visual tokens get the same centrality."""
        uniform = torch.ones(1, num_heads, seq_len, seq_len) / seq_len
        scores = compute_visual_centrality(uniform, visual_indices, agg_mode="mean")
        # All scores should be approximately equal
        assert torch.allclose(scores, scores[0].expand_as(scores), atol=1e-5)


class TestComputeVisualCentralityBatch:
    def test_batch_shape(self, num_heads, seq_len, visual_indices):
        batch_size = 4
        raw = torch.rand(batch_size, num_heads, seq_len, seq_len)
        attn = raw / raw.sum(dim=-1, keepdim=True)
        scores = compute_visual_centrality_batch(attn, visual_indices, agg_mode="mean")
        assert scores.shape == (batch_size, len(visual_indices))

    def test_batch_matches_single(self, layer_attn, visual_indices):
        """Batch variant with batch=1 should match the single version."""
        single = compute_visual_centrality(layer_attn, visual_indices, agg_mode="mean")
        batch = compute_visual_centrality_batch(layer_attn, visual_indices, agg_mode="mean")
        assert torch.allclose(single, batch[0], atol=1e-5)
