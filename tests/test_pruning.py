"""Tests for pruning helpers."""

import pytest
import torch
from sap.pruning import prune_embeddings, prune_embeddings_with_indices, prune_embeddings_batch


class TestPruneEmbeddings:
    def test_basic_pruning(self, doc_embs):
        scores = torch.rand(doc_embs.shape[0])
        pruned = prune_embeddings(doc_embs, scores, ratio=0.5)
        expected_k = max(1, round(doc_embs.shape[0] * 0.5))
        assert pruned.shape == (expected_k, doc_embs.shape[1])

    def test_ratio_1_keeps_all(self, doc_embs):
        scores = torch.rand(doc_embs.shape[0])
        pruned = prune_embeddings(doc_embs, scores, ratio=1.0)
        assert pruned.shape == doc_embs.shape

    def test_small_ratio_keeps_at_least_one(self):
        embs = torch.randn(3, 16)
        scores = torch.rand(3)
        pruned = prune_embeddings(embs, scores, ratio=0.01)
        assert pruned.shape[0] >= 1

    def test_invalid_ratio(self, doc_embs):
        scores = torch.rand(doc_embs.shape[0])
        with pytest.raises(ValueError, match="ratio"):
            prune_embeddings(doc_embs, scores, ratio=0.0)
        with pytest.raises(ValueError, match="ratio"):
            prune_embeddings(doc_embs, scores, ratio=1.5)

    def test_preserves_positional_order(self, doc_embs):
        """Selected indices should be sorted (positional order preserved)."""
        scores = torch.rand(doc_embs.shape[0])
        _, indices = prune_embeddings_with_indices(doc_embs, scores, ratio=0.5)
        assert (indices[1:] > indices[:-1]).all()

    def test_top_scores_selected(self):
        """The token with the highest score should always be selected."""
        embs = torch.randn(10, 16)
        scores = torch.arange(10, dtype=torch.float)  # 0..9, highest=9
        pruned, indices = prune_embeddings_with_indices(embs, scores, ratio=0.3)
        assert 9 in indices


class TestPruneEmbeddingsBatch:
    def test_batch(self):
        embs_list = [torch.randn(20, 16), torch.randn(30, 16)]
        scores_list = [torch.rand(20), torch.rand(30)]
        pruned = prune_embeddings_batch(embs_list, scores_list, ratio=0.5)
        assert len(pruned) == 2
        assert pruned[0].shape[0] == max(1, round(20 * 0.5))
        assert pruned[1].shape[0] == max(1, round(30 * 0.5))
