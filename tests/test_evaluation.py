"""Tests for MaxSim and OSR evaluation metrics."""

import numpy as np
import torch
from sap.evaluation import (
    maxsim_score,
    maxsim_score_batched,
    maxsim_score_matrix,
    compute_osr,
    compute_osr_batch,
)


class TestMaxSimScore:
    def test_identical_embeddings(self):
        """MaxSim of identical query and doc should be sum of squared norms."""
        embs = torch.randn(5, 16)
        score = maxsim_score(embs, embs)
        expected = (embs * embs).sum(dim=-1).sum()
        # Max similarity of q_i against D when D contains q_i is ||q_i||^2
        assert torch.allclose(score, expected, atol=1e-4)

    def test_orthogonal_gives_zero(self):
        """Orthogonal query and doc should give near-zero score."""
        q = torch.zeros(1, 4)
        q[0, 0] = 1.0
        d = torch.zeros(1, 4)
        d[0, 1] = 1.0
        score = maxsim_score(q, d)
        assert abs(score.item()) < 1e-6

    def test_score_is_scalar(self, query_embs, doc_embs):
        score = maxsim_score(query_embs, doc_embs)
        assert score.dim() == 0


class TestMaxSimBatched:
    def test_matches_single(self, query_embs, doc_embs):
        single = maxsim_score(query_embs, doc_embs)
        batched = maxsim_score_batched(
            query_embs.unsqueeze(0), doc_embs.unsqueeze(0)
        )
        assert torch.allclose(single, batched[0], atol=1e-4)


class TestMaxSimMatrix:
    def test_shape(self):
        queries = [torch.randn(3, 16), torch.randn(5, 16)]
        docs = [torch.randn(10, 16), torch.randn(8, 16), torch.randn(12, 16)]
        matrix = maxsim_score_matrix(queries, docs)
        assert matrix.shape == (2, 3)
        assert matrix.dtype == np.float32

    def test_empty(self):
        matrix = maxsim_score_matrix([], [torch.randn(5, 16)])
        assert matrix.shape == (0, 1)


class TestOSR:
    def test_full_retention(self, query_embs, doc_embs):
        """OSR with unpruned doc should be 1.0."""
        osr = compute_osr(query_embs, doc_embs, doc_embs)
        assert abs(osr - 1.0) < 1e-5

    def test_partial_retention(self, query_embs, doc_embs):
        """OSR with subset of doc should be <= 1.0."""
        pruned = doc_embs[:doc_embs.shape[0] // 2]
        osr = compute_osr(query_embs, doc_embs, pruned)
        assert 0.0 <= osr <= 1.0 + 1e-5

    def test_batch(self, query_embs, doc_embs):
        pruned = doc_embs[:doc_embs.shape[0] // 2]
        results = compute_osr_batch(
            [query_embs, query_embs],
            [doc_embs, doc_embs],
            [pruned, doc_embs],
        )
        assert len(results) == 2
        assert abs(results[1] - 1.0) < 1e-5  # full doc -> OSR=1
