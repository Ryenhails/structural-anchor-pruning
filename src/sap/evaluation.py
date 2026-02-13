"""
Evaluation metrics: MaxSim scoring (Eq. 1) and Oracle Score Retention (Eq. 8).
"""

from typing import List, Optional

import torch
import numpy as np


# ---------------------------------------------------------------------------
# MaxSim (Eq. 1)
# ---------------------------------------------------------------------------

def maxsim_score(
    query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the MaxSim late-interaction score between a query and a document.

    .. math::
        S(Q, D) = \\sum_i \\max_j \\; q_i \\cdot v_j

    Args:
        query_embs: Query token embeddings ``[num_query_tokens, dim]``.
        doc_embs: Document (visual) token embeddings ``[num_doc_tokens, dim]``.

    Returns:
        Scalar MaxSim score (0-D tensor).
    """
    # [num_query, num_doc]
    sim = torch.matmul(query_embs, doc_embs.t())
    return sim.max(dim=-1).values.sum()


def maxsim_score_batched(
    query_embs: torch.Tensor,
    doc_embs: torch.Tensor,
) -> torch.Tensor:
    """
    Batched MaxSim: score each (query, document) pair in a batch.

    Args:
        query_embs: ``[batch, num_query_tokens, dim]``.
        doc_embs: ``[batch, num_doc_tokens, dim]``.

    Returns:
        MaxSim scores ``[batch]``.
    """
    # [batch, num_query, num_doc]
    interaction = torch.einsum("bqd,bpd->bqp", query_embs, doc_embs)
    return interaction.max(dim=-1).values.sum(dim=-1)


def maxsim_score_matrix(
    query_embs_list: List[torch.Tensor],
    doc_embs_list: List[torch.Tensor],
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute the full query-document MaxSim score matrix.

    Args:
        query_embs_list: Per-query embeddings, each ``[num_q_tokens_i, dim]``.
        doc_embs_list: Per-document embeddings, each ``[num_d_tokens_j, dim]``.
        device: Device for computation (defaults to CPU).

    Returns:
        Score matrix of shape ``[num_queries, num_docs]`` as a numpy array.
    """
    num_queries = len(query_embs_list)
    num_docs = len(doc_embs_list)
    if num_queries == 0 or num_docs == 0:
        return np.zeros((num_queries, num_docs), dtype=np.float32)

    if device is None:
        device = "cpu"

    # Pad queries into a single batch
    max_q_len = max(q.shape[0] for q in query_embs_list)
    dim = query_embs_list[0].shape[-1]
    dtype = query_embs_list[0].dtype

    Q_batch = torch.zeros((num_queries, max_q_len, dim), dtype=dtype, device=device)
    Q_mask = torch.zeros((num_queries, max_q_len), dtype=torch.bool, device=device)

    for i, q in enumerate(query_embs_list):
        length = q.shape[0]
        Q_batch[i, :length] = q.to(device)
        Q_mask[i, :length] = True

    score_matrix = np.zeros((num_queries, num_docs), dtype=np.float32)

    with torch.inference_mode():
        for j, doc in enumerate(doc_embs_list):
            d = doc.to(device)
            if d.dtype != Q_batch.dtype:
                d = d.to(Q_batch.dtype)
            sim = torch.matmul(Q_batch, d.t())  # [num_queries, max_q_len, num_d_tokens]
            max_sim = sim.max(dim=-1).values * Q_mask  # [num_queries, max_q_len]
            score_matrix[:, j] = max_sim.sum(dim=-1).float().cpu().numpy()

    return score_matrix


# ---------------------------------------------------------------------------
# Oracle Score Retention — OSR (Eq. 8)
# ---------------------------------------------------------------------------

def compute_osr(
    query_embs: torch.Tensor,
    full_doc_embs: torch.Tensor,
    pruned_doc_embs: torch.Tensor,
) -> float:
    """
    Compute Oracle Score Retention for a single query-document pair.

    .. math::
        R = \\frac{\\text{MaxSim}(Q, D_{\\text{pruned}})}{\\text{MaxSim}(Q, D_{\\text{full}})}

    Args:
        query_embs: ``[num_query_tokens, dim]``.
        full_doc_embs: ``[num_full_tokens, dim]`` — unpruned document.
        pruned_doc_embs: ``[num_pruned_tokens, dim]`` — pruned document.

    Returns:
        Score retention ratio (float in [0, 1]).
    """
    full_score = maxsim_score(query_embs, full_doc_embs).item()
    if full_score == 0.0:
        return 1.0
    pruned_score = maxsim_score(query_embs, pruned_doc_embs).item()
    return pruned_score / full_score


def compute_osr_batch(
    query_embs_list: List[torch.Tensor],
    full_doc_embs_list: List[torch.Tensor],
    pruned_doc_embs_list: List[torch.Tensor],
) -> List[float]:
    """
    Compute OSR for multiple query-document pairs.

    Each list entry corresponds to one (query, document) pair.

    Returns:
        List of OSR values.
    """
    return [
        compute_osr(q, f, p)
        for q, f, p in zip(query_embs_list, full_doc_embs_list, pruned_doc_embs_list)
    ]
