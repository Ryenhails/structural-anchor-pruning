"""Synthetic tensor fixtures for testing SAP without GPU or model weights."""

import pytest
import torch


@pytest.fixture
def num_heads():
    return 8


@pytest.fixture
def seq_len():
    return 64


@pytest.fixture
def num_visual():
    return 32


@pytest.fixture
def dim():
    return 128


@pytest.fixture
def visual_indices(num_visual):
    """Visual tokens occupy positions 10..41 (contiguous block)."""
    return torch.arange(10, 10 + num_visual)


@pytest.fixture
def layer_attn(num_heads, seq_len):
    """Random attention weights [1, H, S, S] that sum to 1 along last dim."""
    raw = torch.rand(1, num_heads, seq_len, seq_len)
    return raw / raw.sum(dim=-1, keepdim=True)


@pytest.fixture
def multi_layer_attentions(num_heads, seq_len):
    """18 layers of random attention weights (PaliGemma-like depth)."""
    layers = []
    for _ in range(18):
        raw = torch.rand(1, num_heads, seq_len, seq_len)
        layers.append(raw / raw.sum(dim=-1, keepdim=True))
    return tuple(layers)


@pytest.fixture
def query_embs(dim):
    """Random query embeddings [5, dim]."""
    return torch.randn(5, dim)


@pytest.fixture
def doc_embs(num_visual, dim):
    """Random document embeddings [num_visual, dim]."""
    return torch.randn(num_visual, dim)
