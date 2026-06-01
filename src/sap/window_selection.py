"""
SR-guided window selection (Algorithm 1, paper §3.3).

Given a per-layer Score Retention curve, locate the Structural Plateau by
walking back from the final layer through the alignment region (where SR
drops below the median), then take a window of relative width :math:`\\rho`
ending right before the alignment region begins.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class WindowSelection:
    """Result of :func:`select_sr_window`."""

    layer_indices: List[int]
    """Selected layer indices (0-indexed, inclusive of both ends)."""

    alpha: float
    """Window start as a fraction of total depth, in [0, 1)."""

    beta: float
    """Window end (exclusive) as a fraction of total depth, in (0, 1]."""

    num_layers: int
    """Total number of layers in the backbone."""

    median_sr: float
    """Median of the input per-layer SR curve."""

    alignment_start: int
    """First layer in the trailing alignment region (``num_layers`` if none)."""

    alignment_region: List[int] = field(default_factory=list)
    """Indices of the trailing alignment-region layers (``SR < median`` suffix)."""

    snapped_window: Optional[str] = None
    """Nearest 20%-wide canonical window label (e.g. ``"60-80"``)."""


_SNAP_WINDOWS = [
    (0.0, 0.2, "0-20"),
    (0.1, 0.3, "10-30"),
    (0.2, 0.4, "20-40"),
    (0.3, 0.5, "30-50"),
    (0.4, 0.6, "40-60"),
    (0.5, 0.7, "50-70"),
    (0.6, 0.8, "60-80"),
    (0.7, 0.9, "70-90"),
    (0.8, 1.0, "80-100"),
]


def select_sr_window(
    per_layer_sr: Sequence[float],
    rho: float = 0.2,
) -> WindowSelection:
    """
    SR-guided window selection.

    Steps:
        1. Compute the median :math:`m` of the per-layer SR curve.
        2. Find :math:`l^\\star`, the smallest index such that all layers in
           :math:`[l^\\star, L)` have ``SR < m`` (the trailing alignment region).
        3. Window width :math:`w = \\lceil \\rho L \\rceil`.
        4. The selected window is
           :math:`[\\max(0,\\; l^\\star - w),\\; l^\\star)`.

    Args:
        per_layer_sr: Per-layer Score Retention values, length ``L``.
        rho: Relative window width as a fraction of total depth. The paper
            uses ``rho=0.2`` for all reported results.

    Returns:
        A :class:`WindowSelection` capturing the selected layers, the
        :math:`(\\alpha, \\beta)` window in relative-depth coordinates, and
        diagnostics for downstream analysis.

    Raises:
        ValueError: If ``per_layer_sr`` is empty or ``rho`` is not in (0, 1].
    """
    osr = np.asarray(per_layer_sr, dtype=np.float64)
    L = int(osr.shape[0])
    if L == 0:
        raise ValueError("per_layer_sr must be non-empty.")
    if not 0.0 < rho <= 1.0:
        raise ValueError(f"rho must be in (0, 1], got {rho}")

    median = float(np.median(osr))

    # Longest suffix where SR < median.
    l_star = L
    for i in range(L - 1, -1, -1):
        if osr[i] < median:
            l_star = i
        else:
            break

    w = int(np.ceil(rho * L))
    if l_star == L:
        beta_layer = L
        alpha_layer = max(0, L - w)
    else:
        beta_layer = l_star
        alpha_layer = max(0, l_star - w)

    # Inclusive layer indices [alpha_layer, beta_layer - 1].
    selected = list(range(alpha_layer, beta_layer))

    alpha_frac = alpha_layer / L
    beta_frac = beta_layer / L

    # Snap to the nearest 20%-wide canonical window for table comparison.
    snapped_name: Optional[str] = None
    best_d = float("inf")
    for a, b, name in _SNAP_WINDOWS:
        d = abs(a - alpha_frac) + abs(b - beta_frac)
        if d < best_d:
            best_d = d
            snapped_name = name

    return WindowSelection(
        layer_indices=selected,
        alpha=float(alpha_frac),
        beta=float(beta_frac),
        num_layers=L,
        median_sr=median,
        alignment_start=int(l_star),
        alignment_region=list(range(int(l_star), L)) if l_star < L else [],
        snapped_window=snapped_name,
    )
