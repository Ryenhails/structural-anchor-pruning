"""Tests for SR-guided window selection (Algorithm 1)."""

import numpy as np
import pytest

from sap.window_selection import WindowSelection, select_sr_window


class TestSelectSrWindow:
    def test_returns_window_selection(self):
        sr = [0.5, 0.6, 0.7, 0.8, 0.7, 0.6, 0.5, 0.4]
        out = select_sr_window(sr)
        assert isinstance(out, WindowSelection)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            select_sr_window([])

    def test_bad_rho_raises(self):
        with pytest.raises(ValueError, match="rho"):
            select_sr_window([0.5, 0.6, 0.7], rho=0.0)
        with pytest.raises(ValueError, match="rho"):
            select_sr_window([0.5, 0.6, 0.7], rho=1.5)

    def test_18_layer_paper_shape_colpali(self):
        """Synthetic SR shaped like ColPali's paper curve: plateau in 11-14, drop at end."""
        # SR rises through early layers, plateaus around 11-14, then drops.
        sr = np.array([
            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.88,
            0.90, 0.91, 0.92, 0.93, 0.93, 0.92, 0.91,        # plateau 8-14
            0.75, 0.60, 0.45,                                 # alignment region 15-17
        ])
        assert len(sr) == 18
        out = select_sr_window(sr, rho=0.2)
        # Drop-alignment rule should detect the alignment region near the end
        # and place a 20%-wide window (ceil(0.2 * 18) = 4 layers) ending just before it.
        assert out.num_layers == 18
        assert out.alignment_start >= 15
        assert len(out.layer_indices) == 4
        # Window should land in the upper-mid range (50-80%).
        assert 0.50 <= out.alpha <= 0.65
        assert 0.70 <= out.beta <= 0.85

    def test_28_layer_paper_shape_colqwen2(self):
        """Synthetic SR for a 28-layer backbone with plateau around 18-23."""
        sr = np.array(
            [0.50, 0.55, 0.58, 0.60, 0.63, 0.66, 0.70, 0.74,
             0.78, 0.80, 0.82, 0.85, 0.87, 0.88, 0.89, 0.90,
             0.91, 0.92, 0.92, 0.92, 0.91, 0.90, 0.88,
             0.70, 0.55, 0.42, 0.35, 0.30],
        )
        assert len(sr) == 28
        out = select_sr_window(sr, rho=0.2)
        assert out.num_layers == 28
        assert out.alignment_start <= 25  # SR drops before final 3 layers
        assert len(out.layer_indices) == 6  # ceil(0.2 * 28) = 6

    def test_36_layer_paper_shape_jina(self):
        """Synthetic SR for 36 layers with plateau pushed to 70-90% range."""
        sr = np.linspace(0.45, 0.93, 30).tolist() + [0.85, 0.65, 0.45, 0.35, 0.30, 0.25]
        sr = np.array(sr)
        assert len(sr) == 36
        out = select_sr_window(sr, rho=0.2)
        assert out.num_layers == 36
        assert len(out.layer_indices) == 8  # ceil(0.2 * 36) = 8
        # Plateau-end is around layer 29, so window should end near 30.
        assert out.beta >= 0.70

    def test_snapped_window_is_canonical(self):
        sr = np.array([0.5] * 10 + [0.9] * 6 + [0.3] * 2)  # 18 layers
        out = select_sr_window(sr)
        # 9 canonical labels live in the module.
        assert out.snapped_window in {
            "0-20", "10-30", "20-40", "30-50", "40-60",
            "50-70", "60-80", "70-90", "80-100",
        }

    def test_no_alignment_region_falls_back_to_tail(self):
        """If SR never drops below median at the end, window is placed at the tail."""
        sr = list(range(18))  # monotonically increasing
        out = select_sr_window(sr, rho=0.2)
        # No suffix below median -> alignment_start == L, window ends at L.
        assert out.alignment_start == 18
        assert out.beta == 1.0
        assert out.layer_indices[-1] == 17

    def test_window_width_scales_with_rho(self):
        sr = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3])
        narrow = select_sr_window(sr, rho=0.1)
        wide = select_sr_window(sr, rho=0.4)
        assert len(narrow.layer_indices) <= len(wide.layer_indices)

    def test_alpha_beta_monotonic(self):
        sr = np.random.RandomState(0).rand(24).tolist()
        out = select_sr_window(sr)
        assert 0.0 <= out.alpha < out.beta <= 1.0

    def test_dataclass_serializable(self):
        """WindowSelection fields should be plain JSON-friendly types."""
        out = select_sr_window([0.5, 0.7, 0.9, 0.7, 0.3])
        assert isinstance(out.layer_indices, list)
        assert all(isinstance(i, int) for i in out.layer_indices)
        assert isinstance(out.alpha, float)
        assert isinstance(out.beta, float)
        assert isinstance(out.median_sr, float)
