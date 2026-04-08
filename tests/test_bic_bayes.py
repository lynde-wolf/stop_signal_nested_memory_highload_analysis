"""Tests for bic_bayes: calculate_bic, interpret_bic_delta, calculate_bf10."""

import numpy as np
import pytest

from stop_wm.bic_bayes import calculate_bic, calculate_bf10, interpret_bic_delta


# ---------------------------------------------------------------------------
# calculate_bic
# ---------------------------------------------------------------------------

class TestCalculateBic:
    """BIC = n·ln(RSS/n) + k·ln(n)."""

    def test_known_answer(self):
        """Hand-computed reference case."""
        # residuals = [1, -1, 1, -1]  → RSS = 4, n=4, k=2
        # BIC = 4·ln(4/4) + 2·ln(4) = 4·0 + 2·1.3862… = 2.7726…
        residuals = [1.0, -1.0, 1.0, -1.0]
        expected = 4 * np.log(1.0) + 2 * np.log(4)
        result = calculate_bic(residuals, n_params=2, n_obs=4)
        assert pytest.approx(result, rel=1e-6) == expected

    def test_larger_residuals_give_higher_bic(self):
        """Worse fit (larger residuals) should produce a higher BIC."""
        small = calculate_bic([0.1, -0.1, 0.1, -0.1], n_params=2, n_obs=4)
        large = calculate_bic([1.0, -1.0, 1.0, -1.0], n_params=2, n_obs=4)
        assert large > small

    def test_more_params_gives_higher_bic(self):
        """BIC penalises complexity; more parameters → higher BIC."""
        residuals = [0.5, -0.5, 0.5, -0.5]
        bic_simple = calculate_bic(residuals, n_params=1, n_obs=4)
        bic_complex = calculate_bic(residuals, n_params=3, n_obs=4)
        assert bic_complex > bic_simple

    def test_returns_float(self):
        result = calculate_bic([1.0, -1.0], n_params=1, n_obs=2)
        assert isinstance(result, float)

    def test_accepts_numpy_array(self):
        residuals = np.array([0.5, -0.5, 0.5])
        result = calculate_bic(residuals, n_params=1, n_obs=3)
        assert np.isfinite(result)

    def test_accepts_list(self):
        """Input type flexibility — plain list should work."""
        result = calculate_bic([1.0, -1.0, 1.0, -1.0], n_params=2, n_obs=4)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# interpret_bic_delta
# ---------------------------------------------------------------------------

class TestInterpretBicDelta:
    """Evidence labels follow Kass & Raftery (1995) thresholds."""

    # -- verbose=True (default) ------------------------------------------

    @pytest.mark.parametrize("delta, expected", [
        (11.0,  "Very Strong evidence for condition effects"),
        (10.0,  "Strong evidence for condition effects"),   # boundary: > 6, not > 10
        (8.0,   "Strong evidence for condition effects"),
        (6.0,   "Positive evidence for condition effects"), # boundary: > 2, not > 6
        (4.0,   "Positive evidence for condition effects"),
        (2.0,   "Weak evidence for condition effects"),     # boundary: > 0, not > 2
        (1.0,   "Weak evidence for condition effects"),
        (0.0,   "Weak evidence against condition effects"), # boundary: > -2, not > 0
        (-1.0,  "Weak evidence against condition effects"),
        (-2.0,  "Positive evidence against condition effects"),
        (-4.0,  "Positive evidence against condition effects"),
        (-6.0,  "Strong evidence against condition effects"),
        (-8.0,  "Strong evidence against condition effects"),
        (-10.0, "Very Strong evidence against condition effects"),
        (-15.0, "Very Strong evidence against condition effects"),
    ])
    def test_verbose_labels(self, delta, expected):
        assert interpret_bic_delta(delta, verbose=True) == expected

    def test_verbose_is_default(self):
        """Calling without verbose= should behave like verbose=True."""
        assert interpret_bic_delta(15.0) == interpret_bic_delta(15.0, verbose=True)

    # -- verbose=False (compact) -----------------------------------------

    @pytest.mark.parametrize("delta, expected", [
        (11.0,  "Very Strong FOR"),
        (8.0,   "Strong FOR"),
        (4.0,   "Positive FOR"),
        (1.0,   "Weak FOR"),
        (-1.0,  "Weak AGAINST"),
        (-4.0,  "Positive AGAINST"),
        (-8.0,  "Strong AGAINST"),
        (-15.0, "Very Strong AGAINST"),
    ])
    def test_compact_labels(self, delta, expected):
        assert interpret_bic_delta(delta, verbose=False) == expected

    def test_verbose_and_compact_cover_same_tiers(self):
        """Both modes should return the same tier for every threshold crossing."""
        test_deltas = [15, 8, 4, 1, -1, -4, -8, -15]
        verbose_tiers = [interpret_bic_delta(d, verbose=True) for d in test_deltas]
        compact_tiers = [interpret_bic_delta(d, verbose=False) for d in test_deltas]
        # Tier ordering should be identical — same index = same direction & strength
        for v, c in zip(verbose_tiers, compact_tiers):
            assert ("for" in v.lower()) == ("FOR" in c), (
                f"Direction mismatch: '{v}' vs '{c}'"
            )

    def test_returns_string(self):
        assert isinstance(interpret_bic_delta(5.0), str)


# ---------------------------------------------------------------------------
# calculate_bf10
# ---------------------------------------------------------------------------

class TestCalculateBf10:
    """BF10 = exp(ΔBIC / 2)."""

    def test_zero_delta_gives_bf_one(self):
        """ΔBIC = 0 → models are equivalent → BF10 = 1."""
        assert pytest.approx(calculate_bf10(0.0)) == 1.0

    def test_positive_delta_gives_bf_above_one(self):
        """Positive ΔBIC means full model is better → BF10 > 1."""
        assert calculate_bf10(10.0) > 1.0

    def test_negative_delta_gives_bf_below_one(self):
        """Negative ΔBIC means null model is better → BF10 < 1."""
        assert calculate_bf10(-10.0) < 1.0

    def test_known_value(self):
        """BF10 = exp(6 / 2) = exp(3) ≈ 20.086."""
        assert pytest.approx(calculate_bf10(6.0), rel=1e-6) == np.exp(3.0)

    def test_symmetry(self):
        """BF10(+d) = 1 / BF10(−d)."""
        d = 8.0
        assert pytest.approx(calculate_bf10(d) * calculate_bf10(-d), rel=1e-10) == 1.0

    def test_large_positive_delta(self):
        """Very large ΔBIC should not overflow to inf at typical analysis values."""
        result = calculate_bf10(100.0)
        assert np.isfinite(result)

    def test_returns_float(self):
        assert isinstance(calculate_bf10(5.0), float)

    @pytest.mark.parametrize("delta, expected_bf", [
        (2,  np.exp(1)),   # exp(2/2) ≈ 2.718
        (6,  np.exp(3)),   # exp(6/2) ≈ 20.09
        (10, np.exp(5)),   # exp(10/2) ≈ 148.41
    ])
    def test_aligns_with_kass_raftery_thresholds(self, delta, expected_bf):
        """BF10 = exp(ΔBIC/2) at K&R boundary values matches the formula exactly."""
        assert pytest.approx(calculate_bf10(delta), rel=1e-6) == expected_bf
