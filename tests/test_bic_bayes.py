"""Tests for bic_bayes: calculate_bic, interpret_bic_delta, calculate_bf10, calculate_bf01."""

import numpy as np
import pytest

import pingouin as pg

from stop_wm.bic_bayes import (
    bf10_paired_difference_bic,
    bf10_paired_jzs,
    calculate_bic,
    calculate_bf01,
    calculate_bf10,
    interpret_bic_delta,
)


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

    def test_zero_residuals_gives_negative_inf(self):
        """Perfect fit (all residuals = 0) → RSS = 0 → log(0/n) = -inf → BIC = -inf (#15)."""
        result = calculate_bic([0.0, 0.0, 0.0, 0.0], n_params=2, n_obs=4)
        assert result == float('-inf')


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


# ---------------------------------------------------------------------------
# calculate_bf01
# ---------------------------------------------------------------------------

class TestCalculateBf01:
    """BF01 = exp(-ΔBIC / 2) = 1 / BF10."""

    def test_zero_delta_gives_bf_one(self):
        assert pytest.approx(calculate_bf01(0.0)) == 1.0

    def test_positive_delta_gives_bf_below_one(self):
        """Positive ΔBIC favours the full model → BF01 < 1."""
        assert calculate_bf01(10.0) < 1.0

    def test_negative_delta_gives_bf_above_one(self):
        """Negative ΔBIC favours the null model → BF01 > 1."""
        assert calculate_bf01(-10.0) > 1.0

    def test_reciprocal_of_bf10(self):
        for d in (-8.0, -2.0, 0.0, 3.5, 12.0):
            assert pytest.approx(calculate_bf01(d) * calculate_bf10(d), rel=1e-10) == 1.0

    def test_known_value(self):
        """BF01 = exp(-6 / 2) = exp(-3) ≈ 0.0498."""
        assert pytest.approx(calculate_bf01(6.0), rel=1e-6) == np.exp(-3.0)

    def test_returns_float(self):
        assert isinstance(calculate_bf01(5.0), float)


# ---------------------------------------------------------------------------
# bf10_paired_difference_bic
# ---------------------------------------------------------------------------


class TestBf10PairedDifferenceBic:
    """Paired-difference BF₁₀ uses same BIC residual recipe as calculate_bic."""

    def test_two_point_zero_mean_no_evidence_for_difference(self):
        """d = [1, -1] → δ̂ = 0; H0 and H1 same RSS → ΔBIC < 0 → BF₁₀ < 1."""
        bf10, delta = bf10_paired_difference_bic([1.0, -1.0])
        assert delta < 0
        assert bf10 < 1.0

    def test_matches_manual_bic_difference(self):
        """BF₁₀ equals exp((BIC0 - BIC1) / 2) from explicit calculate_bic calls."""
        d = np.array([0.5, 1.0, -0.2, 0.8])
        n = len(d)
        bic0 = calculate_bic(d, n_params=1, n_obs=n)
        bic1 = calculate_bic(d - d.mean(), n_params=2, n_obs=n)
        bf10, delta = bf10_paired_difference_bic(d)
        assert pytest.approx(delta, rel=1e-9) == bic0 - bic1
        assert pytest.approx(bf10, rel=1e-9) == calculate_bf10(bic0 - bic1)

    def test_clear_shift_favours_alternative(self):
        """Large consistent shift should yield BF₁₀ > 1."""
        d = np.ones(30) * 5.0 + np.random.default_rng(0).normal(0, 0.5, size=30)
        bf10, _ = bf10_paired_difference_bic(d)
        assert bf10 > 10.0

    def test_fewer_than_two_observations_returns_nan(self):
        bf10, delta = bf10_paired_difference_bic([1.0])
        assert np.isnan(bf10) and np.isnan(delta)

    def test_identical_differences_returns_inf(self):
        """Perfect fit under H₁ → BIC_alt = -∞; function documents (inf, inf)."""
        bf10, delta = bf10_paired_difference_bic([3.0, 3.0, 3.0])
        assert bf10 == float('inf') and delta == float('inf')


# ---------------------------------------------------------------------------
# bf10_paired_jzs
# ---------------------------------------------------------------------------


class TestBf10PairedJzs:
    """JZS (Rouder et al. 2009) paired-difference BF₁₀ via pingouin."""

    def test_matches_pingouin_directly(self):
        """Wrapper output should equal pingouin.bayesfactor_ttest applied to the same t."""
        rng = np.random.default_rng(42)
        d = rng.normal(0.5, 1.0, size=25)
        n = len(d)
        t_stat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
        expected = float(pg.bayesfactor_ttest(t_stat, nx=n, paired=True, r=0.707))
        assert pytest.approx(bf10_paired_jzs(d), rel=1e-9) == expected

    def test_clear_shift_favours_alternative(self):
        rng = np.random.default_rng(0)
        d = 5.0 + rng.normal(0, 0.5, size=30)
        assert bf10_paired_jzs(d) > 10.0

    def test_zero_mean_noise_favours_null(self):
        rng = np.random.default_rng(1)
        d = rng.normal(0, 1.0, size=30)
        # Random noise around zero should yield BF₁₀ < 1 on average; this
        # particular seed gives a small-mean draw that supports H₀.
        assert bf10_paired_jzs(d) < 1.0

    def test_fewer_than_two_observations_returns_nan(self):
        assert np.isnan(bf10_paired_jzs([1.0]))
        assert np.isnan(bf10_paired_jzs([]))

    def test_nans_are_dropped(self):
        with_nan = [1.0, 2.0, np.nan, 1.5]
        without = [1.0, 2.0, 1.5]
        assert pytest.approx(bf10_paired_jzs(with_nan), rel=1e-9) == bf10_paired_jzs(without)

    def test_all_zero_differences_returns_one(self):
        """Data perfectly consistent with H₀: BF₁₀ should be 1 (not inf)."""
        assert bf10_paired_jzs([0.0, 0.0, 0.0, 0.0]) == 1.0

    def test_nonzero_constant_differences_returns_inf(self):
        """Zero variance with non-zero mean: BF₁₀ = inf."""
        assert bf10_paired_jzs([3.0, 3.0, 3.0]) == float('inf')

    def test_prior_scale_changes_result(self):
        """Different r values should yield different BFs (no-op check)."""
        rng = np.random.default_rng(7)
        d = rng.normal(0.3, 1.0, size=20)
        bf_med = bf10_paired_jzs(d, r=0.707)
        bf_wide = bf10_paired_jzs(d, r=1.0)
        assert bf_med != bf_wide
