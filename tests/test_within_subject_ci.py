"""Tests for within_subject_ci.calculate_within_subject_ci."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from stop_wm.within_subject_ci import calculate_within_subject_ci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix(data: list[list[float]]) -> pd.DataFrame:
    """Convenience: list-of-rows → participants × conditions DataFrame."""
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Known-answer test
# ---------------------------------------------------------------------------

class TestKnownAnswer:
    """Verify the result against a manual hand-calculation."""

    def test_two_conditions_simple(self):
        """3 participants, 2 conditions — trace through the math by hand."""
        # Participants × conditions
        raw = [[4.0, 6.0],
               [5.0, 7.0],
               [3.0, 5.0]]
        df = _make_matrix(raw)

        # --- Manual calculation ---
        # participant means: [5, 6, 4]  grand mean: 5
        # centred data:
        #   [4-5+5, 6-5+5] = [4, 6]
        #   [5-6+5, 7-6+5] = [4, 6]
        #   [3-4+5, 5-4+5] = [4, 6]
        # variance per condition (ddof=1): both = 0
        # SEM = 0, corrected SEM = 0, CI half-width = 0

        result = calculate_within_subject_ci(df)

        assert result.shape == (2,)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=1e-10)

    def test_known_nonzero_ci(self):
        """4 participants, 3 conditions — independently computed expected values."""
        raw = [[10.0, 12.0, 11.0],
               [20.0, 22.0, 21.0],
               [15.0, 17.0, 16.0],
               [25.0, 27.0, 26.0]]
        df = _make_matrix(raw)

        n_participants = 4
        n_conditions = 3

        # Participant means: [11, 21, 16, 26]  grand mean: 18.5
        # Centred:
        #  row 0: [10-11+18.5, 12-11+18.5, 11-11+18.5] = [17.5, 19.5, 18.5]
        #  row 1: [20-21+18.5, 22-21+18.5, 21-21+18.5] = [17.5, 19.5, 18.5]
        #  row 2: [15-16+18.5, 17-16+18.5, 16-16+18.5] = [17.5, 19.5, 18.5]
        #  row 3: [25-26+18.5, 27-26+18.5, 26-26+18.5] = [17.5, 19.5, 18.5]
        # All rows identical → var per condition = 0
        result = calculate_within_subject_ci(df)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_known_nonzero_variance(self):
        """Participants differ after norming — CI > 0, check against manual formula."""
        raw = [[10.0, 14.0],
               [12.0, 14.0],
               [14.0, 14.0],
               [16.0, 14.0]]
        df = _make_matrix(raw)
        #  participant means: [12, 13, 14, 15]  grand mean: 13.5
        #  centred col 0: [10-12+13.5, 12-13+13.5, 14-14+13.5, 16-15+13.5]
        #                 = [11.5, 12.5, 13.5, 14.5]
        #  centred col 1: [14-12+13.5, 14-13+13.5, 14-14+13.5, 14-15+13.5]
        #                 = [15.5, 14.5, 13.5, 12.5]
        #  var col 0 = var([11.5,12.5,13.5,14.5], ddof=1) = 5/3
        #  var col 1 = var([15.5,14.5,13.5,12.5], ddof=1) = 5/3
        n = 4
        k = 2
        var_expected = np.var([11.5, 12.5, 13.5, 14.5], ddof=1)
        sem = np.sqrt(var_expected / n)
        correction = np.sqrt(k / (k - 1))
        # Morey (2008) within-subject error df: (n-1)(k-1)
        t_crit = stats.t.ppf(0.975, df=(n - 1) * (k - 1))
        expected_ci = t_crit * sem * correction

        result = calculate_within_subject_ci(df)

        np.testing.assert_allclose(result.values, [expected_ci, expected_ci], rtol=1e-6)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaNHandling:
    """Rows with any NaN should be dropped (listwise deletion)."""

    def test_partial_nan_row_dropped(self):
        """A row with one NaN is excluded; remaining rows drive the result."""
        complete = [[10.0, 14.0],
                    [12.0, 14.0],
                    [14.0, 14.0],
                    [16.0, 14.0]]
        with_nan = complete + [[np.nan, 13.0]]
        df_complete = _make_matrix(complete)
        df_with_nan = _make_matrix(with_nan)

        result_complete = calculate_within_subject_ci(df_complete)
        result_with_nan = calculate_within_subject_ci(df_with_nan)

        np.testing.assert_allclose(result_complete, result_with_nan, rtol=1e-6)

    def test_all_nan_returns_nan_array(self):
        """All-NaN matrix → array of NaNs, not an exception."""
        df = _make_matrix([[np.nan, np.nan],
                           [np.nan, np.nan]])
        result = calculate_within_subject_ci(df)
        assert result.shape == (2,)
        assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# Edge-case guards
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Conditions that should raise or behave predictably."""

    def test_single_condition_raises(self):
        """The Morey correction is undefined for k=1; must raise ValueError."""
        df = _make_matrix([[1.0], [2.0], [3.0]])
        with pytest.raises(ValueError, match="at least 2 conditions"):
            calculate_within_subject_ci(df)

    def test_confidence_level_boundary(self):
        """Non-default confidence level should change the CI proportionally."""
        df = _make_matrix([[10.0, 14.0],
                           [12.0, 14.0],
                           [14.0, 14.0],
                           [16.0, 14.0]])
        ci_95 = calculate_within_subject_ci(df, confidence_level=0.95)
        ci_99 = calculate_within_subject_ci(df, confidence_level=0.99)
        # Wider confidence → larger half-width
        assert np.all(ci_99 > ci_95)

    def test_output_shape_matches_n_conditions(self):
        """Return array length equals number of columns."""
        for n_conds in [2, 3, 5]:
            raw = [[float(i + j) for j in range(n_conds)] for i in range(6)]
            df = _make_matrix(raw)
            result = calculate_within_subject_ci(df)
            assert result.shape == (n_conds,)

    def test_single_participant_returns_nan(self):
        """With n=1, within-subject variance is undefined → NaN CIs + warning."""
        df = _make_matrix([[10.0, 14.0]])
        with pytest.warns(RuntimeWarning, match="Only 1 participant"):
            result = calculate_within_subject_ci(df)
        assert result.shape == (2,)
        assert np.all(np.isnan(result))

    def test_invalid_confidence_level_raises(self):
        df = _make_matrix([[1.0, 2.0], [3.0, 4.0]])
        for bad in [0.0, 1.0, -0.1, 1.5, 95]:
            with pytest.raises(ValueError, match="confidence_level"):
                calculate_within_subject_ci(df, confidence_level=bad)

    def test_returns_series_with_column_index(self):
        """Result is a pandas Series keyed by the original column labels."""
        df = pd.DataFrame(
            [[10.0, 14.0], [12.0, 14.0], [14.0, 14.0], [16.0, 14.0]],
            columns=["load_2", "load_4"],
        )
        result = calculate_within_subject_ci(df)
        assert isinstance(result, pd.Series)
        assert list(result.index) == ["load_2", "load_4"]

    def test_return_n_reports_complete_count(self):
        df = _make_matrix([[10.0, 14.0],
                           [12.0, 14.0],
                           [np.nan, 14.0],
                           [16.0, 14.0]])
        result, n_used = calculate_within_subject_ci(df, return_n=True)
        assert n_used == 3
        assert result.shape == (2,)
