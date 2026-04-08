"""Within-subject confidence intervals (Cousineau-Morey method).

Implements the Cousineau (2005) norming procedure with the Morey (2008)
bias-correction factor for computing CIs that reflect only within-subject
variability — appropriate for repeated-measures designs.

References
----------
Cousineau, D. (2005). Confidence intervals in within-subject designs:
    A simpler solution to Loftus and Masson's method.
    *Tutorials in Quantitative Methods for Psychology*, 1(1), 42–45.

Morey, R. D. (2008). Confidence intervals from normalized data:
    A correction to Cousineau (2005).
    *Tutorials in Quantitative Methods for Psychology*, 4(2), 61–64.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def calculate_within_subject_ci(
    data_matrix: pd.DataFrame,
    confidence_level: float = 0.95,
) -> np.ndarray:
    """Compute within-subject CI half-widths for each condition.

    Parameters
    ----------
    data_matrix : DataFrame
        Participants (rows) x conditions (columns).  Rows that contain
        any ``NaN`` are dropped before computation.
    confidence_level : float, optional
        Confidence level for the interval (default 0.95).

    Returns
    -------
    ci_half_width : ndarray
        Half-width of the confidence interval for each condition.
    """
    complete_data = data_matrix.dropna()

    if len(complete_data) == 0:
        return np.array([np.nan] * data_matrix.shape[1])

    participant_means = complete_data.mean(axis=1)
    grand_mean = complete_data.values.mean()

    centered_data = complete_data.subtract(participant_means, axis=0) + grand_mean

    n_participants = len(complete_data)
    n_conditions = complete_data.shape[1]

    if n_conditions < 2:
        raise ValueError(
            "Within-subject CIs require at least 2 conditions; "
            f"got {n_conditions}."
        )

    condition_vars = centered_data.var(axis=0, ddof=1)
    within_subject_sem = np.sqrt(condition_vars / n_participants)

    # Morey (2008) correction factor
    correction_factor = np.sqrt(n_conditions / (n_conditions - 1))
    corrected_sem = within_subject_sem * correction_factor

    t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n_participants - 1)
    ci_half_width = t_critical * corrected_sem

    # Ensure we return a plain ndarray (not a Series carrying the
    # DataFrame's column index) so callers can use positional indexing.
    return np.asarray(ci_half_width, dtype=float)
