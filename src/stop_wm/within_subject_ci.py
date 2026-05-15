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

import warnings

import numpy as np
import pandas as pd
from scipy import stats


def calculate_within_subject_ci(
    data_matrix: pd.DataFrame,
    confidence_level: float = 0.95,
    return_n: bool = False,
    pooled: bool = True,
) -> pd.Series | tuple[pd.Series, int]:
    """Compute within-subject CI half-widths for each condition.

    Parameters
    ----------
    data_matrix : DataFrame
        Participants (rows) x conditions (columns).  Rows that contain
        any ``NaN`` are dropped (listwise deletion) before computation,
        because the per-participant centering step requires a value in
        every condition for every retained subject.
    confidence_level : float, optional
        Confidence level for the interval (default 0.95). Must satisfy
        ``0 < confidence_level < 1``.
    return_n : bool, optional
        If ``True``, also return the number of participants used after
        listwise deletion. Default ``False`` for backwards compatibility.
    pooled : bool, optional
        If ``True`` (default), use a single pooled within-subject
        variance — the mean of the per-condition variances on the
        Cousineau-normalized data — so every condition gets the **same**
        CI half-width. This is the Loftus & Masson (1994) /
        ANOVA-MS_{S×C} flavor and is the more common choice for
        repeated-measures bar plots: it assumes sphericity and produces
        visually comparable error bars across bars. If ``False``, return
        per-condition CI half-widths computed from each column's own
        normalized variance (Cousineau 2005 / Morey 2008 in its strict
        per-condition reading) — bars can have unequal widths if
        sphericity is violated. For ``k = 2`` conditions the two methods
        are algebraically identical.

    Returns
    -------
    ci_half_width : Series
        Half-width of the confidence interval for each condition,
        indexed by the original column labels.
    n_used : int, optional
        Number of participants surviving listwise deletion. Returned
        only when ``return_n=True``.

    Notes
    -----
    Degrees of freedom for the t critical follow Morey (2008): the
    within-subject error term has ``df = (n - 1) * (k - 1)``, where
    ``n`` is the number of participants and ``k`` the number of
    conditions. This differs from the ``n - 1`` Cousineau (2005)
    originally used and produces a slightly tighter CI when ``k`` is
    small.

    The grand mean is computed as the mean of all cells, which after
    listwise deletion equals the mean of the participant means (the
    quantity Cousineau's formula calls for) because the matrix is
    balanced.
    """
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must be in (0, 1); got {confidence_level}."
        )

    n_conditions_total = data_matrix.shape[1]
    columns = data_matrix.columns

    if n_conditions_total < 2:
        raise ValueError(
            "Within-subject CIs require at least 2 conditions; "
            f"got {n_conditions_total}."
        )

    complete_data = data_matrix.dropna()
    n_participants = len(complete_data)

    if n_participants < 2:
        if n_participants == 1:
            warnings.warn(
                "Only 1 participant has complete data; within-subject "
                "variance is undefined. Returning NaN CIs.",
                RuntimeWarning,
                stacklevel=2,
            )
        result = pd.Series(np.nan, index=columns, dtype=float)
        return (result, n_participants) if return_n else result

    n_conditions = complete_data.shape[1]

    participant_means = complete_data.mean(axis=1)
    grand_mean = complete_data.values.mean()
    centered_data = complete_data.subtract(participant_means, axis=0) + grand_mean

    condition_vars = centered_data.var(axis=0, ddof=1)
    if pooled:
        # Pooled within-subject variance: mean of per-condition normed variances.
        # Equivalent to MS_{S×C} from the RM-ANOVA on the balanced matrix.
        condition_vars = pd.Series(condition_vars.mean(), index=columns)
    within_subject_sem = np.sqrt(condition_vars / n_participants)

    # Morey (2008) correction factor: sqrt(k / (k - 1))
    correction_factor = np.sqrt(n_conditions / (n_conditions - 1))
    corrected_sem = within_subject_sem * correction_factor

    # Morey (2008) within-subject error df: (n - 1)(k - 1)
    df = (n_participants - 1) * (n_conditions - 1)
    t_critical = stats.t.ppf((1 + confidence_level) / 2, df=df)
    ci_half_width = pd.Series(
        t_critical * corrected_sem.values,
        index=columns,
        dtype=float,
    )

    return (ci_half_width, n_participants) if return_n else ci_half_width
