"""BIC-based model comparison and Bayes Factor helpers.

Provides three utilities for Bayesian model comparison via the BIC
approximation:

* ``calculate_bic``       — BIC from model residuals
* ``interpret_bic_delta`` — evidence-strength label from ΔBIC
* ``calculate_bf10``      — approximate Bayes Factor from ΔBIC

In all three functions ΔBIC is defined as BIC_null − BIC_full, so
positive values favour the full (effects) model.

References
----------
Kass, R. E., & Raftery, A. E. (1995). Bayes factors.
    *Journal of the American Statistical Association*, 90(430), 773–795.

Wagenmakers, E.-J. (2007). A practical solution to the pervasive problems
    of p values. *Psychonomic Bulletin & Review*, 14(5), 779–804.
"""

from __future__ import annotations

import numpy as np


def calculate_bic(
    residuals: np.ndarray,
    n_params: int,
    n_obs: int,
) -> float:
    """Calculate BIC for a model given its residuals.

    Uses the residual-based formula:

        BIC = n · ln(RSS / n) + k · ln(n)

    where RSS is the sum of squared residuals, k is the number of free
    parameters, and n is the number of observations.

    Parameters
    ----------
    residuals : array-like
        Residuals from the fitted model (observed − predicted).
    n_params : int
        Number of free parameters in the model.
    n_obs : int
        Number of observations used to fit the model.

    Returns
    -------
    float
        BIC value (lower is better).
    """
    residuals = np.asarray(residuals, dtype=float)
    rss = np.sum(residuals ** 2)
    return n_obs * np.log(rss / n_obs) + n_params * np.log(n_obs)


def interpret_bic_delta(delta_bic: float, verbose: bool = True) -> str:
    """Return an evidence-strength label for a given ΔBIC.

    ΔBIC = BIC_null − BIC_full; positive values favour the full model.
    Thresholds follow Kass & Raftery (1995).

    Parameters
    ----------
    delta_bic : float
        BIC_null − BIC_full.
    verbose : bool, optional
        If ``True`` (default) return a full descriptive phrase suitable
        for prose.  If ``False`` return a compact label suitable for
        tables (e.g. ``"Very Strong FOR"``).

    Returns
    -------
    str
        Evidence-strength label.
    """
    if verbose:
        if delta_bic > 10:
            return "Very Strong evidence for condition effects"
        elif delta_bic > 6:
            return "Strong evidence for condition effects"
        elif delta_bic > 2:
            return "Positive evidence for condition effects"
        elif delta_bic > 0:
            return "Weak evidence for condition effects"
        elif delta_bic > -2:
            return "Weak evidence against condition effects"
        elif delta_bic > -6:
            return "Positive evidence against condition effects"
        elif delta_bic > -10:
            return "Strong evidence against condition effects"
        else:
            return "Very Strong evidence against condition effects"
    else:
        if delta_bic > 10:
            return "Very Strong FOR"
        elif delta_bic > 6:
            return "Strong FOR"
        elif delta_bic > 2:
            return "Positive FOR"
        elif delta_bic > 0:
            return "Weak FOR"
        elif delta_bic > -2:
            return "Weak AGAINST"
        elif delta_bic > -6:
            return "Positive AGAINST"
        elif delta_bic > -10:
            return "Strong AGAINST"
        else:
            return "Very Strong AGAINST"


def calculate_bf10(delta_bic: float) -> float:
    """Approximate BF₁₀ from ΔBIC (Wagenmakers, 2007).

    BF₁₀ = exp(ΔBIC / 2),  where ΔBIC = BIC_null − BIC_full.

    BF₁₀ > 1 is evidence for the full model; BF₁₀ < 1 is evidence
    against it.

    Parameters
    ----------
    delta_bic : float
        BIC_null − BIC_full.

    Returns
    -------
    float
        Approximate Bayes Factor favouring the full model over the null.
    """
    return float(np.exp(delta_bic / 2))
