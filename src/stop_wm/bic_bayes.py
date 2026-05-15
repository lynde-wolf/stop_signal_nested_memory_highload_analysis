"""BIC-based model comparison and Bayes Factor helpers.

Two families of Bayes-factor utilities are provided:

**BIC-approximation route** (Wagenmakers, 2007) — useful for omnibus
RM-ANOVA-style comparisons where no closed-form Bayes factor is
readily available:

* ``calculate_bic``                 — BIC from model residuals
* ``interpret_bic_delta``           — Kass & Raftery evidence label from ΔBIC
* ``calculate_bf10``                — BF₁₀ ≈ exp(ΔBIC / 2)
* ``calculate_bf01``                — 1 / BF₁₀
* ``bf10_paired_difference_bic``    — paired-difference BF₁₀ via ΔBIC
  (DEPRECATED for new code — prefer ``bf10_paired_jzs``)

**JZS route** (Rouder et al., 2009) — the standard default Cauchy-prior
Bayes factor for paired/two-sample t-tests. Uses ``pingouin`` under the
hood and is more principled than the BIC approximation for small N:

* ``bf10_paired_jzs``               — paired-difference BF₁₀ via JZS

In every ΔBIC-based function ΔBIC is defined as BIC_null − BIC_full, so
positive values favour the full (effects) model.

When to use which paired-BF function
------------------------------------
``bf10_paired_jzs`` is the recommended default. ``bf10_paired_difference_bic``
is kept for back-compatibility with notebooks that already cite the
BIC value and for cross-validation against the JZS result.

Note on the ``k`` convention
----------------------------
``calculate_bic`` counts only the mean parameters — σ² is treated as a
nuisance and excluded from ``n_params``. Differences cancel for model
comparisons, but absolute BIC values are NOT comparable to statsmodels
or R's BIC, which usually include σ².

References
----------
Kass, R. E., & Raftery, A. E. (1995). Bayes factors.
    *Journal of the American Statistical Association*, 90(430), 773–795.

Rouder, J. N., Speckman, P. L., Sun, D., Morey, R. D., & Iverson, G.
    (2009). Bayesian t tests for accepting and rejecting the null
    hypothesis. *Psychonomic Bulletin & Review*, 16(2), 225–237.

Wagenmakers, E.-J. (2007). A practical solution to the pervasive problems
    of p values. *Psychonomic Bulletin & Review*, 14(5), 779–804.
"""

from __future__ import annotations

import numpy as np
import pingouin as pg


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


def bf10_paired_difference_bic(differences: np.ndarray) -> tuple[float, float]:
    """Approximate BF₁₀ for a paired-difference mean via ΔBIC.

    .. deprecated::
        Prefer :func:`bf10_paired_jzs` for new analyses. The BIC
        approximation under the unit-information prior is crude for
        small N and tends to underestimate evidence relative to the
        JZS Cauchy-prior BF used by JASP and R's ``BayesFactor``
        package. Kept for back-compatibility with existing notebooks.

    Compares two nested Gaussian models on the *paired difference* scores
    ``d_i = y_{A,i} - y_{B,i}``:

    * **H₀** (null): ``d_i ~ N(0, σ²)`` — fitted value 0 for all *i*
      (``k = 1`` free parameter in the BIC penalty).
    * **H₁** (alternative): ``d_i ~ N(δ, σ²)`` with ``δ`` estimated by
      the sample mean (``k = 2``).

    ΔBIC = BIC(H₀) − BIC(H₁); :func:`calculate_bf10` maps that to BF₁₀.
    BF₁₀ > 1 favours a non-zero mean difference.  This matches the
    residual-based BIC convention used by :func:`calculate_bic` and the
    omnibus condition BF in the analysis notebooks.

    Parameters
    ----------
    differences : array-like
        One value per subject (paired difference for one contrast).

    Returns
    -------
    bf10 : float
        Approximate Bayes factor for H₁ vs H₀.
    delta_bic : float
        BIC(H₀) − BIC(H₁).

    Notes
    -----
    If all differences are identical, RSS under H₁ is zero and BIC(H₁)
    is undefined (−∞); the function returns ``(inf, inf)``.
    For fewer than two observations, returns ``(nan, nan)``.
    """
    d = np.asarray(differences, dtype=float).ravel()
    n = int(d.size)
    if n < 2:
        return (float('nan'), float('nan'))

    bic_null = calculate_bic(d, n_params=1, n_obs=n)
    d_centered = d - np.mean(d)
    bic_alt = calculate_bic(d_centered, n_params=2, n_obs=n)

    if not np.isfinite(bic_alt):
        return (float('inf'), float('inf'))

    delta_bic = bic_null - bic_alt
    return (calculate_bf10(delta_bic), float(delta_bic))


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


def bf10_paired_jzs(
    differences: np.ndarray,
    r: float = 0.707,
) -> float:
    """JZS Bayes factor BF₁₀ for a paired-difference mean (Rouder et al., 2009).

    Wraps ``pingouin.bayesfactor_ttest`` so notebooks have a one-call
    drop-in replacement for ``bf10_paired_difference_bic``. The JZS
    prior on effect size is a Cauchy with scale ``r`` (default 0.707,
    the "medium" prior recommended in Rouder et al., 2009 and used by
    JASP).

    Parameters
    ----------
    differences : array-like
        Paired difference scores ``d_i = y_{A,i} - y_{B,i}`` — one per
        subject. NaNs are dropped.
    r : float, optional
        Cauchy prior scale on the standardized effect size. Common
        choices: 0.5 (small), 0.707 (medium, default), 1.0 (wide).

    Returns
    -------
    float
        BF₁₀ favouring H₁ (non-zero mean difference) over H₀ (zero mean).
        ``nan`` for fewer than 2 non-NaN observations; ``inf`` when the
        differences are a non-zero constant (zero variance, non-zero mean);
        ``1.0`` when all differences are exactly zero (data perfectly
        consistent with H₀).
    """
    d = np.asarray(differences, dtype=float).ravel()
    d = d[~np.isnan(d)]
    n = int(d.size)
    if n < 2:
        return float('nan')

    mean = float(d.mean())
    sd = float(d.std(ddof=1))

    if sd == 0:
        return float('inf') if mean != 0 else 1.0

    t = mean / (sd / np.sqrt(n))
    return float(pg.bayesfactor_ttest(t, nx=n, paired=True, r=r))


def calculate_bf01(delta_bic: float) -> float:
    """Approximate BF₀₁ from ΔBIC (reciprocal of BF₁₀).

    BF₀₁ = exp(-ΔBIC / 2) = 1 / BF₁₀, where ΔBIC = BIC_null − BIC_full.

    BF₀₁ > 1 is evidence for the null model; BF₀₁ < 1 is evidence
    against it.

    Parameters
    ----------
    delta_bic : float
        BIC_null − BIC_full.

    Returns
    -------
    float
        Approximate Bayes Factor favouring the null over the full model.
    """
    return float(np.exp(-delta_bic / 2))
