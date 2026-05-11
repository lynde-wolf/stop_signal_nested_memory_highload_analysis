"""Race-model violation diagnostics for stop-signal data, resolved by SSD.

Two diagnostics are provided:

1. **Global-mean diagnostic** (``compute_per_subject_per_ssd``) — the standard
   Verbruggen-2019 check: per (subject, SSD), compare mean failed-stop RT
   against the subject's overall correct-go RT. Cheap and robust to SSD
   sample sizes, but blind to within-session RT drift.

2. **Paired-preceding diagnostic** (``compute_per_subject_per_ssd_preceding``) —
   the stricter Bissett-2019 method: for each failed-stop trial, find the
   *immediately preceding* go trial (same subject, same block) and use that
   trial's RT as the matched reference. Faithful port of
   ``violation_analysis`` in
   ``henrymj-ContextDependence-a1c0d79/utils.py``. Requires at least 2 valid
   pairs per (subject, SSD) and at least 5 subjects per SSD before that SSD
   contributes to group-level summaries.

Also provided:

3. **SSRT short-vs-long comparison** (``ssrt_short_vs_long_comparison``) —
   per subject, compute SSRT twice (with all SSDs vs. with SSDs >= threshold)
   and report the paired difference + paired t-test. This implements
   recommendation 2a of Bissett et al. (2019/2020).

References
----------
Logan, G. D., & Cowan, W. B. (1984). On the ability to inhibit thought and
action. *Psychological Review, 91*, 295-327.

Verbruggen, F., et al. (2019). A consensus guide to capturing the ability to
inhibit actions and impulsive behaviors in the stop-signal task.
*eLife, 8*, e46323.

Bissett, P. G., Jones, H. M., Poldrack, R. A., & Logan, G. D. (2020).
Severe and pervasive violations of independence in response inhibition tasks.
*Science Advances*. (Reference implementation:
``henrymj-ContextDependence-a1c0d79/utils.py::violation_analysis``.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_per_subject_per_ssd(
    df: pd.DataFrame,
    trial_type_col: str,
    rt_col: str,
    correct_col: str,
    ssd_col: str,
    subject_col: str = "participant_id",
    condition_col: str | None = None,
) -> pd.DataFrame:
    """Compute mean(failed-stop RT) − mean(correct-go RT) per (subject, SSD).

    Parameters
    ----------
    df : DataFrame of trial-level data.
    trial_type_col : column with 'go'/'stop'.
    rt_col : RT column (NaN or non-positive treated as no-response).
    correct_col : 0/1 correctness column.
    ssd_col : SSD column (NaN on go trials).
    subject_col : subject identifier.
    condition_col : optional grouping (e.g. memory load).

    Returns
    -------
    DataFrame with one row per (subject, SSD[, condition]) for which the
    subject had at least one failed-stop trial. Columns:
    subject, ssd, [condition], n_failed_stops_at_ssd,
    mean_failed_stop_rt, mean_correct_go_rt, diff_ms, violator_at_ssd.
    """
    rows = []
    grp_keys = [subject_col] + ([condition_col] if condition_col else [])
    grouper = grp_keys[0] if len(grp_keys) == 1 else grp_keys
    for keys, sub in df.groupby(grouper, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        subject = keys[0]
        condition = keys[1] if condition_col else None

        # Reference: subject's overall correct-go RT (not conditional on SSD)
        go = sub[(sub[trial_type_col] == "go") & (sub[correct_col] == 1)][rt_col]
        go = go[go.notna() & (go > 0)]
        if len(go) == 0:
            continue
        mean_go = float(go.mean())

        stop = sub[sub[trial_type_col] == "stop"]
        for ssd, ssd_block in stop.groupby(ssd_col, dropna=False):
            if pd.isna(ssd):
                continue
            fs = ssd_block.loc[ssd_block[correct_col] == 0, rt_col]
            fs = fs[fs.notna() & (fs > 0)]
            if len(fs) == 0:
                continue
            mean_fs = float(fs.mean())
            row = {
                subject_col: subject,
                ssd_col: float(ssd),
                "n_failed_stops_at_ssd": int(len(fs)),
                "mean_failed_stop_rt": mean_fs,
                "mean_correct_go_rt": mean_go,
                "diff_ms": mean_fs - mean_go,
                "violator_at_ssd": bool((mean_fs - mean_go) > 0),
            }
            if condition_col:
                row[condition_col] = condition
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_by_ssd(
    per_subject: pd.DataFrame,
    ssd_col: str,
    condition_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate per-subject (diff_ms, violator_at_ssd) across subjects per SSD.

    Returns one row per (condition, SSD) with mean diff, SEM, n_subjects, and
    p_subjects_violate.
    """
    grp = ([condition_col] if condition_col else []) + [ssd_col]
    out = (
        per_subject.groupby(grp, dropna=False)
        .agg(
            n_subjects=("diff_ms", "size"),
            mean_diff_ms=("diff_ms", "mean"),
            sem_diff_ms=(
                "diff_ms",
                lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan,
            ),
            p_subjects_violate=(
                "violator_at_ssd",
                lambda s: float(np.mean(s.astype(bool))),
            ),
        )
        .reset_index()
    )
    return out


# ============================================================================
# Bissett 2019 paired-preceding diagnostic
# ============================================================================
def compute_per_subject_per_ssd_preceding(
    df: pd.DataFrame,
    trial_type_col: str,
    rt_col: str,
    correct_col: str,
    ssd_col: str,
    subject_col: str = "participant_id",
    block_col: str = "block_num",
    trial_order_col: str = "current_trial",
    condition_col: str | None = None,
    min_pairs_per_subject_ssd: int = 2,
) -> pd.DataFrame:
    """Bissett 2019 paired-preceding race-model violation diagnostic.

    For each stop-failure trial, the matched reference is the *immediately
    preceding* trial within the same (subject, block), provided that
    preceding trial was a go trial with a valid (positive, non-NaN) RT.
    Per (subject, SSD) the function returns the mean signed difference
    ``stop_failure_rt - matched_preceding_go_rt``. A (subject, SSD) cell is
    only kept if ``n_pairs >= min_pairs_per_subject_ssd`` (Bissett 2019
    requires >= 2).

    This is a faithful port of
    ``henrymj-ContextDependence-a1c0d79/utils.py::violation_analysis`` (lines
    165-256), adapted to this project's column conventions.

    Parameters
    ----------
    df : DataFrame of trial-level data, one row per trial.
    trial_type_col : column with 'go' / 'stop'.
    rt_col : RT column. NaN or non-positive values are treated as
        no-response (omission).
    correct_col : 0/1 correctness column. A failed stop is
        ``trial_type == 'stop'`` and ``correct == 0``.
    ssd_col : SSD column.
    subject_col : subject identifier column.
    block_col : block identifier column. Used to forbid pairing across blocks.
    trial_order_col : column whose monotonic increase within
        (subject, block) defines trial order.
    condition_col : optional grouping column carried through (e.g. memory
        load).
    min_pairs_per_subject_ssd : minimum valid (stop-failure, preceding-go)
        pairs required for a (subject, SSD) cell to be kept. Bissett 2019
        uses 2.

    Returns
    -------
    DataFrame with one row per kept (subject, SSD[, condition]) and columns:
    ``subject_col, ssd_col, [condition_col], n_pairs, mean_stop_failure_rt,
    mean_preceding_go_rt, mean_violation_ms, violator_at_ssd``.
    A positive ``mean_violation_ms`` flags a race-model violation.
    """
    sort_keys = [subject_col, block_col, trial_order_col]
    df = df.sort_values(sort_keys).reset_index(drop=True)

    # Build "previous trial" features within (subject, block).
    grp = df.groupby([subject_col, block_col], dropna=False)
    prev_type = grp[trial_type_col].shift(1)
    prev_rt = grp[rt_col].shift(1)

    # Identify failed-stop trials with a real response (Bissett's
    # `dropna(subset=[stopFailRT])` + `> 0` filter).
    is_stop_fail = (
        (df[trial_type_col] == "stop")
        & (df[correct_col] == 0)
        & df[rt_col].notna()
        & (df[rt_col] > 0)
    )

    # The matched RT is the previous trial's RT iff the previous trial was a
    # valid go response (skip omissions; skip stop trials).
    prev_is_valid_go = (
        (prev_type == "go") & prev_rt.notna() & (prev_rt > 0)
    )
    matched_rt = prev_rt.where(prev_is_valid_go, np.nan)

    sf = df.loc[is_stop_fail].copy()
    sf["_match_rt"] = matched_rt.loc[is_stop_fail].values

    group_keys: list[str] = [subject_col]
    if condition_col is not None:
        group_keys.append(condition_col)
    group_keys.append(ssd_col)

    rows = []
    for keys, sub in sf.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        keymap = dict(zip(group_keys, keys))
        ssd_val = keymap[ssd_col]
        if pd.isna(ssd_val):
            continue
        valid = sub.dropna(subset=["_match_rt"])
        n_pairs = int(len(valid))
        if n_pairs < min_pairs_per_subject_ssd:
            continue
        diff = valid[rt_col] - valid["_match_rt"]
        rows.append(
            {
                subject_col: keymap[subject_col],
                ssd_col: float(ssd_val),
                **(
                    {condition_col: keymap[condition_col]} if condition_col else {}
                ),
                "n_pairs": n_pairs,
                "mean_stop_failure_rt": float(valid[rt_col].mean()),
                "mean_preceding_go_rt": float(valid["_match_rt"].mean()),
                "mean_violation_ms": float(diff.mean()),
                "violator_at_ssd": bool(diff.mean() > 0),
            }
        )
    return pd.DataFrame(rows)


def aggregate_by_ssd_preceding(
    per_subject: pd.DataFrame,
    ssd_col: str,
    condition_col: str | None = None,
    min_subjects_per_ssd: int = 5,
) -> pd.DataFrame:
    """Aggregate paired-preceding violations across subjects per SSD.

    Drops any SSD cell that has fewer than ``min_subjects_per_ssd`` subjects
    (Bissett 2019 requires >= 5).
    """
    grp = ([condition_col] if condition_col else []) + [ssd_col]
    out = (
        per_subject.groupby(grp, dropna=False)
        .agg(
            n_subjects=("mean_violation_ms", "size"),
            mean_violation_ms=("mean_violation_ms", "mean"),
            sem_violation_ms=(
                "mean_violation_ms",
                lambda s: float(s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else np.nan,
            ),
            p_subjects_violate=(
                "violator_at_ssd",
                lambda s: float(np.mean(s.astype(bool))),
            ),
        )
        .reset_index()
    )
    return out[out["n_subjects"] >= min_subjects_per_ssd].reset_index(drop=True)


# ============================================================================
# Bissett 2019 SSRT short-vs-long-SSD comparison (recommendation 2a)
# ============================================================================
def ssrt_short_vs_long_comparison(
    df: pd.DataFrame,
    ssrt_fn,
    trial_type_col: str,
    rt_col: str,
    correct_col: str,
    ssd_col: str,
    response_deadline_col: str,
    subject_col: str = "participant_id",
    condition_col: str | None = None,
    ssd_threshold_ms: float = 200.0,
    require_short_and_long_pair: bool = True,
) -> pd.DataFrame:
    """Per-subject SSRT computed with all stop trials vs. SSDs >= threshold.

    Faithful port of ``utils.py::ssrt_comparison`` + ``filter_ssrt_subs``
    from the Bissett 2019 reproducibility code, adapted to this project's
    integration-method SSRT calculator.

    Parameters
    ----------
    df : trial-level DataFrame.
    ssrt_fn : callable with the signature
        ``ssrt_fn(go_rts, go_omission_count, stop_success_rate, mean_ssd,
        response_deadline) -> float`` — i.e. the project's
        ``calculate_ssrt_integration``.
    trial_type_col, rt_col, correct_col, ssd_col, response_deadline_col,
    subject_col, condition_col :
        column names. ``response_deadline_col`` is read from go trials and
        the maximum value across the subject is passed as the deadline
        (Verbruggen 2019, Method 1).
    ssd_threshold_ms : SSDs strictly below this are excluded for the
        ``ssrt_long_ssd`` column. Default 200, matching Bissett 2019.
    require_short_and_long_pair : if True, drop subjects who do not have at
        least one stop-fail and one stop-success at SSD >= threshold
        (Bissett 2019 ``filter_ssrt_subs`` semantics).

    Returns
    -------
    DataFrame with one row per (subject[, condition]) and columns
    ``subject_col, [condition_col], ssrt_all_ssd, ssrt_long_ssd, ssrt_diff_ms,
    n_stop_total, n_stop_long, response_deadline``.
    """

    def _ssrt_for_block(block: pd.DataFrame) -> tuple[float, int, int]:
        go = block[block[trial_type_col] == "go"]
        stop = block[block[trial_type_col] == "stop"]
        if len(stop) == 0:
            return float("nan"), 0, 0
        go_rts = go[rt_col]
        n_go_omissions = int(go_rts.isna().sum() + (go_rts <= 0).sum())
        valid_go_rts = go_rts.dropna()
        valid_go_rts = valid_go_rts[valid_go_rts > 0]
        # Successful stops are coded correct == 1 by convention here.
        n_stop = int(len(stop))
        n_success = int((stop[correct_col] == 1).sum())
        stop_success_rate = n_success / n_stop if n_stop > 0 else float("nan")
        ssd_vals = stop[ssd_col].dropna()
        mean_ssd = float(ssd_vals.mean()) if len(ssd_vals) else float("nan")
        # Response deadline: largest deadline observed for this subject in
        # the source df (matches Bissett's `getmaxRT`).
        deadlines = block[response_deadline_col].dropna()
        deadline = float(deadlines.max()) if len(deadlines) else float("nan")
        ssrt = ssrt_fn(
            valid_go_rts,
            n_go_omissions,
            stop_success_rate,
            mean_ssd,
            deadline,
        )
        return float(ssrt) if ssrt is not None else float("nan"), n_stop, int(
            (stop[ssd_col] >= ssd_threshold_ms).sum()
        )

    keys = [subject_col] + ([condition_col] if condition_col else [])
    rows = []
    for k, sub in df.groupby(keys, dropna=False):
        if not isinstance(k, tuple):
            k = (k,)
        keymap = dict(zip(keys, k))

        # Long-SSD filter: keep only stop trials at >= threshold; keep all go
        # trials (the integration method uses the full go RT distribution).
        long_mask = (
            (sub[trial_type_col] == "go")
            | (sub[ssd_col].isna())
            | (sub[ssd_col] >= ssd_threshold_ms)
        )
        sub_long = sub[long_mask]

        if require_short_and_long_pair:
            long_stop = sub_long[sub_long[trial_type_col] == "stop"]
            n_fail = int(((long_stop[correct_col] == 0)
                          & long_stop[rt_col].notna()).sum())
            n_succ = int((long_stop[correct_col] == 1).sum())
            if n_fail == 0 or n_succ == 0:
                continue

        ssrt_all, n_stop_total, n_stop_long = _ssrt_for_block(sub)
        ssrt_long, _, _ = _ssrt_for_block(sub_long)
        deadlines = sub[response_deadline_col].dropna()
        rows.append(
            {
                **keymap,
                "ssrt_all_ssd": ssrt_all,
                "ssrt_long_ssd": ssrt_long,
                "ssrt_diff_ms": ssrt_all - ssrt_long,
                "n_stop_total": n_stop_total,
                "n_stop_long": n_stop_long,
                "response_deadline": (
                    float(deadlines.max()) if len(deadlines) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)
