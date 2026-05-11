"""Tests for stop_wm.race_model_check."""
import numpy as np
import pandas as pd
import pytest

from stop_wm.race_model_check import (
    aggregate_by_ssd,
    aggregate_by_ssd_preceding,
    compute_per_subject_per_ssd,
    compute_per_subject_per_ssd_preceding,
    ssrt_short_vs_long_comparison,
)


def _make_df():
    """Two subjects with hand-computed expected diffs."""
    rows = [
        # Subject A: go RTs 400, 500, 600 (mean 500). Failed stops at SSD=100: 480, 520 (mean 500). At SSD=300: 600 (>500, violation).
        {"pid": "A", "tt": "go", "rt": 400, "corr": 1, "ssd": np.nan},
        {"pid": "A", "tt": "go", "rt": 500, "corr": 1, "ssd": np.nan},
        {"pid": "A", "tt": "go", "rt": 600, "corr": 1, "ssd": np.nan},
        {"pid": "A", "tt": "stop", "rt": 480, "corr": 0, "ssd": 100},
        {"pid": "A", "tt": "stop", "rt": 520, "corr": 0, "ssd": 100},
        {"pid": "A", "tt": "stop", "rt": -1, "corr": 1, "ssd": 100},  # successful stop, ignored
        {"pid": "A", "tt": "stop", "rt": 600, "corr": 0, "ssd": 300},
        # Subject B: go mean 450. Failed stops at SSD=100: 400 (no violation). No stops at 300.
        {"pid": "B", "tt": "go", "rt": 400, "corr": 1, "ssd": np.nan},
        {"pid": "B", "tt": "go", "rt": 500, "corr": 1, "ssd": np.nan},
        {"pid": "B", "tt": "stop", "rt": 400, "corr": 0, "ssd": 100},
    ]
    return pd.DataFrame(rows)


def test_per_subject_per_ssd_basic():
    df = _make_df()
    out = compute_per_subject_per_ssd(
        df, "tt", "rt", "corr", "ssd", subject_col="pid"
    )
    out = out.set_index(["pid", "ssd"]).sort_index()
    # Subject A at SSD 100: failed-stop mean = 500, go mean = 500, diff = 0, no violation
    assert out.loc[("A", 100.0), "mean_failed_stop_rt"] == pytest.approx(500.0)
    assert out.loc[("A", 100.0), "mean_correct_go_rt"] == pytest.approx(500.0)
    assert out.loc[("A", 100.0), "diff_ms"] == pytest.approx(0.0)
    assert out.loc[("A", 100.0), "violator_at_ssd"] is np.False_ or out.loc[("A", 100.0), "violator_at_ssd"] == False
    # Subject A at SSD 300: 600 > 500, violation
    assert out.loc[("A", 300.0), "diff_ms"] == pytest.approx(100.0)
    assert bool(out.loc[("A", 300.0), "violator_at_ssd"]) is True
    # Subject B at SSD 100: 400 < 450, no violation
    assert out.loc[("B", 100.0), "diff_ms"] == pytest.approx(-50.0)
    assert bool(out.loc[("B", 100.0), "violator_at_ssd"]) is False


def test_per_subject_per_ssd_skips_subjects_without_go_data():
    df = pd.DataFrame([
        {"pid": "X", "tt": "stop", "rt": 400, "corr": 0, "ssd": 100},  # no go trials
    ])
    out = compute_per_subject_per_ssd(df, "tt", "rt", "corr", "ssd", subject_col="pid")
    assert out.empty


def test_per_subject_per_ssd_with_condition():
    # Subject A under 'hi' has its own go reference (rt=550) plus the SSD=300 stop
    rows = [
        {"pid": "A", "tt": "go", "rt": 400, "corr": 1, "ssd": np.nan, "load": "lo"},
        {"pid": "A", "tt": "go", "rt": 500, "corr": 1, "ssd": np.nan, "load": "lo"},
        {"pid": "A", "tt": "stop", "rt": 480, "corr": 0, "ssd": 100, "load": "lo"},
        {"pid": "A", "tt": "go", "rt": 550, "corr": 1, "ssd": np.nan, "load": "hi"},
        {"pid": "A", "tt": "stop", "rt": 600, "corr": 0, "ssd": 300, "load": "hi"},
    ]
    df = pd.DataFrame(rows)
    out = compute_per_subject_per_ssd(
        df, "tt", "rt", "corr", "ssd", subject_col="pid", condition_col="load"
    )
    assert "load" in out.columns
    a_hi = out[(out["pid"] == "A") & (out["load"] == "hi")]
    assert len(a_hi) == 1
    assert a_hi["ssd"].iloc[0] == 300.0
    # Within-condition reference: A/hi go mean = 550, failed-stop = 600 → diff +50, violator
    assert a_hi["mean_correct_go_rt"].iloc[0] == pytest.approx(550.0)
    assert a_hi["diff_ms"].iloc[0] == pytest.approx(50.0)
    assert bool(a_hi["violator_at_ssd"].iloc[0]) is True


def test_aggregate_by_ssd():
    df = _make_df()
    per_subject = compute_per_subject_per_ssd(
        df, "tt", "rt", "corr", "ssd", subject_col="pid"
    )
    agg = aggregate_by_ssd(per_subject, ssd_col="ssd")
    agg = agg.set_index("ssd")
    # SSD=100 has 2 subjects: diffs 0 and -50 → mean -25, p_violate = 0
    assert agg.loc[100.0, "n_subjects"] == 2
    assert agg.loc[100.0, "mean_diff_ms"] == pytest.approx(-25.0)
    assert agg.loc[100.0, "p_subjects_violate"] == pytest.approx(0.0)
    # SSD=300 has 1 subject (A): diff 100, p_violate = 1
    assert agg.loc[300.0, "n_subjects"] == 1
    assert agg.loc[300.0, "p_subjects_violate"] == pytest.approx(1.0)


def test_nan_rt_and_negative_rt_excluded():
    df = pd.DataFrame([
        {"pid": "A", "tt": "go", "rt": 500, "corr": 1, "ssd": np.nan},
        {"pid": "A", "tt": "go", "rt": np.nan, "corr": 0, "ssd": np.nan},
        {"pid": "A", "tt": "go", "rt": -1, "corr": 0, "ssd": np.nan},
        {"pid": "A", "tt": "stop", "rt": 600, "corr": 0, "ssd": 100},
        {"pid": "A", "tt": "stop", "rt": np.nan, "corr": 1, "ssd": 100},
    ])
    out = compute_per_subject_per_ssd(df, "tt", "rt", "corr", "ssd", subject_col="pid")
    assert len(out) == 1
    assert out["mean_correct_go_rt"].iloc[0] == pytest.approx(500.0)
    assert out["mean_failed_stop_rt"].iloc[0] == pytest.approx(600.0)


# ============================================================================
# Bissett 2019 paired-preceding diagnostic
# ============================================================================
def _make_preceding_df():
    """Single subject, single block, hand-built so pairs are unambiguous."""
    rows = [
        # SSD 200 has 2 stop fails, each preceded by a valid go:
        # (rt 700, prev 500) -> +200; (rt 520, prev 480) -> +40; mean = +120
        {"pid": "S1", "blk": 0, "ord": 0, "tt": "go",   "rt": 500,    "ssd": np.nan, "corr": 1},
        {"pid": "S1", "blk": 0, "ord": 1, "tt": "stop", "rt": 700,    "ssd": 200,    "corr": 0},
        {"pid": "S1", "blk": 0, "ord": 2, "tt": "go",   "rt": 600,    "ssd": np.nan, "corr": 1},
        # successful stop is ignored
        {"pid": "S1", "blk": 0, "ord": 3, "tt": "stop", "rt": np.nan, "ssd": 200,    "corr": 1},
        {"pid": "S1", "blk": 0, "ord": 4, "tt": "go",   "rt": 480,    "ssd": np.nan, "corr": 1},
        {"pid": "S1", "blk": 0, "ord": 5, "tt": "stop", "rt": 520,    "ssd": 200,    "corr": 0},
        # SSD 400 has 2 stop fails, each preceded by valid go:
        # (350-300)=+50, (380-320)=+60; mean = +55
        {"pid": "S1", "blk": 0, "ord": 6, "tt": "go",   "rt": 300,    "ssd": np.nan, "corr": 1},
        {"pid": "S1", "blk": 0, "ord": 7, "tt": "stop", "rt": 350,    "ssd": 400,    "corr": 0},
        {"pid": "S1", "blk": 0, "ord": 8, "tt": "go",   "rt": 320,    "ssd": np.nan, "corr": 1},
        {"pid": "S1", "blk": 0, "ord": 9, "tt": "stop", "rt": 380,    "ssd": 400,    "corr": 0},
    ]
    return pd.DataFrame(rows)


def test_preceding_basic_known_answer():
    df = _make_preceding_df()
    out = compute_per_subject_per_ssd_preceding(
        df, trial_type_col="tt", rt_col="rt", correct_col="corr",
        ssd_col="ssd", subject_col="pid", block_col="blk",
        trial_order_col="ord",
    )
    out = out.set_index(["pid", "ssd"]).sort_index()
    # SSD 200: pairs (700-500, 520-480) → mean = 120
    assert out.loc[("S1", 200.0), "n_pairs"] == 2
    assert out.loc[("S1", 200.0), "mean_violation_ms"] == pytest.approx(120.0)
    assert out.loc[("S1", 200.0), "mean_stop_failure_rt"] == pytest.approx(610.0)
    assert out.loc[("S1", 200.0), "mean_preceding_go_rt"] == pytest.approx(490.0)
    assert bool(out.loc[("S1", 200.0), "violator_at_ssd"]) is True
    # SSD 400: pairs (350-300, 380-320) → mean = 55
    assert out.loc[("S1", 400.0), "n_pairs"] == 2
    assert out.loc[("S1", 400.0), "mean_violation_ms"] == pytest.approx(55.0)


def test_preceding_drops_pairs_across_block_boundary():
    # The first stop-fail of block 1 is preceded *in the source order* by a
    # go trial in block 0, but cross-block matching is forbidden, so it must
    # be excluded.
    rows = [
        {"pid": "S1", "blk": 0, "ord": 0, "tt": "go",   "rt": 500, "ssd": np.nan, "corr": 1},
        # block boundary
        {"pid": "S1", "blk": 1, "ord": 0, "tt": "stop", "rt": 600, "ssd": 200,    "corr": 0},  # cross-block, drop
        {"pid": "S1", "blk": 1, "ord": 1, "tt": "go",   "rt": 400, "ssd": np.nan, "corr": 1},
        {"pid": "S1", "blk": 1, "ord": 2, "tt": "stop", "rt": 450, "ssd": 200,    "corr": 0},  # diff +50, kept
        {"pid": "S1", "blk": 1, "ord": 3, "tt": "go",   "rt": 410, "ssd": np.nan, "corr": 1},
        {"pid": "S1", "blk": 1, "ord": 4, "tt": "stop", "rt": 470, "ssd": 200,    "corr": 0},  # diff +60, kept
    ]
    df = pd.DataFrame(rows)
    out = compute_per_subject_per_ssd_preceding(
        df, trial_type_col="tt", rt_col="rt", correct_col="corr",
        ssd_col="ssd", subject_col="pid", block_col="blk",
        trial_order_col="ord",
    )
    assert len(out) == 1
    # Two valid pairs (50, 60) -> mean 55. The first stop-fail had no
    # in-block preceding go and is excluded.
    assert out["n_pairs"].iloc[0] == 2
    assert out["mean_violation_ms"].iloc[0] == pytest.approx(55.0)


def test_preceding_min_pairs_filter():
    # Only one valid pair at SSD=200: should be excluded by default
    # (min_pairs_per_subject_ssd=2).
    rows = [
        {"pid": "S1", "blk": 0, "ord": 0, "tt": "go",   "rt": 500, "ssd": np.nan, "corr": 1},
        {"pid": "S1", "blk": 0, "ord": 1, "tt": "stop", "rt": 600, "ssd": 200,    "corr": 0},
    ]
    df = pd.DataFrame(rows)
    out = compute_per_subject_per_ssd_preceding(
        df, trial_type_col="tt", rt_col="rt", correct_col="corr",
        ssd_col="ssd", subject_col="pid", block_col="blk",
        trial_order_col="ord",
    )
    assert out.empty
    # With min_pairs=1 it should be kept
    out2 = compute_per_subject_per_ssd_preceding(
        df, trial_type_col="tt", rt_col="rt", correct_col="corr",
        ssd_col="ssd", subject_col="pid", block_col="blk",
        trial_order_col="ord", min_pairs_per_subject_ssd=1,
    )
    assert len(out2) == 1
    assert out2["mean_violation_ms"].iloc[0] == pytest.approx(100.0)


def test_preceding_skips_when_prev_is_stop_or_omission():
    rows = [
        # Two stop fails in a row at SSD=200: the second has a stop prev,
        # not a go, so it must be excluded.
        {"pid": "S1", "blk": 0, "ord": 0, "tt": "go",   "rt": 500,   "ssd": np.nan, "corr": 1},
        {"pid": "S1", "blk": 0, "ord": 1, "tt": "stop", "rt": 600,   "ssd": 200,    "corr": 0},  # prev=go(500), diff +100
        {"pid": "S1", "blk": 0, "ord": 2, "tt": "stop", "rt": 550,   "ssd": 200,    "corr": 0},  # prev=stop, drop
        # Stop fail preceded by a go omission: prev RT is NaN, drop
        {"pid": "S1", "blk": 0, "ord": 3, "tt": "go",   "rt": np.nan, "ssd": np.nan, "corr": 0},
        {"pid": "S1", "blk": 0, "ord": 4, "tt": "stop", "rt": 700,   "ssd": 200,    "corr": 0},  # prev=go-omission, drop
    ]
    df = pd.DataFrame(rows)
    out = compute_per_subject_per_ssd_preceding(
        df, trial_type_col="tt", rt_col="rt", correct_col="corr",
        ssd_col="ssd", subject_col="pid", block_col="blk",
        trial_order_col="ord", min_pairs_per_subject_ssd=1,
    )
    # Only one valid pair survives
    assert out["n_pairs"].iloc[0] == 1
    assert out["mean_violation_ms"].iloc[0] == pytest.approx(100.0)


def test_aggregate_by_ssd_preceding_min_subjects_filter():
    # Build per-subject input directly: SSD 200 has 2 subjects (kept),
    # SSD 400 has 1 subject (dropped at min_subs=2).
    per = pd.DataFrame([
        {"pid": "A", "ssd": 200.0, "n_pairs": 3, "mean_stop_failure_rt": 600,
         "mean_preceding_go_rt": 550, "mean_violation_ms": 50.0,
         "violator_at_ssd": True},
        {"pid": "B", "ssd": 200.0, "n_pairs": 2, "mean_stop_failure_rt": 500,
         "mean_preceding_go_rt": 520, "mean_violation_ms": -20.0,
         "violator_at_ssd": False},
        {"pid": "A", "ssd": 400.0, "n_pairs": 5, "mean_stop_failure_rt": 700,
         "mean_preceding_go_rt": 700, "mean_violation_ms": 0.0,
         "violator_at_ssd": False},
    ])
    agg = aggregate_by_ssd_preceding(per, ssd_col="ssd", min_subjects_per_ssd=2)
    assert list(agg["ssd"]) == [200.0]
    row = agg.iloc[0]
    assert row["n_subjects"] == 2
    assert row["mean_violation_ms"] == pytest.approx(15.0)  # mean(50, -20)
    assert row["p_subjects_violate"] == pytest.approx(0.5)


# ============================================================================
# SSRT short-vs-long-SSD comparison
# ============================================================================
def test_ssrt_short_vs_long_basic():
    """A trivial SSRT stub lets us verify mean_ssd shifts up when short SSDs
    are excluded — which is the whole point of this comparison."""

    def fake_ssrt(go_rts, n_omissions, p_inhibit, mean_ssd, deadline):
        # Linear in mean_ssd so we can hand-check the diff.
        return 200.0 - mean_ssd

    rows = [
        {"pid": "S1", "tt": "go",   "rt": 500, "ssd": np.nan, "corr": 1, "dl": 1500},
        {"pid": "S1", "tt": "go",   "rt": 480, "ssd": np.nan, "corr": 1, "dl": 1500},
        # 4 stop trials: 2 short (50, 100) and 2 long (250, 350)
        {"pid": "S1", "tt": "stop", "rt": np.nan, "ssd": 50,  "corr": 1, "dl": 1500},
        {"pid": "S1", "tt": "stop", "rt": 600,    "ssd": 100, "corr": 0, "dl": 1500},
        {"pid": "S1", "tt": "stop", "rt": np.nan, "ssd": 250, "corr": 1, "dl": 1500},
        {"pid": "S1", "tt": "stop", "rt": 580,    "ssd": 350, "corr": 0, "dl": 1500},
    ]
    df = pd.DataFrame(rows)
    out = ssrt_short_vs_long_comparison(
        df, ssrt_fn=fake_ssrt,
        trial_type_col="tt", rt_col="rt", correct_col="corr", ssd_col="ssd",
        response_deadline_col="dl",
        subject_col="pid", ssd_threshold_ms=200.0,
    )
    assert len(out) == 1
    row = out.iloc[0]
    # mean_ssd over all stop trials = (50+100+250+350)/4 = 187.5
    # → ssrt_all = 200 - 187.5 = 12.5
    assert row["ssrt_all_ssd"] == pytest.approx(12.5)
    # mean_ssd over SSD>=200 only = (250+350)/2 = 300
    # → ssrt_long = 200 - 300 = -100
    assert row["ssrt_long_ssd"] == pytest.approx(-100.0)
    assert row["ssrt_diff_ms"] == pytest.approx(112.5)
    assert row["n_stop_total"] == 4
    assert row["n_stop_long"] == 2


def test_ssrt_short_vs_long_drops_subject_without_long_pair():
    """A subject who has no successful or no failed stops at SSD>=200 is
    dropped (Bissett 2019 filter_ssrt_subs)."""

    def fake_ssrt(*args, **kwargs):
        return 1.0

    rows = [
        # Subject A has both a long success and a long fail → kept
        {"pid": "A", "tt": "go",   "rt": 500, "ssd": np.nan, "corr": 1, "dl": 1500},
        {"pid": "A", "tt": "stop", "rt": np.nan, "ssd": 250, "corr": 1, "dl": 1500},
        {"pid": "A", "tt": "stop", "rt": 600,    "ssd": 350, "corr": 0, "dl": 1500},
        # Subject B has only successes at SSD>=200 → dropped
        {"pid": "B", "tt": "go",   "rt": 500, "ssd": np.nan, "corr": 1, "dl": 1500},
        {"pid": "B", "tt": "stop", "rt": np.nan, "ssd": 250, "corr": 1, "dl": 1500},
        {"pid": "B", "tt": "stop", "rt": 600,    "ssd": 100, "corr": 0, "dl": 1500},  # short fail
    ]
    df = pd.DataFrame(rows)
    out = ssrt_short_vs_long_comparison(
        df, ssrt_fn=fake_ssrt,
        trial_type_col="tt", rt_col="rt", correct_col="corr", ssd_col="ssd",
        response_deadline_col="dl",
        subject_col="pid", ssd_threshold_ms=200.0,
    )
    assert list(out["pid"]) == ["A"]


def test_ssrt_short_vs_long_with_condition():
    """Condition column is carried through; SSRT computed within condition."""

    def fake_ssrt(go_rts, n_omissions, p_inhibit, mean_ssd, deadline):
        return mean_ssd  # so we can read the per-condition mean SSD directly

    rows = [
        # load=2: stops at 100, 250 → all-mean=175, long-mean=250
        {"pid": "S1", "tt": "go",   "rt": 500, "ssd": np.nan, "corr": 1, "dl": 1500, "load": 2},
        {"pid": "S1", "tt": "stop", "rt": 600, "ssd": 100,    "corr": 0, "dl": 1500, "load": 2},
        {"pid": "S1", "tt": "stop", "rt": np.nan, "ssd": 250, "corr": 1, "dl": 1500, "load": 2},
        {"pid": "S1", "tt": "stop", "rt": 600, "ssd": 350,    "corr": 0, "dl": 1500, "load": 2},
        # load=4: stops at 250, 450 → all-mean=350, long-mean=350
        {"pid": "S1", "tt": "go",   "rt": 480, "ssd": np.nan, "corr": 1, "dl": 1500, "load": 4},
        {"pid": "S1", "tt": "stop", "rt": np.nan, "ssd": 250, "corr": 1, "dl": 1500, "load": 4},
        {"pid": "S1", "tt": "stop", "rt": 700, "ssd": 450,    "corr": 0, "dl": 1500, "load": 4},
    ]
    df = pd.DataFrame(rows)
    out = ssrt_short_vs_long_comparison(
        df, ssrt_fn=fake_ssrt,
        trial_type_col="tt", rt_col="rt", correct_col="corr", ssd_col="ssd",
        response_deadline_col="dl",
        subject_col="pid", condition_col="load", ssd_threshold_ms=200.0,
    )
    out = out.set_index("load").sort_index()
    # load=2: all=(100+250+350)/3≈233.33, long=(250+350)/2=300
    assert out.loc[2, "ssrt_all_ssd"] == pytest.approx((100 + 250 + 350) / 3)
    assert out.loc[2, "ssrt_long_ssd"] == pytest.approx(300.0)
    # load=4: all=(250+450)/2=350, long=(250+450)/2=350 (no short SSDs here)
    assert out.loc[4, "ssrt_all_ssd"] == pytest.approx(350.0)
    assert out.loc[4, "ssrt_long_ssd"] == pytest.approx(350.0)
    assert out.loc[4, "ssrt_diff_ms"] == pytest.approx(0.0)
