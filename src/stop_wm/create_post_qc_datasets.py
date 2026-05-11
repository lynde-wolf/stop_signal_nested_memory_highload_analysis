"""
Create post-QC datasets containing only subjects that passed all exclusion criteria.

This module filters the stop signal and stop signal + WM metrics to include only
subjects that are not flagged in either task (i.e., passed quality control).
"""
# ============================================================================
# Imports
# ============================================================================
from pathlib import Path

import pandas as pd

try:
    from .config import ProjectConfig
except ImportError:
    from config import ProjectConfig

# ============================================================================
# Functions
# ============================================================================
def _split_by_wm(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split a list of task-named files into (stop_signal_like, wm_like).

    A file is treated as WM-task if ``'wm'`` appears in its task-name stem.
    Mirrors the routing used by ``subjectwise_metrics`` and ``flag_to_exclude``.
    """
    stop, wm = [], []
    for p in paths:
        (wm if 'wm' in p.stem.lower() else stop).append(p)
    return stop, wm


def load_metrics(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate all discovered ``*_metrics.csv`` files.

    Files with ``wm`` in the task name go into the WM DataFrame; everything
    else goes into the simple-stop DataFrame. Already-flagged and post_qc
    outputs are excluded.
    """
    candidates = [
        p for p in sorted(results_dir.glob('*_metrics.csv'))
        if '__flagged' not in p.name and 'post_qc' not in p.name
    ]
    stop_files, wm_files = _split_by_wm(candidates)

    if not stop_files or not wm_files:
        raise FileNotFoundError(
            f'Need at least one stop-signal and one WM metrics file in '
            f'{results_dir}; found stop={[p.name for p in stop_files]}, '
            f'wm={[p.name for p in wm_files]}.'
        )

    stop_metrics = pd.concat([pd.read_csv(p) for p in stop_files], ignore_index=True)
    wm_metrics = pd.concat([pd.read_csv(p) for p in wm_files], ignore_index=True)

    # A subject who completed the experiment under both the pilot exp_ids
    # (stop_signal / stop_signal_wm_task) and the current E3 exp_ids will
    # appear in both files. Keep only their most recent attempt per task.
    def _dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
        if 'completion_date' in df.columns:
            df = df.sort_values('completion_date')
        return df.drop_duplicates('prolific_id', keep='last').reset_index(drop=True)

    stop_metrics = _dedupe_latest(stop_metrics)
    wm_metrics = _dedupe_latest(wm_metrics)

    print(f"Loaded {len(stop_metrics)} subjects from "
          f"{[p.name for p in stop_files]}")
    print(f"Loaded {len(wm_metrics)} subjects from "
          f"{[p.name for p in wm_files]}")

    return stop_metrics, wm_metrics


def load_trial_data(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and concatenate ``all_participants_reshaped_data_*.csv`` files.

    Same WM-vs-stop routing as ``load_metrics``.
    """
    candidates = sorted(results_dir.glob('all_participants_reshaped_data_*.csv'))
    stop_files, wm_files = _split_by_wm(candidates)

    if not stop_files or not wm_files:
        raise FileNotFoundError(
            f'Need at least one stop-signal and one WM combined trial file in '
            f'{results_dir}; found stop={[p.name for p in stop_files]}, '
            f'wm={[p.name for p in wm_files]}.'
        )

    def _concat_with_source(files):
        parts = []
        for p in files:
            df = pd.read_csv(p, low_memory=False)
            df['_source_file'] = p.name
            parts.append(df)
        return pd.concat(parts, ignore_index=True)

    stop_trials = _concat_with_source(stop_files)
    wm_trials = _concat_with_source(wm_files)

    # If a subject has trials from both the pilot exp_ids and the current E3
    # exp_ids, keep only trials from their most-recent source file (E3
    # filenames sort after the pilot files, so we keep the last group per
    # participant_id).
    def _dedupe_by_latest_source(df: pd.DataFrame) -> pd.DataFrame:
        latest_source = (df.groupby('participant_id')['_source_file']
                           .agg(lambda s: sorted(s.unique())[-1]))
        keep_mask = df.apply(
            lambda r: r['_source_file'] == latest_source[r['participant_id']],
            axis=1,
        )
        return df[keep_mask].drop(columns='_source_file').reset_index(drop=True)

    stop_trials = _dedupe_by_latest_source(stop_trials)
    wm_trials = _dedupe_by_latest_source(wm_trials)

    print(f"Loaded {len(stop_trials)} trials from "
          f"{[p.name for p in stop_files]}")
    print(f"Loaded {len(wm_trials)} trials from "
          f"{[p.name for p in wm_files]}")

    return stop_trials, wm_trials


def load_flagged_subjects(results_dir: Path) -> set[str]:
    """
    Load all flagged subjects from both flagged CSV files.

    Args:
        results_dir: Path to the results directory

    Returns:
        Set of prolific_ids that are flagged
    """
    stop_flagged_path = results_dir / 'stop_signal_metrics__flagged.csv'
    wm_flagged_path = results_dir / 'stop_signal_wm_task_metrics__flagged.csv'

    flagged_subjects = set()

    # Load stop signal flagged subjects
    if stop_flagged_path.exists():
        stop_flagged = pd.read_csv(stop_flagged_path)
        stop_flagged_ids = set(stop_flagged['prolific_id'].astype(str))
        flagged_subjects.update(stop_flagged_ids)
        print(
            f"Found {len(stop_flagged_ids)} flagged subjects in "
            "stop_signal_metrics__flagged.csv"
        )

    # Load WM flagged subjects
    if wm_flagged_path.exists():
        wm_flagged = pd.read_csv(wm_flagged_path)
        wm_flagged_ids = set(wm_flagged['prolific_id'].astype(str))
        flagged_subjects.update(wm_flagged_ids)
        print(
            f"Found {len(wm_flagged_ids)} flagged subjects in "
            "stop_signal_wm_task_metrics__flagged.csv"
        )

    print(f"Total unique flagged subjects: {len(flagged_subjects)}")

    return flagged_subjects


def filter_post_qc_subjects(
    stop_metrics: pd.DataFrame,
    wm_metrics: pd.DataFrame,
    flagged_subjects: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """
    Filter metrics to only include subjects that passed QC.

    Args:
        stop_metrics: Full stop signal metrics DataFrame
        wm_metrics: Full WM task metrics DataFrame
        flagged_subjects: Set of flagged prolific_ids

    Returns:
        Tuple of (post_qc_stop_metrics, post_qc_wm_metrics, post_qc_subjects)
    """
    # Convert prolific_id to string for consistent comparison
    stop_metrics['prolific_id'] = stop_metrics['prolific_id'].astype(str)
    wm_metrics['prolific_id'] = wm_metrics['prolific_id'].astype(str)

    # Get subjects that have data in both tasks
    stop_subjects = set(stop_metrics['prolific_id'])
    wm_subjects = set(wm_metrics['prolific_id'])
    subjects_in_both = stop_subjects.intersection(wm_subjects)

    print(f"\nSubjects with stop signal data: {len(stop_subjects)}")
    print(f"Subjects with WM data: {len(wm_subjects)}")
    print(f"Subjects in both tasks: {len(subjects_in_both)}")

    # Find subjects that passed QC (in both tasks and not flagged)
    post_qc_subjects = subjects_in_both - flagged_subjects

    print(f"Subjects that passed QC: {len(post_qc_subjects)}")

    # Filter the DataFrames
    post_qc_subjects_list = list(post_qc_subjects)
    post_qc_stop = stop_metrics[
        stop_metrics['prolific_id'].isin(post_qc_subjects_list)
    ].copy()
    post_qc_wm = wm_metrics[
        wm_metrics['prolific_id'].isin(post_qc_subjects_list)
    ].copy()

    # Sort by prolific_id for consistency
    post_qc_stop = post_qc_stop.sort_values(
        by='prolific_id'
    ).reset_index(drop=True)
    post_qc_wm = post_qc_wm.sort_values(
        by='prolific_id'
    ).reset_index(drop=True)

    return post_qc_stop, post_qc_wm, post_qc_subjects


def filter_post_qc_trials(
    stop_trials: pd.DataFrame,
    wm_trials: pd.DataFrame,
    post_qc_subjects: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter trial-level data to only include subjects that passed QC.

    Args:
        stop_trials: Full stop signal trial-level DataFrame
        wm_trials: Full WM task trial-level DataFrame
        post_qc_subjects: Set of prolific_ids that passed QC

    Returns:
        Tuple of (post_qc_stop_trials, post_qc_wm_trials) DataFrames
    """
    # Convert participant_id to string for consistent comparison
    stop_trials['participant_id'] = stop_trials['participant_id'].astype(str)
    wm_trials['participant_id'] = wm_trials['participant_id'].astype(str)

    # Filter trials to only include post-QC participants
    post_qc_subjects_list = list(post_qc_subjects)
    post_qc_stop_trials = stop_trials[
        stop_trials['participant_id'].isin(post_qc_subjects_list)
    ].copy()
    post_qc_wm_trials = wm_trials[
        wm_trials['participant_id'].isin(post_qc_subjects_list)
    ].copy()

    # Calculate removed trials
    n_removed_stop = len(stop_trials) - len(post_qc_stop_trials)
    n_removed_wm = len(wm_trials) - len(post_qc_wm_trials)

    print("\nTotal trials before QC filtering:")
    print(f"  - Stop signal: {len(stop_trials)}")
    print(f"  - WM task: {len(wm_trials)}")
    print("\nTotal trials after QC filtering:")
    print(f"  - Stop signal: {len(post_qc_stop_trials)}")
    print(f"  - WM task: {len(post_qc_wm_trials)}")
    print("\nTrials removed:")
    print(f"  - Stop signal: {n_removed_stop}")
    print(f"  - WM task: {n_removed_wm}")

    return post_qc_stop_trials, post_qc_wm_trials


def save_post_qc_datasets(
    post_qc_stop: pd.DataFrame,
    post_qc_wm: pd.DataFrame,
    post_qc_stop_trials: pd.DataFrame,
    post_qc_wm_trials: pd.DataFrame,
    results_dir: Path,
) -> None:
    """
    Save the post-QC datasets to CSV files.

    Args:
        post_qc_stop: Post-QC stop signal metrics DataFrame
        post_qc_wm: Post-QC WM task metrics DataFrame
        post_qc_stop_trials: Post-QC stop signal trial-level DataFrame
        post_qc_wm_trials: Post-QC WM task trial-level DataFrame
        results_dir: Path to the results directory
    """
    # Save metrics
    stop_output_path = results_dir / 'post_qc_stop_signal_metrics.csv'
    wm_output_path = results_dir / 'post_qc_stop_signal_wm_metrics.csv'

    post_qc_stop.to_csv(stop_output_path, index=False)
    post_qc_wm.to_csv(wm_output_path, index=False)

    # Save trial-level data
    stop_trials_output_path = results_dir / 'post_qc_stop_signal_trials.csv'
    wm_trials_output_path = results_dir / 'post_qc_stop_signal_wm_trials.csv'

    post_qc_stop_trials.to_csv(stop_trials_output_path, index=False)
    post_qc_wm_trials.to_csv(wm_trials_output_path, index=False)

    print("\nSaved post-QC datasets:")
    print("  Metrics:")
    print(f"    - {stop_output_path} ({len(post_qc_stop)} subjects)")
    print(f"    - {wm_output_path} ({len(post_qc_wm)} subjects)")
    print("  Trial-level data:")
    print(f"    - {stop_trials_output_path} ({len(post_qc_stop_trials)} trials)")
    print(f"    - {wm_trials_output_path} ({len(post_qc_wm_trials)} trials)")


def create_post_qc_datasets() -> None:
    """
    Main function to create post-QC datasets.

    This function:
    1. Loads the full metrics from both tasks
    2. Loads the trial-level data from both tasks
    3. Identifies all flagged subjects from both flagged CSVs
    4. Filters to only include subjects that:
       - Have data in both tasks
       - Are not flagged in either task
    5. Filters trial-level data to only include post-QC subjects
    6. Saves the filtered datasets as:
       - post_qc_stop_signal_metrics.csv
       - post_qc_stop_signal_wm_metrics.csv
       - post_qc_stop_signal_trials.csv
       - post_qc_stop_signal_wm_trials.csv
    """
    print("=" * 80)
    print("Creating Post-QC Datasets")
    print("=" * 80)

    # Initialize configuration
    config = ProjectConfig()

    # Load data
    print("\nStep 1: Loading metrics...")
    stop_metrics, wm_metrics = load_metrics(config.results_dir)

    print("\nStep 2: Loading trial-level data...")
    stop_trials, wm_trials = load_trial_data(config.results_dir)

    print("\nStep 3: Loading flagged subjects...")
    flagged_subjects = load_flagged_subjects(config.results_dir)

    print("\nStep 4: Filtering to post-QC subjects...")
    post_qc_stop, post_qc_wm, post_qc_subjects = filter_post_qc_subjects(
        stop_metrics, wm_metrics, flagged_subjects
    )

    print("\nStep 5: Filtering trial-level data to post-QC subjects...")
    post_qc_stop_trials, post_qc_wm_trials = filter_post_qc_trials(
        stop_trials, wm_trials, post_qc_subjects
    )

    print("\nStep 6: Saving post-QC datasets...")
    save_post_qc_datasets(
        post_qc_stop, post_qc_wm,
        post_qc_stop_trials, post_qc_wm_trials,
        config.results_dir
    )

    print("\n" + "=" * 80)
    print("Post-QC datasets created successfully!")
    print("=" * 80)


def main() -> None:
    """CLI entry-point."""
    create_post_qc_datasets()


# ============================================================================
# Main
# ============================================================================
if __name__ == '__main__':
    main()

