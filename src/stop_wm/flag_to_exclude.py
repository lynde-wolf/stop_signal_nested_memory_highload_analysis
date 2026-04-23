"""Flag participants based on exclusion criteria.

This script flags participants to exclude from the analysis and adds 
a column to all_participants_reshaped_data, stop_signal_metrics.csv, 
and stop_signal_wm_task_metrics.csv indicating they have been flagged.
Another column states what exclusion criteria they were flagged for.

Output:
- stop_signal_metrics__flagged.csv
- stop_signal_wm_task_metrics__flagged.csv
"""
# ============================================================================
# Imports
# ============================================================================
from pathlib import Path

import pandas as pd
from scipy import stats

from stop_wm.config import ProjectConfig
from stop_wm.subjectwise_metrics import WM_LOADS


# Per-load derived column names. Defined once from WM_LOADS so that changing
# the load grid in subjectwise_metrics propagates everywhere.
BINOMIAL_METRICS = {'stop_fail_rate'} | {
    f'dual_stop_fail_rate_wm{load}' for load in WM_LOADS
}

# ============================================================================
# Functions
# ============================================================================
def parse_exclusion_criteria(filepath):
    """Parse exclusion criteria from text file.
    
    Args:
        filepath: Path to exclusion_criteria.txt file
    
    Returns:
        Dictionary mapping metric names to min/max bounds
    """
    criteria = {}
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Parse header to get metric names (skip first empty column)
    header_line = lines[0].strip()
    header_parts = [m.strip() for m in header_line.split('|')]
    metrics = [m for m in header_parts if m and m != '']
    
    # Parse min values (line 2)
    min_line = lines[2].strip()
    min_parts = [v.strip() for v in min_line.split('|')]
    min_values = [v for v in min_parts if v and v != '']
    
    # Parse max values (line 3)
    max_line = lines[3].strip()
    max_parts = [v.strip() for v in max_line.split('|')]
    max_values = [v for v in max_parts if v and v != '']
    
    # Build criteria dictionary
    # metrics has column names, min/max_values have 'min'/'max' label as first element
    # So metric index 0 corresponds to min/max index 1, etc.
    for i, metric_name in enumerate(metrics):
        criteria[metric_name] = {
            'min': float(min_values[i + 1]),  # +1 to skip the 'min' label
            'max': float(max_values[i + 1])   # +1 to skip the 'max' label
        }
    
    return criteria


# Mapping from criteria file names to actual CSV column names
# this is to translate from csv to Patrick it's unclean practice but we're keeping it for now
METRIC_NAME_MAPPING = {
    'go_accuracy': 'go_choice_accuracy',
    'stop_accuracy': 'stop_inhibition_success_rate',
    'go_rt': 'go_mean_rt',
    'go_omission_rate': 'go_omission_rate',
    'go_wm': 'dual_task_go_choice_accuracy',
    'stop_wm': 'dual_task_stop_inhibition_success_rate',
    'memory_accuracy': 'probe_response_accuracy',
    'go_wm_omission': 'dual_task_go_omission_rate',
    'completion_date': 'completion_date',
    # Binomial test thresholds for stop trials (derived metrics)
    'stop_fail_rate': 'stop_fail_rate',
}
# Per-load entries (both binomial stop-fail rates and memory accuracy) are
# expanded from WM_LOADS so the load grid is defined in exactly one place.
for _load in WM_LOADS:
    METRIC_NAME_MAPPING[f'dual_stop_fail_rate_wm{_load}'] = f'dual_stop_fail_rate_wm{_load}'
    METRIC_NAME_MAPPING[f'memory_accuracy_wm{_load}'] = f'probe_wm{_load}_response_accuracy'


# This function calculates the binomial rate metrics for the exclusion checking.
def calculate_binomial_rates(df):
    """Calculate derived binomial rate metrics for exclusion checking.
    
    These rates test whether participants' stop trial responses are significantly
    different from chance (p = 0.5), which would indicate they're not following
    the task instructions or the staircase isn't working properly.
    
    Args:
        df: DataFrame with participant metrics
    
    Returns:
        DataFrame with added binomial rate columns for stop trials
    """
    df = df.copy()

    # Stop trial failed inhibition rate (for stop signal task)
    if 'stop_inhibition_success_rate' in df.columns:
        df['stop_fail_rate'] = 1 - df['stop_inhibition_success_rate']

    # Dual task per-load failed-inhibition rates (load-specific, not overall)
    for load in WM_LOADS:
        src = f'dual_task_stop_wm{load}_inhibition_success_rate'
        if src in df.columns:
            df[f'dual_stop_fail_rate_wm{load}'] = 1 - df[src]

    return df


def check_exclusion(row, criteria):
    """Check if a participant should be excluded based on criteria.
    
    Args:
        row: DataFrame row with participant metrics
        criteria: Dictionary of exclusion criteria with min/max values
    
    Returns:
        tuple: (should_exclude, reasons) where reasons is a list of failed criteria
    """
    reasons = []
    
    # Define binomial test metrics for special labeling
    binomial_metrics = BINOMIAL_METRICS
    
    for metric, bounds in criteria.items():
        # Skip if metric not in row (e.g., WM-specific metrics in stop-only data)
        if metric not in row.index:
            continue
            
        value = row[metric]
        
        # Skip NaN values
        if pd.isna(value):
            continue
        
        # Convert date strings to numeric format for comparison (YYYY.MM.DD -> YYYYMMDD)
        if metric == 'completion_date':
            value = float(str(value).replace('.', ''))
        
        # Check if value is outside bounds
        # For metrics where min == max (like trial counts), we want exact matches
        if bounds['min'] == bounds['max']:
            # Exact match required (e.g., n_go_trials should be exactly 120)
            if value != bounds['min']:
                reasons.append(f"{metric} ({value:.3f}) != required ({bounds['min']})")
        else:
            # For binomial test metrics: Use <= and >= (boundaries excluded)
            # For other metrics: Use < and > (boundaries included in acceptable range)
            # This is because go_omission_rate=0.0 is acceptable (perfect performance),
            # but stop_fail_rate at exact boundary suggests deviation
            if metric in binomial_metrics:
                # Strict boundaries for binomial tests
                if value <= bounds['min']:
                    reasons.append(f"{metric} [BINOMIAL TEST] ({value:.3f}) <= min ({bounds['min']})")
                elif value >= bounds['max']:
                    reasons.append(f"{metric} [BINOMIAL TEST] ({value:.3f}) >= max ({bounds['max']})")
            else:
                # Standard boundaries for other metrics
                if value < bounds['min']:
                    reasons.append(f"{metric} ({value:.3f}) < min ({bounds['min']})")
                elif value > bounds['max']:
                    reasons.append(f"{metric} ({value:.3f}) > max ({bounds['max']})")
    
    should_exclude = len(reasons) > 0
    return should_exclude, reasons

# ============================================================================
# Main
# ============================================================================
def main():
    """Flag participants who fail exclusion criteria."""
    
    # Initialize config
    config = ProjectConfig()
    
    # Parse exclusion criteria from file
    criteria_file = Path(__file__).parent / "exclusion_criteria.txt"
    raw_criteria = parse_exclusion_criteria(criteria_file)
    
    # Map criteria names to actual CSV column names
    criteria = {}
    for old_name, bounds in raw_criteria.items():
        new_name = METRIC_NAME_MAPPING.get(old_name, old_name)
        criteria[new_name] = bounds
        
        # Apply memory_accuracy criteria to each WM level
        if old_name == 'memory_accuracy':
            for load in WM_LOADS:
                level_name = f'memory_accuracy_wm{load}'
                criteria[METRIC_NAME_MAPPING[level_name]] = bounds
    
    print(f"\nLoaded exclusion criteria from: {criteria_file}")
    print("\nCriteria:")
    binomial_metrics = BINOMIAL_METRICS
    for metric, bounds in criteria.items():
        if metric in binomial_metrics:
            print(f"  {metric} [BINOMIAL TEST]: {bounds['min']} - {bounds['max']}")
        else:
            print(f"  {metric}: {bounds['min']} - {bounds['max']}")
    
    # Discover metrics files: anything named <task>_metrics.csv (excluding
    # already-flagged outputs and known derivative tables). Files with 'wm'
    # in the task name are treated as WM-task metrics; the rest are simple
    # stop-signal metrics. If multiple files match a category, concatenate.
    metrics_files = [
        p for p in sorted(config.results_dir.glob('*_metrics.csv'))
        if '__flagged' not in p.name and 'post_qc' not in p.name
    ]
    if not metrics_files:
        raise FileNotFoundError(
            f'No *_metrics.csv files found in {config.results_dir}. '
            f'Run `subjectwise_metrics` first.'
        )

    stop_files, wm_files = [], []
    for p in metrics_files:
        task_name = p.stem.replace('_metrics', '')
        (wm_files if 'wm' in task_name.lower() else stop_files).append(p)

    print(f"\nMetrics files discovered in {config.results_dir}:")
    print(f"  stop-signal: {[p.name for p in stop_files] or '(none)'}")
    print(f"  wm-task:     {[p.name for p in wm_files] or '(none)'}")

    if not stop_files or not wm_files:
        raise FileNotFoundError(
            'Need at least one stop-signal metrics file and one WM metrics file; '
            f'found stop={len(stop_files)}, wm={len(wm_files)}.'
        )

    stop_signal_metrics = pd.concat(
        [pd.read_csv(p) for p in stop_files], ignore_index=True,
    )
    stop_signal_wm_metrics = pd.concat(
        [pd.read_csv(p) for p in wm_files], ignore_index=True,
    )

    # Calculate derived binomial rate metrics
    stop_signal_metrics = calculate_binomial_rates(stop_signal_metrics)
    stop_signal_wm_metrics = calculate_binomial_rates(stop_signal_wm_metrics)
    
    # Check for participants who only completed one task
    stop_ids = set(stop_signal_metrics['prolific_id'])
    wm_ids = set(stop_signal_wm_metrics['prolific_id'])
    only_stop = stop_ids - wm_ids
    only_wm = wm_ids - stop_ids
    
    print(f"\n### Task Completion Check ###")
    print(f"Stop Signal participants: {len(stop_ids)}")
    print(f"WM Task participants: {len(wm_ids)}")
    print(f"Participants in both: {len(stop_ids & wm_ids)}")
    if only_stop:
        print(f"\n⚠️  {len(only_stop)} participant(s) completed ONLY Stop Signal task:")
        for pid in sorted(only_stop):
            print(f"   - {pid}")
    if only_wm:
        print(f"\n⚠️  {len(only_wm)} participant(s) completed ONLY WM task:")
        for pid in sorted(only_wm):
            print(f"   - {pid}")
    
    # Create flagged dataframes
    stop_signal_flagged = []
    stop_signal_wm_flagged = []
    
    # Initialize criterion counters
    stop_signal_criterion_counts = {metric: 0 for metric in criteria.keys()}
    wm_criterion_counts = {metric: 0 for metric in criteria.keys()}
    
    print("\n" + "=" * 80) #aesthetic 
    print("FLAGGING PARTICIPANTS FOR EXCLUSION")
    print("=" * 80) #aesthetic 
    

    # Add incomplete task completion to counters
    stop_signal_criterion_counts['incomplete_task'] = 0
    wm_criterion_counts['incomplete_task'] = 0
    
    # Process stop signal data
    print("\n### STOP SIGNAL TASK ###")
    for index, row in stop_signal_metrics.iterrows():
        participant_id = row['prolific_id']
        should_exclude, reasons = check_exclusion(row, criteria)
        
        # Check if participant didn't complete WM task
        if participant_id in only_stop:
            reasons.append("incomplete_task (did not complete WM task)")
            should_exclude = True

        
        if should_exclude:
            print(f"\n⚠️  {participant_id}")
            for reason in reasons:
                print(f"   - {reason}")
                # Extract metric name from reason string (remove [BINOMIAL TEST] label if present)
                metric_name = reason.split('[BINOMIAL TEST]')[0].strip() if '[BINOMIAL TEST]' in reason else reason.split('(')[0].strip()
                if metric_name in stop_signal_criterion_counts:
                    stop_signal_criterion_counts[metric_name] += 1
            
            # Add to flagged list
            flagged_row = row.copy()
            flagged_row['exclusion_reasons'] = '; '.join(reasons)
            stop_signal_flagged.append(flagged_row)
    
    # Process stop signal + WM data
    print("\n\n### STOP SIGNAL + WORKING MEMORY TASK ###")
    for index, row in stop_signal_wm_metrics.iterrows():
        participant_id = row['prolific_id']
        should_exclude, reasons = check_exclusion(row, criteria)
        
        # Check if participant didn't complete Stop Signal task
        if participant_id in only_wm:
            reasons.append("incomplete_task (did not complete Stop Signal task)")
            should_exclude = True
        
        if should_exclude:
            print(f"\n⚠️  {participant_id}")
            for reason in reasons:
                print(f"   - {reason}")
                # Extract metric name from reason string (remove [BINOMIAL TEST] label if present)
                metric_name = reason.split('[BINOMIAL TEST]')[0].strip() if '[BINOMIAL TEST]' in reason else reason.split('(')[0].strip()
                if metric_name in wm_criterion_counts:
                    wm_criterion_counts[metric_name] += 1
            
            # Add to flagged list
            flagged_row = row.copy()
            flagged_row['exclusion_reasons'] = '; '.join(reasons)
            stop_signal_wm_flagged.append(flagged_row)
    
    # Save flagged participants
    if stop_signal_flagged:
        flagged_df = pd.DataFrame(stop_signal_flagged)
        output_path = config.results_dir / "stop_signal_metrics__flagged.csv"
        flagged_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(flagged_df)} flagged stop signal participants to:")
        print(f"  {output_path}")
    else:
        print("\n✓ No stop signal participants flagged for exclusion")
    
    if stop_signal_wm_flagged:
        flagged_df = pd.DataFrame(stop_signal_wm_flagged)
        output_path = config.results_dir / "stop_signal_wm_task_metrics__flagged.csv"
        flagged_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved {len(flagged_df)} flagged WM task participants to:")
        print(f"  {output_path}")
    else:
        print("\n✓ No WM task participants flagged for exclusion")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    ss_total = len(stop_signal_metrics)
    wm_total = len(stop_signal_wm_metrics)
    print(f"\nStop Signal: {len(stop_signal_flagged)}/{ss_total} flagged")
    print(f"WM Task: {len(stop_signal_wm_flagged)}/{wm_total} flagged")
    
    # Print counts by criterion for Stop Signal task
    print("\n### Stop Signal Task - Exclusions by Criterion ###")
    binomial_metrics = BINOMIAL_METRICS
    for metric, count in sorted(stop_signal_criterion_counts.items()):
        if count > 0:
            if metric in binomial_metrics:
                print(f"  {metric} [BINOMIAL TEST]: {count}")
            else:
                print(f"  {metric}: {count}")
    if sum(stop_signal_criterion_counts.values()) == 0:
        print("  (No participants excluded)")
    
    # Print counts by criterion for WM task
    print("\n### WM Task - Exclusions by Criterion ###")
    for metric, count in sorted(wm_criterion_counts.items()):
        if count > 0:
            if metric in binomial_metrics:
                print(f"  {metric} [BINOMIAL TEST]: {count}")
            else:
                print(f"  {metric}: {count}")
    if sum(wm_criterion_counts.values()) == 0:
        print("  (No participants excluded)")
    


if __name__ == "__main__":
    main()
