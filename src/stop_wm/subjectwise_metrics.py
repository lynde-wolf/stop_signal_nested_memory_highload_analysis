# This script calculates the subjectwise metrics for the stop+wm experiment
# calls reshaped data csv, groups by prolific_id, and calculates the metrics
# the output is a csv where row is a prolific_id and the columns are the metrics

# Core metrics:
# prolific_id,
# completion_date,
# go_accuracy,
# go_mean_correct_rt,
# go_omission_rate,
# stop_success_rate,
# stop_fail_mean_rt,
# min_SSD,max_SSD,mean_SSD,final_SSD

# For the WM block each load level gets the same stats plus:
# wm_rt_<load>, wm_omission_rate_<load>, wm_accuracy_<load>.
# Column names embed the actual load value (e.g. wm2, wm4, wm6 for E3).


# ============================================================================
# Imports
# ============================================================================

from datetime import datetime

import numpy as np
import pandas as pd

from stop_wm.config import ProjectConfig

# Initialize config
config = ProjectConfig()

# WM set sizes for this experiment. Downstream modules (flag_to_exclude,
# calculate_binomial_thresholds) import this so the load grid is defined
# in exactly one place. E1/E2 used [0, 2, 4]; E3 uses [2, 4, 6].
WM_LOADS = [2, 4, 6]

# ============================================================================
# Functions
# ============================================================================
def calculate_ssrt_integration(go_rts, go_omission_count, stop_success_rate, mean_ssd, 
                                response_deadline):
    """
    Calculate SSRT using the integration method with replacement of go omissions.
    
    The integration method:
    1. Takes all Go trial RTs and replaces omissions with the response deadline
    2. Finds the nth percentile of this distribution, where n = P(respond|signal) * 100
    3. SSRT = nth percentile Go RT - mean SSD
    
    Args:
        go_rts: Series/array of Go trial RTs (may include NaN for omissions)
        go_omission_count: Number of go omissions (trials with no response)
        stop_success_rate: Proportion of successful stop trials (P(inhibit))
        mean_ssd: Mean stop signal delay
        response_deadline: Maximum RT allowed (trial duration from task data)
    
    Returns:
        SSRT calculated via integration method, or NaN if cannot be calculated
    
    References:
        Verbruggen, F., et al. (2019). A consensus guide to capturing the ability 
        to inhibit actions and impulsive behaviors in the stop-signal task.
        eLife, 8, e46323.
    """
    if np.isnan(mean_ssd) or np.isnan(stop_success_rate) or np.isnan(response_deadline):
        return np.nan
    
    # Get valid (non-NaN) Go RTs
    valid_go_rts = go_rts.dropna() if hasattr(go_rts, 'dropna') else go_rts[~np.isnan(go_rts)]
    
    if len(valid_go_rts) == 0:
        return np.nan
    
    # Calculate omission rate from go_omission_count and total go trials
    total_go_trials = len(valid_go_rts) + go_omission_count
    omission_rate = go_omission_count / total_go_trials if total_go_trials > 0 else 0
    
    # P(respond|signal) = 1 - P(inhibit) = probability of responding on stop trials
    p_respond = 1 - stop_success_rate
    
    # Correct P(respond) for go omission rate
    # This accounts for trials where participants may not have been attending
    if omission_rate >= 1:
        return np.nan
    corrected_p_respond = p_respond / (1 - omission_rate)
    
    # Handle edge cases where SSRT cannot be estimated
    if corrected_p_respond <= 0 or corrected_p_respond >= 1:
        return np.nan
    
    # Create RT distribution with omissions replaced by response deadline
    replacement_rts = np.full(go_omission_count, response_deadline)
    all_rts = np.concatenate([valid_go_rts.values, replacement_rts])
    
    # Sort the RT distribution
    sorted_rts = np.sort(all_rts)
    
    # Find the nth percentile where n = corrected P(respond|signal) * 100
    # This is the RT at which corrected P(respond|signal) proportion of Go responses have occurred
    nth_rt = np.percentile(sorted_rts, corrected_p_respond * 100)
    
    # SSRT = nth percentile Go RT - mean SSD
    ssrt = nth_rt - mean_ssd
    
    return ssrt


def calculate_stop_signal_metrics(group, participant_id=None):
    """Calculate the metrics for the stop signal task experiment."""
    # Extract participant info
    if participant_id is None:
        participant_id = group['participant_id'].iloc[0]
    
    # Extract completion date from filename pattern
    # The reshaped data doesn't have date, so we'll need to get it from somewhere else
    # For now, use participant_id as placeholder (will be updated later)
    completion_date = participant_id

    # Go trial metrics
    go_trials = group[group['trial_SS_trial_type'] == 'go']
    go_choice_correct = go_trials[go_trials['trial_correct_trial'] == 1.0] #note, a successful go trial is one where the choice response matched the correct go response
    go_no_response = go_trials[go_trials['trial_rt'].isna()]  # Omissions
    go_commission = go_trials[(go_trials['trial_correct_trial'] == 0) & (go_trials['trial_rt'].notna())]  # Commission errors (wrong key pressed)
    
    # Stop trial metrics
    stop_trials = group[group['trial_SS_trial_type'] == 'stop']
    stop_inhibition_success = stop_trials[stop_trials['trial_correct_trial'] == 1]  # Successful stops (no response)
    stop_failed_inhibition = stop_trials[stop_trials['trial_correct_trial'] == 0]  # Failed stops (any response)
    
    number_of_go_trials = go_trials.shape[0]
    number_of_go_correct_trials = go_choice_correct.shape[0]
    number_of_go_omission_trials = go_no_response.shape[0]
    number_of_go_commission_trials = go_commission.shape[0]
    number_of_stop_trials = stop_trials.shape[0]
    print(f"number of stop trials: {number_of_stop_trials}")
    number_of_successful_stop_trials = stop_inhibition_success.shape[0]
    print(f"number of successful stop trials: {number_of_successful_stop_trials}")
    number_of_stop_failed_inhibition = stop_failed_inhibition.shape[0]

    # Calculate metrics using the same style as analyze.py
    go_accuracy = (number_of_go_correct_trials / number_of_go_trials) if number_of_go_trials > 0 else np.nan
    go_mean_correct_rt = go_choice_correct['trial_rt'].mean() if len(go_choice_correct) > 0 else np.nan 
    go_omission_rate = (number_of_go_omission_trials / number_of_go_trials) if number_of_go_trials > 0 else np.nan
    go_commission_rate = (number_of_go_commission_trials / number_of_go_trials) if number_of_go_trials > 0 else np.nan
    stop_success_rate = (number_of_successful_stop_trials / number_of_stop_trials) if number_of_stop_trials > 0 else np.nan
    stop_fail_mean_rt = stop_failed_inhibition['trial_rt'].mean() if len(stop_failed_inhibition) > 0 else np.nan

    # SSD metrics
    ssd_values = group['trial_SSD'].dropna()
    min_ssd = ssd_values.min() if len(ssd_values) > 0 else np.nan
    max_ssd = ssd_values.max() if len(ssd_values) > 0 else np.nan
    mean_ssd = ssd_values.mean() if len(ssd_values) > 0 else np.nan
    final_ssd = ssd_values.iloc[-1] if len(ssd_values) > 0 else np.nan

    # SSRT (Stop Signal Reaction Time) - Integration method with replacement
    # Uses all Go trial RTs, replacing omissions with response deadline
    # Response deadline is extracted from trial_trial_duration in the task data
    go_rts_for_ssrt = go_trials['trial_rt']
    response_deadline = group['trial_trial_duration'].iloc[0]
    ssrt = calculate_ssrt_integration(
        go_rts=go_rts_for_ssrt,
        go_omission_count=number_of_go_omission_trials,
        stop_success_rate=stop_success_rate,
        mean_ssd=mean_ssd,
        response_deadline=response_deadline
    )

    return pd.Series({
        'prolific_id': participant_id,
        'completion_date': participant_id,
        'go_choice_accuracy': go_accuracy,
        'go_mean_rt': go_mean_correct_rt,
        'go_omission_rate': go_omission_rate,
        'go_commission_rate': go_commission_rate,
        'n_go_trials': number_of_go_trials,
        'n_go_omissions': number_of_go_omission_trials,
        'n_go_commissions': number_of_go_commission_trials,
        'stop_inhibition_success_rate': stop_success_rate,
        'stop_failed_mean_rt': stop_fail_mean_rt,
        'n_stop_trials': number_of_stop_trials,
        'min_SSD': min_ssd,
        'max_SSD': max_ssd,
        'mean_SSD': mean_ssd,
        'final_SSD': final_ssd,
        'SSRT': ssrt
    })


def calculate_wm_metrics(group, participant_id=None):
    """Calculate the metrics for the working memory task experiment."""
    if participant_id is None:
        participant_id = group['participant_id'].iloc[0]

    # Overall dual task metrics (all trials aggregated)
    dual_task_go_trials = group[group['stop_trial_SS_trial_type'] == 'go']
    dual_task_go_choice_correct = dual_task_go_trials[
        dual_task_go_trials['stop_trial_correct_trial'] == 1.0]
    dual_task_go_no_response = dual_task_go_trials[
        dual_task_go_trials['stop_trial_rt'].isna()]  # Omissions
    dual_task_go_commission = dual_task_go_trials[
        (dual_task_go_trials['stop_trial_correct_trial'] == 0) & 
        (dual_task_go_trials['stop_trial_rt'].notna())]  # Commission errors (wrong key pressed)
    
    # Overall dual task stop trials
    dual_task_stop_trials = group[group['stop_trial_SS_trial_type'] == 'stop']
    dual_task_stop_inhibition_success = dual_task_stop_trials[
        dual_task_stop_trials['stop_trial_correct_trial'] == 1]  # Successful stops (no response)
    dual_task_stop_failed_inhibition = dual_task_stop_trials[
        dual_task_stop_trials['stop_trial_correct_trial'] == 0]  # Failed stops (any response)
    
    number_of_dual_task_go_trials = dual_task_go_trials.shape[0]
    number_of_dual_task_go_correct_trials = dual_task_go_choice_correct.shape[0]
    number_of_dual_task_go_omission_trials = dual_task_go_no_response.shape[0]
    number_of_dual_task_go_commission_trials = dual_task_go_commission.shape[0]
    number_of_dual_task_stop_trials = dual_task_stop_trials.shape[0]
    number_of_dual_task_successful_stop_trials = dual_task_stop_inhibition_success.shape[0]
    number_of_dual_task_stop_failed_inhibition = dual_task_stop_failed_inhibition.shape[0]

    # Calculate metrics using the same style as analyze.py
    dual_task_go_choice_accuracy = (number_of_dual_task_go_correct_trials / number_of_dual_task_go_trials) if number_of_dual_task_go_trials > 0 else np.nan
    dual_task_go_mean_rt = dual_task_go_choice_correct['stop_trial_rt'].mean() if len(dual_task_go_choice_correct) > 0 else np.nan
    dual_task_go_omission_rate = (number_of_dual_task_go_omission_trials / number_of_dual_task_go_trials) if number_of_dual_task_go_trials > 0 else np.nan
    dual_task_go_commission_rate = (number_of_dual_task_go_commission_trials / number_of_dual_task_go_trials) if number_of_dual_task_go_trials > 0 else np.nan
    
    dual_task_stop_inhibition_success_rate = (number_of_dual_task_successful_stop_trials / number_of_dual_task_stop_trials) if number_of_dual_task_stop_trials > 0 else np.nan
    dual_task_stop_failed_mean_rt = dual_task_stop_failed_inhibition['stop_trial_rt'].mean() if len(dual_task_stop_failed_inhibition) > 0 else np.nan

    # Working memory load specific metrics
    wm_metrics = {}
    for wm_load in WM_LOADS:
        wm_trials = group[group['memory_trial_stimLength'] == wm_load]

        # Dual task go trials for this WM load
        dual_task_go_wm_trials = wm_trials[
            wm_trials['stop_trial_SS_trial_type'] == 'go']
        dual_task_go_wm_choice_correct = dual_task_go_wm_trials[
            dual_task_go_wm_trials['stop_trial_correct_trial'] == 1.0]
        dual_task_go_wm_no_response = dual_task_go_wm_trials[
            dual_task_go_wm_trials['stop_trial_rt'].isna()]  # Omissions
        dual_task_go_wm_commission = dual_task_go_wm_trials[
            (dual_task_go_wm_trials['stop_trial_correct_trial'] == 0) & 
            (dual_task_go_wm_trials['stop_trial_rt'].notna())]  # Commission errors (wrong key pressed)

        # Dual task stop trials for this WM load
        dual_task_stop_wm_trials = wm_trials[
            wm_trials['stop_trial_SS_trial_type'] == 'stop']
        dual_task_stop_wm_inhibition_success = dual_task_stop_wm_trials[
            dual_task_stop_wm_trials['stop_trial_correct_trial'] == 1]  # Successful stops (no response)
        dual_task_stop_wm_failed_inhibition = dual_task_stop_wm_trials[
            dual_task_stop_wm_trials['stop_trial_correct_trial'] == 0]  # Failed stops (any response)
        # Failed stops where the choice response matched the correct go response
        dual_task_stop_wm_failed_choice_correct = dual_task_stop_wm_trials[
            (dual_task_stop_wm_trials['stop_trial_correct_trial'] == 0) & 
            (dual_task_stop_wm_trials['stop_trial_response'] == dual_task_stop_wm_trials['stop_trial_correct_response'])
        ]  # Failed stops with correct choice
        
        number_of_dual_task_go_wm_trials = dual_task_go_wm_trials.shape[0]
        number_of_dual_task_go_wm_correct_trials = dual_task_go_wm_choice_correct.shape[0]
        number_of_dual_task_go_wm_omission_trials = dual_task_go_wm_no_response.shape[0]
        number_of_dual_task_go_wm_commission_trials = dual_task_go_wm_commission.shape[0]
        number_of_dual_task_stop_wm_trials = dual_task_stop_wm_trials.shape[0]
        number_of_dual_task_successful_stop_wm_trials = dual_task_stop_wm_inhibition_success.shape[0]
        number_of_dual_task_stop_wm_failed_inhibition = dual_task_stop_wm_failed_inhibition.shape[0]
        number_of_dual_task_stop_wm_failed_choice_correct = dual_task_stop_wm_failed_choice_correct.shape[0]

        # Calculate metrics using the same style
        wm_metrics[f'dual_task_go_wm{wm_load}_choice_accuracy'] = (
            number_of_dual_task_go_wm_correct_trials / number_of_dual_task_go_wm_trials
            if number_of_dual_task_go_wm_trials > 0 else np.nan)
        wm_metrics[f'dual_task_go_wm{wm_load}_mean_rt'] = (
            dual_task_go_wm_choice_correct['stop_trial_rt'].mean()
            if len(dual_task_go_wm_choice_correct) > 0 else np.nan)
        wm_metrics[f'dual_task_go_wm{wm_load}_omission_rate'] = (
            number_of_dual_task_go_wm_omission_trials / number_of_dual_task_go_wm_trials
            if number_of_dual_task_go_wm_trials > 0 else np.nan)
        wm_metrics[f'dual_task_go_wm{wm_load}_commission_rate'] = (
            number_of_dual_task_go_wm_commission_trials / number_of_dual_task_go_wm_trials
            if number_of_dual_task_go_wm_trials > 0 else np.nan)
        wm_metrics[f'dual_task_go_wm{wm_load}_n_trials'] = number_of_dual_task_go_wm_trials
        wm_metrics[f'dual_task_go_wm{wm_load}_n_omissions'] = number_of_dual_task_go_wm_omission_trials
        wm_metrics[f'dual_task_go_wm{wm_load}_n_commissions'] = number_of_dual_task_go_wm_commission_trials

        wm_metrics[f'dual_task_stop_wm{wm_load}_inhibition_success_rate'] = (
            number_of_dual_task_successful_stop_wm_trials / number_of_dual_task_stop_wm_trials
            if number_of_dual_task_stop_wm_trials > 0 else np.nan)
        wm_metrics[f'dual_task_stop_wm{wm_load}_failed_choice_accuracy'] = (
            number_of_dual_task_stop_wm_failed_choice_correct / number_of_dual_task_stop_wm_failed_inhibition
            if number_of_dual_task_stop_wm_failed_inhibition > 0 else np.nan)
        wm_metrics[f'dual_task_stop_wm{wm_load}_failed_mean_rt'] = (
            dual_task_stop_wm_failed_inhibition['stop_trial_rt'].mean()
            if len(dual_task_stop_wm_failed_inhibition) > 0 else np.nan)
        wm_metrics[f'dual_task_stop_wm{wm_load}_n_trials'] = number_of_dual_task_stop_wm_trials

        # SSD metrics for this WM load
        wm_ssd_values = wm_trials['stop_trial_SSD'].dropna()
        wm_metrics[f'min_SSD_wm{wm_load}'] = (
            wm_ssd_values.min() if len(wm_ssd_values) > 0 else np.nan)
        wm_metrics[f'max_SSD_wm{wm_load}'] = (
            wm_ssd_values.max() if len(wm_ssd_values) > 0 else np.nan)
        wm_metrics[f'mean_SSD_wm{wm_load}'] = (
            wm_ssd_values.mean() if len(wm_ssd_values) > 0 else np.nan)
        wm_metrics[f'final_SSD_wm{wm_load}'] = (
            wm_ssd_values.iloc[-1] if len(wm_ssd_values) > 0 else np.nan)
        
        # SSRT for this WM load - Integration method with replacement
        # Response deadline extracted from stop_trial_trial_duration in task data
        wm_go_rts = dual_task_go_wm_trials['stop_trial_rt']
        wm_stop_success_rate = wm_metrics[f'dual_task_stop_wm{wm_load}_inhibition_success_rate']
        wm_mean_ssd = wm_metrics[f'mean_SSD_wm{wm_load}']
        wm_response_deadline = wm_trials['stop_trial_trial_duration'].iloc[0] if len(wm_trials) > 0 else np.nan
        wm_metrics[f'SSRT_wm{wm_load}'] = calculate_ssrt_integration(
            go_rts=wm_go_rts,
            go_omission_count=number_of_dual_task_go_wm_omission_trials,
            stop_success_rate=wm_stop_success_rate,
            mean_ssd=wm_mean_ssd,
            response_deadline=wm_response_deadline
        )

    # Probe response metrics (include all probe trial types)
    probe_trials = group[group['memory_recognition_condition'].isin(
        ['in memory set', 'not in memory set', 'no memory set'])]
    probe_choice_correct = probe_trials[
        probe_trials['memory_recognition_correct_trial'] == 1.0]

    number_of_probe_trials = probe_trials.shape[0]
    number_of_probe_correct_trials = probe_choice_correct.shape[0]

    # Calculate metrics using the same style as analyze.py
    probe_response_accuracy = (number_of_probe_correct_trials / number_of_probe_trials
                               if number_of_probe_trials > 0 else np.nan)
    probe_mean_rt = (probe_choice_correct['memory_recognition_rt'].mean()
                     if len(probe_choice_correct) > 0 else np.nan)

    # =========================================================================
    # Probe accuracy conditional on stop signal outcome
    # =========================================================================
    # Trials with successful stops (inhibition success)
    successful_stop_trials = group[
        (group['stop_trial_SS_trial_type'] == 'stop') & 
        (group['stop_trial_correct_trial'] == 1)]
    # Trials with failed stops (inhibition failure)
    failed_stop_trials = group[
        (group['stop_trial_SS_trial_type'] == 'stop') & 
        (group['stop_trial_correct_trial'] == 0)]
    # Go trials (no stop signal)
    go_trials_for_probe = group[group['stop_trial_SS_trial_type'] == 'go']
    
    # Overall probe accuracy on successful stop trials
    successful_stop_probe_correct = successful_stop_trials[
        successful_stop_trials['memory_recognition_correct_trial'] == 1.0]
    n_successful_stop_trials = successful_stop_trials.shape[0]
    n_successful_stop_probe_correct = successful_stop_probe_correct.shape[0]
    wm_metrics['probe_accuracy_on_successful_stop'] = (
        n_successful_stop_probe_correct / n_successful_stop_trials
        if n_successful_stop_trials > 0 else np.nan)
    wm_metrics['n_successful_stop_trials_with_probe'] = n_successful_stop_trials
    
    # Overall probe accuracy on failed stop trials
    failed_stop_probe_correct = failed_stop_trials[
        failed_stop_trials['memory_recognition_correct_trial'] == 1.0]
    n_failed_stop_trials = failed_stop_trials.shape[0]
    n_failed_stop_probe_correct = failed_stop_probe_correct.shape[0]
    wm_metrics['probe_accuracy_on_failed_stop'] = (
        n_failed_stop_probe_correct / n_failed_stop_trials
        if n_failed_stop_trials > 0 else np.nan)
    wm_metrics['n_failed_stop_trials_with_probe'] = n_failed_stop_trials
    
    # Overall probe accuracy on go trials (no stop signal)
    go_probe_correct = go_trials_for_probe[
        go_trials_for_probe['memory_recognition_correct_trial'] == 1.0]
    n_go_trials_for_probe = go_trials_for_probe.shape[0]
    n_go_probe_correct = go_probe_correct.shape[0]
    wm_metrics['probe_accuracy_on_go_trials'] = (
        n_go_probe_correct / n_go_trials_for_probe
        if n_go_trials_for_probe > 0 else np.nan)
    wm_metrics['n_go_trials_with_probe'] = n_go_trials_for_probe

    # Probe metrics by WM load
    for wm_load in WM_LOADS:
        wm_probe_trials = probe_trials[
            probe_trials['memory_trial_stimLength'] == wm_load]
        wm_probe_choice_correct = wm_probe_trials[
            wm_probe_trials['memory_recognition_correct_trial'] == 1.0]

        number_of_wm_probe_trials = wm_probe_trials.shape[0]
        number_of_wm_probe_correct_trials = wm_probe_choice_correct.shape[0]

        # Calculate metrics using the same style as analyze.py
        wm_metrics[f'probe_wm{wm_load}_response_accuracy'] = (
            number_of_wm_probe_correct_trials / number_of_wm_probe_trials
            if number_of_wm_probe_trials > 0 else np.nan)
        wm_metrics[f'probe_wm{wm_load}_mean_rt'] = (
            wm_probe_choice_correct['memory_recognition_rt'].mean()
            if len(wm_probe_choice_correct) > 0 else np.nan)
        wm_metrics[f'probe_wm{wm_load}_n_trials'] = number_of_wm_probe_trials
        
        # Probe accuracy conditional on stop outcome BY WM LOAD
        # Successful stops at this WM load
        wm_successful_stop_trials = successful_stop_trials[
            successful_stop_trials['memory_trial_stimLength'] == wm_load]
        wm_successful_stop_probe_correct = wm_successful_stop_trials[
            wm_successful_stop_trials['memory_recognition_correct_trial'] == 1.0]
        n_wm_successful_stop = wm_successful_stop_trials.shape[0]
        n_wm_successful_stop_correct = wm_successful_stop_probe_correct.shape[0]
        wm_metrics[f'probe_wm{wm_load}_accuracy_on_successful_stop'] = (
            n_wm_successful_stop_correct / n_wm_successful_stop
            if n_wm_successful_stop > 0 else np.nan)
        wm_metrics[f'probe_wm{wm_load}_n_successful_stop_trials'] = n_wm_successful_stop
        
        # Failed stops at this WM load
        wm_failed_stop_trials = failed_stop_trials[
            failed_stop_trials['memory_trial_stimLength'] == wm_load]
        wm_failed_stop_probe_correct = wm_failed_stop_trials[
            wm_failed_stop_trials['memory_recognition_correct_trial'] == 1.0]
        n_wm_failed_stop = wm_failed_stop_trials.shape[0]
        n_wm_failed_stop_correct = wm_failed_stop_probe_correct.shape[0]
        wm_metrics[f'probe_wm{wm_load}_accuracy_on_failed_stop'] = (
            n_wm_failed_stop_correct / n_wm_failed_stop
            if n_wm_failed_stop > 0 else np.nan)
        wm_metrics[f'probe_wm{wm_load}_n_failed_stop_trials'] = n_wm_failed_stop
        
        # Go trials at this WM load
        wm_go_trials = go_trials_for_probe[
            go_trials_for_probe['memory_trial_stimLength'] == wm_load]
        wm_go_probe_correct = wm_go_trials[
            wm_go_trials['memory_recognition_correct_trial'] == 1.0]
        n_wm_go_trials = wm_go_trials.shape[0]
        n_wm_go_correct = wm_go_probe_correct.shape[0]
        wm_metrics[f'probe_wm{wm_load}_accuracy_on_go_trials'] = (
            n_wm_go_correct / n_wm_go_trials
            if n_wm_go_trials > 0 else np.nan)
        wm_metrics[f'probe_wm{wm_load}_n_go_trials'] = n_wm_go_trials

    # SSD metrics
    ssd_values = group['stop_trial_SSD'].dropna()
    min_ssd = ssd_values.min() if len(ssd_values) > 0 else np.nan
    max_ssd = ssd_values.max() if len(ssd_values) > 0 else np.nan
    mean_ssd = ssd_values.mean() if len(ssd_values) > 0 else np.nan
    final_ssd = ssd_values.iloc[-1] if len(ssd_values) > 0 else np.nan

    # SSRT (Stop Signal Reaction Time) - Integration method with replacement
    # Uses all Go trial RTs, replacing omissions with response deadline
    # Response deadline extracted from stop_trial_trial_duration in task data
    go_rts_for_ssrt = dual_task_go_trials['stop_trial_rt']
    response_deadline = group['stop_trial_trial_duration'].iloc[0]
    ssrt = calculate_ssrt_integration(
        go_rts=go_rts_for_ssrt,
        go_omission_count=number_of_dual_task_go_omission_trials,
        stop_success_rate=dual_task_stop_inhibition_success_rate,
        mean_ssd=mean_ssd,
        response_deadline=response_deadline
    )

    # Combine all metrics
    result = {
        'prolific_id': participant_id,
        #'completion_date': participant_id,
        # Overall aggregated dual task metrics
        'dual_task_go_choice_accuracy': dual_task_go_choice_accuracy,
        'dual_task_go_mean_rt': dual_task_go_mean_rt,
        'dual_task_go_omission_rate': dual_task_go_omission_rate,
        'dual_task_go_commission_rate': dual_task_go_commission_rate,
        'dual_task_n_go_trials': number_of_dual_task_go_trials,
        'dual_task_n_go_omissions': number_of_dual_task_go_omission_trials,
        'dual_task_n_go_commissions': number_of_dual_task_go_commission_trials,
        'dual_task_stop_inhibition_success_rate': (
            dual_task_stop_inhibition_success_rate),
        'dual_task_stop_failed_mean_rt': dual_task_stop_failed_mean_rt,
        'dual_task_n_stop_trials': number_of_dual_task_stop_trials,
        # Overall probe metrics
        'probe_response_accuracy': probe_response_accuracy,
        'probe_mean_rt': probe_mean_rt,
        'n_probe_trials': number_of_probe_trials,
        # SSD metrics
        'min_SSD': min_ssd,
        'max_SSD': max_ssd,
        'mean_SSD': mean_ssd,
        'final_SSD': final_ssd,
        'SSRT': ssrt,
        'total_stop_trials': number_of_dual_task_stop_trials,
        'total_probe_trials': number_of_probe_trials
    }

    # Add WM load specific metrics
    result.update(wm_metrics)

    return pd.Series(result)

# ============================================================================
# Helper Functions
# ============================================================================
def extract_completion_dates(preprocessed_dir, task_name):
    """Extract completion dates from preprocessed data filenames.
    
    Args:
        preprocessed_dir: Path to preprocessed data directory
        task_name: Name of the task (e.g., 'stop_signal', 'stop_signal_wm_task')
    
    Returns:
        Dictionary mapping prolific_id to completion_date in YYYY.MM.DD format
    """
    participant_dates = {}
    
    for participant_dir in preprocessed_dir.iterdir():
        if participant_dir.name == 'approved':
            continue
        if not participant_dir.is_dir():
            continue

        prolific_id = participant_dir.name
        
        # Look for task directory
        task_dir = participant_dir / task_name
        if task_dir.exists():
            csv_files = list(task_dir.glob('*.csv'))
            if csv_files:
                filename = csv_files[0].name
                # Extract timestamp from filename: sub-{id}_task-{task}_date-{timestamp}.csv
                if '_date-' in filename:
                    timestamp_str = filename.split('_date-')[1].replace('.csv', '')
                    # Convert Unix milliseconds timestamp to readable date format
                    timestamp_ms = int(timestamp_str)
                    timestamp_s = timestamp_ms / 1000  # Convert milliseconds to seconds
                    date_obj = datetime.fromtimestamp(timestamp_s)
                    readable_date = date_obj.strftime('%Y.%m.%d')
                    participant_dates[prolific_id] = readable_date
    
    return participant_dates


# ============================================================================
# Main
# ============================================================================
COMBINED_CSV_PREFIX = 'all_participants_reshaped_data_'


def _metrics_fn_for(task_name: str):
    """Route a task name to the appropriate metrics function.

    Tasks with 'wm' in the name use the WM metrics function; everything else
    (e.g. simple stop-signal control) uses the stop-signal metrics function.
    """
    return calculate_wm_metrics if 'wm' in task_name.lower() else calculate_stop_signal_metrics


def _compute_task_metrics(task_name, csv_path, preprocessed_dir, results_dir):
    """Compute and save metrics for one combined trial-level CSV."""
    fn = _metrics_fn_for(task_name)
    print(f"Processing {task_name} with {fn.__name__}...")

    data = pd.read_csv(csv_path)
    if 'participant_id' not in data.columns:
        print(f"  Skipping {task_name}: no participant_id column in {csv_path.name}")
        return

    dates = extract_completion_dates(preprocessed_dir, task_name)

    rows = []
    failed = []
    for pid, group in data.groupby('participant_id'):
        try:
            rows.append(fn(group, participant_id=pid))
        except Exception as exc:
            failed.append((pid, exc))

    if not rows:
        print(f"  No metrics rows produced for {task_name}")
        if failed:
            print(f"  ({len(failed)} participants failed)")
        return

    metrics = pd.DataFrame(rows)
    metrics['completion_date'] = metrics['prolific_id'].map(dates)

    out_path = results_dir / f'{task_name}_metrics.csv'
    metrics.to_csv(out_path, index=False)
    print(f"  Saved {len(metrics)} participants -> {out_path}")
    if failed:
        print(f"  Skipped {len(failed)} participant(s) due to errors; "
              f"first: {failed[0][0]} ({failed[0][1]})")


def main():
    """Discover combined reshaped CSVs and compute metrics for each task."""
    results_dir = config.results_dir
    combined_files = sorted(results_dir.glob(f'{COMBINED_CSV_PREFIX}*.csv'))

    if not combined_files:
        print(f"No '{COMBINED_CSV_PREFIX}*.csv' files found in {results_dir}. "
              f"Run `clean_shape` first.")
        return

    print(f"Found {len(combined_files)} combined trial CSV(s) in {results_dir}")
    for csv_path in combined_files:
        task_name = csv_path.stem[len(COMBINED_CSV_PREFIX):]
        _compute_task_metrics(
            task_name, csv_path, config.active_preprocessed_dir(), results_dir,
        )

    print("All metrics calculated.")


if __name__ == "__main__":
    main()