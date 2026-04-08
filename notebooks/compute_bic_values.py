"""
Compute missing BIC and BF₁₀ values for statistical tests in experiment 1 analysis.
Uses the same approach as the write-up: subject-level metrics for ANOVAs and t-tests.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from scipy.stats import ttest_rel

# Set up paths and imports
sys.path.insert(0, '/sessions/admiring-peaceful-darwin/mnt/working_memory_inhibition/experiment_1/src')
from stop_wm.bic_bayes import calculate_bic, calculate_bf10, interpret_bic_delta

# Load data
metrics_data_wm = pd.read_csv('/sessions/admiring-peaceful-darwin/mnt/working_memory_inhibition/experiment_1/data/results/post_qc_stop_signal_wm_metrics.csv')

print("="*90)
print("COMPUTING MISSING BIC AND BF₁₀ VALUES FOR EXPERIMENT 1")
print("="*90)

def compute_bic_ttest_paired(group1, group2):
    """Compute BIC for paired t-test."""
    # Remove NaNs
    mask = ~(np.isnan(group1) | np.isnan(group2))
    g1 = group1[mask]
    g2 = group2[mask]
    
    if len(g1) < 2:
        return None, None, len(g1)
    
    # Null: grand mean
    grand_mean = (np.mean(g1) + np.mean(g2)) / 2
    residuals_null = np.concatenate([g1 - grand_mean, g2 - grand_mean])
    
    # Full: separate means
    residuals_full = np.concatenate([g1 - np.mean(g1), g2 - np.mean(g2)])
    
    n_obs = len(residuals_null)
    bic_null = calculate_bic(residuals_null, n_params=1, n_obs=n_obs)
    bic_full = calculate_bic(residuals_full, n_params=2, n_obs=n_obs)
    delta_bic = bic_null - bic_full
    bf10 = calculate_bf10(delta_bic)
    
    return delta_bic, bf10, len(g1)

def compute_bic_anova_3way(wm0, wm2, wm4):
    """Compute BIC for 3-way repeated measures ANOVA."""
    # Remove NaNs from all
    mask = ~(np.isnan(wm0) | np.isnan(wm2) | np.isnan(wm4))
    w0 = wm0[mask]
    w2 = wm2[mask]
    w4 = wm4[mask]
    
    if len(w0) < 2:
        return None, None, len(w0)
    
    # Null: grand mean
    grand_mean = (np.mean(w0) + np.mean(w2) + np.mean(w4)) / 3
    residuals_null = np.concatenate([w0 - grand_mean, w2 - grand_mean, w4 - grand_mean])
    
    # Full: separate means for each condition
    residuals_full = np.concatenate([
        w0 - np.mean(w0),
        w2 - np.mean(w2),
        w4 - np.mean(w4)
    ])
    
    n_obs = len(residuals_null)
    bic_null = calculate_bic(residuals_null, n_params=1, n_obs=n_obs)
    bic_full = calculate_bic(residuals_full, n_params=3, n_obs=n_obs)
    delta_bic = bic_null - bic_full
    bf10 = calculate_bf10(delta_bic)
    
    return delta_bic, bf10, len(w0)

# ============================================================================
# 1. GO TRIAL PERFORMANCE ANOVA (Accuracy and RT, 3-level WM load)
# ============================================================================
print("\n" + "="*90)
print("1. GO TRIAL PERFORMANCE ANOVA - Accuracy and RT (3-level WM load)")
print("="*90)

complete_go = metrics_data_wm.dropna(subset=[
    'dual_task_go_wm0_choice_accuracy', 'dual_task_go_wm2_choice_accuracy', 
    'dual_task_go_wm4_choice_accuracy',
    'dual_task_go_wm0_mean_rt', 'dual_task_go_wm2_mean_rt', 'dual_task_go_wm4_mean_rt'
])

print(f"N = {len(complete_go)}")

# 1a. Go Accuracy
print("\n  Go Trial Accuracy:")
delta_bic_go_acc, bf10_go_acc, n = compute_bic_anova_3way(
    complete_go['dual_task_go_wm0_choice_accuracy'].values,
    complete_go['dual_task_go_wm2_choice_accuracy'].values,
    complete_go['dual_task_go_wm4_choice_accuracy'].values
)
if delta_bic_go_acc is not None:
    print(f"    ΔBIC = {delta_bic_go_acc:.2f}, BF₁₀ = {bf10_go_acc:.2f}")

# 1b. Go RT
print("\n  Go Trial RT:")
delta_bic_go_rt, bf10_go_rt, n = compute_bic_anova_3way(
    complete_go['dual_task_go_wm0_mean_rt'].values,
    complete_go['dual_task_go_wm2_mean_rt'].values,
    complete_go['dual_task_go_wm4_mean_rt'].values
)
if delta_bic_go_rt is not None:
    print(f"    ΔBIC = {delta_bic_go_rt:.2f}, BF₁₀ = {bf10_go_rt:.2f}")

# ============================================================================
# 2. PROBE ACCURACY ANOVA (3-level WM load)
# ============================================================================
print("\n" + "="*90)
print("2. PROBE ACCURACY ANOVA (3-level WM load)")
print("="*90)

complete_probe = metrics_data_wm.dropna(subset=[
    'probe_wm0_response_accuracy', 'probe_wm2_response_accuracy',
    'probe_wm4_response_accuracy'
])

print(f"N = {len(complete_probe)}")

delta_bic_probe_acc, bf10_probe_acc, n = compute_bic_anova_3way(
    complete_probe['probe_wm0_response_accuracy'].values,
    complete_probe['probe_wm2_response_accuracy'].values,
    complete_probe['probe_wm4_response_accuracy'].values
)
if delta_bic_probe_acc is not None:
    print(f"  Probe Accuracy ANOVA: ΔBIC = {delta_bic_probe_acc:.2f}, BF₁₀ = {bf10_probe_acc:.2f}")

# ============================================================================
# 3. PROBE ACCURACY BY STOP SIGNAL OUTCOME - t-tests
# ============================================================================
print("\n" + "="*90)
print("3. PROBE ACCURACY BY STOP SIGNAL OUTCOME - t-tests")
print("="*90)

# Overall
print("\n  Overall (all WM loads collapsed):")

# Success vs Failed
delta_bic, bf10, n = compute_bic_ttest_paired(
    metrics_data_wm['probe_accuracy_on_successful_stop'].values,
    metrics_data_wm['probe_accuracy_on_failed_stop'].values
)
if delta_bic is not None:
    print(f"    Successful vs Failed: N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# Success vs Go
delta_bic, bf10, n = compute_bic_ttest_paired(
    metrics_data_wm['probe_accuracy_on_successful_stop'].values,
    metrics_data_wm['probe_accuracy_on_go_trials'].values
)
if delta_bic is not None:
    print(f"    Successful vs Go: N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# Failed vs Go
delta_bic, bf10, n = compute_bic_ttest_paired(
    metrics_data_wm['probe_accuracy_on_failed_stop'].values,
    metrics_data_wm['probe_accuracy_on_go_trials'].values
)
if delta_bic is not None:
    print(f"    Failed vs Go: N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# By WM Load
for wm in [0, 2, 4]:
    print(f"\n  WM Load {wm}:")
    
    success_col = f'probe_wm{wm}_accuracy_on_successful_stop'
    failed_col = f'probe_wm{wm}_accuracy_on_failed_stop'
    go_col = f'probe_wm{wm}_accuracy_on_go_trials'
    
    # Success vs Failed
    delta_bic, bf10, n = compute_bic_ttest_paired(
        metrics_data_wm[success_col].values,
        metrics_data_wm[failed_col].values
    )
    if delta_bic is not None:
        print(f"    Successful vs Failed: N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")
    
    # Success vs Go
    delta_bic, bf10, n = compute_bic_ttest_paired(
        metrics_data_wm[success_col].values,
        metrics_data_wm[go_col].values
    )
    if delta_bic is not None:
        print(f"    Successful vs Go: N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")
    
    # Failed vs Go
    delta_bic, bf10, n = compute_bic_ttest_paired(
        metrics_data_wm[failed_col].values,
        metrics_data_wm[go_col].values
    )
    if delta_bic is not None:
        print(f"    Failed vs Go: N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# ============================================================================
# Now load trial-level data for sequential dependencies
# ============================================================================
print("\n" + "="*90)
print("SEQUENTIAL DEPENDENCIES (trial-level analyses)")
print("="*90)

trial_data = pd.read_csv('/sessions/admiring-peaceful-darwin/mnt/working_memory_inhibition/experiment_1/data/results/post_qc_stop_signal_wm_trials.csv')

# Prepare data
trial_data = trial_data.sort_values(['participant_id', 'block_num', 'current_trial']).reset_index(drop=True)

# Create variables
trial_data['is_stop'] = (trial_data['stop_trial_SS_trial_type'] == 'stop')
trial_data['is_go'] = (trial_data['stop_trial_SS_trial_type'] == 'go')
trial_data['stop_success'] = (
    trial_data['is_stop'] & (trial_data['stop_trial_correct_trial'] == 1)
).astype(float)
trial_data['probe_correct'] = trial_data['memory_recognition_correct_trial']
trial_data['wm_load'] = trial_data['memory_trial_stimLength']

# Lagged variables
trial_data['prev_probe_correct'] = (
    trial_data.groupby(['participant_id', 'block_num'])['probe_correct'].shift(1)
)

trial_data['is_commission'] = (
    (trial_data['probe_correct'] == 0) & 
    (~trial_data['memory_recognition_rt'].isna())
)
trial_data['prev_is_commission'] = (
    trial_data.groupby(['participant_id', 'block_num'])['is_commission'].shift(1)
)

trial_data['is_omission'] = (
    (trial_data['probe_correct'] == 0) & 
    (trial_data['memory_recognition_rt'].isna())
)
trial_data['prev_is_omission'] = (
    trial_data.groupby(['participant_id', 'block_num'])['is_omission'].shift(1)
)

# Go trial outcome variables (stop_trial_correct_trial is 1 for correct go, 0 for incorrect)
trial_data['go_trial_correct'] = trial_data[trial_data['is_go']]['stop_trial_correct_trial']
trial_data['go_trial_rt'] = trial_data[trial_data['is_go']]['stop_trial_rt']

# ============================================================================
# 4. SEQUENTIAL DEPENDENCIES - Prior Probe Outcome
# ============================================================================
print("\n" + "="*90)
print("4. SEQUENTIAL DEPENDENCIES - Prior Probe Outcome")
print("="*90)

# 4a. SSD
print("\n  Prior Probe Outcome → SSD:")
stop_with_lag = trial_data[
    trial_data['is_stop'] & trial_data['prev_probe_correct'].notna()
].copy()

subj_ssd = []
for pid in stop_with_lag['participant_id'].unique():
    pid_data = stop_with_lag[stop_with_lag['participant_id'] == pid]
    ssd_correct = pid_data[pid_data['prev_probe_correct'] == 1]['stop_trial_SSD'].mean()
    ssd_incorrect = pid_data[pid_data['prev_probe_correct'] == 0]['stop_trial_SSD'].mean()
    if not np.isnan(ssd_correct) and not np.isnan(ssd_incorrect):
        subj_ssd.append({'ssd_correct': ssd_correct, 'ssd_incorrect': ssd_incorrect})

if subj_ssd:
    ssd_df = pd.DataFrame(subj_ssd)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        ssd_df['ssd_correct'].values, ssd_df['ssd_incorrect'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# 4b. Commission-only stop success
print("\n  Prior Commission Error → Stop Success:")
stop_commission = trial_data[
    trial_data['is_stop'] & trial_data['prev_is_commission'].notna()
].copy()

subj_stop_comm = []
for pid in stop_commission['participant_id'].unique():
    pid_data = stop_commission[stop_commission['participant_id'] == pid]
    success_after_comm = pid_data[pid_data['prev_is_commission'] == 1]['stop_success'].mean()
    success_after_correct = pid_data[pid_data['prev_is_commission'] == 0]['stop_success'].mean()
    if not np.isnan(success_after_comm) and not np.isnan(success_after_correct):
        subj_stop_comm.append({'success_comm': success_after_comm, 'success_correct': success_after_correct})

if subj_stop_comm:
    stop_comm_df = pd.DataFrame(subj_stop_comm)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        stop_comm_df['success_comm'].values, stop_comm_df['success_correct'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# ============================================================================
# 5. SEQUENTIAL DEPENDENCIES - Prior Probe Omission
# ============================================================================
print("\n" + "="*90)
print("5. SEQUENTIAL DEPENDENCIES - Prior Probe Omission")
print("="*90)

# 5a. Probe RT
print("\n  Prior Probe Omission → Probe RT:")
probe_omission = trial_data[
    ~trial_data['is_go'] & trial_data['prev_is_omission'].notna() & 
    ~trial_data['memory_recognition_rt'].isna()
].copy()

subj_probe_rt = []
for pid in probe_omission['participant_id'].unique():
    pid_data = probe_omission[probe_omission['participant_id'] == pid]
    rt_omission = pid_data[pid_data['prev_is_omission'] == 1]['memory_recognition_rt'].mean()
    rt_response = pid_data[pid_data['prev_is_omission'] == 0]['memory_recognition_rt'].mean()
    if not np.isnan(rt_omission) and not np.isnan(rt_response):
        subj_probe_rt.append({'rt_omission': rt_omission, 'rt_response': rt_response})

if subj_probe_rt:
    probe_rt_df = pd.DataFrame(subj_probe_rt)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        probe_rt_df['rt_omission'].values, probe_rt_df['rt_response'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# 5b. Go Accuracy
print("\n  Prior Probe Omission → Go Accuracy:")
go_omission = trial_data[
    trial_data['is_go'] & trial_data['prev_is_omission'].notna()
].copy()

subj_go_acc = []
for pid in go_omission['participant_id'].unique():
    pid_data = go_omission[go_omission['participant_id'] == pid]
    acc_omission = pid_data[pid_data['prev_is_omission'] == 1]['stop_trial_correct_trial'].mean()
    acc_response = pid_data[pid_data['prev_is_omission'] == 0]['stop_trial_correct_trial'].mean()
    if not np.isnan(acc_omission) and not np.isnan(acc_response):
        subj_go_acc.append({'acc_omission': acc_omission, 'acc_response': acc_response})

if subj_go_acc:
    go_acc_df = pd.DataFrame(subj_go_acc)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        go_acc_df['acc_omission'].values, go_acc_df['acc_response'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# 5c. Probe Commission Rate
print("\n  Prior Probe Omission → Probe Commission Rate:")
probe_for_comm = trial_data[
    ~trial_data['is_go'] & trial_data['prev_is_omission'].notna()
].copy()

subj_probe_comm = []
for pid in probe_for_comm['participant_id'].unique():
    pid_data = probe_for_comm[probe_for_comm['participant_id'] == pid]
    
    after_omission = pid_data[pid_data['prev_is_omission'] == 1]
    n_resp_omission = (~after_omission['memory_recognition_rt'].isna()).sum()
    comm_after_omission = (after_omission['is_commission'] == 1).sum() / n_resp_omission if n_resp_omission > 0 else np.nan
    
    after_response = pid_data[pid_data['prev_is_omission'] == 0]
    n_resp_response = (~after_response['memory_recognition_rt'].isna()).sum()
    comm_after_response = (after_response['is_commission'] == 1).sum() / n_resp_response if n_resp_response > 0 else np.nan
    
    if not np.isnan(comm_after_omission) and not np.isnan(comm_after_response):
        subj_probe_comm.append({'comm_omission': comm_after_omission, 'comm_response': comm_after_response})

if subj_probe_comm:
    probe_comm_df = pd.DataFrame(subj_probe_comm)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        probe_comm_df['comm_omission'].values, probe_comm_df['comm_response'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# ============================================================================
# 6. SEQUENTIAL DEPENDENCIES - Prior Commission Error
# ============================================================================
print("\n" + "="*90)
print("6. SEQUENTIAL DEPENDENCIES - Prior Commission Error")
print("="*90)

# 6a. Stop Success
print("\n  Prior Commission Error → Stop Success:")
stop_comm_lag = trial_data[
    trial_data['is_stop'] & trial_data['prev_is_commission'].notna()
].copy()

subj_stop_comm_lag = []
for pid in stop_comm_lag['participant_id'].unique():
    pid_data = stop_comm_lag[stop_comm_lag['participant_id'] == pid]
    success_after_comm = pid_data[pid_data['prev_is_commission'] == 1]['stop_success'].mean()
    success_after_correct = pid_data[pid_data['prev_is_commission'] == 0]['stop_success'].mean()
    if not np.isnan(success_after_comm) and not np.isnan(success_after_correct):
        subj_stop_comm_lag.append({'success_comm': success_after_comm, 'success_correct': success_after_correct})

if subj_stop_comm_lag:
    stop_comm_lag_df = pd.DataFrame(subj_stop_comm_lag)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        stop_comm_lag_df['success_comm'].values, stop_comm_lag_df['success_correct'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# 6b. Probe RT
print("\n  Prior Commission Error → Probe RT:")
probe_comm_lag = trial_data[
    ~trial_data['is_go'] & trial_data['prev_is_commission'].notna() & 
    ~trial_data['memory_recognition_rt'].isna()
].copy()

subj_probe_rt_comm = []
for pid in probe_comm_lag['participant_id'].unique():
    pid_data = probe_comm_lag[probe_comm_lag['participant_id'] == pid]
    rt_after_comm = pid_data[pid_data['prev_is_commission'] == 1]['memory_recognition_rt'].mean()
    rt_after_correct = pid_data[pid_data['prev_is_commission'] == 0]['memory_recognition_rt'].mean()
    if not np.isnan(rt_after_comm) and not np.isnan(rt_after_correct):
        subj_probe_rt_comm.append({'rt_comm': rt_after_comm, 'rt_correct': rt_after_correct})

if subj_probe_rt_comm:
    probe_rt_comm_df = pd.DataFrame(subj_probe_rt_comm)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        probe_rt_comm_df['rt_comm'].values, probe_rt_comm_df['rt_correct'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# 6c. Go RT
print("\n  Prior Commission Error → Go RT:")
go_comm_lag = trial_data[
    trial_data['is_go'] & trial_data['prev_is_commission'].notna() & 
    ~trial_data['stop_trial_rt'].isna()
].copy()

subj_go_rt_comm = []
for pid in go_comm_lag['participant_id'].unique():
    pid_data = go_comm_lag[go_comm_lag['participant_id'] == pid]
    rt_after_comm = pid_data[pid_data['prev_is_commission'] == 1]['stop_trial_rt'].mean()
    rt_after_correct = pid_data[pid_data['prev_is_commission'] == 0]['stop_trial_rt'].mean()
    if not np.isnan(rt_after_comm) and not np.isnan(rt_after_correct):
        subj_go_rt_comm.append({'rt_comm': rt_after_comm, 'rt_correct': rt_after_correct})

if subj_go_rt_comm:
    go_rt_comm_df = pd.DataFrame(subj_go_rt_comm)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        go_rt_comm_df['rt_comm'].values, go_rt_comm_df['rt_correct'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# 6d. SSD
print("\n  Prior Commission Error → SSD:")
stop_ssd_comm = trial_data[
    trial_data['is_stop'] & trial_data['prev_is_commission'].notna()
].copy()

subj_ssd_comm = []
for pid in stop_ssd_comm['participant_id'].unique():
    pid_data = stop_ssd_comm[stop_ssd_comm['participant_id'] == pid]
    ssd_after_comm = pid_data[pid_data['prev_is_commission'] == 1]['stop_trial_SSD'].mean()
    ssd_after_correct = pid_data[pid_data['prev_is_commission'] == 0]['stop_trial_SSD'].mean()
    if not np.isnan(ssd_after_comm) and not np.isnan(ssd_after_correct):
        subj_ssd_comm.append({'ssd_comm': ssd_after_comm, 'ssd_correct': ssd_after_correct})

if subj_ssd_comm:
    ssd_comm_df = pd.DataFrame(subj_ssd_comm)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        ssd_comm_df['ssd_comm'].values, ssd_comm_df['ssd_correct'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# 6e. Go Accuracy
print("\n  Prior Commission Error → Go Accuracy:")
go_acc_comm = trial_data[
    trial_data['is_go'] & trial_data['prev_is_commission'].notna()
].copy()

subj_go_acc_comm = []
for pid in go_acc_comm['participant_id'].unique():
    pid_data = go_acc_comm[go_acc_comm['participant_id'] == pid]
    acc_after_comm = pid_data[pid_data['prev_is_commission'] == 1]['stop_trial_correct_trial'].mean()
    acc_after_correct = pid_data[pid_data['prev_is_commission'] == 0]['stop_trial_correct_trial'].mean()
    if not np.isnan(acc_after_comm) and not np.isnan(acc_after_correct):
        subj_go_acc_comm.append({'acc_comm': acc_after_comm, 'acc_correct': acc_after_correct})

if subj_go_acc_comm:
    go_acc_comm_df = pd.DataFrame(subj_go_acc_comm)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        go_acc_comm_df['acc_comm'].values, go_acc_comm_df['acc_correct'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# 6f. Go Omission Rate
print("\n  Prior Commission Error → Go Omission Rate:")
go_omiss_comm = trial_data[
    trial_data['is_go'] & trial_data['prev_is_commission'].notna()
].copy()

subj_go_omiss_comm = []
for pid in go_omiss_comm['participant_id'].unique():
    pid_data = go_omiss_comm[go_omiss_comm['participant_id'] == pid]
    
    # Go omission: correct_trial == 0 AND rt is NaN
    after_comm = pid_data[pid_data['prev_is_commission'] == 1]
    omiss_after_comm = (
        ((after_comm['stop_trial_correct_trial'] == 0) & (after_comm['stop_trial_rt'].isna())).sum() / 
        len(after_comm)
    ) if len(after_comm) > 0 else np.nan
    
    after_correct = pid_data[pid_data['prev_is_commission'] == 0]
    omiss_after_correct = (
        ((after_correct['stop_trial_correct_trial'] == 0) & (after_correct['stop_trial_rt'].isna())).sum() / 
        len(after_correct)
    ) if len(after_correct) > 0 else np.nan
    
    if not np.isnan(omiss_after_comm) and not np.isnan(omiss_after_correct):
        subj_go_omiss_comm.append({'omiss_comm': omiss_after_comm, 'omiss_correct': omiss_after_correct})

if subj_go_omiss_comm:
    go_omiss_comm_df = pd.DataFrame(subj_go_omiss_comm)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        go_omiss_comm_df['omiss_comm'].values, go_omiss_comm_df['omiss_correct'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# 6g. Probe Omission Rate
print("\n  Prior Commission Error → Probe Omission Rate:")
probe_omiss_comm = trial_data[
    ~trial_data['is_go'] & trial_data['prev_is_commission'].notna()
].copy()

subj_probe_omiss_comm = []
for pid in probe_omiss_comm['participant_id'].unique():
    pid_data = probe_omiss_comm[probe_omiss_comm['participant_id'] == pid]
    omiss_after_comm = pid_data[pid_data['prev_is_commission'] == 1]['is_omission'].mean()
    omiss_after_correct = pid_data[pid_data['prev_is_commission'] == 0]['is_omission'].mean()
    if not np.isnan(omiss_after_comm) and not np.isnan(omiss_after_correct):
        subj_probe_omiss_comm.append({'omiss_comm': omiss_after_comm, 'omiss_correct': omiss_after_correct})

if subj_probe_omiss_comm:
    probe_omiss_comm_df = pd.DataFrame(subj_probe_omiss_comm)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        probe_omiss_comm_df['omiss_comm'].values, probe_omiss_comm_df['omiss_correct'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

# 6h. Probe Commission Rate
print("\n  Prior Commission Error → Probe Commission Rate:")
probe_comm_rate = trial_data[
    ~trial_data['is_go'] & trial_data['prev_is_commission'].notna()
].copy()

subj_probe_comm_rate = []
for pid in probe_comm_rate['participant_id'].unique():
    pid_data = probe_comm_rate[probe_comm_rate['participant_id'] == pid]
    
    after_comm = pid_data[pid_data['prev_is_commission'] == 1]
    n_resp_comm = (~after_comm['memory_recognition_rt'].isna()).sum()
    comm_after_comm = (after_comm['is_commission'] == 1).sum() / n_resp_comm if n_resp_comm > 0 else np.nan
    
    after_correct = pid_data[pid_data['prev_is_commission'] == 0]
    n_resp_correct = (~after_correct['memory_recognition_rt'].isna()).sum()
    comm_after_correct = (after_correct['is_commission'] == 1).sum() / n_resp_correct if n_resp_correct > 0 else np.nan
    
    if not np.isnan(comm_after_comm) and not np.isnan(comm_after_correct):
        subj_probe_comm_rate.append({'comm_after_comm': comm_after_comm, 'comm_after_correct': comm_after_correct})

if subj_probe_comm_rate:
    probe_comm_rate_df = pd.DataFrame(subj_probe_comm_rate)
    delta_bic, bf10, n = compute_bic_ttest_paired(
        probe_comm_rate_df['comm_after_comm'].values, probe_comm_rate_df['comm_after_correct'].values
    )
    if delta_bic is not None:
        print(f"    N={n}, ΔBIC = {delta_bic:.2f}, BF₁₀ = {bf10:.2f}")

print("\n" + "="*90)
print("COMPUTATION COMPLETE")
print("="*90)
