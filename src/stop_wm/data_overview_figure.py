"""
Create figures that show an overview for the stop+wm experiment.

This module provides comprehensive data visualization including:
- Summary statistics tables for both tasks
- Histograms of RT distributions
- Boxplots with within-subject confidence intervals

"""
# ============================================================================
# Imports
# ============================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from stop_wm.config import ProjectConfig

# Initialize config
config = ProjectConfig()

# Set plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# ============================================================================
# Helper Functions
# ============================================================================

def calculate_within_subject_ci(data, subject_col, value_col, ci=0.95):
    """
    Calculate within-subject confidence intervals using the Cousineau-Morey method.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data containing subject-level measurements
    subject_col : str
        Column name for subject identifier
    value_col : str
        Column name for the values to compute CI for
    ci : float
        Confidence interval level (default: 0.95)
    
    Returns
    -------
    float
        The half-width of the confidence interval
    """
    # Calculate subject means
    subject_means = data.groupby(subject_col)[value_col].mean()
    grand_mean = subject_means.mean()
    
    # Center each subject's data around the grand mean
    centered_data = data.copy()
    for subject in data[subject_col].unique():
        subject_mask = data[subject_col] == subject
        subject_mean = subject_means[subject]
        centered_data.loc[subject_mask, value_col] = (
            data.loc[subject_mask, value_col] - subject_mean + grand_mean
        )
    
    # Calculate standard error of the centered data
    se = centered_data.groupby(subject_col)[value_col].mean().sem()
    
    # Apply Morey correction factor for within-subject designs
    n_conditions = data.groupby(subject_col).size().max()
    correction = np.sqrt(n_conditions / (n_conditions - 1))
    se_corrected = se * correction
    
    # Calculate CI using t-distribution
    from scipy import stats
    n_subjects = data[subject_col].nunique()
    t_crit = stats.t.ppf((1 + ci) / 2, n_subjects - 1)

    return t_crit * se_corrected


def create_summary_statistics_table(metrics_data, task_name):
    """
    Create a summary statistics table for a given task.
    
    Parameters
    ----------
    metrics_data : pd.DataFrame
        Metrics data for the task
    task_name : str
        Name of the task (for labeling)
    
    Returns
    -------
    pd.DataFrame
        Summary statistics table
    """
    summary_stats = {
        'Metric': [],
        'Mean': [],
        'SD': [],
        'Min': [],
        'Max': [],
        'N': []
    }
    
    # Common metrics
    if 'go_mean_rt' in metrics_data.columns:
        summary_stats['Metric'].append('Go RT (ms)')
        summary_stats['Mean'].append(f"{metrics_data['go_mean_rt'].mean():.2f}")
        summary_stats['SD'].append(f"{metrics_data['go_mean_rt'].std():.2f}")
        summary_stats['Min'].append(f"{metrics_data['go_mean_rt'].min():.2f}")
        summary_stats['Max'].append(f"{metrics_data['go_mean_rt'].max():.2f}")
        summary_stats['N'].append(len(metrics_data))
    
    if 'stop_failed_mean_rt' in metrics_data.columns:
        summary_stats['Metric'].append('Failed Stop RT (ms)')
        summary_stats['Mean'].append(f"{metrics_data['stop_failed_mean_rt'].mean():.2f}")
        summary_stats['SD'].append(f"{metrics_data['stop_failed_mean_rt'].std():.2f}")
        summary_stats['Min'].append(f"{metrics_data['stop_failed_mean_rt'].min():.2f}")
        summary_stats['Max'].append(f"{metrics_data['stop_failed_mean_rt'].max():.2f}")
        summary_stats['N'].append(len(metrics_data))
    
    if 'go_choice_accuracy' in metrics_data.columns:
        summary_stats['Metric'].append('Go Accuracy (%)')
        acc_mean = metrics_data['go_choice_accuracy'].mean() * 100
        acc_std = metrics_data['go_choice_accuracy'].std() * 100
        acc_min = metrics_data['go_choice_accuracy'].min() * 100
        acc_max = metrics_data['go_choice_accuracy'].max() * 100
        summary_stats['Mean'].append(f"{acc_mean:.2f}")
        summary_stats['SD'].append(f"{acc_std:.2f}")
        summary_stats['Min'].append(f"{acc_min:.2f}")
        summary_stats['Max'].append(f"{acc_max:.2f}")
        summary_stats['N'].append(len(metrics_data))
    
    if 'stop_inhibition_success_rate' in metrics_data.columns:
        summary_stats['Metric'].append('Stop Inhibition Success (%)')
        inh_mean = metrics_data['stop_inhibition_success_rate'].mean() * 100
        inh_std = metrics_data['stop_inhibition_success_rate'].std() * 100
        inh_min = metrics_data['stop_inhibition_success_rate'].min() * 100
        inh_max = metrics_data['stop_inhibition_success_rate'].max() * 100
        summary_stats['Mean'].append(f"{inh_mean:.2f}")
        summary_stats['SD'].append(f"{inh_std:.2f}")
        summary_stats['Min'].append(f"{inh_min:.2f}")
        summary_stats['Max'].append(f"{inh_max:.2f}")
        summary_stats['N'].append(len(metrics_data))
    
    if 'mean_SSD' in metrics_data.columns:
        summary_stats['Metric'].append('Mean SSD (ms)')
        summary_stats['Mean'].append(f"{metrics_data['mean_SSD'].mean():.2f}")
        summary_stats['SD'].append(f"{metrics_data['mean_SSD'].std():.2f}")
        summary_stats['Min'].append(f"{metrics_data['mean_SSD'].min():.2f}")
        summary_stats['Max'].append(f"{metrics_data['mean_SSD'].max():.2f}")
        summary_stats['N'].append(len(metrics_data))
    
    # WM-specific metrics
    if 'probe_mean_rt' in metrics_data.columns:
        summary_stats['Metric'].append('Probe RT (ms)')
        summary_stats['Mean'].append(f"{metrics_data['probe_mean_rt'].mean():.2f}")
        summary_stats['SD'].append(f"{metrics_data['probe_mean_rt'].std():.2f}")
        summary_stats['Min'].append(f"{metrics_data['probe_mean_rt'].min():.2f}")
        summary_stats['Max'].append(f"{metrics_data['probe_mean_rt'].max():.2f}")
        summary_stats['N'].append(len(metrics_data))
    
    if 'probe_response_accuracy' in metrics_data.columns:
        summary_stats['Metric'].append('Probe Accuracy (%)')
        probe_mean = metrics_data['probe_response_accuracy'].mean() * 100
        probe_std = metrics_data['probe_response_accuracy'].std() * 100
        probe_min = metrics_data['probe_response_accuracy'].min() * 100
        probe_max = metrics_data['probe_response_accuracy'].max() * 100
        summary_stats['Mean'].append(f"{probe_mean:.2f}")
        summary_stats['SD'].append(f"{probe_std:.2f}")
        summary_stats['Min'].append(f"{probe_min:.2f}")
        summary_stats['Max'].append(f"{probe_max:.2f}")
        summary_stats['N'].append(len(metrics_data))
    
    return pd.DataFrame(summary_stats)


def plot_stop_vs_go_rt_basic(trial_data, ax):
    """
    Create histogram comparing stop (failed) vs go RT for basic stop signal task.
    
    Parameters
    ----------
    trial_data : pd.DataFrame
        Trial-level data
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    # Get go trials (with responses)
    go_trials = trial_data[
        (trial_data['trial_SS_trial_type'] == 'go') & 
        (trial_data['trial_rt'].notna()) &
        (trial_data['trial_rt'] > 0)
    ]['trial_rt']
    
    # Get failed stop trials (stop trials with responses)
    stop_trials = trial_data[
        (trial_data['trial_SS_trial_type'] == 'stop') & 
        (trial_data['trial_rt'].notna()) &
        (trial_data['trial_rt'] > 0)
    ]['trial_rt']
    
    # Create histogram
    bins = np.linspace(0, 2000, 50)
    ax.hist(
        go_trials, bins=bins, alpha=0.6,
        label=f'Go RT (n={len(go_trials)})',
        color='blue', edgecolor='black'
    )
    ax.hist(
        stop_trials, bins=bins, alpha=0.6,
        label=f'Failed Stop RT (n={len(stop_trials)})',
        color='red', edgecolor='black'
    )
    
    ax.set_xlabel('Response Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Basic Stop Signal Task: Go vs Failed Stop RT')
    ax.legend()
    ax.set_xlim(0, 2000)


def plot_stop_vs_go_rt_by_level(trial_data, ax):
    """
    Create histograms comparing stop (failed) vs go RT for each WM level.

    Parameters
    ----------
    trial_data : pd.DataFrame
        Trial-level data for WM task
    ax : matplotlib.axes.Axes
        Axes array to plot on
    """
    levels = [0, 2, 4]
    colors_go = ['#1f77b4', '#ff7f0e', '#2ca02c']
    colors_stop = ['#d62728', '#ff1744', '#c51162']

    # All levels on one plot
    bins = np.linspace(0, 2000, 50)

    for i, level in enumerate(levels):
        # Get go trials for this level
        go_trials = trial_data[
            (trial_data['stop_trial_SS_trial_type'] == 'go') &
            (trial_data['stop_trial_rt'].notna()) &
            (trial_data['stop_trial_rt'] > 0) &
            (trial_data['memory_trial_stimLength'] == level)
        ]['stop_trial_rt']

        # Get failed stop trials for this level
        stop_trials = trial_data[
            (trial_data['stop_trial_SS_trial_type'] == 'stop') &
            (trial_data['stop_trial_rt'].notna()) &
            (trial_data['stop_trial_rt'] > 0) &
            (trial_data['memory_trial_stimLength'] == level)
        ]['stop_trial_rt']

        ax.hist(
            go_trials, bins=bins, alpha=0.4,
            label=f'WM{level} Go (n={len(go_trials)})',
            color=colors_go[i], edgecolor='black', linewidth=0.5
        )
        ax.hist(
            stop_trials, bins=bins, alpha=0.4,
            label=f'WM{level} Stop (n={len(stop_trials)})',
            color=colors_stop[i], edgecolor='black', linewidth=0.5
        )

    ax.set_xlabel('Response Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('WM Task: Go vs Failed Stop RT by WM Level')
    ax.legend(ncol=2, fontsize=9)
    ax.set_xlim(0, 2000)


def plot_probe_rt_by_level(trial_data, ax):
    """
    Create histogram of probe RT by WM level.

    Parameters
    ----------
    trial_data : pd.DataFrame
        Trial-level data for WM task
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    levels = [0, 2, 4]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    bins = np.linspace(0, 3000, 50)

    for i, level in enumerate(levels):
        probe_rts = trial_data[
            (trial_data['memory_recognition_rt'].notna()) &
            (trial_data['memory_recognition_rt'] > 0) &
            (trial_data['memory_trial_stimLength'] == level)
        ]['memory_recognition_rt']

        ax.hist(
            probe_rts, bins=bins, alpha=0.5,
            label=f'WM{level} (n={len(probe_rts)})',
            color=colors[i], edgecolor='black', linewidth=0.5
        )

    ax.set_xlabel('Response Time (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Probe RT by WM Level')
    ax.legend()
    ax.set_xlim(0, 3000)


def plot_rt_by_wm_level_barplot(trial_data, ax):
    """
    Create bar plot of mean RT by WM level with within-subject CIs.

    Parameters
    ----------
    trial_data : pd.DataFrame
        Trial-level data for WM task
    ax : matplotlib.axes.Axes
        Axes to plot on
    """
    # Prepare data for plotting
    levels = [0, 2, 4]

    # Calculate subject means for each level
    rt_by_level = []
    for level in levels:
        level_data = trial_data[
            (trial_data['stop_trial_SS_trial_type'] == 'go') &
            (trial_data['stop_trial_rt'].notna()) &
            (trial_data['stop_trial_rt'] > 0) &
            (trial_data['memory_trial_stimLength'] == level)
        ].groupby('participant_id')['stop_trial_rt'].mean().reset_index()
        level_data['wm_level'] = level
        rt_by_level.append(level_data)

    rt_df = pd.concat(rt_by_level, ignore_index=True)

    # Calculate means and within-subject CIs
    means = []
    cis = []
    for level in levels:
        level_data = rt_df[rt_df['wm_level'] == level]
        means.append(level_data['stop_trial_rt'].mean())

        # Calculate within-subject CI
        # Create a long-form dataset for CI calculation
        level_subset = trial_data[
            (trial_data['stop_trial_SS_trial_type'] == 'go') &
            (trial_data['stop_trial_rt'].notna()) &
            (trial_data['stop_trial_rt'] > 0) &
            (trial_data['memory_trial_stimLength'] == level)
        ][['participant_id', 'stop_trial_rt']]

        if len(level_subset) > 0:
            ci = calculate_within_subject_ci(
                level_subset, 'participant_id', 'stop_trial_rt'
            )
            cis.append(ci)
        else:
            cis.append(0)

    # Create bar plot with error bars
    x_positions = np.arange(len(levels))
    bars = ax.bar(
        x_positions, means, yerr=cis,
        color=['#1f77b4', '#ff7f0e', '#2ca02c'],
        alpha=0.7,
        capsize=8,
        error_kw={'linewidth': 2, 'ecolor': 'black'}
    )

    # Customize the plot
    ax.set_xlabel('Working Memory Level', fontsize=12)
    ax.set_ylabel('Mean Go RT (ms)', fontsize=12)
    ax.set_title(
        'Go RT by WM Level\n(Within-Subject 95% CI)',
        fontsize=14, fontweight='bold'
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['WM0', 'WM2', 'WM4'])
    ax.set_ylim(0, max(means) * 1.2)

    # Add value labels on top of bars
    for i, (bar, mean, ci) in enumerate(zip(bars, means, cis)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2., height + ci + 10,
            f'{mean:.1f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )


# ============================================================================
# Main Functions
# ============================================================================

def create_data_overview_figure():
    """
    Create comprehensive data overview figures for the stop+wm experiment.

    This function creates multiple figures:
    1. Summary statistics tables
    2. RT histograms for basic stop signal task
    3. RT histograms by WM level
    4. Probe RT histograms
    5. Bar plots with within-subject CIs
    """
    print("Loading data...")
    
    # Load metrics data (post-QC only)
    stop_signal_metrics = pd.read_csv(
        config.results_dir / "post_qc_stop_signal_metrics.csv"
    )
    wm_task_metrics = pd.read_csv(
        config.results_dir / "post_qc_stop_signal_wm_metrics.csv"
    )

    print(
        f"Post-QC participants: {len(stop_signal_metrics)} (basic), "
        f"{len(wm_task_metrics)} (WM)"
    )

    # Load trial-level data (already filtered to post-QC participants)
    print("Loading post-QC trial-level data...")
    stop_signal_trials = pd.read_csv(
        config.results_dir / "post_qc_stop_signal_trials.csv"
    )
    wm_task_trials = pd.read_csv(
        config.results_dir / "post_qc_stop_signal_wm_trials.csv"
    )

    print(
        f"Loaded trials: {len(stop_signal_trials)} (basic), "
        f"{len(wm_task_trials)} (WM)"
    )
    
    # ========================================================================
    # Figure 1: Summary Statistics Tables
    # ========================================================================
    print("\nCreating summary statistics tables...")

    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle(
        'Data Summary Statistics', fontsize=16, fontweight='bold'
    )

    # Basic stop signal task
    stop_summary = create_summary_statistics_table(
        stop_signal_metrics, "Basic Stop Signal"
    )
    axes1[0].axis('tight')
    axes1[0].axis('off')
    table1 = axes1[0].table(
        cellText=stop_summary.values,
        colLabels=stop_summary.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2)
    axes1[0].set_title(
        'Basic Stop Signal Task', fontsize=14, fontweight='bold', pad=20
    )

    # WM task
    wm_summary = create_summary_statistics_table(wm_task_metrics, "WM Task")
    axes1[1].axis('tight')
    axes1[1].axis('off')
    table2 = axes1[1].table(
        cellText=wm_summary.values,
        colLabels=wm_summary.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)
    axes1[1].set_title(
        'Stop Signal + WM Task', fontsize=14, fontweight='bold', pad=20
    )

    plt.tight_layout()
    fig1.savefig(
        config.figures_dir / 'data_summary_statistics.png',
        dpi=300, bbox_inches='tight'
    )
    print(f"Saved: {config.figures_dir / 'data_summary_statistics.png'}")
    
    # ========================================================================
    # Figure 2: Basic Stop Signal RT Distributions
    # ========================================================================
    print("\nCreating basic stop signal RT histograms...")

    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    plot_stop_vs_go_rt_basic(stop_signal_trials, ax2)
    plt.tight_layout()
    fig2.savefig(
        config.figures_dir / 'stop_vs_go_rt_basic.png',
        dpi=300, bbox_inches='tight'
    )
    print(f"Saved: {config.figures_dir / 'stop_vs_go_rt_basic.png'}")

    # ========================================================================
    # Figure 3: WM Task RT Distributions by Level
    # ========================================================================
    print("\nCreating WM task RT histograms by level...")

    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
    plot_stop_vs_go_rt_by_level(wm_task_trials, ax3)
    plt.tight_layout()
    fig3.savefig(
        config.figures_dir / 'stop_vs_go_rt_by_wm_level.png',
        dpi=300, bbox_inches='tight'
    )
    print(f"Saved: {config.figures_dir / 'stop_vs_go_rt_by_wm_level.png'}")

    # ========================================================================
    # Figure 4: Probe RT by Level
    # ========================================================================
    print("\nCreating probe RT histograms...")

    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
    plot_probe_rt_by_level(wm_task_trials, ax4)
    plt.tight_layout()
    fig4.savefig(
        config.figures_dir / 'probe_rt_by_wm_level.png',
        dpi=300, bbox_inches='tight'
    )
    print(f"Saved: {config.figures_dir / 'probe_rt_by_wm_level.png'}")

    # ========================================================================
    # Figure 5: Bar plots with Within-Subject CIs
    # ========================================================================
    print("\nCreating bar plots with within-subject confidence intervals...")

    fig5, ax5 = plt.subplots(1, 1, figsize=(10, 7))
    plot_rt_by_wm_level_barplot(wm_task_trials, ax5)
    plt.tight_layout()
    fig5.savefig(
        config.figures_dir / 'go_rt_by_wm_level_barplot.png',
        dpi=300, bbox_inches='tight'
    )
    print(
        f"Saved: {config.figures_dir / 'go_rt_by_wm_level_barplot.png'}"
    )

    print("\n" + "="*70)
    print("All figures created successfully!")
    print(f"Figures saved to: {config.figures_dir}")
    print("="*70)

    plt.show()


# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    create_data_overview_figure()
