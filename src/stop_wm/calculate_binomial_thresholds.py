#!/usr/bin/env python3
"""Calculate binomial test thresholds for stop signal tasks.

This script calculates the critical values (min and max) for binomial tests
with null hypothesis p = 0.5, to identify participants whose stop trial 
response rates deviate significantly from chance.

These thresholds are used in exclusion_criteria.txt to flag participants
who may not be following task instructions or whose staircase isn't working.
"""

from scipy.stats import binom

# Task parameters
STOP_SIGNAL_N_STOP = 60  # Stop signal task stop trials
DUAL_TASK_N_STOP = 144   # Dual task total stop trials
DUAL_TASK_WM0_N_STOP = 48  # Dual task WM load 0 stop trials
DUAL_TASK_WM2_N_STOP = 48  # Dual task WM load 2 stop trials
DUAL_TASK_WM4_N_STOP = 48  # Dual task WM load 4 stop trials

# Test parameters
P = 0.5  # Null hypothesis (chance level)
ALPHA = 0.05  # Significance level (two-tailed)


def calculate_threshold_rate(n_trials, p=P, alpha=ALPHA):
    """Calculate binomial test threshold as a rate.
    
    Args:
        n_trials: Number of trials
        p: Null hypothesis probability (0.5)
        alpha: Significance level (0.05)
    
    Returns:
        tuple: (min_rate, max_rate) rounded to 3 decimal places
    """
    min_count = int(binom.ppf(alpha/2, n_trials, p))
    max_count = int(binom.ppf(1 - alpha/2, n_trials, p))
    min_rate = round(min_count / n_trials, 3)
    max_rate = round(max_count / n_trials, 3)
    return min_rate, max_rate


def main():
    """Calculate and display all binomial test thresholds."""
    
    print("=" * 80)
    print("BINOMIAL TEST THRESHOLDS FOR STOP TRIALS")
    print("=" * 80)
    print(f"\nNull Hypothesis: p = {P} (chance level)")
    print(f"Significance Level: α = {ALPHA} (two-tailed, 95% CI)")
    print("\nThese test whether participants' FAILED INHIBITION rates on stop")
    print("trials deviate significantly from 50%, which would indicate:")
    print("  - Too low: Always stopping (not following staircase)")
    print("  - Too high: Never stopping (always responding)")
    
    print("\n" + "=" * 80)
    print("STOP SIGNAL TASK")
    print("=" * 80)
    
    min_rate, max_rate = calculate_threshold_rate(STOP_SIGNAL_N_STOP)
    print(f"\nStop trials: n = {STOP_SIGNAL_N_STOP}")
    print(f"  Expected failed inhibitions (50%): {STOP_SIGNAL_N_STOP * P:.0f}")
    print(f"  Acceptable range: {int(binom.ppf(ALPHA/2, STOP_SIGNAL_N_STOP, P))} - {int(binom.ppf(1 - ALPHA/2, STOP_SIGNAL_N_STOP, P))} responses")
    print(f"  Rate thresholds: {min_rate:.4f} - {max_rate:.4f}")
    
    print("\n" + "=" * 80)
    print("DUAL TASK - BY WORKING MEMORY LOAD")
    print("=" * 80)
    print("\nNote: We test each WM load separately (not overall) to detect")
    print("load-specific performance issues.")
    
    # WM Load 0
    print(f"\n### WM Load 0 ###")
    min_rate, max_rate = calculate_threshold_rate(DUAL_TASK_WM0_N_STOP)
    print(f"Stop trials: n = {DUAL_TASK_WM0_N_STOP}")
    print(f"  Expected failed inhibitions (50%): {DUAL_TASK_WM0_N_STOP * P:.0f}")
    print(f"  Acceptable range: {int(binom.ppf(ALPHA/2, DUAL_TASK_WM0_N_STOP, P))} - {int(binom.ppf(1 - ALPHA/2, DUAL_TASK_WM0_N_STOP, P))} responses")
    print(f"  Rate thresholds: {min_rate:.4f} - {max_rate:.4f}")
    
    # WM Load 2
    print(f"\n### WM Load 2 ###")
    min_rate, max_rate = calculate_threshold_rate(DUAL_TASK_WM2_N_STOP)
    print(f"Stop trials: n = {DUAL_TASK_WM2_N_STOP}")
    print(f"  Expected failed inhibitions (50%): {DUAL_TASK_WM2_N_STOP * P:.0f}")
    print(f"  Acceptable range: {int(binom.ppf(ALPHA/2, DUAL_TASK_WM2_N_STOP, P))} - {int(binom.ppf(1 - ALPHA/2, DUAL_TASK_WM2_N_STOP, P))} responses")
    print(f"  Rate thresholds: {min_rate:.4f} - {max_rate:.4f}")
    
    # WM Load 4
    print(f"\n### WM Load 4 ###")
    min_rate, max_rate = calculate_threshold_rate(DUAL_TASK_WM4_N_STOP)
    print(f"Stop trials: n = {DUAL_TASK_WM4_N_STOP}")
    print(f"  Expected failed inhibitions (50%): {DUAL_TASK_WM4_N_STOP * P:.0f}")
    print(f"  Acceptable range: {int(binom.ppf(ALPHA/2, DUAL_TASK_WM4_N_STOP, P))} - {int(binom.ppf(1 - ALPHA/2, DUAL_TASK_WM4_N_STOP, P))} responses")
    print(f"  Rate thresholds: {min_rate:.4f} - {max_rate:.4f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY FOR exclusion_criteria.txt")
    print("=" * 80)
    
    print("\nCopy these values to exclusion_criteria.txt:")
    print("\nMetric names (header):")
    print("stop_fail_rate | dual_stop_fail_rate_wm0 | dual_stop_fail_rate_wm2 | dual_stop_fail_rate_wm4")
    
    # Calculate all thresholds
    stop_min, stop_max = calculate_threshold_rate(STOP_SIGNAL_N_STOP)
    wm0_min, wm0_max = calculate_threshold_rate(DUAL_TASK_WM0_N_STOP)
    wm2_min, wm2_max = calculate_threshold_rate(DUAL_TASK_WM2_N_STOP)
    wm4_min, wm4_max = calculate_threshold_rate(DUAL_TASK_WM4_N_STOP)
    
    print("\nMin values:")
    print(f"{stop_min:.3f} | {wm0_min:.3f} | {wm2_min:.3f} | {wm4_min:.3f}")
    
    print("\nMax values:")
    print(f"{stop_max:.3f} | {wm0_max:.3f} | {wm2_max:.3f} | {wm4_max:.3f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

