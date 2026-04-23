# CLAUDE.md — Instructions for AI Assistants

For project overview, directory structure, pipeline, and conventions, see **README.md**.

## Pipeline quick reference

New server exports arrive as battery JSON files in `raw_data/` (`battery_<id>_<timestamp>.json`). Full run order:

```bash
unpack-battery                      # Step 1: battery JSON → per-subject CSVs in preprocessed_data/
approved-sync <pid1> <pid2> ...     # Step 1.5: curate approved/ subset (symlinks)
clean_shape                         # Step 2: reshape to trial-level, combine across subjects
subjectwise_metrics                 # Step 3: participant-level metrics
flag_to_exclude                     # Step 4: apply exclusion_criteria.txt → __flagged CSVs
```

`approved-sync` maintains `data/preprocessed_data/approved/<pid>` as symlinks to the canonical `data/preprocessed_data/<pid>`. The pipeline prefers `approved/` when non-empty (via `config.active_preprocessed_dir()`), so running `approved-sync` with each new reviewer list is enough to scope every downstream step. Default `approved-sync` is **replace** semantics — pass the full current list each batch. Use `--add` to only append.

`unpack-battery` with no args processes all `battery_*.json` in `raw_data/`; pass a path to target one file. See README for the battery JSON schema and the per-file (`preprocess`) alternative.

**Task-agnostic pipeline:** `clean_shape` and `subjectwise_metrics` both discover tasks from the filesystem — no exp_id is hardcoded. `clean_shape` produces one `all_participants_reshaped_data_<exp_name>.csv` per exp_name found in `data/preprocessed_data/<pid>/<exp_name>/`; `subjectwise_metrics` globs those and routes each to `calculate_wm_metrics` (if `'wm'` in name) or `calculate_stop_signal_metrics`. Output: `<exp_name>_metrics.csv`. Adding a new exp_id requires no code changes unless the `wm`-in-the-name heuristic fails for it.

This file documents style and formatting instructions to follow when creating or editing notebooks in this project.

---

## Notebook style guide

### Structure & cell ordering

Every notebook follows this structure, without exception:

1. **Cell 0 (Markdown)** — Title (`# Title: Subtitle`), one-line description, then `## Table of Contents` with a numbered list matching the section headers below
2. **Cell 1 (Code)** — All imports, config initialization, path definitions, and data loading
3. **Cell 2–3 (Markdown + Code)** — The within-subject CI helper function
4. **Remaining cells (alternating Markdown + Code)** — One markdown header per code section, numbered to match the TOC

Each markdown header cell precedes exactly one code cell. The markdown cell names and briefly describes what the code below does — including any statistical assumptions being tested or corrections applied.

---

### Imports cell

Always the very first code cell. Format:

```python
#imports

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import scipy.stats as stats

from stop_wm.config import ProjectConfig

# Initialize config — go up one level from notebooks/ to project root
project_root = Path.cwd().parent
config = ProjectConfig(project_root=project_root)

# Pathing
trial_wise_data_wm_path = config.results_dir / 'post_qc_stop_signal_wm_trials.csv'
trial_wise_data_stop_path = config.results_dir / 'post_qc_stop_signal_trials.csv'
subject_wise_metrics_wm_path = config.results_dir / 'post_qc_stop_signal_wm_metrics.csv'
subject_wise_metrics_stop_path = config.results_dir / 'post_qc_stop_signal_metrics.csv'

# Load data
trial_wise_data_wm = pd.read_csv(trial_wise_data_wm_path)
trial_wise_data_stop = pd.read_csv(trial_wise_data_stop_path)
metrics_data_wm = pd.read_csv(subject_wise_metrics_wm_path)
metrics_data_stop = pd.read_csv(subject_wise_metrics_stop_path)
```

Rules:
- Imports grouped (stdlib → third-party → project), no blank lines between groups
- Paths always go through `config`, never hardcoded
- Data loaded in the same cell as path definitions

---

### Section headers

**In markdown cells:** `## N. Section Name` — H2, numbered, matching the TOC entry exactly.

**In code cells:** major subsections delimited with `# === SECTION TITLE ===`. No other header style.

**Printed to console:** `print("="*70)` followed by the section title on the next line. Use `print("="*70)` again as a closing divider for long sections.

---

### Within-subject confidence intervals

The Cousineau (2005) + Morey (2008) within-subject CI function is defined in every notebook that produces visualizations. It is always the second numbered section. Use the canonical implementation from `analysis_notebook.ipynb` — do not simplify or modify it. Function signature: `calculate_within_subject_ci(data_matrix, confidence_level=0.95)`.

---

### Statistical output format

Always report: statistic, degrees of freedom, p-value, effect size, and a plain-English significance statement. Use this format:

```
t(df) = X.XXXX, p = 0.XXXXXX
Significant: Yes (α = 0.05)
Effect size (Cohen's d): X.XXXX (medium effect)
```

For RM ANOVAs, always run Mauchly's sphericity test first and apply Greenhouse-Geisser correction if violated. For pairwise comparisons, apply Bonferroni correction and state it explicitly. BIC model comparisons should also report BF₁₀.

Participant counts with complete data are always printed at the start of any analysis section: `Participants with complete data for all conditions: N`.

---

### Data filtering pattern

Use this pattern consistently when aligning across datasets:

```python
participants_both = list(set(metrics_data_stop['prolific_id']) &
                         set(metrics_data_wm['prolific_id']))

stop_data = metrics_data_stop[metrics_data_stop['prolific_id'].isin(participants_both)]
stop_data = stop_data.dropna(subset=['target_column'])

wm_data = metrics_data_wm[metrics_data_wm['prolific_id'].isin(participants_both)]
wm_data = wm_data.dropna(subset=['wm_col_1', 'wm_col_2'])

common_participants = list(set(stop_data['prolific_id']) & set(wm_data['prolific_id']))
stop_data_filtered = stop_data[stop_data['prolific_id'].isin(common_participants)]
wm_data_filtered = wm_data[wm_data['prolific_id'].isin(common_participants)]
```

---

### Commission vs. omission errors

These are distinct and must not be conflated. Definitions:

- **Omission error**: `memory_recognition_rt` is NaN (no response given). Even though `memory_recognition_correct_trial` is coded as 0 for these rows, they are not errors of commission.
- **Commission error**: `memory_recognition_correct_trial == 0` AND `memory_recognition_rt.notna()` (a response was given but was wrong).

Whenever grouping by "probe correct vs. probe incorrect," document explicitly which error types are included. If a sequential-dependency analysis uses `prev_probe_correct == 0`, add a comment noting whether omissions are included and why, or filter them out: `.loc[trial_data['memory_recognition_rt'].notna()]`.

---

### Figures

**Figure size:** `(10, 6)` for single plots; `(14, 5)` for two horizontal subplots; `(18, 5)` for three.

**Titles:** `fontsize=14, fontweight='bold'`. Subtitles (e.g. `\n(Within-Subject 95% CI)`) appended with a newline inside the title string.

**Axis labels:** `fontsize=13, fontweight='bold'`.

**Tick labels:** `fontsize=11`.

**Grid:** always `ax.grid(True, alpha=0.3, axis='y')` — light horizontal gridlines only.

**Always call** `plt.tight_layout()` then `plt.show()`.

**Color palettes:**

| Use case | Colors |
|---|---|
| 3 WM load conditions (0, 2, 4) | `['#2E86AB', '#A23B72', '#F18F01']` |
| Simple Stop + 3 WM loads | `['#C1121F', '#2E86AB', '#A23B72', '#F18F01']` |
| Two conditions (e.g. correct/incorrect) | `['#2E86AB', '#A23B72']` |
| Omission comparison | `['#E07A5F', '#81B29A']` |

**Bar plot configuration:**

```python
bars = ax.bar(x_pos, all_means, yerr=all_cis, capsize=8,
              alpha=0.8, color=colors, edgecolor='black', linewidth=1.5,
              error_kw={'linewidth': 2, 'ecolor': 'black'})
```

Always add value labels above bars:

```python
for bar, mean, ci in zip(bars, all_means, all_cis):
    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + ci + offset,
            f'{mean:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
```

Add a chance line (`y=0.5`) for accuracy plots and a reference line for baseline conditions:

```python
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
ax.axhline(y=simple_stop_mean, color='#C1121F', linestyle='--', linewidth=2.5,
           label='Simple Stop Reference')
```

**Histograms:** always add mean (red) and median (orange) reference lines with legend labels.

---

### Variable naming

| Suffix / pattern | Meaning |
|---|---|
| `trial_wise_data_{type}` | Trial-level DataFrame |
| `subject_wise_metrics_{type}` or `metrics_data_{type}` | Participant-level aggregated metrics |
| `{metric}_matrix` | Wide-format DataFrame: rows = participants, columns = conditions |
| `df_long_{metric}` | Long-format DataFrame for pingouin/statsmodels |
| `{metric}_means` | Array of condition means |
| `{metric}_cis` or `all_cis` | Array of CI half-widths |
| `common_participants` | Filtered list of IDs with complete data across datasets |

---

### Comments

Comments explain intent and methodology, not just mechanics. Steps in multi-step procedures get numbered comments (`# Step 1: ...`). Never leave a complex operation without a note on why it's done that way.

---

### Deprecated notebooks

Obsolete notebooks go in `notebooks/deprec/`. They are not deleted.
