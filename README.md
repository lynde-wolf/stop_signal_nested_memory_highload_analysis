# Experiment 3: Nested Stop Signal + Working Memory (High Load)

Context for collaborators and AI assistants.

## Project overview

Experiment 3 replicates the **nested design** from Experiment 1 (stop signal embedded between memory encoding and probe) but with a **higher working memory load** — set sizes of 2, 4, and 6 letters rather than 0, 2, and 4 (`exp_id: stop_signal_wm_task_e3`).

The experiment also includes a **control block** using a simple stop-signal task with shapes only (no memory load), `exp_id: simple_stop_signal_e3`. Task code for the control block lives in `experiment_3_task/stop_signal/`.

- Task code: `experiment_3_task/stop_signal_wm_task/`
  (GitHub: lynde-wolf/stop_signal_nested_memory_highload_task)
- Control block task: `experiment_3_task/stop_signal/`
- Pilot data: previously ingested subject folders sit under `data/preprocessed_data/` (older pilot run used exp_ids `stop_signal` / `stop_signal_wm_task`).

Data is collected via Prolific and hosted on an Experiment Factory server. Raw exports arrive as **battery JSON files** containing multiple experiments × subjects.

## Directory structure

```
experiment_3/
├── src/stop_wm/                  # Core analysis package
│   ├── config.py                 # Path configuration (reads .env)
│   ├── unpack_battery.py         # Battery JSON → per-subject CSVs (Step 1)
│   ├── serverExperimentalData.py # Per-file JSON → per-subject CSVs (Step 1 alt)
│   ├── clean_shape.py            # Raw CSVs → reshaped trial-level CSVs (Step 2)
│   ├── flag_to_exclude.py        # QC exclusion flagging (Step 3)
│   ├── create_post_qc_datasets.py# Apply exclusions → final trial datasets (Step 3)
│   ├── subjectwise_metrics.py    # Trial data → participant-level metrics (Step 4)
│   ├── calculate_binomial_thresholds.py
│   ├── bic_bayes.py
│   ├── within_subject_ci.py
│   └── utils.py
├── tests/                        # pytest test suite
├── notebooks/                    # Exploratory analysis notebooks
├── raw_data/                     # Raw data — NOT on git
│   └── battery_<id>_<yyyy.mm.dd.hhmmss>.json   # Battery exports (batches)
├── data/                         # All pipeline outputs — NOT on git
│   ├── preprocessed_data/        # Per-subject CSVs, one folder per prolific_id
│   │   └── <prolific_id>/<exp_name>/sub-<pid>_task-<exp>_date-<ts>.csv
│   ├── results/
│   └── figures/
├── pyproject.toml                # Dependencies (uv), CLI entry-points
├── .env.example
└── .gitignore
```

## Environment setup

```bash
uv venv
uv sync
uv pip install -e .          # register CLI entry-points
cp .env.example .env         # optional: edit paths
```

## Preprocessing pipeline

### Step 1: Unpack battery JSON → per-subject CSVs (CLI: `unpack-battery`)

When new data arrives as a battery JSON (`raw_data/battery_<id>_<timestamp>.json`):

```bash
unpack-battery                                        # unpacks ALL battery_*.json in raw_data/
unpack-battery raw_data/battery_313_<timestamp>.json  # or a specific file
```

Battery JSON format: top-level keys are experiment names (e.g. `simple_stop_signal_e3`, `stop_signal_wm_task_e3`, `race_ethnicity_RMR_survey_rdoc`); each maps to `[{"subject": "...", "data": "<python-dict-literal>"}, ...]`. The `data` string is a Python dict literal (parsed with `ast.literal_eval`) whose `trialdata` field is a JSON-encoded list of trials.

Output: one CSV per subject per experiment at
`data/preprocessed_data/<prolific_id>/<exp_name>/sub-<pid>_task-<exp>_date-<ts>.csv`.

### Step 1 (alt): Per-subject JSONs → CSVs (CLI: `preprocess`)

For the older per-file JSON format (filename pattern `sub-<pid>_battery-<n>_..._task-<exp>_asgn-<n>_data.json`). Same output layout.

### Step 1.5: Curate approved set (CLI: `approved-sync`)

Prolific assignments need manual review before their data enters analysis. The pipeline operates on a curated subset via `data/preprocessed_data/approved/`, which contains **symlinks** (not copies) back to the canonical subject folders at `data/preprocessed_data/<pid>/`.

```bash
approved-sync <pid1> <pid2> ...           # inline IDs (default: --replace)
approved-sync --from-file approved_ids.txt
approved-sync --add <pid>                 # add without removing existing
```

Default behavior is **replace**: symlinks for IDs not in the supplied list are removed, so passing the complete current approved list each batch keeps the set in sync. `clean_shape` and `subjectwise_metrics` automatically operate on `approved/` when it is non-empty, and fall back to the full `preprocessed_data/` set when it is empty or missing.

### Step 2: Clean + reshape (CLI: `clean_shape`)

Reads per-subject CSVs and produces trial-level reshaped files per subject plus combined files:
- `data/results/all_participants_reshaped_data_<exp_name>.csv`

### Step 3: QC and exclusion

`flag_to_exclude.py` + `create_post_qc_datasets.py`. Produces `post_qc_stop_signal_*` files in `data/results/`.

### Step 4: Subject-wise metrics (CLI: `subjectwise_metrics`)

Computes participant-level metrics (SSRT, accuracy, RT, etc.) from post-QC trial data.

### Full run

```bash
unpack-battery           # Step 1
clean_shape              # Step 2
# Step 3 — run QC modules as needed
subjectwise_metrics      # Step 4
```

## Task discovery

`clean_shape` and `subjectwise_metrics` are **task-agnostic** — they discover tasks from the file system rather than hardcoding exp_ids.

- `clean_shape` walks `data/preprocessed_data/<pid>/<exp_name>/` and produces one combined CSV per exp_name: `data/results/all_participants_reshaped_data_<exp_name>.csv`.
- `subjectwise_metrics` globs those combined CSVs and routes each to a metrics function: **tasks with `wm` in the name** → `calculate_wm_metrics`; everything else → `calculate_stop_signal_metrics`. Output: `data/results/<exp_name>_metrics.csv`.

This means batteries using the `_e3` suffix (`simple_stop_signal_e3`, `stop_signal_wm_task_e3`) and older pilot tasks (`stop_signal`, `stop_signal_wm_task`) both flow through the pipeline without code changes. New exp_ids work automatically as long as the `wm`-in-the-name heuristic routes them correctly.

## Path configuration

All paths flow through `src/stop_wm/config.py` (`ProjectConfig`). Env vars in `.env`:
- `STOPWM_DATADIR` — processed outputs (default `./data`)
- `STOPWM_RAWDATADIR` — raw data (default `./raw_data`)

Raw data is never committed to git.

## Key conventions

- Raw data is read-only. Never modify files in `raw_data/`.
- Notebooks are for exploration — analysis logic lives in `src/`.
- Tests live in `tests/` and run with `pytest`.
- Mirror conventions from `experiment_1/` and `experiment_2/` where applicable.

## Module inventory

| Module | Purpose | CLI |
|---|---|---|
| `config.py` | Path configuration | — |
| `unpack_battery.py` | Battery JSON → per-subject CSVs | `unpack-battery` |
| `approved.py` | Curate `preprocessed_data/approved/` symlinks | `approved-sync` |
| `serverExperimentalData.py` | Per-file JSON → per-subject CSVs | `preprocess` |
| `clean_shape.py` | Raw CSVs → reshaped trial-level data | `clean_shape` |
| `flag_to_exclude.py` | QC exclusion flagging | — |
| `create_post_qc_datasets.py` | Apply exclusions → final datasets | — |
| `subjectwise_metrics.py` | Trial data → participant metrics | `subjectwise_metrics` |
| `calculate_binomial_thresholds.py` | Chance-level thresholds | — |
| `bic_bayes.py` | BIC / BF₁₀ helpers | — |
| `within_subject_ci.py` | Cousineau-Morey within-subject CIs | — |
| `utils.py` | `load_json` and shared helpers | — |
