# Experiment 3: Nested Stop Signal + Working Memory (High Load)

Context for collaborators and AI assistants.

## Project overview

Experiment 3 replicates the **nested design** from Experiment 1 (stop signal embedded between memory encoding and probe) but with a **higher working memory load** — set sizes of 2, 4, and 6 letters rather than 0, 2, and 4 (`exp_id: stop_signal_wm_task`).

The experiment also includes a **control block** using a simple stop-signal task with shapes only (no memory load), `exp_id: stop_signalc`. Task code for the control block lives in `experiment_3_task/stop_signal/`.

- Task code: `experiment_3_task/stop_signal_wm_task/`
  (GitHub: lynde-wolf/stop_signal_nested_memory_highload_task)
- Control block task: `experiment_3_task/stop_signal/`
- Pilot data: `raw_data/pilot_data/`

This analysis codebase is currently being set up to mirror `experiment_1/` and `experiment_2/`.

## Directory structure

```
experiment_3/
├── src/stop_wm/            # Core analysis package (to be populated)
│   └── config.py           # Path configuration (reads from .env)
├── tests/                  # pytest test suite
├── notebooks/              # Exploratory analysis notebooks
├── raw_data/               # Raw data from experiment server — NOT on git
│   └── pilot_data/         # Pilot run data
├── data/                   # All pipeline outputs — NOT on git
│   ├── preprocessed_data/
│   ├── results/
│   └── figures/
├── pyproject.toml          # Dependencies (managed with uv)
├── .env.example            # Template for local path config — copy to .env
└── .gitignore
```

## Environment setup

```bash
uv venv
uv sync
cp .env.example .env   # then edit .env with your local paths if needed
```

## Path configuration

All paths flow through `src/stop_wm/config.py`. Two env vars can be set in `.env`:
- `EXP3_DATADIR` — where processed outputs go (defaults to `./data`)
- `EXP3_RAWDATADIR` — where raw data lives (defaults to `./raw_data`)

Raw data is never committed to git.

## Key conventions

- Mirror the conventions from `experiment_1/` and `experiment_2/` where applicable.
- Raw data is read-only. Never modify files in `raw_data/`.
- Notebooks in `notebooks/` are for exploration — analysis logic lives in `src/`.
- Tests live in `tests/` and run with `pytest`.

## Data source

Data is collected via Prolific and hosted on an Experiment Factory server (study collection 134). Raw exports arrive as battery JSON files containing multiple experiments and subjects.
