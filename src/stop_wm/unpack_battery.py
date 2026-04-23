"""Unpack an Experiment Factory battery JSON export into per-subject preprocessed CSVs.

Battery exports from the experiment server are JSON files structured as:

    {
        "<experiment_name>": [
            {"subject": "<id>", "data": "<python-dict-literal-string>"},
            ...
        ],
        ...
    }

Each subject's ``data`` string is a Python dict literal (single-quoted keys)
containing ``trialdata`` (a JSON-encoded list of trial dicts), ``prolific_id``,
``dateTime``, and other session metadata.

This module:
  1. Parses the battery JSON.
  2. For every subject x experiment, extracts the trial data into a DataFrame.
  3. Saves each DataFrame as a CSV under
     ``<preprocessed_data_dir>/<prolific_id>/<exp_name>/
       sub-<prolific_id>_task-<exp_name>_date-<dateTime>.csv``
     -- the same layout that ``serverExperimentalData.py`` produces.

Usage
-----
CLI::

    unpack-battery raw_data/battery_313_2026.04.22.204751.json

Programmatically::

    from stop_wm.unpack_battery import unpack_battery
    from stop_wm.config import ProjectConfig

    config = ProjectConfig()
    stats = unpack_battery(
        json_path="raw_data/battery_313_2026.04.22.204751.json",
        preprocessed_data_dir=config.preprocessed_data_dir,
    )
"""

import ast
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from stop_wm.config import ProjectConfig

logger = logging.getLogger(__name__)


def _parse_subject_data(raw_data_string: str) -> dict:
    """Parse the single-quoted Python dict literal stored in ``data``."""
    try:
        return ast.literal_eval(raw_data_string)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(
            f'Could not parse subject data string '
            f'(first 120 chars: {raw_data_string[:120]!r})'
        ) from exc


def _extract_trialdata(subject_dict: dict) -> Optional[pd.DataFrame]:
    """Extract the trial-level DataFrame from a parsed subject dict."""
    trialdata_raw = subject_dict.get('trialdata')
    if trialdata_raw is None:
        return None

    if isinstance(trialdata_raw, str):
        trialdata_raw = json.loads(trialdata_raw)

    if not trialdata_raw:
        return None

    return pd.DataFrame(trialdata_raw)


def _save_subject_csv(
    df: pd.DataFrame,
    prolific_id: str,
    exp_name: str,
    date_time: str,
    preprocessed_data_dir: Path,
) -> Path:
    """Save a subject's trial data to CSV in the standard layout."""
    out_dir = preprocessed_data_dir / prolific_id / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f'sub-{prolific_id}_task-{exp_name}_date-{date_time}.csv'
    out_path = out_dir / fname
    df.to_csv(out_path, index=False)
    return out_path


def unpack_battery(
    json_path: str | Path,
    preprocessed_data_dir: Optional[Path] = None,
    experiments: Optional[list[str]] = None,
) -> dict:
    """Unpack a battery JSON export into per-subject preprocessed CSVs.

    Parameters
    ----------
    json_path : str or Path
        Path to the battery JSON file.
    preprocessed_data_dir : Path, optional
        Where to write CSVs. Defaults to ``ProjectConfig().preprocessed_data_dir``.
    experiments : list of str, optional
        If given, only unpack these experiment keys.

    Returns
    -------
    dict
        Summary statistics with keys ``battery_file``, ``experiments_processed``,
        ``subjects_saved``, ``subjects_skipped``, ``files_written``.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f'Battery JSON not found: {json_path}')

    if preprocessed_data_dir is None:
        config = ProjectConfig()
        preprocessed_data_dir = config.preprocessed_data_dir

    preprocessed_data_dir = Path(preprocessed_data_dir)
    preprocessed_data_dir.mkdir(parents=True, exist_ok=True)

    with open(json_path, 'r') as fh:
        battery_data = json.load(fh)

    if not isinstance(battery_data, dict):
        raise ValueError(
            f'Expected top-level dict in battery JSON, got {type(battery_data).__name__}'
        )

    exp_keys = experiments if experiments else list(battery_data.keys())

    saved_count = 0
    skipped_count = 0
    files_written: list[str] = []

    for exp_name in exp_keys:
        if exp_name not in battery_data:
            logger.warning(f'Experiment key {exp_name!r} not found in JSON - skipping')
            continue

        subjects = battery_data[exp_name]
        logger.info(
            f'Processing experiment {exp_name!r}: {len(subjects)} subject(s)'
        )

        for entry in subjects:
            subject_label = entry.get('subject', '<unknown>')
            try:
                subject_dict = _parse_subject_data(entry['data'])
            except (KeyError, ValueError) as exc:
                logger.error(
                    f'  Skipping subject {subject_label} in {exp_name}: {exc}'
                )
                skipped_count += 1
                continue

            prolific_id = subject_dict.get('prolific_id') or subject_label
            if not prolific_id:
                logger.warning(
                    f'  No prolific_id for subject {subject_label} - skipping'
                )
                skipped_count += 1
                continue

            df = _extract_trialdata(subject_dict)
            if df is None or df.empty:
                logger.warning(
                    f'  No trial data for {prolific_id} in {exp_name} - skipping'
                )
                skipped_count += 1
                continue

            date_time = str(subject_dict.get('dateTime', 'unknown'))

            out_path = _save_subject_csv(
                df, prolific_id, exp_name, date_time,
                preprocessed_data_dir,
            )
            files_written.append(str(out_path))
            saved_count += 1
            logger.info(f'  Saved {prolific_id}/{exp_name} ({len(df)} trials)')

    stats = {
        'battery_file': str(json_path),
        'experiments_processed': len(exp_keys),
        'subjects_saved': saved_count,
        'subjects_skipped': skipped_count,
        'files_written': files_written,
    }

    logger.info(
        f'Done - saved {saved_count} file(s), skipped {skipped_count}'
    )
    return stats


def main() -> None:
    """CLI entry-point: ``unpack-battery <json_path> [--experiments ...]``."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='Unpack a battery JSON export into preprocessed CSVs.',
    )
    parser.add_argument(
        'json_path',
        type=Path,
        nargs='?',
        default=None,
        help='Path to the battery JSON file. If omitted, all *.json in raw_data_dir are processed.',
    )
    parser.add_argument(
        '--experiments',
        nargs='*',
        default=None,
        help='Only unpack these experiment keys (default: all).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Override the preprocessed data output directory.',
    )
    args = parser.parse_args()

    if args.json_path is not None:
        targets = [args.json_path]
    else:
        config = ProjectConfig()
        targets = sorted(config.raw_data_dir.glob('battery_*.json'))
        if not targets:
            parser.error(f'No battery_*.json files found in {config.raw_data_dir}')

    total_saved = 0
    total_skipped = 0
    for target in targets:
        stats = unpack_battery(
            json_path=target,
            preprocessed_data_dir=args.output_dir,
            experiments=args.experiments,
        )
        total_saved += stats['subjects_saved']
        total_skipped += stats['subjects_skipped']
        print(f'\nSummary for {stats["battery_file"]}')
        print(f'  Experiments processed: {stats["experiments_processed"]}')
        print(f'  Subjects saved:        {stats["subjects_saved"]}')
        print(f'  Subjects skipped:      {stats["subjects_skipped"]}')
        print(f'  Files written:         {len(stats["files_written"])}')

    if len(targets) > 1:
        print(f'\nOverall: saved {total_saved}, skipped {total_skipped} across {len(targets)} batteries')


if __name__ == '__main__':
    main()
