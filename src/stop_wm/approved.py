"""Manage the ``approved/`` subfolder of ``preprocessed_data``.

Prolific assignments need manual review before their data enters analysis.
Rather than copying or moving data around when a reviewer approves or rejects
a participant, we keep the canonical per-subject folders in
``data/preprocessed_data/<pid>/`` and maintain a curated set of **symlinks**
in ``data/preprocessed_data/approved/<pid>`` pointing back at those folders.

The downstream pipeline (``clean_shape``, ``subjectwise_metrics``) prefers
``approved/`` when it is non-empty, so the analysis always operates on the
reviewer-approved set without duplicating data on disk.

Usage
-----
From the CLI::

    approved-sync <pid1> <pid2> ...                  # pass IDs inline
    approved-sync --from-file approved_ids.txt       # one ID per line

``--replace`` (default) removes symlinks not in the supplied list; use
``--add`` to only add new approvals without touching existing ones.
"""

import argparse
import logging
from pathlib import Path

from stop_wm.config import ProjectConfig

logger = logging.getLogger(__name__)


def _read_ids_from_file(path: Path) -> list[str]:
    """Read one prolific_id per line; ignore blanks and '#'-comments."""
    ids = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        ids.append(line)
    return ids


def sync_approved(
    approved_ids: list[str],
    preprocessed_data_dir: Path,
    approved_data_dir: Path,
    replace: bool = True,
) -> dict:
    """Ensure ``approved_data_dir`` contains symlinks to each approved subject.

    Parameters
    ----------
    approved_ids : list of str
        Prolific IDs to approve.
    preprocessed_data_dir : Path
        Root of the per-subject data (symlink targets live here).
    approved_data_dir : Path
        Where the curated symlinks should be written. Typically
        ``preprocessed_data_dir / 'approved'``.
    replace : bool
        If True (default), remove existing approved symlinks whose ID is not
        in ``approved_ids``. If False, only add missing links.

    Returns
    -------
    dict
        Summary: ``added``, ``kept``, ``removed``, ``missing`` (IDs with no
        source folder), ``not_a_symlink`` (paths present but not symlinks).
    """
    approved_data_dir.mkdir(parents=True, exist_ok=True)
    requested = set(approved_ids)

    existing = {p.name: p for p in approved_data_dir.iterdir() if p.name != '.DS_Store'}

    added, kept, removed, missing, not_a_symlink = [], [], [], [], []

    for pid in sorted(requested):
        source = preprocessed_data_dir / pid
        link = approved_data_dir / pid

        if not source.exists():
            missing.append(pid)
            logger.warning(f'No source folder for {pid} at {source} - skipping')
            continue

        if link.exists() or link.is_symlink():
            if link.is_symlink():
                kept.append(pid)
                continue
            not_a_symlink.append(str(link))
            logger.warning(f'{link} exists and is not a symlink - leaving alone')
            continue

        # Relative symlink so the tree is portable
        link.symlink_to(Path('..') / pid)
        added.append(pid)
        logger.info(f'Approved: {pid}')

    if replace:
        for name, path in existing.items():
            if name not in requested and path.is_symlink():
                path.unlink()
                removed.append(name)
                logger.info(f'Un-approved: {name}')

    return {
        'added': added,
        'kept': kept,
        'removed': removed,
        'missing': missing,
        'not_a_symlink': not_a_symlink,
    }


def main() -> None:
    """CLI: ``approved-sync [pid ...] [--from-file FILE] [--add]``."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(
        description='Populate preprocessed_data/approved/ with symlinks to '
                    'reviewer-approved subject folders.',
    )
    parser.add_argument(
        'ids', nargs='*', help='Prolific IDs to approve (space-separated).',
    )
    parser.add_argument(
        '--from-file', type=Path, default=None,
        help='Read IDs from a file (one per line). Merged with inline IDs.',
    )
    parser.add_argument(
        '--add', action='store_true',
        help='Only add new approvals; do not remove existing symlinks not in the list.',
    )
    args = parser.parse_args()

    ids = list(args.ids)
    if args.from_file:
        ids.extend(_read_ids_from_file(args.from_file))

    if not ids:
        parser.error('Provide IDs inline or via --from-file.')

    config = ProjectConfig()
    stats = sync_approved(
        approved_ids=ids,
        preprocessed_data_dir=config.preprocessed_data_dir,
        approved_data_dir=config.approved_data_dir,
        replace=not args.add,
    )

    print(f'\nApproved set at {config.approved_data_dir}')
    print(f'  added:         {len(stats["added"])} {stats["added"] or ""}')
    print(f'  already ok:    {len(stats["kept"])}')
    print(f'  removed:       {len(stats["removed"])} {stats["removed"] or ""}')
    if stats['missing']:
        print(f'  missing src:   {len(stats["missing"])} {stats["missing"]}')
    if stats['not_a_symlink']:
        print(f'  conflicts:     {stats["not_a_symlink"]}')


if __name__ == '__main__':
    main()
