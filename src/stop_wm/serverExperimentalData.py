import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from stop_wm.utils import load_json
from stop_wm.config import ProjectConfig


@dataclass
class ServerExperimentalData:
    """Represents experimental data for a single participant session."""

    subject: str
    battery_id: int
    study_collection_id: int
    fname: str
    exp_name: str
    date_time: Optional[str] = None
    prolific_id: Optional[str] = None
    trialdata: Optional[pd.DataFrame] = None

    @classmethod
    def from_row(cls, row: pd.Series, raw_data_dir: Path) -> 'ServerExperimentalData':
        """Create ServerExperimentalData instance from a unified.csv row."""
        fname_path = raw_data_dir / 'results_export' / row['fname']
        data = load_json(fname_path)

        trialdata_raw = data.get('trialdata')
        if isinstance(trialdata_raw, str):
            trialdata_raw = json.loads(trialdata_raw)

        trialdata = pd.DataFrame(trialdata_raw) if trialdata_raw else None

        return cls(
            subject=row['subject'],
            battery_id=row['battery_id'],
            study_collection_id=row['study_collection_id'],
            fname=row['fname'],
            exp_name=row['exp_name'],
            date_time=data.get('dateTime'),
            prolific_id=data.get('prolific_id'),
            trialdata=trialdata,
        )

    @classmethod
    def from_json_file(cls, json_file_path: Path) -> 'ServerExperimentalData':
        """Create ServerExperimentalData instance directly from a JSON file."""
        data = load_json(json_file_path)

        # Extract experiment name from filename
        # Format: sub-{prolific_id}_battery-{battery}_order-{order}_task-{exp_name}_asgn-{asgn}_data.json
        filename = json_file_path.stem
        exp_name = None
        battery_id = None
        
        # Find task name by looking for pattern between 'task-' and '_asgn-'
        if 'task-' in filename and '_asgn-' in filename:
            start_idx = filename.find('task-') + 5  # 5 = len('task-')
            end_idx = filename.find('_asgn-')
            exp_name = filename[start_idx:end_idx]
        
        # Extract battery ID
        filename_parts = filename.split('_')
        for part in filename_parts:
            if part.startswith('battery-'):
                try:
                    battery_id = int(part.replace('battery-', ''))
                except ValueError:
                    battery_id = 0

        if not exp_name:
            raise ValueError(f'Could not extract experiment name from {json_file_path}')

        # Parse trial data
        trialdata_raw = data.get('trialdata')
        if isinstance(trialdata_raw, str):
            trialdata_raw = json.loads(trialdata_raw)

        trialdata = pd.DataFrame(trialdata_raw) if trialdata_raw else None

        return cls(
            subject=data.get('prolific_id', ''),
            battery_id=battery_id or 0,
            study_collection_id=68,  # Default for stop+WM study
            fname=json_file_path.name,
            exp_name=exp_name,
            date_time=data.get('dateTime'),
            prolific_id=data.get('prolific_id'),
            trialdata=trialdata,
        )
    # the definition of task type -> csv name below
    def get_task_type(self) -> str:
        """Get the task type from the experiment name."""
        return self.exp_name

    def get_output_path(self, base_dir: Path) -> Path:
        """Generate the output path for this experimental data."""
        outdir = base_dir / self.prolific_id / self.exp_name
        outdir.mkdir(parents=True, exist_ok=True)
        return (
            outdir
            / f'sub-{self.prolific_id}_task-{self.exp_name}_date-{self.date_time}.csv'
        )

    def save(self, base_dir: Path) -> None:
        """Save the experimental data to CSV."""
        if self.trialdata is None or self.trialdata.empty:
            logging.warning(f'No trial data for {self.fname}')
            return

        output_path = self.get_output_path(base_dir)
        self.trialdata.to_csv(output_path, index=False)
        logging.info(f'Saved preprocessed data to {output_path}')


def process_all_json_files():
    """Process all JSON files and save as CSV."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize configuration
    config = ProjectConfig()
    raw_data_dir = config.raw_data_dir
    preprocessed_data_dir = config.preprocessed_data_dir
    
    logging.info('Starting JSON to CSV preprocessing...')
    
    # Find all JSON files
    json_files = list(raw_data_dir.glob('**/*.json'))
    logging.info(f'Found {len(json_files)} JSON files')
    
    if not json_files:
        logging.error(f'No JSON files found in {raw_data_dir}')
        return
    
    # Process each file
    processed_count = 0
    failed_count = 0
    
    for json_file in json_files:
        try:
            # Create ServerExperimentalData instance
            exp_data = ServerExperimentalData.from_json_file(json_file)
            
            # Save to CSV
            exp_data.save(preprocessed_data_dir)
            processed_count += 1
            
        except Exception as e:
            logging.error(f'Failed to process {json_file.name}: {e}')
            failed_count += 1
    
    logging.info(f'Preprocessing complete!')
    logging.info(f'Successfully processed: {processed_count} files')
    logging.info(f'Failed to process: {failed_count} files')
    logging.info(f'Preprocessed data saved to: {preprocessed_data_dir}')


if __name__ == '__main__':
    process_all_json_files()
