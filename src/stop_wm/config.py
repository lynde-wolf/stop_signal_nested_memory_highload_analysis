"""
Configuration file for data paths and settings.

Paths can be overridden via a .env file in the project root:
  STOPWM_DATADIR    - processed data outputs (results, figures, preprocessed)
  STOPWM_RAWDATADIR - raw participant JSON files (not committed to git)
"""

from dataclasses import dataclass
from pathlib import Path
import dotenv


@dataclass
class ProjectConfig:
    """Configuration class for project paths and settings."""

    data_dir: Path = None
    raw_data_dir: Path = None
    project_root: Path = None
    results_dir: Path = None
    preprocessed_data_dir: Path = None
    figures_dir: Path = None

    def __post_init__(self) -> None:
        """Create directories if they don't exist."""

        if self.project_root is None:
            self.project_root = Path.cwd()

        dotenv.load_dotenv()
        data_dir_env = dotenv.get_key(dotenv.find_dotenv(), 'STOPWM_DATADIR')
        raw_data_dir_env = dotenv.get_key(dotenv.find_dotenv(), 'STOPWM_RAWDATADIR')

        if self.data_dir is None and data_dir_env:
            print('Using STOPWM_DATADIR from .env file: ', data_dir_env)
            self.data_dir = Path(data_dir_env)
        elif self.data_dir is None:
            self.data_dir = self.project_root / 'data'

        if self.raw_data_dir is None and raw_data_dir_env:
            print('Using STOPWM_RAWDATADIR from .env file: ', raw_data_dir_env)
            self.raw_data_dir = Path(raw_data_dir_env)
        elif self.raw_data_dir is None:
            self.raw_data_dir = self.project_root / 'raw_data'

        if self.results_dir is None:
            self.results_dir = self.data_dir / 'results'

        if self.preprocessed_data_dir is None:
            self.preprocessed_data_dir = self.data_dir / 'preprocessed_data'

        if self.figures_dir is None:
            self.figures_dir = self.data_dir / 'figures'

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Note: raw_data_dir is intentionally not auto-created —
        # it is external data that must be downloaded separately.


if __name__ == '__main__':
    # Example usage
    config = ProjectConfig()
    print('Project Root:             ', config.project_root)
    print('Raw Data Directory:       ', config.raw_data_dir)
    print('Data Directory:           ', config.data_dir)
    print('Preprocessed Data Dir:    ', config.preprocessed_data_dir)
    print('Results Directory:        ', config.results_dir)
    print('Figures Directory:        ', config.figures_dir)