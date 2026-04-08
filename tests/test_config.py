from pathlib import Path, PosixPath

from stop_wm.config import ProjectConfig


def test_ProjectConfig_instance():
    config = ProjectConfig(data_dir=Path('/tmp/data'))
    assert config.data_dir == PosixPath('/tmp/data')
    assert config.results_dir == PosixPath('/tmp/data/results')
    assert config.preprocessed_data_dir == PosixPath('/tmp/data/preprocessed_data')
    assert config.figures_dir == PosixPath('/tmp/data/figures')
