import json
from pathlib import Path


def load_json(fpath: Path) -> dict:
    """Load a JSON file into a dictionary.

    Args:
        fpath (Path): The path to the JSON file.

    Returns:
        dict: The dictionary loaded from the JSON file.
    """
    with open(fpath, 'r') as fp:
        return json.load(fp)
