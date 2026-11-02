"""Writing result dictionaries to JSON."""
from __future__ import annotations

import json
import os


def write_json(data: dict, output_dir: str, filename: str) -> str:
    """Write `data` as indented JSON to output_dir/filename; returns the path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path
