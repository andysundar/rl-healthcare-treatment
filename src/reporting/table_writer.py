from __future__ import annotations

from pathlib import Path
import json
import pandas as pd


def write_csv_json(df: pd.DataFrame, csv_path: Path, json_path: Path):
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient='records', indent=2)


def write_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2))
