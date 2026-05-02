from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path | None = None) -> dict:
    config_path = path or ROOT / "config" / "project_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def month_sequence(start: int, end: int) -> list[int]:
    year = start // 100
    month = start % 100
    out: list[int] = []
    while year * 100 + month <= end:
        out.append(year * 100 + month)
        month += 1
        if month == 13:
            year += 1
            month = 1
    return out
