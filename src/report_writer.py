"""Write test run results as JSON files."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import TestRunResult


def write_report(
    result: TestRunResult,
    output_dir: Path | None = None,
) -> str:
    """Write a TestRunResult as a JSON file.

    Returns the path of the written file.
    """
    base = output_dir or Path("output/results")
    base.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{result.persona_id}_{timestamp}.json"
    file_path = base / filename

    data = result.to_json_dict()
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return str(file_path)
