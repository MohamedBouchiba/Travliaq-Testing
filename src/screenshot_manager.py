"""Save and organize screenshots from browser-use agent history."""

import base64
from pathlib import Path
from typing import List


def save_screenshots(
    screenshots_b64: List[str],
    persona_id: str,
    run_id: str,
    output_base: Path | None = None,
) -> List[str]:
    """Decode base64 screenshots and save them to disk.

    Returns list of saved file paths (relative to project root).
    """
    if not screenshots_b64:
        return []

    base = output_base or Path("output/screenshots")
    run_dir = base / f"{persona_id}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for i, b64_data in enumerate(screenshots_b64):
        if not b64_data:
            continue
        try:
            img_data = base64.b64decode(b64_data)
            file_path = run_dir / f"step_{i:03d}.png"
            file_path.write_bytes(img_data)
            saved_paths.append(str(file_path))
        except Exception:
            continue

    return saved_paths
