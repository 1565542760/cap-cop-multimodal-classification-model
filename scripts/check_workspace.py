from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cap_cop.config import load_model_configs
from cap_cop.paths import project_root


def main() -> None:
    root = project_root()
    models = load_model_configs()
    report = {
        "project_root": str(root),
        "model_config_count": len(models),
        "blocked_configs": [model.name for model in models if "blocked" in model.status],
        "canonical_result_lock_pending_configs": [
            model.name for model in models if "canonical" in model.status and "lock_pending" in model.status
        ],
        "external_masks_dirs_present": [
            str(path.relative_to(root))
            for path in (root / "data" / "deidentified_release_candidate" / "external").rglob("masks")
            if path.is_dir()
        ],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
