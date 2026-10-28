"""Export immutable cohort metadata, excluding all prediction columns."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cap_cop.data.model_contracts import read_stable_cohort_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Authorized legacy cohort artifact.")
    parser.add_argument("--output", type=Path, required=True, help="Local patient-level CSV; do not commit.")
    args = parser.parse_args()
    frame = read_stable_cohort_metadata(args.source, legacy=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False)
    print(f"Exported {len(frame)} metadata rows; no predictions exported.")


if __name__ == "__main__":
    main()
