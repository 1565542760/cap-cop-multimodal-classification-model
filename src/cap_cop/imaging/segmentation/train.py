from __future__ import annotations

import argparse

from cap_cop.imaging.segmentation.splits import verify_all_strict_oof_folds


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate locked lesion-segmentation training fold contract.")
    parser.add_argument("--check-folds-only", action="store_true", help="Do not train; only verify strict-OOF fold separation.")
    args = parser.parse_args()
    if args.check_folds_only:
        print(verify_all_strict_oof_folds())
        return
    raise SystemExit(
        "Training entry is intentionally disabled in this local public workspace. "
        "The locked publication package carries the 20 trained checkpoints for inference; "
        "full retraining requires the governed expert-label dataset and should be run only when explicitly authorised."
    )


if __name__ == "__main__":
    main()
