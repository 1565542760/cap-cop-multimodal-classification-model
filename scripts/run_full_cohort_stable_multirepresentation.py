"""Full-cohort entry point for stable multi-representation CT."""

import sys
from run_full_cohort_whole_chest_representations import main


if __name__ == "__main__":
    # Use the existing full-cohort implementation without changing any model rule.
    if not any(arg == "--phase" or arg.startswith("--phase=") for arg in sys.argv[1:]):
        sys.argv.extend(["--phase", "stable_model"])
    raise SystemExit(main())
