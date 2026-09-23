"""
run_augmented_threshold_search.py

Backward-compatible wrapper for the canonical shared threshold-search
runner:

    Buildings/scripts/run_threshold_search.py

Why this wrapper exists
-----------------------
Historically the project had a separate command named:

    run_augmented_threshold_search.py

However, threshold selection itself must NOT use augmentation on the
validation set.

Training augmentation may be ON or OFF in experiment.json, but:

    validation augmentation = OFF

for every model.

Therefore the canonical threshold implementation is now shared by:

    U-Net
    DeepLabV3 / DeepLabV3+
    SAM-LoRA

This wrapper preserves the old command/path so previous workflows and
notes continue to work, while avoiding a second duplicated threshold
implementation.

PowerShell
----------
The historical command still works:

    & $py "Buildings\\scripts\\run_augmented_threshold_search.py"

All command-line arguments are forwarded automatically because the
canonical runner parses the original sys.argv.
"""

from pathlib import Path
import sys


# ============================================================
# SCRIPT DIRECTORY
# ============================================================

SCRIPT_DIR = Path(
    __file__
).resolve().parent


if str(
    SCRIPT_DIR
) not in sys.path:

    sys.path.insert(
        0,
        str(
            SCRIPT_DIR
        ),
    )


# ============================================================
# CANONICAL RUNNER
# ============================================================

from run_threshold_search import (
    main,
)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=" * 76
    )

    print(
        "COMPATIBILITY NOTICE"
    )

    print(
        "=" * 76
    )

    print(
        "run_augmented_threshold_search.py now uses the canonical "
        "shared run_threshold_search.py implementation."
    )

    print(
        "Training augmentation may be ON, but validation augmentation "
        "remains OFF during threshold selection."
    )

    print(
        "=" * 76
    )

    print()


    main()
