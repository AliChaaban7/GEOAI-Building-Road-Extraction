"""
run_master_optuna.py

Launcher for Buildings Master Optuna.

Master Optuna keeps model family, source, split, and seed fixed while
searching the model-specific space defined in:
    Buildings/config/optional/master_optuna_search_space.json

Each trial keeps its best Validation-IoU checkpoint. The exact winning
checkpoint becomes the final Master Optuna model; there is no second training.
External test data are never accessed during the study.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BUILDINGS_ROOT / "config" / "experiment.json"
DEFAULT_SEARCH_SPACE = (
    BUILDINGS_ROOT
    / "config"
    / "optional"
    / "master_optuna_search_space.json"
)

if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDINGS_ROOT))

from src.optional.master_optuna import run_master_optuna_study
from src.pipeline.model_registry import normalize_model_type


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model-level Master Optuna for the Buildings module."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
        help="Experiment config. Model and source are fixed from this file.",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default=str(DEFAULT_SEARCH_SPACE),
        help="Master Optuna JSON search-space file.",
    )
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--trial-epochs", type=int, default=40)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()
    search_space_path = Path(args.search_space).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    if not search_space_path.exists():
        raise FileNotFoundError(
            f"Master Optuna search-space file not found: {search_space_path}"
        )

    config = load_json(config_path)
    config["model_type"] = normalize_model_type(config.get("model_type", ""))

    if not config.get("source_type"):
        raise ValueError("Master Optuna requires source_type in the experiment config.")

    if not config.get("experiment_id"):
        raise ValueError("Master Optuna requires experiment_id in the experiment config.")

    if int(args.trials) < 1:
        raise ValueError("--trials must be >= 1")
    if int(args.trial_epochs) < 1:
        raise ValueError("--trial-epochs must be >= 1")

    config["use_master_optuna"] = True
    config["use_optuna"] = False

    selection = run_master_optuna_study(
        base_config=config,
        search_space_path=search_space_path,
        n_trials=int(args.trials),
        trial_epochs=int(args.trial_epochs),
    )

    print("\nMASTER OPTUNA RESULT")
    print("--------------------")
    print(f"Experiment: {selection['experiment_id']}")
    print(f"Best Trial: {selection['best_trial']}")
    print(
        "BEST Validation IoU: "
        f"{selection['best_validation_iou_percent']:.2f}%"
    )
    print(f"Best Epoch: {selection.get('best_epoch')}")
    print(f"Winning Model: {selection['best_model_path']}")
    print("Second Training of Winner: NO")
    print("External Test Accessed: NO")


if __name__ == "__main__":
    main()
