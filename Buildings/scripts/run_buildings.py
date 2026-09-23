"""
Buildings manual controller — final simplified version.

User-facing stages:
    standard
    augmentation
    optuna
    master_optuna
    threshold
    post
    final

There is intentionally:
- no Auto Optimize
- no Smoke Test
- no visible Freeze stage

Scientific reproducibility is preserved because the FINAL stage performs
the technical freeze internally immediately before the external test.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys


BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BUILDINGS_ROOT.parent
SCRIPTS = BUILDINGS_ROOT / "scripts"
DEFAULT_CONFIG = BUILDINGS_ROOT / "config" / "experiment.json"

if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDINGS_ROOT))

from src.pipeline.model_registry import (
    get_model_spec,
    normalize_model_type,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one manual Buildings workflow stage."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG),
    )

    parser.add_argument(
        "--stage",
        choices=(
            "standard",
            "augmentation",
            "optuna",
            "master_optuna",
            "threshold",
            "post",
            "final",
        ),
        default="standard",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--trial-epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--final-epochs",
        type=int,
        default=40,
    )

    return parser.parse_args()


def load_json(path: Path):
    return json.loads(
        Path(path).read_text(encoding="utf-8")
    )


def run(command):
    subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        check=True,
    )


def experiment_output(config: dict) -> Path:
    return (
        BUILDINGS_ROOT
        / "outputs"
        / "experiments"
        / config["experiment_id"]
    )


def semantic_freeze_command(
    python: str,
    config_path: Path,
    config: dict,
):
    output = experiment_output(config)

    command = [
        python,
        str(SCRIPTS / "freeze_final_settings.py"),
        "--config",
        str(config_path),
        "--overwrite",
    ]

    threshold_summary = (
        output / "validation_threshold_summary.json"
    )
    post_summary = (
        output / "validation_postprocess_summary.json"
    )
    optuna_summary = (
        output / "optuna_selection_summary.json"
    )

    if threshold_summary.exists():
        command += [
            "--threshold-summary",
            str(threshold_summary),
        ]

    if post_summary.exists():
        command += [
            "--postprocess-summary",
            str(post_summary),
        ]

    if bool(config.get("use_optuna", False)):
        if not optuna_summary.exists():
            raise FileNotFoundError(
                "This experiment is marked as Optuna-based, but the "
                f"Optuna summary is missing: {optuna_summary}"
            )

        command += [
            "--optuna-summary",
            str(optuna_summary),
        ]

    return command


def maskrcnn_freeze_command(
    python: str,
    config_path: Path,
    config: dict,
):
    output = experiment_output(config)

    command = [
        python,
        str(SCRIPTS / "freeze_maskrcnn_settings.py"),
        "--config",
        str(config_path),
    ]

    threshold_summary = (
        output / "validation_threshold_summary.json"
    )
    post_summary = (
        output / "validation_postprocess_summary.json"
    )
    optuna_summary = (
        output / "optuna_selection_summary.json"
    )

    if threshold_summary.exists():
        command += [
            "--threshold-summary",
            str(threshold_summary),
        ]
    else:
        command += [
            "--score-threshold",
            "0.50",
            "--mask-threshold",
            "0.50",
        ]

    if post_summary.exists():
        command += [
            "--postprocess-summary",
            str(post_summary),
        ]

    if bool(config.get("use_optuna", False)):
        if not optuna_summary.exists():
            raise FileNotFoundError(
                "This experiment is marked as Optuna-based, but the "
                f"Optuna summary is missing: {optuna_summary}"
            )

        command += [
            "--optuna-summary",
            str(optuna_summary),
        ]

    return command


def run_semantic(
    args,
    python: str,
    config_path: Path,
    config: dict,
):
    if args.stage in {
        "standard",
        "augmentation",
    }:
        run(
            [
                python,
                str(SCRIPTS / "run_train.py"),
                "--config",
                str(config_path),
                "--epochs",
                str(args.epochs),
            ]
        )

    elif args.stage == "optuna":
        run(
            [
                python,
                str(SCRIPTS / "run_optuna.py"),
                "--config",
                str(config_path),
                "--trials",
                str(args.trials),
                "--trial-epochs",
                str(args.trial_epochs),
                "--train-final",
                "--final-epochs",
                str(args.final_epochs),
            ]
        )

    elif args.stage == "threshold":
        run(
            [
                python,
                str(SCRIPTS / "run_threshold_search.py"),
                "--config",
                str(config_path),
            ]
        )

    elif args.stage == "post":
        run(
            [
                python,
                str(SCRIPTS / "run_postprocess.py"),
                "--config",
                str(config_path),
            ]
        )

    elif args.stage == "final":
        # Internal technical freeze.
        # The user does not need to manage this as a separate UI step.
        run(
            semantic_freeze_command(
                python,
                config_path,
                config,
            )
        )

        output = experiment_output(config)

        run(
            [
                python,
                str(SCRIPTS / "run_final_test.py"),
                "--config",
                str(config_path),
                "--settings",
                str(output / "final_settings.json"),
            ]
        )


def run_instance(
    args,
    python: str,
    config_path: Path,
    config: dict,
):
    if args.stage in {
        "standard",
        "augmentation",
    }:
        run(
            [
                python,
                str(SCRIPTS / "run_maskrcnn_train.py"),
                "--config",
                str(config_path),
                "--epochs",
                str(args.epochs),
            ]
        )

    elif args.stage == "optuna":
        run(
            [
                python,
                str(SCRIPTS / "run_maskrcnn_optuna.py"),
                "--config",
                str(config_path),
                "--trials",
                str(args.trials),
                "--trial-epochs",
                str(args.trial_epochs),
                "--train-final",
                "--final-epochs",
                str(args.final_epochs),
            ]
        )

    elif args.stage == "threshold":
        run(
            [
                python,
                str(SCRIPTS / "run_maskrcnn_validation.py"),
                "--config",
                str(config_path),
                "--mode",
                "threshold",
            ]
        )

    elif args.stage == "post":
        run(
            [
                python,
                str(SCRIPTS / "run_maskrcnn_validation.py"),
                "--config",
                str(config_path),
                "--mode",
                "post",
            ]
        )

    elif args.stage == "final":
        # Internal technical freeze.
        run(
            maskrcnn_freeze_command(
                python,
                config_path,
                config,
            )
        )

        output = experiment_output(config)

        run(
            [
                python,
                str(SCRIPTS / "run_maskrcnn_final_test.py"),
                "--config",
                str(config_path),
                "--settings",
                str(output / "final_settings.json"),
            ]
        )


def main():
    args = parse_args()

    config_path = Path(args.config).resolve()

    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment config not found: {config_path}"
        )

    config = load_json(config_path)

    model_type = normalize_model_type(
        config.get("model_type", "")
    )

    spec = get_model_spec(model_type)

    python = sys.executable

    # ========================================================
    # MASTER OPTUNA — shared across all model families
    # ========================================================

    if args.stage == "master_optuna":
        run(
            [
                python,
                str(SCRIPTS / "run_master_optuna.py"),
                "--config",
                str(config_path),
                "--trials",
                str(args.trials),
                "--trial-epochs",
                str(args.trial_epochs),
            ]
        )
        return

    if spec.family == "semantic":
        run_semantic(
            args,
            python,
            config_path,
            config,
        )
    else:
        run_instance(
            args,
            python,
            config_path,
            config,
        )


if __name__ == "__main__":
    main()
