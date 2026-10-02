"""
Roads/scripts/run_master_optuna.py

Road Master Optuna launcher with dynamic backbone search.

For the selected model family, this launcher replaces the backbone search
list with ALL READY backbones from config/backbone_catalog.json.

The final Aerial/Satellite/Drone test areas are never accessed here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROADS_ROOT = Path(__file__).resolve().parents[1]

if str(ROADS_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROADS_ROOT),
    )


from src.optional.backbone_search import (
    resolve_master_search_space,
)

from src.optional.master_optuna import (
    run_master_optuna_study,
)


def load_json(
    path: str | Path,
) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def save_json(
    payload: dict,
    path: str | Path,
) -> Path:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Road Master Optuna across all ready backbones "
            "for one selected model."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            ROADS_ROOT
            / "config"
            / "experiment.json"
        ),
    )

    parser.add_argument(
        "--search-space",
        type=str,
        default=str(
            ROADS_ROOT
            / "config"
            / "optional"
            / "master_optuna_search_space.json"
        ),
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--trial-epochs",
        type=int,
        default=12,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(
        args.config
    ).expanduser().resolve()

    search_space_path = Path(
        args.search_space
    ).expanduser().resolve()

    base_config = load_json(
        config_path
    )

    base_search_space = load_json(
        search_space_path
    )

    model_type = str(
        base_config.get(
            "model_type",
            "",
        )
    ).strip().lower()

    experiment_id = str(
        base_config.get(
            "experiment_id",
            "roads_master_optuna",
        )
    )

    (
        resolved_search_space,
        backbone_values,
    ) = resolve_master_search_space(
        base_search_space=base_search_space,
        model_type=model_type,
        roads_root=ROADS_ROOT,
    )

    resolved_path = (
        ROADS_ROOT
        / "outputs"
        / "experiments"
        / experiment_id
        / "master_optuna"
        / "resolved_master_optuna_search_space.json"
    )

    save_json(
        resolved_search_space,
        resolved_path,
    )

    print()
    print("=" * 78)
    print("ROAD MASTER OPTUNA — BACKBONES")
    print("=" * 78)
    print(f"Model: {model_type}")
    print("Backbones searched:")

    for backbone in backbone_values:
        print(
            f"  - {backbone}"
        )

    print("External Final Test Used: NO")
    print("=" * 78)

    result = run_master_optuna_study(
        base_config=base_config,
        search_space_path=resolved_path,
        n_trials=int(
            args.trials
        ),
        trial_epochs=int(
            args.trial_epochs
        ),
    )

    if isinstance(
        result,
        dict,
    ):
        print(
            json.dumps(
                result,
                indent=2,
                default=str,
            )
        )


if __name__ == "__main__":
    main()
