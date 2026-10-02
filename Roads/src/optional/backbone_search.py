"""
Roads/src/optional/backbone_search.py

Backbone catalog integration for Road Master Optuna.

Policy
------
- Manual training: selected UI backbone only.
- General Optuna: selected UI backbone stays fixed.
- Master Optuna: ALL READY backbones for the selected model are searched.
- SAM-LoRA ViT backbones become ready only when their official checkpoint exists.
- External final-test data are never used here.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


ROADS_ROOT = Path(__file__).resolve().parents[2]

CATALOG_PATH = (
    ROADS_ROOT
    / "config"
    / "backbone_catalog.json"
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


def normalize_model_type(
    model_type: str,
) -> str:
    value = str(
        model_type
    ).strip().lower()

    aliases = {
        "u-net": "unet",
        "u_net": "unet",
        "deeplab": "deeplabv3",
        "deeplabv3+": "deeplabv3",
        "deeplabv3plus": "deeplabv3",
        "connect_net": "connectnet",
        "multitask": "multitask_road_extractor",
        "multi_task_road_extractor": "multitask_road_extractor",
        "sam-lora": "sam_lora",
        "samlora": "sam_lora",
    }

    return aliases.get(
        value,
        value,
    )


def load_backbone_catalog(
    path: str | Path = CATALOG_PATH,
) -> dict:
    return load_json(
        path
    )


def model_backbone_entries(
    model_type: str,
    catalog: dict | None = None,
) -> list[dict]:
    catalog = (
        catalog
        if catalog is not None
        else load_backbone_catalog()
    )

    model_key = normalize_model_type(
        model_type
    )

    models = catalog.get(
        "models",
        {},
    )

    if model_key not in models:
        raise KeyError(
            f"Model '{model_key}' is missing from backbone_catalog.json"
        )

    entries = models[
        model_key
    ].get(
        "backbones",
        [],
    )

    if not isinstance(
        entries,
        list,
    ):
        raise TypeError(
            f"Backbone catalog for '{model_key}' must contain a list."
        )

    return [
        dict(entry)
        for entry
        in entries
    ]


def _checkpoint_ready(
    entry: dict,
    roads_root: Path,
) -> bool:
    checkpoint = entry.get(
        "checkpoint"
    )

    if not checkpoint:
        return False

    path = Path(
        str(
            checkpoint
        )
    )

    if not path.is_absolute():
        path = (
            roads_root.parent
            / path
        )

    return path.exists()


def entry_ready(
    entry: dict,
    roads_root: Path = ROADS_ROOT,
) -> bool:
    ready = entry.get(
        "ready",
        True,
    )

    if ready == "checkpoint":
        return _checkpoint_ready(
            entry,
            roads_root,
        )

    return bool(
        ready
    )


def ready_master_backbones(
    model_type: str,
    roads_root: Path = ROADS_ROOT,
    catalog: dict | None = None,
) -> list:
    values = []

    for entry in model_backbone_entries(
        model_type,
        catalog=catalog,
    ):
        if not entry.get(
            "master_optuna",
            False,
        ):
            continue

        if not entry_ready(
            entry,
            roads_root=roads_root,
        ):
            continue

        values.append(
            entry.get(
                "id"
            )
        )

    return values


def _patch_model_space(
    node: Any,
    model_type: str,
    backbone_values: list,
) -> bool:
    if not isinstance(
        node,
        dict,
    ):
        return False

    # Already model-specific.
    if "parameters" in node:
        parameters = node.get(
            "parameters"
        )

        if isinstance(
            parameters,
            dict,
        ):
            declared_model = normalize_model_type(
                node.get(
                    "model_type",
                    model_type,
                )
            )

            if declared_model == model_type:
                backbone_spec = parameters.get(
                    "backbone",
                    {},
                )

                if not isinstance(
                    backbone_spec,
                    dict,
                ):
                    backbone_spec = {}

                backbone_spec.update(
                    {
                        "type": "categorical",
                        "values": deepcopy(
                            backbone_values
                        ),
                    }
                )

                parameters[
                    "backbone"
                ] = backbone_spec

                return True

    # Named model block.
    for key, value in node.items():
        if (
            normalize_model_type(
                key
            )
            == model_type
            and isinstance(
                value,
                dict,
            )
        ):
            parameters = value.setdefault(
                "parameters",
                {},
            )

            backbone_spec = parameters.get(
                "backbone",
                {},
            )

            if not isinstance(
                backbone_spec,
                dict,
            ):
                backbone_spec = {}

            backbone_spec.update(
                {
                    "type": "categorical",
                    "values": deepcopy(
                        backbone_values
                    ),
                }
            )

            parameters[
                "backbone"
            ] = backbone_spec

            return True

    # Recurse.
    for value in node.values():
        if isinstance(
            value,
            dict,
        ):
            if _patch_model_space(
                value,
                model_type,
                backbone_values,
            ):
                return True

    return False


def resolve_master_search_space(
    base_search_space: dict,
    model_type: str,
    roads_root: Path = ROADS_ROOT,
) -> tuple[dict, list]:
    """
    Inject every READY Master-Optuna backbone into the selected model space.
    """

    model_type = normalize_model_type(
        model_type
    )

    backbones = ready_master_backbones(
        model_type,
        roads_root=roads_root,
    )

    if not backbones:
        raise ValueError(
            "No ready Master Optuna backbone candidates exist for "
            f"model '{model_type}'."
        )

    resolved = deepcopy(
        base_search_space
    )

    if not _patch_model_space(
        resolved,
        model_type,
        backbones,
    ):
        raise ValueError(
            "Could not find the selected model's parameter block inside "
            "master_optuna_search_space.json."
        )

    return (
        resolved,
        backbones,
    )
