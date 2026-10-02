"""
ui/utils/config_io.py

Shared JSON/configuration helpers for the project-level GeoAI Master UI.

This module is deliberately backend-neutral:
- it does not import Buildings/src
- it does not import Roads/src
- it does not modify backend configuration files
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Iterable


# ============================================================
# JSON
# ============================================================

def load_json(
    path: str | Path,
    default=None,
):
    """
    Load JSON safely.

    If the file does not exist or cannot be parsed, return a copy of
    `default`. When no default is supplied, return an empty dictionary.
    """

    path = Path(path)

    fallback = (
        {}
        if default is None
        else deepcopy(default)
    )

    if not path.exists():
        return fallback

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return fallback


def save_json(
    data: dict,
    path: str | Path,
) -> Path:
    """
    Save JSON using UTF-8 and human-readable indentation.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return path


def first_json(
    folder: str | Path,
    names: Iterable[str],
) -> dict:
    """
    Return the first non-empty JSON payload found in `folder`.
    """

    folder = Path(folder)

    for name in names:
        payload = load_json(
            folder / name,
            {},
        )

        if payload:
            return payload

    return {}


# ============================================================
# TOKENS / RUNTIME NAMES
# ============================================================

def safe_token(
    value: Any,
) -> str:
    """
    Convert a label/experiment name into a filesystem-safe token.
    """

    value = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(value).strip().lower(),
    )

    return re.sub(
        r"_+",
        "_",
        value,
    ).strip("_")


def now_token() -> str:
    """
    Timestamp suitable for runtime config/log filenames.
    """

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


# ============================================================
# NESTED CONFIG HELPERS
# ============================================================

def nested_first(
    payload: dict,
    paths,
    default=None,
):
    """
    Return the first existing nested value.

    Example:
        nested_first(
            payload,
            [
                ("training", "batch_size"),
                ("batch_size",),
            ],
        )
    """

    for path in paths:
        current = payload

        try:
            for key in path:
                current = current[key]
        except (KeyError, TypeError):
            continue

        if current is not None:
            return current

    return default


def model_config_path(
    config_root: Path,
    model_key: str,
) -> Path | None:
    """
    Resolve a model JSON without assuming one single folder layout.
    """

    candidates = (
        config_root
        / "models"
        / f"{model_key}.json",

        config_root
        / f"{model_key}.json",
    )

    for path in candidates:
        if path.exists():
            return path

    return None


def read_backbones(
    config_root: Path,
    model_key: str,
) -> list[str]:
    """
    Read supported/configured backbones from a model JSON.

    Returns an empty list when no explicit backbone information exists.
    """

    path = model_config_path(
        config_root,
        model_key,
    )

    if path is None:
        return []

    payload = load_json(
        path,
        {},
    )

    for key in (
        "supported_backbones",
        "backbones",
    ):
        value = payload.get(
            key
        )

        if isinstance(value, dict) and value:
            return [
                str(item)
                for item in value.keys()
            ]

        if isinstance(value, list) and value:
            return [
                str(item)
                for item in value
            ]

    architecture = payload.get(
        "architecture",
        {},
    )

    if isinstance(
        architecture,
        dict,
    ):
        for key in (
            "supported_backbones",
            "backbones",
        ):
            value = architecture.get(
                key
            )

            if isinstance(
                value,
                dict,
            ) and value:
                return [
                    str(item)
                    for item in value.keys()
                ]

            if isinstance(
                value,
                list,
            ) and value:
                return [
                    str(item)
                    for item in value
                ]

    default = (
        payload.get(
            "default_backbone"
        )
        or payload.get(
            "backbone"
        )
        or (
            architecture.get(
                "backbone"
            )
            if isinstance(
                architecture,
                dict,
            )
            else None
        )
    )

    return (
        [str(default)]
        if default
        else []
    )


# ============================================================
# METRIC HELPERS
# ============================================================

def percentage(
    payload: dict,
    *keys: str,
):
    """
    Read a metric and return a percentage.

    Convention:
    - keys ending with `_percent` are assumed already in percent.
    - values in [0, 1] are converted to [0, 100].
    - larger values are returned unchanged.
    """

    for key in keys:
        value = payload.get(
            key
        )

        if value is None:
            continue

        try:
            value = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if key.endswith(
            "_percent"
        ):
            return value

        if 0.0 <= value <= 1.0:
            return value * 100.0

        return value

    return None


def as_bool(
    value,
    default: bool = False,
) -> bool:
    """
    Conservative boolean coercion for JSON/UI values.
    """

    if value is None:
        return bool(
            default
        )

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        (int, float),
    ):
        return bool(
            value
        )

    text = str(
        value
    ).strip().lower()

    if text in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if text in {
        "0",
        "false",
        "no",
        "off",
        "",
    }:
        return False

    return bool(
        default
    )
