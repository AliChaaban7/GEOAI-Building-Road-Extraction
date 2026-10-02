"""
ui/pages/roads.py

Project-level Road Extraction page.

Current Road workflow
---------------------
Mixed-source Train/Validation:
    Aerial + Satellite + Drone

Model set:
- U-Net
- DeepLabV3+
- ConnectNet
- MultiTask Road Extractor
- SAM-LoRA

Final test:
- Aerial
- Satellite
- Drone
- one frozen configuration
- no per-source tuning

Important
---------
ConnectNet and MultiTask Road Extractor remain ArcGIS Learn models.
Their training launcher is exposed, but validation/final inference controls
remain disabled until their exact ArcGIS inference integration is verified.
"""

from __future__ import annotations

from pathlib import Path
import sys
import textwrap

THIS_FILE = Path(__file__).resolve()
PAGES_ROOT = THIS_FILE.parent
UI_ROOT = PAGES_ROOT.parent
PROJECT_ROOT = UI_ROOT.parent

for value in (
    str(PROJECT_ROOT),
    str(UI_ROOT),
):
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from copy import deepcopy
from pathlib import Path
import re
import sys

import streamlit as st

from components.experiment_form import (
    render_backend_banner,
    render_dataset_banner,
)
from components.header import (
    render_header,
    render_section,
)
from components.map_view import (
    render_final_overlay,
)
from components.results_table import (
    render_results_table,
)
from utils.config_io import (
    load_json,
    now_token,
    safe_token,
    save_json,
)
from utils.paths import (
    PROJECT_ROOT,
    ROADS_ARCGIS_TRAIN_SCRIPT,
    ROADS_CONFIG_ROOT,
    ROADS_EXPERIMENTS_ROOT,
    ROADS_FINAL_TEST_SCRIPT,
    ROADS_FREEZE_SCRIPT,
    ROADS_MASTER_OPTUNA_SCRIPT,
    ROADS_OPTUNA_SCRIPT,
    ROADS_PATHS_CONFIG,
    ROADS_POSTPROCESS_SCRIPT,
    ROADS_THRESHOLD_SCRIPT,
    ROADS_TRAIN_SCRIPT,
    ROAD_UI_CONFIG_ROOT,
    ROAD_UI_LOG_ROOT,
)
from utils.process_runner import (
    CommandSpec,
    run_streamed,
)
from utils.result_reader import (
    experiment_config,
    experiment_folders,
    final_metric_payloads,
    result_rows,
)


# ============================================================
# INTERNAL REPRODUCIBILITY
# ============================================================

# Kept fixed in backend configs for deterministic Train/Validation splitting.
# This value is intentionally not exposed in the UI.
FIXED_SPLIT_SEED = 42


# ============================================================
# HTML HELPER
# ============================================================

def _html(
    content: str,
) -> None:
    cleaned = "\n".join(
        line.strip()
        for line in textwrap.dedent(content).splitlines()
        if line.strip()
    )

    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(
            cleaned,
            unsafe_allow_html=True,
        )


# ============================================================
# ROAD MODEL REGISTRY — UI METADATA ONLY
# ============================================================

MODEL_INFO = {
    "unet": {
        "label": "U-Net",
        "backend": "PyTorch",
        "supports_optuna": True,
        "validation_ready": True,
        "final_ready": True,
    },
    "deeplabv3": {
        "label": "DeepLabV3+",
        "backend": "PyTorch",
        "supports_optuna": True,
        "validation_ready": True,
        "final_ready": True,
    },
    "connectnet": {
        "label": "ConnectNet",
        "backend": "ArcGIS Learn",
        "supports_optuna": False,
        "validation_ready": False,
        "final_ready": False,
    },
    "multitask_road_extractor": {
        "label": "MultiTask Road Extractor",
        "backend": "ArcGIS Learn",
        "supports_optuna": False,
        "validation_ready": False,
        "final_ready": False,
    },
    "sam_lora": {
        "label": "SAM-LoRA",
        "backend": "PyTorch",
        "supports_optuna": True,
        "validation_ready": True,
        "final_ready": True,
    },
}



# ============================================================
# ROAD BACKBONE CATALOG
# ============================================================

BACKBONE_CATALOG_PATH = (
    ROADS_CONFIG_ROOT
    / "backbone_catalog.json"
)


def load_backbone_catalog() -> dict:
    return load_json(
        BACKBONE_CATALOG_PATH,
        {},
    )


def model_backbone_entries(
    model_key: str,
) -> list[dict]:
    """
    Resolve the Road backbone catalog for one model.

    Priority
    --------
    1. Roads/config/backbone_catalog.json
       This is the authoritative UI backbone catalog.

    2. Roads/config/models/<model>.json
       Used only as a conservative compatibility fallback.

    Safety
    ------
    A legacy model JSON must not automatically turn every historical
    backbone into a runnable or Master-Optuna candidate.

    Therefore:
    - backbone_catalog.json remains authoritative when entries exist;
    - readiness and Master-Optuna eligibility must be explicit there;
    - the legacy fallback exposes only the configured default backbone;
    - the legacy fallback is never automatically added to Master Optuna.
    """

    catalog = load_backbone_catalog()

    model_catalog = (
        catalog
        .get(
            "models",
            {},
        )
        .get(
            model_key,
            {},
        )
    )

    entries = model_catalog.get(
        "backbones",
        [],
    )

    # --------------------------------------------------------
    # PRIMARY SOURCE:
    # Roads/config/backbone_catalog.json
    # --------------------------------------------------------

    if (
        isinstance(
            entries,
            list,
        )
        and entries
    ):

        normalized_entries = []

        for raw_entry in entries:

            if not isinstance(
                raw_entry,
                dict,
            ):
                continue

            entry = dict(
                raw_entry
            )

            backbone_id = entry.get(
                "id"
            )

            # Ignore malformed entries instead of inventing an
            # architecture identifier.
            if backbone_id is None:
                continue

            if isinstance(
                backbone_id,
                str,
            ):
                entry[
                    "id"
                ] = (
                    backbone_id
                    .strip()
                    .lower()
                )

            # Keep readiness explicit. Existing supported values
            # such as True, False and "checkpoint" remain valid.
            entry.setdefault(
                "ready",
                False,
            )

            # Master Optuna membership must also be explicit.
            entry.setdefault(
                "master_optuna",
                False,
            )

            entry.setdefault(
                "label",
                str(
                    entry.get(
                        "id"
                    )
                ),
            )

            normalized_entries.append(
                entry
            )

        if normalized_entries:
            return normalized_entries

    # ========================================================
    # SAFE LEGACY FALLBACK
    # ========================================================
    #
    # Older UI behavior converted every value in
    # supported_backbones/backbones to:
    #
    #     ready = True
    #     master_optuna = True
    #
    # That can expose stale architecture names that are no longer
    # implemented by the active Roads Python model code.
    #
    # The fallback therefore exposes ONLY the configured default.
    # ========================================================

    config = load_json(
        ROADS_CONFIG_ROOT
        / "models"
        / f"{model_key}.json",
        {},
    )

    default = (
        config.get(
            "default_backbone"
        )
        or config.get(
            "backbone"
        )
    )

    if default is None:
        return []

    if isinstance(
        default,
        str,
    ):
        default = (
            default
            .strip()
            .lower()
        )

    return [
        {
            "id":
                default,

            "label":
                str(
                    default
                ),

            # Keep the declared default available for normal manual
            # training when the central catalog is missing.
            "ready":
                True,

            # Never silently place a legacy/fallback architecture
            # inside Master Optuna.
            "master_optuna":
                False,

            "source":
                "model_json_default_fallback",
        }
    ]


def backbone_checkpoint_path(
    entry: dict,
) -> Path | None:
    checkpoint = entry.get(
        "checkpoint"
    )

    if not checkpoint:
        return None

    path = Path(
        str(
            checkpoint
        )
    )

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path


def backbone_entry_ready(
    entry: dict,
) -> bool:
    ready = entry.get(
        "ready",
        True,
    )

    if ready == "checkpoint":
        path = backbone_checkpoint_path(
            entry
        )

        return bool(
            path
            and path.exists()
        )

    return bool(
        ready
    )


def backbone_label(
    entry: dict,
) -> str:
    """
    Clean UI label only.

    Readiness is handled separately by the warning below the selector.
    The dropdown therefore stays visually clean:
        ResNet50
        ViT-B
        ViT-L
        ViT-H
    """
    label = str(
        entry.get(
            "label",
            entry.get(
                "id",
                "Default",
            ),
        )
    )

    # Keep the visible selector concise.
    label = label.replace(
        " Encoder",
        "",
    )

    return label


def ready_master_backbones_for_ui(
    model_key: str,
) -> list:
    return [
        entry.get(
            "id"
        )
        for entry
        in model_backbone_entries(
            model_key
        )
        if entry.get(
            "master_optuna",
            False,
        )
        and backbone_entry_ready(
            entry
        )
    ]


# ============================================================
# ROAD DATASET DISCOVERY
# ============================================================

TILE_RE = re.compile(
    r"(?:^|_)(128|192|256|384|512)(?:_|$)",
    re.IGNORECASE,
)


def discover_road_datasets() -> list[dict]:
    """
    Discover enabled mixed Road Classified-Tile datasets.

    The reader prefers explicit JSON metadata. If tile_size is omitted,
    it can read the supported tile size from the dataset id.
    It does not invent filesystem paths.
    """

    payload = load_json(
        ROADS_CONFIG_ROOT
        / "datasets.json",
        {},
    )

    rows = []

    def walk(
        node,
        node_key=None,
    ):
        if not isinstance(
            node,
            dict,
        ):
            return

        dataset_id = (
            node.get(
                "dataset_id"
            )
            or node_key
        )

        tile_size = (
            node.get(
                "tile_size"
            )
            or node.get(
                "chip_size"
            )
        )

        if (
            tile_size is None
            and dataset_id
        ):
            match = TILE_RE.search(
                str(
                    dataset_id
                )
            )

            if match:
                tile_size = int(
                    match.group(
                        1
                    )
                )

        path = (
            node.get(
                "path"
            )
            or node.get(
                "dataset_path"
            )
            or node.get(
                "folder"
            )
            or node.get(
                "root"
            )
        )

        source = str(
            node.get(
                "source_type",
                node.get(
                    "source",
                    "",
                ),
            )
        ).lower()

        sources = node.get(
            "sources"
        )

        mixed = (
            source
            == "mixed"
            or (
                isinstance(
                    sources,
                    list,
                )
                and {
                    "aerial",
                    "satellite",
                    "drone",
                }.issubset(
                    {
                        str(
                            value
                        ).lower()
                        for value
                        in sources
                    }
                )
            )
            or (
                dataset_id
                and "mixed"
                in str(
                    dataset_id
                ).lower()
            )
        )

        if (
            dataset_id
            and tile_size
            is not None
            and path
        ):
            rows.append(
                {
                    "dataset_id":
                        str(
                            dataset_id
                        ),

                    "tile_size":
                        int(
                            tile_size
                        ),

                    "path":
                        path,

                    "mixed":
                        bool(
                            mixed
                        ),

                    "enabled":
                        bool(
                            node.get(
                                "enabled",
                                True,
                            )
                        ),
                }
            )

        for key, value in node.items():
            if isinstance(
                value,
                dict,
            ):
                walk(
                    value,
                    key,
                )

    walk(
        payload
    )

    filtered = {}

    for row in rows:

        if not row[
            "enabled"
        ]:
            continue

        if not row[
            "mixed"
        ]:
            continue

        filtered[
            (
                row[
                    "dataset_id"
                ],
                row[
                    "tile_size"
                ],
            )
        ] = row

    return list(
        filtered.values()
    )


# ============================================================
# COMMANDS
# ============================================================

def _require_script(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Road backend script not found:\n{path}"
        )


def run_road_stage(
    title: str,
    command: list[str],
    log_stem: str,
) -> int:

    return run_streamed(
        CommandSpec(
            title=title,
            command=command,
            log_path=(
                ROAD_UI_LOG_ROOT
                / (
                    f"{safe_token(log_stem)}_"
                    f"{now_token()}.log"
                )
            ),
            cwd=PROJECT_ROOT,
        )
    )


def baseline_train_command(
    model_key: str,
    config_path: Path,
    epochs: int,
) -> list[str]:

    backend = MODEL_INFO[
        model_key
    ][
        "backend"
    ]

    if backend == "ArcGIS Learn":
        _require_script(
            ROADS_ARCGIS_TRAIN_SCRIPT
        )

        return [
            sys.executable,
            str(
                ROADS_ARCGIS_TRAIN_SCRIPT
            ),
            "--config",
            str(
                config_path
            ),
        ]

    _require_script(
        ROADS_TRAIN_SCRIPT
    )

    return [
        sys.executable,
        str(
            ROADS_TRAIN_SCRIPT
        ),
        "--config",
        str(
            config_path
        ),
        "--epochs",
        str(
            int(
                epochs
            )
        ),
    ]


def optuna_command(
    config_path: Path,
    trials: int,
    trial_epochs: int,
    final_epochs: int,
) -> list[str]:

    _require_script(
        ROADS_OPTUNA_SCRIPT
    )

    return [
        sys.executable,
        str(
            ROADS_OPTUNA_SCRIPT
        ),
        "--config",
        str(
            config_path
        ),
        "--trials",
        str(
            int(
                trials
            )
        ),
        "--trial-epochs",
        str(
            int(
                trial_epochs
            )
        ),
        "--train-final",
        "--final-epochs",
        str(
            int(
                final_epochs
            )
        ),
    ]


def master_optuna_command(
    config_path: Path,
    trials: int,
    trial_epochs: int,
) -> list[str]:

    _require_script(
        ROADS_MASTER_OPTUNA_SCRIPT
    )

    return [
        sys.executable,
        str(
            ROADS_MASTER_OPTUNA_SCRIPT
        ),
        "--config",
        str(
            config_path
        ),
        "--trials",
        str(
            int(
                trials
            )
        ),
        "--trial-epochs",
        str(
            int(
                trial_epochs
            )
        ),
    ]


def threshold_command(
    config_path: Path,
) -> list[str]:

    _require_script(
        ROADS_THRESHOLD_SCRIPT
    )

    return [
        sys.executable,
        str(
            ROADS_THRESHOLD_SCRIPT
        ),
        "--config",
        str(
            config_path
        ),
    ]


def postprocess_command(
    config_path: Path,
) -> list[str]:

    _require_script(
        ROADS_POSTPROCESS_SCRIPT
    )

    return [
        sys.executable,
        str(
            ROADS_POSTPROCESS_SCRIPT
        ),
        "--config",
        str(
            config_path
        ),
    ]


def freeze_command(
    config_path: Path,
) -> list[str]:

    _require_script(
        ROADS_FREEZE_SCRIPT
    )

    return [
        sys.executable,
        str(
            ROADS_FREEZE_SCRIPT
        ),
        "--config",
        str(
            config_path
        ),
        "--overwrite",
    ]


def final_test_command(
    config_path: Path,
    source: str,
) -> list[str]:

    _require_script(
        ROADS_FINAL_TEST_SCRIPT
    )

    return [
        sys.executable,
        str(
            ROADS_FINAL_TEST_SCRIPT
        ),
        "--config",
        str(
            config_path
        ),
        "--source",
        str(
            source
        ),
    ]


# ============================================================
# 1. SETUP & TRAIN
# ============================================================

def render_setup_and_train() -> None:

    render_section(
        "1",
        "Setup & Train",
        (
            "Train one Road model on the mixed Aerial + Satellite + "
            "Drone Road-surface dataset."
        ),
    )

    datasets = discover_road_datasets()

    if not datasets:
        st.error(
            "No enabled mixed Road dataset could be resolved from "
            "Roads/config/datasets.json."
        )
        return

    label_to_key = {
        info[
            "label"
        ]:
            key
        for key, info
        in MODEL_INFO.items()
    }

    primary_controls = st.container(
        key="roads_primary_controls"
    )

    c1, c2, c3, c4 = primary_controls.columns(
        4
    )

    with c1:
        model_label = st.selectbox(
            "Model",
            list(
                label_to_key.keys()
            ),
            key="roads_model",
        )

    model_key = label_to_key[
        model_label
    ]

    model_info = MODEL_INFO[
        model_key
    ]

    with c2:
        st.text_input(
            "Training Source",
            value=(
                "Mixed: Aerial + Satellite + Drone"
            ),
            disabled=True,
            key="roads_source",
        )

    tile_options = sorted(
        {
            int(
                row[
                    "tile_size"
                ]
            )
            for row in datasets
        }
    )

    with c3:
        tile_size = st.selectbox(
            "Tile Size",
            tile_options,
            index=(
                tile_options.index(
                    256
                )
                if 256
                in tile_options
                else 0
            ),
            key="roads_tile",
        )


    # --------------------------------------------------------
    # BACKBONE
    # --------------------------------------------------------

    backbone_entries = model_backbone_entries(
        model_key
    )

    # --------------------------------------------------------
    # ArcGIS Learn models may legitimately use backbone=None.
    #
    # ConnectNet / MultiTaskRoadExtractor are trained through the
    # separate ArcGIS Learn launcher. An empty backbone catalog must
    # therefore NOT stop the Setup & Train section before the Train
    # button is rendered.
    # --------------------------------------------------------

    if (
        not backbone_entries
        and model_info[
            "backend"
        ]
        == "ArcGIS Learn"
    ):

        backbone = None
        selected_backbone_entry = None
        backbone_ready = True

        with c4:
            st.text_input(
                "Backbone",
                value="ArcGIS Learn default",
                disabled=True,
                key=(
                    f"roads_backbone_"
                    f"{model_key}_arcgis"
                ),
            )

    elif not backbone_entries:

        st.error(
            "No backbone options are configured for this Road model."
        )
        return

    else:

        backbone_ids = [
            entry.get(
                "id"
            )
            for entry
            in backbone_entries
        ]

        catalog_payload = load_backbone_catalog()

        default_backbone = (
            catalog_payload
            .get(
                "models",
                {},
            )
            .get(
                model_key,
                {},
            )
            .get(
                "default_backbone"
            )
        )

        default_index = (
            backbone_ids.index(
                default_backbone
            )
            if default_backbone
            in backbone_ids
            else 0
        )

        with c4:
            backbone = st.selectbox(
                "Backbone",
                backbone_ids,
                index=default_index,
                format_func=lambda value: backbone_label(
                    next(
                        entry
                        for entry
                        in backbone_entries
                        if entry.get(
                            "id"
                        )
                        == value
                    )
                ),
                key=(
                    f"roads_backbone_"
                    f"{model_key}"
                ),
            )

        selected_backbone_entry = next(
            entry
            for entry
            in backbone_entries
            if entry.get(
                "id"
            )
            == backbone
        )

        backbone_ready = backbone_entry_ready(
            selected_backbone_entry
        )

        if not backbone_ready:
            checkpoint_path = backbone_checkpoint_path(
                selected_backbone_entry
            )

            st.warning(
                "This backbone is supported but is not ready on this machine "
                "because its required pretrained checkpoint is missing."
                + (
                    f" Expected: {checkpoint_path}"
                    if checkpoint_path
                    else ""
                )
            )


    # --------------------------------------------------------
    # EFFECTIVE ARCGIS ARCHITECTURE / BACKBONE
    # --------------------------------------------------------
    #
    # For native PyTorch models, the selected UI backbone is used
    # unchanged.
    #
    # For ArcGIS Learn road models, backbone_catalog.json may map a
    # visible UI choice to an effective ArcGIS configuration:
    #
    #   ArcGIS Default (Hourglass)
    #       -> backbone = None
    #       -> mtl_model = hourglass
    #
    #   ResNet-*
    #       -> backbone = resnet*
    #       -> mtl_model = linknet
    #
    # The backend reads these saved experiment values and merges them
    # into the static model config without modifying the source JSON.
    # --------------------------------------------------------

    effective_backbone = backbone
    arcgis_mtl_model = None

    if (
        model_info[
            "backend"
        ]
        == "ArcGIS Learn"
        and selected_backbone_entry
        is not None
    ):

        effective_backbone = (
            selected_backbone_entry.get(
                "arcgis_backbone",
                backbone,
            )
        )

        arcgis_mtl_model = (
            selected_backbone_entry.get(
                "mtl_model"
            )
        )


    dataset = next(
        (
            row
            for row in datasets
            if int(
                row[
                    "tile_size"
                ]
            )
            == int(
                tile_size
            )
        ),
        None,
    )

    if dataset:
        render_dataset_banner(
            dataset[
                "dataset_id"
            ],
            dataset.get(
                "path"
            ),
        )

    render_backend_banner(
        model_info[
            "backend"
        ]
    )

    if model_key == "sam_lora":
        selected_sam_backbone = backbone_label(
            selected_backbone_entry
        )

        _html(
            f"""
            <div class="architecture-banner">
                <b>SAM-LoRA Road architecture</b><br>
                {selected_sam_backbone} · LoRA rank 8 · alpha 16 ·
                Q + V adapters · frozen base SAM · frozen prompt encoder ·
                trainable mask decoder
            </div>
            """
        )

    elif model_key == "connectnet":
        architecture_text = (
            str(arcgis_mtl_model).title()
            if arcgis_mtl_model
            else "ArcGIS default"
        )

        backbone_text = (
            str(effective_backbone)
            if effective_backbone
            is not None
            else "None / Hourglass internal encoder"
        )

        st.caption(
            "ArcGIS Learn ConnectNet · "
            f"Architecture: {architecture_text} · "
            f"Backbone: {backbone_text}."
        )

    elif model_key == "multitask_road_extractor":
        architecture_text = (
            str(arcgis_mtl_model).title()
            if arcgis_mtl_model
            else "ArcGIS default"
        )

        backbone_text = (
            str(effective_backbone)
            if effective_backbone
            is not None
            else "None / Hourglass internal encoder"
        )

        st.caption(
            "ArcGIS Learn MultiTaskRoadExtractor · "
            f"Architecture: {architecture_text} · "
            f"Backbone: {backbone_text}."
        )

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    split_controls = st.container(
        key="roads_split_controls"
    )

    s1, s2 = split_controls.columns(
        2
    )

    with s1:
        train_percent = st.number_input(
            "Training %",
            min_value=60,
            max_value=90,
            value=80,
            step=5,
            key="roads_train_percent",
        )

    validation_percent = (
        100
        - int(
            train_percent
        )
    )

    with s2:
        st.number_input(
            "Validation %",
            min_value=10,
            max_value=40,
            value=validation_percent,
            disabled=True,
            key="roads_validation_percent",
        )

    st.caption(
        f"Split: {int(train_percent)}% Training / "
        f"{int(validation_percent)}% Validation · "
        "External Test: Separate"
    )

    auto_experiment_name = (
        f"{model_key}_"
        f"{int(tile_size)}_"
        "mixed_roads"
    )

    previous_auto_name = st.session_state.get(
        "_roads_auto_experiment_name"
    )

    current_name = st.session_state.get(
        "roads_experiment_name"
    )

    # Keep the automatic experiment ID synchronized with the selected
    # model/tile while preserving a genuinely custom name typed by the user.
    known_auto_pattern = re.compile(
        r"^(?:unet|deeplabv3|connectnet|multitask_road_extractor|"
        r"sam_lora)_\\d+_mixed_roads$",
        re.IGNORECASE,
    )

    should_refresh_auto_name = (
        current_name is None
        or current_name == previous_auto_name
        or (
            isinstance(
                current_name,
                str,
            )
            and known_auto_pattern.fullmatch(
                current_name.strip()
            )
            is not None
        )
    )

    if should_refresh_auto_name:
        st.session_state[
            "roads_experiment_name"
        ] = auto_experiment_name

    st.session_state[
        "_roads_auto_experiment_name"
    ] = auto_experiment_name

    experiment_name = st.text_input(
        "Experiment Name",
        key="roads_experiment_name",
    )

    # --------------------------------------------------------
    # OPTIONAL
    # --------------------------------------------------------

    optimization_controls = st.container(
        key="roads_optimization_controls"
    )

    o1, o2, o3 = optimization_controls.columns(
        3
    )

    with o1:
        use_augmentation = st.checkbox(
            "Use Augmentation",
            value=False,
            key="roads_augmentation",
        )

    with o2:
        use_optuna = st.checkbox(
            "General Optuna",
            value=False,
            disabled=(
                not model_info[
                    "supports_optuna"
                ]
            ),
            key="roads_optuna",
        )

    with o3:
        use_master_optuna = st.checkbox(
            "Master Optuna",
            value=False,
            disabled=(
                not model_info[
                    "supports_optuna"
                ]
            ),
            key="roads_master_optuna",
        )

    optimization_conflict = bool(
        use_optuna
        and use_master_optuna
    )

    if optimization_conflict:
        st.error(
            "Choose General Optuna or Master Optuna, not both."
        )


    master_backbone_pool = (
        ready_master_backbones_for_ui(
            model_key
        )
    )

    if use_master_optuna:
        if master_backbone_pool:
            st.info(
                "Master Optuna will search ALL ready backbones for this model: "
                + ", ".join(
                    "Default"
                    if value is None
                    else str(value)
                    for value
                    in master_backbone_pool
                )
                + ". The Backbone dropdown above does not restrict Master Optuna."
            )
        else:
            st.warning(
                "Master Optuna has no ready backbone candidates for this model."
            )

    if not model_info[
        "supports_optuna"
    ]:
        st.info(
            "Optuna is not exposed for this ArcGIS Learn model. "
            "No fabricated PyTorch replacement is used."
        )

    # --------------------------------------------------------
    # TRAINING SCHEDULE
    # --------------------------------------------------------

    render_section(
        "1A",
        "Training Schedule",
        (
            "Control the epoch budget for standard training and "
            "the trial budgets used by Optuna."
        ),
    )

    schedule = st.container(
        key="roads_training_schedule"
    )

    if use_master_optuna:

        m1, m2 = schedule.columns(
            2
        )

        with m1:
            trials = st.number_input(
                "Master Trials",
                min_value=1,
                value=15,
                step=1,
                key="roads_master_trials",
            )

        with m2:
            trial_epochs = st.number_input(
                "Epochs / Master Trial",
                min_value=1,
                value=12,
                step=1,
                key="roads_master_trial_epochs",
            )

        # Master Optuna keeps the exact winning trial checkpoint.
        epochs = 40
        final_epochs = 40

        st.caption(
            "Master Optuna compares complete candidates and keeps the exact "
            "best Validation-IoU checkpoint; no second winner training is required."
        )

    elif use_optuna:

        p1, p2, p3 = schedule.columns(
            3
        )

        with p1:
            trials = st.number_input(
                "Trials",
                min_value=1,
                value=10,
                step=1,
                key="roads_optuna_trials",
            )

        with p2:
            trial_epochs = st.number_input(
                "Epochs / Trial",
                min_value=1,
                value=12,
                step=1,
                key="roads_optuna_trial_epochs",
            )

        with p3:
            final_epochs = st.number_input(
                "Final Epochs",
                min_value=1,
                value=40,
                step=1,
                key="roads_optuna_final_epochs",
            )

        epochs = 40

    else:

        epochs = schedule.number_input(
            "Epochs",
            min_value=1,
            value=40,
            step=1,
            key="roads_epochs",
        )

        trials = 10
        trial_epochs = 12
        final_epochs = 40

    # --------------------------------------------------------
    # RUNTIME CONFIG
    # --------------------------------------------------------

    base_config = load_json(
        ROADS_CONFIG_ROOT
        / "experiment.json",
        {},
    )

    runtime = deepcopy(
        base_config
    )

    runtime.update(
        {
            "experiment_id":
                safe_token(
                    experiment_name
                ),

            "approach":
                "roads",

            "model_type":
                model_key,

            "backbone":
                effective_backbone,

            "source_type":
                "mixed",

            "tile_size":
                int(
                    tile_size
                ),

            "dataset_id":
                (
                    dataset[
                        "dataset_id"
                    ]
                    if dataset
                    else None
                ),

            "use_augmentation":
                bool(
                    use_augmentation
                ),

            "use_optuna":
                bool(
                    use_optuna
                ),

            "use_master_optuna":
                bool(
                    use_master_optuna
                ),

            "use_threshold_search":
                False,

            "use_postprocessing":
                False,

            "use_cross_validation":
                False,

            "use_reporting":
                False,

            "use_error_analysis":
                False,

            "training_schedule": {
                "epochs":
                    int(
                        epochs
                    ),

                "trials":
                    int(
                        trials
                    ),

                "trial_epochs":
                    int(
                        trial_epochs
                    ),

                "final_epochs":
                    int(
                        final_epochs
                    ),
            },

            "data_split": {
                "train_percent":
                    float(
                        train_percent
                    ),

                "validation_percent":
                    float(
                        validation_percent
                    ),

                "seed":
                    int(
                        FIXED_SPLIT_SEED
                    ),
            },
        }
    )


    # Preserve the visible ArcGIS choice as provenance while keeping
    # `backbone` equal to the effective value passed to ArcGIS Learn.
    if model_info[
        "backend"
    ] == "ArcGIS Learn":

        runtime[
            "backbone_choice"
        ] = backbone

        runtime[
            "arcgis_mtl_model"
        ] = arcgis_mtl_model


    button_text = (
        "Run Master Optuna"
        if use_master_optuna
        else "Run General Optuna"
        if use_optuna
        else "Train Road Model"
    )

    action_container_key = (
        "roads_action_master"
        if use_master_optuna
        else "roads_action_optuna"
        if use_optuna
        else "roads_action_train"
    )

    action_container = st.container(
        key=action_container_key
    )

    action_clicked = action_container.button(
        button_text,
        type="primary",
        use_container_width=True,
        disabled=(
            optimization_conflict
            or (
                not use_master_optuna
                and not backbone_ready
            )
            or (
                use_master_optuna
                and not master_backbone_pool
            )
        ),
        key="roads_train_button",
    )

    if action_clicked:

        config_path = save_json(
            runtime,
            ROAD_UI_CONFIG_ROOT
            / (
                f"{safe_token(experiment_name)}_"
                f"train_{now_token()}.json"
            ),
        )

        if use_master_optuna:
            command = master_optuna_command(
                config_path,
                trials=int(
                    trials
                ),
                trial_epochs=int(
                    trial_epochs
                ),
            )

        elif use_optuna:
            command = optuna_command(
                config_path,
                trials=int(
                    trials
                ),
                trial_epochs=int(
                    trial_epochs
                ),
                final_epochs=int(
                    final_epochs
                ),
            )

        else:
            command = baseline_train_command(
                model_key,
                config_path,
                epochs=int(
                    epochs
                ),
            )

        run_road_stage(
            button_text,
            command,
            f"{experiment_name}_train",
        )


# ============================================================
# 2. VALIDATION OPTIONS
# ============================================================

def render_validation_options() -> None:

    render_section(
        "2",
        "Validation Options",
        (
            "Select threshold and Road morphology using validation only. "
            "The independent source-specific tests remain untouched."
        ),
    )

    folders = experiment_folders(
        ROADS_EXPERIMENTS_ROOT
    )

    if not folders:
        st.info(
            "No trained Road experiment is available yet."
        )
        return

    selected = st.selectbox(
        "Experiment",
        [
            folder.name
            for folder
            in folders
        ],
        key="roads_validation_experiment",
    )

    folder = (
        ROADS_EXPERIMENTS_ROOT
        / selected
    )

    config = experiment_config(
        folder
    )

    if not config:
        st.error(
            "Could not load this Road experiment configuration."
        )
        return

    model_key = str(
        config.get(
            "model_type",
            "",
        )
    ).lower()

    model_info = MODEL_INFO.get(
        model_key
    )

    if (
        not model_info
        or not model_info[
            "validation_ready"
        ]
    ):
        st.info(
            "Validation threshold/post-processing for this ArcGIS Learn "
            "model remains disabled until its exact inference integration "
            "is verified."
        )
        return

    v1, v2 = st.columns(
        2
    )

    with v1:
        optimize_threshold = st.checkbox(
            "Optimize Threshold",
            value=False,
            key="roads_validation_threshold",
        )

    with v2:
        use_postprocessing = st.checkbox(
            "Road Post-processing",
            value=False,
            key="roads_validation_postprocess",
        )

    if st.button(
        "Apply Validation Options",
        type="primary",
        use_container_width=True,
        key="roads_validation_button",
    ):

        if (
            not optimize_threshold
            and not use_postprocessing
        ):
            st.info(
                "Select at least one validation option."
            )
            return

        runtime = deepcopy(
            config
        )

        threshold_exists = (
            folder
            / "validation_threshold_summary.json"
        ).exists()

        runtime[
            "use_threshold_search"
        ] = bool(
            optimize_threshold
            or use_postprocessing
            or threshold_exists
        )

        runtime[
            "use_postprocessing"
        ] = bool(
            use_postprocessing
        )

        config_path = save_json(
            runtime,
            ROAD_UI_CONFIG_ROOT
            / (
                f"{safe_token(selected)}_"
                f"validation_{now_token()}.json"
            ),
        )

        failed = False

        if (
            optimize_threshold
            or (
                use_postprocessing
                and not threshold_exists
            )
        ):
            code = run_road_stage(
                "Road Threshold Search",
                threshold_command(
                    config_path
                ),
                f"{selected}_threshold",
            )

            failed = (
                code
                != 0
            )

        if (
            use_postprocessing
            and not failed
        ):
            run_road_stage(
                "Road Post-processing Search",
                postprocess_command(
                    config_path
                ),
                f"{selected}_postprocess",
            )


# ============================================================
# 3. RESULTS
# ============================================================

def _results_backbone_for_experiment(
    experiment_name: str,
) -> str:
    """
    Resolve the actual backbone selected for a Road experiment.

    Priority:
    1. experiment configuration
    2. training summary
    3. Optuna selection summary
    4. Master Optuna selection summary
    5. frozen final settings

    This keeps the Results table tied to the experiment artifacts rather
    than displaying the generic backend type.
    """

    folder = (
        ROADS_EXPERIMENTS_ROOT
        / str(
            experiment_name
        )
    )

    payloads = [
        experiment_config(
            folder
        ),
        load_json(
            folder
            / "training_summary.json",
            {},
        ),
        load_json(
            folder
            / "optuna_selection_summary.json",
            {},
        ),
        load_json(
            folder
            / "master_optuna_selection_summary.json",
            {},
        ),
        load_json(
            folder
            / "final_settings.json",
            {},
        ),
    ]

    direct_keys = (
        "backbone",
        "selected_backbone",
        "best_backbone",
        "winning_backbone",
    )

    def find_backbone(
        value,
    ):
        if isinstance(
            value,
            dict,
        ):

            for key in direct_keys:

                candidate = value.get(
                    key
                )

                if (
                    candidate is not None
                    and str(
                        candidate
                    ).strip()
                ):
                    return str(
                        candidate
                    ).strip()

            # Common nested result/config containers.
            for key in (
                "best_params",
                "best_parameters",
                "selected_config",
                "winning_config",
                "winner",
                "trial_params",
                "params",
                "model_config",
                "experiment_config",
                "frozen_settings",
                "settings",
            ):

                if key in value:

                    result = find_backbone(
                        value[
                            key
                        ]
                    )

                    if result:
                        return result

        return None

    for payload in payloads:

        result = find_backbone(
            payload
        )

        if result:
            return result

    return "—"


def render_results() -> None:

    render_section(
        "3",
        "Results",
        (
            "Road model settings, validation metrics and independent "
            "source-specific final-test metrics."
        ),
    )

    dataframe = result_rows(
        "roads",
        ROADS_EXPERIMENTS_ROOT,
    )

    # --------------------------------------------------------
    # ROAD RESULTS TABLE PRESENTATION
    # --------------------------------------------------------
    # Requested UI:
    # - remove Backend
    # - add the backbone used by each experiment
    # - remove clDice
    #
    # The backend is still preserved internally in experiment metadata;
    # this only changes the Results table presentation.
    # --------------------------------------------------------

    if hasattr(
        dataframe,
        "columns",
    ):

        dataframe = dataframe.copy()

        columns_to_drop = []

        for column in list(
            dataframe.columns
        ):

            normalized = (
                str(
                    column
                )
                .strip()
                .lower()
                .replace(
                    "_",
                    "",
                )
                .replace(
                    " ",
                    "",
                )
                .replace(
                    "-",
                    "",
                )
            )

            if normalized == "backend":
                columns_to_drop.append(
                    column
                )

            if normalized == "cldice":
                columns_to_drop.append(
                    column
                )

        if columns_to_drop:

            dataframe = dataframe.drop(
                columns=columns_to_drop,
                errors="ignore",
            )

        # Rebuild Backbone from the actual experiment artifacts so the
        # table shows the backbone chosen for that experiment.
        if "Backbone" in dataframe.columns:

            dataframe = dataframe.drop(
                columns=[
                    "Backbone"
                ],
            )

        if "Experiment" in dataframe.columns:

            backbone_values = [
                _results_backbone_for_experiment(
                    experiment_name
                )
                for experiment_name
                in dataframe[
                    "Experiment"
                ].tolist()
            ]

            model_index = (
                list(
                    dataframe.columns
                ).index(
                    "Model"
                )
                + 1
                if "Model"
                in dataframe.columns
                else 1
            )

            dataframe.insert(
                model_index,
                "Backbone",
                backbone_values,
            )

    render_results_table(
        dataframe
    )


# ============================================================
# FINAL SOURCE CONFIG
# ============================================================

def _road_final_tests() -> dict:

    paths = load_json(
        ROADS_PATHS_CONFIG,
        {},
    )

    final_tests = paths.get(
        "final_tests",
        {},
    )

    return (
        final_tests
        if isinstance(
            final_tests,
            dict,
        )
        else {}
    )


# ============================================================
# 4. FINAL TEST
# ============================================================

def render_final_test() -> None:

    render_section(
        "4",
        "Final Test",
        (
            "Choose the trained Road model and exact experiment to test. "
            "Only experiments with an existing trained checkpoint are shown. "
            "The selected experiment's frozen Validation settings are then "
            "used unchanged for Aerial, Satellite and Drone."
        ),
    )

    # --------------------------------------------------------
    # DISCOVER TRAINED / FINAL-READY EXPERIMENTS
    # --------------------------------------------------------

    all_folders = experiment_folders(
        ROADS_EXPERIMENTS_ROOT
    )

    if not all_folders:

        st.info(
            "No trained Road experiment is available."
        )
        return

    trained_by_model = {}

    for candidate_folder in all_folders:

        candidate_config = experiment_config(
            candidate_folder
        )

        if not candidate_config:
            continue

        candidate_model = str(
            candidate_config.get(
                "model_type",
                "",
            )
        ).lower()

        model_info = MODEL_INFO.get(
            candidate_model
        )

        if (
            not model_info
            or not model_info.get(
                "final_ready",
                False,
            )
        ):
            continue

        candidate_checkpoint = (
            candidate_folder
            / "best_model.pth"
        )

        if not candidate_checkpoint.exists():
            continue

        trained_by_model.setdefault(
            candidate_model,
            [],
        ).append(
            (
                candidate_folder,
                candidate_config,
            )
        )

    if not trained_by_model:

        st.warning(
            "No final-test-ready Road experiment with best_model.pth "
            "was found."
        )
        return

    # --------------------------------------------------------
    # MODEL SELECTOR FOR FINAL TEST
    # --------------------------------------------------------
    # Final Test owns its own model selector.
    # Setup & Train is used only as a preferred default when that model
    # actually has a trained final-ready experiment.

    available_model_keys = [
        key
        for key in MODEL_INFO.keys()
        if key in trained_by_model
    ]

    available_model_labels = [
        MODEL_INFO[
            key
        ][
            "label"
        ]
        for key in available_model_keys
    ]

    setup_model_label = st.session_state.get(
        "roads_model"
    )

    preferred_model_index = 0

    if setup_model_label in available_model_labels:

        preferred_model_index = (
            available_model_labels.index(
                setup_model_label
            )
        )

    cached_final_model = st.session_state.get(
        "roads_final_model"
    )

    if (
        cached_final_model is not None
        and cached_final_model
        not in available_model_labels
    ):
        st.session_state.pop(
            "roads_final_model",
            None,
        )

    selected_model_label = st.selectbox(
        "Model to Final-Test",
        available_model_labels,
        index=preferred_model_index,
        key="roads_final_model",
        help=(
            "Only models that already have at least one trained "
            "final-test-ready experiment are listed."
        ),
    )

    selected_model_key = (
        available_model_keys[
            available_model_labels.index(
                selected_model_label
            )
        ]
    )

    selected_model_info = MODEL_INFO[
        selected_model_key
    ]

    # --------------------------------------------------------
    # EXACT TRAINED EXPERIMENT
    # --------------------------------------------------------

    compatible = trained_by_model[
        selected_model_key
    ]

    compatible = sorted(
        compatible,
        key=lambda item: item[0].name.lower(),
    )

    experiment_names = [
        folder.name
        for folder, _
        in compatible
    ]

    setup_experiment_name = safe_token(
        st.session_state.get(
            "roads_experiment_name",
            "",
        )
    )

    preferred_experiment_index = (
        experiment_names.index(
            setup_experiment_name
        )
        if setup_experiment_name
        in experiment_names
        else 0
    )

    cached_final_experiment = st.session_state.get(
        "roads_final_experiment"
    )

    if (
        cached_final_experiment is not None
        and cached_final_experiment
        not in experiment_names
    ):
        st.session_state.pop(
            "roads_final_experiment",
            None,
        )

    selected = st.selectbox(
        "Trained Experiment",
        experiment_names,
        index=preferred_experiment_index,
        key="roads_final_experiment",
        help=(
            "Choose the exact completed training run whose checkpoint "
            "will be frozen and tested."
        ),
    )

    folder = (
        ROADS_EXPERIMENTS_ROOT
        / selected
    )

    config = experiment_config(
        folder
    )

    if not config:

        st.error(
            "Could not load this Road experiment configuration."
        )
        return

    model_key = str(
        config.get(
            "model_type",
            "",
        )
    ).lower()

    if model_key != selected_model_key:

        st.error(
            "Final Test blocked because the selected experiment does not "
            "belong to the selected model."
        )
        return

    checkpoint_path = (
        folder
        / "best_model.pth"
    )

    if not checkpoint_path.exists():

        st.error(
            "Final Test blocked because this experiment does not contain "
            "best_model.pth."
        )
        return

    # --------------------------------------------------------
    # EXACT MODEL / CHECKPOINT CONFIRMATION
    # --------------------------------------------------------

    training_summary = load_json(
        folder
        / "training_summary.json",
        {},
    )

    optuna_summary = load_json(
        folder
        / "optuna_selection_summary.json",
        {},
    )

    master_summary = load_json(
        folder
        / "master_optuna_selection_summary.json",
        {},
    )

    backbone = config.get(
        "backbone",
        "—",
    )

    tile_size = config.get(
        "tile_size",
        "—",
    )

    if bool(
        config.get(
            "use_master_optuna",
            False,
        )
    ):
        training_mode = "Master Optuna"

    elif bool(
        config.get(
            "use_optuna",
            False,
        )
    ):
        training_mode = "General Optuna"

    elif bool(
        config.get(
            "use_augmentation",
            False,
        )
    ):
        training_mode = "Augmentation"

    else:
        training_mode = "Baseline"

    best_validation_iou = (
        training_summary.get(
            "best_validation_iou"
        )
        or training_summary.get(
            "best_val_iou"
        )
        or optuna_summary.get(
            "best_value"
        )
        or master_summary.get(
            "best_value"
        )
    )

    if best_validation_iou is not None:

        try:
            best_validation_iou = float(
                best_validation_iou
            )

            if best_validation_iou <= 1.0:
                best_validation_iou *= 100.0

            best_validation_text = (
                f"{best_validation_iou:.2f}%"
            )

        except Exception:
            best_validation_text = "—"

    else:
        best_validation_text = "—"

    _html(
        f"""
        <div class="info-banner">
            <b>MODEL THAT WILL BE FINAL-TESTED</b><br>
            <b>Model:</b> {selected_model_info.get("label", selected_model_key)}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Experiment:</b> {selected}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Backbone:</b> {backbone}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Tile:</b> {tile_size}
            <br>
            <b>Training:</b> {training_mode}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Best Validation IoU:</b> {best_validation_text}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Checkpoint:</b> Ready
        </div>
        """
    )

    st.caption(
        f"Checkpoint: {checkpoint_path}"
    )

    # --------------------------------------------------------
    # FINAL-TEST SOURCE
    # --------------------------------------------------------

    final_tests = _road_final_tests()

    source_options = []

    for source in (
        "aerial",
        "satellite",
        "drone",
    ):
        source_config = final_tests.get(
            source,
            {},
        )

        if source_config.get(
            "enabled",
            False,
        ):
            source_options.append(
                source
            )

    if not source_options:

        st.warning(
            "No Road final-test source is currently enabled in paths.json."
        )
        return

    source = st.selectbox(
        "Final-test Source",
        source_options,
        key="roads_final_source",
    )

    source_config = final_tests.get(
        source,
        {},
    )

    raster = source_config.get(
        "raster"
    )

    ground_truth = source_config.get(
        "ground_truth"
    )

    native_resolution = source_config.get(
        "native_resolution_m"
    )

    threshold_summary = load_json(
        folder
        / "validation_threshold_summary.json",
        {},
    )

    post_summary = load_json(
        folder
        / "validation_postprocess_summary.json",
        {},
    )

    threshold_value = (
        threshold_summary.get(
            "best_threshold"
        )
        if threshold_summary
        else 0.5
    )

    post_enabled = bool(
        post_summary
    )

    _html(
        f"""
        <div class="info-banner">
            <b>Source:</b> {source.title()}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Raster:</b> {"Configured" if raster else "Missing"}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Ground Truth:</b> {"Configured" if ground_truth else "Missing"}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Native Resolution:</b>
            {native_resolution if native_resolution is not None else "—"} m
            <br>
            <b>Frozen Threshold:</b> {float(threshold_value):.3f}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Post-processing:</b> {"ON" if post_enabled else "OFF"}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Settings Source:</b> Validation
        </div>
        """
    )

    if (
        not raster
        or not ground_truth
    ):
        st.warning(
            "This source cannot be tested yet because its raster or "
            "Ground Truth path is still missing."
        )
        return

    source_final_folder = (
        folder
        / "final_test"
        / source
    )

    receipt = (
        source_final_folder
        / "final_test_receipt.json"
    )

    if receipt.exists():
        st.warning(
            f"{source.title()} final test already has a receipt."
        )

    confirmed = st.checkbox(
        (
            f"I confirm final testing of {selected_model_info.get('label', selected_model_key)} "
            f"using experiment '{selected}' and its frozen Validation settings."
        ),
        value=False,
        disabled=receipt.exists(),
        key="roads_final_confirm",
    )

    if st.button(
        (
            f"Freeze & Run {selected_model_info.get('label', selected_model_key)} "
            f"Final Test — {source.title()}"
        ),
        type="primary",
        use_container_width=True,
        disabled=(
            not confirmed
            or receipt.exists()
        ),
        key="roads_final_button",
    ):

        runtime = deepcopy(
            config
        )

        # Hard-bind Final Test to the model / experiment shown above.
        runtime[
            "experiment_id"
        ] = selected

        runtime[
            "model_type"
        ] = selected_model_key

        runtime[
            "backbone"
        ] = backbone

        runtime[
            "tile_size"
        ] = int(
            tile_size
        )

        runtime[
            "use_threshold_search"
        ] = bool(
            threshold_summary
        )

        runtime[
            "use_postprocessing"
        ] = bool(
            post_summary
        )

        config_path = save_json(
            runtime,
            ROAD_UI_CONFIG_ROOT
            / (
                f"{safe_token(selected)}_"
                f"final_{source}_"
                f"{now_token()}.json"
            ),
        )

        freeze_code = run_road_stage(
            (
                "Freeze Road Final Settings — "
                f"{selected_model_info.get('label', selected_model_key)}"
            ),
            freeze_command(
                config_path
            ),
            f"{selected}_freeze",
        )

        if freeze_code == 0:
            run_road_stage(
                (
                    f"{selected_model_info.get('label', selected_model_key)} "
                    f"Final Test — {source.title()}"
                ),
                final_test_command(
                    config_path,
                    source,
                ),
                (
                    f"{selected}_"
                    f"{source}_final"
                ),
            )


# ============================================================
# 5. MAP RESULTS
# ============================================================

def render_map_results() -> None:

    render_section(
        "5",
        "Map Results",
        (
            "The map automatically follows the model, trained experiment "
            "and source selected in Final Test. No second result selector "
            "is used."
        ),
    )

    # --------------------------------------------------------
    # FOLLOW THE EXACT FINAL-TEST SELECTION
    # --------------------------------------------------------

    selected_model_label = st.session_state.get(
        "roads_final_model"
    )

    selected_experiment = st.session_state.get(
        "roads_final_experiment"
    )

    selected_source = st.session_state.get(
        "roads_final_source"
    )

    if (
        not selected_model_label
        or not selected_experiment
        or not selected_source
    ):

        st.info(
            "Choose a model, trained experiment and source in Final Test "
            "first. The completed result will appear here automatically."
        )
        return

    folder = (
        ROADS_EXPERIMENTS_ROOT
        / selected_experiment
    )

    if not folder.exists():

        st.warning(
            "The experiment selected in Final Test no longer exists."
        )
        return

    config = experiment_config(
        folder
    )

    if not config:

        st.error(
            "Could not load the selected Road experiment configuration."
        )
        return

    selected_model_key = str(
        config.get(
            "model_type",
            "",
        )
    ).lower()

    expected_model_label = (
        MODEL_INFO.get(
            selected_model_key,
            {},
        ).get(
            "label",
            selected_model_key,
        )
    )

    if selected_model_label != expected_model_label:

        st.warning(
            "The Final Test model selection changed. Re-select the trained "
            "experiment before viewing its map."
        )
        return

    # --------------------------------------------------------
    # FIND THE COMPLETED RESULT FOR THAT EXACT SOURCE
    # --------------------------------------------------------

    finals = final_metric_payloads(
        folder,
        "roads",
    )

    metrics = None

    for source, payload in finals:

        if str(
            source
        ).lower() == str(
            selected_source
        ).lower():

            metrics = payload
            break

    if metrics is None:

        st.info(
            (
                f"No completed {selected_source.title()} Final Test exists "
                f"yet for '{selected_experiment}'. Run that selected Final "
                "Test first; its map will then appear here automatically."
            )
        )
        return

    # --------------------------------------------------------
    # DISPLAY EXACT SELECTED MODEL / EXPERIMENT / SOURCE
    # --------------------------------------------------------

    final_tests = _road_final_tests()

    source_config = final_tests.get(
        selected_source,
        {},
    )

    raster = (
        metrics.get(
            "test_raster"
        )
        or metrics.get(
            "test_image_path"
        )
        or source_config.get(
            "raster"
        )
    )

    ground_truth = (
        metrics.get(
            "ground_truth_projected"
        )
        or metrics.get(
            "ground_truth_path"
        )
        or metrics.get(
            "ground_truth"
        )
        or source_config.get(
            "ground_truth"
        )
    )

    prediction = (
        metrics.get(
            "prediction_fc"
        )
        or metrics.get(
            "final_prediction_fc"
        )
        or metrics.get(
            "final_prediction"
        )
        or metrics.get(
            "prediction"
        )
    )

    if not raster:

        st.error(
            "The selected final result does not contain a test raster path."
        )
        return

    if not ground_truth:

        st.error(
            "The selected final result does not contain a Ground Truth path."
        )
        return

    if not prediction:

        st.error(
            "The selected final result does not contain a prediction path."
        )
        return

    model_info = MODEL_INFO.get(
        selected_model_key,
        {},
    )

    backbone = config.get(
        "backbone",
        "—",
    )

    tile_size = config.get(
        "tile_size",
        "—",
    )

    iou = metrics.get(
        "iou_percent"
    )

    precision = metrics.get(
        "precision_percent"
    )

    recall = metrics.get(
        "recall_percent"
    )

    f1 = metrics.get(
        "f1_percent"
    )

    def metric_text(
        value,
    ):

        try:
            return f"{float(value):.2f}%"

        except Exception:
            return "—"

    _html(
        f"""
        <div class="info-banner">
            <b>SELECTED FINAL RESULT</b><br>
            <b>Model:</b> {model_info.get("label", selected_model_key)}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Experiment:</b> {selected_experiment}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Backbone:</b> {backbone}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Tile:</b> {tile_size}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Source:</b> {selected_source.title()}
            <br>
            <b>IoU:</b> {metric_text(iou)}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Precision:</b> {metric_text(precision)}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>Recall:</b> {metric_text(recall)}
            &nbsp;&nbsp;·&nbsp;&nbsp;
            <b>F1:</b> {metric_text(f1)}
        </div>
        """
    )

    render_final_overlay(
        raster_path=raster,
        ground_truth_path=ground_truth,
        prediction_path=prediction,
        caption=(
            f"{selected_experiment} — "
            f"{selected_source.title()} Final Test"
        ),
        key_prefix=(
            f"roads_map_"
            f"{safe_token(selected_experiment)}_"
            f"{selected_source}"
        ),
    )


# ============================================================
# PAGE
# ============================================================

def render() -> None:

    with st.container(
        key="roads_page"
    ):
        render_header(
            title="Road Extraction",
            subtitle=(
                "Approach 2 · Mixed-source Road-surface extraction with "
                "strict validation selection and frozen final testing."
            ),
            eyebrow="APPROACH 2",
        )

        render_setup_and_train()
        render_validation_options()
        render_results()
        render_final_test()
        render_map_results()

# ============================================================
# DIRECT STREAMLIT PAGE SUPPORT
# ============================================================

if __name__ == "__main__":
    render()
