"""
ui/pages/buildings.py

Native Building Extraction page for the ONE shared thesis Streamlit UI.

Architecture:
    ui/app.py
        -> ui/pages/buildings.py
            -> Buildings/scripts/*
            -> Buildings/src/*
            -> Buildings/config/*
            -> Buildings/outputs/*

There is NO dependency on Buildings/ui/.

The page preserves the complete Buildings workflow:
1. Setup & Train
2. Validation Options
3. Results
4. Final Test
5. Map Results

Final Test safety:
- only experiment folders containing best_model.pth are selectable
- runtime experiment_id is hard-bound to the selected folder
- validation/final settings remain experiment-specific
- Map Results follows the exact experiment selected in Final Test
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re
import subprocess
import sys
import textwrap

import pandas as pd
import streamlit as st


# ============================================================
# PATHS — ONE SHARED UI
# ============================================================

THIS_FILE = Path(__file__).resolve()
PAGES_ROOT = THIS_FILE.parent
UI_ROOT = PAGES_ROOT.parent
PROJECT_ROOT = UI_ROOT.parent

BUILDINGS_ROOT = (
    PROJECT_ROOT
    / "Buildings"
)

CONFIG_ROOT = BUILDINGS_ROOT / "config"
SCRIPTS_ROOT = BUILDINGS_ROOT / "scripts"
OUTPUTS_ROOT = BUILDINGS_ROOT / "outputs"
EXPERIMENTS_ROOT = OUTPUTS_ROOT / "experiments"

DATASETS_CONFIG = CONFIG_ROOT / "datasets.json"
EXPERIMENT_CONFIG = CONFIG_ROOT / "experiment.json"
TEST_ZONES_CONFIG = CONFIG_ROOT / "test_zones.json"
PATHS_CONFIG = CONFIG_ROOT / "paths.json"

RUN_BUILDINGS = SCRIPTS_ROOT / "run_buildings.py"

UI_OUTPUT_ROOT = OUTPUTS_ROOT / "ui"
UI_CONFIG_ROOT = UI_OUTPUT_ROOT / "configs"
UI_LOG_ROOT = UI_OUTPUT_ROOT / "logs"

UI_CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
UI_LOG_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# INTERNAL REPRODUCIBILITY
# ============================================================

# Kept fixed in backend configs for deterministic Train/Validation splitting.
# This value is intentionally not exposed in the UI.
FIXED_SPLIT_SEED = 42


for value in (
    str(BUILDINGS_ROOT),
    str(UI_ROOT),
):
    while value in sys.path:
        sys.path.remove(value)

# Buildings first so `src.*` resolves to the Buildings backend.
sys.path.insert(0, str(BUILDINGS_ROOT))
sys.path.insert(1, str(UI_ROOT))


from src.pipeline.model_registry import MODEL_REGISTRY, get_model_spec
from components.header import render_header


# ============================================================
# HTML
# ============================================================

def _html(
    content: str,
) -> None:
    cleaned = "\n".join(
        line.strip()
        for line in textwrap.dedent(
            content
        ).splitlines()
        if line.strip()
    )

    if hasattr(
        st,
        "html",
    ):
        st.html(
            cleaned
        )
    else:
        st.markdown(
            cleaned,
            unsafe_allow_html=True,
        )


BUILDING_ROAD_MATCH_CSS = r"""
<style>

/* =========================================================
   BUILDINGS — IMAGE OUTSIDE CONTENT CARDS
   ========================================================= */

.stApp {
    background: #07101d !important;
}

.stApp:has(.st-key-buildings_page) {
    background:
        linear-gradient(
            180deg,
            rgba(4, 15, 28, 0.60) 0px,
            rgba(4, 15, 28, 0.46) 180px,
            rgba(4, 15, 28, 0.40) 360px,
            rgba(5, 17, 30, 0.68) 620px,
            #07101d 920px,
            #07101d 100%
        ),
        linear-gradient(
            90deg,
            rgba(5, 18, 34, 0.52) 0%,
            rgba(5, 18, 34, 0.27) 55%,
            rgba(5, 18, 34, 0.14) 100%
        ),
        url("https://images.unsplash.com/photo-1516280740075-5c4bef5821b4?auto=format&fit=crop&fm=jpg&q=85&w=2400")
        center 110px / cover no-repeat fixed !important;
}

.block-container {
    max-width: 1320px !important;
    padding-top: 0.25rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-bottom: 4rem !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
}

/* Hide only the legacy standalone Building header. */
.header {
    display: none !important;
}

/*
The aerial BUILDING image belongs to the FULL page background,
outside the Building workspace/content cards.
*/
.st-key-buildings_page {
    position: relative !important;
    isolation: auto !important;
    overflow: visible !important;
    margin-top: 0.45rem !important;
    padding: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
}

.st-key-buildings_page > div {
    position: relative;
    z-index: 1;
}

/* ---------------------------------------------------------
   CLEAN BUILDING HERO — NO PHOTO INSIDE.
   --------------------------------------------------------- */

.st-key-buildings_page .hero {
    position: relative !important;
    overflow: hidden !important;
    min-height: 165px !important;

    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;

    padding:
        1.55rem
        1.7rem !important;

    border:
        1px solid
        rgba(71, 183, 255, 0.24) !important;

    border-radius:
        18px !important;

    background:
        linear-gradient(
            120deg,
            rgba(9, 27, 46, 0.98),
            rgba(10, 37, 58, 0.96)
        ) !important;

    box-shadow:
        0 18px 44px rgba(0, 0, 0, 0.28),
        inset 0 1px 0 rgba(255, 255, 255, 0.035) !important;
}

.st-key-buildings_page .hero::after {
    content: none !important;
}

.st-key-buildings_page .hero > * {
    position: relative;
    z-index: 1;
    max-width: 790px;
}

.st-key-buildings_page .hero-eyebrow {
    color: #5bd8ff !important;
    letter-spacing: 0.16em !important;
    font-weight: 900 !important;
}

.st-key-buildings_page .hero-title {
    color: #ffffff !important;
    font-size: clamp(1.85rem, 3vw, 2.65rem) !important;
    font-weight: 900 !important;
    letter-spacing: -0.03em !important;
    text-shadow: none !important;
}

.st-key-buildings_page .hero-subtitle {
    color: #c0d4e6 !important;
}

/* ---------------------------------------------------------
   SECTION HEADERS
   --------------------------------------------------------- */

.st-key-buildings_page .section-header {
    margin:
        1.55rem
        0
        0.82rem
        0 !important;

    padding:
        0.86rem
        0.95rem !important;

    border:
        1px solid
        rgba(148, 163, 184, 0.15) !important;

    border-left:
        3px solid
        #20aef4 !important;

    border-radius:
        13px !important;

    background:
        linear-gradient(
            100deg,
            rgba(16, 36, 58, 0.98),
            rgba(10, 28, 47, 0.97)
        ) !important;

    box-shadow:
        0 8px 24px rgba(0, 0, 0, 0.12) !important;
}

.st-key-buildings_page .section-title {
    color: #f8fafc !important;
    font-size: 1.02rem !important;
    font-weight: 900 !important;
}

.st-key-buildings_page .section-subtitle {
    color: #8fa4bb !important;
}

/* ---------------------------------------------------------
   BUILDING SETUP CARDS
   --------------------------------------------------------- */

.st-key-buildings_page
[data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"])
[data-testid="stColumn"] {
    padding:
        0.72rem
        0.72rem
        0.64rem !important;

    border-radius:
        13px !important;

    border:
        1px solid
        rgba(124, 170, 207, 0.22) !important;

    background:
        linear-gradient(
            145deg,
            rgba(25, 48, 73, 0.97),
            rgba(14, 34, 55, 0.98)
        ) !important;

    box-shadow:
        0 10px 25px rgba(0, 0, 0, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.03) !important;
}

.st-key-buildings_page
[data-testid="stHorizontalBlock"]:has([data-testid="stSelectbox"])
[data-testid="stColumn"] label {
    color: #d9e9f7 !important;
    font-size: 0.77rem !important;
    font-weight: 900 !important;
}

.st-key-buildings_page div[data-baseweb="select"] > div,
.st-key-buildings_page [data-testid="stTextInput"] input,
.st-key-buildings_page [data-testid="stNumberInput"] input {
    min-height: 46px !important;
    color: #f8fafc !important;

    border:
        1px solid
        rgba(140, 183, 218, 0.20) !important;

    border-radius:
        10px !important;

    background:
        rgba(19, 40, 63, 0.96) !important;
}

.st-key-buildings_page div[data-baseweb="select"] span,
.st-key-buildings_page div[data-baseweb="select"] svg {
    color: #e3f0fa !important;
    fill: #e3f0fa !important;
}

/* ---------------------------------------------------------
   BUILDING INFO / SPLIT STRIPS
   --------------------------------------------------------- */

.st-key-buildings_page .data-line,
.st-key-buildings_page .split-line,
.st-key-buildings_page .small-info {
    margin:
        0.50rem
        0 !important;

    padding:
        0.75rem
        0.88rem !important;

    border:
        1px solid
        rgba(83, 178, 238, 0.18) !important;

    border-radius:
        11px !important;

    background:
        linear-gradient(
            90deg,
            rgba(18, 53, 84, 0.92),
            rgba(11, 34, 57, 0.90)
        ) !important;

    color:
        #d5e7f5 !important;
}

/* ---------------------------------------------------------
   BUILDING CHECKBOX / OPTIMIZATION CARDS
   --------------------------------------------------------- */

.st-key-buildings_page
[data-testid="stHorizontalBlock"]:has([data-testid="stCheckbox"])
[data-testid="stColumn"] {
    min-height: 66px !important;
    display: flex !important;
    align-items: center !important;

    padding:
        0.58rem
        0.72rem !important;

    border:
        1px solid
        rgba(119, 155, 190, 0.18) !important;

    border-radius:
        12px !important;

    background:
        linear-gradient(
            145deg,
            rgba(21, 43, 67, 0.97),
            rgba(12, 29, 48, 0.98)
        ) !important;
}

.st-key-buildings_page [data-testid="stCheckbox"] label {
    color: #d8e8f6 !important;
    font-weight: 850 !important;
}

/* ---------------------------------------------------------
   BUILDING BUTTONS
   --------------------------------------------------------- */

.st-key-buildings_page .stButton > button {
    min-height: 47px !important;
    border-radius: 10px !important;
    font-weight: 900 !important;

    transition:
        transform 0.16s ease,
        filter 0.16s ease,
        box-shadow 0.16s ease !important;
}

.st-key-buildings_page .stButton > button[kind="primary"] {
    color: #ffffff !important;
    border: 0 !important;

    background:
        linear-gradient(
            90deg,
            #0588ff,
            #16a9f4
        ) !important;

    box-shadow:
        0 11px 25px
        rgba(0, 0, 0, 0.22) !important;
}

.st-key-buildings_page .stButton > button[kind="primary"]:hover {
    filter: brightness(1.08) !important;
    transform: translateY(-1px) !important;
}

.st-key-buildings_page .stButton > button:not([kind="primary"]) {
    color: #dbeafe !important;

    border:
        1px solid
        rgba(125, 211, 252, 0.17) !important;

    background:
        rgba(18, 39, 62, 0.86) !important;
}

/* ---------------------------------------------------------
   BUILDING RESULTS / METRICS
   --------------------------------------------------------- */

.st-key-buildings_page [data-testid="stAlert"] {
    border-radius: 11px !important;

    border:
        1px solid
        rgba(148, 163, 184, 0.14) !important;

    background:
        rgba(15, 31, 49, 0.92) !important;
}

.st-key-buildings_page [data-testid="stMetric"] {
    padding: 0.82rem !important;

    border:
        1px solid
        rgba(125, 211, 252, 0.13) !important;

    border-radius:
        12px !important;

    background:
        linear-gradient(
            145deg,
            rgba(17, 35, 56, 0.97),
            rgba(10, 25, 42, 0.98)
        ) !important;

    box-shadow:
        0 12px 30px
        rgba(0, 0, 0, 0.14) !important;
}

.st-key-buildings_page [data-testid="stMetricLabel"] {
    color: #8fa4bb !important;
}

.st-key-buildings_page [data-testid="stMetricValue"] {
    color: #f8fafc !important;
}

.st-key-buildings_page [data-testid="stDataFrame"],
.st-key-buildings_page [data-testid="stExpander"] {
    border-radius: 13px !important;
    overflow: hidden !important;

    box-shadow:
        0 12px 30px
        rgba(0, 0, 0, 0.14) !important;
}

@media (max-width: 900px) {
    .st-key-buildings_page {
        padding: 0 !important;
        background: transparent !important;
    }

    .st-key-buildings_page .hero {
        min-height: 150px !important;
    }
}

</style>
"""


# ============================================================
# HELPERS
# ============================================================

def load_json(path: Path, default=None):
    if not Path(path).exists():
        return {} if default is None else default

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8")
        )
    except Exception:
        return {} if default is None else default


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def safe_token(value):
    value = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        str(value).strip().lower(),
    )
    return re.sub(r"_+", "_", value).strip("_")


def now_token():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def first_json(folder: Path, names):
    for name in names:
        payload = load_json(folder / name, {})
        if payload:
            return payload
    return {}


def percentage(payload, *keys):
    for key in keys:
        value = payload.get(key)

        if value is None:
            continue

        try:
            value = float(value)
        except Exception:
            continue

        if key.endswith("_percent"):
            return value

        return value * 100.0 if 0 <= value <= 1 else value

    return None


def fmt(value):
    if value is None:
        return "—"

    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


# ============================================================
# MODELS / BACKBONES
# ============================================================

def model_name(model_key):
    return MODEL_REGISTRY[model_key].display_name


def model_key_from_name(name):
    for key, spec in MODEL_REGISTRY.items():
        if spec.display_name == name:
            return key

    raise KeyError(name)


def load_model_config(model_key):
    for path in (
        CONFIG_ROOT / "models" / f"{model_key}.json",
        CONFIG_ROOT / f"{model_key}.json",
    ):
        payload = load_json(path, {})
        if payload:
            return payload

    return {}


def backbones_for(model_key):
    payload = load_model_config(model_key)

    for key in ("supported_backbones", "backbones"):
        value = payload.get(key)

        if isinstance(value, dict) and value:
            return list(value.keys())

        if isinstance(value, list) and value:
            return [str(item) for item in value]

    return list(
        get_model_spec(model_key).supported_backbones
    )


# ============================================================
# DATASETS
# ============================================================

DATASET_RE = re.compile(
    r"^(?P<source>.+)_(?P<dtype>ct|rcnn)_(?P<tile>\d+)$",
    re.IGNORECASE,
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


def discover_datasets(payload):
    rows = []

    def walk(node):
        if not isinstance(node, dict):
            return

        for key, value in node.items():
            if not isinstance(value, dict):
                continue

            match = DATASET_RE.match(str(key))

            if match:
                rows.append(
                    {
                        "dataset_id": str(key),
                        "source": match.group("source").lower(),
                        "dataset_type": match.group("dtype").lower(),
                        "tile_size": int(match.group("tile")),
                        "path": (
                            value.get("path")
                            or value.get("dataset_path")
                            or value.get("folder")
                            or value.get("root")
                        ),
                    }
                )

            walk(value)

    walk(payload)

    return list(
        {
            row["dataset_id"]: row
            for row in rows
        }.values()
    )


DATASETS = discover_datasets(
    load_json(DATASETS_CONFIG, {})
)


def dataset_type_for_model(model_key):
    return (
        "ct"
        if get_model_spec(model_key).family == "semantic"
        else "rcnn"
    )


def sources_for(dataset_type):
    return sorted(
        {
            row["source"]
            for row in DATASETS
            if row["dataset_type"] == dataset_type
        }
    )


def tiles_for(source, dataset_type):
    return sorted(
        {
            int(row["tile_size"])
            for row in DATASETS
            if row["source"] == source
            and row["dataset_type"] == dataset_type
        }
    )


def dataset_for(source, dataset_type, tile_size):
    for row in DATASETS:
        if (
            row["source"] == source
            and row["dataset_type"] == dataset_type
            and int(row["tile_size"]) == int(tile_size)
        ):
            return row

    return None


@st.cache_data(show_spinner=False, ttl=120)
def dataset_file_info(dataset_path):
    path = Path(dataset_path)

    if not path.exists():
        return {
            "images": None,
            "labels": None,
            "paired": None,
        }

    def files_in(folder):
        if not folder.exists():
            return []

        return [
            item
            for item in folder.rglob("*")
            if item.is_file()
            and item.suffix.lower() in IMAGE_EXTENSIONS
        ]

    images = files_in(path / "images")
    labels = files_in(path / "labels")

    image_stems = {item.stem for item in images}
    label_stems = {item.stem for item in labels}

    return {
        "images": len(images),
        "labels": len(labels),
        "paired": len(image_stems & label_stems),
    }


# ============================================================
# EXPERIMENTS / RESULTS
# ============================================================

def experiment_folders():
    if not EXPERIMENTS_ROOT.exists():
        return []

    return sorted(
        [
            folder
            for folder in EXPERIMENTS_ROOT.iterdir()
            if folder.is_dir()
            and not folder.name.lower().startswith(
                ("smoke_", "ui_smoke_")
            )
        ],
        key=lambda folder: folder.stat().st_mtime,
        reverse=True,
    )


def experiment_config(folder):
    for path in (
        folder / "experiment_config_used.json",
        folder / "config_used.json",
    ):
        payload = load_json(path, {})

        if payload.get("experiment_id"):
            return payload

        for key in ("experiment", "experiment_config"):
            nested = payload.get(key)
            if isinstance(nested, dict) and nested.get("experiment_id"):
                return nested

    # Backward-compatible UI-config fallback.
    candidates = []

    for path in UI_CONFIG_ROOT.glob("*.json"):
        payload = load_json(path, {})

        if payload.get("experiment_id") == folder.name:
            candidates.append(
                (path.stat().st_mtime, payload)
            )

    if candidates:
        return sorted(candidates, reverse=True)[0][1]

    return {}


def split_info(folder, config):
    summary = first_json(
        folder,
        [
            "split_summary.json",
            "manifest_summary.json",
        ],
    )

    data_split = config.get("data_split", {})

    train_percent = summary.get(
        "train_percent",
        data_split.get("train_percent"),
    )

    validation_percent = summary.get(
        "validation_percent",
        data_split.get("validation_percent"),
    )

    seed = summary.get(
        "seed",
        data_split.get("seed"),
    )

    return {
        "train_percent": train_percent,
        "validation_percent": validation_percent,
        "seed": seed,
        "train_count": summary.get("train_count"),
        "validation_count": summary.get("validation_count"),
        "selected_count": summary.get("total_selected_count"),
    }


def infer_legacy_metadata(folder_name, config):
    """
    Fill only safe metadata for older experiment folders whose original
    experiment_config_used.json is incomplete.

    We infer Model / Source / Tile from the experiment ID when possible.
    We do NOT invent split metrics or final results.
    """
    name = str(folder_name).lower()

    model = config.get("model_type")

    if not model:
        if (
            "sam_lora" in name
            or "sam-lora" in name
            or "samlora" in name
        ):
            model = "sam_lora"
        elif "deeplabv3" in name or "deeplab" in name:
            model = "deeplabv3"
        elif "maskrcnn" in name or "mask_rcnn" in name:
            model = "maskrcnn"
        elif "unet" in name or "u_net" in name:
            model = "unet"

    source = config.get("source_type")

    if not source:
        for candidate in (
            "satellite",
            "aerial",
            "drone",
        ):
            if candidate in name:
                source = candidate
                break

    tile = config.get("tile_size")

    if tile in (None, ""):
        tile_match = re.search(
            r"(?:^|_)(256|320|384|512|640|768|1024)(?:_|$)",
            name,
        )

        if tile_match:
            tile = int(
                tile_match.group(1)
            )

    backbone = config.get("backbone")

    return {
        "model": model or "—",
        "source": source or "—",
        "tile": tile if tile not in (None, "") else "—",
        "backbone": backbone or "—",
    }


def experiment_result_rows():
    """
    Professional result table:
    ONE ROW = ONE EXPERIMENT.

    All stages are represented as columns instead of creating repeated
    rows for Training / Optuna / Threshold / Post / Final Test.
    """
    rows = []

    for folder in experiment_folders():
        config = experiment_config(
            folder
        )

        split = split_info(
            folder,
            config,
        )

        meta = infer_legacy_metadata(
            folder.name,
            config,
        )

        training = first_json(
            folder,
            [
                "training_summary.json",
                "maskrcnn_training_summary.json",
            ],
        )

        optuna = first_json(
            folder,
            [
                "optuna_selection_summary.json",
                "run_optuna_summary.json",
            ],
        )

        master_optuna = first_json(
            folder,
            [
                "master_optuna_selection_summary.json",
            ],
        )

        threshold = first_json(
            folder,
            [
                "validation_threshold_summary.json",
            ],
        )

        post = first_json(
            folder,
            [
                "validation_postprocess_summary.json",
            ],
        )

        final = first_json(
            folder / "final_test",
            [
                "final_metrics.json",
            ],
        )

        # ------------------------------------------------------
        # Experiment settings
        # ------------------------------------------------------

        augmented = bool(
            config.get(
                "use_augmentation",
                False,
            )
        )

        if not augmented:
            augmented = (
                "augmented"
                in folder.name.lower()
            )

        optuna_used = bool(
            optuna
            or config.get(
                "use_optuna",
                False,
            )
        )

        master_optuna_used = bool(
            master_optuna
            or config.get(
                "use_master_optuna",
                False,
            )
        )

        train_percent = split.get(
            "train_percent"
        )

        validation_percent = split.get(
            "validation_percent"
        )

        split_text = (
            f"{float(train_percent):.0f}/{float(validation_percent):.0f}"
            if train_percent is not None
            and validation_percent is not None
            else "—"
        )

        # ------------------------------------------------------
        # Validation result from model training
        # ------------------------------------------------------

        validation_iou = percentage(
            training,
            "best_validation_iou_percent",
            "best_val_iou",
            "best_validation_iou",
            "validation_iou",
        )

        # Master Optuna promotes the exact winning trial checkpoint, so its
        # single winning Validation IoU is also the experiment Validation IoU.
        if validation_iou is None and master_optuna:
            validation_iou = percentage(
                master_optuna,
                "best_validation_iou_percent",
                "best_validation_iou",
            )

        # ------------------------------------------------------
        # Optuna
        # ------------------------------------------------------

        optuna_iou = percentage(
            optuna,
            "best_value_percent",
            "best_validation_iou_percent",
            "best_validation_iou",
            "best_value",
        )

        best_trial = (
            optuna.get(
                "best_trial_number",
                optuna.get(
                    "best_trial",
                ),
            )
            if optuna
            else None
        )

        # ------------------------------------------------------
        # Threshold
        # ------------------------------------------------------

        threshold_value = (
            threshold.get(
                "best_threshold",
                threshold.get(
                    "score_threshold",
                ),
            )
            if threshold
            else None
        )

        # ------------------------------------------------------
        # Post-processing
        # ------------------------------------------------------

        post_used = bool(
            post
            or config.get(
                "use_postprocessing",
                False,
            )
        )

        # ------------------------------------------------------
        # External testing zone / Final test
        # ------------------------------------------------------

        final_iou = percentage(
            final,
            "iou_percent",
            "iou",
        )

        precision = percentage(
            final,
            "precision_percent",
            "precision",
        )

        recall = percentage(
            final,
            "recall_percent",
            "recall",
        )

        final_f1 = percentage(
            final,
            "f1_percent",
            "f1",
        )

        source_key = str(meta["source"]).lower()
        source_test_options = FINAL_TEST_OPTIONS.get(source_key, [])

        if source_test_options:
            test_zone = source_test_options[0]["label"]
        else:
            test_zone_config = TEST_ZONES.get(source_key, {})
            test_zone = (
                test_zone_config.get("display_name")
                or (
                    f"{str(meta['source']).title()} Test Zone"
                    if meta["source"] != "—"
                    else "—"
                )
            )

        # ------------------------------------------------------
        # Status
        # ------------------------------------------------------

        if final:
            status = "Final Tested"
        elif post:
            status = "Post-processed"
        elif threshold:
            status = "Threshold Tuned"
        elif master_optuna:
            status = "Master Optuna Complete"
        elif optuna:
            status = "Optuna Complete"
        elif training:
            status = "Trained"
        else:
            status = "Incomplete"

        row = {
            "Experiment":
                folder.name,

            "Model":
                meta["model"],

            "Backbone":
                meta["backbone"],

            "Source":
                meta["source"],

            "Tile":
                meta["tile"],

            "Split":
                split_text,

            "Seed":
                (
                    split.get(
                        "seed"
                    )
                    if split.get(
                        "seed"
                    )
                    is not None
                    else "—"
                ),

            "Aug":
                "ON"
                if augmented
                else "OFF",

            "Optuna":
                "ON"
                if optuna_used
                else "OFF",

            "Master Optuna":
                "ON"
                if master_optuna_used
                else "OFF",

            "Validation IoU":
                validation_iou,

            "Optuna Val IoU":
                optuna_iou,

            "Threshold":
                threshold_value,

            "Post-processing":
                "ON"
                if post_used
                else "OFF",

            "Test Zone":
                test_zone,

            "Test IoU":
                final_iou,

            "Precision":
                precision,

            "Recall":
                recall,

            "F1":
                final_f1,

            "Status":
                status,
        }

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CONFIG CREATION
# ============================================================

BASE_CONFIG = load_json(EXPERIMENT_CONFIG, {})
TEST_ZONES = load_json(TEST_ZONES_CONFIG, {})

FINAL_TEST_OPTIONS = {
    "satellite": [
        {
            "label": "Area_04_Clip",
            "test_image_id": "satellite_area_04",
            "gt_label": "Satellite_labeling",
            "ground_truth_id": "satellite_labeling",
        }
    ],
    "aerial": [
        {
            "label": "Cliped_img",
            "test_image_id": "aerial_test_zone",
            "gt_label": "Arial_label_testing",
            "ground_truth_id": "aerial_label_testing",
        }
    ],
    "drone": [
        {
            "label": "drone_v00601",
            "test_image_id": "drone_v00601",
            "gt_label": "gt_drone",
            "ground_truth_id": "gt_drone",
        }
    ],
}


# ============================================================
# MAP RESULTS HELPERS
# ============================================================

def final_tested_folders():
    """
    Return only experiments that already have final_metrics.json.

    Map Results is visualization only. It never launches inference.
    """
    result = []

    for folder in experiment_folders():
        metrics_path = (
            folder
            / "final_test"
            / "final_metrics.json"
        )

        if metrics_path.exists():
            result.append(folder)

    return result


def source_test_option(source):
    options = FINAL_TEST_OPTIONS.get(
        str(source).lower(),
        [],
    )

    return options[0] if options else {}


def map_result_paths(folder, config, final_metrics):
    """
    Resolve the three map sources:
      - final testing raster
      - ground truth polygons
      - final prediction polygons

    Prefer paths written by the final-test script itself.
    Fall back to paths.json / known final-test outputs when needed.
    """
    paths_config = load_json(
        PATHS_CONFIG,
        {},
    )

    source = str(
        config.get(
            "source_type",
            "",
        )
    ).lower()

    option = source_test_option(source)

    # -----------------------------
    # Testing raster
    # -----------------------------
    test_image_path = (
        final_metrics.get("test_image_path")
        or final_metrics.get("test_raster")
    )

    if not test_image_path:
        test_image_id = option.get(
            "test_image_id"
        )

        test_image_path = (
            paths_config.get(
                "test_images",
                {},
            ).get(
                test_image_id
            )
            if test_image_id
            else None
        )

    # -----------------------------
    # Ground truth
    # -----------------------------
    ground_truth_path = (
        final_metrics.get("ground_truth_path")
        or final_metrics.get("ground_truth")
    )

    # Mask R-CNN stores a projected GT in the final prediction GDB.
    if not ground_truth_path:
        ground_truth_path = str(
            folder
            / "final_test"
            / "final_prediction.gdb"
            / "ground_truth_projected"
        )

    # If that does not exist, the ArcGIS existence check below will
    # fall back to the original source GT from paths.json.
    original_ground_truth_id = option.get(
        "ground_truth_id"
    )

    original_ground_truth_path = (
        paths_config.get(
            "ground_truth",
            {},
        ).get(
            original_ground_truth_id
        )
        if original_ground_truth_id
        else None
    )

    # -----------------------------
    # Final prediction
    # -----------------------------
    prediction_path = (
        final_metrics.get("final_prediction_fc")
        or final_metrics.get("final_prediction")
        or final_metrics.get("prediction_fc")
        or final_metrics.get("prediction")
    )

    if not prediction_path:
        prediction_path = str(
            folder
            / "final_test"
            / "final_prediction.gdb"
            / "final_prediction"
        )

    return {
        "source": source,
        "test_zone_label": option.get(
            "label",
            source.title() + " Test Zone",
        ),
        "ground_truth_label": option.get(
            "gt_label",
            "Ground Truth",
        ),
        "test_image_path": test_image_path,
        "ground_truth_path": ground_truth_path,
        "original_ground_truth_path": original_ground_truth_path,
        "prediction_path": prediction_path,
    }


def arcgis_exists(dataset_path):
    if not dataset_path:
        return False

    try:
        import arcpy

        return bool(
            arcpy.Exists(
                str(dataset_path)
            )
        )
    except Exception:
        return False


@st.cache_data(
    show_spinner=False,
    ttl=300,
)
def arcgis_feature_to_geojson(
    dataset_path,
    layer_name,
):
    """
    Convert an ArcGIS feature class to WGS84 GeoJSON for PyDeck.

    The conversion is temporary and does not modify the source data.
    """
    import tempfile

    import arcpy

    dataset_path = str(
        dataset_path
    )

    if not arcpy.Exists(
        dataset_path
    ):
        raise FileNotFoundError(
            f"ArcGIS feature class not found: {dataset_path}"
        )

    with tempfile.TemporaryDirectory(
        prefix="buildings_map_"
    ) as temp_folder:
        out_json = (
            Path(temp_folder)
            / "layer.geojson"
        )

        arcpy.conversion.FeaturesToJSON(
            dataset_path,
            str(out_json),
            "FORMATTED",
            "NO_Z_VALUES",
            "NO_M_VALUES",
            "GEOJSON",
            "WGS84",
        )

        payload = json.loads(
            out_json.read_text(
                encoding="utf-8"
            )
        )

    for feature in payload.get(
        "features",
        [],
    ):
        properties = feature.setdefault(
            "properties",
            {},
        )

        properties[
            "MapLayer"
        ] = layer_name

    return payload


def _walk_geojson_coordinates(value):
    """
    Yield lon/lat pairs recursively from GeoJSON coordinates.
    """
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield float(value[0]), float(value[1])
        return

    if isinstance(
        value,
        (list, tuple),
    ):
        for item in value:
            yield from _walk_geojson_coordinates(
                item
            )


def geojson_bounds(payload):
    xs = []
    ys = []

    for feature in payload.get(
        "features",
        [],
    ):
        geometry = feature.get(
            "geometry"
        ) or {}

        for x, y in _walk_geojson_coordinates(
            geometry.get(
                "coordinates",
                [],
            )
        ):
            xs.append(x)
            ys.append(y)

    if not xs:
        return None

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


@st.cache_data(
    show_spinner=False,
    ttl=300,
)
def arcgis_raster_bitmap(
    raster_path,
    max_dimension=1200,
):
    """
    Create a lightweight RGB bitmap of the real final testing raster
    and return WGS84 deck.gl corner bounds.

    This is visualization only. The source raster is never modified.
    """
    import io

    import arcpy
    import numpy as np
    from PIL import Image

    raster_path = str(
        raster_path
    )

    if not arcpy.Exists(
        raster_path
    ):
        raise FileNotFoundError(
            f"ArcGIS raster not found: {raster_path}"
        )

    raster = arcpy.Raster(
        raster_path
    )

    array = np.asarray(
        arcpy.RasterToNumPyArray(
            raster
        )
    )

    # ArcPy commonly returns:
    #   single band -> H x W
    #   multiband   -> Bands x H x W
    if array.ndim == 2:
        rgb = np.stack(
            [array, array, array],
            axis=-1,
        )

    elif array.ndim == 3:
        if array.shape[0] <= 16:
            bands = array[:3]

            if bands.shape[0] == 1:
                bands = np.repeat(
                    bands,
                    3,
                    axis=0,
                )
            elif bands.shape[0] == 2:
                bands = np.concatenate(
                    [
                        bands,
                        bands[:1],
                    ],
                    axis=0,
                )

            rgb = np.moveaxis(
                bands,
                0,
                -1,
            )
        else:
            rgb = array[..., :3]

    else:
        raise ValueError(
            f"Unsupported raster array shape: {array.shape}"
        )

    height, width = rgb.shape[:2]

    step = max(
        1,
        int(
            max(
                height,
                width,
            )
            / int(max_dimension)
        ),
    )

    rgb = rgb[
        ::step,
        ::step,
        :
    ].astype(
        "float32",
        copy=False,
    )

    # Percentile stretch gives a readable display for both 8-bit
    # imagery and higher-bit-depth remote-sensing imagery.
    stretched = np.zeros(
        rgb.shape,
        dtype="uint8",
    )

    for band_index in range(3):
        band = rgb[
            :,
            :,
            band_index,
        ]

        finite = band[
            np.isfinite(
                band
            )
        ]

        if finite.size == 0:
            continue

        low = float(
            np.percentile(
                finite,
                2,
            )
        )

        high = float(
            np.percentile(
                finite,
                98,
            )
        )

        if high <= low:
            low = float(
                np.min(
                    finite
                )
            )
            high = float(
                np.max(
                    finite
                )
            )

        if high <= low:
            continue

        normalized = (
            (band - low)
            / (high - low)
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        stretched[
            :,
            :,
            band_index,
        ] = (
            normalized
            * 255.0
        ).astype(
            "uint8"
        )

    image = Image.fromarray(
        stretched,
        mode="RGB",
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=85,
        optimize=True,
    )

    image_bytes = buffer.getvalue()

    spatial_reference = (
        raster.spatialReference
    )

    if (
        spatial_reference is None
        or spatial_reference.name
        in (
            None,
            "",
            "Unknown",
        )
    ):
        raise ValueError(
            "Testing raster has no valid spatial reference."
        )

    extent = raster.extent

    wgs84 = arcpy.SpatialReference(
        4326
    )

    source_corners = [
        (extent.XMin, extent.YMin),  # lower-left
        (extent.XMin, extent.YMax),  # upper-left
        (extent.XMax, extent.YMax),  # upper-right
        (extent.XMax, extent.YMin),  # lower-right
    ]

    bounds = []

    for x, y in source_corners:
        geometry = arcpy.PointGeometry(
            arcpy.Point(
                x,
                y,
            ),
            spatial_reference,
        )

        projected = geometry.projectAs(
            wgs84
        )

        bounds.append(
            [
                float(projected.firstPoint.X),
                float(projected.firstPoint.Y),
            ]
        )

    return {
        "image_bytes": image_bytes,
        "bounds": bounds,
    }



@st.cache_data(
    show_spinner=False,
    ttl=300,
)
def arcgis_testing_image_with_overlays(
    raster_path,
    ground_truth_path=None,
    prediction_path=None,
    show_ground_truth=True,
    show_prediction=True,
    max_dimension=1600,
):
    """
    Render the real external testing raster with Ground Truth and
    Final Prediction polygon outlines directly on top of the image.

    Ground Truth:
        thick RED outline

    Final Prediction:
        thinner BLUE outline

    The source raster and feature classes are never modified.
    """
    import io

    import arcpy
    import numpy as np
    from PIL import Image, ImageDraw

    raster_path = str(raster_path)

    if not arcpy.Exists(raster_path):
        raise FileNotFoundError(
            f"Testing raster not found: {raster_path}"
        )

    raster = arcpy.Raster(raster_path)

    raster_sr = raster.spatialReference

    if (
        raster_sr is None
        or raster_sr.name in (None, "", "Unknown")
    ):
        raise ValueError(
            "Testing raster has no valid spatial reference."
        )

    extent = raster.extent

    array = np.asarray(
        arcpy.RasterToNumPyArray(raster)
    )

    # ----------------------------------------------------------
    # Convert ArcGIS raster array to RGB.
    # ----------------------------------------------------------

    if array.ndim == 2:
        rgb = np.stack(
            [array, array, array],
            axis=-1,
        )

    elif array.ndim == 3:
        # ArcPy commonly returns Bands x H x W.
        if array.shape[0] <= 16:
            bands = array[:3]

            if bands.shape[0] == 1:
                bands = np.repeat(
                    bands,
                    3,
                    axis=0,
                )

            elif bands.shape[0] == 2:
                bands = np.concatenate(
                    [
                        bands,
                        bands[:1],
                    ],
                    axis=0,
                )

            rgb = np.moveaxis(
                bands,
                0,
                -1,
            )

        else:
            rgb = array[..., :3]

    else:
        raise ValueError(
            f"Unsupported testing raster shape: {array.shape}"
        )

    original_height, original_width = rgb.shape[:2]

    step = max(
        1,
        int(
            max(
                original_height,
                original_width,
            )
            / int(max_dimension)
        ),
    )

    rgb = rgb[
        ::step,
        ::step,
        :
    ].astype(
        "float32",
        copy=False,
    )

    # ----------------------------------------------------------
    # Percentile stretch for readable remote-sensing display.
    # ----------------------------------------------------------

    stretched = np.zeros(
        rgb.shape,
        dtype="uint8",
    )

    for band_index in range(3):
        band = rgb[
            :,
            :,
            band_index,
        ]

        finite = band[
            np.isfinite(band)
        ]

        if finite.size == 0:
            continue

        low = float(
            np.percentile(
                finite,
                2,
            )
        )

        high = float(
            np.percentile(
                finite,
                98,
            )
        )

        if high <= low:
            low = float(
                np.min(finite)
            )
            high = float(
                np.max(finite)
            )

        if high <= low:
            continue

        normalized = (
            (band - low)
            / (high - low)
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        stretched[
            :,
            :,
            band_index,
        ] = (
            normalized * 255.0
        ).astype("uint8")

    base = Image.fromarray(
        stretched,
        mode="RGB",
    ).convert("RGBA")

    preview_width, preview_height = base.size

    x_span = float(
        extent.XMax - extent.XMin
    )

    y_span = float(
        extent.YMax - extent.YMin
    )

    if x_span <= 0 or y_span <= 0:
        raise ValueError(
            "Testing raster has an invalid extent."
        )

    def world_to_pixel(x, y):
        px = (
            (float(x) - float(extent.XMin))
            / x_span
            * preview_width
        )

        # Image Y increases downward, map Y increases upward.
        py = (
            (float(extent.YMax) - float(y))
            / y_span
            * preview_height
        )

        return (
            int(round(px)),
            int(round(py)),
        )

    def draw_feature_class(
        image,
        feature_class,
        line_color,
        line_width,
    ):
        if (
            not feature_class
            or not arcpy.Exists(
                str(feature_class)
            )
        ):
            return image, 0

        feature_class = str(
            feature_class
        )

        desc = arcpy.Describe(
            feature_class
        )

        feature_sr = getattr(
            desc,
            "spatialReference",
            None,
        )

        overlay = Image.new(
            "RGBA",
            image.size,
            (0, 0, 0, 0),
        )

        drawer = ImageDraw.Draw(
            overlay
        )

        feature_count = 0

        with arcpy.da.SearchCursor(
            feature_class,
            ["SHAPE@"],
        ) as cursor:

            for row in cursor:
                geometry = row[0]

                if geometry is None:
                    continue

                try:
                    if (
                        feature_sr is not None
                        and feature_sr.name
                        not in (
                            None,
                            "",
                            "Unknown",
                        )
                        and feature_sr.factoryCode
                        != raster_sr.factoryCode
                    ):
                        geometry = geometry.projectAs(
                            raster_sr
                        )
                except Exception:
                    # If the factory codes are unavailable but both
                    # datasets are already aligned, continue as-is.
                    pass

                geometry_extent = getattr(
                    geometry,
                    "extent",
                    None,
                )

                if geometry_extent is not None:
                    if (
                        geometry_extent.XMax < extent.XMin
                        or geometry_extent.XMin > extent.XMax
                        or geometry_extent.YMax < extent.YMin
                        or geometry_extent.YMin > extent.YMax
                    ):
                        continue

                feature_count += 1

                # ArcPy polygon parts may contain None separators.
                # Draw every ring as an outline so the testing image
                # remains visible and holes are not incorrectly filled.
                for part in geometry:
                    ring = []

                    for point in part:
                        if point is None:
                            if len(ring) >= 2:
                                drawer.line(
                                    ring + [ring[0]],
                                    fill=line_color,
                                    width=line_width,
                                    joint="curve",
                                )

                            ring = []
                            continue

                        ring.append(
                            world_to_pixel(
                                point.X,
                                point.Y,
                            )
                        )

                    if len(ring) >= 2:
                        drawer.line(
                            ring + [ring[0]],
                            fill=line_color,
                            width=line_width,
                            joint="curve",
                        )

        return (
            Image.alpha_composite(
                image,
                overlay,
            ),
            feature_count,
        )

    gt_count = 0
    prediction_count = 0

    # Draw GT first with a thicker line. Prediction is then drawn
    # with a thinner line so both remain visible when they overlap.
    if show_ground_truth:
        base, gt_count = draw_feature_class(
            base,
            ground_truth_path,
            (220, 38, 38, 255),
            5,
        )

    if show_prediction:
        base, prediction_count = draw_feature_class(
            base,
            prediction_path,
            (37, 99, 235, 255),
            2,
        )

    # ----------------------------------------------------------
    # Add a small readable legend directly on the image.
    # ----------------------------------------------------------

    legend = Image.new(
        "RGBA",
        base.size,
        (0, 0, 0, 0),
    )

    legend_draw = ImageDraw.Draw(
        legend
    )

    legend_x = 18
    legend_y = 18

    box_width = 245
    box_height = 34

    if (
        show_ground_truth
        and show_prediction
    ):
        box_height = 62

    legend_draw.rounded_rectangle(
        [
            legend_x,
            legend_y,
            legend_x + box_width,
            legend_y + box_height,
        ],
        radius=8,
        fill=(15, 23, 42, 190),
    )

    current_y = legend_y + 10

    if show_ground_truth:
        legend_draw.line(
            [
                (legend_x + 12, current_y + 7),
                (legend_x + 45, current_y + 7),
            ],
            fill=(220, 38, 38, 255),
            width=5,
        )

        legend_draw.text(
            (
                legend_x + 56,
                current_y,
            ),
            "Ground Truth",
            fill=(255, 255, 255, 255),
        )

        current_y += 28

    if show_prediction:
        legend_draw.line(
            [
                (legend_x + 12, current_y + 7),
                (legend_x + 45, current_y + 7),
            ],
            fill=(37, 99, 235, 255),
            width=3,
        )

        legend_draw.text(
            (
                legend_x + 56,
                current_y,
            ),
            "Final Prediction",
            fill=(255, 255, 255, 255),
        )

    base = Image.alpha_composite(
        base,
        legend,
    ).convert("RGB")

    buffer = io.BytesIO()

    base.save(
        buffer,
        format="JPEG",
        quality=92,
        optimize=True,
    )

    return {
        "image_bytes": buffer.getvalue(),
        "ground_truth_count_drawn": gt_count,
        "prediction_count_drawn": prediction_count,
        "preview_width": preview_width,
        "preview_height": preview_height,
    }


def bitmap_bounds(bitmap):
    points = bitmap.get(
        "bounds",
        [],
    )

    if not points:
        return None

    xs = [
        float(point[0])
        for point in points
    ]

    ys = [
        float(point[1])
        for point in points
    ]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def merge_bounds(bounds_list):
    valid = [
        bounds
        for bounds in bounds_list
        if bounds is not None
    ]

    if not valid:
        return None

    return (
        min(item[0] for item in valid),
        min(item[1] for item in valid),
        max(item[2] for item in valid),
        max(item[3] for item in valid),
    )


def view_state_from_bounds(bounds):
    """
    Build a reasonable initial map view for the selected test area.
    """
    import math

    if bounds is None:
        return {
            "latitude": 33.85,
            "longitude": 35.86,
            "zoom": 8.0,
        }

    min_x, min_y, max_x, max_y = bounds

    longitude = (
        min_x + max_x
    ) / 2.0

    latitude = (
        min_y + max_y
    ) / 2.0

    span = max(
        abs(max_x - min_x),
        abs(max_y - min_y),
        1e-6,
    )

    zoom = (
        math.log2(
            360.0 / span
        )
        - 1.6
    )

    zoom = max(
        3.0,
        min(
            19.0,
            zoom,
        ),
    )

    return {
        "latitude": latitude,
        "longitude": longitude,
        "zoom": zoom,
    }


def build_training_config(
    experiment_id,
    model_key,
    backbone,
    source,
    tile_size,
    training_percent,
    validation_percent,
    seed,
    augmentation,
    optuna,
    master_optuna,
):
    config = dict(BASE_CONFIG)

    config.update(
        {
            "experiment_id": safe_token(experiment_id),
            "approach": "buildings",
            "model_type": model_key,
            "backbone": backbone,
            "source_type": source,
            "tile_size": int(tile_size),
            "dataset_type": dataset_type_for_model(model_key),

            # Final test stays separate.
            "test_image_id": None,
            "ground_truth_id": None,

            "data_split": {
                "train_percent": float(training_percent),
                "validation_percent": float(validation_percent),
                "seed": int(seed),
            },

            "use_augmentation": bool(augmentation),
            "use_optuna": bool(optuna),
            "use_master_optuna": bool(master_optuna),
            "use_threshold_search": False,
            "use_postprocessing": False,
            "use_normalization": bool(
                BASE_CONFIG.get("use_normalization", False)
            ),
            "use_resampling": bool(
                BASE_CONFIG.get("use_resampling", False)
            ),
            "use_error_analysis": bool(
                BASE_CONFIG.get("use_error_analysis", False)
            ),
            "use_reporting": True,
        }
    )

    return config


# ============================================================
# BACKEND
# ============================================================

def backend_command(
    config_path,
    stage,
    epochs=40,
    trials=10,
    trial_epochs=12,
    final_epochs=40,
):
    return [
        sys.executable,
        str(RUN_BUILDINGS),
        "--config",
        str(config_path),
        "--stage",
        stage,
        "--epochs",
        str(int(epochs)),
        "--trials",
        str(int(trials)),
        "--trial-epochs",
        str(int(trial_epochs)),
        "--final-epochs",
        str(int(final_epochs)),
    ]


def run_backend(command, title, log_path):
    st.markdown(f"**{title}**")

    status = st.empty()
    output = st.empty()

    status.info("Running...")

    lines = []

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert process.stdout is not None

        for line in process.stdout:
            lines.append(line.rstrip("\n"))
            log_file.write(line)
            log_file.flush()

            output.code(
                "\n".join(lines[-120:]),
                language="text",
            )

        code = process.wait()

    if code == 0:
        status.success("Completed.")
    else:
        status.error(f"Failed with exit code {code}.")

    return code


# ============================================================
# PAGE
# ============================================================

def render() -> None:

    _html(
        BUILDING_ROAD_MATCH_CSS
    )

    with st.container(
        key="buildings_page"
    ):

        render_header(
            title="Building Extraction",
            subtitle=(
                "Approach 1 · Complete Building extraction workflow with "
                "training, validation optimization, independent final testing "
                "and GIS visualization."
            ),
            eyebrow="APPROACH 1",
        )

        # ============================================================
        # CHECKS
        # ============================================================

        if not RUN_BUILDINGS.exists():
            st.error(f"Backend controller not found: {RUN_BUILDINGS}")
            st.stop()

        if not DATASETS:
            st.error("No datasets were found in datasets.json.")
            st.stop()


        # ============================================================
        # 1. SETUP & TRAIN
        # ============================================================

        st.markdown(
            """
            <div class="section-header">
                <div class="section-title">1. Setup & Train</div>
                <div class="section-subtitle">
                    Select the model, dataset, split, and optional training settings.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            selected_model_name = st.selectbox(
                "Model",
                [model_name(key) for key in MODEL_REGISTRY],
            )

        model_key = model_key_from_name(selected_model_name)
        model_spec = get_model_spec(model_key)
        dataset_type = dataset_type_for_model(model_key)

        source_options = sources_for(dataset_type)

        with c2:
            source = st.selectbox(
                "Source",
                source_options,
                index=(
                    source_options.index("satellite")
                    if "satellite" in source_options
                    else 0
                ),
            )

        tile_options = tiles_for(source, dataset_type)

        with c3:
            tile_size = st.selectbox(
                "Tile Size",
                tile_options,
            )

        backbones = backbones_for(model_key)
        default_backbone = get_model_spec(model_key).default_backbone

        with c4:
            backbone = st.selectbox(
                "Backbone",
                backbones,
                index=(
                    backbones.index(default_backbone)
                    if default_backbone in backbones
                    else 0
                ),
            )

        if model_key == "sam_lora":
            st.info(
                "SAM-LoRA baseline is enabled. "
                "Backbone options are read from sam_lora.json. "
                "Training uses the shared semantic workflow with Auto-LR. "
                "Optuna and Master Optuna remain disabled until their "
                "SAM-LoRA integration is implemented."
            )

        dataset = dataset_for(
            source,
            dataset_type,
            tile_size,
        )

        if dataset:
            info = dataset_file_info(
                str(dataset.get("path", ""))
            )

            st.markdown(
                f"""
                <div class="data-line">
                    Dataset: {dataset["dataset_id"]}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    Images: {info["images"] if info["images"] is not None else "—"}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    Labels: {info["labels"] if info["labels"] is not None else "—"}
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    Paired: {info["paired"] if info["paired"] is not None else "—"}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------
        # SPLIT CONTROL
        # ------------------------------------------------------------

        st.markdown("**Data Split**")

        s1, s2 = st.columns(2)

        with s1:
            training_percent = st.number_input(
                "Training %",
                min_value=60,
                max_value=90,
                value=80,
                step=5,
            )

        validation_percent = 100 - int(training_percent)

        with s2:
            st.number_input(
                "Validation %",
                min_value=10,
                max_value=40,
                value=validation_percent,
                disabled=True,
            )

        st.markdown(
            f"""
            <div class="split-line">
                Split: {training_percent}% Training / {validation_percent}% Validation
                &nbsp;&nbsp;|&nbsp;&nbsp;
                External Test: Separate
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------
        # OPTIONAL SETTINGS
        # ------------------------------------------------------------

        experiment_name = st.text_input(
            "Experiment Name",
            value=f"{model_key}_{tile_size}_{source}",
        )

        st.markdown("**Optional**")

        o1, o2, o3 = st.columns(3)

        with o1:
            use_augmentation = st.checkbox(
                "Use Augmentation",
                value=False,
                key="train_use_augmentation",
                disabled=(
                    not model_spec.supports_augmentation
                ),
            )

        with o2:
            use_optuna = st.checkbox(
                "Use Optuna",
                value=False,
                key="train_use_optuna",
                disabled=(
                    not model_spec.supports_optuna
                ),
                help=(
                    None
                    if model_spec.supports_optuna
                    else (
                        "Optuna is not enabled for this model yet. "
                        "Use the normal baseline workflow first."
                    )
                ),
            )

        with o3:
            use_master_optuna = st.checkbox(
                "Use Master Optuna",
                value=False,
                key="train_use_master_optuna",
                disabled=(
                    not model_spec.supports_optuna
                ),
                help=(
                    None
                    if model_spec.supports_optuna
                    else (
                        "Master Optuna is not enabled for this model yet."
                    )
                ),
            )

        # Defensive capability guard. The checkboxes above are disabled for
        # unsupported models (currently SAM-LoRA Optuna), but keep the runtime
        # configuration safe even if Streamlit session state contains an old value.
        if not model_spec.supports_optuna:
            use_optuna = False
            use_master_optuna = False

        optimization_conflict = bool(
            use_optuna and use_master_optuna
        )

        if optimization_conflict:
            st.error(
                "Choose either Optuna or Master Optuna, not both in the same experiment."
            )

        effective_augmentation = use_augmentation

        if use_master_optuna:
            effective_augmentation = False
            st.info(
                "Master Optuna keeps Model, Source and Split fixed. "
                "It searches Tile Size, Backbone and supported training parameters "
                "from master_optuna_search_space.json. Every trial keeps its best "
                "Validation-IoU checkpoint; the global winner is used directly with "
                "NO second training."
            )

        elif model_key == "maskrcnn" and use_augmentation and use_optuna:
            st.info(
                "Mask R-CNN Optuna uses clean trials. "
                "Run augmentation as a separate training experiment for comparison."
            )
            effective_augmentation = False

        if use_master_optuna:
            p1, p2 = st.columns(2)

            with p1:
                trials = st.number_input(
                    "Master Trials",
                    min_value=1,
                    value=30,
                    step=1,
                )

            with p2:
                trial_epochs = st.number_input(
                    "Epochs / Master Trial",
                    min_value=1,
                    value=40,
                    step=1,
                    help=(
                        "Each trial is a complete candidate. Its best Validation-IoU "
                        "checkpoint can become the final Master Optuna winner."
                    ),
                )

            # Master Optuna does NOT retrain the winner. This value is kept only
            # because backend_command is shared with the focused Optuna workflow.
            final_epochs = 40
            epochs = 40

            st.caption(
                "Winner policy: highest Validation IoU across all trials → keep that "
                "exact best_model.pth → no second training."
            )

        elif use_optuna:
            p1, p2, p3 = st.columns(3)

            with p1:
                trials = st.number_input(
                    "Trials",
                    min_value=1,
                    value=10,
                    step=1,
                )

            with p2:
                trial_epochs = st.number_input(
                    "Epochs / Trial",
                    min_value=1,
                    value=12,
                    step=1,
                )

            with p3:
                final_epochs = st.number_input(
                    "Final Epochs",
                    min_value=1,
                    value=40,
                    step=1,
                )

            epochs = 40

        else:
            epochs = st.number_input(
                "Epochs",
                min_value=1,
                value=40,
                step=1,
            )

            trials = 10
            trial_epochs = 12
            final_epochs = 40

        config = build_training_config(
            experiment_id=experiment_name,
            model_key=model_key,
            backbone=backbone,
            source=source,
            tile_size=tile_size,
            training_percent=training_percent,
            validation_percent=validation_percent,
            seed=FIXED_SPLIT_SEED,
            augmentation=effective_augmentation,
            optuna=use_optuna,
            master_optuna=use_master_optuna,
        )

        button_text = (
            "Run Master Optuna"
            if use_master_optuna
            else "Run Optuna"
            if use_optuna
            else "Train Model"
        )

        if st.button(
            button_text,
            type="primary",
            width="stretch",
            disabled=optimization_conflict,
        ):
            config_path = save_json(
                config,
                UI_CONFIG_ROOT
                / f"{safe_token(experiment_name)}_train_{now_token()}.json",
            )

            stage = (
                "master_optuna"
                if use_master_optuna
                else "optuna"
                if use_optuna
                else "augmentation"
                if effective_augmentation
                else "standard"
            )

            run_backend(
                backend_command(
                    config_path,
                    stage,
                    epochs=epochs,
                    trials=trials,
                    trial_epochs=trial_epochs,
                    final_epochs=final_epochs,
                ),
                button_text,
                UI_LOG_ROOT
                / f"{safe_token(experiment_name)}_{stage}_{now_token()}.log",
            )


        # ============================================================
        # 2. VALIDATION OPTIONS
        # ============================================================

        st.markdown(
            """
            <div class="section-header">
                <div class="section-title">2. Validation Options</div>
                <div class="section-subtitle">
                    Optional. Apply these only to an already trained experiment.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        folders = experiment_folders()

        if not folders:
            st.info("No trained experiment is available yet.")
        else:
            selected_validation = st.selectbox(
                "Experiment",
                [folder.name for folder in folders],
                key="validation_experiment",
            )

            validation_folder = (
                EXPERIMENTS_ROOT / selected_validation
            )

            v1, v2 = st.columns(2)

            with v1:
                optimize_threshold = st.checkbox(
                    "Optimize Threshold",
                    value=False,
                    key=f"validation_optimize_threshold_{safe_token(selected_validation)}",
                )

            with v2:
                use_postprocessing = st.checkbox(
                    "Use Post-processing",
                    value=False,
                    key=f"validation_use_postprocessing_{safe_token(selected_validation)}",
                )

            if st.button(
                "Apply Validation Options",
                type="primary",
                width="stretch",
            ):
                if not optimize_threshold and not use_postprocessing:
                    st.info("Select at least one option.")
                else:
                    existing_config = experiment_config(
                        validation_folder
                    )

                    if not existing_config:
                        st.error(
                            "Could not load this experiment configuration."
                        )
                    else:
                        threshold_exists = (
                            validation_folder
                            / "validation_threshold_summary.json"
                        ).exists()

                        runtime = dict(existing_config)

                        # Bind validation to the exact experiment folder selected
                        # in the shared UI. This prevents stale config metadata from
                        # redirecting validation to another experiment directory.
                        runtime["experiment_id"] = selected_validation

                        runtime["use_threshold_search"] = bool(
                            optimize_threshold
                            or use_postprocessing
                            or threshold_exists
                        )

                        runtime["use_postprocessing"] = bool(
                            use_postprocessing
                        )

                        runtime_path = save_json(
                            runtime,
                            UI_CONFIG_ROOT
                            / f"{safe_token(selected_validation)}_validation_{now_token()}.json",
                        )

                        failed = False

                        if optimize_threshold or (
                            use_postprocessing
                            and not threshold_exists
                        ):
                            code = run_backend(
                                backend_command(
                                    runtime_path,
                                    "threshold",
                                ),
                                "Threshold Optimization",
                                UI_LOG_ROOT
                                / f"{safe_token(selected_validation)}_threshold_{now_token()}.log",
                            )

                            failed = code != 0

                        if use_postprocessing and not failed:
                            run_backend(
                                backend_command(
                                    runtime_path,
                                    "post",
                                ),
                                "Post-processing",
                                UI_LOG_ROOT
                                / f"{safe_token(selected_validation)}_post_{now_token()}.log",
                            )


        # ============================================================
        # 3. RESULTS
        # ============================================================

        st.markdown(
            """
            <div class="section-header">
                <div class="section-title">3. Results</div>
                <div class="section-subtitle">
                    One experiment per row with validation results and external testing-zone results.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        results = experiment_result_rows()

        if results.empty:
            st.info(
                "No results are available yet."
            )

        else:
            display = results.copy()

            # --------------------------------------------------------
            # Arrow-safe metadata
            # --------------------------------------------------------

            text_columns = (
                "Experiment",
                "Model",
                "Backbone",
                "Source",
                "Tile",
                "Split",
                "Aug",
                "Optuna",
                "Master Optuna",
                "Post-processing",
                "Test Zone",
                "Status",
            )

            for column in text_columns:
                if column in display.columns:
                    display[column] = (
                        display[column]
                        .fillna("—")
                        .astype(str)
                        .replace(
                            {
                                "": "—",
                                "nan": "—",
                                "None": "—",
                            }
                        )
                    )

            numeric_columns = (
                "Validation IoU",
                "Optuna Val IoU",
                "Threshold",
                "Test IoU",
                "Precision",
                "Recall",
                "F1",
            )

            for column in numeric_columns:
                if column in display.columns:
                    display[column] = pd.to_numeric(
                        display[column],
                        errors="coerce",
                    ).round(2)

            # --------------------------------------------------------
            # Main professional table
            # --------------------------------------------------------

            table_columns = [
                "Experiment",
                "Model",
                "Backbone",
                "Source",
                "Tile",
                "Split",
                "Aug",
                "Optuna",
                "Master Optuna",
                "Validation IoU",
                "Optuna Val IoU",
                "Threshold",
                "Post-processing",
                "Test Zone",
                "Test IoU",
                "Precision",
                "Recall",
                "F1",
                "Status",
            ]

            st.dataframe(
                display[
                    table_columns
                ],
                width="stretch",
                hide_index=True,
                height=390,
                column_config={
                    "Experiment":
                        st.column_config.TextColumn(
                            "Experiment",
                            width="large",
                        ),

                    "Model":
                        st.column_config.TextColumn(
                            "Model",
                            width="small",
                        ),

                    "Backbone":
                        st.column_config.TextColumn(
                            "Backbone",
                            width="medium",
                        ),

                    "Source":
                        st.column_config.TextColumn(
                            "Source",
                            width="small",
                        ),

                    "Tile":
                        st.column_config.TextColumn(
                            "Tile",
                            width="small",
                        ),

                    "Split":
                        st.column_config.TextColumn(
                            "Split",
                            width="small",
                        ),

                    "Aug":
                        st.column_config.TextColumn(
                            "Aug",
                            width="small",
                        ),

                    "Optuna":
                        st.column_config.TextColumn(
                            "Optuna",
                            width="small",
                        ),

                    "Master Optuna":
                        st.column_config.TextColumn(
                            "Master",
                            help="Model-level Master Optuna search using the model-specific JSON search space.",
                            width="small",
                        ),

                    "Validation IoU":
                        st.column_config.NumberColumn(
                            "Validation IoU %",
                            help="Best IoU measured on the validation split during model training.",
                            format="%.2f",
                            width="small",
                        ),

                    "Optuna Val IoU":
                        st.column_config.NumberColumn(
                            "Optuna IoU %",
                            format="%.2f",
                            width="small",
                        ),

                    "Threshold":
                        st.column_config.NumberColumn(
                            "Threshold",
                            format="%.2f",
                            width="small",
                        ),

                    "Post-processing":
                        st.column_config.TextColumn(
                            "Post",
                            help="Whether validation-selected post-processing is available/used.",
                            width="small",
                        ),

                    "Test Zone":
                        st.column_config.TextColumn(
                            "Testing Zone",
                            help="External source-specific testing zone: Satellite, Aerial, or Drone.",
                            width="medium",
                        ),

                    "Test IoU":
                        st.column_config.NumberColumn(
                            "Test IoU %",
                            help="Final IoU measured on the external source-specific testing zone.",
                            format="%.2f",
                            width="small",
                        ),

                    "Precision":
                        st.column_config.NumberColumn(
                            "Precision %",
                            format="%.2f",
                            width="small",
                        ),

                    "Recall":
                        st.column_config.NumberColumn(
                            "Recall %",
                            format="%.2f",
                            width="small",
                        ),

                    "F1":
                        st.column_config.NumberColumn(
                            "F1 %",
                            format="%.2f",
                            width="small",
                        ),

                    "Status":
                        st.column_config.TextColumn(
                            "Status",
                            width="medium",
                        ),
                },
            )

            st.caption(
                "Validation IoU comes from the validation split. "
                "Test IoU is the final IoU from the external Satellite, Aerial, or Drone testing zone. "
                "Blank cells mean that stage has not been run."
            )

            # --------------------------------------------------------
            # Details
            # --------------------------------------------------------

            selected_details = st.selectbox(
                "View Experiment Details",
                results[
                    "Experiment"
                ].tolist(),
                key="details_experiment",
            )

            details_folder = (
                EXPERIMENTS_ROOT
                / selected_details
            )

            details_config = experiment_config(
                details_folder
            )

            details_split = split_info(
                details_folder,
                details_config,
            )

            details_training = first_json(
                details_folder,
                [
                    "training_summary.json",
                    "maskrcnn_training_summary.json",
                ],
            )

            details_optuna = first_json(
                details_folder,
                [
                    "optuna_selection_summary.json",
                    "run_optuna_summary.json",
                ],
            )

            details_master_optuna = first_json(
                details_folder,
                [
                    "master_optuna_selection_summary.json",
                ],
            )

            details_threshold = first_json(
                details_folder,
                [
                    "validation_threshold_summary.json",
                ],
            )

            details_post = first_json(
                details_folder,
                [
                    "validation_postprocess_summary.json",
                ],
            )

            details_final = first_json(
                details_folder
                / "final_test",
                [
                    "final_metrics.json",
                ],
            )

            with st.expander(
                "Experiment Details",
                expanded=False,
            ):
                detail_meta = infer_legacy_metadata(
                    selected_details,
                    details_config,
                )

                st.write(
                    {
                        "Experiment":
                            selected_details,

                        "Model":
                            detail_meta[
                                "model"
                            ],

                        "Backbone":
                            detail_meta[
                                "backbone"
                            ],

                        "Source":
                            detail_meta[
                                "source"
                            ],

                        "Tile Size":
                            detail_meta[
                                "tile"
                            ],

                        "Training %":
                            details_split.get(
                                "train_percent"
                            ),

                        "Validation %":
                            details_split.get(
                                "validation_percent"
                            ),

                        "Training Samples":
                            details_split.get(
                                "train_count"
                            ),

                        "Validation Samples":
                            details_split.get(
                                "validation_count"
                            ),

                        "Selected Samples":
                            details_split.get(
                                "selected_count"
                            ),

                        "Augmentation":
                            details_config.get(
                                "use_augmentation",
                                "augmented"
                                in selected_details.lower(),
                            ),

                        "Optuna":
                            bool(
                                details_optuna
                                or details_config.get(
                                    "use_optuna",
                                    False,
                                )
                            ),

                        "Master Optuna":
                            bool(
                                details_master_optuna
                                or details_config.get(
                                    "use_master_optuna",
                                    False,
                                )
                            ),
                    }
                )

                if details_training:
                    st.markdown(
                        "**Training**"
                    )

                    st.write(
                        {
                            "Best Validation IoU":
                                percentage(
                                    details_training,
                                    "best_validation_iou_percent",
                                    "best_val_iou",
                                    "best_validation_iou",
                                    "validation_iou",
                                )
                        }
                    )

                if details_optuna:
                    st.markdown(
                        "**Optuna**"
                    )

                    st.write(
                        {
                            "Best Trial":
                                details_optuna.get(
                                    "best_trial_number",
                                    details_optuna.get(
                                        "best_trial",
                                    ),
                                ),

                            "Best Validation IoU":
                                percentage(
                                    details_optuna,
                                    "best_value_percent",
                                    "best_validation_iou_percent",
                                    "best_validation_iou",
                                    "best_value",
                                ),

                            "Best Parameters":
                                details_optuna.get(
                                    "best_params",
                                    {},
                                ),
                        }
                    )

                if details_master_optuna:
                    st.markdown(
                        "**Master Optuna**"
                    )

                    st.write(
                        {
                            "Best Trial":
                                details_master_optuna.get(
                                    "best_trial"
                                ),

                            "Validation IoU":
                                percentage(
                                    details_master_optuna,
                                    "best_validation_iou_percent",
                                    "best_validation_iou",
                                ),

                            "Best Epoch":
                                details_master_optuna.get(
                                    "best_epoch"
                                ),

                            "Epochs / Trial":
                                details_master_optuna.get(
                                    "trial_epochs"
                                ),

                            "Second Training":
                                "NO",

                            "Best Configuration":
                                details_master_optuna.get(
                                    "best_configuration",
                                    details_master_optuna.get(
                                        "best_params",
                                        {},
                                    ),
                                ),
                        }
                    )

                if details_threshold:
                    st.markdown(
                        "**Threshold**"
                    )

                    st.write(
                        {
                            "Threshold":
                                details_threshold.get(
                                    "best_threshold",
                                    details_threshold.get(
                                        "score_threshold",
                                    ),
                                ),

                            "Validation IoU":
                                percentage(
                                    details_threshold,
                                    "best_validation_iou_percent",
                                    "validation_iou_percent",
                                    "iou_percent",
                                ),

                            "F1":
                                percentage(
                                    details_threshold,
                                    "best_f1_percent",
                                    "f1_percent",
                                ),
                        }
                    )

                if details_post:
                    st.markdown(
                        "**Post-processing**"
                    )

                    st.write(
                        {
                            "Minimum Area m²":
                                details_post.get(
                                    "best_min_area_m2",
                                    details_post.get(
                                        "min_area_m2",
                                    ),
                                ),

                            "Validation IoU":
                                percentage(
                                    details_post,
                                    "best_validation_iou_percent",
                                    "validation_iou_percent",
                                    "iou_percent",
                                ),

                            "F1":
                                percentage(
                                    details_post,
                                    "best_f1_percent",
                                    "f1_percent",
                                ),
                        }
                    )

                if details_final:
                    st.markdown(
                        "**Final Test**"
                    )

                    st.write(
                        {
                            "Final IoU":
                                percentage(
                                    details_final,
                                    "iou_percent",
                                    "iou",
                                ),

                            "Precision":
                                percentage(
                                    details_final,
                                    "precision_percent",
                                    "precision",
                                ),

                            "Recall":
                                percentage(
                                    details_final,
                                    "recall_percent",
                                    "recall",
                                ),

                            "F1":
                                percentage(
                                    details_final,
                                    "f1_percent",
                                    "f1",
                                ),
                        }
                    )


        # ============================================================
        # 4. FINAL TEST
        # ============================================================

        st.markdown(
            """
            <div class="section-header">
                <div class="section-title">4. Final Test</div>
                <div class="section-subtitle">
                    Source-specific external test. Threshold and post-processing settings
                    are taken automatically from validation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        folders = [
            folder
            for folder in experiment_folders()
            if (
                folder
                / "best_model.pth"
            ).exists()
        ]

        if not folders:
            st.info(
                "No final-test-ready trained experiment is available. "
                "A completed experiment must contain best_model.pth."
            )
        else:
            selected_final = st.selectbox(
                "Experiment",
                [folder.name for folder in folders],
                key="final_experiment",
            )

            final_folder = EXPERIMENTS_ROOT / selected_final
            final_config = experiment_config(final_folder)
            final_checkpoint = final_folder / "best_model.pth"

            if not final_config:
                st.error("Could not load this experiment configuration.")
            else:
                source = str(final_config.get("source_type", "")).lower()
                source_options = FINAL_TEST_OPTIONS.get(source, [])

                if not source_options:
                    st.error(f"No final testing zone is configured for source: {source}")
                else:
                    selected_zone_label = st.selectbox(
                        "Testing Zone",
                        [option["label"] for option in source_options],
                        key=f"final_test_zone_{safe_token(selected_final)}",
                    )

                    selected_zone = next(
                        option
                        for option in source_options
                        if option["label"] == selected_zone_label
                    )

                    test_image_id = selected_zone["test_image_id"]
                    ground_truth_id = selected_zone["ground_truth_id"]
                    ground_truth_label = selected_zone["gt_label"]

                    st.markdown(
                        f"""
                        <div class="data-line">
                            Source: <b>{source.title()}</b>
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            Testing Zone: <b>{selected_zone_label}</b>
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            Ground Truth: <b>{ground_truth_label}</b>
                            <br>
                            Checkpoint: <b>{final_checkpoint}</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    threshold_summary = load_json(
                        final_folder / "validation_threshold_summary.json",
                        {},
                    )
                    post_summary = load_json(
                        final_folder / "validation_postprocess_summary.json",
                        {},
                    )

                    threshold_value = 0.50

                    if threshold_summary:
                        threshold_value = threshold_summary.get(
                            "best_threshold",
                            threshold_summary.get("score_threshold", 0.50),
                        )

                    if post_summary:
                        threshold_value = post_summary.get(
                            "best_threshold",
                            threshold_value,
                        )

                    post_enabled = bool(post_summary)

                    st.markdown(
                        f"""
                        <div class="split-line">
                            Selected Threshold: <b>{float(threshold_value):.2f}</b>
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            Post-processing: <b>{"ON" if post_enabled else "OFF"}</b>
                            &nbsp;&nbsp;|&nbsp;&nbsp;
                            Settings source: <b>Validation</b>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    receipt = final_folder / "final_test" / "final_test_receipt.json"

                    if receipt.exists():
                        st.warning("Final test already exists for this experiment.")

                    confirmed = st.checkbox(
                        "I confirm that model selection is complete.",
                        value=False,
                        key=f"final_model_selection_confirmed_{safe_token(selected_final)}",
                        disabled=receipt.exists(),
                    )

                    if st.button(
                        "Run Final Test",
                        type="primary",
                        width="stretch",
                        disabled=(not confirmed or receipt.exists()),
                    ):
                        runtime = dict(final_config)

                        # CRITICAL: Final Test must use the exact trained experiment
                        # selected above. Do not trust a stale experiment_id stored
                        # inside an older config file.
                        runtime["experiment_id"] = selected_final

                        runtime["test_image_id"] = test_image_id
                        runtime["ground_truth_id"] = ground_truth_id
                        runtime["use_threshold_search"] = bool(
                            threshold_summary or post_summary
                        )
                        runtime["use_postprocessing"] = bool(post_summary)

                        runtime_path = save_json(
                            runtime,
                            UI_CONFIG_ROOT
                            / f"{safe_token(selected_final)}_final_{now_token()}.json",
                        )

                        run_backend(
                            backend_command(runtime_path, "final"),
                            "Final Test",
                            UI_LOG_ROOT
                            / f"{safe_token(selected_final)}_final_{now_token()}.log",
                        )


        # ============================================================
        # 5. MAP RESULTS
        # ============================================================

        st.markdown(
            """
            <div class="section-header">
                <div class="section-title">5. Map Results</div>
                <div class="section-subtitle">
                    Ground Truth and Final Prediction are drawn directly on the real external testing image.
                    This section only visualizes completed Final Test outputs.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        map_folders = final_tested_folders()
        completed_map_names = {
            folder.name
            for folder in map_folders
        }

        selected_map = st.session_state.get(
            "final_experiment"
        )

        if not selected_map:
            st.info(
                "Select a trained experiment in Final Test first. "
                "Map Results follows that exact selection automatically."
            )

        elif selected_map not in completed_map_names:
            st.info(
                f"No completed Final Test exists yet for '{selected_map}'. "
                "Run that selected Final Test first; its map will then appear here."
            )

        else:
            map_folder = (
                EXPERIMENTS_ROOT
                / selected_map
            )

            map_config = experiment_config(
                map_folder
            )

            map_metrics = load_json(
                map_folder
                / "final_test"
                / "final_metrics.json",
                {},
            )

            map_paths = map_result_paths(
                map_folder,
                map_config,
                map_metrics,
            )

            st.markdown(
                f"""
                <div class="data-line">
                    Source: <b>{map_paths["source"].title()}</b>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    Testing Zone: <b>{map_paths["test_zone_label"]}</b>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    Ground Truth: <b>{map_paths["ground_truth_label"]}</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # --------------------------------------------------------
            # Final-test metrics
            # --------------------------------------------------------

            metric_iou = percentage(
                map_metrics,
                "iou_percent",
                "iou",
            )

            metric_precision = percentage(
                map_metrics,
                "precision_percent",
                "precision",
            )

            metric_recall = percentage(
                map_metrics,
                "recall_percent",
                "recall",
            )

            metric_f1 = percentage(
                map_metrics,
                "f1_percent",
                "f1",
            )

            mc1, mc2, mc3, mc4 = st.columns(
                4
            )

            mc1.metric(
                "Test IoU",
                (
                    f"{metric_iou:.2f}%"
                    if metric_iou is not None
                    else "—"
                ),
            )

            mc2.metric(
                "Precision",
                (
                    f"{metric_precision:.2f}%"
                    if metric_precision is not None
                    else "—"
                ),
            )

            mc3.metric(
                "Recall",
                (
                    f"{metric_recall:.2f}%"
                    if metric_recall is not None
                    else "—"
                ),
            )

            mc4.metric(
                "F1",
                (
                    f"{metric_f1:.2f}%"
                    if metric_f1 is not None
                    else "—"
                ),
            )

            # --------------------------------------------------------
            # Overlay controls
            # --------------------------------------------------------

            ctrl1, ctrl2 = st.columns(
                2
            )

            with ctrl1:
                show_gt = st.checkbox(
                    "Ground Truth",
                    value=True,
                    key=f"overlay_gt_{safe_token(selected_map)}",
                )

            with ctrl2:
                show_prediction = st.checkbox(
                    "Final Prediction",
                    value=True,
                    key=f"overlay_prediction_{safe_token(selected_map)}",
                )

            if not (
                show_gt
                or show_prediction
            ):
                st.info(
                    "Select Ground Truth, Final Prediction, or both."
                )

            else:
                try:
                    test_image_path = map_paths.get(
                        "test_image_path"
                    )

                    if not (
                        test_image_path
                        and arcgis_exists(
                            test_image_path
                        )
                    ):
                        st.error(
                            "The external testing raster could not be found."
                        )

                    else:
                        ground_truth_path = map_paths.get(
                            "ground_truth_path"
                        )

                        if not arcgis_exists(
                            ground_truth_path
                        ):
                            ground_truth_path = map_paths.get(
                                "original_ground_truth_path"
                            )

                        prediction_path = map_paths.get(
                            "prediction_path"
                        )

                        if (
                            show_gt
                            and not arcgis_exists(
                                ground_truth_path
                            )
                        ):
                            st.warning(
                                "Ground Truth feature class could not be found."
                            )

                        if (
                            show_prediction
                            and not arcgis_exists(
                                prediction_path
                            )
                        ):
                            st.warning(
                                "Final Prediction feature class could not be found."
                            )

                        overlay_result = (
                            arcgis_testing_image_with_overlays(
                                raster_path=test_image_path,
                                ground_truth_path=ground_truth_path,
                                prediction_path=prediction_path,
                                show_ground_truth=(
                                    show_gt
                                    and arcgis_exists(
                                        ground_truth_path
                                    )
                                ),
                                show_prediction=(
                                    show_prediction
                                    and arcgis_exists(
                                        prediction_path
                                    )
                                ),
                            )
                        )

                        st.image(
                            overlay_result[
                                "image_bytes"
                            ],
                            caption=(
                                f"{map_paths['test_zone_label']} — "
                                "Ground Truth and Final Prediction"
                            ),
                            width="stretch",
                        )

                        st.markdown(
                            """
                            <div style="
                                display:flex;
                                gap:22px;
                                align-items:center;
                                font-size:0.84rem;
                                color:#334155;
                                margin:0.25rem 0 0.4rem 0;
                            ">
                                <span>
                                    <span style="
                                        display:inline-block;
                                        width:28px;
                                        border-top:5px solid #dc2626;
                                        margin-right:6px;
                                        vertical-align:3px;
                                    "></span>
                                    Ground Truth
                                </span>
                                <span>
                                    <span style="
                                        display:inline-block;
                                        width:28px;
                                        border-top:3px solid #2563eb;
                                        margin-right:6px;
                                        vertical-align:3px;
                                    "></span>
                                    Final Prediction
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.caption(
                            "The polygons are overlaid directly on the real external testing image. "
                            "Red = Ground Truth; Blue = Final Prediction. "
                            "No inference or model processing is rerun."
                        )

                except Exception as exc:
                    st.error(
                        f"Could not create the testing-image overlay: {exc}"
                    )


        st.markdown(
            """
            <div class="footer">
                Building Extraction · Simple Manual Interface
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DIRECT PAGE SUPPORT
# ============================================================

if __name__ == "__main__":
    render()
