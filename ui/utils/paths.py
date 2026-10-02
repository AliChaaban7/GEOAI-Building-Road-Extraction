"""
ui/utils/paths.py

Central filesystem paths for the project-level GeoAI Master UI.

Design rules
------------
- Buildings/ remains fully isolated.
- Roads/ remains fully isolated.
- The Master UI writes only to master_outputs/ui/.
- No backend config, checkpoint, output, or source file is overwritten here.
"""

from __future__ import annotations

from pathlib import Path


# ============================================================
# ROOTS
# ============================================================

UI_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = UI_ROOT.parent


# ============================================================
# APPROACH 1 — BUILDINGS
# ============================================================

BUILDINGS_ROOT = PROJECT_ROOT / "Buildings"

BUILDINGS_CONFIG_ROOT = BUILDINGS_ROOT / "config"
BUILDINGS_SCRIPTS_ROOT = BUILDINGS_ROOT / "scripts"
BUILDINGS_SRC_ROOT = BUILDINGS_ROOT / "src"
BUILDINGS_OUTPUTS_ROOT = BUILDINGS_ROOT / "outputs"
BUILDINGS_EXPERIMENTS_ROOT = (
    BUILDINGS_OUTPUTS_ROOT
    / "experiments"
)
BUILDINGS_CHECKPOINTS_ROOT = (
    BUILDINGS_ROOT
    / "checkpoints"
)
BUILDINGS_UI_ROOT = (
    BUILDINGS_ROOT
    / "ui"
)


# ============================================================
# APPROACH 2 — ROADS
# ============================================================

ROADS_ROOT = PROJECT_ROOT / "Roads"

ROADS_CONFIG_ROOT = ROADS_ROOT / "config"
ROADS_SCRIPTS_ROOT = ROADS_ROOT / "scripts"
ROADS_SRC_ROOT = ROADS_ROOT / "src"
ROADS_OUTPUTS_ROOT = ROADS_ROOT / "outputs"
ROADS_EXPERIMENTS_ROOT = (
    ROADS_OUTPUTS_ROOT
    / "experiments"
)
ROADS_CHECKPOINTS_ROOT = (
    ROADS_ROOT
    / "checkpoints"
)
ROADS_UI_ROOT = (
    ROADS_ROOT
    / "ui"
)


# ============================================================
# PROJECT-LEVEL OUTPUTS
# ============================================================

MASTER_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "master_outputs"
)

UI_OUTPUT_ROOT = (
    MASTER_OUTPUT_ROOT
    / "ui"
)

UI_CONFIG_ROOT = (
    UI_OUTPUT_ROOT
    / "configs"
)

UI_LOG_ROOT = (
    UI_OUTPUT_ROOT
    / "logs"
)


# ============================================================
# APPROACH-SPECIFIC MASTER UI RUNTIME FOLDERS
# ============================================================

BUILDING_UI_CONFIG_ROOT = (
    UI_CONFIG_ROOT
    / "buildings"
)

ROAD_UI_CONFIG_ROOT = (
    UI_CONFIG_ROOT
    / "roads"
)

BUILDING_UI_LOG_ROOT = (
    UI_LOG_ROOT
    / "buildings"
)

ROAD_UI_LOG_ROOT = (
    UI_LOG_ROOT
    / "roads"
)


# ============================================================
# COMMON CONFIG FILES
# ============================================================

BUILDINGS_EXPERIMENT_CONFIG = (
    BUILDINGS_CONFIG_ROOT
    / "experiment.json"
)

BUILDINGS_DATASETS_CONFIG = (
    BUILDINGS_CONFIG_ROOT
    / "datasets.json"
)

BUILDINGS_PATHS_CONFIG = (
    BUILDINGS_CONFIG_ROOT
    / "paths.json"
)

BUILDINGS_TEST_ZONES_CONFIG = (
    BUILDINGS_CONFIG_ROOT
    / "test_zones.json"
)


ROADS_EXPERIMENT_CONFIG = (
    ROADS_CONFIG_ROOT
    / "experiment.json"
)

ROADS_DATASETS_CONFIG = (
    ROADS_CONFIG_ROOT
    / "datasets.json"
)

ROADS_PATHS_CONFIG = (
    ROADS_CONFIG_ROOT
    / "paths.json"
)

ROADS_TEST_ZONE_CONFIG = (
    ROADS_CONFIG_ROOT
    / "test_zone.json"
)

ROADS_UI_SCHEMA_CONFIG = (
    ROADS_CONFIG_ROOT
    / "ui_schema.json"
)


# ============================================================
# BACKEND CONTROLLERS / IMPORTANT SCRIPTS
# ============================================================

BUILDINGS_CONTROLLER = (
    BUILDINGS_SCRIPTS_ROOT
    / "run_buildings.py"
)

ROADS_TRAIN_SCRIPT = (
    ROADS_SCRIPTS_ROOT
    / "run_train.py"
)

ROADS_ARCGIS_TRAIN_SCRIPT = (
    ROADS_SCRIPTS_ROOT
    / "run_arcgis_train.py"
)

ROADS_OPTUNA_SCRIPT = (
    ROADS_SCRIPTS_ROOT
    / "run_optuna.py"
)

ROADS_MASTER_OPTUNA_SCRIPT = (
    ROADS_SCRIPTS_ROOT
    / "run_master_optuna.py"
)

ROADS_THRESHOLD_SCRIPT = (
    ROADS_SCRIPTS_ROOT
    / "run_threshold_search.py"
)

ROADS_POSTPROCESS_SCRIPT = (
    ROADS_SCRIPTS_ROOT
    / "run_postprocess.py"
)

ROADS_FREEZE_SCRIPT = (
    ROADS_SCRIPTS_ROOT
    / "freeze_final_settings.py"
)

ROADS_FINAL_TEST_SCRIPT = (
    ROADS_SCRIPTS_ROOT
    / "run_final_test.py"
)


# ============================================================
# DIRECTORY CREATION
# ============================================================

def ensure_master_ui_directories() -> None:
    """
    Create only Master-UI runtime directories.

    This function deliberately does NOT create or modify:
    - Buildings/
    - Roads/
    - backend experiment folders
    - backend checkpoint folders
    """

    folders = (
        MASTER_OUTPUT_ROOT,
        UI_OUTPUT_ROOT,
        UI_CONFIG_ROOT,
        UI_LOG_ROOT,
        BUILDING_UI_CONFIG_ROOT,
        ROAD_UI_CONFIG_ROOT,
        BUILDING_UI_LOG_ROOT,
        ROAD_UI_LOG_ROOT,
    )

    for folder in folders:
        folder.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# SAFETY / STATUS HELPERS
# ============================================================

def project_structure_status() -> dict:
    """
    Return a lightweight status dictionary for the Master UI.

    This is read-only and does not create anything.
    """

    return {
        "project_root": PROJECT_ROOT.exists(),
        "ui_root": UI_ROOT.exists(),
        "buildings_root": BUILDINGS_ROOT.exists(),
        "roads_root": ROADS_ROOT.exists(),
        "buildings_controller": BUILDINGS_CONTROLLER.exists(),
        "roads_train_script": ROADS_TRAIN_SCRIPT.exists(),
        "roads_arcgis_train_script": ROADS_ARCGIS_TRAIN_SCRIPT.exists(),
        "roads_optuna_script": ROADS_OPTUNA_SCRIPT.exists(),
        "roads_master_optuna_script": ROADS_MASTER_OPTUNA_SCRIPT.exists(),
        "roads_threshold_script": ROADS_THRESHOLD_SCRIPT.exists(),
        "roads_postprocess_script": ROADS_POSTPROCESS_SCRIPT.exists(),
        "roads_freeze_script": ROADS_FREEZE_SCRIPT.exists(),
        "roads_final_test_script": ROADS_FINAL_TEST_SCRIPT.exists(),
    }


def require_backend_roots() -> None:
    """
    Fail clearly if the Master UI is not inside the thesis repository.
    """

    if not BUILDINGS_ROOT.exists():
        raise FileNotFoundError(
            f"Buildings module not found:\n{BUILDINGS_ROOT}"
        )

    if not ROADS_ROOT.exists():
        raise FileNotFoundError(
            f"Roads module not found:\n{ROADS_ROOT}"
        )
