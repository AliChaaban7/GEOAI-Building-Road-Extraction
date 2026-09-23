from pathlib import Path

PROJECT_ROOT = Path(r"D:\Ali Chaaban Thesis Project\GeoAI_Thesis_Codebase")

BUILDINGS = PROJECT_ROOT / "Buildings"
APPROACH_2 = PROJECT_ROOT / "approach_2_road_extraction"
APPROACH_3 = PROJECT_ROOT / "approach_3_future_work"

folders = [
    # Main project folders
    "shared",
    "master_outputs",
    "master_outputs/final_figures",
    "thesis_docs",
    "thesis_docs/supervisor_reports",

    # Empty future approaches
    "approach_2_road_extraction",
    "approach_3_future_work",

    # Buildings approach folders
    "Buildings/config",
    "Buildings/config/models",
    "Buildings/config/optional",

    "Buildings/src",
    "Buildings/src/models",
    "Buildings/src/optional",

    "Buildings/scripts",
    "Buildings/ui",

    "Buildings/outputs",
    "Buildings/outputs/experiments",
    "Buildings/outputs/optuna",
    "Buildings/outputs/figures",

    "Buildings/notebooks",
    "Buildings/docs"
]

files = [
    # Main project files
    "README.md",
    "requirements.txt",
    "environment.yml",

    # Shared files
    "shared/gis_utils.py",
    "shared/metrics.py",
    "shared/visualization.py",
    "shared/file_utils.py",
    "shared/report_utils.py",

    # Master output files
    "master_outputs/master_results.csv",
    "master_outputs/approach_comparison.csv",

    # Thesis docs
    "thesis_docs/methodology_notes.md",
    "thesis_docs/final_results_notes.md",

    # Buildings files
    "Buildings/README.md",

    "Buildings/config/experiment.json",
    "Buildings/config/paths.json",
    "Buildings/config/datasets.json",
    "Buildings/config/general_params.json",

    "Buildings/config/models/unet.json",
    "Buildings/config/models/deeplabv3.json",
    "Buildings/config/models/maskrcnn.json",

    "Buildings/config/optional/augmentation.json",
    "Buildings/config/optional/normalization.json",
    "Buildings/config/optional/resampling.json",
    "Buildings/config/optional/error_analysis.json",
    "Buildings/config/optional/reporting.json",
    "Buildings/config/optional/optuna.json",

    "Buildings/src/utils.py",
    "Buildings/src/train.py",
    "Buildings/src/inference.py",
    "Buildings/src/postprocess.py",
    "Buildings/src/pipeline.py",
    "Buildings/src/report.py",

    "Buildings/src/models/__init__.py",
    "Buildings/src/models/factory.py",
    "Buildings/src/models/backbones.py",
    "Buildings/src/models/unet.py",
    "Buildings/src/models/deeplabv3.py",
    "Buildings/src/models/maskrcnn.py",

    "Buildings/src/optional/augmentation.py",
    "Buildings/src/optional/normalization.py",
    "Buildings/src/optional/resampling.py",
    "Buildings/src/optional/error_analysis.py",
    "Buildings/src/optional/reporting_tools.py",
    "Buildings/src/optional/optuna_search.py",

    "Buildings/scripts/run_pipeline.py",
    "Buildings/scripts/run_train.py",
    "Buildings/scripts/run_inference.py",
    "Buildings/scripts/run_postprocess.py",
    "Buildings/scripts/run_optuna.py",

    "Buildings/ui/app.py",
    "Buildings/ui/config_page.py",
    "Buildings/ui/run_page.py",
    "Buildings/ui/results_page.py",

    "Buildings/outputs/master_results.csv",

    "Buildings/notebooks/01_dataset_quality_exploration.ipynb",
    "Buildings/notebooks/02_training_experiment_review.ipynb",
    "Buildings/notebooks/03_inference_and_prediction_review.ipynb",
    "Buildings/notebooks/04_results_analysis_and_model_comparison.ipynb",

    "Buildings/docs/methodology.md",
    "Buildings/docs/configuration_notes.md",
    "Buildings/docs/results_notes.md"
]


def create_project_structure():
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)

    for folder in folders:
        folder_path = PROJECT_ROOT / folder
        folder_path.mkdir(parents=True, exist_ok=True)

    for file in files:
        file_path = PROJECT_ROOT / file
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.write_text("", encoding="utf-8")

    print("Project structure created successfully.")
    print(f"Main project path: {PROJECT_ROOT}")
    print(f"Buildings path: {BUILDINGS}")
    print(f"Approach 2 path: {APPROACH_2}  [empty for now]")
    print(f"Approach 3 path: {APPROACH_3}  [empty for now]")


if __name__ == "__main__":
    create_project_structure()