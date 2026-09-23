"""
test_inference_preview.py

Create prediction preview images from validation chips.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utils import (
    load_experiment_config,
    load_model_config,
    get_backbone_info,
    create_experiment_output_folder,
    save_json
)

from src.models.factory import build_model

from src.inference import (
    get_best_threshold,
    load_model_checkpoint,
    save_validation_prediction_previews
)


def main():
    config_path = ROOT / "config" / "experiment.json"

    experiment_config = load_experiment_config(config_path)

    model_type = experiment_config["model_type"]
    model_config = load_model_config(model_type, ROOT)

    backbone_info = get_backbone_info(
        experiment_config=experiment_config,
        model_config=model_config
    )

    output_folder = create_experiment_output_folder(
        experiment_config=experiment_config,
        root=ROOT
    )

    best_model_path = output_folder / "best_model.pth"
    val_manifest_csv = output_folder / "val_manifest.csv"

    model, device, params = build_model(
        experiment_config=experiment_config,
        model_config=model_config,
        backbone_info=backbone_info,
        move_to_device=True
    )

    model, checkpoint = load_model_checkpoint(
        model=model,
        checkpoint_path=best_model_path,
        device=device
    )

    best_threshold = get_best_threshold(
        output_folder=output_folder,
        default_threshold=0.5
    )

    print("\n========== INFERENCE PREVIEW TEST ==========")
    print(f"Experiment ID: {experiment_config['experiment_id']}")
    print(f"Model Type: {experiment_config['model_type']}")
    print(f"Backbone: {experiment_config['backbone']}")
    print(f"Device: {device}")
    print(f"Best Model: {best_model_path}")
    print(f"Best Threshold: {best_threshold}")
    print(f"Validation Manifest: {val_manifest_csv}")
    print("============================================\n")

    summary = save_validation_prediction_previews(
        model=model,
        val_manifest_csv=val_manifest_csv,
        output_folder=output_folder,
        device=device,
        threshold=best_threshold,
        max_samples=12
    )

    save_json(
        summary,
        output_folder / "inference_preview_summary.json"
    )

    print("\n========== INFERENCE PREVIEW FINISHED ==========")
    print(f"Saved Count: {summary['saved_count']}")
    print(f"Preview Folder: {summary['preview_folder']}")
    print(f"Summary JSON: {output_folder / 'inference_preview_summary.json'}")
    print("================================================\n")


if __name__ == "__main__":
    main()