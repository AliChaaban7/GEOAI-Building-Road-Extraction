"""
test_model_creation.py

Flexible model creation test for Buildings module.

Supports:
- Semantic segmentation models:
    - U-Net
    - DeepLabV3

- Instance segmentation models:
    - Mask R-CNN

This script only tests model creation and dummy forward pass.
It does not train the model.
"""

from pathlib import Path
import sys
import json

import torch


# ============================================================
# Paths
# ============================================================

BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BUILDINGS_ROOT.parent

sys.path.append(str(BUILDINGS_ROOT))
sys.path.append(str(PROJECT_ROOT))


# ============================================================
# Imports
# ============================================================

from src.models.factory import build_model


# ============================================================
# JSON helpers
# ============================================================

def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Config helpers
# ============================================================

def load_experiment_config():
    return load_json(BUILDINGS_ROOT / "config" / "experiment.json")


def load_model_config(model_type):
    return load_json(BUILDINGS_ROOT / "config" / "models" / f"{model_type}.json")


def normalize_model_type(model_type):
    model_type = str(model_type).lower().strip()

    aliases = {
        "u-net": "unet",
        "u_net": "unet",
        "deeplab": "deeplabv3",
        "deep_lab_v3": "deeplabv3",
        "mask_rcnn": "maskrcnn",
        "mask-r-cnn": "maskrcnn",
        "rcnn": "maskrcnn"
    }

    return aliases.get(model_type, model_type)


def get_backbone_info(experiment_config, model_config):
    backbone = experiment_config.get(
        "backbone",
        model_config.get("default_backbone", None)
    )

    if backbone is None:
        return {}

    if "supported_backbones" in model_config:
        return model_config["supported_backbones"].get(
            backbone,
            {"name": backbone}
        )

    if "backbones" in model_config:
        return model_config["backbones"].get(
            backbone,
            {"name": backbone}
        )

    return {"name": backbone}


# ============================================================
# Model helpers
# ============================================================

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def extract_semantic_logits(outputs):
    if isinstance(outputs, dict):
        if "out" in outputs:
            return outputs["out"]

        first_key = list(outputs.keys())[0]
        return outputs[first_key]

    return outputs


def test_semantic_forward(model, device, tile_size):
    model.eval()

    dummy_input = torch.randn(
        1,
        3,
        tile_size,
        tile_size,
        device=device
    )

    with torch.no_grad():
        outputs = model(dummy_input)

    logits = extract_semantic_logits(outputs)

    print(f"Dummy Input Shape: {tuple(dummy_input.shape)}")
    print(f"Model Output Shape: {tuple(logits.shape)}")

    if logits.ndim != 4:
        raise ValueError(
            "Semantic model output should be 4D: [B, C, H, W]. "
            f"Got shape: {tuple(logits.shape)}"
        )

    if logits.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1. Got: {logits.shape[0]}"
        )

    if logits.shape[2] != tile_size or logits.shape[3] != tile_size:
        raise ValueError(
            "Output spatial size does not match tile size.\n"
            f"Expected: {tile_size} x {tile_size}\n"
            f"Got: {logits.shape[2]} x {logits.shape[3]}"
        )

    return logits


def test_maskrcnn_forward(model, device, tile_size):
    model.eval()

    dummy_image = torch.rand(
        3,
        tile_size,
        tile_size,
        device=device
    )

    with torch.no_grad():
        outputs = model([dummy_image])

    print(f"Dummy Input Shape: {tuple(dummy_image.shape)}")
    print(f"Output Type: {type(outputs)}")

    if not isinstance(outputs, list):
        raise ValueError(
            "Mask R-CNN output should be a list of dictionaries."
        )

    if len(outputs) != 1:
        raise ValueError(
            f"Expected one output dictionary. Got: {len(outputs)}"
        )

    output = outputs[0]

    expected_keys = ["boxes", "labels", "scores", "masks"]

    print(f"Output Keys: {list(output.keys())}")

    for key in expected_keys:
        if key not in output:
            raise KeyError(
                f"Mask R-CNN output is missing key: {key}"
            )

    print(f"Boxes Shape: {tuple(output['boxes'].shape)}")
    print(f"Labels Shape: {tuple(output['labels'].shape)}")
    print(f"Scores Shape: {tuple(output['scores'].shape)}")
    print(f"Masks Shape: {tuple(output['masks'].shape)}")

    if output["boxes"].ndim != 2:
        raise ValueError("boxes should have shape [N, 4].")

    if output["masks"].ndim != 4:
        raise ValueError("masks should have shape [N, 1, H, W].")

    print(
        "Note: Random dummy image may produce 0 detections. "
        "That is normal for this test."
    )

    return output


# ============================================================
# Main
# ============================================================

def main():
    experiment_config = load_experiment_config()

    model_type = normalize_model_type(
        experiment_config.get("model_type")
    )

    model_config = load_model_config(model_type)
    backbone_info = get_backbone_info(
        experiment_config=experiment_config,
        model_config=model_config
    )

    tile_size = int(experiment_config.get("tile_size", 256))
    backbone = experiment_config.get(
        "backbone",
        model_config.get("default_backbone", "none")
    )

    print("\n========== MODEL CREATION TEST ==========")
    print(f"Experiment ID: {experiment_config.get('experiment_id')}")
    print(f"Model Type: {model_type}")
    print(f"Backbone: {backbone}")

    model, device = build_model(
        experiment_config=experiment_config,
        model_config=model_config,
        backbone_info=backbone_info,
        move_to_device=True
    )

    total_params, trainable_params = count_parameters(model)

    print(f"Device: {device}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    if model_type in ["unet", "deeplabv3"]:
        test_semantic_forward(
            model=model,
            device=device,
            tile_size=tile_size
        )

    elif model_type == "maskrcnn":
        test_maskrcnn_forward(
            model=model,
            device=device,
            tile_size=tile_size
        )

    else:
        raise ValueError(
            f"Unsupported model_type in test script: {model_type}"
        )

    print("Model creation test completed successfully.")
    print("=========================================\n")


if __name__ == "__main__":
    main()