"""
Validation-only probability-threshold search for Road extraction.

Supported native models:
- U-Net
- DeepLabV3+
- SAM-LoRA

Scientific rule
---------------
Threshold is selected ONLY on Validation.

External Aerial / Satellite / Drone test scenes are never accessed.

Default search values are read from general_params.json.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch


ROAD_ROOT = Path(__file__).resolve().parents[1]

if str(ROAD_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROAD_ROOT),
    )


from src.inference import restore_model_from_checkpoint
from src.pipeline import prepare_experiment
from src.train import build_dataloaders


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Validation-only Road threshold search."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    return parser.parse_args()


def save_json(
    data,
    path,
):

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )


def normalize_validation_context(
    context,
    config_path,
):
    """
    Normalize the experiment context required by the Road validation
    DataLoader and checkpoint loader.

    prepare_experiment() versions in the current Road pipeline do not all
    expose the same compatibility keys. Validation must nevertheless reuse
    the exact already-trained experiment, manifests and model configuration.
    """

    if not isinstance(
        context,
        dict,
    ):
        raise TypeError(
            "prepare_experiment() must return a dictionary context."
        )

    # --------------------------------------------------------
    # Output / checkpoint folder
    # --------------------------------------------------------
    output_folder_value = context.get(
        "output_folder"
    )

    if output_folder_value is None:
        raise KeyError(
            "prepare_experiment() did not return 'output_folder'."
        )

    output_folder = Path(
        output_folder_value
    )

    checkpoint_folder = Path(
        context.get(
            "checkpoint_folder",
            output_folder,
        )
        or output_folder
    )

    context[
        "output_folder"
    ] = output_folder

    context[
        "checkpoint_folder"
    ] = checkpoint_folder

    # --------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------
    experiment_config = context.get(
        "experiment_config"
    )

    if not isinstance(
        experiment_config,
        dict,
    ):
        if config_path is None:
            raise KeyError(
                "Validation context is missing 'experiment_config' "
                "and no --config path was supplied."
            )

        config_file = Path(
            config_path
        )

        if not config_file.exists():
            raise FileNotFoundError(
                f"Experiment config not found:\n{config_file}"
            )

        experiment_config = json.loads(
            config_file.read_text(
                encoding="utf-8"
            )
        )

        context[
            "experiment_config"
        ] = experiment_config

    # --------------------------------------------------------
    # Model type / model configuration
    # --------------------------------------------------------
    model_type = str(
        context.get(
            "model_type",
            experiment_config.get(
                "model_type",
                "",
            ),
        )
    ).strip().lower()

    aliases = {
        "u-net": "unet",
        "u_net": "unet",
        "deeplab": "deeplabv3",
        "deeplab_v3": "deeplabv3",
        "deeplabv3+": "deeplabv3",
        "deeplabv3plus": "deeplabv3",
        "sam-lora": "sam_lora",
        "sam lora": "sam_lora",
        "samlora": "sam_lora",
    }

    model_type = aliases.get(
        model_type,
        model_type,
    )

    if not model_type:
        raise KeyError(
            "Could not resolve model_type for threshold validation."
        )

    context[
        "model_type"
    ] = model_type

    model_config = context.get(
        "model_config"
    )

    if not isinstance(
        model_config,
        dict,
    ) or not model_config:

        model_config_candidates = [
            ROAD_ROOT
            / "config"
            / "models"
            / f"{model_type}.json",

            ROAD_ROOT
            / "config"
            / f"{model_type}.json",
        ]

        model_config_path = next(
            (
                path
                for path
                in model_config_candidates
                if path.exists()
            ),
            None,
        )

        if model_config_path is None:
            raise FileNotFoundError(
                "Road model configuration was not found.\n"
                "Checked:\n"
                + "\n".join(
                    str(path)
                    for path
                    in model_config_candidates
                )
            )

        model_config = json.loads(
            model_config_path.read_text(
                encoding="utf-8"
            )
        )

        context[
            "model_config"
        ] = model_config

    # --------------------------------------------------------
    # General parameters
    # --------------------------------------------------------
    general_params = context.get(
        "general_params"
    )

    if not isinstance(
        general_params,
        dict,
    ) or not general_params:

        general_params_path = (
            ROAD_ROOT
            / "config"
            / "general_params.json"
        )

        if not general_params_path.exists():
            raise FileNotFoundError(
                "Road general parameters not found:\n"
                f"{general_params_path}"
            )

        context[
            "general_params"
        ] = json.loads(
            general_params_path.read_text(
                encoding="utf-8"
            )
        )

    # --------------------------------------------------------
    # Existing Train / Validation manifests
    #
    # Threshold search NEVER regenerates or reshuffles the split.
    # It reuses the exact manifests created for this experiment.
    # --------------------------------------------------------
    train_manifest = (
        context.get(
            "train_manifest"
        )
        or context.get(
            "train_manifest_csv"
        )
        or (
            output_folder
            / "train_manifest.csv"
        )
    )

    validation_manifest = (
        context.get(
            "validation_manifest"
        )
        or context.get(
            "val_manifest"
        )
        or context.get(
            "validation_manifest_csv"
        )
        or (
            output_folder
            / "val_manifest.csv"
        )
    )

    train_manifest = Path(
        train_manifest
    )

    validation_manifest = Path(
        validation_manifest
    )

    if not train_manifest.exists():
        raise FileNotFoundError(
            "Existing Road train manifest not found:\n"
            f"{train_manifest}"
        )

    if not validation_manifest.exists():
        raise FileNotFoundError(
            "Existing Road validation manifest not found:\n"
            f"{validation_manifest}"
        )

    context[
        "train_manifest"
    ] = train_manifest

    context[
        "validation_manifest"
    ] = validation_manifest

    return context


def resolve_thresholds(
    context,
    explicit,
):

    if explicit:

        values = [
            float(value)
            for value
            in explicit
        ]

    else:

        general = context[
            "general_params"
        ]

        block = general.get(
            "threshold_search",
            general.get(
                "thresholds",
                {},
            ),
        )

        values = block.get(
            "values"
        )

        if values is None:

            low = block.get(
                "minimum",
                block.get(
                    "min",
                    0.30,
                ),
            )

            high = block.get(
                "maximum",
                block.get(
                    "max",
                    0.70,
                ),
            )

            step = block.get(
                "step",
                0.05,
            )

            low = float(low)
            high = float(high)
            step = float(step)

            values = []

            current = low

            while current <= high + 1e-12:

                values.append(
                    round(
                        current,
                        10,
                    )
                )

                current += step

    values = sorted(
        {
            float(value)
            for value
            in values
        }
    )

    for value in values:

        if not 0.0 <= value <= 1.0:

            raise ValueError(
                f"Invalid threshold: {value}"
            )

    if not values:

        raise ValueError(
            "No threshold values resolved."
        )

    return values


class Counts:

    def __init__(self):

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(
        self,
        prediction,
        target,
    ):

        prediction = prediction.bool()
        target = target.bool()

        self.tp += int(
            (
                prediction
                & target
            ).sum().item()
        )

        self.fp += int(
            (
                prediction
                & ~target
            ).sum().item()
        )

        self.fn += int(
            (
                ~prediction
                & target
            ).sum().item()
        )

        self.tn += int(
            (
                ~prediction
                & ~target
            ).sum().item()
        )

    def metrics(self):

        tp = float(self.tp)
        fp = float(self.fp)
        fn = float(self.fn)

        union = tp + fp + fn

        iou = (
            tp / union
            if union > 0
            else 1.0
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        return {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


@torch.no_grad()
def main():

    args = parse_arguments()

    context = prepare_experiment(
        config_path=args.config,
        force_regenerate_manifests=False,
    )

    context = normalize_validation_context(
        context=context,
        config_path=args.config,
    )

    if context[
        "model_backend"
    ] != "pytorch":

        raise RuntimeError(
            "This threshold launcher currently supports "
            "native PyTorch Road models only."
        )

    output_folder = Path(
        context[
            "output_folder"
        ]
    )

    checkpoint = (
        Path(
            args.checkpoint
        )
        if args.checkpoint
        else (
            Path(
                context[
                    "checkpoint_folder"
                ]
            )
            / "best_model.pth"
        )
    )

    if not checkpoint.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint}"
        )

    thresholds = resolve_thresholds(
        context,
        args.thresholds,
    )

    (
        model,
        _,
        device,
    ) = restore_model_from_checkpoint(
        checkpoint_path=checkpoint,
        context=context,
        device=args.device,
        strict=True,
    )

    data = build_dataloaders(
        context=context,
        train_augmentation=None,
        validation_augmentation=None,
    )

    accumulators = {
        threshold:
            Counts()

        for threshold
        in thresholds
    }

    for batch in data[
        "validation_loader"
    ]:

        if isinstance(
            batch,
            dict,
        ):

            images = batch.get(
                "image",
                batch.get(
                    "images"
                ),
            )

            masks = batch.get(
                "mask",
                batch.get(
                    "masks",
                ),
            )

        else:

            images = batch[0]
            masks = batch[1]

        images = (
            images.float()
            .to(
                device,
                non_blocking=True,
            )
        )

        masks = (
            masks.float()
            .to(
                device,
                non_blocking=True,
            )
        )

        logits = model(
            images
        )

        from src.models import extract_model_logits

        logits = extract_model_logits(
            logits
        )

        probabilities = torch.sigmoid(
            logits
        )

        targets = masks >= 0.5

        for threshold in thresholds:

            prediction = (
                probabilities
                >= threshold
            )

            accumulators[
                threshold
            ].update(
                prediction,
                targets,
            )

    rows = []

    for threshold in thresholds:

        metrics = accumulators[
            threshold
        ].metrics()

        row = {
            "threshold":
                threshold,

            **metrics,

            "iou_percent":
                metrics[
                    "iou"
                ]
                * 100.0,

            "precision_percent":
                metrics[
                    "precision"
                ]
                * 100.0,

            "recall_percent":
                metrics[
                    "recall"
                ]
                * 100.0,

            "f1_percent":
                metrics[
                    "f1"
                ]
                * 100.0,
        }

        rows.append(
            row
        )

    best = max(
        rows,
        key=lambda row: row[
            "iou"
        ],
    )

    search_folder = (
        output_folder
        / "validation_threshold_search"
    )

    search_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        search_folder
        / "validation_threshold_search.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    summary = {
        "selection_dataset":
            "validation",

        "selection_metric":
            "pixel_iou",

        "checkpoint":
            str(
                checkpoint
            ),

        "thresholds":
            thresholds,

        "best_threshold":
            float(
                best[
                    "threshold"
                ]
            ),

        "best_validation_iou":
            float(
                best[
                    "iou"
                ]
            ),

        "best_validation_iou_percent":
            float(
                best[
                    "iou_percent"
                ]
            ),

        "best_precision":
            float(
                best[
                    "precision"
                ]
            ),

        "best_recall":
            float(
                best[
                    "recall"
                ]
            ),

        "best_f1":
            float(
                best[
                    "f1"
                ]
            ),

        "external_test_used":
            False,

        "final_test_used":
            False,

        "search_csv":
            str(
                csv_path
            ),
    }

    save_json(
        summary,
        search_folder
        / "validation_threshold_summary.json",
    )

    save_json(
        summary,
        output_folder
        / "validation_threshold_summary.json",
    )

    print()
    print(
        "=" * 72
    )

    print(
        "ROAD VALIDATION THRESHOLD SEARCH COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Best threshold : {best['threshold']:.2f}"
    )

    print(
        f"Validation IoU : {best['iou_percent']:.4f}%"
    )

    print(
        "Final test used: NO"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()