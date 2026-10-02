"""
Validation-only Road morphology search.

Searches frozen Road-surface post-processing parameters after threshold
selection.

Required first:
    validation_threshold_summary.json

Search happens ONLY on validation chips.

No final-test scene or Ground Truth is accessed.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch


ROAD_ROOT = Path(__file__).resolve().parents[1]

if str(ROAD_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROAD_ROOT),
    )


from src.inference import restore_model_from_checkpoint
from src.pipeline import prepare_experiment
from src.postprocessing.road_mask import (
    RoadPostprocessingConfig,
    postprocess_road_mask,
)
from src.train import build_dataloaders


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Validation-only Road post-processing search."
        )
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
        "--threshold-summary",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    return parser.parse_args()


def load_json(
    path,
):

    return json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )


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
    Normalize compatibility keys needed by Road validation-only
    post-processing.

    This does NOT retrain, regenerate manifests, or touch final-test scenes.
    It reuses the exact trained experiment, checkpoint and manifests.
    """

    if not isinstance(
        context,
        dict,
    ):
        raise TypeError(
            "prepare_experiment() must return a dictionary context."
        )

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

    context[
        "output_folder"
    ] = output_folder

    context[
        "checkpoint_folder"
    ] = Path(
        context.get(
            "checkpoint_folder",
            output_folder,
        )
        or output_folder
    )

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
            "Could not resolve model_type for Road post-processing."
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

        candidates = [
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
                in candidates
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
                    in candidates
                )
            )

        context[
            "model_config"
        ] = json.loads(
            model_config_path.read_text(
                encoding="utf-8"
            )
        )

    general_params = context.get(
        "general_params"
    )

    if not isinstance(
        general_params,
        dict,
    ) or not general_params:

        general_path = (
            ROAD_ROOT
            / "config"
            / "general_params.json"
        )

        if not general_path.exists():
            raise FileNotFoundError(
                f"Road general_params.json not found:\n{general_path}"
            )

        context[
            "general_params"
        ] = json.loads(
            general_path.read_text(
                encoding="utf-8"
            )
        )

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
            f"Existing Road train manifest not found:\n{train_manifest}"
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


def read_search_grid():

    path = (
        ROAD_ROOT
        / "config"
        / "optional"
        / "postprocessing.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing post-processing config:\n{path}"
        )

    config = load_json(
        path
    )

    # ========================================================
    # AUTHORITATIVE ROAD POST-PROCESSING CONFIG
    # ========================================================
    #
    # The Road configuration uses the richer schema:
    #
    # operations.remove_small_components.candidate_min_component_pixels
    # operations.morphological_closing.candidate_radius_pixels
    # operations.bridge_small_gaps.candidate_gap_pixels
    # operations.fill_small_holes.candidate_max_hole_pixels
    #
    # Older/flat search-grid layouts are still accepted as a
    # compatibility fallback, but the nested operations schema is
    # preferred and preserved.
    # ========================================================

    operations = config.get(
        "operations",
        {},
    )

    def nested_values(
        operation_name,
        candidate_key,
    ):

        operation = operations.get(
            operation_name,
            {},
        )

        if not isinstance(
            operation,
            dict,
        ):
            return None

        enabled = bool(
            operation.get(
                "enabled",
                True,
            )
        )

        if not enabled:
            return [0]

        values = operation.get(
            candidate_key
        )

        if isinstance(
            values,
            list,
        ):
            return values

        return None

    # --------------------------------------------------------
    # Preferred nested Roads schema
    # --------------------------------------------------------

    components = nested_values(
        "remove_small_components",
        "candidate_min_component_pixels",
    )

    closing = nested_values(
        "morphological_closing",
        "candidate_radius_pixels",
    )

    bridge = nested_values(
        "bridge_small_gaps",
        "candidate_gap_pixels",
    )

    holes = nested_values(
        "fill_small_holes",
        "candidate_max_hole_pixels",
    )

    # --------------------------------------------------------
    # Compatibility fallback for older/flat schemas
    # --------------------------------------------------------

    search_block = config.get(
        "search",
        {},
    )

    def flat_values(
        *names,
    ):

        if not isinstance(
            search_block,
            dict,
        ):
            return None

        for name in names:

            value = search_block.get(
                name
            )

            if value is None:
                continue

            if isinstance(
                value,
                dict,
            ):
                value = value.get(
                    "values"
                )

            if isinstance(
                value,
                list,
            ):
                return value

        return None

    if components is None:
        components = flat_values(
            "minimum_component_size",
            "minimum_component_sizes",
            "component_sizes",
            "min_component_pixels",
        )

    if closing is None:
        closing = flat_values(
            "closing_radius",
            "closing_radii",
            "closing",
        )

    if holes is None:
        holes = flat_values(
            "maximum_hole_size",
            "maximum_hole_sizes",
            "hole_sizes",
            "max_hole_pixels",
        )

    if bridge is None:
        bridge = flat_values(
            "bridge_gap_radius",
            "bridge_gap_radii",
            "bridge_radius",
            "bridge",
        )

    missing = []

    if components is None:
        missing.append(
            "operations.remove_small_components."
            "candidate_min_component_pixels"
        )

    if closing is None:
        missing.append(
            "operations.morphological_closing."
            "candidate_radius_pixels"
        )

    if holes is None:
        missing.append(
            "operations.fill_small_holes."
            "candidate_max_hole_pixels"
        )

    if bridge is None:
        missing.append(
            "operations.bridge_small_gaps."
            "candidate_gap_pixels"
        )

    if missing:

        raise ValueError(
            "Missing post-processing search grids:\n"
            + "\n".join(
                f"- {name}"
                for name
                in missing
            )
        )

    def clean_values(
        values,
        name,
    ):

        cleaned = sorted(
            {
                int(value)
                for value
                in values
            }
        )

        if not cleaned:
            raise ValueError(
                f"No candidates configured for {name}."
            )

        if any(
            value < 0
            for value
            in cleaned
        ):
            raise ValueError(
                f"{name} candidates must be >= 0."
            )

        return cleaned

    components = clean_values(
        components,
        "minimum_component_size",
    )

    closing = clean_values(
        closing,
        "closing_radius",
    )

    holes = clean_values(
        holes,
        "maximum_hole_size",
    )

    bridge = clean_values(
        bridge,
        "bridge_gap_radius",
    )

    # The existing Road config requests evaluation of the original
    # unprocessed mask. Ensure the all-zero baseline can be tested.
    search_policy = config.get(
        "search",
        {},
    )

    if (
        isinstance(
            search_policy,
            dict,
        )
        and bool(
            search_policy.get(
                "evaluate_original_mask",
                False,
            )
        )
    ):

        for values in (
            components,
            closing,
            holes,
            bridge,
        ):
            if 0 not in values:
                values.insert(
                    0,
                    0,
                )

    return {
        "minimum_component_size":
            components,

        "closing_radius":
            closing,

        "maximum_hole_size":
            holes,

        "bridge_gap_radius":
            bridge,
    }


def metrics_from_counts(
    tp,
    fp,
    fn,
):

    tp = float(tp)
    fp = float(fp)
    fn = float(fn)

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
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall > 0
        else 0.0
    )

    return (
        iou,
        precision,
        recall,
        f1,
    )


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
            "Current validation morphology search supports "
            "native PyTorch Road models."
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

    threshold_summary = (
        Path(
            args.threshold_summary
        )
        if args.threshold_summary
        else (
            output_folder
            / "validation_threshold_summary.json"
        )
    )

    if not threshold_summary.exists():

        raise FileNotFoundError(
            "Run threshold search first:\n"
            f"{threshold_summary}"
        )

    threshold_payload = load_json(
        threshold_summary
    )

    if bool(
        threshold_payload.get(
            "final_test_used",
            False,
        )
    ):

        raise RuntimeError(
            "Invalid threshold summary: final test was used."
        )

    threshold = float(
        threshold_payload[
            "best_threshold"
        ]
    )

    grid = read_search_grid()

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

    # --------------------------------------------------------------
    # Cache validation prediction/GT masks once.
    # --------------------------------------------------------------

    samples = []

    from src.models import extract_model_logits

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
                device
            )
        )

        logits = extract_model_logits(
            model(
                images
            )
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
            .detach()
            .cpu()
            .numpy()
        )

        targets = (
            masks
            .detach()
            .cpu()
            .numpy()
        )

        for index in range(
            probabilities.shape[
                0
            ]
        ):

            prediction = (
                probabilities[
                    index,
                    0,
                ]
                >= threshold
            ).astype(
                np.uint8
            )

            target = (
                targets[
                    index,
                    0,
                ]
                >= 0.5
            ).astype(
                np.uint8
            )

            samples.append(
                (
                    prediction,
                    target,
                )
            )

    rows = []

    best = None

    combinations = itertools.product(
        grid[
            "minimum_component_size"
        ],
        grid[
            "closing_radius"
        ],
        grid[
            "maximum_hole_size"
        ],
        grid[
            "bridge_gap_radius"
        ],
    )

    for (
        component_size,
        closing_radius,
        hole_size,
        bridge_radius,
    ) in combinations:

        config = RoadPostprocessingConfig(
            minimum_component_size=(
                component_size
            ),
            closing_radius=(
                closing_radius
            ),
            maximum_hole_size=(
                hole_size
            ),
            bridge_gap_radius=(
                bridge_radius
            ),
        )

        tp = 0
        fp = 0
        fn = 0

        for (
            prediction,
            target,
        ) in samples:

            processed = postprocess_road_mask(
                mask=prediction,
                config=config,
            )

            processed = (
                np.asarray(
                    processed
                )
                > 0
            )

            target_bool = (
                target
                > 0
            )

            tp += int(
                np.logical_and(
                    processed,
                    target_bool,
                ).sum()
            )

            fp += int(
                np.logical_and(
                    processed,
                    ~target_bool,
                ).sum()
            )

            fn += int(
                np.logical_and(
                    ~processed,
                    target_bool,
                ).sum()
            )

        (
            iou,
            precision,
            recall,
            f1,
        ) = metrics_from_counts(
            tp,
            fp,
            fn,
        )

        row = {
            "minimum_component_size":
                component_size,

            "closing_radius":
                closing_radius,

            "maximum_hole_size":
                hole_size,

            "bridge_gap_radius":
                bridge_radius,

            "iou":
                iou,

            "iou_percent":
                iou
                * 100.0,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,
        }

        rows.append(
            row
        )

        if (
            best is None
            or iou
            > best[
                "iou"
            ]
        ):

            best = row

    search_folder = (
        output_folder
        / "validation_postprocess_search"
    )

    search_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        search_folder
        / "validation_postprocess_search.csv"
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

        "threshold":
            threshold,

        "selected_config": {
            "minimum_component_size":
                int(
                    best[
                        "minimum_component_size"
                    ]
                ),

            "closing_radius":
                int(
                    best[
                        "closing_radius"
                    ]
                ),

            "maximum_hole_size":
                int(
                    best[
                        "maximum_hole_size"
                    ]
                ),

            "bridge_gap_radius":
                int(
                    best[
                        "bridge_gap_radius"
                    ]
                ),
        },

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
        / "validation_postprocess_summary.json",
    )

    save_json(
        summary,
        output_folder
        / "validation_postprocess_summary.json",
    )

    print()
    print(
        "=" * 72
    )

    print(
        "ROAD POST-PROCESSING SEARCH COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Validation IoU : "
        f"{best['iou_percent']:.4f}%"
    )

    print(
        "Selected config:"
    )

    for key, value in summary[
        "selected_config"
    ].items():

        print(
            f"  {key:<26}: {value}"
        )

    print(
        "Final test used: NO"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()