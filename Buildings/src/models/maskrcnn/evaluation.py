"""
Mask R-CNN validation, threshold-search, and post-processing utilities.

All model selection remains validation-only. The external geographic test
scene is not accessed by this module.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import torch

try:
    from scipy import ndimage
except Exception as exc:
    raise ImportError(
        "scipy is required for Mask R-CNN connected-component handling."
    ) from exc


DEFAULT_SCORE_THRESHOLD = 0.50
DEFAULT_MASK_THRESHOLD = 0.50


def target_union_mask(
    target: Dict[
        str,
        torch.Tensor,
    ],
) -> np.ndarray:
    masks = target.get(
        "masks"
    )

    if (
        masks is None
        or len(
            masks
        ) == 0
    ):
        return np.zeros(
            (
                1,
                1,
            ),
            dtype=bool,
        )

    return (
        masks
        .detach()
        .cpu()
        .numpy()
        .astype(
            bool
        )
        .any(
            axis=0
        )
    )


def output_union_mask(
    output: Dict[
        str,
        torch.Tensor,
    ],
    shape: Tuple[
        int,
        int,
    ],
    score_threshold: float,
    mask_threshold: float,
) -> np.ndarray:
    pred_mask = np.zeros(
        shape,
        dtype=bool,
    )

    scores = output.get(
        "scores"
    )

    masks = output.get(
        "masks"
    )

    if (
        scores is None
        or masks is None
        or len(
            scores
        ) == 0
    ):
        return pred_mask

    scores_np = (
        scores
        .detach()
        .cpu()
        .numpy()
    )

    masks_np = (
        masks
        .detach()
        .cpu()
        .numpy()[
            :,
            0,
        ]
    )

    for score, mask in zip(
        scores_np,
        masks_np,
    ):
        if float(
            score
        ) >= float(
            score_threshold
        ):
            pred_mask |= (
                mask
                >= float(
                    mask_threshold
                )
            )

    return pred_mask


def remove_small_components(
    binary_mask: np.ndarray,
    min_pixels: int,
) -> np.ndarray:
    min_pixels = int(
        min_pixels
    )

    if min_pixels <= 0:
        return binary_mask.astype(
            bool
        )

    labeled, count = (
        ndimage.label(
            binary_mask.astype(
                bool
            )
        )
    )

    if count == 0:
        return binary_mask.astype(
            bool
        )

    sizes = np.bincount(
        labeled.ravel()
    )

    keep = (
        sizes
        >= min_pixels
    )

    keep[0] = False

    return keep[
        labeled
    ]


def metrics_from_counts(
    tp: int,
    fp: int,
    fn: int,
) -> Dict[
    str,
    float,
]:
    eps = 1e-12

    iou = (
        tp
        / max(
            tp
            + fp
            + fn,
            eps,
        )
    )

    precision = (
        tp
        / max(
            tp
            + fp,
            eps,
        )
    )

    recall = (
        tp
        / max(
            tp
            + fn,
            eps,
        )
    )

    f1 = (
        2
        * precision
        * recall
        / max(
            precision
            + recall,
            eps,
        )
    )

    return {
        "iou":
            float(
                iou
            ),

        "iou_percent":
            float(
                iou
                * 100.0
            ),

        "precision":
            float(
                precision
            ),

        "precision_percent":
            float(
                precision
                * 100.0
            ),

        "recall":
            float(
                recall
            ),

        "recall_percent":
            float(
                recall
                * 100.0
            ),

        "f1":
            float(
                f1
            ),

        "f1_percent":
            float(
                f1
                * 100.0
            ),
    }


@torch.no_grad()
def evaluate_validation(
    model,
    loader,
    device,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
    min_component_pixels: int = 0,
) -> Dict[
    str,
    float,
]:
    model.eval()

    tp = 0
    fp = 0
    fn = 0

    image_count = 0

    for images, targets in loader:
        image = images[
            0
        ].to(
            device,
            non_blocking=True,
        )

        target = targets[
            0
        ]

        h = int(
            image.shape[
                -2
            ]
        )

        w = int(
            image.shape[
                -1
            ]
        )

        output = model(
            [
                image
            ]
        )[0]

        pred = output_union_mask(
            output,
            shape=(
                h,
                w,
            ),
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )

        pred = remove_small_components(
            pred,
            min_component_pixels,
        )

        target_masks = target.get(
            "masks"
        )

        if (
            target_masks is None
            or len(
                target_masks
            ) == 0
        ):
            gt = np.zeros(
                (
                    h,
                    w,
                ),
                dtype=bool,
            )

        else:
            gt = (
                target_masks
                .detach()
                .cpu()
                .numpy()
                .astype(
                    bool
                )
                .any(
                    axis=0
                )
            )

        tp += int(
            np.logical_and(
                pred,
                gt,
            ).sum()
        )

        fp += int(
            np.logical_and(
                pred,
                ~gt,
            ).sum()
        )

        fn += int(
            np.logical_and(
                ~pred,
                gt,
            ).sum()
        )

        image_count += 1

    metrics = metrics_from_counts(
        tp,
        fp,
        fn,
    )

    metrics.update(
        {
            "tp_pixels":
                int(
                    tp
                ),

            "fp_pixels":
                int(
                    fp
                ),

            "fn_pixels":
                int(
                    fn
                ),

            "validation_images":
                int(
                    image_count
                ),

            "score_threshold":
                float(
                    score_threshold
                ),

            "mask_threshold":
                float(
                    mask_threshold
                ),

            "min_component_pixels":
                int(
                    min_component_pixels
                ),
        }
    )

    return metrics


@torch.no_grad()
def validation_pixel_iou(
    model,
    loader,
    device,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    mask_threshold: float = DEFAULT_MASK_THRESHOLD,
) -> float:
    return evaluate_validation(
        model=model,
        loader=loader,
        device=device,
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
        min_component_pixels=0,
    )[
        "iou"
    ]


def search_thresholds(
    model,
    val_loader,
    device,
    score_thresholds: Iterable[
        float
    ],
    mask_thresholds: Iterable[
        float
    ],
) -> Tuple[
    pd.DataFrame,
    dict,
]:
    rows = []

    for score in score_thresholds:
        for mask in mask_thresholds:
            metrics = evaluate_validation(
                model,
                val_loader,
                device,
                score_threshold=float(
                    score
                ),
                mask_threshold=float(
                    mask
                ),
                min_component_pixels=0,
            )

            rows.append(
                metrics
            )

            print(
                f"Score {float(score):.2f} | "
                f"Mask {float(mask):.2f} | "
                f"IoU {metrics['iou_percent']:.2f}% | "
                f"P {metrics['precision_percent']:.2f}% | "
                f"R {metrics['recall_percent']:.2f}%"
            )

    df = pd.DataFrame(
        rows
    )

    idx = df[
        "iou"
    ].idxmax()

    best = df.loc[
        idx
    ].to_dict()

    return (
        df,
        best,
    )


def search_postprocessing(
    model,
    val_loader,
    device,
    score_threshold: float,
    mask_threshold: float,
    area_thresholds_m2: Iterable[
        float
    ],
    pixel_size_m: float,
) -> Tuple[
    pd.DataFrame,
    dict,
]:
    pixel_area = (
        float(
            pixel_size_m
        )
        ** 2
    )

    rows = []

    for area_m2 in area_thresholds_m2:
        min_pixels = (
            0
            if float(
                area_m2
            ) <= 0
            else int(
                math.ceil(
                    float(
                        area_m2
                    )
                    / pixel_area
                )
            )
        )

        metrics = evaluate_validation(
            model,
            val_loader,
            device,
            score_threshold=float(
                score_threshold
            ),
            mask_threshold=float(
                mask_threshold
            ),
            min_component_pixels=min_pixels,
        )

        metrics[
            "min_area_m2"
        ] = float(
            area_m2
        )

        metrics[
            "pixel_size_m"
        ] = float(
            pixel_size_m
        )

        rows.append(
            metrics
        )

        print(
            f"Min Area {float(area_m2):.2f} m2 | "
            f"IoU {metrics['iou_percent']:.2f}% | "
            f"P {metrics['precision_percent']:.2f}% | "
            f"R {metrics['recall_percent']:.2f}%"
        )

    df = pd.DataFrame(
        rows
    )

    idx = df[
        "iou"
    ].idxmax()

    best = df.loc[
        idx
    ].to_dict()

    return (
        df,
        best,
    )


__all__ = [
    "DEFAULT_MASK_THRESHOLD",
    "DEFAULT_SCORE_THRESHOLD",
    "evaluate_validation",
    "metrics_from_counts",
    "output_union_mask",
    "remove_small_components",
    "search_postprocessing",
    "search_thresholds",
    "target_union_mask",
    "validation_pixel_iou",
]
