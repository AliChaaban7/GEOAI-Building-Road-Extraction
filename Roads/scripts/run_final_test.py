"""
Independent final Road testing.

Sources:
- aerial
- satellite
- drone

All sources use:
- same trained checkpoint
- same threshold
- same Road post-processing settings

No source-specific tuning is allowed.

Final metrics:
- strict polygon IoU
- Precision
- Recall
- F1

clDice remains a secondary raster/connectivity metric and can be added
to the resulting final metrics by the Road connectivity evaluator when
the aligned final-test mask is available.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import sys


ROAD_ROOT = Path(__file__).resolve().parents[1]

if str(ROAD_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROAD_ROOT),
    )


from src.gis_inference import (
    run_georeferenced_raster_inference,
)

from src.pipeline import prepare_experiment


def parse_arguments():

    parser = argparse.ArgumentParser(
        description="Run frozen independent Road final test."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--source",
        choices=(
            "aerial",
            "satellite",
            "drone",
            "all",
        ),
        default="satellite",
    )

    parser.add_argument(
        "--settings",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--allow-rerun",
        action="store_true",
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


def get_arcpy():

    try:

        import arcpy

    except Exception as exc:

        raise RuntimeError(
            "Final GIS test must run in an ArcGIS Pro "
            "Python environment."
        ) from exc

    return arcpy


def spatial_reference_equal(
    first,
    second,
):

    if first is None or second is None:
        return False

    try:

        if (
            first.factoryCode
            and second.factoryCode
        ):

            return (
                int(
                    first.factoryCode
                )
                ==
                int(
                    second.factoryCode
                )
            )

    except Exception:
        pass

    return (
        str(
            first.exportToString()
        )
        ==
        str(
            second.exportToString()
        )
    )


def project_ground_truth_if_needed(
    ground_truth,
    raster_path,
    output_gdb,
    source,
):

    arcpy = get_arcpy()

    raster_sr = arcpy.Raster(
        raster_path
    ).spatialReference

    gt_sr = arcpy.Describe(
        ground_truth
    ).spatialReference

    output = os.path.join(
        str(
            output_gdb
        ),
        f"{source}_ground_truth_projected",
    )

    if arcpy.Exists(
        output
    ):

        arcpy.management.Delete(
            output
        )

    if spatial_reference_equal(
        gt_sr,
        raster_sr,
    ):

        arcpy.management.CopyFeatures(
            ground_truth,
            output,
        )

    else:

        arcpy.management.Project(
            ground_truth,
            output,
            raster_sr,
        )

    return output


def feature_count(
    feature_class,
):

    arcpy = get_arcpy()

    return int(
        arcpy.management.GetCount(
            feature_class
        )[0]
    )


def feature_area(
    feature_class,
):

    arcpy = get_arcpy()

    if feature_count(
        feature_class
    ) == 0:

        return 0.0

    return sum(
        float(
            row[0]
        )

        for row
        in arcpy.da.SearchCursor(
            feature_class,
            [
                "SHAPE@AREA",
            ],
        )
    )


def strict_polygon_metrics(
    prediction,
    ground_truth,
    workspace,
):

    arcpy = get_arcpy()

    prediction_count = feature_count(
        prediction
    )

    ground_truth_count = feature_count(
        ground_truth
    )

    prediction_area = feature_area(
        prediction
    )

    ground_truth_area = feature_area(
        ground_truth
    )

    intersection = os.path.join(
        str(
            workspace
        ),
        "_road_final_intersection",
    )

    if arcpy.Exists(
        intersection
    ):

        arcpy.management.Delete(
            intersection
        )

    intersection_count = 0
    intersection_area = 0.0

    try:

        if (
            prediction_count > 0
            and ground_truth_count > 0
        ):

            arcpy.analysis.Intersect(
                [
                    prediction,
                    ground_truth,
                ],
                intersection,
            )

            intersection_count = (
                feature_count(
                    intersection
                )
            )

            intersection_area = (
                feature_area(
                    intersection
                )
            )

    finally:

        if arcpy.Exists(
            intersection
        ):

            arcpy.management.Delete(
                intersection
            )

    union_area = (
        prediction_area
        + ground_truth_area
        - intersection_area
    )

    iou = (
        intersection_area
        / union_area
        if union_area > 0
        else 0.0
    )

    precision = (
        intersection_area
        / prediction_area
        if prediction_area > 0
        else 0.0
    )

    recall = (
        intersection_area
        / ground_truth_area
        if ground_truth_area > 0
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
        if precision
        + recall
        > 0
        else 0.0
    )

    return {
        "prediction_count":
            prediction_count,

        "ground_truth_count":
            ground_truth_count,

        "intersection_count":
            intersection_count,

        "prediction_area":
            prediction_area,

        "ground_truth_area":
            ground_truth_area,

        "intersection_area":
            intersection_area,

        "union_area":
            union_area,

        "iou":
            iou,

        "iou_percent":
            iou
            * 100.0,

        "precision":
            precision,

        "precision_percent":
            precision
            * 100.0,

        "recall":
            recall,

        "recall_percent":
            recall
            * 100.0,

        "f1":
            f1,

        "f1_percent":
            f1
            * 100.0,
    }


def ensure_file_gdb(
    folder,
):

    arcpy = get_arcpy()

    folder = Path(
        folder
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    gdb = (
        folder
        / "final_prediction.gdb"
    )

    if not arcpy.Exists(
        str(
            gdb
        )
    ):

        arcpy.management.CreateFileGDB(
            str(
                folder
            ),
            gdb.name,
        )

    return gdb


def utc_now_iso():
    """
    Return one timezone-aware UTC timestamp.

    The explicit module alias avoids the datetime class/module name collision
    that caused the previous final-test crash.
    """

    return (
        dt.datetime.now(
            dt.timezone.utc
        ).isoformat()
    )


def same_path_reference(
    first,
    second,
):
    """
    Compare two path-like references without requiring either path to exist.
    """

    if first is None or second is None:
        return False

    return (
        os.path.normcase(
            os.path.normpath(
                str(
                    first
                )
            )
        )
        ==
        os.path.normcase(
            os.path.normpath(
                str(
                    second
                )
            )
        )
    )


def write_final_test_receipt(
    receipt_path,
    source,
    metrics_path,
    checkpoint,
    threshold,
    postprocessing,
    recovery_mode=None,
):
    """
    Write the source-specific final-test receipt.

    Recovery never changes the frozen checkpoint, threshold, or Road
    post-processing settings.
    """

    receipt = {
        "source":
            source,

        "completed":
            True,

        "metrics_path":
            str(
                metrics_path
            ),

        "checkpoint":
            checkpoint,

        "threshold":
            float(
                threshold
            ),

        "postprocessing":
            postprocessing,

        "retuning_after_test":
            False,

        "timestamp_utc":
            utc_now_iso(),

        "recovery_mode":
            recovery_mode,

        "external_test_inference_rerun":
            False
            if recovery_mode
            else None,
    }

    Path(
        receipt_path
    ).write_text(
        json.dumps(
            receipt,
            indent=2,
        ),
        encoding="utf-8",
    )

    return receipt


def build_final_result(
    source_config,
    source,
    raster,
    ground_truth,
    projected_gt,
    checkpoint,
    threshold,
    postprocessing,
    prediction_fc,
    metrics,
    binary_raster=None,
    probability_raster=None,
    recovery_mode=None,
):
    """
    Build the per-source final metrics payload.
    """

    return {
        "source":
            source,

        "native_resolution_m":
            source_config.get(
                "native_resolution_m"
            ),

        "test_raster":
            raster,

        "ground_truth":
            ground_truth,

        "ground_truth_projected":
            projected_gt,

        "checkpoint":
            checkpoint,

        "threshold":
            float(
                threshold
            ),

        "postprocessing":
            postprocessing,

        "prediction_fc":
            prediction_fc,

        "binary_raster":
            binary_raster,

        "probability_raster":
            probability_raster,

        **metrics,

        "cldice":
            None,

        "cldice_note":
            (
                "Secondary clDice requires an aligned rasterized "
                "Ground Truth mask. It is intentionally not fabricated "
                "from polygon-area overlap."
            ),

        "per_source_retuning":
            False,

        "settings_frozen":
            True,

        "completed_at_utc":
            utc_now_iso(),

        "recovery_mode":
            recovery_mode,

        "external_test_inference_rerun":
            False
            if recovery_mode
            else None,
    }


def recover_from_existing_metrics(
    metrics_path,
    receipt_path,
    arcpy,
    source,
    checkpoint,
    threshold,
    postprocessing,
):
    """
    Safe recovery when final_metrics.json exists but receipt writing failed.
    """

    metrics_path = Path(
        metrics_path
    )

    if not metrics_path.exists():
        return None

    existing = load_json(
        metrics_path
    )

    required = (
        "source",
        "checkpoint",
        "threshold",
        "prediction_fc",
        "iou_percent",
        "precision_percent",
        "recall_percent",
        "f1_percent",
    )

    if any(
        key not in existing
        for key in required
    ):
        raise RuntimeError(
            "\nA partial final_metrics.json exists but is incomplete:\n"
            f"{metrics_path}\n\n"
            "The final test will not be rerun automatically."
        )

    settings_match = (
        str(
            existing.get(
                "source"
            )
        ).lower()
        ==
        str(
            source
        ).lower()

        and same_path_reference(
            existing.get(
                "checkpoint"
            ),
            checkpoint,
        )

        and abs(
            float(
                existing.get(
                    "threshold"
                )
            )
            - float(
                threshold
            )
        )
        < 1e-12

        and existing.get(
            "postprocessing"
        )
        == postprocessing
    )

    prediction_fc = existing.get(
        "prediction_fc"
    )

    if not settings_match:
        raise RuntimeError(
            "\nExisting final metrics do not match the currently frozen "
            f"settings for {source}.\n"
            "Refusing automatic reuse or rerun."
        )

    if not arcpy.Exists(
        prediction_fc
    ):
        raise FileNotFoundError(
            "\nExisting final metrics reference a missing prediction:\n"
            f"{prediction_fc}"
        )

    existing[
        "recovery_mode"
    ] = "existing_final_metrics"

    existing[
        "external_test_inference_rerun"
    ] = False

    metrics_path.write_text(
        json.dumps(
            existing,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_final_test_receipt(
        receipt_path=receipt_path,
        source=source,
        metrics_path=metrics_path,
        checkpoint=checkpoint,
        threshold=threshold,
        postprocessing=postprocessing,
        recovery_mode="existing_final_metrics",
    )

    print()
    print(
        "=" * 72
    )

    print(
        f"RECOVERED FINAL ROAD TEST — {source.upper()}"
    )

    print(
        "=" * 72
    )

    print(
        "Existing final metrics were validated."
    )

    print(
        "External raster inference was NOT rerun."
    )

    print(
        f"IoU       : {float(existing['iou_percent']):.2f}%"
    )

    print(
        f"Precision : {float(existing['precision_percent']):.2f}%"
    )

    print(
        f"Recall    : {float(existing['recall_percent']):.2f}%"
    )

    print(
        f"F1        : {float(existing['f1_percent']):.2f}%"
    )

    print(
        f"Receipt   : {receipt_path}"
    )

    print(
        "=" * 72
    )

    return existing


def recover_from_existing_prediction(
    source_config,
    source_folder,
    receipt_path,
    output_gdb,
    source,
    raster,
    ground_truth,
    checkpoint,
    threshold,
    postprocessing,
    output_name,
):
    """
    Recover the exact failure case where tiled inference and polygon output
    completed, but Python crashed before final_metrics.json and the receipt
    were written.

    The existing prediction is reused. External raster model inference is
    NOT repeated.
    """

    arcpy = get_arcpy()

    expected_prediction_fc = os.path.join(
        str(
            output_gdb
        ),
        output_name,
    )

    if not arcpy.Exists(
        expected_prediction_fc
    ):
        return None

    print()
    print(
        "=" * 72
    )

    print(
        f"PARTIAL FINAL-TEST RECOVERY — {source.upper()}"
    )

    print(
        "=" * 72
    )

    print(
        "A completed prediction feature class already exists:"
    )

    print(
        expected_prediction_fc
    )

    print(
        "The external raster inference will NOT be repeated."
    )

    print(
        "Only final evaluation metadata / receipt will be rebuilt."
    )

    print(
        "=" * 72
    )

    projected_gt = (
        project_ground_truth_if_needed(
            ground_truth=ground_truth,

            raster_path=raster,

            output_gdb=output_gdb,

            source=source,
        )
    )

    metrics = strict_polygon_metrics(
        prediction=expected_prediction_fc,

        ground_truth=projected_gt,

        workspace=output_gdb,
    )

    result = build_final_result(
        source_config=source_config,
        source=source,
        raster=raster,
        ground_truth=ground_truth,
        projected_gt=projected_gt,
        checkpoint=checkpoint,
        threshold=threshold,
        postprocessing=postprocessing,
        prediction_fc=expected_prediction_fc,
        metrics=metrics,
        binary_raster=None,
        probability_raster=None,
        recovery_mode="existing_prediction_after_interrupted_run",
    )

    metrics_path = (
        Path(
            source_folder
        )
        / "final_metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_final_test_receipt(
        receipt_path=receipt_path,
        source=source,
        metrics_path=metrics_path,
        checkpoint=checkpoint,
        threshold=threshold,
        postprocessing=postprocessing,
        recovery_mode="existing_prediction_after_interrupted_run",
    )

    print()
    print(
        f"Recovered IoU       : {float(result['iou_percent']):.2f}%"
    )

    print(
        f"Recovered Precision : {float(result['precision_percent']):.2f}%"
    )

    print(
        f"Recovered Recall    : {float(result['recall_percent']):.2f}%"
    )

    print(
        f"Recovered F1        : {float(result['f1_percent']):.2f}%"
    )

    print(
        f"Metrics             : {metrics_path}"
    )

    print(
        f"Receipt             : {receipt_path}"
    )

    return result


def run_source(
    context,
    paths_config,
    frozen,
    source,
    device,
    allow_rerun,
):

    arcpy = get_arcpy()

    source_config = (
        paths_config[
            "final_tests"
        ][
            source
        ]
    )

    if not bool(
        source_config.get(
            "enabled",
            False,
        )
    ):

        raise RuntimeError(
            f"{source} final test is disabled."
        )

    raster = source_config.get(
        "raster"
    )

    ground_truth = source_config.get(
        "ground_truth"
    )

    if not raster or not ground_truth:

        raise RuntimeError(
            f"\n{source.title()} final-test paths are not complete.\n\n"
            f"Raster       : {raster}\n"
            f"Ground Truth : {ground_truth}"
        )

    if not arcpy.Exists(
        raster
    ):

        raise FileNotFoundError(
            f"Final raster not found:\n{raster}"
        )

    if not arcpy.Exists(
        ground_truth
    ):

        raise FileNotFoundError(
            f"Ground Truth not found:\n{ground_truth}"
        )

    experiment_output = Path(
        context[
            "output_folder"
        ]
    )

    source_folder = (
        experiment_output
        / "final_test"
        / source
    )

    source_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    receipt_path = (
        source_folder
        / "final_test_receipt.json"
    )

    metrics_path = (
        source_folder
        / "final_metrics.json"
    )

    if (
        receipt_path.exists()
        and not allow_rerun
    ):

        raise RuntimeError(
            f"\nFinal test already completed for {source}.\n"
            "Use --allow-rerun only for deliberate debugging."
        )

    output_gdb = ensure_file_gdb(
        source_folder
    )

    checkpoint = frozen[
        "checkpoint"
    ]

    threshold = float(
        frozen[
            "probability_threshold"
        ]
    )

    postprocessing = (
        frozen.get(
            "postprocessing"
        )
        if frozen.get(
            "postprocessing_enabled",
            False,
        )
        else None
    )

    output_name = (
        f"road_prediction_{source}"
    )

    # ---------------------------------------------------------
    # SAFE RECOVERY 1
    # final_metrics.json exists but receipt writing failed.
    # ---------------------------------------------------------

    if not allow_rerun:

        recovered_metrics = (
            recover_from_existing_metrics(
                metrics_path=metrics_path,
                receipt_path=receipt_path,
                arcpy=arcpy,
                source=source,
                checkpoint=checkpoint,
                threshold=threshold,
                postprocessing=postprocessing,
            )
        )

        if recovered_metrics is not None:
            return recovered_metrics

    # ---------------------------------------------------------
    # SAFE RECOVERY 2
    # Prediction polygon exists after completed tiled inference, but
    # final metrics / receipt writing was interrupted.
    # ---------------------------------------------------------

    if not allow_rerun:

        recovered_prediction = (
            recover_from_existing_prediction(
                source_config=source_config,
                source_folder=source_folder,
                receipt_path=receipt_path,
                output_gdb=output_gdb,
                source=source,
                raster=raster,
                ground_truth=ground_truth,
                checkpoint=checkpoint,
                threshold=threshold,
                postprocessing=postprocessing,
                output_name=output_name,
            )
        )

        if recovered_prediction is not None:
            return recovered_prediction

    # ---------------------------------------------------------
    # NORMAL ONE-TIME INFERENCE
    # ---------------------------------------------------------

    inference = (
        run_georeferenced_raster_inference(
            context=context,

            raster_path=raster,

            checkpoint_path=checkpoint,

            threshold=threshold,

            output_folder=source_folder,

            output_geodatabase=output_gdb,

            output_name=output_name,

            postprocess_config=postprocessing,

            device=device,

            tile_size=None,

            overlap=None,

            save_probability_raster=True,

            polygonize=True,
        )
    )

    prediction_fc = inference[
        "prediction_fc"
    ]

    projected_gt = (
        project_ground_truth_if_needed(
            ground_truth=ground_truth,

            raster_path=raster,

            output_gdb=output_gdb,

            source=source,
        )
    )

    metrics = strict_polygon_metrics(
        prediction=prediction_fc,

        ground_truth=projected_gt,

        workspace=output_gdb,
    )

    result = build_final_result(
        source_config=source_config,
        source=source,
        raster=raster,
        ground_truth=ground_truth,
        projected_gt=projected_gt,
        checkpoint=checkpoint,
        threshold=threshold,
        postprocessing=postprocessing,
        prediction_fc=prediction_fc,
        metrics=metrics,
        binary_raster=inference.get(
            "binary_raster"
        ),
        probability_raster=inference.get(
            "probability_raster"
        ),
        recovery_mode=None,
    )

    metrics_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_final_test_receipt(
        receipt_path=receipt_path,
        source=source,
        metrics_path=metrics_path,
        checkpoint=checkpoint,
        threshold=threshold,
        postprocessing=postprocessing,
        recovery_mode=None,
    )

    return result

def main():

    args = parse_arguments()

    context = prepare_experiment(
        config_path=args.config,
        force_regenerate_manifests=False,
    )

    if context[
        "model_backend"
    ] != "pytorch":

        raise RuntimeError(
            "\nThis final-test launcher currently uses the native "
            "PyTorch GIS inference engine.\n"
            "ConnectNet/MultiTaskRoadExtractor need their "
            "ArcGIS Learn inference path."
        )

    output_folder = Path(
        context[
            "output_folder"
        ]
    )

    settings_path = (
        Path(
            args.settings
        )
        if args.settings
        else (
            output_folder
            / "final_settings.json"
        )
    )

    if not settings_path.exists():

        raise FileNotFoundError(
            "\nFrozen final settings not found:\n"
            f"{settings_path}\n\n"
            "Run freeze_final_settings.py first."
        )

    frozen = load_json(
        settings_path
    )

    if not bool(
        frozen.get(
            "settings_frozen",
            False,
        )
    ):

        raise RuntimeError(
            "Settings are not frozen. Final test blocked."
        )

    paths_config = load_json(
        ROAD_ROOT
        / "config"
        / "paths.json"
    )

    sources = (
        [
            "aerial",
            "satellite",
            "drone",
        ]
        if args.source == "all"
        else [
            args.source,
        ]
    )

    results = {}

    for source in sources:

        print()
        print(
            "=" * 72
        )

        print(
            f"FINAL ROAD TEST — {source.upper()}"
        )

        print(
            "=" * 72
        )

        results[
            source
        ] = run_source(
            context=context,

            paths_config=paths_config,

            frozen=frozen,

            source=source,

            device=args.device,

            allow_rerun=args.allow_rerun,
        )

    combined_path = (
        output_folder
        / "final_test"
        / "final_test_summary.json"
    )

    combined_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_path.write_text(
        json.dumps(
            {
                "same_frozen_settings":
                    True,

                "per_source_retuning":
                    False,

                "results":
                    results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Final summary: {combined_path}"
    )


if __name__ == "__main__":
    main()