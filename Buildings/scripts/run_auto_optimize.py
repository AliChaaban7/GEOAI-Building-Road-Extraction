"""
Unified Auto Optimize for ALL three Buildings models:
- U-Net
- DeepLabV3
- Mask R-CNN

Simple professional stages:
    1. Professional Standard
    2. + Augmentation
    3. After Optuna
    4. Optuna + Threshold
    5. Optuna + Post-processing
    6. Select + Freeze champion

The external source-specific test is NOT used during selection.
Use --run-final-test only after the champion has been frozen.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import copy
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone

BUILDINGS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BUILDINGS_ROOT.parent
SCRIPTS_ROOT = BUILDINGS_ROOT / "scripts"
MAIN_CONFIG = BUILDINGS_ROOT / "config" / "experiment.json"

if str(BUILDINGS_ROOT) not in sys.path:
    sys.path.insert(0, str(BUILDINGS_ROOT))

from src.pipeline.model_registry import get_model_spec, normalize_model_type


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run(command, label):
    print("\n" + "=" * 86)
    print(label)
    print("=" * 86)
    print("Command:", " ".join(str(x) for x in command))
    print("=" * 86 + "\n")
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def token(value):
    text = str(value).strip().lower()
    out = "".join(c if c.isalnum() or c in "_-" else "_" for c in text)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def make_config(base, experiment_id, *, aug, optuna, threshold, post):
    cfg = copy.deepcopy(base)
    cfg["experiment_id"] = experiment_id
    cfg["use_augmentation"] = bool(aug)
    cfg["use_optuna"] = bool(optuna)
    cfg["use_threshold_search"] = bool(threshold)
    cfg["use_postprocessing"] = bool(post)
    return cfg


def semantic_threshold_script():
    candidates = [
        SCRIPTS_ROOT / "run_threshold_search.py",
        SCRIPTS_ROOT / "run_augmented_threshold_search.py",
    ]
    for path in candidates:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            if "validation" in text and "validation_threshold_summary" in text:
                return path
    raise FileNotFoundError(
        "Validation-only semantic threshold script not found. "
        "Expected run_threshold_search.py or the validation-only run_augmented_threshold_search.py."
    )


def metric_percent(path):
    payload = load_json(path)
    for key in (
        "best_validation_iou_percent",
        "iou_percent",
        "validation_iou_percent",
        "threshold_050_iou_percent",
    ):
        if payload.get(key) is not None:
            return float(payload[key])
    for key in ("best_validation_iou", "best_val_iou", "iou"):
        if payload.get(key) is not None:
            value = float(payload[key])
            return value * 100.0 if value <= 1.0 else value
    raise KeyError(f"No validation IoU found in {path}")


def copy_snapshot(source, destination):
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def parse_args():
    parser = argparse.ArgumentParser(description="Auto Optimize all Buildings model families.")
    parser.add_argument("--config", type=str, default=str(MAIN_CONFIG))
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--trial-epochs", type=int, default=12)
    parser.add_argument("--final-epochs", type=int, default=40)
    parser.add_argument("--run-final-test", action="store_true")
    parser.add_argument("--overwrite-auto-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    python = sys.executable
    base_path = Path(args.config).resolve()
    base = load_json(base_path)

    model_type = normalize_model_type(base.get("model_type", ""))
    spec = get_model_spec(model_type)
    source = str(base.get("source_type", "")).lower()
    tile = int(base.get("tile_size"))
    if not source:
        raise ValueError("source_type is required")

    run_id = token(args.run_id or f"{model_type}_{tile}_{source}_auto")
    auto_root = BUILDINGS_ROOT / "outputs" / "auto" / run_id
    if auto_root.exists() and not args.overwrite_auto_run:
        raise FileExistsError(
            f"Auto run already exists: {auto_root}\nUse a new --run-id or --overwrite-auto-run."
        )
    auto_root.mkdir(parents=True, exist_ok=True)
    configs = auto_root / "configs"
    snaps = auto_root / "snapshots"
    configs.mkdir(exist_ok=True)
    snaps.mkdir(exist_ok=True)

    ids = {
        "standard": f"{run_id}_standard",
        "augmentation": f"{run_id}_augmentation",
        "optuna": f"{run_id}_optuna",
    }
    cfgs = {
        "standard": make_config(base, ids["standard"], aug=False, optuna=False, threshold=False, post=False),
        "augmentation": make_config(base, ids["augmentation"], aug=True, optuna=False, threshold=False, post=False),
        "optuna": make_config(base, ids["optuna"], aug=False, optuna=True, threshold=True, post=True),
    }
    cfg_paths = {
        name: save_json(cfg, configs / f"{name}.json")
        for name, cfg in cfgs.items()
    }

    summary = {
        "run_id": run_id,
        "started_at_utc": utc_now(),
        "model_type": model_type,
        "model_display_name": spec.display_name,
        "model_family": spec.family,
        "backbone": base.get("backbone"),
        "source_type": source,
        "tile_size": tile,
        "dataset_type": base.get("dataset_type"),
        "external_test_used_during_selection": False,
        "stages": {},
    }
    save_json(summary, auto_root / "auto_summary.json")

    if spec.family == "semantic":
        threshold_script = semantic_threshold_script()
        train_script = SCRIPTS_ROOT / "run_train.py"
        post_script = SCRIPTS_ROOT / "run_postprocess.py"

        # 1. STANDARD
        run([python, str(train_script), "--config", str(cfg_paths["standard"]), "--epochs", str(args.epochs)],
            "AUTO 1/6 — PROFESSIONAL STANDARD")
        run([python, str(threshold_script), "--config", str(cfg_paths["standard"]), "--thresholds", "0.50"],
            "STANDARD — VALIDATION @ 0.50")
        out_std = BUILDINGS_ROOT / "outputs" / "experiments" / ids["standard"]
        std_snap = copy_snapshot(out_std / "validation_threshold_summary.json", snaps / "standard.json")

        # 2. AUGMENTATION
        run([python, str(train_script), "--config", str(cfg_paths["augmentation"]), "--epochs", str(args.epochs)],
            "AUTO 2/6 — + AUGMENTATION")
        run([python, str(threshold_script), "--config", str(cfg_paths["augmentation"]), "--thresholds", "0.50"],
            "AUGMENTATION — VALIDATION @ 0.50")
        out_aug = BUILDINGS_ROOT / "outputs" / "experiments" / ids["augmentation"]
        aug_snap = copy_snapshot(out_aug / "validation_threshold_summary.json", snaps / "augmentation.json")

        # 3. OPTUNA
        run([
            python, str(SCRIPTS_ROOT / "run_optuna.py"),
            "--config", str(cfg_paths["optuna"]),
            "--trials", str(args.trials),
            "--trial-epochs", str(args.trial_epochs),
            "--train-final",
            "--final-epochs", str(args.final_epochs),
        ], "AUTO 3/6 — PROFESSIONAL OPTUNA")
        out_opt = BUILDINGS_ROOT / "outputs" / "experiments" / ids["optuna"]
        run([python, str(threshold_script), "--config", str(cfg_paths["optuna"]), "--thresholds", "0.50"],
            "OPTUNA — VALIDATION @ 0.50")
        opt_snap = copy_snapshot(out_opt / "validation_threshold_summary.json", snaps / "optuna_050.json")

        # 4. THRESHOLD
        run([python, str(threshold_script), "--config", str(cfg_paths["optuna"])],
            "AUTO 4/6 — OPTUNA + THRESHOLD")
        thr_summary = out_opt / "validation_threshold_summary.json"

        # 5. POST
        run([python, str(post_script), "--config", str(cfg_paths["optuna"])],
            "AUTO 5/6 — OPTUNA + THRESHOLD + POST")
        post_summary = out_opt / "validation_postprocess_summary.json"

        optuna_summary = out_opt / "optuna_selection_summary.json"
        if not optuna_summary.exists():
            optuna_summary = out_opt / "run_optuna_summary.json"

        stages = {
            "professional_standard": {
                "validation_iou_percent": metric_percent(std_snap),
                "experiment_output": out_std,
                "config": cfgs["standard"],
                "threshold_summary": std_snap,
                "post_summary": None,
                "optuna_summary": None,
            },
            "augmentation": {
                "validation_iou_percent": metric_percent(aug_snap),
                "experiment_output": out_aug,
                "config": cfgs["augmentation"],
                "threshold_summary": aug_snap,
                "post_summary": None,
                "optuna_summary": None,
            },
            "after_optuna": {
                "validation_iou_percent": metric_percent(opt_snap),
                "experiment_output": out_opt,
                "config": cfgs["optuna"],
                "threshold_summary": opt_snap,
                "post_summary": None,
                "optuna_summary": optuna_summary,
            },
            "optuna_threshold": {
                "validation_iou_percent": metric_percent(thr_summary),
                "experiment_output": out_opt,
                "config": cfgs["optuna"],
                "threshold_summary": thr_summary,
                "post_summary": None,
                "optuna_summary": optuna_summary,
            },
            "optuna_post": {
                "validation_iou_percent": metric_percent(post_summary),
                "experiment_output": out_opt,
                "config": cfgs["optuna"],
                "threshold_summary": thr_summary,
                "post_summary": post_summary,
                "optuna_summary": optuna_summary,
            },
        }

    else:
        # MASK R-CNN / INSTANCE SEGMENTATION
        train_script = SCRIPTS_ROOT / "run_maskrcnn_train.py"
        val_script = SCRIPTS_ROOT / "run_maskrcnn_validation.py"
        optuna_script = SCRIPTS_ROOT / "run_maskrcnn_optuna.py"

        run([python, str(train_script), "--config", str(cfg_paths["standard"]), "--epochs", str(args.epochs)],
            "AUTO 1/6 — MASK R-CNN PROFESSIONAL STANDARD")
        run([python, str(val_script), "--config", str(cfg_paths["standard"]), "--mode", "fixed", "--score-threshold", "0.50", "--mask-threshold", "0.50"],
            "STANDARD — VALIDATION @ SCORE .50 / MASK .50")
        out_std = BUILDINGS_ROOT / "outputs" / "experiments" / ids["standard"]
        std_snap = copy_snapshot(out_std / "validation_fixed_summary.json", snaps / "standard.json")

        run([python, str(train_script), "--config", str(cfg_paths["augmentation"]), "--epochs", str(args.epochs)],
            "AUTO 2/6 — MASK R-CNN + AUGMENTATION")
        run([python, str(val_script), "--config", str(cfg_paths["augmentation"]), "--mode", "fixed", "--score-threshold", "0.50", "--mask-threshold", "0.50"],
            "AUGMENTATION — VALIDATION @ SCORE .50 / MASK .50")
        out_aug = BUILDINGS_ROOT / "outputs" / "experiments" / ids["augmentation"]
        aug_snap = copy_snapshot(out_aug / "validation_fixed_summary.json", snaps / "augmentation.json")

        run([
            python, str(optuna_script), "--config", str(cfg_paths["optuna"]),
            "--trials", str(args.trials),
            "--trial-epochs", str(args.trial_epochs),
            "--train-final",
            "--final-epochs", str(args.final_epochs),
        ], "AUTO 3/6 — MASK R-CNN PROFESSIONAL OPTUNA")
        out_opt = BUILDINGS_ROOT / "outputs" / "experiments" / ids["optuna"]
        run([python, str(val_script), "--config", str(cfg_paths["optuna"]), "--mode", "fixed", "--score-threshold", "0.50", "--mask-threshold", "0.50"],
            "OPTUNA — VALIDATION @ SCORE .50 / MASK .50")
        opt_snap = copy_snapshot(out_opt / "validation_fixed_summary.json", snaps / "optuna_050.json")

        run([python, str(val_script), "--config", str(cfg_paths["optuna"]), "--mode", "threshold"],
            "AUTO 4/6 — MASK R-CNN OPTUNA + THRESHOLD")
        thr_summary = out_opt / "validation_threshold_summary.json"

        run([python, str(val_script), "--config", str(cfg_paths["optuna"]), "--mode", "post"],
            "AUTO 5/6 — MASK R-CNN OPTUNA + POST")
        post_summary = out_opt / "validation_postprocess_summary.json"
        optuna_summary = out_opt / "optuna_selection_summary.json"

        stages = {
            "professional_standard": {
                "validation_iou_percent": metric_percent(std_snap),
                "experiment_output": out_std,
                "config": cfgs["standard"],
                "threshold_summary": None,
                "post_summary": None,
                "optuna_summary": None,
                "fixed_score": 0.50,
                "fixed_mask": 0.50,
            },
            "augmentation": {
                "validation_iou_percent": metric_percent(aug_snap),
                "experiment_output": out_aug,
                "config": cfgs["augmentation"],
                "threshold_summary": None,
                "post_summary": None,
                "optuna_summary": None,
                "fixed_score": 0.50,
                "fixed_mask": 0.50,
            },
            "after_optuna": {
                "validation_iou_percent": metric_percent(opt_snap),
                "experiment_output": out_opt,
                "config": cfgs["optuna"],
                "threshold_summary": None,
                "post_summary": None,
                "optuna_summary": optuna_summary,
                "fixed_score": 0.50,
                "fixed_mask": 0.50,
            },
            "optuna_threshold": {
                "validation_iou_percent": metric_percent(thr_summary),
                "experiment_output": out_opt,
                "config": cfgs["optuna"],
                "threshold_summary": thr_summary,
                "post_summary": None,
                "optuna_summary": optuna_summary,
            },
            "optuna_post": {
                "validation_iou_percent": metric_percent(post_summary),
                "experiment_output": out_opt,
                "config": cfgs["optuna"],
                "threshold_summary": thr_summary,
                "post_summary": post_summary,
                "optuna_summary": optuna_summary,
            },
        }

    # ========================================================
    # SELECT CHAMPION ON VALIDATION
    # ========================================================
    best_stage = max(stages, key=lambda k: stages[k]["validation_iou_percent"])
    selected = stages[best_stage]
    selected_cfg = copy.deepcopy(selected["config"])
    selected_cfg["auto_selected_stage"] = best_stage
    selected_cfg["auto_run_id"] = run_id
    selected_cfg_path = save_json(selected_cfg, configs / "final_selected.json")

    if spec.family == "semantic":
        freeze = [
            python, str(SCRIPTS_ROOT / "freeze_final_settings.py"),
            "--config", str(selected_cfg_path),
            "--threshold-summary", str(selected["threshold_summary"]),
        ]
        if selected["post_summary"] is not None:
            freeze += ["--postprocess-summary", str(selected["post_summary"])]
        if selected["optuna_summary"] is not None and Path(selected["optuna_summary"]).exists():
            freeze += ["--optuna-summary", str(selected["optuna_summary"])]
    else:
        freeze = [
            python, str(SCRIPTS_ROOT / "freeze_maskrcnn_settings.py"),
            "--config", str(selected_cfg_path),
        ]
        if selected.get("threshold_summary") is not None:
            freeze += ["--threshold-summary", str(selected["threshold_summary"])]
        else:
            freeze += ["--score-threshold", str(selected.get("fixed_score", 0.50)),
                       "--mask-threshold", str(selected.get("fixed_mask", 0.50))]
        if selected.get("post_summary") is not None:
            freeze += ["--postprocess-summary", str(selected["post_summary"])]
        if selected.get("optuna_summary") is not None and Path(selected["optuna_summary"]).exists():
            freeze += ["--optuna-summary", str(selected["optuna_summary"])]

    run(freeze, "AUTO 6/6 — FREEZE CHAMPION")
    final_settings = Path(selected["experiment_output"]) / "final_settings.json"

    # UI-ready stage summary
    labels = {
        "professional_standard": "Professional Standard",
        "augmentation": "+ Augmentation",
        "after_optuna": "After Optuna",
        "optuna_threshold": "Optuna + Threshold",
        "optuna_post": "Optuna + Post-processing",
    }
    summary["stages"] = {
        key: {
            "label": labels[key],
            "validation_iou_percent": float(stages[key]["validation_iou_percent"]),
            "selected": key == best_stage,
        }
        for key in labels
    }
    summary["best_stage"] = best_stage
    summary["best_validation_iou_percent"] = float(selected["validation_iou_percent"])
    summary["selected_experiment_id"] = selected_cfg["experiment_id"]
    summary["selected_config"] = str(selected_cfg_path)
    summary["final_settings"] = str(final_settings)
    summary["settings_frozen"] = True

    if args.run_final_test:
        if spec.family == "semantic":
            final_command = [
                python, str(SCRIPTS_ROOT / "run_final_test.py"),
                "--config", str(selected_cfg_path),
                "--settings", str(final_settings),
            ]
        else:
            final_command = [
                python, str(SCRIPTS_ROOT / "run_maskrcnn_final_test.py"),
                "--config", str(selected_cfg_path),
                "--settings", str(final_settings),
            ]
        run(final_command, "AUTO FINAL — EXPLICIT SOURCE-SPECIFIC TEST")
        metrics_path = Path(selected["experiment_output"]) / "final_test" / "final_metrics.json"
        summary["final_test"] = {
            "completed": True,
            "metrics_path": str(metrics_path),
            "metrics": load_json(metrics_path),
        }
    else:
        summary["final_test"] = {
            "completed": False,
            "ready": True,
            "note": "Champion selected and frozen. External test intentionally not run.",
        }

    summary["finished_at_utc"] = utc_now()
    save_json(summary, auto_root / "auto_summary.json")
    save_json({
        "run_id": run_id,
        "model_type": model_type,
        "model_family": spec.family,
        "source_type": source,
        "tile_size": tile,
        "results": list(summary["stages"].values()),
        "best_stage": best_stage,
        "best_validation_iou_percent": summary["best_validation_iou_percent"],
        "final_test": summary["final_test"],
    }, auto_root / "ui_results.json")

    print("\n" + "=" * 86)
    print("AUTO OPTIMIZE FINISHED")
    print("=" * 86)
    print(f"Model: {spec.display_name}")
    print(f"Family: {spec.family}")
    print(f"Source: {source}")
    print(f"Tile Size: {tile}")
    print("-" * 86)
    for key in labels:
        marker = "  <-- CHAMPION" if key == best_stage else ""
        print(f"{labels[key]:30s}: {stages[key]['validation_iou_percent']:6.2f}%{marker}")
    print("-" * 86)
    print(f"Best Stage: {labels[best_stage]}")
    print(f"Best Validation IoU: {summary['best_validation_iou_percent']:.2f}%")
    print("Settings Frozen: YES")
    print("External Test:", "COMPLETED" if args.run_final_test else "NOT RUN (ready)")
    print(f"UI Results: {auto_root / 'ui_results.json'}")
    print("=" * 86 + "\n")


if __name__ == "__main__":
    main()
