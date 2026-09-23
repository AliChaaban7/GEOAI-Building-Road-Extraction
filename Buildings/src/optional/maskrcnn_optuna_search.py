"""
Model-aware Mask R-CNN Optuna workflow.

Uses the same shared Mask R-CNN engine as professional standard training.
The external geographic test scene is never accessed.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Dict

import optuna
import pandas as pd
import torch

from src.instance.maskrcnn_engine import (
    make_loaders,
    create_maskrcnn_model,
    make_optimizer,
    make_scheduler,
    train_one_epoch,
    validation_pixel_iou,
    save_checkpoint,
    seed_everything,
    train_model as train_maskrcnn_model,
)


def sample_params(trial: optuna.Trial) -> Dict:
    optimizer = trial.suggest_categorical(
        "optimizer", ["adamw", "sgd"]
    )
    params = {
        "optimizer": optimizer,
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 3e-4, log=True
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay", 1e-6, 1e-3, log=True
        ),
        "scheduler": trial.suggest_categorical(
            "scheduler", ["none", "step", "cosine"]
        ),
        "momentum": 0.9,
    }
    if optimizer == "sgd":
        params["momentum"] = trial.suggest_float(
            "momentum", 0.80, 0.95
        )
    return params


def run_trial(
    trial,
    train_pairs,
    val_pairs,
    device,
    output_folder: Path,
    epochs: int,
    batch_size: int,
    pretrained: bool,
    seed: int,
):
    seed_everything(seed)
    params = sample_params(trial)
    trial_folder = Path(output_folder) / "trials" / f"trial_{trial.number:03d}"
    trial_folder.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = make_loaders(
        train_pairs,
        val_pairs,
        batch_size=batch_size,
        train_augmentation=None,
    )

    model = create_maskrcnn_model(
        pretrained=pretrained
    ).to(device)
    optimizer = make_optimizer(
        model,
        optimizer_name=params["optimizer"],
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        momentum=params["momentum"],
    )
    scheduler = make_scheduler(
        optimizer,
        scheduler_name=params["scheduler"],
        epochs=epochs,
    )

    best_iou = -1.0
    best_epoch = -1
    history = []

    print("\n" + "-" * 70)
    print(f"MASK R-CNN OPTUNA TRIAL {trial.number}")
    print("-" * 70)
    for key, value in params.items():
        print(f"{key}: {value}")
    print("Objective Dataset: VALIDATION ONLY")
    print("Augmentation: OFF")
    print("External Test Used: NO")
    print("-" * 70)

    for epoch in range(1, int(epochs) + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_iou = validation_pixel_iou(
            model, val_loader, device,
            score_threshold=0.50,
            mask_threshold=0.50,
        )
        if scheduler is not None:
            scheduler.step()
        current_lr = float(optimizer.param_groups[0]["lr"])
        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_iou": float(val_iou),
            "val_iou_percent": float(val_iou * 100.0),
            "learning_rate": current_lr,
        })
        print(
            f"Epoch {epoch:03d}/{epochs} | Loss {train_loss:.4f} | "
            f"Val IoU {val_iou * 100:.2f}%"
        )
        if val_iou > best_iou:
            best_iou = float(val_iou)
            best_epoch = int(epoch)
            save_checkpoint(
                trial_folder / "best_model.pth",
                model,
                epoch,
                val_iou,
                params,
                extra_meta={
                    "phase": "optuna_trial",
                    "trial_number": int(trial.number),
                },
            )
        pd.DataFrame(history).to_csv(
            trial_folder / "training_log.csv", index=False
        )

    summary = {
        "trial": int(trial.number),
        "best_epoch": best_epoch,
        "best_validation_iou": best_iou,
        "best_validation_iou_percent": best_iou * 100.0,
        "params": params,
        "external_test_used": False,
    }
    (trial_folder / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    del model, optimizer
    if scheduler is not None:
        del scheduler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_iou


def run_maskrcnn_optuna_workflow(
    train_pairs,
    val_pairs,
    output_folder: Path,
    study_name: str,
    n_trials: int = 10,
    trial_epochs: int = 12,
    train_final: bool = False,
    final_epochs: int = 40,
    batch_size: int = 1,
    pretrained: bool = True,
    seed: int = 42,
) -> Dict:
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)

    storage = f"sqlite:///{(output_folder / 'optuna_study.db').as_posix()}"
    study = optuna.create_study(
        direction="maximize",
        study_name=str(study_name),
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    remaining = max(0, int(n_trials) - len(completed))

    print("\n========== MASK R-CNN OPTUNA START ==========")
    print(f"Study Name: {study_name}")
    print(f"Train Samples: {len(train_pairs)}")
    print(f"Validation Samples: {len(val_pairs)}")
    print(f"Trials Target: {n_trials}")
    print(f"Trial Epochs: {trial_epochs}")
    print("Augmentation: OFF")
    print("Objective: validation pixel IoU @ score=.50, mask=.50")
    print("External Test Used: NO")
    print("=============================================\n")

    if remaining > 0:
        study.optimize(
            lambda trial: run_trial(
                trial,
                train_pairs=train_pairs,
                val_pairs=val_pairs,
                device=device,
                output_folder=output_folder,
                epochs=trial_epochs,
                batch_size=batch_size,
                pretrained=pretrained,
                seed=seed,
            ),
            n_trials=remaining,
            gc_after_trial=True,
        )

    if not any(
        t.state == optuna.trial.TrialState.COMPLETE
        for t in study.trials
    ):
        raise RuntimeError("Optuna completed without a successful trial.")

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    best_params.setdefault("momentum", 0.9)

    study.trials_dataframe().to_csv(
        output_folder / "trials.csv", index=False
    )

    best_payload = {
        "study_name": str(study_name),
        "best_trial": int(best_trial.number),
        "best_validation_iou": float(best_trial.value),
        "best_validation_iou_percent": float(best_trial.value * 100.0),
        "best_params": best_params,
        "external_test_used": False,
    }
    (output_folder / "best_params.json").write_text(
        json.dumps(best_payload, indent=2), encoding="utf-8"
    )

    final_summary = None
    if train_final:
        final_summary = train_maskrcnn_model(
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            output_folder=output_folder,
            epochs=final_epochs,
            batch_size=batch_size,
            optimizer_name=best_params["optimizer"],
            learning_rate=float(best_params["learning_rate"]),
            weight_decay=float(best_params["weight_decay"]),
            scheduler_name=best_params["scheduler"],
            momentum=float(best_params.get("momentum", 0.9)),
            pretrained=pretrained,
            seed=seed,
            train_augmentation=None,
            phase="final_optuna_model",
        )

    summary = {
        **best_payload,
        "train_final": bool(train_final),
        "final_training": final_summary,
        "output_folder": str(output_folder),
    }
    (output_folder / "optuna_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
