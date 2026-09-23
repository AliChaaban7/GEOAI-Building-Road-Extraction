"""
train_instance.py

Training utilities for Mask R-CNN instance segmentation.

Used for:
- Mask R-CNN building extraction
- RCNN tile datasets
"""

from pathlib import Path
import json
import time

import pandas as pd
import torch


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    return obj


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2)


def move_targets_to_device(targets, device):
    moved_targets = []

    for target in targets:
        moved_target = {}

        for key, value in target.items():
            if torch.is_tensor(value):
                moved_target[key] = value.to(device)
            else:
                moved_target[key] = value

        moved_targets.append(moved_target)

    return moved_targets


def get_training_value(model_config, key, default):
    if isinstance(model_config, dict):
        if "training" in model_config and key in model_config["training"]:
            return model_config["training"][key]

        if key in model_config:
            return model_config[key]

    return default


def calculate_total_loss(loss_dict):
    total_loss = sum(loss for loss in loss_dict.values())
    return total_loss


def train_one_epoch(model, data_loader, optimizer, device, epoch):
    model.train()

    running_loss = 0.0
    running_losses = {}

    valid_batches = 0

    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader, start=1):
        # Skip empty target batches
        if all(target["boxes"].numel() == 0 for target in targets):
            continue

        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        total_loss = calculate_total_loss(loss_dict)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: invalid loss at batch {batch_idx}. Skipping batch.")
            continue

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        optimizer.step()

        running_loss += float(total_loss.detach().cpu())
        valid_batches += 1

        for key, value in loss_dict.items():
            running_losses.setdefault(key, 0.0)
            running_losses[key] += float(value.detach().cpu())

        if batch_idx % 10 == 0:
            avg_loss = running_loss / max(valid_batches, 1)
            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx}/{len(data_loader)} | "
                f"Avg Loss: {avg_loss:.4f}"
            )

    avg_total_loss = running_loss / max(valid_batches, 1)

    avg_losses = {
        key: value / max(valid_batches, 1)
        for key, value in running_losses.items()
    }

    elapsed = time.time() - start_time

    return {
        "train_loss": avg_total_loss,
        "train_losses": avg_losses,
        "valid_batches": valid_batches,
        "elapsed_seconds": elapsed
    }


@torch.no_grad()
def validate_one_epoch(model, data_loader, device):
    """
    Torchvision Mask R-CNN returns losses only in train mode when targets are passed.

    For validation loss, we temporarily use train mode with no_grad.
    This is normal for a simple training-loss validation check.
    """

    model.train()

    running_loss = 0.0
    running_losses = {}

    valid_batches = 0

    for images, targets in data_loader:
        if all(target["boxes"].numel() == 0 for target in targets):
            continue

        images = [image.to(device) for image in images]
        targets = move_targets_to_device(targets, device)

        loss_dict = model(images, targets)
        total_loss = calculate_total_loss(loss_dict)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            continue

        running_loss += float(total_loss.detach().cpu())
        valid_batches += 1

        for key, value in loss_dict.items():
            running_losses.setdefault(key, 0.0)
            running_losses[key] += float(value.detach().cpu())

    avg_total_loss = running_loss / max(valid_batches, 1)

    avg_losses = {
        key: value / max(valid_batches, 1)
        for key, value in running_losses.items()
    }

    return {
        "val_loss": avg_total_loss,
        "val_losses": avg_losses,
        "valid_batches": valid_batches
    }


def save_checkpoint(
    model,
    output_path,
    epoch,
    experiment_config,
    model_config,
    optimizer,
    train_loss,
    val_loss,
    is_best
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "experiment_config": experiment_config,
        "model_config": model_config,
        "model_type": experiment_config.get("model_type"),
        "backbone": experiment_config.get("backbone"),
        "tile_size": experiment_config.get("tile_size"),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "is_best": is_best
    }

    torch.save(checkpoint, output_path)


def train_maskrcnn(
    model,
    train_loader,
    val_loader,
    device,
    output_folder,
    experiment_config,
    model_config
):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    epochs = int(get_training_value(model_config, "epochs", 20))
    learning_rate = float(get_training_value(model_config, "learning_rate", 0.0001))
    weight_decay = float(get_training_value(model_config, "weight_decay", 0.00001))

    optimizer_name = str(get_training_value(model_config, "optimizer", "adamw")).lower()

    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    best_val_loss = None
    best_epoch = None

    log_rows = []

    best_model_path = output_folder / "best_model.pth"
    last_model_path = output_folder / "last_model.pth"
    training_log_csv = output_folder / "training_log.csv"
    training_summary_json = output_folder / "training_summary.json"

    print("\n========== MASK R-CNN TRAINING START ==========")
    print(f"Epochs: {epochs}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Device: {device}")
    print(f"Output Folder: {output_folder}")
    print("================================================\n")

    for epoch in range(1, epochs + 1):
        train_result = train_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        val_result = validate_one_epoch(
            model=model,
            data_loader=val_loader,
            device=device
        )

        train_loss = train_result["train_loss"]
        val_loss = val_result["val_loss"]

        is_best = False

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            is_best = True

            save_checkpoint(
                model=model,
                output_path=best_model_path,
                epoch=epoch,
                experiment_config=experiment_config,
                model_config=model_config,
                optimizer=optimizer,
                train_loss=train_loss,
                val_loss=val_loss,
                is_best=True
            )

        save_checkpoint(
            model=model,
            output_path=last_model_path,
            epoch=epoch,
            experiment_config=experiment_config,
            model_config=model_config,
            optimizer=optimizer,
            train_loss=train_loss,
            val_loss=val_loss,
            is_best=False
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "is_best": is_best,
            "train_valid_batches": train_result["valid_batches"],
            "val_valid_batches": val_result["valid_batches"],
            "elapsed_seconds": train_result["elapsed_seconds"]
        }

        for key, value in train_result["train_losses"].items():
            row[f"train_{key}"] = value

        for key, value in val_result["val_losses"].items():
            row[f"val_{key}"] = value

        log_rows.append(row)

        pd.DataFrame(log_rows).to_csv(training_log_csv, index=False)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Best Val Loss: {best_val_loss:.4f}"
        )

        if is_best:
            print(f"Best model saved at epoch {epoch}")

    summary = {
        "experiment_id": experiment_config.get("experiment_id"),
        "model_type": experiment_config.get("model_type"),
        "backbone": experiment_config.get("backbone"),
        "tile_size": experiment_config.get("tile_size"),
        "epochs": epochs,
        "optimizer": optimizer_name,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_model_path": str(best_model_path),
        "last_model_path": str(last_model_path),
        "training_log_csv": str(training_log_csv)
    }

    save_json(summary, training_summary_json)

    print("\n========== MASK R-CNN TRAINING FINISHED ==========")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Model Path: {best_model_path}")
    print(f"Training Log CSV: {training_log_csv}")
    print(f"Training Summary JSON: {training_summary_json}")
    print("==================================================\n")

    return summary