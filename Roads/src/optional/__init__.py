"""Optional modules for the Road Extraction approach.

This package initializer intentionally performs no eager imports.

Why
---
Importing one optional module such as:

    from src.optional.augmentation import build_training_augmentation

must not automatically import Master Optuna, the pipeline package,
SAM-LoRA, post-processing, or other unrelated optional features.

Import each optional feature directly from the module that owns it.

Examples
--------
    from src.optional.augmentation import build_training_augmentation
    from src.optional.master_optuna import run_master_optuna_study
    from src.optional.backbone_search import resolve_master_search_space
"""

from __future__ import annotations


__all__: list[str] = []
