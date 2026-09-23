"""
backbones.py

Shared backbone utilities.
For the first test, pretrained weights are disabled by default to avoid internet download problems.
"""

import torch


def get_device():
    """
    Return CUDA device if available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    """
    Count trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def should_use_pretrained(experiment_config, backbone_info):
    """
    Decide whether pretrained weights should be loaded.

    Important:
    By default we return False because torchvision may try to download weights.
    Later, we can activate pretrained weights if needed.
    """
    allow_pretrained = experiment_config.get("allow_pretrained_weights", False)
    config_pretrained = backbone_info.get("pretrained", False)

    return bool(allow_pretrained and config_pretrained)