"""
ArcGIS Learn ConnectNet wrapper for Approach 2 - Road Extraction.

Important
---------
ConnectNet is NOT reimplemented in custom PyTorch.

The real installed Esri implementation is used:

    arcgis.learn.ConnectNet

Installed API
-------------
ConnectNet inherits MultiTaskRoadExtractor.

Road architecture:
    mtl_model = "hourglass" by default

Optional advanced parameters:
    gaussian_thresh
    orient_bin_size
    orient_theta

The import remains lazy so the native PyTorch Road pipeline can operate
without initializing ArcGIS Learn.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _as_dict(
    value,
) -> Dict[str, Any]:

    if isinstance(
        value,
        dict,
    ):
        return dict(
            value
        )

    return {}


def _first(
    *values,
):

    for value in values:

        if value is not None:

            return value

    return None


# ---------------------------------------------------------------------
# ArcGIS import
# ---------------------------------------------------------------------

def _get_connectnet_class():
    """
    Import the actual Esri ConnectNet only when needed.
    """

    try:

        from arcgis.learn import (
            ConnectNet,
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not import arcgis.learn.ConnectNet.\n\n"
            "ConnectNet requires a properly initialized ArcGIS Pro "
            "Python session/license.\n\n"
            "Use ArcGIS Pro Notebook if standalone Python cannot "
            "initialize the product license."
        ) from exc

    return ConnectNet


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

def resolve_connectnet_config(
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve ConnectNet constructor settings.
    """

    model_config = dict(
        model_config
        or {}
    )

    architecture = _as_dict(
        model_config.get(
            "architecture"
        )
    )

    advanced = _as_dict(
        model_config.get(
            "advanced"
        )
    )

    training = _as_dict(
        model_config.get(
            "training"
        )
    )

    mtl_model = str(
        _first(
            model_config.get(
                "mtl_model"
            ),

            architecture.get(
                "mtl_model"
            ),

            architecture.get(
                "name"
            ),

            "hourglass",
        )
    ).lower()

    if mtl_model not in {
        "hourglass",
        "linknet",
    }:

        raise ValueError(
            "ConnectNet mtl_model must be 'hourglass' or 'linknet'."
        )

    backbone = _first(
        model_config.get(
            "backbone"
        ),

        architecture.get(
            "backbone"
        ),
    )

    # Esri ignores backbone for hourglass.
    if mtl_model == "hourglass":

        backbone = None

    pretrained_path = _first(
        model_config.get(
            "pretrained_path"
        ),

        model_config.get(
            "pretrained_model"
        ),
    )

    gaussian_thresh = _first(
        advanced.get(
            "gaussian_thresh"
        ),

        model_config.get(
            "gaussian_thresh"
        ),
    )

    orient_bin_size = _first(
        advanced.get(
            "orient_bin_size"
        ),

        model_config.get(
            "orient_bin_size"
        ),
    )

    orient_theta = _first(
        advanced.get(
            "orient_theta"
        ),

        model_config.get(
            "orient_theta"
        ),
    )

    if gaussian_thresh is not None:

        gaussian_thresh = float(
            gaussian_thresh
        )

        if not (
            0.0
            <= gaussian_thresh
            <= 1.0
        ):

            raise ValueError(
                "gaussian_thresh must be between 0 and 1."
            )

    if orient_bin_size is not None:

        orient_bin_size = int(
            orient_bin_size
        )

        if orient_bin_size <= 0:

            raise ValueError(
                "orient_bin_size must be > 0."
            )

    if orient_theta is not None:

        orient_theta = int(
            orient_theta
        )

        if orient_theta <= 0:

            raise ValueError(
                "orient_theta must be > 0."
            )

    return {
        "mtl_model":
            mtl_model,

        "backbone":
            backbone,

        "pretrained_path":
            pretrained_path,

        "gaussian_thresh":
            gaussian_thresh,

        "orient_bin_size":
            orient_bin_size,

        "orient_theta":
            orient_theta,

        "batch_size":
            int(
                training.get(
                    "batch_size",
                    4,
                )
            ),

        "epochs":
            int(
                training.get(
                    "epochs",
                    40,
                )
            ),

        "learning_rate":
            float(
                training.get(
                    "learning_rate",
                    1e-4,
                )
            ),

        "weight_decay":
            float(
                training.get(
                    "weight_decay",
                    1e-5,
                )
            ),
    }


# ---------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------

def build_connectnet(
    data,
    model_config: Dict[str, Any],
    pretrained_path: Optional[
        str
    ] = None,
):
    """
    Build the real ArcGIS Learn ConnectNet model.
    """

    ConnectNet = (
        _get_connectnet_class()
    )

    config = (
        resolve_connectnet_config(
            model_config
        )
    )

    if pretrained_path is not None:

        config[
            "pretrained_path"
        ] = pretrained_path

    kwargs = {
        "mtl_model":
            config[
                "mtl_model"
            ],
    }

    # Do not override Esri's internal defaults unless explicitly set.
    if config[
        "gaussian_thresh"
    ] is not None:

        kwargs[
            "gaussian_thresh"
        ] = config[
            "gaussian_thresh"
        ]

    if config[
        "orient_bin_size"
    ] is not None:

        kwargs[
            "orient_bin_size"
        ] = config[
            "orient_bin_size"
        ]

    if config[
        "orient_theta"
    ] is not None:

        kwargs[
            "orient_theta"
        ] = config[
            "orient_theta"
        ]

    model = ConnectNet(
        data=data,

        backbone=config[
            "backbone"
        ],

        pretrained_path=(
            config[
                "pretrained_path"
            ]
        ),

        **kwargs,
    )

    return model


# Compatibility aliases.
build_model = build_connectnet
create_connectnet = build_connectnet