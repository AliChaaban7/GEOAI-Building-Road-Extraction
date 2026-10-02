"""
ArcGIS Learn MultiTaskRoadExtractor wrapper.

No custom replacement architecture is implemented here.

The real Esri model is used:

    arcgis.learn.MultiTaskRoadExtractor

Installed constructor
---------------------
MultiTaskRoadExtractor(
    data,
    backbone=None,
    pretrained_path=None,
    *args,
    **kwargs
)

Default thesis architecture
---------------------------
mtl_model = "hourglass"

Advanced Esri parameters remain optional so Esri's documented defaults
are preserved unless our experiment config explicitly overrides them.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


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


def _get_multitask_class():
    """
    Lazy ArcGIS Learn import.
    """

    try:

        from arcgis.learn import (
            MultiTaskRoadExtractor,
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not import arcgis.learn.MultiTaskRoadExtractor.\n\n"
            "Run this backend from an initialized ArcGIS Pro "
            "Python session, preferably an ArcGIS Pro Notebook."
        ) from exc

    return MultiTaskRoadExtractor


def resolve_multitask_config(
    model_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve Esri MultiTaskRoadExtractor settings.
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
            "MultiTaskRoadExtractor mtl_model must be "
            "'hourglass' or 'linknet'."
        )

    backbone = _first(
        model_config.get(
            "backbone"
        ),

        architecture.get(
            "backbone"
        ),
    )

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


def build_multitask_road_extractor(
    data,
    model_config: Dict[str, Any],
    pretrained_path: Optional[
        str
    ] = None,
):
    """
    Build the real Esri MultiTaskRoadExtractor.
    """

    MultiTaskRoadExtractor = (
        _get_multitask_class()
    )

    config = (
        resolve_multitask_config(
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

    model = MultiTaskRoadExtractor(
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


build_model = build_multitask_road_extractor
create_multitask_road_extractor = (
    build_multitask_road_extractor
)