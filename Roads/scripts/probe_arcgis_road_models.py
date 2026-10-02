"""
Inspect the actual ArcGIS Learn Road-model API installed in the user's
ArcGIS Pro environment.

Nothing is trained or modified.

The result is written to:

    Roads/outputs/diagnostics/arcgis_learn_road_api.json

We use this before implementing:
- ConnectNet wrapper
- MultiTaskRoadExtractor wrapper
- ArcGIS Learn Road trainer

This prevents us from guessing signatures from another ArcGIS version.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import sys
import traceback


ROAD_ROOT = (
    Path(
        __file__
    )
    .resolve()
    .parents[
        1
    ]
)

if str(
    ROAD_ROOT
) not in sys.path:

    sys.path.insert(
        0,
        str(
            ROAD_ROOT
        ),
    )


def safe_signature(
    value,
):

    try:

        return str(
            inspect.signature(
                value
            )
        )

    except Exception as exc:

        return (
            f"<signature unavailable: {exc}>"
        )


def safe_source_file(
    value,
):

    try:

        return inspect.getfile(
            value
        )

    except Exception:

        return None


def safe_doc(
    value,
    maximum_length=3000,
):

    try:

        document = (
            inspect.getdoc(
                value
            )
            or ""
        )

        return document[
            :maximum_length
        ]

    except Exception:

        return ""


def describe_callable(
    value,
):

    return {
        "signature":
            safe_signature(
                value
            ),

        "module":
            getattr(
                value,
                "__module__",
                None,
            ),

        "source_file":
            safe_source_file(
                value
            ),

        "doc":
            safe_doc(
                value
            ),
    }


def describe_class(
    class_object,
):

    result = {
        "name":
            getattr(
                class_object,
                "__name__",
                str(
                    class_object
                ),
            ),

        "constructor":
            safe_signature(
                class_object
            ),

        "module":
            getattr(
                class_object,
                "__module__",
                None,
            ),

        "source_file":
            safe_source_file(
                class_object
            ),

        "mro": [
            cls.__name__
            for cls
            in getattr(
                class_object,
                "__mro__",
                []
            )
        ],

        "doc":
            safe_doc(
                class_object
            ),

        "methods":
            {},
    }

    method_names = (
        "fit",
        "lr_find",
        "save",
        "load",
        "predict",
        "predict_video",
        "show_results",
        "show_results_multispectral",
        "from_model",
        "export",
        "available_metrics",
        "unfreeze",
        "freeze",
    )

    for name in method_names:

        if not hasattr(
            class_object,
            name,
        ):

            continue

        value = getattr(
            class_object,
            name
        )

        result[
            "methods"
        ][
            name
        ] = describe_callable(
            value
        )

    return result


def main():

    result = {
        "success":
            False,
    }

    try:

        import arcgis

        from arcgis.learn import (
            ConnectNet,
            MultiTaskRoadExtractor,
            prepare_data,
        )

        result[
            "arcgis_version"
        ] = getattr(
            arcgis,
            "__version__",
            None,
        )

        result[
            "prepare_data"
        ] = describe_callable(
            prepare_data
        )

        result[
            "ConnectNet"
        ] = describe_class(
            ConnectNet
        )

        result[
            "MultiTaskRoadExtractor"
        ] = describe_class(
            MultiTaskRoadExtractor
        )

        result[
            "success"
        ] = True

    except Exception as exc:

        result[
            "error"
        ] = str(
            exc
        )

        result[
            "traceback"
        ] = traceback.format_exc()

    output_folder = (
        ROAD_ROOT
        / "outputs"
        / "diagnostics"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_folder
        / "arcgis_learn_road_api.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            result,
            file,
            indent=2,
        )

    print()
    print(
        "=" * 72
    )

    print(
        "ARCGIS LEARN ROAD API PROBE"
    )

    print(
        "=" * 72
    )

    print(
        f"Success          : "
        f"{result['success']}"
    )

    print(
        f"ArcGIS API       : "
        f"{result.get('arcgis_version')}"
    )

    if result.get(
        "prepare_data"
    ):

        print(
            f"prepare_data     : "
            f"{result['prepare_data']['signature']}"
        )

    if result.get(
        "ConnectNet"
    ):

        print(
            f"ConnectNet       : "
            f"{result['ConnectNet']['constructor']}"
        )

    if result.get(
        "MultiTaskRoadExtractor"
    ):

        print(
            f"MultiTask        : "
            f"{result['MultiTaskRoadExtractor']['constructor']}"
        )

    if result.get(
        "error"
    ):

        print(
            f"Error            : "
            f"{result['error']}"
        )

    print(
        f"Full JSON        : "
        f"{output_path}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()