"""
Static inspection of the installed ArcGIS Learn Road-model API.

This script DOES NOT import:

    arcpy
    arcgis
    arcgis.learn

Therefore ArcGIS Pro product-license initialization is not required.

It scans the installed arcgis Python source code and extracts information
about:

    ConnectNet
    MultiTaskRoadExtractor
    prepare_data

Output
------
Roads/outputs/diagnostics/
    arcgis_learn_road_api_static.json
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------
# Roads root
# ---------------------------------------------------------------------

ROAD_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


# ---------------------------------------------------------------------
# Locate ArcGIS Python package
# ---------------------------------------------------------------------

def find_arcgis_package() -> Path:
    """
    Find the installed 'arcgis' package without importing it.
    """

    candidates = [
        Path(sys.prefix)
        / "Lib"
        / "site-packages"
        / "arcgis",

        Path(sys.prefix)
        / "lib"
        / "site-packages"
        / "arcgis",

        Path(
            r"C:\Program Files\ArcGIS\Pro\bin\Python"
            r"\envs\arcgispro-py3\Lib\site-packages\arcgis"
        ),
    ]

    for path in candidates:

        if path.exists():

            return path

    raise FileNotFoundError(
        "Could not locate the installed ArcGIS Python package.\n\n"
        "Checked:\n"
        + "\n".join(
            str(path)
            for path in candidates
        )
    )


# ---------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------

def expression_to_text(
    node,
) -> str:
    """
    Convert a simple AST expression into readable text.
    """

    if node is None:

        return ""

    try:

        return ast.unparse(
            node
        )

    except Exception:

        return "<expression>"


def function_signature(
    node: ast.FunctionDef
    | ast.AsyncFunctionDef,
) -> str:
    """
    Reconstruct a readable function signature from AST.
    """

    args = node.args

    items = []

    # --------------------------------------------------------------
    # Positional-only + normal positional arguments
    # --------------------------------------------------------------

    positional = (
        list(args.posonlyargs)
        + list(args.args)
    )

    defaults = list(
        args.defaults
    )

    number_without_default = (
        len(positional)
        - len(defaults)
    )

    for index, argument in enumerate(
        positional
    ):

        text = argument.arg

        if argument.annotation is not None:

            text += (
                ": "
                + expression_to_text(
                    argument.annotation
                )
            )

        if index >= number_without_default:

            default_index = (
                index
                - number_without_default
            )

            text += (
                "="
                + expression_to_text(
                    defaults[
                        default_index
                    ]
                )
            )

        items.append(
            text
        )

    # --------------------------------------------------------------
    # *args
    # --------------------------------------------------------------

    if args.vararg is not None:

        items.append(
            "*"
            + args.vararg.arg
        )

    elif args.kwonlyargs:

        items.append(
            "*"
        )

    # --------------------------------------------------------------
    # Keyword-only arguments
    # --------------------------------------------------------------

    for (
        argument,
        default,
    ) in zip(
        args.kwonlyargs,
        args.kw_defaults,
    ):

        text = argument.arg

        if argument.annotation is not None:

            text += (
                ": "
                + expression_to_text(
                    argument.annotation
                )
            )

        if default is not None:

            text += (
                "="
                + expression_to_text(
                    default
                )
            )

        items.append(
            text
        )

    # --------------------------------------------------------------
    # **kwargs
    # --------------------------------------------------------------

    if args.kwarg is not None:

        items.append(
            "**"
            + args.kwarg.arg
        )

    signature = (
        "("
        + ", ".join(items)
        + ")"
    )

    if node.returns is not None:

        signature += (
            " -> "
            + expression_to_text(
                node.returns
            )
        )

    return signature


# ---------------------------------------------------------------------
# Source inspection
# ---------------------------------------------------------------------

TARGET_CLASSES = {
    "ConnectNet",
    "MultiTaskRoadExtractor",
}

TARGET_FUNCTIONS = {
    "prepare_data",
}

METHODS_OF_INTEREST = {
    "__init__",
    "fit",
    "lr_find",
    "save",
    "load",
    "predict",
    "show_results",
    "from_model",
    "export",
    "freeze",
    "unfreeze",
}


def inspect_python_file(
    path: Path,
):
    """
    Parse one Python file and return matching symbols.
    """

    try:

        source = path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        tree = ast.parse(
            source,
            filename=str(path),
        )

    except Exception:

        return []

    matches = []

    for node in ast.walk(
        tree
    ):

        # ----------------------------------------------------------
        # Classes
        # ----------------------------------------------------------

        if (
            isinstance(
                node,
                ast.ClassDef,
            )
            and node.name
            in TARGET_CLASSES
        ):

            methods = {}

            for item in node.body:

                if (
                    isinstance(
                        item,
                        (
                            ast.FunctionDef,
                            ast.AsyncFunctionDef,
                        ),
                    )
                    and item.name
                    in METHODS_OF_INTEREST
                ):

                    methods[
                        item.name
                    ] = {
                        "signature":
                            function_signature(
                                item
                            ),

                        "line":
                            int(
                                item.lineno
                            ),
                    }

            matches.append(
                {
                    "type":
                        "class",

                    "name":
                        node.name,

                    "file":
                        str(
                            path
                        ),

                    "line":
                        int(
                            node.lineno
                        ),

                    "bases": [
                        expression_to_text(
                            base
                        )
                        for base
                        in node.bases
                    ],

                    "methods":
                        methods,
                }
            )

        # ----------------------------------------------------------
        # Functions
        # ----------------------------------------------------------

        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and node.name
            in TARGET_FUNCTIONS
        ):

            matches.append(
                {
                    "type":
                        "function",

                    "name":
                        node.name,

                    "file":
                        str(
                            path
                        ),

                    "line":
                        int(
                            node.lineno
                        ),

                    "signature":
                        function_signature(
                            node
                        ),
                }
            )

    return matches


# ---------------------------------------------------------------------
# Text references
# ---------------------------------------------------------------------

def find_text_references(
    package_root: Path,
    symbol: str,
    maximum_results: int = 30,
):
    """
    Find Python files containing the symbol text.

    Useful for discovering re-exports and helper modules.
    """

    results = []

    for path in package_root.rglob(
        "*.py"
    ):

        try:

            text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        except Exception:

            continue

        if symbol not in text:

            continue

        line_numbers = []

        for (
            line_number,
            line,
        ) in enumerate(
            text.splitlines(),
            start=1,
        ):

            if symbol in line:

                line_numbers.append(
                    line_number
                )

        results.append(
            {
                "file":
                    str(
                        path
                    ),

                "lines":
                    line_numbers[
                        :20
                    ],
            }
        )

        if len(
            results
        ) >= maximum_results:

            break

    return results


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    package_root = (
        find_arcgis_package()
    )

    print()
    print(
        "=" * 72
    )

    print(
        "STATIC ARCGIS LEARN ROAD API PROBE"
    )

    print(
        "=" * 72
    )

    print(
        f"ArcGIS package   : "
        f"{package_root}"
    )

    matches = []

    python_files = list(
        package_root.rglob(
            "*.py"
        )
    )

    print(
        f"Python files     : "
        f"{len(python_files)}"
    )

    for path in python_files:

        matches.extend(
            inspect_python_file(
                path
            )
        )

    references = {
        symbol:
            find_text_references(
                package_root,
                symbol,
            )

        for symbol in (
            "ConnectNet",
            "MultiTaskRoadExtractor",
            "prepare_data",
        )
    }

    output = {
        "success":
            True,

        "method":
            "static_source_inspection",

        "license_required":
            False,

        "arcgis_package_root":
            str(
                package_root
            ),

        "matches":
            matches,

        "references":
            references,
    }

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
        / "arcgis_learn_road_api_static.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
        )

    print()

    for match in matches:

        print(
            "-" * 72
        )

        print(
            f"{match['type'].upper()}: "
            f"{match['name']}"
        )

        print(
            f"File             : "
            f"{match['file']}"
        )

        if match[
            "type"
        ] == "function":

            print(
                f"Signature        : "
                f"{match['signature']}"
            )

        else:

            print(
                f"Bases            : "
                f"{match['bases']}"
            )

            constructor = (
                match[
                    "methods"
                ].get(
                    "__init__"
                )
            )

            if constructor:

                print(
                    f"Constructor      : "
                    f"{constructor['signature']}"
                )

            for method_name in (
                "fit",
                "lr_find",
                "save",
                "load",
                "predict",
                "from_model",
            ):

                method = (
                    match[
                        "methods"
                    ].get(
                        method_name
                    )
                )

                if method:

                    print(
                        f"{method_name:<16} : "
                        f"{method['signature']}"
                    )

    print(
        "-" * 72
    )

    print(
        f"Full JSON        : "
        f"{output_path}"
    )

    print(
        "=" * 72
    )

    if not matches:

        print()
        print(
            "No direct class/function definitions were found."
        )

        print(
            "The JSON still contains text references that we can "
            "use to locate re-exported or dynamically defined symbols."
        )


if __name__ == "__main__":
    main()