"""
Build RGB-only copies of existing Road Classified-Tiles datasets WITHOUT
re-exporting from ArcGIS Pro and WITHOUT modifying the originals.

Key fix in this version
-----------------------
The older script copied every TIFF into CT_<tile>_Roads_RGB and then tried
to replace 4-band TIFFs in place. ArcPy can keep those TIFFs locked, which
causes Windows WinError 32.

This version NEVER rewrites a TIFF in place.

For each source dataset:
    CT_<tile>_Roads
it creates:
    CT_<tile>_Roads_RGB

It copies all non-image files/folders normally, then rebuilds the RGB
"images" folder directly from the ORIGINAL source images:
    - 3-band image  -> ordinary file copy
    - >3-band image -> ArcPy writes Bands 1,2,3 directly to NEW target TIFF
    - <3-band image -> error

The original CT_<tile>_Roads dataset is never changed.

Run with the ArcGIS Pro Python environment.

Examples
--------
Only 256:
    python make_road_rgb_datasets.py --tiles 256 --overwrite

Several sizes:
    python make_road_rgb_datasets.py --tiles 128 256 512 --overwrite

All CT_<number>_Roads folders found under the root:
    python make_road_rgb_datasets.py --overwrite
"""

from __future__ import annotations

import argparse
import re
import shutil
import uuid
from pathlib import Path


DEFAULT_ROOT = Path(
    r"D:\Ali Chaaban Thesis Project\Data\Exported_Data_Roads"
)

DATASET_PATTERN = re.compile(
    r"^CT_(\d+)_Roads$",
    re.IGNORECASE,
)

IMAGE_EXTENSIONS = {
    ".tif",
    ".tiff",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create RGB-only copies of CT_<tile>_Roads datasets "
            "without re-exporting."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=(
            "Folder containing CT_<tile>_Roads datasets. "
            f"Default: {DEFAULT_ROOT}"
        ),
    )

    parser.add_argument(
        "--tiles",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Optional tile sizes, e.g. --tiles 128 256 512. "
            "If omitted, all CT_<number>_Roads folders are processed."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete and rebuild an existing *_RGB output folder. "
            "The original non-RGB dataset is NEVER modified."
        ),
    )

    return parser.parse_args()


def get_arcpy():
    try:
        import arcpy
    except Exception as exc:
        raise RuntimeError(
            "arcpy could not be imported.\n"
            "Run this script with the ArcGIS Pro Python environment."
        ) from exc

    return arcpy


def discover_datasets(
    root: Path,
    requested_tiles: list[int] | None,
) -> list[tuple[int, Path]]:

    if not root.exists():
        raise FileNotFoundError(
            f"Road export root does not exist:\n{root}"
        )

    requested = (
        {int(value) for value in requested_tiles}
        if requested_tiles
        else None
    )

    found = []

    for path in root.iterdir():

        if not path.is_dir():
            continue

        match = DATASET_PATTERN.fullmatch(
            path.name
        )

        if match is None:
            continue

        tile = int(
            match.group(1)
        )

        if (
            requested is not None
            and tile not in requested
        ):
            continue

        found.append(
            (tile, path)
        )

    found.sort(
        key=lambda item: item[0]
    )

    if requested is not None:

        found_tiles = {
            tile
            for tile, _
            in found
        }

        missing = sorted(
            requested - found_tiles
        )

        if missing:
            raise FileNotFoundError(
                "Requested Road dataset folder(s) not found:\n"
                + "\n".join(
                    str(
                        root
                        / f"CT_{tile}_Roads"
                    )
                    for tile
                    in missing
                )
            )

    if not found:
        raise FileNotFoundError(
            "No CT_<tile>_Roads folders found under:\n"
            f"{root}"
        )

    return found


def find_images_directory(
    dataset_root: Path,
) -> Path:

    direct = (
        dataset_root
        / "images"
    )

    if direct.is_dir():
        return direct

    direct_upper = (
        dataset_root
        / "Images"
    )

    if direct_upper.is_dir():
        return direct_upper

    matches = [
        path
        for path
        in dataset_root.rglob("*")
        if (
            path.is_dir()
            and path.name.lower()
            == "images"
        )
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            "Could not find an images folder inside:\n"
            f"{dataset_root}"
        )

    raise RuntimeError(
        "More than one images folder was found. "
        "Refusing to guess:\n"
        + "\n".join(
            str(path)
            for path
            in matches
        )
    )


def list_image_files(
    images_root: Path,
) -> list[Path]:

    files = [
        path
        for path
        in images_root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    files.sort(
        key=lambda path: str(path).lower()
    )

    return files


def raster_band_count(
    arcpy,
    raster_path: Path,
) -> int:

    result = (
        arcpy.management.GetRasterProperties(
            str(
                raster_path
            ),
            "BANDCOUNT",
        )
    )

    return int(
        result.getOutput(0)
    )


def copy_non_image_dataset_content(
    source: Path,
    target: Path,
    source_images: Path,
) -> Path:
    """
    Copy the full ArcGIS training dataset EXCEPT the images subtree.

    Returns the target images-directory path corresponding to source_images.
    """

    relative_images = (
        source_images.relative_to(
            source
        )
    )

    target_images = (
        target
        / relative_images
    )

    target.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Copy every top-level item except the branch that contains source_images.
    # We then rebuild that images branch from source TIFFs.
    for item in source.iterdir():

        destination = (
            target
            / item.name
        )

        try:
            relative_images.parts[
                0
            ]
        except Exception:
            raise RuntimeError(
                "Could not resolve relative images folder."
            )

        if item.name == relative_images.parts[0]:

            # If images is directly under source, skip it completely.
            if len(
                relative_images.parts
            ) == 1:
                continue

            # For unusual nested layouts, copy the parent branch and remove
            # only the copied images subtree afterward.
            if item.is_dir():

                shutil.copytree(
                    item,
                    destination,
                )

                copied_images = (
                    target
                    / relative_images
                )

                if copied_images.exists():
                    shutil.rmtree(
                        copied_images
                    )

            else:
                shutil.copy2(
                    item,
                    destination,
                )

            continue

        if item.is_dir():

            shutil.copytree(
                item,
                destination,
            )

        else:

            shutil.copy2(
                item,
                destination,
            )

    target_images.mkdir(
        parents=True,
        exist_ok=True,
    )

    return target_images


def write_three_band_copy(
    arcpy,
    source_image: Path,
    target_image: Path,
) -> None:
    """
    Write Bands 1,2,3 directly from source_image to a NEW target TIFF.

    No in-place overwrite is performed.
    """

    target_image.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if target_image.exists():
        raise FileExistsError(
            f"Target image unexpectedly exists:\n{target_image}"
        )

    layer_name = (
        "road_rgb_"
        + uuid.uuid4().hex
    )

    try:

        arcpy.management.MakeRasterLayer(
            str(
                source_image
            ),
            layer_name,
            band_index="1;2;3",
        )

        arcpy.management.CopyRaster(
            layer_name,
            str(
                target_image
            ),
        )

    finally:

        try:
            arcpy.management.Delete(
                layer_name
            )
        except Exception:
            pass

        try:
            arcpy.ClearWorkspaceCache_management()
        except Exception:
            pass

    bands = raster_band_count(
        arcpy,
        target_image,
    )

    if bands != 3:
        raise RuntimeError(
            "Band extraction did not produce exactly 3 bands.\n"
            f"Source: {source_image}\n"
            f"Target: {target_image}\n"
            f"Output bands: {bands}"
        )


def verify_auxiliary_files(
    source: Path,
    target: Path,
) -> None:

    source_map = (
        source
        / "map.txt"
    )

    target_map = (
        target
        / "map.txt"
    )

    if (
        source_map.exists()
        and not target_map.exists()
    ):
        raise RuntimeError(
            "map.txt was not preserved in the RGB copy."
        )

    for label_name in (
        "labels",
        "Labels",
    ):

        source_labels = (
            source
            / label_name
        )

        if not source_labels.is_dir():
            continue

        target_labels = (
            target
            / label_name
        )

        if not target_labels.is_dir():
            raise RuntimeError(
                "Labels folder was not preserved:\n"
                f"{target_labels}"
            )

        source_count = sum(
            1
            for path
            in source_labels.rglob("*")
            if path.is_file()
        )

        target_count = sum(
            1
            for path
            in target_labels.rglob("*")
            if path.is_file()
        )

        if source_count != target_count:
            raise RuntimeError(
                "Label count changed while creating RGB dataset.\n"
                f"Source labels: {source_count}\n"
                f"Target labels: {target_count}"
            )


def process_dataset(
    arcpy,
    tile: int,
    source: Path,
    overwrite: bool,
) -> dict:

    target = source.with_name(
        source.name
        + "_RGB"
    )

    print()
    print(
        "=" * 78
    )
    print(
        f"CT_{tile}_Roads -> CT_{tile}_Roads_RGB"
    )
    print(
        "=" * 78
    )
    print(
        f"Source: {source}"
    )
    print(
        f"Target: {target}"
    )

    if target.exists():

        if not overwrite:

            raise FileExistsError(
                "RGB target already exists:\n"
                f"{target}\n\n"
                "Use --overwrite to rebuild it."
            )

        print(
            "Deleting previous RGB target..."
        )

        shutil.rmtree(
            target
        )

    source_images = find_images_directory(
        source
    )

    source_files = list_image_files(
        source_images
    )

    if not source_files:
        raise RuntimeError(
            "No TIFF training images were found in:\n"
            f"{source_images}"
        )

    print(
        f"Images found: {len(source_files)}"
    )
    print(
        "Copying dataset metadata/labels (images rebuilt separately)..."
    )

    target_images = (
        copy_non_image_dataset_content(
            source=source,
            target=target,
            source_images=source_images,
        )
    )

    verify_auxiliary_files(
        source,
        target,
    )

    already_rgb = 0
    converted = 0
    problems = []

    total = len(
        source_files
    )

    print(
        "Building RGB image folder from original source images..."
    )

    for index, source_image in enumerate(
        source_files,
        start=1,
    ):

        relative = (
            source_image.relative_to(
                source_images
            )
        )

        target_image = (
            target_images
            / relative
        )

        try:

            bands = raster_band_count(
                arcpy,
                source_image,
            )

            if bands == 3:

                target_image.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                shutil.copy2(
                    source_image,
                    target_image,
                )

                already_rgb += 1

            elif bands > 3:

                write_three_band_copy(
                    arcpy=arcpy,
                    source_image=source_image,
                    target_image=target_image,
                )

                converted += 1

            else:

                problems.append(
                    (
                        source_image,
                        f"{bands} band(s)",
                    )
                )

        except Exception as exc:

            problems.append(
                (
                    source_image,
                    str(
                        exc
                    ),
                )
            )

        if (
            index == 1
            or index % 250 == 0
            or index == total
        ):

            print(
                f"Processed {index}/{total}"
                f" | Already RGB: {already_rgb}"
                f" | Converted: {converted}"
                f" | Problems: {len(problems)}"
            )

    if problems:

        print()
        print(
            "FAILED — original dataset is still untouched."
        )

        for path, error in problems[:40]:

            print(
                f"  {path}: {error}"
            )

        raise RuntimeError(
            f"RGB creation failed for CT_{tile}_Roads_RGB"
        )

    # ----------------------------------------------------------
    # Final verification
    # ----------------------------------------------------------

    target_files = list_image_files(
        target_images
    )

    if len(
        target_files
    ) != total:

        raise RuntimeError(
            "Image count mismatch after RGB creation.\n"
            f"Source: {total}\n"
            f"Target: {len(target_files)}"
        )

    print(
        "Final verification: every target image must have exactly 3 bands..."
    )

    invalid = []

    for index, target_image in enumerate(
        target_files,
        start=1,
    ):

        try:

            bands = raster_band_count(
                arcpy,
                target_image,
            )

            if bands != 3:

                invalid.append(
                    (
                        target_image,
                        bands,
                    )
                )

        except Exception as exc:

            invalid.append(
                (
                    target_image,
                    f"ERROR: {exc}",
                )
            )

        if (
            index % 1000 == 0
            or index == len(
                target_files
            )
        ):
            print(
                f"Verified {index}/{len(target_files)}"
                f" | Invalid: {len(invalid)}"
            )

    if invalid:

        print()
        print(
            "FINAL VERIFICATION FAILED"
        )

        for path, bands in invalid[:40]:

            print(
                f"  {path}: {bands}"
            )

        raise RuntimeError(
            f"RGB verification failed for CT_{tile}_Roads_RGB"
        )

    print()
    print(
        "=" * 78
    )
    print(
        "RGB DATASET COMPLETE"
    )
    print(
        "=" * 78
    )
    print(
        f"Images total       : {total}"
    )
    print(
        f"Already 3-band     : {already_rgb}"
    )
    print(
        f"Converted >3 -> RGB: {converted}"
    )
    print(
        f"Verified 3-band    : {total}/{total}"
    )
    print(
        f"Output             : {target}"
    )
    print(
        "Original dataset   : UNTOUCHED"
    )
    print(
        "=" * 78
    )

    return {
        "tile":
            tile,
        "total":
            total,
        "already_rgb":
            already_rgb,
        "converted":
            converted,
        "target":
            str(
                target
            ),
    }


def main():

    args = parse_args()

    root = (
        args.root
        .expanduser()
        .resolve()
    )

    datasets = discover_datasets(
        root=root,
        requested_tiles=args.tiles,
    )

    print(
        "=" * 78
    )
    print(
        "ROAD RGB DATASET CREATOR — NO IN-PLACE TIFF REPLACEMENT"
    )
    print(
        "=" * 78
    )
    print(
        f"Root: {root}"
    )

    for tile, source in datasets:

        print(
            f"  {source.name} -> {source.name}_RGB"
        )

    print(
        "=" * 78
    )

    arcpy = get_arcpy()

    results = []

    for tile, source in datasets:

        results.append(
            process_dataset(
                arcpy=arcpy,
                tile=tile,
                source=source,
                overwrite=bool(
                    args.overwrite
                ),
            )
        )

    print()
    print(
        "=" * 78
    )
    print(
        "ALL REQUESTED RGB DATASETS COMPLETE"
    )
    print(
        "=" * 78
    )

    for result in results:

        print(
            f"CT_{result['tile']}_Roads_RGB"
            f" | Total: {result['total']}"
            f" | Already RGB: {result['already_rgb']}"
            f" | Converted: {result['converted']}"
        )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
