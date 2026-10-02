"""
ui/components/experiment_form.py

Reusable Road/Building experiment-form components.

HTML banners use compact rendering to avoid raw HTML code blocks.
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import streamlit as st


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}


def _html(
    content: str,
) -> None:
    cleaned = "\n".join(
        line.strip()
        for line in textwrap.dedent(content).splitlines()
        if line.strip()
    )

    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(
            cleaned,
            unsafe_allow_html=True,
        )


@st.cache_data(
    show_spinner=False,
    ttl=120,
)
def dataset_file_info(
    dataset_path: str,
) -> dict:

    path = Path(
        dataset_path
    )

    if not path.exists():
        return {
            "images": None,
            "labels": None,
            "paired": None,
        }

    def files_in(
        folder: Path,
    ):
        if not folder.exists():
            return []

        return [
            item
            for item in folder.rglob("*")
            if item.is_file()
            and item.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

    images = files_in(
        path / "images"
    )

    labels = files_in(
        path / "labels"
    )

    image_stems = {
        item.stem
        for item in images
    }

    label_stems = {
        item.stem
        for item in labels
    }

    return {
        "images": len(images),
        "labels": len(labels),
        "paired": len(
            image_stems
            & label_stems
        ),
    }


def render_dataset_banner(
    dataset_id: str,
    dataset_path: str | None,
) -> None:

    info = (
        dataset_file_info(
            str(dataset_path)
        )
        if dataset_path
        else {
            "images": None,
            "labels": None,
            "paired": None,
        }
    )

    _html(
        f"""
        <div class="info-banner">
            <b>Dataset:</b> {dataset_id}
            <span class="banner-separator">·</span>
            <b>Images:</b> {info["images"] if info["images"] is not None else "—"}
            <span class="banner-separator">·</span>
            <b>Labels:</b> {info["labels"] if info["labels"] is not None else "—"}
            <span class="banner-separator">·</span>
            <b>Paired:</b> {info["paired"] if info["paired"] is not None else "—"}
        </div>
        """
    )


def render_split_banner(
    train_percent: int,
    validation_percent: int,
    seed: int,
) -> None:

    _html(
        f"""
        <div class="success-banner">
            <b>Training:</b> {train_percent}%
            <span class="banner-separator">·</span>
            <b>Validation:</b> {validation_percent}%
            <span class="banner-separator">·</span>
            <b>Seed:</b> {seed}
            <span class="banner-separator">·</span>
            <b>Final Test:</b> Independent
        </div>
        """
    )


def render_backend_banner(
    backend: str,
) -> None:

    _html(
        f"""
        <div class="backend-badge">
            Backend <b>{backend}</b>
        </div>
        """
    )
