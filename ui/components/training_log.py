"""
ui/components/training_log.py

Read-only persisted log viewer.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_log_file(
    path: str | Path,
    max_lines: int = 160,
) -> None:
    """
    Display the end of a saved backend log.
    """

    path = Path(
        path
    )

    if not path.exists():
        st.info(
            "No log file is available."
        )
        return

    lines = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    st.code(
        "\n".join(
            lines[
                -int(
                    max_lines
                ):
            ]
        ),
        language="text",
    )
