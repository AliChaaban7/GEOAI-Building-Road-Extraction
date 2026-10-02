"""
ui/components/results_table.py

Reusable experiment result table for Buildings, Roads and Comparison.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


NUMERIC_COLUMNS = (
    "Validation IoU",
    "Threshold",
    "Test IoU",
    "Precision",
    "Recall",
    "F1",
    "clDice",
)


def render_results_table(
    dataframe: pd.DataFrame,
    height: int = 420,
) -> None:
    """
    Render a professional experiment-results dataframe.
    """

    if dataframe.empty:
        st.info(
            "No experiment results are available yet."
        )
        return

    display = dataframe.copy()

    for column in NUMERIC_COLUMNS:
        if column not in display.columns:
            continue

        display[
            column
        ] = pd.to_numeric(
            display[
                column
            ],
            errors="coerce",
        ).round(
            2
        )

    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        height=height,
    )

    st.caption(
        "Validation metrics come from the validation split. "
        "Final metrics come only from the independent final-test area. "
        "Blank cells mean that stage has not been completed."
    )

