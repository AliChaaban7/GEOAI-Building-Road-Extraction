"""
ui/pages/dashboard.py

Professional read-only thesis results dashboard for the ONE shared GeoAI UI.

Performance / compatibility goals
---------------------------------
- Cache experiment discovery so filters and tabs do not repeatedly rescan all
  Building and Road experiment folders.
- Keep Arrow-safe dataframe dtypes for Streamlit rendering.
- Never mix integers/floats with the display placeholder "—" in one column.
- Use the current Streamlit width API instead of deprecated
  use_container_width.
- Keep Validation IoU and independent Final IoU clearly separated.
- Use Final IoU as the primary final-test overlap metric.
- Do NOT display clDice on this dashboard.
- Preserve experiment traceability: model, source, tile size, split,
  augmentation, Optuna, Master Optuna, threshold, post-processing and status.
- Show validation-to-final generalization gap when both metrics exist.

Data source
-----------
All values are read from the existing experiment artifacts through:
    ui.utils.result_reader.result_rows()

No training, validation, inference, freezing or post-processing is launched here.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import streamlit as st

try:
    import altair as alt
except Exception:
    alt = None

from ui.components.header import (
    render_header,
    render_section,
)
from ui.utils.paths import (
    BUILDINGS_EXPERIMENTS_ROOT,
    ROADS_EXPERIMENTS_ROOT,
)
from ui.utils.result_reader import (
    result_rows,
)


# ============================================================
# DASHBOARD CONSTANTS
# ============================================================

FINAL_IOU_COLUMN = "Final IoU"
VALIDATION_IOU_COLUMN = "Validation IoU"

PERCENT_METRICS = (
    VALIDATION_IOU_COLUMN,
    FINAL_IOU_COLUMN,
    "Precision",
    "Recall",
    "F1",
)

TEXT_COLUMNS = (
    "Approach",
    "Experiment",
    "Model",
    "Backend",
    "Source",
    "Tile",
    "Split",
    "Aug",
    "Optuna",
    "Master Optuna",
    "Post-processing",
    "Status",
    "Final Source",
)

MAIN_TABLE_COLUMNS = (
    "Approach",
    "Experiment",
    "Model",
    "Backend",
    "Source",
    "Tile",
    "Split",
    "Aug",
    "Optuna",
    "Master Optuna",
    VALIDATION_IOU_COLUMN,
    "Threshold",
    "Post-processing",
    "Status",
    "Final Source",
    FINAL_IOU_COLUMN,
    "Precision",
    "Recall",
    "F1",
    "Generalization Gap",
)


# ============================================================
# STYLE
# ============================================================

DASHBOARD_CSS = """
<style>

.st-key-results_dashboard_shell {
    padding-top: 0.2rem;
}

.st-key-results_dashboard_shell [data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(16, 35, 55, 0.97),
        rgba(8, 24, 40, 0.98)
    );
    border: 1px solid rgba(125, 211, 252, 0.14);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    min-height: 122px;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.14);
}

.st-key-results_dashboard_shell [data-testid="stMetricLabel"] {
    color: #9fb3c8 !important;
    font-size: 0.80rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em;
}

.st-key-results_dashboard_shell [data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-weight: 800 !important;
}

.dashboard-note {
    border: 1px solid rgba(125, 211, 252, 0.14);
    background: rgba(13, 31, 49, 0.82);
    color: #cbd5e1;
    border-radius: 12px;
    padding: 0.82rem 0.95rem;
    margin: 0.15rem 0 1.0rem 0;
    line-height: 1.45;
}

.dashboard-note b {
    color: #f8fafc;
}

.metric-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.25rem 0 0.8rem 0;
}

.metric-chip {
    border: 1px solid rgba(125, 211, 252, 0.16);
    background: rgba(15, 35, 54, 0.82);
    color: #dbeafe;
    border-radius: 999px;
    padding: 0.35rem 0.7rem;
    font-size: 0.78rem;
    font-weight: 700;
}

</style>
"""


# ============================================================
# DATA PREPARATION
# ============================================================

def _to_arrow_safe_text(
    series: pd.Series,
) -> pd.Series:
    """
    Convert one display/category column to a consistent Arrow-safe text dtype.

    Missing values stay as pandas <NA> internally. The visible "—" placeholder
    is applied only in display-only dataframes.
    """

    return (
        series
        .astype("string")
        .replace(
            {
                "nan": pd.NA,
                "None": pd.NA,
                "<NA>": pd.NA,
            }
        )
    )


@st.cache_data(
    ttl=15,
    show_spinner=False,
)
def _combined_results() -> pd.DataFrame:
    """
    Read Buildings + Roads experiment artifacts.

    Cached briefly because result_rows() may scan many experiment folders and
    JSON files. Filtering and switching dashboard tabs should not trigger the
    same filesystem scan repeatedly.
    """

    buildings = result_rows(
        "buildings",
        BUILDINGS_EXPERIMENTS_ROOT,
    )

    roads = result_rows(
        "roads",
        ROADS_EXPERIMENTS_ROOT,
    )

    frames = [
        frame
        for frame in (
            buildings,
            roads,
        )
        if not frame.empty
    ]

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    return _prepare_dashboard_frame(
        combined
    )


def _prepare_dashboard_frame(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert backend result columns to stable dashboard columns.

    Rules
    -----
    - Test IoU -> Final IoU.
    - clDice is removed entirely.
    - metric columns stay numeric.
    - descriptive/category columns are converted to pandas StringDtype.
    - Generalization Gap = Validation IoU - Final IoU.
    """

    frame = dataframe.copy()

    if "Test IoU" in frame.columns:
        frame = frame.rename(
            columns={
                "Test IoU": FINAL_IOU_COLUMN,
            }
        )

    if "clDice" in frame.columns:
        frame = frame.drop(
            columns=["clDice"]
        )

    # --------------------------------------------------------
    # Numeric metrics
    # --------------------------------------------------------

    for column in PERCENT_METRICS:
        if column not in frame.columns:
            frame[column] = pd.NA

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).astype("float64")

    if "Threshold" not in frame.columns:
        frame["Threshold"] = pd.NA

    frame["Threshold"] = pd.to_numeric(
        frame["Threshold"],
        errors="coerce",
    ).astype("float64")

    frame["Generalization Gap"] = (
        frame[VALIDATION_IOU_COLUMN]
        - frame[FINAL_IOU_COLUMN]
    ).astype("float64")

    # --------------------------------------------------------
    # Ensure every expected column exists
    # --------------------------------------------------------

    for column in MAIN_TABLE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    # --------------------------------------------------------
    # Text / categorical columns
    #
    # Tile intentionally becomes text in the dashboard.
    # It is an identifier/setting here, not a quantity used for math.
    # This prevents Arrow failures such as integer + "—" in one column.
    # --------------------------------------------------------

    for column in TEXT_COLUMNS:
        frame[column] = _to_arrow_safe_text(
            frame[column]
        )

    return frame


# ============================================================
# GENERIC HELPERS
# ============================================================

def _unique_values(
    dataframe: pd.DataFrame,
    column: str,
) -> list[str]:

    if column not in dataframe.columns:
        return []

    values = (
        dataframe[column]
        .dropna()
        .astype("string")
        .str.strip()
    )

    values = values[
        values.ne("")
    ]

    return sorted(
        values
        .dropna()
        .unique()
        .tolist()
    )


def _safe_mean(
    series: pd.Series,
) -> float | None:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(
        values.mean()
    )


def _safe_max(
    series: pd.Series,
) -> float | None:

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    return float(
        values.max()
    )


def _fmt_percent(
    value: float | None,
) -> str:

    if value is None or pd.isna(value):
        return "—"

    return f"{float(value):.2f}%"


def _fmt_number(
    value,
    digits: int = 2,
) -> str:

    if value is None or pd.isna(value):
        return "—"

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _fmt_text(
    value,
) -> str:

    if value is None or pd.isna(value):
        return "—"

    text = str(value).strip()

    return text if text else "—"


def _filter_frame(
    dataframe: pd.DataFrame,
    approach: str,
    model: str,
    final_source: str,
    status: str,
    tile: str,
) -> pd.DataFrame:

    frame = dataframe.copy()

    filters = (
        ("Approach", approach),
        ("Model", model),
        ("Final Source", final_source),
        ("Status", status),
        ("Tile", tile),
    )

    for column, selected in filters:
        if selected == "All":
            continue

        frame = frame[
            frame[column]
            .astype("string")
            .eq(str(selected))
        ]

    return frame


def _existing_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
) -> list[str]:

    return [
        column
        for column in columns
        if column in dataframe.columns
    ]


def _display_safe_frame(
    dataframe: pd.DataFrame,
    text_columns: Iterable[str],
) -> pd.DataFrame:
    """
    Return a display-only dataframe with guaranteed uniform text columns.

    The "—" placeholder is used only here, after all numeric computations are
    complete, so Arrow never has to reconcile integers and strings.
    """

    display = dataframe.copy()

    for column in text_columns:
        if column not in display.columns:
            continue

        display[column] = (
            display[column]
            .astype("string")
            .fillna("—")
        )

    return display


# ============================================================
# REFRESH
# ============================================================

def _render_refresh_control() -> None:
    """
    Manual cache invalidation for newly completed experiment artifacts.
    """

    refresh_left, refresh_right = st.columns(
        [1, 5]
    )

    with refresh_left:
        refresh = st.button(
            "Refresh Results",
            key="dashboard_refresh_results",
            width="stretch",
            help=(
                "Clear the short dashboard cache and rescan Building "
                "and Road experiment artifacts."
            ),
        )

    with refresh_right:
        st.caption(
            "Dashboard data is cached for 15 seconds to keep filters and "
            "visualizations responsive."
        )

    if refresh:
        _combined_results.clear()
        st.rerun()


# ============================================================
# FILTERS
# ============================================================

def _render_filters(
    combined: pd.DataFrame,
) -> pd.DataFrame:

    render_section(
        "A",
        "Dashboard Filters",
        "Filter the dashboard without changing any experiment artifact.",
    )

    c1, c2, c3, c4, c5 = st.columns(
        [1.1, 1.25, 1.1, 1.15, 0.8]
    )

    with c1:
        approach = st.selectbox(
            "Approach",
            ["All"] + _unique_values(combined, "Approach"),
            key="dashboard_approach",
        )

    with c2:
        model = st.selectbox(
            "Model",
            ["All"] + _unique_values(combined, "Model"),
            key="dashboard_model",
        )

    with c3:
        final_source = st.selectbox(
            "Final Source",
            ["All"] + _unique_values(combined, "Final Source"),
            key="dashboard_final_source",
        )

    with c4:
        status = st.selectbox(
            "Status",
            ["All"] + _unique_values(combined, "Status"),
            key="dashboard_status",
        )

    with c5:
        tile = st.selectbox(
            "Tile",
            ["All"] + _unique_values(combined, "Tile"),
            key="dashboard_tile",
        )

    filtered = _filter_frame(
        dataframe=combined,
        approach=approach,
        model=model,
        final_source=final_source,
        status=status,
        tile=tile,
    )

    st.caption(
        f"Showing {len(filtered):,} result row(s) from "
        f"{filtered['Experiment'].nunique() if not filtered.empty else 0:,} "
        "unique experiment(s)."
    )

    return filtered


# ============================================================
# KPI CARDS
# ============================================================

def _render_kpis(
    filtered: pd.DataFrame,
) -> None:

    render_section(
        "B",
        "Performance Overview",
        "Current filtered view of validation and independent final-test performance.",
    )

    experiment_count = (
        int(filtered["Experiment"].nunique())
        if not filtered.empty
        else 0
    )

    final_rows = filtered[
        filtered[FINAL_IOU_COLUMN].notna()
    ]

    final_test_rows = int(
        len(final_rows)
    )

    avg_validation = _safe_mean(
        filtered[VALIDATION_IOU_COLUMN]
    )

    avg_final = _safe_mean(
        filtered[FINAL_IOU_COLUMN]
    )

    highest_final = _safe_max(
        filtered[FINAL_IOU_COLUMN]
    )

    avg_gap = _safe_mean(
        filtered["Generalization Gap"]
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric(
        "Experiments",
        f"{experiment_count:,}",
    )

    c2.metric(
        "Final-test rows",
        f"{final_test_rows:,}",
    )

    c3.metric(
        "Avg Validation IoU",
        _fmt_percent(avg_validation),
    )

    c4.metric(
        "Avg Final IoU",
        _fmt_percent(avg_final),
    )

    c5.metric(
        "Highest Final IoU",
        _fmt_percent(highest_final),
    )

    c6.metric(
        "Avg Val−Final Gap",
        _fmt_percent(avg_gap),
        help=(
            "Validation IoU minus Final IoU. Positive values mean the "
            "validation score is higher than the independent final-test score."
        ),
    )

    st.markdown(
        """
        <div class="metric-chip-row">
            <div class="metric-chip">Primary final metric: Final IoU</div>
            <div class="metric-chip">Precision</div>
            <div class="metric-chip">Recall</div>
            <div class="metric-chip">F1</div>
            <div class="metric-chip">Validation IoU kept separate</div>
            <div class="metric-chip">clDice excluded from dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def _final_metric_frame(
    filtered: pd.DataFrame,
) -> pd.DataFrame:

    frame = filtered[
        filtered[FINAL_IOU_COLUMN].notna()
    ].copy()

    if frame.empty:
        return frame

    frame["Experiment Label"] = (
        frame["Approach"].fillna("Unknown").astype("string")
        + " · "
        + frame["Experiment"].fillna("Unknown").astype("string")
        + " · "
        + frame["Final Source"].fillna("Unknown").astype("string")
    )

    return frame


def _render_final_iou_chart(
    final_frame: pd.DataFrame,
) -> None:

    st.markdown("#### Independent Final IoU by Experiment")

    if final_frame.empty:
        st.info(
            "No completed Final IoU values are available for the current filters."
        )
        return

    plot = final_frame[
        [
            "Experiment Label",
            "Approach",
            "Model",
            "Final Source",
            FINAL_IOU_COLUMN,
        ]
    ].copy()

    plot = plot.sort_values(
        FINAL_IOU_COLUMN,
        ascending=False,
    )

    if alt is not None:
        chart = (
            alt.Chart(plot)
            .mark_bar(
                cornerRadiusEnd=5,
            )
            .encode(
                y=alt.Y(
                    "Experiment Label:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(
                        labelLimit=360,
                    ),
                ),
                x=alt.X(
                    f"{FINAL_IOU_COLUMN}:Q",
                    title="Final IoU (%)",
                    scale=alt.Scale(
                        domain=[0, 100],
                    ),
                ),
                color=alt.Color(
                    "Approach:N",
                    title="Approach",
                ),
                tooltip=[
                    alt.Tooltip("Approach:N"),
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Final Source:N"),
                    alt.Tooltip(
                        f"{FINAL_IOU_COLUMN}:Q",
                        format=".2f",
                    ),
                ],
            )
            .properties(
                height=max(
                    280,
                    min(
                        760,
                        36 * len(plot),
                    ),
                )
            )
        )

        st.altair_chart(
            chart,
            width="stretch",
        )

    else:
        st.bar_chart(
            plot.set_index("Experiment Label")[
                FINAL_IOU_COLUMN
            ],
            width="stretch",
        )


def _render_metric_comparison_chart(
    final_frame: pd.DataFrame,
) -> None:

    st.markdown("#### Final-test Metric Profile")

    if final_frame.empty:
        st.info(
            "No completed final-test metric rows are available."
        )
        return

    metric_columns = [
        metric
        for metric in (
            FINAL_IOU_COLUMN,
            "Precision",
            "Recall",
            "F1",
        )
        if metric in final_frame.columns
        and final_frame[metric].notna().any()
    ]

    if not metric_columns:
        st.info(
            "No final-test metric values are available for the current filters."
        )
        return

    plot = final_frame[
        [
            "Experiment Label",
            "Model",
            "Final Source",
        ]
        + metric_columns
    ].copy()

    melted = plot.melt(
        id_vars=[
            "Experiment Label",
            "Model",
            "Final Source",
        ],
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Value",
    ).dropna(
        subset=["Value"]
    )

    # Keep chart data types explicit.
    melted["Metric"] = melted["Metric"].astype("string")
    melted["Value"] = pd.to_numeric(
        melted["Value"],
        errors="coerce",
    ).astype("float64")

    if alt is not None:
        chart = (
            alt.Chart(melted)
            .mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3,
            )
            .encode(
                x=alt.X(
                    "Experiment Label:N",
                    title=None,
                    axis=alt.Axis(
                        labelAngle=-35,
                        labelLimit=260,
                    ),
                ),
                y=alt.Y(
                    "Value:Q",
                    title="Metric (%)",
                    scale=alt.Scale(
                        domain=[0, 100],
                    ),
                ),
                xOffset="Metric:N",
                color=alt.Color(
                    "Metric:N",
                    title="Metric",
                ),
                tooltip=[
                    alt.Tooltip("Experiment Label:N"),
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Final Source:N"),
                    alt.Tooltip("Metric:N"),
                    alt.Tooltip("Value:Q", format=".2f"),
                ],
            )
            .properties(
                height=390,
            )
        )

        st.altair_chart(
            chart,
            width="stretch",
        )

    else:
        pivot = melted.pivot_table(
            index="Experiment Label",
            columns="Metric",
            values="Value",
            aggfunc="mean",
        )

        st.bar_chart(
            pivot,
            width="stretch",
        )


def _render_validation_vs_final(
    final_frame: pd.DataFrame,
) -> None:

    st.markdown("#### Validation vs Independent Final IoU")

    plot = final_frame.dropna(
        subset=[
            VALIDATION_IOU_COLUMN,
            FINAL_IOU_COLUMN,
        ]
    ).copy()

    if plot.empty:
        st.info(
            "Validation and Final IoU are not simultaneously available "
            "for the current filters."
        )
        return

    if alt is not None:
        points = (
            alt.Chart(plot)
            .mark_circle(
                size=115,
                opacity=0.88,
            )
            .encode(
                x=alt.X(
                    f"{VALIDATION_IOU_COLUMN}:Q",
                    title="Validation IoU (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                y=alt.Y(
                    f"{FINAL_IOU_COLUMN}:Q",
                    title="Final IoU (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                color=alt.Color(
                    "Approach:N",
                    title="Approach",
                ),
                shape=alt.Shape(
                    "Model:N",
                    title="Model",
                ),
                tooltip=[
                    alt.Tooltip("Experiment:N"),
                    alt.Tooltip("Approach:N"),
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Final Source:N"),
                    alt.Tooltip(
                        f"{VALIDATION_IOU_COLUMN}:Q",
                        format=".2f",
                    ),
                    alt.Tooltip(
                        f"{FINAL_IOU_COLUMN}:Q",
                        format=".2f",
                    ),
                    alt.Tooltip(
                        "Generalization Gap:Q",
                        format=".2f",
                    ),
                ],
            )
        )

        reference = pd.DataFrame(
            {
                "x": pd.Series(
                    [0.0, 100.0],
                    dtype="float64",
                ),
                "y": pd.Series(
                    [0.0, 100.0],
                    dtype="float64",
                ),
            }
        )

        equality_line = (
            alt.Chart(reference)
            .mark_line(
                strokeDash=[7, 5],
                opacity=0.55,
            )
            .encode(
                x=alt.X("x:Q"),
                y=alt.Y("y:Q"),
            )
        )

        st.altair_chart(
            equality_line + points,
            width="stretch",
        )

        st.caption(
            "The dashed diagonal represents equal Validation and Final IoU. "
            "Points below the line have lower independent final-test IoU."
        )

    else:
        st.scatter_chart(
            plot,
            x=VALIDATION_IOU_COLUMN,
            y=FINAL_IOU_COLUMN,
            color="Approach",
            width="stretch",
        )


def _render_gap_chart(
    final_frame: pd.DataFrame,
) -> None:

    st.markdown("#### Validation-to-Final Generalization Gap")

    plot = final_frame.dropna(
        subset=["Generalization Gap"]
    ).copy()

    if plot.empty:
        st.info(
            "No generalization-gap values are available for the current filters."
        )
        return

    plot = plot.sort_values(
        "Generalization Gap",
        ascending=False,
    )

    if alt is not None:
        chart = (
            alt.Chart(plot)
            .mark_bar(
                cornerRadiusEnd=4,
            )
            .encode(
                y=alt.Y(
                    "Experiment Label:N",
                    sort="-x",
                    title=None,
                    axis=alt.Axis(
                        labelLimit=360,
                    ),
                ),
                x=alt.X(
                    "Generalization Gap:Q",
                    title="Validation IoU − Final IoU (percentage points)",
                ),
                color=alt.condition(
                    alt.datum["Generalization Gap"] >= 0,
                    alt.value("#f59e0b"),
                    alt.value("#22c55e"),
                ),
                tooltip=[
                    alt.Tooltip("Experiment:N"),
                    alt.Tooltip("Model:N"),
                    alt.Tooltip("Final Source:N"),
                    alt.Tooltip(
                        "Generalization Gap:Q",
                        format=".2f",
                    ),
                ],
            )
            .properties(
                height=max(
                    260,
                    min(
                        700,
                        34 * len(plot),
                    ),
                )
            )
        )

        st.altair_chart(
            chart,
            width="stretch",
        )

    else:
        st.bar_chart(
            plot.set_index("Experiment Label")[
                "Generalization Gap"
            ],
            width="stretch",
        )


# ============================================================
# APPROACH / MODEL SUMMARY
# ============================================================

def _aggregate_summary(
    filtered: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:

    if filtered.empty:
        return pd.DataFrame()

    working = filtered.copy()

    grouped = (
        working.groupby(
            group_column,
            dropna=False,
        )
        .agg(
            Experiments=("Experiment", "nunique"),
            Final_Test_Rows=(FINAL_IOU_COLUMN, "count"),
            Avg_Validation_IoU=(VALIDATION_IOU_COLUMN, "mean"),
            Avg_Final_IoU=(FINAL_IOU_COLUMN, "mean"),
            Avg_Precision=("Precision", "mean"),
            Avg_Recall=("Recall", "mean"),
            Avg_F1=("F1", "mean"),
            Avg_Gap=("Generalization Gap", "mean"),
        )
        .reset_index()
    )

    grouped = grouped.rename(
        columns={
            "Avg_Validation_IoU": "Avg Validation IoU",
            "Avg_Final_IoU": "Avg Final IoU",
            "Avg_Precision": "Avg Precision",
            "Avg_Recall": "Avg Recall",
            "Avg_F1": "Avg F1",
            "Avg_Gap": "Avg Val−Final Gap",
            "Final_Test_Rows": "Final-test Rows",
        }
    )

    numeric_columns = (
        "Avg Validation IoU",
        "Avg Final IoU",
        "Avg Precision",
        "Avg Recall",
        "Avg F1",
        "Avg Val−Final Gap",
    )

    for column in numeric_columns:
        if column in grouped.columns:
            grouped[column] = pd.to_numeric(
                grouped[column],
                errors="coerce",
            ).astype("float64").round(2)

    if group_column in grouped.columns:
        grouped[group_column] = (
            grouped[group_column]
            .astype("string")
            .fillna("—")
        )

    # Explicit integer count columns avoid object dtype inference.
    for count_column in (
        "Experiments",
        "Final-test Rows",
    ):
        if count_column in grouped.columns:
            grouped[count_column] = pd.to_numeric(
                grouped[count_column],
                errors="coerce",
            ).fillna(0).astype("int64")

    return grouped


def _render_summary_tables(
    filtered: pd.DataFrame,
) -> None:

    render_section(
        "D",
        "Approach & Model Summary",
        "Aggregated descriptive statistics for the currently filtered results.",
    )

    t1, t2 = st.tabs(
        [
            "By Approach",
            "By Model",
        ]
    )

    with t1:
        approach_summary = _aggregate_summary(
            filtered,
            "Approach",
        )

        if approach_summary.empty:
            st.info(
                "No approach summary is available."
            )
        else:
            st.dataframe(
                approach_summary,
                hide_index=True,
                width="stretch",
            )

    with t2:
        model_summary = _aggregate_summary(
            filtered,
            "Model",
        )

        if model_summary.empty:
            st.info(
                "No model summary is available."
            )
        else:
            st.dataframe(
                model_summary,
                hide_index=True,
                width="stretch",
            )


# ============================================================
# EXPERIMENT TABLE
# ============================================================

def _render_experiment_table(
    filtered: pd.DataFrame,
) -> None:

    render_section(
        "E",
        "Experiment Explorer",
        (
            "Complete experiment-level traceability. Final IoU replaces the "
            "old Test IoU display name, and clDice is intentionally excluded."
        ),
    )

    if filtered.empty:
        st.info(
            "No experiment rows match the current filters."
        )
        return

    display_columns = _existing_columns(
        filtered,
        MAIN_TABLE_COLUMNS,
    )

    display = filtered[
        display_columns
    ].copy()

    # --------------------------------------------------------
    # Numeric columns remain numeric
    # --------------------------------------------------------

    for column in (
        VALIDATION_IOU_COLUMN,
        FINAL_IOU_COLUMN,
        "Precision",
        "Recall",
        "F1",
        "Generalization Gap",
    ):
        if column in display.columns:
            display[column] = pd.to_numeric(
                display[column],
                errors="coerce",
            ).astype("float64").round(2)

    if "Threshold" in display.columns:
        display["Threshold"] = pd.to_numeric(
            display["Threshold"],
            errors="coerce",
        ).astype("float64").round(3)

    # --------------------------------------------------------
    # Text columns remain text
    #
    # IMPORTANT:
    # Tile is never mixed between int and "—". It is always string here.
    # --------------------------------------------------------

    display = _display_safe_frame(
        display,
        text_columns=(
            "Approach",
            "Experiment",
            "Model",
            "Backend",
            "Source",
            "Tile",
            "Split",
            "Aug",
            "Optuna",
            "Master Optuna",
            "Post-processing",
            "Status",
            "Final Source",
        ),
    )

    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        height=560,
        column_config={
            VALIDATION_IOU_COLUMN: st.column_config.NumberColumn(
                VALIDATION_IOU_COLUMN,
                format="%.2f%%",
            ),
            FINAL_IOU_COLUMN: st.column_config.NumberColumn(
                FINAL_IOU_COLUMN,
                format="%.2f%%",
            ),
            "Precision": st.column_config.NumberColumn(
                "Precision",
                format="%.2f%%",
            ),
            "Recall": st.column_config.NumberColumn(
                "Recall",
                format="%.2f%%",
            ),
            "F1": st.column_config.NumberColumn(
                "F1",
                format="%.2f%%",
            ),
            "Generalization Gap": st.column_config.NumberColumn(
                "Val−Final Gap",
                format="%.2f",
                help=(
                    "Validation IoU minus Final IoU in percentage points."
                ),
            ),
            "Threshold": st.column_config.NumberColumn(
                "Threshold",
                format="%.3f",
            ),
        },
    )

    export = display.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download filtered dashboard CSV",
        data=export,
        file_name="thesis_results_dashboard_filtered.csv",
        mime="text/csv",
        width="content",
        key="dashboard_download_csv",
    )


# ============================================================
# EXPERIMENT DETAIL VIEW
# ============================================================

def _render_experiment_detail(
    filtered: pd.DataFrame,
) -> None:

    render_section(
        "F",
        "Selected Experiment Detail",
        "Inspect one experiment/final-source result without changing any artifact.",
    )

    if filtered.empty:
        st.info(
            "No experiment is available for inspection."
        )
        return

    detail = filtered.copy()

    detail["Selection Label"] = (
        detail["Approach"].fillna("Unknown").astype("string")
        + " · "
        + detail["Experiment"].fillna("Unknown").astype("string")
        + " · "
        + detail["Final Source"].fillna("Unknown").astype("string")
    )

    labels = (
        detail["Selection Label"]
        .astype("string")
        .tolist()
    )

    selected = st.selectbox(
        "Experiment result",
        labels,
        key="dashboard_detail_selection",
    )

    row = detail.loc[
        detail["Selection Label"].eq(selected)
    ].iloc[0]

    a1, a2, a3, a4 = st.columns(4)

    a1.metric(
        "Validation IoU",
        _fmt_percent(
            row.get(
                VALIDATION_IOU_COLUMN
            )
        ),
    )

    a2.metric(
        "Final IoU",
        _fmt_percent(
            row.get(
                FINAL_IOU_COLUMN
            )
        ),
    )

    a3.metric(
        "F1",
        _fmt_percent(
            row.get("F1")
        ),
    )

    a4.metric(
        "Val−Final Gap",
        _fmt_percent(
            row.get(
                "Generalization Gap"
            )
        ),
    )

    b1, b2, b3, b4 = st.columns(4)

    b1.metric(
        "Precision",
        _fmt_percent(
            row.get("Precision")
        ),
    )

    b2.metric(
        "Recall",
        _fmt_percent(
            row.get("Recall")
        ),
    )

    b3.metric(
        "Threshold",
        _fmt_number(
            row.get("Threshold"),
            3,
        ),
    )

    b4.metric(
        "Tile",
        _fmt_text(
            row.get("Tile")
        ),
    )

    # --------------------------------------------------------
    # Arrow-safe detail table
    #
    # Both Field and Value are explicitly strings.
    # This fixes the previous "Expected bytes, got int" error.
    # --------------------------------------------------------

    detail_pairs = (
        ("Approach", row.get("Approach")),
        ("Experiment", row.get("Experiment")),
        ("Model", row.get("Model")),
        ("Backend", row.get("Backend")),
        ("Training Source", row.get("Source")),
        ("Final Source", row.get("Final Source")),
        ("Tile", row.get("Tile")),
        ("Split", row.get("Split")),
        ("Augmentation", row.get("Aug")),
        ("Optuna", row.get("Optuna")),
        ("Master Optuna", row.get("Master Optuna")),
        ("Post-processing", row.get("Post-processing")),
        ("Status", row.get("Status")),
    )

    details = pd.DataFrame(
        {
            "Field": pd.Series(
                [
                    str(field)
                    for field, _ in detail_pairs
                ],
                dtype="string",
            ),
            "Value": pd.Series(
                [
                    _fmt_text(value)
                    for _, value in detail_pairs
                ],
                dtype="string",
            ),
        }
    )

    st.dataframe(
        details,
        hide_index=True,
        width="stretch",
        height=420,
    )


# ============================================================
# MAIN PAGE
# ============================================================

def render() -> None:

    st.markdown(
        DASHBOARD_CSS,
        unsafe_allow_html=True,
    )

    dashboard = st.container(
        key="results_dashboard_shell"
    )

    with dashboard:
        render_header(
            title="GeoAI Thesis Results Dashboard",
            subtitle=(
                "Professional read-only analytics for Building and Road "
                "experiments across validation and independent final testing."
            ),
            eyebrow="CROSS-APPROACH ANALYTICS",
        )

        st.markdown(
            """
            <div class="dashboard-note">
                <b>Scientific reading rule:</b> Validation IoU is shown separately
                from independent <b>Final IoU</b>. Precision, Recall and F1 describe
                the final-test prediction quality. clDice is intentionally excluded
                from this dashboard.
            </div>
            """,
            unsafe_allow_html=True,
        )

        _render_refresh_control()

        combined = _combined_results()

        if combined.empty:
            st.info(
                "No Building or Road experiment artifacts are available yet."
            )
            return

        filtered = _render_filters(
            combined
        )

        if filtered.empty:
            st.warning(
                "No experiment rows match the selected filters."
            )
            return

        _render_kpis(
            filtered
        )

        render_section(
            "C",
            "Performance Visualizations",
            (
                "Final-test overlap, metric profile and validation-to-final "
                "generalization behavior for the current filters."
            ),
        )

        final_frame = _final_metric_frame(
            filtered
        )

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Final IoU",
                "Metric Profile",
                "Validation vs Final",
                "Generalization Gap",
            ]
        )

        with tab1:
            _render_final_iou_chart(
                final_frame
            )

        with tab2:
            _render_metric_comparison_chart(
                final_frame
            )

        with tab3:
            _render_validation_vs_final(
                final_frame
            )

        with tab4:
            _render_gap_chart(
                final_frame
            )

        _render_summary_tables(
            filtered
        )

        _render_experiment_table(
            filtered
        )

        _render_experiment_detail(
            filtered
        )


if __name__ == "__main__":
    render()
