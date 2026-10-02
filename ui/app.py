"""
ui/app.py

Website-style Master UI for the thesis project.

This version uses the EXISTING ui/pages/ folder.
It does not require ui/views/.

Important:
- Streamlit's automatic legacy pages sidebar is hidden.
- Navigation is controlled here like a website.
- Page modules are loaded directly from file paths.
- HTML blocks are dedented before rendering so Streamlit does not show
  raw HTML as Markdown code.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import textwrap

import streamlit as st


# ============================================================
# PATHS
# ============================================================

UI_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = UI_ROOT.parent
PAGES_ROOT = UI_ROOT / "pages"

for value in (
    str(PROJECT_ROOT),
    str(UI_ROOT),
):
    while value in sys.path:
        sys.path.remove(value)

    sys.path.insert(
        0,
        value,
    )


# ============================================================
# LOCAL UTILITIES
# ============================================================

from utils.paths import (
    BUILDINGS_ROOT,
    ROADS_ROOT,
    ensure_master_ui_directories,
)


# ============================================================
# HTML HELPER
# ============================================================

def html(
    content: str,
) -> None:
    """
    Render raw HTML without allowing Markdown to interpret indented
    fragments as code blocks.

    Streamlit's Markdown parser can break multiline HTML at blank lines,
    so the markup is compacted before rendering.
    """
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


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="GeoAI Extraction Platform",
    page_icon="ðŸ›°ï¸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ensure_master_ui_directories()


# ============================================================
# GLOBAL CSS
# ============================================================

css_path = (
    UI_ROOT
    / "styles"
    / "main.css"
)

if css_path.exists():

    css_text = css_path.read_text(
        encoding="utf-8"
    )

    html(
        f"""
        <style>
        {css_text}

        [data-testid="stSidebar"] {{
            display: none !important;
        }}

        [data-testid="stSidebarNav"] {{
            display: none !important;
        }}

        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        </style>
        """
    )


# ============================================================
# DIRECT LOCAL PAGE LOADER
# ============================================================

def load_page_module(
    page_name: str,
):
    """
    Load one file from ui/pages/ directly.
    """

    page_path = (
        PAGES_ROOT
        / f"{page_name}.py"
    )

    if not page_path.exists():
        raise FileNotFoundError(
            f"Master UI page not found:\n{page_path}"
        )

    module_name = (
        f"geoai_master_page_{page_name}"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        page_path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise ImportError(
            f"Could not load page module:\n{page_path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# NAVIGATION STATE
# ============================================================

PAGE_KEYS = (
    "home",
    "buildings",
    "roads",
    "dashboard",
)

PAGE_LABELS = {
    "home": "Home",
    "buildings": "Buildings",
    "roads": "Roads",
    "dashboard": "Results Dashboard",
}

if (
    "master_ui_page"
    not in st.session_state
):
    st.session_state.master_ui_page = "home"


def go_to(
    page: str,
) -> None:

    if page not in PAGE_KEYS:
        page = "home"

    st.session_state.master_ui_page = page
    st.rerun()


# ============================================================
# PROFESSIONAL WEBSITE TOP NAVIGATION
# ============================================================

with st.container(
    key="website_topbar"
):

    brand_col, home_col, buildings_col, roads_col, dashboard_col = (
        st.columns(
            [
                3.5,
                1.0,
                1.25,
                1.0,
                1.35,
            ],
            vertical_alignment="center",
        )
    )

    with brand_col:

        html(
            """
            <div class="site-brand">
                <div class="site-brand-mark">
                    GEO<span>AI</span>
                </div>

                <div class="site-brand-copy">
                    Building & Road Extraction Platform
                </div>
            </div>
            """
        )

    navigation = (
        (
            home_col,
            "Home",
            "home",
            "master_nav_home",
        ),
        (
            buildings_col,
            "Buildings",
            "buildings",
            "master_nav_buildings",
        ),
        (
            roads_col,
            "Roads",
            "roads",
            "master_nav_roads",
        ),
        (
            dashboard_col,
            "Results Dashboard",
            "dashboard",
            "master_nav_dashboard",
        ),
    )

    for (
        column,
        label,
        page_key,
        widget_key,
    ) in navigation:

        with column:

            active = (
                st.session_state.master_ui_page
                == page_key
            )

            if st.button(
                label,
                key=widget_key,
                use_container_width=True,
                type=(
                    "primary"
                    if active
                    else "secondary"
                ),
            ):
                go_to(
                    page_key
                )


# ============================================================
# BACKEND STATUS
# ============================================================

buildings_ok = BUILDINGS_ROOT.exists()
roads_ok = ROADS_ROOT.exists()

html(
    f"""
    <div class="system-strip">

        <div>
            <span class="system-dot {'online' if buildings_ok else 'offline'}"></span>
            Buildings backend
        </div>

        <div>
            <span class="system-dot {'online' if roads_ok else 'offline'}"></span>
            Roads backend
        </div>

        <div class="system-strip-right">
            Multi-source Remote Sensing Â· Deep Learning Â· GIS
        </div>

    </div>
    """
)


# ============================================================
# PAGE RENDER
# ============================================================

current_page = (
    st.session_state.master_ui_page
)

try:

    module = load_page_module(
        current_page
    )

    render_function = getattr(
        module,
        "render",
        None,
    )

    if not callable(
        render_function
    ):
        st.error(
            f"ui/pages/{current_page}.py does not contain render()."
        )

    else:
        render_function()

except Exception as exc:

    page_label = PAGE_LABELS.get(
        current_page,
        current_page,
    )

    st.error(
        f"Could not load the {page_label} page."
    )

    st.exception(
        exc
    )

