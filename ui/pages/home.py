"""
ui/pages/home.py

Professional website-style landing page for the GeoAI Master UI.
"""

from __future__ import annotations

import textwrap

import streamlit as st


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
# NAVIGATION
# ============================================================

def _go(
    page: str,
) -> None:

    st.session_state.master_ui_page = page
    st.rerun()


# ============================================================
# PAGE
# ============================================================

def render() -> None:

    # ========================================================
    # HERO
    # ========================================================

    html(
        """
        <section class="web-hero">

            <div class="web-hero-kicker">
                MASTER THESIS Â· GEOAI Â· REMOTE SENSING
            </div>

            <div class="web-hero-title">
                Deep Learning for
                <span>Building & Road Extraction</span>
            </div>

            <div class="web-hero-copy">
                A unified research platform for training, validating,
                optimizing and evaluating geospatial deep-learning models
                across aerial, satellite and drone imagery.
            </div>

            <div class="web-hero-tags">
                <span>Semantic Segmentation</span>
                <span>Instance Segmentation</span>
                <span>ArcGIS Pro</span>
                <span>PyTorch</span>
                <span>Optuna</span>
                <span>GIS Evaluation</span>
            </div>

        </section>
        """
    )

    # ========================================================
    # OVERVIEW METRICS
    # ========================================================

    k1, k2, k3, k4 = st.columns(
        4
    )

    cards = (
        (
            k1,
            "02",
            "Extraction Approaches",
            "Buildings + Roads",
        ),
        (
            k2,
            "06",
            "Model Families",
            "CNN Â· Instance Â· Foundation",
        ),
        (
            k3,
            "03",
            "Imagery Sources",
            "Aerial Â· Satellite Â· Drone",
        ),
        (
            k4,
            "01",
            "Unified Platform",
            "Training â†’ GIS Results",
        ),
    )

    for (
        column,
        value,
        label,
        detail,
    ) in cards:

        with column:

            html(
                f"""
                <div class="metric-card">

                    <div class="metric-value">
                        {value}
                    </div>

                    <div class="metric-label">
                        {label}
                    </div>

                    <div class="metric-detail">
                        {detail}
                    </div>

                </div>
                """
            )

    # ========================================================
    # RESEARCH MODULES
    # ========================================================

    html(
        """
        <div class="web-section-heading">

            <div class="web-section-kicker">
                RESEARCH MODULES
            </div>

            <div class="web-section-title">
                Choose an extraction workflow
            </div>

            <div class="web-section-copy">
                Each approach keeps its own data, model configuration,
                checkpoints and outputs while sharing one professional
                thesis-level interface.
            </div>

        </div>
        """
    )

    left, right = st.columns(
        2,
        gap="large",
    )

    # --------------------------------------------------------
    # BUILDINGS
    # --------------------------------------------------------

    with left:

        html(
            """
            <div class="module-card building-module">

                <div class="module-card-top">
                    <div class="module-number">APPROACH 01</div>
                    <div class="module-icon">â–¦</div>
                </div>

                <div class="module-name">
                    Building Extraction
                </div>

                <div class="module-description">
                    Complete building extraction workflow with training,
                    validation optimization, independent final testing,
                    quantitative results and GIS visualization.
                </div>

                <div class="module-models">
                    <span>U-Net</span>
                    <span>DeepLabV3+</span>
                    <span>Mask R-CNN</span>
                    <span>SAM-LoRA</span>
                </div>

            </div>
            """
        )

        if st.button(
            "Open Building Extraction  â†’",
            key="home_open_buildings",
            use_container_width=True,
            type="primary",
        ):
            _go(
                "buildings"
            )

    # --------------------------------------------------------
    # ROADS
    # --------------------------------------------------------

    with right:

        html(
            """
            <div class="module-card road-module">

                <div class="module-card-top">
                    <div class="module-number">APPROACH 02</div>
                    <div class="module-icon">â•±â•²</div>
                </div>

                <div class="module-name">
                    Road Extraction
                </div>

                <div class="module-description">
                    Mixed-source road-surface extraction with validation-only
                    optimization and one frozen configuration for aerial,
                    satellite and drone final testing.
                </div>

                <div class="module-models">
                    <span>U-Net</span>
                    <span>DeepLabV3+</span>
                    <span>ConnectNet</span>
                    <span>MultiTask</span>
                    <span>SAM-LoRA</span>
                </div>

            </div>
            """
        )

        if st.button(
            "Open Road Extraction  â†’",
            key="home_open_roads",
            use_container_width=True,
            type="primary",
        ):
            _go(
                "roads"
            )

    # ========================================================
    # METHODOLOGY
    # ========================================================

    html(
        """
        <div class="web-section-heading workflow-heading">

            <div class="web-section-kicker">
                METHODOLOGY
            </div>

            <div class="web-section-title">
                Controlled experimental workflow
            </div>

        </div>

        <div class="website-workflow">

            <div class="workflow-node">
                <div class="workflow-number">01</div>
                <div class="workflow-name">Data</div>
                <div class="workflow-copy">
                    Multi-source imagery and labeled masks
                </div>
            </div>

            <div class="workflow-arrow">â†’</div>

            <div class="workflow-node">
                <div class="workflow-number">02</div>
                <div class="workflow-name">Train</div>
                <div class="workflow-copy">
                    Learn model weights
                </div>
            </div>

            <div class="workflow-arrow">â†’</div>

            <div class="workflow-node">
                <div class="workflow-number">03</div>
                <div class="workflow-name">Validation</div>
                <div class="workflow-copy">
                    Select model settings
                </div>
            </div>

            <div class="workflow-arrow">â†’</div>

            <div class="workflow-node">
                <div class="workflow-number">04</div>
                <div class="workflow-name">Optimize</div>
                <div class="workflow-copy">
                    Optuna Â· threshold Â· post-processing
                </div>
            </div>

            <div class="workflow-arrow">â†’</div>

            <div class="workflow-node">
                <div class="workflow-number">05</div>
                <div class="workflow-name">Final Test</div>
                <div class="workflow-copy">
                    Independent geographic evaluation
                </div>
            </div>

            <div class="workflow-arrow">â†’</div>

            <div class="workflow-node">
                <div class="workflow-number">06</div>
                <div class="workflow-name">GIS Results</div>
                <div class="workflow-copy">
                    Metrics Â· polygons Â· visualization
                </div>
            </div>

        </div>
        """
    )

    # ========================================================
    # RESULTS DASHBOARD
    # ========================================================

    html(
        """
        <div class="comparison-hero">

            <div class="comparison-kicker">
                THESIS RESULTS
            </div>

            <div class="comparison-title">
                Compare experiments in one dashboard
            </div>

            <div class="comparison-copy">
                Inspect validation and independent final-test metrics
                from Building and Road experiments in one read-only view.
            </div>

        </div>
        """
    )

    if st.button(
        "Open Results Dashboard",
        key="home_open_dashboard",
        use_container_width=True,
    ):
        _go(
            "dashboard"
        )


# ============================================================
# DIRECT STREAMLIT PAGE SUPPORT
# ============================================================

if __name__ == "__main__":
    render()

