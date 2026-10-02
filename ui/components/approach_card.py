"""
ui/components/approach_card.py

Reusable approach-selection card for the Master UI Home page.

Used for:
- Approach 1 â€” Building Extraction
- Approach 2 â€” Road Extraction
"""

from __future__ import annotations

from html import escape

import streamlit as st


def render_approach_card(
    title: str,
    subtitle: str,
    models: list[str],
    icon: str,
    target_page: str,
    accent: str,
    key: str,
) -> None:
    """
    Render one thesis-approach card and navigation button.

    Parameters
    ----------
    title:
        Visible approach title.

    subtitle:
        Short secondary label such as "Approach 1".

    models:
        Models shown inside the card.

    icon:
        Emoji/icon displayed at the top of the card.

    target_page:
        Page name stored in st.session_state.master_ui_page.

    accent:
        CSS accent color for this card.

    key:
        Unique Streamlit button key.
    """

    safe_title = escape(str(title))
    safe_subtitle = escape(str(subtitle))
    safe_icon = escape(str(icon))
    safe_accent = escape(str(accent))

    model_items = "".join(
        f"<span>{escape(str(model))}</span>"
        for model in models
    )

    st.markdown(
        f"""
        <div
            class="approach-card"
            style="--approach-accent:{safe_accent};"
        >

            <div class="approach-icon">
                {safe_icon}
            </div>

            <div class="approach-title">
                {safe_title}
            </div>

            <div class="approach-subtitle">
                {safe_subtitle}
            </div>

            <div class="approach-model-list">
                {model_items}
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        f"Open {title}",
        key=key,
        use_container_width=True,
    ):
        st.session_state.master_ui_page = target_page
        st.rerun()

