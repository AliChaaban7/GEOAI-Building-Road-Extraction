"""
ui/components/header.py

Shared headings for the Master UI.

HTML is compacted before rendering so Streamlit never interprets
indented HTML as Markdown code.
"""

from __future__ import annotations

import textwrap

import streamlit as st


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


def render_header(
    title: str,
    subtitle: str,
    eyebrow: str | None = None,
) -> None:

    eyebrow_html = (
        f'<div class="hero-eyebrow">{eyebrow}</div>'
        if eyebrow
        else ""
    )

    _html(
        f"""
        <section class="hero">
            {eyebrow_html}
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </section>
        """
    )


def render_section(
    number: str | int,
    title: str,
    subtitle: str,
) -> None:

    _html(
        f"""
        <div class="section-header">
            <div class="section-index">{number}</div>
            <div class="section-content">
                <div class="section-title">{title}</div>
                <div class="section-subtitle">{subtitle}</div>
            </div>
        </div>
        """
    )


def render_subsection(
    title: str,
    subtitle: str | None = None,
) -> None:

    subtitle_html = (
        f'<div class="subsection-subtitle">{subtitle}</div>'
        if subtitle
        else ""
    )

    _html(
        f"""
        <div class="subsection-header">
            <div class="subsection-title">{title}</div>
            {subtitle_html}
        </div>
        """
    )


def render_status_banner(
    title: str,
    text: str,
    status: str = "info",
) -> None:

    status = str(
        status
    ).strip().lower()

    if status not in {
        "info",
        "success",
        "warning",
        "neutral",
    }:
        status = "neutral"

    _html(
        f"""
        <div class="status-banner status-{status}">
            <div class="status-banner-title">{title}</div>
            <div class="status-banner-text">{text}</div>
        </div>
        """
    )
