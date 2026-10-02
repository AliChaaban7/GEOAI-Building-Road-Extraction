"""
ui/utils/process_runner.py

Isolated backend subprocess execution for the Master UI.

Why subprocesses?
-----------------
Buildings and Roads each use a top-level Python package named `src`.
Importing both backends into one Streamlit process risks module collisions.

Therefore:
    Master UI
        ├── subprocess -> Buildings backend
        └── subprocess -> Roads backend

No backend training module is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Iterable

import streamlit as st

from ui.utils.paths import PROJECT_ROOT


# ============================================================
# COMMAND SPEC
# ============================================================

@dataclass(frozen=True)
class CommandSpec:
    title: str
    command: list[str]
    log_path: Path
    cwd: Path = PROJECT_ROOT


# ============================================================
# HELPERS
# ============================================================

def python_executable() -> str:
    """
    Use the same Python interpreter running Streamlit.
    """

    return sys.executable


def command_text(
    command: Iterable[str],
) -> str:
    """
    Human-readable command string for logs/debugging.
    """

    parts = []

    for item in command:
        item = str(
            item
        )

        parts.append(
            f'"{item}"'
            if " " in item
            else item
        )

    return " ".join(
        parts
    )


# ============================================================
# STREAMED EXECUTION
# ============================================================

def run_streamed(
    spec: CommandSpec,
) -> int:
    """
    Execute a backend command and stream the most recent output into UI.

    Full output is persisted to `spec.log_path`.
    """

    spec.log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    st.markdown(
        f"**{spec.title}**"
    )

    status_box = st.empty()
    command_box = st.empty()
    output_box = st.empty()

    status_box.info(
        "Running…"
    )

    command_box.caption(
        command_text(
            spec.command
        )
    )

    lines: list[str] = []

    try:
        with spec.log_path.open(
            "w",
            encoding="utf-8",
        ) as log_file:

            process = subprocess.Popen(
                spec.command,
                cwd=str(
                    spec.cwd
                ),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            if process.stdout is None:
                raise RuntimeError(
                    "Backend process did not expose stdout."
                )

            for line in process.stdout:
                clean = line.rstrip(
                    "\n"
                )

                lines.append(
                    clean
                )

                log_file.write(
                    line
                )

                log_file.flush()

                output_box.code(
                    "\n".join(
                        lines[-140:]
                    ),
                    language="text",
                )

            code = process.wait()

    except Exception as exc:
        status_box.error(
            f"Execution failed: {exc}"
        )
        return 1

    if code == 0:
        status_box.success(
            "Completed."
        )
    else:
        status_box.error(
            f"Failed with exit code {code}."
        )

    return int(
        code
    )
