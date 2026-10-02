"""
Roads DeepLabV3 / DeepLabV3+ package exports.

This package initializer intentionally avoids hard-coded imports of optional
classes such as ``DeepLabV3PlusDecoder``.

Why
---
The implementation in ``deeplabv3.py`` is the source of truth.  Keeping a
second hand-written symbol list here can become stale and can make the whole
package fail to import before the model factory even has a chance to discover
the available builder.

The package therefore forwards public attributes to ``deeplabv3.py``
dynamically.  Existing callers can continue importing through:

    src.models.deeplabv3

while the real implementation remains in:

    src.models.deeplabv3.deeplabv3
"""

from __future__ import annotations

from . import deeplabv3 as _impl


# ---------------------------------------------------------------------------
# PUBLIC EXPORTS
# ---------------------------------------------------------------------------
#
# Prefer an explicit __all__ supplied by the implementation.  If the
# implementation does not define one, expose its non-private attributes.
#
# We deliberately DO NOT name DeepLabV3PlusDecoder (or any other optional
# class) here.  A symbol is available from this package only when the current
# implementation actually defines it.
# ---------------------------------------------------------------------------

_impl_all = getattr(
    _impl,
    "__all__",
    None,
)

if _impl_all is None:
    __all__ = tuple(
        name
        for name in vars(_impl)
        if not name.startswith("_")
    )
else:
    __all__ = tuple(
        str(name)
        for name in _impl_all
    )


def __getattr__(
    name: str,
):
    """
    Forward package-level attribute access to ``deeplabv3.py``.

    This keeps the package compatible with factories that use patterns such as:

        import src.models.deeplabv3 as module
        hasattr(module, "build_model")
        getattr(module, "build_model")

    without maintaining a duplicate list of implementation symbols.
    """

    try:
        return getattr(
            _impl,
            name,
        )

    except AttributeError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}. "
            "The requested symbol is also not defined by "
            "src.models.deeplabv3.deeplabv3."
        ) from exc


def __dir__():
    """
    Include forwarded implementation symbols in dir(package).
    """

    return sorted(
        set(
            globals()
        )
        | set(
            dir(
                _impl
            )
        )
    )
