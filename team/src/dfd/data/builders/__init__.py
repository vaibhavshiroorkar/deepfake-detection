"""Manifest builders, one module per dataset.

Every builder exposes ``build(data_root: Path) -> tuple[list[Record], dict]`` where the dict is a report
of measured counts. Builders for datasets whose files have not been seen raise ``LayoutNotVerified``.
"""

from __future__ import annotations


class LayoutNotVerified(NotImplementedError):
    """The folder layout of this dataset has not been seen, so no builder is implemented yet."""
