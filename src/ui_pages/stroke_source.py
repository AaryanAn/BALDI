"""Route Evaluate/Save/Clear between hand (Gestures) and IR (IRTracker) stroke stores."""

from __future__ import annotations

from typing import Protocol


class _PathsProvider(Protocol):
    def snapshot_paths(self) -> list: ...
    def clear_path(self) -> None: ...


def snapshot_active_paths(
    active_source: str,
    gesture: _PathsProvider,
    ir: _PathsProvider | None,
) -> list:
    if ir is not None and active_source == "ir":
        return ir.snapshot_paths()
    return gesture.snapshot_paths()


def clear_active_paths(
    active_source: str,
    gesture: _PathsProvider,
    ir: _PathsProvider | None,
) -> None:
    if ir is not None and active_source == "ir":
        ir.clear_path()
    else:
        gesture.clear_path()
