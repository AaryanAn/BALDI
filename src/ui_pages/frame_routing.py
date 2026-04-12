"""Which annotated frame is written to disk while Record is active."""

import numpy as np


def raw_frame_for_active_source(
    active_source: str,
    gesture_annotated: np.ndarray,
    ir_annotated: np.ndarray,
) -> np.ndarray:
    """Return the BGR frame that should be fed to VideoWriter for the current UI source."""
    if active_source == "ir":
        return ir_annotated
    return gesture_annotated
