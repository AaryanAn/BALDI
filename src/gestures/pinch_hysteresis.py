"""Thumb–index pinch hysteresis (normalized distance). Pure logic for tests + Gestures."""


def next_pinch_state(
    was_pinching: bool,
    dist_norm: float,
    enter: float,
    exit: float,
) -> bool:
    """
    dist_norm: thumb–index distance divided by min(frame width, height).

    Enter pinch when not pinching and dist_norm < enter.
    Exit pinch when pinching and dist_norm > exit.
    """
    if was_pinching:
        if dist_norm > exit:
            return False
        return True
    if dist_norm < enter:
        return True
    return False
