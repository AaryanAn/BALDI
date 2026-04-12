import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gestures.pinch_hysteresis import next_pinch_state


def test_pinch_enters_when_close():
    assert next_pinch_state(False, 0.03, enter=0.042, exit=0.062) is True


def test_pinch_stays_open_in_band():
    assert next_pinch_state(True, 0.05, enter=0.042, exit=0.062) is True


def test_pinch_exits_when_far():
    assert next_pinch_state(True, 0.10, enter=0.042, exit=0.062) is False


def test_pinch_stays_open_until_exit_threshold():
    assert next_pinch_state(True, 0.062, enter=0.042, exit=0.062) is True


def test_pinch_not_entered_when_far():
    assert next_pinch_state(False, 0.08, enter=0.042, exit=0.062) is False
