import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ui_pages.recording_paths import next_recording_path


def test_next_recording_path_starts_at_zero(tmp_path):
    p = next_recording_path(tmp_path)
    assert p == tmp_path / "recording_0.mp4"


def test_next_recording_path_increments_with_existing_files(tmp_path):
    (tmp_path / "recording_0.mp4").write_bytes(b"x")
    assert next_recording_path(tmp_path).name == "recording_1.mp4"
    (tmp_path / "recording_1.mp4").write_bytes(b"x")
    assert next_recording_path(tmp_path).name == "recording_2.mp4"
