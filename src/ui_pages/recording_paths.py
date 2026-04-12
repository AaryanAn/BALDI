"""Filename generation for screen recordings (matches webcam Record button)."""

from pathlib import Path


def next_recording_path(recordings_dir: Path) -> Path:
    """Next file: recording_{N}.mp4 where N = number of existing .mp4 files."""
    n = len(list(recordings_dir.glob("*.mp4")))
    return recordings_dir / f"recording_{n}.mp4"
