#!/usr/bin/env python3
"""
Copy src/team_templates/<Letter>/*.npy into src/templates/<lowercase>/
with new numeric indices (next free after existing files).

Run from repo root after git fetch. Does not delete sources unless --remove-sources.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TEAM = REPO / "src" / "team_templates"
MAIN = REPO / "src" / "templates"


def next_free_index(letter_dir: Path) -> int:
    m = -1
    for p in letter_dir.glob("*.npy"):
        try:
            m = max(m, int(p.stem))
        except ValueError:
            continue
    return m + 1


def merge_one_letter(team_letter_dir: Path, dest: Path) -> int:
    stems = sorted(
        {p.stem for p in team_letter_dir.glob("*.npy")},
        key=lambda s: int(s) if s.isdigit() else s,
    )
    n = 0
    idx = next_free_index(dest)
    for stem in stems:
        src = team_letter_dir / f"{stem}.npy"
        if not src.is_file():
            continue
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / f"{idx}.npy")
        js = team_letter_dir / f"{stem}.json"
        if js.is_file():
            try:
                meta = json.loads(js.read_text())
                meta["index"] = idx
                meta["label"] = dest.name
                (dest / f"{idx}.json").write_text(json.dumps(meta))
            except Exception:
                shutil.copy2(js, dest / f"{idx}.json")
        else:
            (dest / f"{idx}.json").write_text(json.dumps({"label": dest.name, "index": idx}))
        idx += 1
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--remove-sources",
        action="store_true",
        help="Delete merged .npy/.json from src/team_templates (keeps .gitkeep if present)",
    )
    args = ap.parse_args()

    if not TEAM.is_dir():
        print("No src/team_templates", file=sys.stderr)
        return 1

    total = 0
    for team_letter_dir in sorted(TEAM.iterdir(), key=lambda p: p.name):
        if not team_letter_dir.is_dir() or team_letter_dir.name.startswith("."):
            continue
        name = team_letter_dir.name
        if len(name) != 1 or not name.isalpha():
            print(f"skip (not single letter): {name}")
            continue
        dest = MAIN / name.lower()
        n = merge_one_letter(team_letter_dir, dest)
        print(f"{name} -> {dest.name}: +{n} samples")
        total += n

    print(f"Total merged: {total} samples into {MAIN}")

    if args.remove_sources:
        for team_letter_dir in sorted(TEAM.iterdir(), key=lambda p: p.name):
            if not team_letter_dir.is_dir() or team_letter_dir.name.startswith("."):
                continue
            for p in team_letter_dir.glob("*"):
                if p.name == ".gitkeep":
                    continue
                p.unlink()
        print("Removed merged files from team_templates (dirs may be empty)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
