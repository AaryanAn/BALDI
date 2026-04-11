#!/usr/bin/env python3
"""
Import samples from origin/templates-Declan into src/team_templates on top of base commit.

- Files that exist only on Declan: copied as-is.
- Same path but different bytes vs base: Declan's blob is stored under the next free index.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "src" / "team_templates"
BRANCH = "origin/templates-Declan"
BASE = "e11bd4a842e47b6511b2baf0c31562948fe28ada"


def git_show(branch: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{branch}:{path}"], cwd=REPO)


def tree_paths(ref: str, prefix: str) -> set[str]:
    out = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", ref, prefix],
        cwd=REPO,
    ).decode()
    return {p for p in out.strip().splitlines() if p}


def next_free_index(letter_dir: Path) -> int:
    nums = []
    for p in letter_dir.glob("*.npy"):
        try:
            nums.append(int(p.stem))
        except ValueError:
            continue
    return max(nums) + 1 if nums else 0


def main() -> int:
    e11 = tree_paths(BASE, "src/team_templates")
    dec = tree_paths(BRANCH, "src/team_templates")
    only_dec = sorted(dec - e11)
    both = sorted(e11 & dec)

    n_copy = 0
    for rel in only_dec:
        if ".gitkeep" in rel:
            continue
        dest = REPO / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(git_show(BRANCH, rel))
        n_copy += 1

    n_renum = 0
    for rel in both:
        if not rel.endswith(".npy"):
            continue
        r = subprocess.run(
            ["git", "diff", "--quiet", BASE, BRANCH, "--", rel],
            cwd=REPO,
        )
        if r.returncode == 0:
            continue
        letter = Path(rel).parts[2]
        dest_dir = ROOT / letter
        dest_dir.mkdir(parents=True, exist_ok=True)
        idx = next_free_index(dest_dir)
        (dest_dir / f"{idx}.npy").write_bytes(git_show(BRANCH, rel))
        json_rel = rel[:-4] + ".json"
        try:
            raw = git_show(BRANCH, json_rel)
            meta = json.loads(raw.decode())
            meta["index"] = idx
            meta["label"] = letter
            (dest_dir / f"{idx}.json").write_text(json.dumps(meta))
        except Exception:
            (dest_dir / f"{idx}.json").write_text(json.dumps({"label": letter, "index": idx}))
        n_renum += 1

    print(f"Declan: copied {n_copy} new paths, added {n_renum} alternate samples for conflicting indices")
    return 0


if __name__ == "__main__":
    sys.exit(main())
