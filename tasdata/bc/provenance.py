"""What recipe produced this checkpoint? Recorded at save time, because it cannot be recovered later.

`data/bc_phase1/runlength.pt` carries two blocks of conclusions and records **no `steps`, no `batch`, no
`seed`, and no commit.** When its timing lift turned out not to reproduce, "an outlier seed" and "an
unrecorded recipe difference" fitted the evidence equally well and there was no way to tell them apart. The
finding it supported was voided under either reading, but *why* it happened is still unknown and now
unknowable for that file.

Five fields, written into every checkpoint from here:

* `steps`, `batch`, `seed` -- the three that were missing
* `git_sha` + `git_dirty` -- which code, and whether it was committed
* `frame_size` -- resolution, now that it is a flag rather than a constant

`git_dirty` matters as much as the SHA: a clean SHA means the run is reproducible from history, a dirty one
means it is not, and silently recording only the SHA would imply the first when the second is true.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def git_state(root: Path | None = None) -> dict:
    """Commit and working-tree cleanliness. Never raises -- absent git is recorded, not fatal."""
    root = Path(root or ROOT)
    out: dict = {"git_sha": None, "git_dirty": None}
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if sha.returncode == 0:
            out["git_sha"] = sha.stdout.strip()
            st = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=20)
            if st.returncode == 0:
                out["git_dirty"] = bool(st.stdout.strip())
    except Exception as e:                          # noqa: BLE001 -- provenance must never break a save
        out["git_error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def recipe(*, steps: int, batch: int, seed: int, frame_size: int, **extra) -> dict:
    """The block of fields every saved checkpoint must carry."""
    return {"steps": int(steps), "batch": int(batch), "seed": int(seed),
            "frame_size": int(frame_size), **git_state(), **extra}
