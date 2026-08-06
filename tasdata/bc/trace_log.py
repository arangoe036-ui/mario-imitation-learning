"""Per-frame episode retention. Written to disk, always, for every episode.

Six consecutive reports stalled because the next question needed per-frame data the previous run
discarded. The pattern was always the same: each question looked answerable from a summary, so the
frames were dropped, and the following question needed them.

Design rules, each earned by a specific failure today:

* **Raw values are persisted alongside every interpretation.** The enemy ID table covers a fraction
  of SMB's enemies and the missing ones fell into an `unknown` bucket that destroyed the evidence.
  Here the raw byte is always kept, so a wrong or incomplete lookup costs a label, not the data.
* **No guard clause silently drops a region.** `clip_test` excluded transitions involving x=0 to
  suppress boundary noise, and a pipe transit is always x -> 0, so the one test that could have
  caught it was blind by construction. Nothing is excluded here; filtering happens at analysis time
  where it can be seen and reversed.
* **y is absolute.** ``ADDR_Y_PIXEL`` alone wraps at 256, which is why a ``y > 200`` pit test could
  never fire.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..ram import ADDR_Y_PIXEL, ADDR_Y_VIEWPORT, on_ground, read_smb, y_absolute

ADDR_X_SPEED = 0x0057
ENEMY_SLOTS = 5
ADDR_ENEMY_TYPE, ADDR_ENEMY_XPAGE, ADDR_ENEMY_XLO = 0x0016, 0x006E, 0x0087
ADDR_ENEMY_YPAGE, ADDR_ENEMY_YLO = 0x00B6, 0x00CF
ADDR_ENEMY_ACTIVE = 0x000F

#: Best-available mapping of SMB enemy type bytes to names.
#:
#: **UNVERIFIED against this ROM.** The previous table covered five IDs and everything else fell
#: through to `unknown`, which made 62 deaths uninterpretable. This is wider but is still a lookup
#: taken from documentation rather than measured here, so the **raw byte is always persisted** and
#: any name in an analysis can be recomputed. Treat a name as a hint and the byte as the datum.
SMB_ENEMY_IDS: dict[int, str] = {
    0x00: "green_koopa", 0x01: "red_koopa", 0x02: "buzzy_beetle", 0x03: "hammer_bro",
    0x04: "goomba_alt", 0x05: "blooper", 0x06: "goomba", 0x07: "green_paratroopa",
    0x08: "grey_cheep", 0x09: "red_cheep", 0x0A: "podoboo", 0x0B: "piranha_plant",
    0x0C: "green_paratroopa_jump", 0x0D: "red_paratroopa", 0x0E: "green_koopa_wall",
    0x0F: "lakitu", 0x10: "spiny", 0x11: "bowser_flame", 0x12: "cheep_school",
    0x13: "bowser", 0x14: "air_bubble", 0x15: "toad_or_princess",
    0x24: "koopa_shell", 0x25: "koopa_shell_moving",
}


@dataclass
class EpisodeTrace:
    """One episode, every frame, plus the enemy field at death."""

    seed: int
    #: (x, y_absolute, speed, buttons, player_state, grounded, world, stage, area)
    #: -- `grounded` appended 2026-08-04; `world`/`stage`/`area` appended 2026-08-05 (block 59)
    frames: list = field(default_factory=list)
    death: dict | None = None
    ended: str = "budget"

    def record(self, obs, byte: int) -> None:
        # Fields are APPENDED, never inserted: every reader indexes f[0] for x, f[3] for buttons and
        # f[4] for player_state, so appending stays backward-compatible with the traces on disk.
        # `grounded` is here because the behaviour statistics that matter -- airborne fraction,
        # A-onsets while grounded, A still held while airborne -- cannot be derived from x and y.
        #
        # `world`, `stage` and `area` are here because **two completions of 1-1 sat on disk
        # mislabelled `stuck` and `budget`** and could not be confirmed from the traces: a stage
        # advance is the only unambiguous evidence a level was finished, and the trace could not
        # record it. Worse, `x_position` is per-AREA, so without `area` a trace silently mixes two
        # coordinate systems the moment Mario enters a pipe -- which is the only route by which
        # anything here has ever completed the level.
        st = read_smb(obs.ram, obs.framecount)
        self.frames.append((int(st.x_position), y_absolute(obs.ram),
                            int(obs.ram[ADDR_X_SPEED]), int(byte), int(st.player_state),
                            int(on_ground(obs.ram)),
                            int(st.world), int(st.stage), int(st.area)))

    def record_death(self, obs) -> None:
        st = read_smb(obs.ram, obs.framecount)
        enemies = []
        for i in range(ENEMY_SLOTS):
            raw = int(obs.ram[ADDR_ENEMY_TYPE + i])
            ex = int(obs.ram[ADDR_ENEMY_XPAGE + i]) * 256 + int(obs.ram[ADDR_ENEMY_XLO + i])
            ey = int(obs.ram[ADDR_ENEMY_YPAGE + i]) * 256 + int(obs.ram[ADDR_ENEMY_YLO + i])
            enemies.append({"slot": i, "raw_id": raw,
                            "name": SMB_ENEMY_IDS.get(raw),      # may be None; raw is the datum
                            "active": int(obs.ram[ADDR_ENEMY_ACTIVE + i]),
                            "x": ex, "y": ey,
                            "dx": ex - int(st.x_position)})
        self.death = {"x": int(st.x_position), "y_absolute": y_absolute(obs.ram),
                      "y_wrapped": int(obs.ram[ADDR_Y_PIXEL]),
                      "y_page": int(obs.ram[ADDR_Y_VIEWPORT]),
                      "player_state": int(st.player_state),
                      "frame_index": len(self.frames), "enemies": enemies}
        self.ended = "died"

    def to_dict(self) -> dict:
        return {"seed": self.seed, "ended": self.ended, "death": self.death,
                "n_frames": len(self.frames), "frames": self.frames}


def write_traces(path: Path | str, traces: list[EpisodeTrace], **meta) -> Path:
    """Persist to disk. Frames are the point; summaries are derived later, never instead."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"schema": "per-frame (x, y_absolute, speed_byte, buttons, player_state, grounded, "
                   "world, stage, area); grounded absent before 2026-08-04; world/stage/area "
                   "absent before 2026-08-05 (block 59) -- a trace without them CANNOT show a "
                   "level completion, and x is per-AREA so it mixes coordinate systems inside a "
                   "pipe; enemy raw_id always persisted",
         "enemy_table_unverified": True, "n_episodes": len(traces), **meta,
         "episodes": [t.to_dict() for t in traces]}, separators=(",", ":")))
    return path


def load_traces(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())
