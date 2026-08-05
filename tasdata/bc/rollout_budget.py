"""The episode terminator, in ONE place. `STALL = 300` used to live in twelve scripts.

**Why this file exists.** Block 57 measured the terminator that had governed every rollout this project ever
ran: end an episode after 300 frames without a new maximum x. Paired against a loosened rule on identical
states and RNG seeds, the *median* furthest position was unchanged (899→900, 900→916) — the walls are real —
but **23–26% of episodes legitimately freeze x for more than 300 frames**, the top decile gained 82–127 px,
and level completions went from 0 to 2 per arm. Worse, the freeze p99 sat exactly at the loosened cap of
1201, so **even the loosened measurement was censored.**

Twelve copies of a constant is how a number becomes untraceable and how a measurement artifact survives
fifty-seven blocks. One definition, imported.

**The values are measured, not guessed** — see `data/freeze_distribution.json` and the constants below.

**Every artifact must record `terminator` beside `measurement_basis`**, because every historical reach figure
in this project is a lower bound and a reader needs to know which rule produced a number. Use
`describe()` for that.
"""
from __future__ import annotations

#: Legacy rule. Retained ONLY so old numbers can be reproduced and labelled; never use for new work.
LEGACY_STALL = 300
LEGACY_CAP_FRAMES = 3000

#: Chosen from the MEASURED distribution of *recovered* freezes -- the longest x-freeze in an episode that
#: was afterwards followed by new progress. That is the quantity a terminator has to clear: how long must
#: you wait before concluding the episode will never recover?
#:
#: Measured over 400 episodes from the 1-1 level start with the terminator off
#: (`data/freeze_and_completion.json`): pooled recovered-freeze median 211/102, p90 1666/582,
#: p99 3523/1469, **p99.9 5848 (script) and 3425 (policy), max 6412**. A first guess of 1800 was
#: BELOW the measured p99.9 and is not used. 6500 sits above the observed maximum.
#:
#: Note the rule is now nearly inert for from-the-start episodes: **400 of 400 ended in death**, none
#: by stall. It is a safety net, not a measurement instrument -- which is the point.
STALL = 6500
#: 1-1 is 3,266 px; the flagpole descent and castle walk add several hundred frames during which x is
#: frozen or absent. A completion scored `stuck` or `budget` is how two of them hid on disk.
CAP_FRAMES = 12_000


def describe(stall: int = STALL, cap: int = CAP_FRAMES) -> dict:
    """The block that belongs in every artifact beside `measurement_basis`."""
    return {"stall_frames": int(stall), "cap_frames": int(cap),
            "legacy_stall": LEGACY_STALL, "legacy_cap_frames": LEGACY_CAP_FRAMES,
            "note": ("every reach figure produced before block 58 used stall=300/cap=3000 and is a "
                     "LOWER BOUND -- including the +367 px encoder depth result and block 56's "
                     "'0 of 720 completions at T=0.7'")}
