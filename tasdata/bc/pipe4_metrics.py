"""Pipe-4 metrics, computed identically for every arm.

One module so the baseline (read off ``data/traces/p1_200.json``) and any distilled policy are
scored by the same code. Every definition here is stated in terms of the trace-log frame tuple
``(x, y_absolute, speed, buttons, player_state)`` so it can be recomputed from any retained run.

Definitions, each named for its semantics per LEDGER.md §2:

* ``A_HOLD_ONSET_WINDOW`` -- an A-hold is counted when its **onset** falls inside x 880-924, the
  trigger range the requirement sweep enumerated. A hold that starts before 880 and continues into
  the window is *not* counted, because the question is whether the policy initiates a long hold at
  the right place. Holds that begin in the window and run past it are counted in full.
* ``stuck_at_pipe4`` -- max_x in 896-928, the band the 29 stuck episodes occupy. This is at the
  face, deliberately: it is a *failure* label, not a clearance threshold.
* ``cleared_past_975`` -- max_x > 975, past pipe 4's far edge. Same threshold the requirement sweep
  and the search scorer used, so the three are comparable.
* ``arrived_at_880`` -- max_x >= 880, the handover. Conditional clearance uses this as denominator.

No guard clause drops a region: every episode contributes to the denominators, and an episode that
never reaches 880 is counted as a non-arrival rather than discarded.
"""

from __future__ import annotations

import numpy as np

from ..buttons import NES_BUTTON_BITS

A_BIT = NES_BUTTON_BITS["A"]
A_HOLD_ONSET_WINDOW = (880, 924)
STUCK_BAND = (896, 928)
CLEAR_X = 975
ARRIVE_X = 880

#: The hold the requirement sweep proved sufficient from real states (min over 39 clearing configs).
REQUIRED_HOLD = 12

#: Clearance thresholds, each named for its semantics and each *past* an obstacle's far edge.
#:
#: `pipe1`/`pipe2` are the project's canonical 470/630 so figures stay comparable with the 21.5%
#: and 62% history. `pipe3` is 735, derived rather than chosen: the max_x histogram of 200 baseline
#: episodes has a 37-episode spike in the 720-735 bin and then **nothing at all in 736-783**, so 735
#: is the last x at which an episode can be stalled against pipe 3's face. `pipe4` is the 975 the
#: requirement sweep and the search scorer both used.
#:
#: `past720` is deliberately absent -- LEDGER.md §3 voids it as an arrival at a face.
PIPE_THRESHOLDS = {"pipe1": 470, "pipe2": 630, "pipe3": 735, "pipe4": CLEAR_X}

#: Expert press rates, for the ratio that exposes a degenerate marginal. See `button_marginals`.
EXPERT_RATES = {"Up": 0.001, "Down": 0.007, "Left": 0.030, "Right": 0.453,
                "Start": 0.0, "Select": 0.0, "B": 0.514, "A": 0.152}


def button_marginals(frames) -> dict:
    """Press rate per button, plus the ratio to the expert's own rate and A-hold occupancy.

    **Required beside every clearance figure** (LEDGER.md §2). A policy that holds A on 85% of frames
    against the expert's 15% clears obstacles by being permanently airborne, and its clearance rate is
    a statement about button rates rather than about skill. That went unnoticed for six reports
    because the marginals were never printed next to the rates.
    """
    from ..buttons import NES_BUTTON_ORDER

    bits = np.asarray([f[3] for f in frames], dtype=np.int64)
    if not len(bits):
        return {"frames": 0}
    rates = {n: float(((bits & NES_BUTTON_BITS[n]) > 0).mean()) for n in NES_BUTTON_ORDER}
    a = (bits & A_BIT) > 0
    runs, i = [], 0
    while i < len(a):
        if a[i] and (i == 0 or not a[i - 1]):
            j = i
            while j < len(a) and a[j]:
                j += 1
            runs.append(j - i)
            i = j
        else:
            i += 1
    return {
        "frames": int(len(bits)),
        "rates": {k: round(v, 3) for k, v in rates.items()},
        "over_expert": {k: (round(rates[k] / EXPERT_RATES[k], 2) if EXPERT_RATES.get(k) else None)
                        for k in rates},
        "frames_inside_a_hold_pct": round(float(a.mean()) * 100, 1),
        "a_holds": hold_stats(runs),
        "expert_rates": EXPERT_RATES,
    }


def clearance(max_xs, thresholds=None) -> dict:
    """Wilson-bounded clearance at each named threshold."""
    from .overnight_lib import wilson

    thresholds = thresholds or PIPE_THRESHOLDS
    xs = np.asarray(list(max_xs))
    n = len(xs)
    out = {}
    for name, th in thresholds.items():
        k = int((xs > th).sum())
        lo, hi = wilson(k, n) if n else (0.0, 0.0)
        out[name] = {"threshold_x": th, "k": k, "n": n,
                     "rate": (k / n) if n else None, "ci": [lo, hi], "method": "Wilson"}
    return out


def a_hold_onsets(frames, window=A_HOLD_ONSET_WINDOW) -> list[int]:
    """Lengths of A-holds whose first pressed frame has x inside `window`.

    The run is followed to its end wherever that falls, so a hold beginning at x=890 and lasting
    30 frames is reported as 30 rather than truncated at the window edge.
    """
    lo, hi = window
    xs = np.asarray([f[0] for f in frames], dtype=np.int64)
    a = (np.asarray([f[3] for f in frames], dtype=np.int64) & A_BIT) > 0
    out = []
    n = len(a)
    i = 0
    while i < n:
        if a[i] and (i == 0 or not a[i - 1]) and lo <= xs[i] <= hi:
            j = i
            while j < n and a[j]:
                j += 1
            out.append(int(j - i))
            i = j
        else:
            i += 1
    return out


def hold_stats(holds: list[int]) -> dict:
    if not holds:
        return {"n": 0, "median": None, "p90": None, "max": None, "frac_ge_required": None}
    h = np.asarray(holds, dtype=float)
    return {
        "n": len(holds),
        "median": float(np.median(h)),
        "p90": float(np.percentile(h, 90)),
        "max": int(h.max()),
        "frac_ge_required": float((h >= REQUIRED_HOLD).mean()),
        "required_hold": REQUIRED_HOLD,
    }


def episode_metrics(frames) -> dict:
    xs = [f[0] for f in frames]
    mx = int(max(xs)) if xs else 0
    return {
        "max_x": mx,
        "arrived_at_880": mx >= ARRIVE_X,
        "stuck_at_pipe4": STUCK_BAND[0] <= mx <= STUCK_BAND[1],
        "cleared_past_975": mx > CLEAR_X,
        "a_holds_880_924": a_hold_onsets(frames),
    }


def arm_metrics(episodes) -> dict:
    """`episodes` is a list of frame-lists (or objects with `.frames`)."""
    rows = [episode_metrics(e.frames if hasattr(e, "frames") else e["frames"]
                            if isinstance(e, dict) else e) for e in episodes]
    n = len(rows)
    holds = [h for r in rows for h in r["a_holds_880_924"]]
    arrived = sum(r["arrived_at_880"] for r in rows)
    cleared = sum(r["cleared_past_975"] for r in rows)
    return {
        "n": n,
        "measurement_basis": "single_life",
        "grounded_enforced": False,          # this is live play; no jump filter is applied
        "stuck_at_pipe4": sum(r["stuck_at_pipe4"] for r in rows),
        "arrived_at_880": arrived,
        "cleared_past_975": cleared,
        "cleared_given_arrived": (cleared / arrived) if arrived else None,
        "x_median": float(np.median([r["max_x"] for r in rows])) if n else None,
        "a_hold_880_924": hold_stats(holds),
        "episodes_with_a_hold_onset": sum(1 for r in rows if r["a_holds_880_924"]),
        "rows": rows,
    }
