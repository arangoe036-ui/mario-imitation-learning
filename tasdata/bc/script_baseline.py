"""§3: score every arm against the best fixed-rate script, per obstacle.

**"Clears pipe N" cannot distinguish learning from tuning a marginal.** A three-button script -- Right+B
held, A flipped as a coin at p=0.85 -- matches or beats every learned checkpoint in this project at pipes
1 and 2. So a raw clearance rate is not evidence of skill, and neither is an improvement in one.

Worse, the self-imitation acceptance filter scored rollouts on ``gained = max_x - start_x``, which is
**achievable by raising the A-rate alone**. The training loop has therefore been optimising the marginal
all along; death escalation, the reckless models and the composition "gain" are one mechanism.

This module makes degeneracy worth zero by construction:

* :func:`advantage` -- clearance minus the best fixed-rate script's clearance at the same obstacle, with
  a Newcombe interval. Reported beside every rate (LEDGER.md §2 requires `vs_script`).
* :func:`rollout_credit` -- the replacement training signal. An obstacle cleared is worth
  ``1 - p_script(obstacle)``, so clearing something the script clears 82.5% of the time earns 0.175 while
  clearing pipe 4 (script 6%) earns 0.94. **A policy that raises its A-rate to look like the script earns
  almost nothing.**

The baseline is **measured, never assumed**: :func:`build` reads the script arms out of the artifacts on
disk and takes the per-obstacle maximum over every fixed-rate arm at n=200. Taking the max is deliberate
-- the opponent is the *strongest* fixed-rate script at each obstacle, not a convenient one. If a future
arm beats the current table, rebuild and the bar rises.
"""

from __future__ import annotations

import json
from pathlib import Path

from .overnight_lib import diff_ci, wilson
from .pipe4_metrics import PIPE_THRESHOLDS

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "data/script_baseline.json"

#: Artifacts that contain fixed-rate script arms, and how to reach them.
#: (path, dotted-ish accessor description, arm-name predicate)
SOURCES = (
    ("data/p1_script_control.json", "arms"),
    ("data/p2_marginals.json", "arms"),
    ("data/p1_control_ladder.json", "arms"),
)


def _iter_script_arms():
    """Yield (source, arm_label, n, clearance-dict) for every *fixed-rate* arm on disk."""
    for rel, key in SOURCES:
        p = ROOT / rel
        if not p.exists():
            continue
        blob = json.loads(p.read_text())
        for label, row in (blob.get(key) or {}).items():
            if not isinstance(row, dict) or "clearance" not in row:
                continue
            # a learned arm is anything carrying a checkpoint; those are not the opponent
            if row.get("learned") or row.get("checkpoint"):
                continue
            n = row.get("n")
            if not n or n < 200:          # n=20 screens must not set the bar
                continue
            yield rel, label, n, row["clearance"]


def build(write: bool = True) -> dict:
    """Per-obstacle best fixed-rate script clearance, measured from artifacts on disk."""
    best: dict[str, dict] = {}
    considered = []
    for rel, label, n, cl in _iter_script_arms():
        considered.append(f"{rel}:{label} (n={n})")
        for ob, th in PIPE_THRESHOLDS.items():
            row = cl.get(ob)
            if not row:
                continue
            cur = best.get(ob)
            if cur is None or row["rate"] > cur["rate"]:
                lo, hi = wilson(row["k"], n)
                best[ob] = {"obstacle": ob, "threshold_x": th, "k": row["k"], "n": n,
                            "rate": row["rate"], "ci": [lo, hi], "method": "Wilson",
                            "arm": label, "source": rel}
    out = {"note": "best fixed-rate script clearance per obstacle; the opponent is the strongest "
                   "script at each obstacle, taken independently per obstacle",
           "arms_considered": considered,
           "measurement_basis": "single_life", "baseline": best}
    if write and best:
        CACHE.write_text(json.dumps(out, indent=2, default=str))
    return out


def baseline(rebuild: bool = False) -> dict:
    """The per-obstacle table, rebuilt from artifacts if missing or if asked."""
    if rebuild or not CACHE.exists():
        return build()["baseline"]
    return json.loads(CACHE.read_text())["baseline"]


def advantage(k: int, n: int, obstacle: str, table: dict | None = None) -> dict:
    """Clearance minus the best fixed-rate script's, at one obstacle. Newcombe interval."""
    table = table or baseline()
    b = table.get(obstacle)
    if not b:
        return {"obstacle": obstacle, "error": "no script baseline for this obstacle"}
    lo, hi = diff_ci(b["k"], b["n"], k, n)
    return {"obstacle": obstacle, "threshold_x": b["threshold_x"],
            "rate": k / n if n else None, "script_rate": b["rate"], "script_arm": b["arm"],
            "advantage_pp": ((k / n) - b["rate"]) * 100 if n else None,
            "ci_pp": [lo * 100, hi * 100], "method": "Newcombe",
            "beats_script": bool(lo > 0), "loses_to_script": bool(hi < 0)}


def vs_script(max_xs, table: dict | None = None) -> dict:
    """`vs_script` block for an arm: per-obstacle advantage over the best fixed-rate script."""
    table = table or baseline()
    xs = list(max_xs)
    n = len(xs)
    out = {"n": n, "measurement_basis": "single_life", "per_obstacle": {}}
    for ob, th in PIPE_THRESHOLDS.items():
        k = sum(1 for x in xs if x > th)
        out["per_obstacle"][ob] = advantage(k, n, ob, table)
    beats = [ob for ob, r in out["per_obstacle"].items() if r.get("beats_script")]
    out["beats_script_at"] = beats
    out["beats_script_at_pipe2"] = "pipe2" in beats
    return out


#: Levels for which a measured fixed-rate script baseline exists. The obstacle thresholds in
#: `PIPE_THRESHOLDS` are **1-1 x-coordinates**; applying them to another level's x would be
#: meaningless, so credit is refused rather than faked. See `MISSING_BASELINE_NOTE`.
BASELINE_LEVELS = frozenset({"1-1"})

MISSING_BASELINE_NOTE = (
    "484 of the 500 trajectory start points are outside 1-1 and have no measured script baseline, so "
    "`rollout_credit` returns None for them and the acceptance filter must exclude them. The old "
    "signal covered all 500 but was the gameable one. The fix is a per-start script *reach* table -- "
    "run the fixed-rate script from each start point and credit a rollout by its quantile in that "
    "script's max_x distribution, which needs no terrain knowledge and no per-level thresholds."
)


def rollout_credit(max_x: int, deaths: int = 0, table: dict | None = None,
                   death_penalty: float = 0.5, label: str = "1-1") -> float | None:
    """Training signal for the self-imitation acceptance filter.

    An obstacle cleared is worth ``1 - p_script(obstacle)``: what a fixed-rate script would have got
    for free is subtracted, so **raising the A-rate earns close to nothing**. Replaces
    ``gained = max_x - start_x``, which raising the A-rate maximises directly.

    Returns **None** when `label` has no measured script baseline -- the caller must drop the rollout
    rather than fall back to progress, because progress is the signal being retired.
    """
    if label not in BASELINE_LEVELS:
        return None
    table = table or baseline()
    credit = 0.0
    for ob in PIPE_THRESHOLDS:
        if max_x > PIPE_THRESHOLDS[ob]:
            b = table.get(ob)
            credit += (1.0 - b["rate"]) if b else 1.0
    return credit - death_penalty * deaths


REACH_TABLE = ROOT / "data/reach_table.json"


def reach_table(path: Path | None = None) -> dict:
    """Per-start script reach distributions, keyed ``"<seed>:<frame_index>"``."""
    p = path or REACH_TABLE
    if not p.exists():
        raise FileNotFoundError(
            f"{p.name} missing -- run scripts/build_reach_table.py. Falling back to progress is "
            "forbidden: progress is the signal being retired.")
    return json.loads(p.read_text())["states"]


def reach_quantile(max_x: float, state_key: str, table: dict) -> float | None:
    """Where a rollout lands in the script's own distribution from the same start.

    Mid-quantile, so ties are credited half -- matching the script exactly earns 0.5 rather than 1.0.
    Bounded on [0, 1], which is the saturation the directive flagged; :func:`reach_margin` is the
    unbounded fallback.
    """
    row = table.get(state_key)
    if not row:
        return None
    xs = row["script_max_x"]
    below = sum(1 for v in xs if v < max_x)
    ties = sum(1 for v in xs if v == max_x)
    return (below + 0.5 * ties) / len(xs)


def reach_margin(max_x: float, state_key: str, table: dict) -> float | None:
    """``(max_x - script_median) / script_IQR``. None when the IQR is 0 (undefined, not zero)."""
    row = table.get(state_key)
    if not row or not row.get("iqr"):
        return None
    return (max_x - row["median"]) / row["iqr"]


def reach_credit(max_x: float, state_key: str, table: dict, deaths: int = 0,
                 death_penalty: float = 0.25) -> float | None:
    """Training signal for starts outside 1-1's obstacle table. None if the start is unknown."""
    q = reach_quantile(max_x, state_key, table)
    if q is None:
        return None
    return q - death_penalty * deaths


def report_line(label: str, v: dict) -> str:
    parts = []
    for ob, r in v["per_obstacle"].items():
        if "advantage_pp" not in r or r["advantage_pp"] is None:
            continue
        mark = "+" if r["beats_script"] else ("-" if r["loses_to_script"] else " ")
        parts.append(f"{ob} {r['advantage_pp']:+6.1f}{mark}")
    return f"{label:20s} vs best script: " + "  ".join(parts)
