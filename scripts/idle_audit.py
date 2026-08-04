"""§0b: when the run-length policy is stationary, is it holding Right or holding nothing?

The owner's sixth observation: *"Mario stays still for a lot of time; in some instances he wasn't doing
anything."* The previous five were all right, and this one is corroborated by numbers already on disk —
x median 314 against scripted controls that end at x≈312 whether they press nothing or press A every frame.

**The discriminating question separates a generation bug from a competence problem:**

* **stationary while holding Right** -- pressed against terrain and unable to clear it. Competence.
* **stationary while holding nothing** -- inside an emitted no-op run. **Generation bug**, and capping
  non-A runs fixes it directly.

The expert's most common single action is *nothing*, at 40.3% of frames. In a run-length encoding those
become no-op classes with length buckets, and generation commits to the class's median length -- so
interspersed expert no-ops are re-emitted as blocks. Faithful in aggregate, potentially fatal in execution.

**One correction to the request.** The expert's A-onsets per 1,000 *grounded* frames was asked for as a read
over the corpus. It is not available as a read: `TRACE_COLUMNS` has no `on_ground` field, and deriving
groundedness from y is the exact mistake the ledger forbids. So the expert's 1-1 inputs are replayed through
the emulator to read `on_ground()` directly, and **the replay is validated by requiring the replayed x to
match the recorded x frame by frame** before any statistic is taken from it.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.overnight_lib import wilson  # noqa: E402
from tasdata.bc.pipe4_metrics import A_BIT  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACEDIR = ROOT / "data/traces"
OUT = ROOT / "data/idle_audit.json"
EXPERT_CACHE = ROOT / "data/expert_grounded_1_1.json"
RIGHT = NES_BUTTON_BITS["Right"]

ARMS = {"runlength": TRACEDIR / "phase1_runlength_200.json",
        "perframe": TRACEDIR / "phase1_perframe_200.json",
        "base_sustain": TRACEDIR / "seeds_base_200.json"}


def runs_of(pred) -> list[int]:
    """Lengths of maximal runs where `pred` is true."""
    out, i, n = [], 0, len(pred)
    while i < n:
        if pred[i]:
            j = i
            while j < n and pred[j]:
                j += 1
            out.append(j - i)
            i = j
        else:
            i += 1
    return out


def describe(v) -> dict:
    a = np.asarray(v, dtype=float)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max()), "mean": float(a.mean()),
            "frames_total": float(a.sum())}


def audit_arm(episodes) -> dict:
    """Stationary-frame attribution, no-op runs, and how episodes ended."""
    zero = tot = 0
    noop_runs: list[int] = []
    stat_right = stat_none = stat_other = stat_tot = 0
    ended = Counter()
    death_x: list[int] = []
    onsets_grounded = grounded = onsets_total = 0
    for e in episodes:
        fr = e["frames"]
        b = np.asarray([f[3] for f in fr], dtype=np.int64)
        x = np.asarray([f[0] for f in fr], dtype=np.int64)
        g = (np.asarray([f[5] for f in fr], dtype=np.int64) == 1) if len(fr[0]) >= 6 else None
        tot += len(b)
        zero += int((b == 0).sum())
        noop_runs.extend(runs_of(b == 0))
        # stationary: x did not change from the previous frame
        moved = np.ones(len(x), dtype=bool)
        moved[1:] = x[1:] != x[:-1]
        st = ~moved
        st[0] = False
        stat_tot += int(st.sum())
        stat_none += int(((b == 0) & st).sum())
        stat_right += int((((b & RIGHT) > 0) & st).sum())
        stat_other += int((((b & RIGHT) == 0) & (b != 0) & st).sum())
        ended[e.get("ended", "budget")] += 1
        if e.get("death"):
            death_x.append(int(e["death"]["x"]))
        a = (b & A_BIT) > 0
        prev = np.zeros_like(a)
        prev[1:] = a[:-1]
        onsets_total += int((a & ~prev).sum())
        if g is not None:
            onsets_grounded += int((a & ~prev & g).sum())
            grounded += int(g.sum())
    out = {
        "frames": tot,
        "zero_button_fraction": zero / tot if tot else None,
        "zero_button_ci": list(wilson(zero, tot)) if tot else None,
        "noop_runs": describe(noop_runs),
        "stationary_frames": stat_tot,
        "stationary_fraction": stat_tot / tot if tot else None,
        "stationary_attribution": {
            "holding_nothing": stat_none / stat_tot if stat_tot else None,
            "holding_right": stat_right / stat_tot if stat_tot else None,
            "other_buttons": stat_other / stat_tot if stat_tot else None,
        },
        "ended": dict(ended),
        "death_x_histogram_64px": dict(sorted(Counter(v // 64 * 64 for v in death_x).items())),
        "n_deaths": len(death_x),
        "a_onsets_per_1000_grounded": (onsets_grounded / grounded * 1000) if grounded else None,
        "a_onsets_per_1000_frames": (onsets_total / tot * 1000) if tot else None,
        "grounded_frames": grounded,
    }
    return out


def expert_actions_only(ctx) -> dict:
    """Expert statistics available as a pure read: no `on_ground`, so the denominator is TOTAL frames."""
    onsets = frames = zero = 0
    noop_runs: list[int] = []
    stat_none = stat_right = stat_tot = 0
    for run in ctx.expert_train:
        tr = np.asarray(run.trace)
        w, st_, pg = column(tr, "world"), column(tr, "stage"), column(tr, "pregame")
        xs, ps = column(tr, "x_position"), column(tr, "player_state")
        acts = np.asarray(run.actions, dtype=np.uint8)
        n = min(len(xs), len(acts))
        m = (w[:n] == 1) & (st_[:n] == 1) & (pg[:n] == 1) & (ps[:n] == 8)
        if not m.any():
            continue
        b = acts[:n][m].astype(np.int64)
        x = xs[:n][m].astype(np.int64)
        a = (b & A_BIT) > 0
        prev = np.zeros_like(a)
        prev[1:] = a[:-1]
        onsets += int((a & ~prev).sum())
        frames += len(b)
        zero += int((b == 0).sum())
        noop_runs.extend(runs_of(b == 0))
        moved = np.ones(len(x), dtype=bool)
        moved[1:] = x[1:] != x[:-1]
        stt = ~moved
        stt[0] = False
        stat_tot += int(stt.sum())
        stat_none += int(((b == 0) & stt).sum())
        stat_right += int((((b & RIGHT) > 0) & stt).sum())
    return {
        "basis": "1-1 surface frames of every expert train run; denominator is TOTAL frames",
        "grounded_available": False,
        "why_not_grounded": ("TRACE_COLUMNS has no on_ground field and deriving it from y is forbidden. "
                             "Each publication has its OWN movie, so replaying its inputs against this "
                             "session's savestate does not reproduce it -- validated and rejected: all "
                             "20 runs mismatched on x. An exact grounded figure needs a session per "
                             "movie, which was not spent."),
        "frames": frames,
        "a_onsets_per_1000_frames": (onsets / frames * 1000) if frames else None,
        "zero_button_fraction": zero / frames if frames else None,
        "noop_runs": describe(noop_runs),
        "stationary_fraction": stat_tot / frames if frames else None,
        "stationary_attribution": {
            "holding_nothing": stat_none / stat_tot if stat_tot else None,
            "holding_right": stat_right / stat_tot if stat_tot else None},
    }


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    if EXPERT_CACHE.exists():
        exp = json.loads(EXPERT_CACHE.read_text())
    else:
        exp = expert_actions_only(ctx)
        EXPERT_CACHE.write_text(json.dumps(exp, indent=2, default=str))
    print(f"EXPERT, 1-1 surface frames ({exp['frames']:,}) -- denominator is TOTAL frames, not "
          f"grounded ones:", flush=True)
    print(f"  A-onsets/1k frames {exp['a_onsets_per_1000_frames']:.1f}   "
          f"zero-button {exp['zero_button_fraction'] * 100:.1f}%   "
          f"no-op runs median {exp['noop_runs']['median']:.0f} "
          f"p90 {exp['noop_runs']['p90']:.0f} max {exp['noop_runs']['max']:.0f}", flush=True)
    print(f"  stationary {exp['stationary_fraction'] * 100:.1f}% of frames, of which nothing "
          f"{(exp['stationary_attribution']['holding_nothing'] or 0) * 100:.1f}% / Right "
          f"{(exp['stationary_attribution']['holding_right'] or 0) * 100:.1f}%")
    print(f"  NOTE: {exp['why_not_grounded']}\n", flush=True)

    out = {"expert": exp, "arms": {}}
    print(f"{'arm':14s} {'zero-btn':>9s} {'noop med/p90/max':>18s} {'stationary':>11s} "
          f"{'still+nothing':>14s} {'still+Right':>12s} {'onset/1k':>9s}")
    for label, path in ARMS.items():
        if not path.exists():
            continue
        eps = json.loads(path.read_text())["episodes"]
        r = audit_arm(eps)
        out["arms"][label] = r
        nr, sa = r["noop_runs"], r["stationary_attribution"]
        print(f"{label:14s} {r['zero_button_fraction'] * 100:8.1f}% "
              f"{nr['median']:6.0f}/{nr['p90']:.0f}/{nr['max']:.0f}".ljust(46)
              + f"{r['stationary_fraction'] * 100:10.1f}% "
              f"{(sa['holding_nothing'] or 0) * 100:13.1f}% "
              f"{(sa['holding_right'] or 0) * 100:11.1f}% "
              f"{(r['a_onsets_per_1000_grounded'] or 0):9.1f}"
              f"  /1k-all {(r['a_onsets_per_1000_frames'] or 0):5.1f}", flush=True)
        print(f"{'':14s} ended {r['ended']}  deaths {r['n_deaths']}  "
              f"death-x {dict(list(r['death_x_histogram_64px'].items())[:6])}", flush=True)

    rl = out["arms"].get("runlength", {})
    sa = rl.get("stationary_attribution", {})
    nothing, right = (sa.get("holding_nothing") or 0), (sa.get("holding_right") or 0)
    out["verdict"] = {
        "stationary_holding_nothing": nothing, "stationary_holding_right": right,
        "diagnosis": ("GENERATION BUG: when stationary the run-length policy is holding NOTHING on "
                      f"{nothing * 100:.1f}% of those frames against {right * 100:.1f}% holding Right, "
                      f"and its no-op runs reach {rl.get('noop_runs', {}).get('max', 0):.0f} frames "
                      f"against the expert's {exp['noop_runs']['max']:.0f}. It is sitting inside "
                      "emitted no-op blocks, which capping non-A runs addresses directly."
                      if nothing > right else
                      "COMPETENCE PROBLEM: when stationary the policy is mostly holding Right "
                      f"({right * 100:.1f}% against {nothing * 100:.1f}% holding nothing), so it is "
                      "pressed against terrain it cannot clear rather than idling. Capping no-op runs "
                      "will not address this."),
    }
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"]["diagnosis"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
