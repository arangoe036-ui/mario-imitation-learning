"""Does `capped` beat a fixed-rate script running at *its own* button rates?

`capped` closed the script gap by 31-36 pp at every obstacle, but its A marginal is **0.572 — 3.8x the
expert's 0.152** — and capping non-A runs necessarily hands A-runs more wall-clock. So the reach may be
another marginal shift. This project spent two days establishing that reach bought that way is not skill,
and the existing script curve cannot settle it: it jumps from p(A)=0.50 (pipe 2: 10.0%) to p(A)=0.85 (68.5%),
and 0.572 sits in that gap while `capped` is at 61.0%. Interpolation suggests `capped` wins, which is exactly
why it is measured.

**Two controls, because "rate-matched" has two defensible readings and they bracket the answer.**

* **`rate_matched`** -- every one of `capped`'s five marginals reproduced independently per frame
  (A .572, B .744, Right .616, Down .013, Left .065). This is the literal request. It isolates *temporal
  structure* -- which buttons go together and when -- from the marginals, because the marginals are equal by
  construction.
* **`rate_matched_strong`** -- Right and B held on **every** frame, A at .572, Left/Down at `capped`'s rates.
  Every earlier script in this project held Right+B permanently, and holding them is strictly better for
  travelling right, so this is the **stronger** opponent at the same A-rate. `capped`'s Right of 0.616 is a
  consequence of its own run structure, not a constraint the opponent has to accept.

Both are reported. The strong one is the bar; the literal one answers the question as asked.

Paired: seeds 0-199, the same seeds `capped` ran, same thresholds, single life. Medians are reported with
**max and p99** beside them, per the rule earned by a no-op run distribution whose median was correct at 10
while its maximum was 347.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import session_when_free  # noqa: E402
from scripts.phase1_duration import PIPE2_WINDOW, _Ep  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402
from tasdata.bc.pipe4_metrics import (  # noqa: E402
    PIPE_THRESHOLDS,
    a_hold_onsets,
    button_marginals,
    clearance,
    hold_stats,
)
from tasdata.bc.script_baseline import behaviour_stats, conditional_rates  # noqa: E402
from tasdata.bc.tokens import LIVE_MASK  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACEDIR = ROOT / "data/traces"
OUT = ROOT / "data/rate_matched_control.json"
CAPPED = TRACEDIR / "variant_capped_200.json"

#: `capped`'s measured marginals, from data/phase1_variants.json.
CAPPED_RATES = {"A": 0.572, "B": 0.744, "Right": 0.616, "Down": 0.013, "Left": 0.065}

ARMS = {
    "rate_matched": {**CAPPED_RATES},
    "rate_matched_strong": {**CAPPED_RATES, "B": 1.0, "Right": 1.0},
}
N_EVAL, CAP_FRAMES, STALL, CHUNK = 200, 3000, 300, 20


def scripted_episode(session, start, seed: int, rates: dict) -> EpisodeTrace:
    """Each button drawn independently per frame at its own fixed rate."""
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    best = since = 0
    names = sorted(rates)                      # fixed order so a seed means one thing
    for _ in range(CAP_FRAMES):
        byte = 0
        for nm in names:
            p = rates[nm]
            if p >= 1.0 or (p > 0.0 and rng.random() < p):
                byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        obs = session.step(byte)
        t.record(obs, byte)
        st = read_smb(obs.ram, obs.framecount)
        if st.player_state in (0x06, 0x0B):
            t.record_death(obs)
            break
        if st.x_position > best:
            best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL:
                t.ended = "stuck"
                break
    return t


def resumable(path: Path, n: int, make):
    if path.exists() and json.loads(path.read_text()).get("n_episodes") == n:
        return [_Ep(e) for e in json.loads(path.read_text())["episodes"]]
    partial = path.with_suffix(".partial.json")
    traces = [_Ep(e) for e in (json.loads(partial.read_text())["episodes"]
                               if partial.exists() else [])]
    while len(traces) < n:
        for i in range(len(traces), min(len(traces) + CHUNK, n)):
            traces.append(make(i))
        partial.write_text(json.dumps({"episodes": [t.to_dict() for t in traces]},
                                      separators=(",", ":")))
        print(f"    {path.stem}: {len(traces)}/{n}", flush=True)
    path.write_text(json.dumps(
        {"schema": "per-frame (x, y_absolute, speed_byte, buttons, player_state, grounded)",
         "n_episodes": len(traces), "episodes": [t.to_dict() for t in traces]},
        separators=(",", ":")))
    partial.unlink(missing_ok=True)
    return traces


def noop_runs(frames) -> dict:
    b = np.asarray([f[3] for f in frames], dtype=np.int64)
    out, i = [], 0
    while i < len(b):
        if b[i] == 0:
            j = i
            while j < len(b) and b[j] == 0:
                j += 1
            out.append(j - i)
            i = j
        else:
            i += 1
    if not out:
        return {"n": 0}
    a = np.asarray(out, dtype=float)
    return {"n": len(out), "median": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def with_tail(h: dict, vals) -> dict:
    """Add p99 to a hold_stats block. Medians are not reported alone any more."""
    a = np.asarray(vals, dtype=float)
    return {**h, "p99": (float(np.percentile(a, 99)) if a.size else None)}


def score(label: str, traces) -> dict:
    xs = [max(f[0] for f in t.frames) for t in traces]
    frames = [f for t in traces for f in t.frames]
    h2 = [h for t in traces for h in a_hold_onsets(t.frames, PIPE2_WINDOW)]
    row = {"label": label, "n": len(traces), "measurement_basis": "single_life",
           "x_median": float(np.median(xs)), "x_p99": float(np.percentile(xs, 99)),
           "x_max": int(max(xs)),
           "a_hold_pipe2": with_tail(hold_stats(h2), h2),
           "noop_runs": noop_runs(frames),
           "clearance": clearance(xs),
           "conditional": conditional_rates(xs),
           "button_marginals": button_marginals(frames),
           "behaviour": behaviour_stats(frames),
           "ended": {k: sum(1 for t in traces if t.ended == k) for k in ("died", "stuck")}}
    m = row["button_marginals"]["rates"]
    print(f"  {label:20s} A {m['A']:.3f} B {m['B']:.3f} R {m['Right']:.3f} | "
          f"p1 {row['clearance']['pipe1']['rate'] * 100:5.1f} "
          f"p2 {row['clearance']['pipe2']['rate'] * 100:5.1f} "
          f"p3 {row['clearance']['pipe3']['rate'] * 100:5.1f} "
          f"p4 {row['clearance']['pipe4']['rate'] * 100:5.1f} | "
          f"x med {row['x_median']:4.0f} max {row['x_max']:5d}", flush=True)
    return row


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")

    capped_eps = json.loads(CAPPED.read_text())["episodes"]
    capped = score("capped (policy)", [_Ep(e) for e in capped_eps])
    out = {"target_rates": CAPPED_RATES, "arms": {"capped": capped},
           "note": ("two readings of 'rate-matched': all five marginals reproduced, and the stronger "
                    "opponent that holds Right+B permanently at the same A-rate"),
           "seeds": "0-199, paired with capped"}

    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        for label, rates in ARMS.items():
            print(f"[{label}] {rates}", flush=True)
            tr = resumable(TRACEDIR / f"{label}_200.json", N_EVAL,
                           lambda i, r=rates: scripted_episode(s, start, i, r))
            out["arms"][label] = score(label, tr)
    finally:
        s.close()

    # capped versus each control, unconditional and conditional on arrival
    out["comparisons"] = {}
    for label in ARMS:
        ctl = out["arms"][label]
        per_ob, cond = {}, {}
        for ob in PIPE_THRESHOLDS:
            a, b = ctl["clearance"][ob], capped["clearance"][ob]
            lo, hi = diff_ci(a["k"], a["n"], b["k"], b["n"])
            per_ob[ob] = {"script": a["rate"], "capped": b["rate"],
                          "advantage_pp": (b["rate"] - a["rate"]) * 100,
                          "ci_pp": [lo * 100, hi * 100],
                          "capped_beats": bool(lo > 0), "capped_loses": bool(hi < 0)}
            ca, cb = ctl["conditional"][ob], capped["conditional"][ob]
            if ca["n_arrived"] and cb["n_arrived"]:
                clo, chi = diff_ci(ca["k"], ca["n_arrived"], cb["k"], cb["n_arrived"])
                cond[ob] = {"script": ca["rate"], "script_n": ca["n_arrived"],
                            "capped": cb["rate"], "capped_n": cb["n_arrived"],
                            "advantage_pp": (cb["rate"] - ca["rate"]) * 100,
                            "ci_pp": [clo * 100, chi * 100],
                            "capped_beats": bool(clo > 0), "capped_loses": bool(chi < 0)}
        out["comparisons"][label] = {"unconditional": per_ob, "conditional_on_arrival": cond}
        print(f"\ncapped vs {label} (pp, Newcombe):")
        for ob in PIPE_THRESHOLDS:
            r = per_ob[ob]
            c = cond.get(ob)
            mark = "BEATS" if r["capped_beats"] else ("loses" if r["capped_loses"] else "n.s.")
            cs = (f"  cond {c['advantage_pp']:+6.1f} [{c['ci_pp'][0]:+6.1f},{c['ci_pp'][1]:+6.1f}]"
                  f" {'BEATS' if c['capped_beats'] else ('loses' if c['capped_loses'] else 'n.s.')}"
                  if c else "  cond n/a")
            print(f"  {ob:6s} uncond {r['advantage_pp']:+6.1f} "
                  f"[{r['ci_pp'][0]:+6.1f},{r['ci_pp'][1]:+6.1f}] {mark:5s}{cs}", flush=True)

    strong = out["comparisons"]["rate_matched_strong"]
    lit = out["comparisons"]["rate_matched"]
    beats_strong = [o for o, r in strong["unconditional"].items() if r["capped_beats"]]
    beats_lit = [o for o, r in lit["unconditional"].items() if r["capped_beats"]]
    out["verdict"] = {
        "beats_rate_matched_at": beats_lit,
        "beats_rate_matched_strong_at": beats_strong,
        "statement": (
            f"SKILL SIGNAL: at its own button rates `capped` beats the rate-matched script at "
            f"{beats_lit} and the stronger Right+B-held script at {beats_strong or 'nothing'}. "
            f"The reach is not reproducible by a marginal running at the same rate."
            if beats_strong else
            (f"PARTIAL: `capped` beats the literally rate-matched script at {beats_lit}, but not the "
             f"stronger script that holds Right+B at the same A-rate. Its advantage over the matched "
             f"marginal comes at least partly from holding Right and B more effectively rather than "
             f"from obstacle skill."
             if beats_lit else
             "NO SKILL SIGNAL: `capped` does not beat a fixed-rate script at its own marginals. The cap "
             "is a marginal shift wearing a new representation. The >=12 hold capability is still real "
             "and still needed, but the reach is not evidence of skill.")),
    }
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"]["statement"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
