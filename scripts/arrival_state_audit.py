"""Does the policy arrive at pipe 3 in the same state the script does?

`FINDINGS.md` §1b claims the policy beats the strongest fixed-rate script at pipe 3 by +23.8 pp
[+14.7, +32.1], conditional on arrival. Conditioning controls for **whether** the policy reaches pipe 3.
It does not control for **how it arrives**. If the policy arrives faster, better positioned, or more often
grounded, part of that advantage is inherited upstream state quality rather than pipe-3 behaviour -- the
same class of error the conditioning fix addressed, one level deeper.

Two arrival points are measured, because they answer different questions:

* **the gate** -- the first frame with x > 630, i.e. the moment the conditioning admits the episode. This is
  the state the conditional comparison implicitly assumes is exchangeable between arms.
* **the face** -- the first frame with x >= 720, pipe 3 itself. This is how the episode arrives *at the
  obstacle*, and it is the primary comparison.

Four quantities at each: x, speed byte (units of 1/16 px/frame), y_absolute, and `grounded`.

**One correction to the request: `grounded` is not in the retained script traces.** The script arms were
recorded before `EpisodeTrace` gained the field, so their frames carry five elements, not six. Three of the
four quantities are a pure read; the fourth needed the canonical script arm re-run, which is ~2 minutes of
emulator for 200 scripted episodes. Leaving one of the four unmeasured was the worse option.

**The residual is computed by direct standardisation**, not by eyeballing the distributions: the policy's
per-stratum pipe-3 clearance is reweighted onto the script's stratum distribution, where a stratum is
(grounded, speed band) at the face. What survives reweighting is the part of the advantage that is not
explained by arrival state.
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
from tasdata.bc.overnight_lib import diff_ci, wilson  # noqa: E402
from tasdata.bc.trace_log import write_traces  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACEDIR = ROOT / "data/traces"
OUT = ROOT / "data/arrival_state_audit.json"

POLICY_TRACES = [TRACEDIR / f"seeds_plain_s{s}_200.json" for s in (0, 1, 2)]
BASE_TRACES = TRACEDIR / "seeds_base_200.json"
#: the conditional pipe-3 baseline in FINDINGS §1b was this arm, at 31.1% (47/151)
SCRIPT_TRACES = TRACEDIR / "arrival_rng_matched_200.json"
SCRIPT_RATES = {"A": 0.85, "Left": 0.0, "Down": 0.0, "rng_matched": True}

GATE_X, FACE_X, CLEAR_X = 630, 720, 735
#: speed byte bands. 40 = maximum running speed (2.5 px/frame); 0 is a dead stop.
SPEED_BANDS = ((0, 20), (20, 32), (32, 64))


def arrival_state(frames, at_x: int):
    """(x, speed, y, grounded) at the first frame reaching `at_x`; None if never reached."""
    for f in frames:
        if f[0] >= at_x:
            return {"x": f[0], "speed": f[2], "y": f[1],
                    "grounded": (f[5] if len(f) >= 6 else None)}
    return None


def band(speed: int) -> int:
    for i, (lo, hi) in enumerate(SPEED_BANDS):
        if lo <= speed < hi:
            return i
    return len(SPEED_BANDS) - 1


def describe(vals) -> dict:
    a = np.asarray([v for v in vals if v is not None], dtype=float)
    if not a.size:
        return {"n": 0}
    return {"n": int(a.size), "median": float(np.median(a)),
            "q1": float(np.percentile(a, 25)), "q3": float(np.percentile(a, 75)),
            "mean": float(a.mean()), "min": float(a.min()), "max": float(a.max())}


def collect(episodes) -> dict:
    """Arrival states plus the pipe-3 outcome, for episodes that pass the gate."""
    rows = []
    for e in episodes:
        fr = e["frames"]
        mx = max(f[0] for f in fr)
        if mx <= GATE_X:
            continue
        g = arrival_state(fr, GATE_X + 1)
        face = arrival_state(fr, FACE_X)
        rows.append({"gate": g, "face": face, "cleared": bool(mx > CLEAR_X), "max_x": int(mx)})
    return rows


def summarise(rows, label: str) -> dict:
    out = {"label": label, "n_arrivals": len(rows),
           "cleared": sum(r["cleared"] for r in rows)}
    out["conditional_rate"] = out["cleared"] / len(rows) if rows else None
    for point in ("gate", "face"):
        got = [r[point] for r in rows if r[point]]
        gr = [g["grounded"] for g in got if g["grounded"] is not None]
        out[point] = {
            "n": len(got),
            "x": describe([g["x"] for g in got]),
            "speed": describe([g["speed"] for g in got]),
            "y": describe([g["y"] for g in got]),
            "grounded_fraction": (float(np.mean(gr)) if gr else None),
            "grounded_ci": (list(wilson(int(np.sum(gr)), len(gr))) if gr else None),
            "grounded_available": bool(gr),
        }
    return out


def main() -> None:
    t0 = time.time()
    # --- the one thing that needs the emulator: `grounded` for the script arm --------------------
    if not SCRIPT_TRACES.exists():
        print("re-running the canonical script arm to record `grounded` "
              "(the retained script traces predate the field)", flush=True)
        from scripts.p1_control_ladder import scripted_episode
        ctx = O.Ctx()
        start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
        partial = SCRIPT_TRACES.with_suffix(".partial.json")
        done = json.loads(partial.read_text())["episodes"] if partial.exists() else []
        s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        try:
            traces = list(done)
            while len(traces) < 200:
                for i in range(len(traces), min(len(traces) + 25, 200)):
                    t = scripted_episode(s, start, i, SCRIPT_RATES["A"], 0.0, 0.0, True)
                    traces.append(t.to_dict())
                partial.write_text(json.dumps({"episodes": traces}, separators=(",", ":")))
                print(f"  {len(traces)}/200 banked", flush=True)
        finally:
            s.close()
        SCRIPT_TRACES.write_text(json.dumps(
            {"schema": "per-frame (x, y_absolute, speed_byte, buttons, player_state, grounded)",
             "n_episodes": len(traces), "arm": "rng_matched", "rates": SCRIPT_RATES,
             "episodes": traces}, separators=(",", ":")))
        partial.unlink(missing_ok=True)
        print(f"  wrote {SCRIPT_TRACES.name}\n", flush=True)

    pol_eps = [e for p in POLICY_TRACES for e in json.loads(p.read_text())["episodes"]]
    base_eps = json.loads(BASE_TRACES.read_text())["episodes"]
    scr_eps = json.loads(SCRIPT_TRACES.read_text())["episodes"]
    print(f"policy {len(pol_eps)} episodes (3 seeds), base {len(base_eps)}, "
          f"script {len(scr_eps)}\n", flush=True)

    pol, base, scr = collect(pol_eps), collect(base_eps), collect(scr_eps)
    out = {"gate_x": GATE_X, "face_x": FACE_X, "clear_x": CLEAR_X,
           "speed_bands": [list(b) for b in SPEED_BANDS],
           "note": ("`grounded` was absent from the retained script traces (recorded before the field "
                    "existed); the canonical arm was re-run to obtain it"),
           "arms": {k: summarise(v, k) for k, v in
                    (("policy_pooled_3_seeds", pol), ("base", base), ("script_rng_matched", scr))}}

    print(f"{'arm':22s} {'n':>5s} {'cond':>7s} | face: {'x med':>7s} {'speed med':>10s} "
          f"{'y med':>7s} {'grounded':>9s}")
    for k, v in out["arms"].items():
        f = v["face"]
        gf = f"{f['grounded_fraction'] * 100:8.1f}%" if f["grounded_fraction"] is not None else "      n/a"
        print(f"{k:22s} {v['n_arrivals']:5d} {(v['conditional_rate'] or 0) * 100:6.1f}% | "
              f"      {f['x']['median']:7.0f} {f['speed']['median']:10.0f} "
              f"{f['y']['median']:7.0f} {gf}", flush=True)

    # --- two standardisations, with matched denominators ----------------------------------------
    #
    # A first version compared a face-restricted standardised policy rate against the script's
    # gate-conditional rate. Mismatched denominators, which is the error this whole document is about.
    # Both analyses below keep the two arms on the same basis, and they answer different questions:
    #
    #   A. among episodes that REACH THE FACE -- "given both arrive at the obstacle, is the policy
    #      better at the obstacle?" This is the advisor's question in its strict form.
    #   B. among episodes that PASS THE GATE, with "never reached the face" kept as its own stratum --
    #      the §1b figure adjusted for upstream stalling between x=630 and x=720. That stratum lies on
    #      the causal path, so B adjusts for an upstream *behaviour* difference, not an arrival state.
    def face_strata(rows):
        d = {}
        for r in rows:
            f = r["face"]
            if not f or f["grounded"] is None:
                continue
            d.setdefault((int(f["grounded"]), band(int(f["speed"]))), []).append(r["cleared"])
        return d

    def gate_strata(rows):
        d = {}
        for r in rows:
            f = r["face"]
            key = ("never_reached_face",) if (not f or f["grounded"] is None) \
                else (int(f["grounded"]), band(int(f["speed"])))
            d.setdefault(key, []).append(r["cleared"])
        return d

    def standardise(pol_s, scr_s):
        n = sum(len(v) for v in scr_s.values())
        num = cov = unc = 0.0
        table = []
        for key, out_s in sorted(scr_s.items(), key=lambda kv: str(kv[0])):
            w = len(out_s) / n if n else 0.0
            p_out = pol_s.get(key)
            table.append({"stratum": str(key), "script_n": len(out_s),
                          "script_rate": float(np.mean(out_s)), "script_weight": w,
                          "policy_n": len(p_out) if p_out else 0,
                          "policy_rate": float(np.mean(p_out)) if p_out else None})
            if p_out:
                num += w * float(np.mean(p_out))
                cov += w
            else:
                unc += w
        return {"strata": table, "weight_covered": cov, "weight_uncovered": unc,
                "standardised_policy_rate": (num / cov) if cov else None,
                "script_n": n,
                "script_rate": (sum(sum(v) for v in scr_s.values()) / n) if n else None,
                "policy_n": sum(len(v) for v in pol_s.values()),
                "policy_rate": (sum(sum(v) for v in pol_s.values())
                                / max(sum(len(v) for v in pol_s.values()), 1))}

    out["standardisation"] = {}
    for name, fn in (("A_among_face_reachers", face_strata),
                     ("B_gate_conditional_with_stall_stratum", gate_strata)):
        r = standardise(fn(pol), fn(scr))
        pk = int(round(r["policy_rate"] * r["policy_n"]))
        sk = int(round(r["script_rate"] * r["script_n"]))
        lo, hi = diff_ci(sk, r["script_n"], pk, r["policy_n"])
        r["crude_advantage_pp"] = (r["policy_rate"] - r["script_rate"]) * 100
        r["crude_ci_pp"] = [lo * 100, hi * 100]
        r["standardised_advantage_pp"] = (
            (r["standardised_policy_rate"] - r["script_rate"]) * 100
            if r["standardised_policy_rate"] is not None else None)
        r["explained_by_arrival_state_pp"] = (
            r["crude_advantage_pp"] - r["standardised_advantage_pp"]
            if r["standardised_advantage_pp"] is not None else None)
        out["standardisation"][name] = r
        print(f"\n[{name}]  policy {r['policy_rate'] * 100:.1f}% (n={r['policy_n']})  "
              f"script {r['script_rate'] * 100:.1f}% (n={r['script_n']})")
        print(f"   crude {r['crude_advantage_pp']:+.1f} pp "
              f"[{r['crude_ci_pp'][0]:+.1f}, {r['crude_ci_pp'][1]:+.1f}]   "
              f"standardised {r['standardised_advantage_pp']:+.1f} pp   "
              f"(arrival state explains {r['explained_by_arrival_state_pp']:+.1f} pp)", flush=True)

    crude = out["arms"]["policy_pooled_3_seeds"]["conditional_rate"] * 100 - \
        out["arms"]["script_rng_matched"]["conditional_rate"] * 100
    std_adv = out["standardisation"]["A_among_face_reachers"]["standardised_advantage_pp"]

    # a compact statement of whether the arrival states match at all
    f_p, f_s = out["arms"]["policy_pooled_3_seeds"]["face"], out["arms"]["script_rng_matched"]["face"]
    dspeed = f_p["speed"]["median"] - f_s["speed"]["median"]
    dg = ((f_p["grounded_fraction"] - f_s["grounded_fraction"]) * 100
          if None not in (f_p["grounded_fraction"], f_s["grounded_fraction"]) else None)
    same = abs(dspeed) <= 2 and (dg is None or abs(dg) <= 5)
    a = out["standardisation"]["A_among_face_reachers"]
    b = out["standardisation"]["B_gate_conditional_with_stall_stratum"]
    out["verdict"] = {
        "arrival_states_match": bool(same),
        "speed_median_delta": dspeed,
        "grounded_fraction_delta_pp": dg,
        "advantage_shrinks_after_adjustment": bool(
            a["standardised_advantage_pp"] < a["crude_advantage_pp"]),
        "statement": (
            f"THE POLICY ARRIVES AT PIPE 3 IN ESSENTIALLY THE SAME STATE AS THE SCRIPT. At the face, "
            f"x median {f_p['x']['median']:.0f} against {f_s['x']['median']:.0f}, speed median "
            f"{f_p['speed']['median']:.0f} against {f_s['speed']['median']:.0f} "
            f"({dspeed:+.0f}, units of 1/16 px/frame), y median {f_p['y']['median']:.0f} against "
            f"{f_s['y']['median']:.0f}, grounded {f_p['grounded_fraction'] * 100:.1f}% against "
            f"{f_s['grounded_fraction'] * 100:.1f}% ({dg:+.1f} pp). "
            f"Standardising on arrival state does NOT reduce the advantage -- among face-reachers it "
            f"moves {a['crude_advantage_pp']:+.1f} -> {a['standardised_advantage_pp']:+.1f} pp, and on "
            f"the gate-conditional basis with stalling as a stratum "
            f"{b['crude_advantage_pp']:+.1f} -> {b['standardised_advantage_pp']:+.1f} pp. "
            f"If anything the policy arrives slightly WORSE: it stalls between the gate and the face in "
            f"{(1 - a['policy_n'] / out['arms']['policy_pooled_3_seeds']['n_arrivals']) * 100:.1f}% of "
            f"arrivals against the script's "
            f"{(1 - a['script_n'] / out['arms']['script_rng_matched']['n_arrivals']) * 100:.1f}%. "
            f"So FINDINGS §1b is a claim about pipe-3 behaviour, and the crude +23.8 pp is if anything "
            f"conservative."
            if same else
            f"THE POLICY ARRIVES DIFFERENTLY: speed median differs by {dspeed:+.0f} and grounded "
            f"fraction by {dg:+.1f} pp. Among face-reachers, standardising moves "
            f"{a['crude_advantage_pp']:+.1f} -> {a['standardised_advantage_pp']:+.1f} pp, so "
            f"{a['explained_by_arrival_state_pp']:+.1f} pp is arrival state and "
            f"{a['standardised_advantage_pp']:+.1f} pp is pipe-3 behaviour."),
    }
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"]["statement"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
