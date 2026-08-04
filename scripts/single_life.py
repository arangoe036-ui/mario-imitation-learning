"""P1 + P2: re-measure the headline chain result single-life, and close three cheap gaps.

The multi-life harness rewards dying: a policy that dies respawns and gets another attempt at
the same obstacle inside one episode, while a policy that gets *stuck* burns its remaining
frames and gets no retry. Two policies of equal skill score differently according to how they
happen to fail. Single life is canonical from here.

That matters most for the chain-position result (+45.7 pp, earliest vs latest TAS data), which
is the project's one surviving headline claim and was measured under the old harness. The
inflation is not uniform — it scales with how often an arm dies, and these two arms die at
visibly different rates.

Also here:

* **P2(a)** per-episode rows, so the death-x *distribution* can be reported instead of its
  median, and so "cleared pipe 2 then stuck at 720" can be separated from "cleared pipe 2 then
  died".
* **P2(b)** an empirical geometry probe: jump at each obstacle and record the landing y, which
  gives its height in tiles without a ROM read.
* **P2(c)** the trivial baseline the new rate table lacks: scripted Right+B held permanently.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from tasdata.bc.overnight_lib import calibrate, diff_ci, load_policy, wilson  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.tokens import LIVE_MASK  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/single_life.json"
SEEDS = 200
MAX_FRAMES = 2500
STALL = 300
GROUND_Y = 176
RIGHT, B, A = NES_BUTTON_BITS["Right"], NES_BUTTON_BITS["B"], NES_BUTTON_BITS["A"]
ENEMY_TYPE, ENEMY_XPAGE, ENEMY_XLO = 0x0016, 0x006E, 0x0087
ENEMY_NAMES = {0x00: "green_koopa", 0x06: "goomba", 0x03: "buzzy_beetle"}

CHAIN = {
    "earliest": [ROOT / f"data/bc_overnight/chain_earliest_seed{s}.pt" for s in (0, 1, 2)],
    "latest": [ROOT / f"data/bc_overnight/chain_latest_seed{s}.pt" for s in (0, 1, 2)],
}
ROUND3 = ROOT / "data/bc_overnight/round3_ratio1to1.pt"


def enemies(ram):
    out = []
    for i in range(5):
        t = int(ram[ENEMY_TYPE + i])
        if t == 0 and int(ram[ENEMY_XPAGE + i]) == 0:
            continue
        out.append((ENEMY_NAMES.get(t, f"type_{t:#04x}"),
                    int(ram[ENEMY_XPAGE + i]) * 256 + int(ram[ENEMY_XLO + i])))
    return out


def episode(session, policy, cfg, thr, start, seed):
    """One life. Ends at first death, stall, or frame budget."""
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), dtype=np.uint8)
    win[:] = _resize_gray(obs.rgb, (84, 84))
    maxx = best = read_smb(obs.ram, obs.framecount).x_position
    since = 0
    death = None
    ended = "budget"
    for _ in range(MAX_FRAMES):
        with torch.no_grad():
            logits = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        bits = rng.random(8) < 1.0 / (1.0 + np.exp(-logits))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]:
                byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        obs = session.step(byte)
        win = np.roll(win, -1, axis=0)
        win[-1] = _resize_gray(obs.rgb, (84, 84))
        st = read_smb(obs.ram, obs.framecount)
        maxx = max(maxx, st.x_position)
        if st.player_state in (0x06, 0x0B):
            near = sorted(enemies(obs.ram), key=lambda e: abs(e[1] - st.x_position))
            death = {"x": int(st.x_position), "y": int(st.y_position),
                     "cause": "pit" if st.y_position > 200 else (
                         f"enemy:{near[0][0]}" if near
                         and abs(near[0][1] - st.x_position) < 24 else "unknown")}
            ended = "died"
            break
        if st.x_position > best:
            best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL:
                ended = "stuck"
                break
    return {"seed": seed, "max_x": int(maxx), "ended": ended, "death": death}


def summarise(rows):
    xs = np.array([r["max_x"] for r in rows])
    n = len(rows)

    def rate(th):
        k = int((xs > th).sum())
        lo, hi = wilson(k, n)
        return {"k": k, "n": n, "rate": k / n, "ci": [lo, hi]}

    d = [r["death"] for r in rows if r["death"]]
    # P2(a): the joint -- among episodes clearing pipe 2, how did they end?
    past2 = [r for r in rows if r["max_x"] > 630]
    return {
        "n": n, "pipe1": rate(470), "pipe2": rate(630), "past720": rate(720),
        "x_median": float(np.median(xs)), "x_p90": float(np.percentile(xs, 90)),
        "x_max": int(xs.max()),
        "ended": dict(Counter(r["ended"] for r in rows)),
        "deaths": len(d), "death_causes": dict(Counter(x["cause"] for x in d)),
        "death_x_histogram_32px": dict(sorted(Counter(
            [int(x["x"]) // 32 * 32 for x in d]).items())),
        # Cause x bin, not pooled. A pooled cause count cannot say whether the deaths at an
        # obstacle's face are enemies or falls, which is the question that decides how the
        # obstacle is characterised.
        "cause_by_bin_32px": {
            str(b): dict(Counter(x["cause"] for x in d if int(x["x"]) // 32 * 32 == b))
            for b in sorted({int(x["x"]) // 32 * 32 for x in d})},
        "deaths_detail": [{"x": int(x["x"]), "y": int(x["y"]), "cause": x["cause"]}
                          for x in d],
        "death_x_median": float(np.median([x["x"] for x in d])) if d else None,
        "deaths_past_470": int(sum(1 for x in d if x["x"] > 470)),
        "deaths_past_630": int(sum(1 for x in d if x["x"] > 630)),
        "cleared_pipe2_outcomes": dict(Counter(r["ended"] for r in past2)),
        "n_cleared_pipe2": len(past2),
    }


def scripted_baseline(session, start, *, mode: str, frames=MAX_FRAMES):
    """P2(c): Right+B held permanently, single life. And a jump-probe variant."""
    obs = session.reset(start.frame)
    maxx = best = read_smb(obs.ram, obs.framecount).x_position
    since = 0
    ended = "budget"
    death = None
    landings = []
    prev_y = None
    for i in range(frames):
        byte = RIGHT | B
        if mode == "jump" and (i % 48) < 20:
            byte |= A
        obs = session.step(byte)
        st = read_smb(obs.ram, obs.framecount)
        maxx = max(maxx, st.x_position)
        if prev_y is not None and st.y_position == prev_y and st.y_position < GROUND_Y - 8:
            landings.append((int(st.x_position), int(st.y_position)))
        prev_y = st.y_position
        if st.player_state in (0x06, 0x0B):
            near = sorted(enemies(obs.ram), key=lambda e: abs(e[1] - st.x_position))
            death = {"x": int(st.x_position), "cause": (
                "pit" if st.y_position > 200 else
                (f"enemy:{near[0][0]}" if near
                 and abs(near[0][1] - st.x_position) < 24 else "unknown"))}
            ended = "died"
            break
        if st.x_position > best:
            best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL:
                ended = "stuck"
                break
    # elevated resting spots => obstacle tops; height in 16px tiles above ground
    tops = {}
    for x, y in landings:
        tops.setdefault(x // 16 * 16, set()).add(y)
    heights = sorted({(x, min(ys), round((GROUND_Y - min(ys)) / 16, 1))
                      for x, ys in tops.items()})
    return {"mode": mode, "max_x": int(maxx), "ended": ended, "death": death,
            "elevated_rests": heights[:20]}


def main() -> None:
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    out = {"seeds": SEEDS, "harness": "single life; episode ends at first death"}

    # ---------- P2(c) trivial baselines ----------
    print("P2(c) scripted baselines, single life")
    with FceuxSession(O.ROM, O.MOVIE, ctx.frames_needed()) as session:
        base = {m: scripted_baseline(session, start, mode=m) for m in ("hold", "jump")}
    for m, r in base.items():
        print(f"  Right+B {m:5s}: max_x {r['max_x']:4d} ended {r['ended']:6s} "
              f"death {r['death']}")
        print(f"     elevated rests (x, y, tiles above ground): {r['elevated_rests'][:8]}")
    out["scripted_baseline"] = base

    # ---------- P1 chain arms, single life ----------
    print("\nP1 chain position, single life")
    chain = {}
    for arm, paths in CHAIN.items():
        per_seed = []
        for i, p in enumerate(paths):
            policy, cfg, _ = load_policy(p)
            cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
            thr = cal.vector.astype(np.float64)
            with FceuxSession(O.ROM, O.MOVIE, ctx.frames_needed()) as session:
                rows = [episode(session, policy, cfg, thr, start, s) for s in range(SEEDS)]
            s = summarise(rows)
            per_seed.append(s)
            print(f"  {arm:9s} seed{i}: pipe1 {s['pipe1']['rate'] * 100:5.1f}%  "
                  f"pipe2 {s['pipe2']['rate'] * 100:5.1f}%  deaths {s['deaths']:3d}  "
                  f"x_med {s['x_median']:.0f}  x_max {s['x_max']}")
        k1 = sum(s["pipe1"]["k"] for s in per_seed)
        k2 = sum(s["pipe2"]["k"] for s in per_seed)
        n = sum(s["pipe1"]["n"] for s in per_seed)
        chain[arm] = {"per_seed": per_seed, "pooled_n": n,
                      "pipe1": {"k": k1, "rate": k1 / n, "ci": list(wilson(k1, n))},
                      "pipe2": {"k": k2, "rate": k2 / n, "ci": list(wilson(k2, n))},
                      "deaths": sum(s["deaths"] for s in per_seed)}
        print(f"  {arm:9s} POOLED n={n}: pipe1 {k1}/{n} = {k1 / n * 100:.1f}%  "
              f"pipe2 {k2}/{n} = {k2 / n * 100:.1f}%  deaths {chain[arm]['deaths']}")

    e, l = chain["earliest"], chain["latest"]
    diffs = {}
    for p in ("pipe1", "pipe2"):
        lo, hi = diff_ci(l[p]["k"], l["pooled_n"], e[p]["k"], e["pooled_n"])
        diffs[p] = {"delta": e[p]["rate"] - l[p]["rate"], "ci": [lo, hi],
                    "excludes_zero": bool(lo > 0 or hi < 0)}
        print(f"  earliest - latest {p}: {diffs[p]['delta'] * 100:+.1f} pp "
              f"[{lo * 100:+.1f}, {hi * 100:+.1f}]  "
              f"{'EXCLUDES zero' if diffs[p]['excludes_zero'] else 'includes zero'}")
    out["chain"] = chain
    out["chain_difference"] = diffs

    # ---------- P2(a) round3 rows ----------
    print("\nP2(a) round3_ratio1to1 death distribution and the pipe-2 joint")
    policy, cfg, _ = load_policy(ROUND3)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)
    with FceuxSession(O.ROM, O.MOVIE, ctx.frames_needed()) as session:
        rows = [episode(session, policy, cfg, thr, start, s) for s in range(SEEDS)]
    r3 = summarise(rows)
    out["round3_single_life"] = r3
    print(f"  pipe1 {r3['pipe1']['rate'] * 100:.1f}%  pipe2 {r3['pipe2']['rate'] * 100:.1f}%  "
          f"past720 {r3['past720']['rate'] * 100:.1f}%")
    print(f"  death x histogram (32px bins): {r3['death_x_histogram_32px']}")
    print(f"  deaths past 470: {r3['deaths_past_470']}  past 630: {r3['deaths_past_630']}")
    print(f"  of {r3['n_cleared_pipe2']} episodes clearing pipe 2, outcomes: "
          f"{r3['cleared_pipe2_outcomes']}")

    verdict = ("YES -- the earliest-vs-latest advantage survives single life: "
               f"{diffs['pipe1']['delta'] * 100:+.1f} pp on pipe 1 "
               f"[{diffs['pipe1']['ci'][0] * 100:+.1f}, {diffs['pipe1']['ci'][1] * 100:+.1f}]"
               if diffs["pipe1"]["excludes_zero"] else
               "NO -- the earliest-vs-latest pipe-1 advantage does not exclude zero single-life")
    out["binary"] = verdict
    print("\n" + "=" * 78)
    print(f"BINARY: {verdict}")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
