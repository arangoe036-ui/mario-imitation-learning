"""§2(b): is stage 2's founding result a marginal artifact?

The result the whole stage-2 -> stage-3 -> composition lineage descends from is

    arm A (bernoulli only) 29.5%  ->  arm B (+onset reweight 10x) 59.5%  at pipe 1,  +30.0 pp

and `tasdata/bc/arms.py:48,49` shows the two arms differ **only** in `onset_weight` (1.0 vs 10.0).
`data/loss_bias_probe.json` shows onset reweighting inflates the A marginal. So the founding win may be a
marginal shift wearing the clothes of a learning result.

**The test.** Take arm A -- the plain-BCE arm -- and raise only its A *sampling rate* to arm B's measured
marginal, changing nothing else about the network. If pipe-1 clearance then matches arm B's, onset
reweighting contributed nothing beyond moving a button rate.

The rate is raised by adding a constant `delta` to the A logit, found by bisection so that the mean
predicted probability on held-out expert rows equals arm B's measured live rate. That is a pure marginal
intervention: it cannot add state-dependent behaviour, because it is the same constant at every frame.
The **realised** live rate is reported next to the target so any mismatch between the offline fit and live
play is visible rather than assumed.

Three arms at n=200, single life, identical seeds. The archived 29.5% / 59.5% figures are **not** reused:
they predate the single-life harness and LEDGER.md §2 forbids comparing across measurement bases, so both
arms are re-measured here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import session_when_free  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci, load_policy, random_rows  # noqa: E402
from tasdata.bc.pipe4_metrics import button_marginals, clearance  # noqa: E402
from tasdata.bc.script_baseline import vs_script  # noqa: E402
from tasdata.bc.tokens import LIVE_MASK  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace, write_traces  # noqa: E402
from tasdata.bc.train import make_loader  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARM_A = ROOT / "data/bc3/A_bernoulli_only_step3000_recal.pt"
ARM_B = ROOT / "data/bc3/B_bernoulli_onset10x_step3000_recal.pt"
OUT = ROOT / "data/stage2_marginal_test.json"
TRACEDIR = ROOT / "data/traces"
A_INDEX = NES_BUTTON_ORDER.index("A")
N, CAP, STALL = 200, 3000, 300
PROBE_ROWS = 6000


def episode(session, policy, cfg, start, seed: int, logit_bias=None) -> EpisodeTrace:
    """Per-button sampling, optionally with a constant added to one or more logits."""
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, 84, 84), np.uint8)
    win[:] = _resize_gray(obs.rgb, (84, 84))
    best = since = 0
    for _ in range(CAP):
        with torch.no_grad():
            lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        if logit_bias is not None:
            lg = lg + logit_bias
        bits = rng.random(8) < 1.0 / (1.0 + np.exp(-lg))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]:
                byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        obs = session.step(byte)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (84, 84))
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


def mean_p_a(policy, ds, rows, delta: float = 0.0) -> float:
    ps = []
    with torch.no_grad():
        for obs, _p, _b, _o in make_loader(Subset(ds, rows), batch_size=256, shuffle=False,
                                           num_workers=0):
            lg = policy(obs)[:, A_INDEX] + delta
            ps.append(torch.sigmoid(lg).numpy())
    return float(np.concatenate(ps).mean())


def fit_delta(policy, ds, rows, target: float) -> tuple[float, float]:
    """Bisect a constant logit offset so the mean predicted p(A) matches `target`."""
    lo, hi = -8.0, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if mean_p_a(policy, ds, rows, mid) < target:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return d, mean_p_a(policy, ds, rows, d)


def measure(session, policy, cfg, start, label: str, logit_bias=None) -> dict:
    """Measure one arm, resuming from its retained trace file if it is already complete.

    The environment restarts long jobs, and this script is three n=200 evaluations. Re-measuring an
    arm that already has 200 retained episodes wastes the emulator and, worse, makes a restart able to
    lose the whole run. Frames are the datum, so a completed arm is read back rather than re-run.
    """
    path = TRACEDIR / f"stage2_{label}_200.json"
    if path.exists():
        blob = json.loads(path.read_text())
        if blob.get("n_episodes") == N:
            eps = blob["episodes"]
            xs = [max(f[0] for f in e["frames"]) for e in eps]
            marg = button_marginals([f for e in eps for f in e["frames"]])
            cl = clearance(xs)
            print(f"  {label:24s} A {marg['rates']['A']:.3f}  "
                  f"pipe1 {cl['pipe1']['rate'] * 100:5.1f}%  "
                  f"pipe2 {cl['pipe2']['rate'] * 100:5.1f}%  "
                  f"x_med {float(np.median(xs)):.0f}   (resumed from {path.name})", flush=True)
            return {"n": N, "measurement_basis": "single_life",
                    "x_median": float(np.median(xs)), "clearance": cl,
                    "button_marginals": marg, "vs_script": vs_script(xs),
                    "ended": {k: sum(1 for e in eps if e["ended"] == k)
                              for k in ("died", "stuck")},
                    "resumed": True}
    traces = [episode(session, policy, cfg, start, i, logit_bias) for i in range(N)]
    write_traces(path, traces, arm=label)
    xs = [max(f[0] for f in t.frames) for t in traces]
    marg = button_marginals([f for t in traces for f in t.frames])
    cl = clearance(xs)
    row = {"n": N, "measurement_basis": "single_life", "x_median": float(np.median(xs)),
           "clearance": cl, "button_marginals": marg, "vs_script": vs_script(xs),
           "ended": {k: sum(1 for t in traces if t.ended == k) for k in ("died", "stuck")}}
    print(f"  {label:24s} A {marg['rates']['A']:.3f}  pipe1 {cl['pipe1']['rate'] * 100:5.1f}%  "
          f"pipe2 {cl['pipe2']['rate'] * 100:5.1f}%  x_med {row['x_median']:.0f}", flush=True)
    return row


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    ds = ctx.dataset(ctx.expert_train)
    rows = random_rows(ds, PROBE_ROWS, seed=1)
    pa, cfg_a, _ = load_policy(ARM_A)
    pb, cfg_b, _ = load_policy(ARM_B)
    print(f"arm A {ARM_A.name}  (plain BCE, onset_weight=1.0)")
    print(f"arm B {ARM_B.name}  (onset_weight=10.0)\n", flush=True)

    out = {"n": N, "measurement_basis": "single_life",
           "archived_figures_not_reused": {
               "arm_A_pipe1": 0.295, "arm_B_pipe1": 0.595, "delta_pp": 30.0,
               "why": "they predate the single-life harness; LEDGER.md §2 forbids comparing across "
                      "measurement bases, so both arms are re-measured here"},
           "arms": {}}

    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        print("re-measuring both arms, n=200 single life, identical seeds:", flush=True)
        out["arms"]["armA_plain_bce"] = measure(s, pa, cfg_a, start, "armA_plain_bce")
        out["arms"]["armB_onset10x"] = measure(s, pb, cfg_b, start, "armB_onset10x")

        target = out["arms"]["armB_onset10x"]["button_marginals"]["rates"]["A"]
        delta, fitted = fit_delta(pa, ds, rows, target)
        print(f"\nraising arm A's A-rate to arm B's {target:.3f}: offline logit delta "
              f"{delta:+.3f} (offline fit {fitted:.3f})", flush=True)

        # The offline fit is measured on rows the EXPERT visits; arm A visits different states (it
        # dies around x=435), so the realised live rate differs. A first attempt at delta=+1.032
        # realised 0.349 against a 0.219 target -- the intervention overshot and the comparison was
        # therefore not marginal-matched at all. Calibrate on the LIVE distribution instead, with
        # short probe runs, and only then spend the full n=200.
        probe_n = 40
        trace_log = []
        lo_d, hi_d = 0.0, delta
        for it in range(6):
            mid = (lo_d + hi_d) / 2
            bias = np.zeros(8, dtype=np.float64)
            bias[A_INDEX] = mid
            ts = [episode(s, pa, cfg_a, start, i, bias) for i in range(probe_n)]
            got = float(np.mean([(f[3] & NES_BUTTON_BITS["A"]) > 0
                                 for t in ts for f in t.frames]))
            trace_log.append({"iteration": it, "delta": mid, "realised_a_rate": got,
                              "probe_n": probe_n})
            print(f"    live probe {it}: delta {mid:+.3f} -> realised A {got:.3f} "
                  f"(target {target:.3f})", flush=True)
            if abs(got - target) < 0.008:
                break
            if got < target:
                lo_d = mid
            else:
                hi_d = mid
        delta = trace_log[-1]["delta"]
        bias = np.zeros(8, dtype=np.float64)
        bias[A_INDEX] = delta
        out["marginal_intervention"] = {
            "target_a_rate": target, "logit_delta": delta,
            "offline_fitted_p_a": fitted, "offline_delta_rejected": trace_log[0]["delta"],
            "live_calibration": trace_log,
            "why_live": ("the offline fit is taken on rows the expert visits; arm A visits its own "
                         "states, so an offline-matched delta overshot the live rate (0.349 against "
                         "a 0.219 target) and the comparison would not have been marginal-matched")}
        # The first attempt used an offline-fitted delta that overshot the live rate (0.349 against a
        # 0.219 target). Its trace is kept once, as evidence, and never overwritten again -- an
        # unconditional rename here defeated the resume and re-measured this arm on every run.
        stale = TRACEDIR / "stage2_armA_plain_bce_A_raised_200.json"
        overshot = TRACEDIR / "stage2_armA_plain_bce_A_raised_200.overshot.json"
        if stale.exists() and not overshot.exists():
            stale.rename(overshot)
        out["arms"]["armA_plain_bce_A_raised"] = measure(
            s, pa, cfg_a, start, "armA_plain_bce_A_raised", bias)
    finally:
        s.close()

    A_, B_, R_ = (out["arms"]["armA_plain_bce"], out["arms"]["armB_onset10x"],
                  out["arms"]["armA_plain_bce_A_raised"])
    out["marginal_intervention"]["realised_a_rate"] = R_["button_marginals"]["rates"]["A"]
    out["marginal_intervention"]["realised_vs_target_gap"] = (
        R_["button_marginals"]["rates"]["A"] - target)

    def cmp(x, y, name):
        lo, hi = diff_ci(x["clearance"][name]["k"], N, y["clearance"][name]["k"], N)
        return {"a": x["clearance"][name]["rate"], "b": y["clearance"][name]["rate"],
                "delta_pp": (y["clearance"][name]["rate"] - x["clearance"][name]["rate"]) * 100,
                "ci_pp": [lo * 100, hi * 100], "excludes_zero": bool(lo > 0 or hi < 0)}

    out["comparisons"] = {
        "onset_reweight_effect_A_to_B": {p: cmp(A_, B_, p) for p in ("pipe1", "pipe2")},
        "marginal_only_A_to_Araised": {p: cmp(A_, R_, p) for p in ("pipe1", "pipe2")},
        "residual_Araised_to_B": {p: cmp(R_, B_, p) for p in ("pipe1", "pipe2")},
    }
    res1 = out["comparisons"]["residual_Araised_to_B"]["pipe1"]
    ab1 = out["comparisons"]["onset_reweight_effect_A_to_B"]["pipe1"]
    marg1 = out["comparisons"]["marginal_only_A_to_Araised"]["pipe1"]
    # "the residual does not exclude zero" is NOT "the residual is zero". Report the decomposition
    # and the power needed to resolve what is left, rather than collapsing it to a binary claim.
    share = marg1["delta_pp"] / ab1["delta_pp"] if ab1["delta_pp"] else None
    out["decomposition"] = {
        "total_founding_effect_pp": ab1["delta_pp"],
        "explained_by_marginal_pp": marg1["delta_pp"],
        "share_explained_by_marginal": share,
        "residual_pp": res1["delta_pp"], "residual_ci_pp": res1["ci_pp"],
        "residual_resolved": res1["excludes_zero"],
        "power_note": ("LEDGER.md: ~600 episodes per arm are needed to detect 8 pp. The residual is "
                       f"{res1['delta_pp']:+.1f} pp at n={N}, so this test is underpowered to say "
                       "whether it is real; it is not evidence that the residual is zero."),
    }
    explained = (share is not None and share >= 0.5) and not res1["excludes_zero"]
    out["verdict"] = (
        f"MOST OF STAGE 2'S FOUNDING RESULT IS A MARGINAL SHIFT: a single constant added to one "
        f"logit -- which cannot add state-dependent behaviour -- reproduces "
        f"{marg1['delta_pp']:+.1f} pp of the {ab1['delta_pp']:+.1f} pp founding effect at pipe 1 "
        f"({(share or 0) * 100:.0f}%), taking arm A from "
        f"{A_['clearance']['pipe1']['rate'] * 100:.1f}% to "
        f"{R_['clearance']['pipe1']['rate'] * 100:.1f}% at a matched A-rate of "
        f"{R_['button_marginals']['rates']['A']:.3f} against arm B's {target:.3f}. The residual "
        f"against arm B's {B_['clearance']['pipe1']['rate'] * 100:.1f}% is {res1['delta_pp']:+.1f} pp "
        f"[{res1['ci_pp'][0]:+.1f}, {res1['ci_pp'][1]:+.1f}], which does NOT exclude zero and is "
        f"also NOT shown to be zero -- ~600 episodes per arm would be needed to resolve 8 pp."
        if explained else
        f"ONSET REWEIGHTING DID MORE THAN SHIFT THE MARGINAL: with arm A's A-rate raised to arm B's "
        f"{target:.3f}, pipe 1 reaches {R_['clearance']['pipe1']['rate'] * 100:.1f}% against arm B's "
        f"{B_['clearance']['pipe1']['rate'] * 100:.1f}%, a residual of {res1['delta_pp']:+.1f} pp "
        f"[{res1['ci_pp'][0]:+.1f}, {res1['ci_pp'][1]:+.1f}] that excludes zero. The founding "
        f"+{ab1['delta_pp']:.1f} pp is not fully explained by the marginal.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
