"""§4 evaluation of arms B / R / RT. **CPU only, and this process must never touch MPS.**

Training ran on MPS in `scripts/scaleup_train.py`; this file is the other half of that split. It spawns
FCEUX, so a single `torch.backends.mps.is_available()` anywhere in its import graph would poison every
rollout irreversibly. `pick_device` is not called at all here -- tensors stay on the CPU default.

Per arm, everything the fifty-third directive asks for:

* **corrected timing lift at pipe 1 and pipe 2** -- onsets against **non-A run starts**, stratified per
  16-px bin, bootstrapped over onsets. The confounded all-non-onset-frames form is reported beside it, since
  that is what the fifty-first block measured and the comparison is the point.
* **`button_marginals`** against the expert's, **clearance** at every obstacle, **`vs_script` on both bars**,
  **A-hold distribution with max and p99**, **airborne fraction** against the expert's 61.1%.

The **`capped`** generation rule throughout (non-A runs capped at 4 frames), because that is the adopted rule
and changing it here would confound the scale-up with the generation rule.

**The rate-matched bar needs its own rollouts** (a script at each arm's own five marginals), so it runs as a
second stage: the primary numbers land first and survive an interruption.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import session_when_free  # noqa: E402
from scripts.onset_lift_boundary_matched import boot_lift, stratified_lift  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402
from tasdata.bc.pipe4_metrics import (  # noqa: E402
    A_BIT,
    a_hold_onsets,
    button_marginals,
    clearance,
    hold_stats,
)
from tasdata.bc.runlength import N_BUCKETS, class_lengths, joint_size  # noqa: E402
from tasdata.bc.script_baseline import behaviour_stats, vs_script  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import column, read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "data/bc_scaleup"
TRACEDIR = ROOT / "data/traces"
OUT = OUTDIR / "eval_summary.json"
RUNS128 = ROOT / "data/runs128"

ARMS = ["B_84_d64_L1", "R_128_d64_L1", "RT_128_d128_L2"]
CAP_NON_A = 4
# ⚠ The terminator now comes from the ONE shared constant. These were local copies of 3000/300 --
# the censored legacy rule -- and block 60 caught `scripted_episode` still using them while the
# calling script's artifact declared STALL=6500. A local copy of a shared constant is how an
# artifact ends up describing a run it did not perform.
from tasdata.bc import rollout_budget as _RB  # noqa: E402
N_EVAL, CHUNK = 200, 20
CAP_FRAMES, STALL = _RB.CAP_FRAMES, _RB.STALL
PIPE2_WINDOW = (530, 645)
EXPERT_AIRBORNE = 0.611
EXPERT_ONSETS_PER_1K = 27.5
BIN = 16
#: same windows as the fifty-first and the §1 read, so the three are comparable
LIFT_WINDOWS = {"goomba_288": (180, 320), "pipe1_432": (370, 480), "pipe2_592": (530, 645)}
LIVE_MASK = 0xFF


# ------------------------------------------------------------------ rollouts

def rollout(session, policy, cfg, start, seed, lut, byte_of) -> EpisodeTrace:
    """`capped` generation at the arm's own resolution."""
    s = cfg.frame_size
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    win = np.zeros((cfg.stack, s, s), np.uint8)
    win[:] = _resize_gray(obs.rgb, (s, s))
    best = since = frames = 0
    held_byte, remaining = None, 0
    while frames < CAP_FRAMES:
        if remaining <= 0:
            with torch.no_grad():
                p = torch.softmax(policy(torch.from_numpy(win[None]).float().div_(255.0))[0],
                                  dim=-1).numpy()
            c = int(rng.choice(len(p), p=p / p.sum()))
            b, length = int(byte_of[c]), max(1, int(lut[c]))
            if not (b & A_BIT):
                length = min(length, CAP_NON_A)
            held_byte, remaining = b, length
        byte = held_byte
        remaining -= 1
        obs = session.step(byte)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (s, s))
        t.record(obs, byte)
        frames += 1
        st = read_smb(obs.ram, obs.framecount)
        if st.player_state in (0x06, 0x0B):
            t.record_death(obs)
            return t
        if st.x_position > best:
            best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL:
                t.ended = "stuck"
                return t
    return t


def scripted_episode(session, start, seed: int, rates: dict) -> EpisodeTrace:
    """Each button drawn independently per frame at its own fixed rate -- the rate-matched bar."""
    t = EpisodeTrace(seed=seed)
    rng = np.random.default_rng(seed)
    obs = session.reset(start.frame)
    best = since = 0
    names = sorted(rates)
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


class _Ep:
    def __init__(self, e):
        self.__dict__.update(e)
        self.raw = e

    def to_dict(self):
        return self.raw


def resumable(path: Path, n: int, make):
    """200 episodes never survive this machine's kill cadence in one go; bank every CHUNK."""
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


# ------------------------------------------------------------------ scoring

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
    """`hold_stats` reports the centre; the ledger requires max and p99 beside every median."""
    a = np.asarray(list(vals), dtype=float)
    if a.size:
        h = {**h, "p99": float(np.percentile(a, 99)), "max": float(a.max())}
    return h


def score(label: str, traces) -> dict:
    xs = [max(f[0] for f in t.frames) for t in traces]
    frames = [f for t in traces for f in t.frames]
    h2v = [h for t in traces for h in a_hold_onsets(t.frames, PIPE2_WINDOW)]
    hav = [h for t in traces for h in a_hold_onsets(t.frames, (0, 10 ** 9))]
    b = np.asarray([f[3] for f in frames], dtype=np.int64)
    a = (b & A_BIT) > 0
    prev = np.zeros_like(a)
    prev[1:] = a[:-1]
    beh = behaviour_stats(frames)
    return {"label": label, "n": len(traces), "measurement_basis": "single_life",
            "loss": "plain cross-entropy", "seeds": 1,
            "grounded_enforced": False, "generation_rule": "capped (non-A runs <= 4 frames)",
            "x_median": float(np.median(xs)), "x_max": int(max(xs)),
            "a_hold_pipe2": with_tail(hold_stats(h2v), h2v),
            "a_hold_anywhere": with_tail(hold_stats(hav), hav),
            "a_onsets_per_1000_frames": float((a & ~prev).sum()) / len(b) * 1000,
            "expert_onsets_per_1000_frames": EXPERT_ONSETS_PER_1K,
            "noop_runs": noop_runs(frames),
            "clearance": clearance(xs), "vs_script_best_fixed_rate": vs_script(xs),
            "button_marginals": button_marginals(frames),
            "behaviour": beh, "expert_airborne": EXPERT_AIRBORNE,
            "airborne_minus_expert_pp": (beh["airborne_fraction"] - EXPERT_AIRBORNE) * 100
            if beh.get("airborne_fraction") is not None else None,
            "ended": {k: sum(1 for t in traces if getattr(t, "ended", None) == k)
                      for k in ("died", "stuck")}}


# ------------------------------------------------- corrected timing lift

def timing_lift(policy, cfg, ctx, corpus: str, vocab) -> dict:
    """Onsets vs non-A run starts, stratified, per obstacle -- the §1 estimator on this arm."""
    from tasdata.bc.data import FrameStackDataset
    from tasdata.dataset import load_run_dir
    names = [r.name for r in ctx.expert_train]
    root = ROOT / "data/runs" if corpus == "runs" else RUNS128
    runs = [load_run_dir(root / n) for n in names]
    ds = FrameStackDataset(runs, vocab, stack=4, label_mode="buttons",
                           frame_size=cfg.frame_size)
    n_cls = joint_size(vocab.size)
    a_mask = np.array([(vocab.decode_byte(c // N_BUCKETS) & A_BIT) > 0 for c in range(n_cls)])
    lo_all = min(v[0] for v in LIFT_WINDOWS.values())
    hi_all = max(v[1] for v in LIFT_WINDOWS.values())

    recs = []
    for run_id, entry in enumerate(ds.index):
        tr = np.asarray(runs[run_id].trace)
        w, s, pg = column(tr, "world"), column(tr, "stage"), column(tr, "pregame")
        x, ps = column(tr, "x_position"), column(tr, "player_state")
        tokens = ds.tokens[run_id]
        last = len(tokens) - 1
        has_a = np.array([(vocab.decode_byte(int(t)) & A_BIT) > 0 for t in tokens])
        off = int(ds.offsets[run_id])
        fidx = entry.frame_indices
        for row in range(entry.n_obs):
            m = min(int(fidx[row]) + ds.label_offset, last)
            if m >= len(x) or not (w[m] == 1 and s[m] == 1 and pg[m] == 1 and ps[m] == 8):
                continue
            if not (lo_all <= x[m] <= hi_all):
                continue
            p = max(m - 1, 0)
            recs.append((int(x[m]), bool(has_a[m] and not has_a[p] and m > 0),
                         bool(m == 0 or tokens[p] != tokens[m]), bool(has_a[m]), off + row))
    xs = np.asarray([r[0] for r in recs])
    ons = np.asarray([r[1] for r in recs])
    starts = np.asarray([r[2] for r in recs])
    tok_a = np.asarray([r[3] for r in recs])
    rows = [r[4] for r in recs]
    nonA = starts & ~tok_a

    pa = []
    for i in range(0, len(rows), 256):
        ob = torch.stack([ds[j][0] for j in rows[i:i + 256]])
        with torch.no_grad():
            p = torch.softmax(policy(ob), dim=-1).numpy()
        pa.extend(p[:, a_mask].sum(axis=1).tolist())
    pa = np.asarray(pa)

    out = {"estimator": "stratified 16px bins, bootstrap over onsets",
           "n_frames": len(recs), "obstacles": {}}
    for name, (lo, hi) in LIFT_WINDOWS.items():
        m = (xs >= lo) & (xs <= hi)
        if not m.any() or not (ons & m).any():
            out["obstacles"][name] = {"n_onsets": 0, "measurable": False}
            continue
        new_l, _, _ = stratified_lift(xs[m], pa[m], ons[m], nonA[m])
        old_l, _, _ = stratified_lift(xs[m], pa[m], ons[m], ~ons[m])
        out["obstacles"][name] = {
            "n_onsets": int((ons & m).sum()), "measurable": True,
            "corrected_lift": new_l,
            "corrected_ci": boot_lift(xs[m], pa[m], ons[m], nonA[m]),
            "confounded_lift_all_non_onset": old_l,
            "confounded_ci": boot_lift(xs[m], pa[m], ons[m], ~ons[m])}
    return out


# ------------------------------------------------------------------ driver

def load_arm(name: str):
    blob = torch.load(OUTDIR / f"{name}.pt", map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig.from_dict(cfg)
    policy = BCPolicy(cfg)
    policy.load_state_dict(blob["model_state"])
    policy.eval()
    return policy, cfg, blob


def main() -> None:
    t0 = time.time()
    only = [a for a in (sys.argv[1:] or ARMS)]
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    lut_cache: dict[str, np.ndarray] = {}
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out.setdefault("arms", {})
    out.update({
        "n_eval": N_EVAL, "generation_rule": "capped", "measurement_basis": "single_life",
        "device": "cpu -- this process spawns FCEUX and must never touch MPS",
        "bars": ("vs_script_best_fixed_rate is the measured best-fixed-rate table (pipe1 87.0, "
                 "pipe2 82.5, pipe3 23.5, pipe4 8.0); vs_script_rate_matched is a script at the "
                 "arm's OWN five marginals, rolled out fresh in stage 2")})

    for name in only:
        if not (OUTDIR / f"{name}.pt").exists():
            print(f"[{name}] no checkpoint yet -- skipping", flush=True)
            continue
        policy, cfg, blob = load_arm(name)
        print(f"\n[{name}] {cfg.frame_size}x{cfg.frame_size} d_model {cfg.d_model} "
              f"L{cfg.n_layers} steps {blob.get('step')} corpus {blob.get('corpus')}", flush=True)

        key = blob.get("corpus", "runs")
        if key not in lut_cache:
            idx_p = ROOT / f"data/runlength_index_{key}.npz"
            z = np.load(idx_p)
            lut_cache[key] = class_lengths({k: z[k] for k in ("rows", "joints", "lengths")},
                                           joint_size(ctx.vocab.size))
        lut = lut_cache[key]
        byte_of = np.array([ctx.vocab.decode_byte(c // N_BUCKETS)
                            for c in range(joint_size(ctx.vocab.size))], dtype=np.int64)

        tp = TRACEDIR / f"scaleup_{name}_{N_EVAL}.json"
        s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        try:
            traces = resumable(tp, N_EVAL,
                               lambda i: rollout(s, policy, cfg, start, i, lut, byte_of))
        finally:
            s.close()
        rec = score(name, traces)
        rec.update({"frame_size": cfg.frame_size, "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers, "train_steps": blob.get("step"),
                    "corpus": blob.get("corpus"), "traces": str(tp.relative_to(ROOT))})
        print("    computing corrected timing lift ...", flush=True)
        rec["timing_lift"] = timing_lift(policy, cfg, ctx, blob.get("corpus", "runs"), ctx.vocab)
        out["arms"][name] = rec
        OUT.write_text(json.dumps(out, indent=2, default=str))
        c = rec["clearance"]
        tl = rec["timing_lift"]["obstacles"]
        print(f"    p1 {c['pipe1']['rate'] * 100:.1f} p2 {c['pipe2']['rate'] * 100:.1f} "
              f"p3 {c['pipe3']['rate'] * 100:.1f} p4 {c['pipe4']['rate'] * 100:.1f} | "
              f"x_med {rec['x_median']:.0f} | A {rec['button_marginals']['rates']['A']:.3f} | "
              f"airb {rec['behaviour']['airborne_fraction'] * 100:.1f}%", flush=True)
        print(f"    corrected lift  goomba {tl['goomba_288'].get('corrected_lift')}  "
              f"pipe1 {tl['pipe1_432'].get('corrected_lift')}  "
              f"pipe2 {tl['pipe2_592'].get('corrected_lift')}", flush=True)

    # ---- stage 2: the rate-matched bar, one script per arm's own marginals ----
    for name in only:
        rec = out["arms"].get(name)
        if not rec or "vs_script_rate_matched" in rec:
            continue
        r = rec["button_marginals"]["rates"]
        rates = {k: float(r[k]) for k in ("A", "B", "Right", "Down", "Left")}
        strong = {**rates, "B": 1.0, "Right": 1.0}
        print(f"\n[{name}] rate-matched control at its own marginals "
              f"A {rates['A']:.3f} B {rates['B']:.3f} Right {rates['Right']:.3f}", flush=True)
        tp = TRACEDIR / f"scaleup_{name}_ratematched_strong_{N_EVAL}.json"
        s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        try:
            ctrl = resumable(tp, N_EVAL, lambda i: scripted_episode(s, start, i, strong))
        finally:
            s.close()
        cs = score(f"{name}_ratematched_strong", ctrl)
        pol_xs_rate = {k: rec["clearance"][k]["rate"] for k in rec["clearance"]}
        bar = {}
        for ob in cs["clearance"]:
            kp = int(round(rec["clearance"][ob]["rate"] * N_EVAL))
            kc = int(round(cs["clearance"][ob]["rate"] * N_EVAL))
            lo, hi = diff_ci(kc, N_EVAL, kp, N_EVAL)
            bar[ob] = {"policy_rate": pol_xs_rate[ob], "script_rate": cs["clearance"][ob]["rate"],
                       "difference_pp": (pol_xs_rate[ob] - cs["clearance"][ob]["rate"]) * 100,
                       "ci_pp": [lo * 100, hi * 100], "method": "Newcombe"}
        rec["vs_script_rate_matched"] = {
            "control_rates": strong,
            "control_note": ("the stronger 'Right+B held' reading, which is the harder bar; see "
                             "LEDGER 46th-block append"),
            "control_score": cs, "per_obstacle": bar}
        out["arms"][name] = rec
        OUT.write_text(json.dumps(out, indent=2, default=str))
        print("    " + "  ".join(f"{k} {v['difference_pp']:+.1f}" for k, v in bar.items()),
              flush=True)

    # ---- kill condition ----
    def lift(name, ob):
        a = out["arms"].get(name, {}).get("timing_lift", {}).get("obstacles", {}).get(ob, {})
        return a.get("corrected_lift"), a.get("corrected_ci")
    moved = []
    for name in ("R_128_d64_L1", "RT_128_d128_L2"):
        for ob in ("pipe1_432", "pipe2_592"):
            v, ci = lift(name, ob)
            if v is not None and ci is not None and ci[0] > 0:
                moved.append(f"{name}/{ob} {v:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    out["kill_condition"] = {
        "statement": ("if neither arm moves the corrected timing lift positive (interval excluding "
                      "zero) at BOTH pipe 1 and pipe 2, neither resolution nor capacity is the limit"),
        "positive_intervals": moved, "fired": len(moved) == 0}
    out["verdict"] = (
        f"KILL CONDITION FIRED: neither 128x128 nor a bigger transformer moved the corrected timing "
        f"lift positive at pipe 1 or pipe 2. **Neither resolution nor capacity is the limit**, and the "
        f"remaining candidates are the objective (balanced obstacle-window reweighting) and the frame "
        f"stack (4 -> 8, which carries approach velocity)."
        if out["kill_condition"]["fired"] else
        f"TIMING LIFT MOVED POSITIVE: {'; '.join(moved)}. The observation or the capacity was a real "
        f"limit; which one is told by whether R alone sufficed or RT was needed.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
