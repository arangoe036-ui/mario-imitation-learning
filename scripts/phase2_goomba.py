"""Phase 2, obstacle 1: sweep the Goomba from `capped`'s own states, distil, measure against both bars.

The Goomba at x≈288 is the dominant loss and it is also the pipe-1 regression, since pipe 1 sits at 432:
**70 of `capped`'s 146 deaths fall in x 272–319.** It is the best-specified obstacle in the level — 75 of 80
scripted timings clear it — and it sits in the representation's weak spot, because it needs a short precisely
timed jump rather than a long hold. Length buckets are near-exact below 6 frames, so short actions *are*
expressible; the deficit is timing, not resolution.

**The clearance threshold is derived, not chosen.** `capped`'s max_x histogram piles up at 272 (7), 288 (17)
and 304 (45), and the band **x 320–431 is completely empty** — no episode ends there. So x>320 is the first x
past the obstacle's far edge, and it is robust: the count is identical for every threshold from 320 to 384.
`capped`'s current rate is **130/200 = 65.0%**.

Five stages, each banked so a restart costs at most one chunk:

1. start states from `capped`'s own traces — grounded, `player_state==8`, x in 140–240, spread across seeds
2. sweep (trigger x, hold) with `on_ground()` **required** at the jump frame and a degeneracy assertion
3. re-run the winners with frame recording, and re-encode them as run-length training samples
4. distil into the run-length head with plain cross-entropy, a 1:1 mix and few epochs — **not** 13 epochs
   over near-identical segments, which is what failed at pipe 4
5. evaluate at n=200 against **both** bars: the best fixed-rate script and one rate-matched to the *new*
   policy's own marginals, which has to be re-run because the marginals move
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import session_when_free  # noqa: E402
from scripts.phase1_duration import PIPE2_WINDOW, _Ep  # noqa: E402
from scripts.phase1_variants import CAP_NON_A, noop_runs, rollout  # noqa: E402
from scripts.rate_matched_control import scripted_episode  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.overnight_lib import diff_ci  # noqa: E402
from tasdata.bc.pipe4_metrics import (  # noqa: E402
    A_BIT,
    PIPE_THRESHOLDS,
    a_hold_onsets,
    button_marginals,
    clearance,
    hold_stats,
)
from tasdata.bc.runlength import (  # noqa: E402
    N_BUCKETS,
    class_lengths,
    encode_joint,
    joint_size,
)
from tasdata.bc.script_baseline import behaviour_stats, conditional_rates, vs_script  # noqa: E402
from tasdata.bc.trace_log import EpisodeTrace  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS  # noqa: E402
from tasdata.ram import on_ground, read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRACEDIR = ROOT / "data/traces"
CAPPED_TRACES = TRACEDIR / "variant_capped_200.json"
BASE_CKPT = ROOT / "data/bc_phase1/runlength.pt"
IDX = ROOT / "data/phase1_runlength_index.npz"
NEW_CKPT = ROOT / "data/bc_phase1/goomba_distilled.pt"
OUT = ROOT / "data/phase2_goomba.json"
STATES = ROOT / "data/goomba_states.json"
SWEEP = ROOT / "data/goomba_sweep.json"
DEMOS = ROOT / "data/goomba_demos.npz"

RIGHT, B = NES_BUTTON_BITS["Right"], NES_BUTTON_BITS["B"]
CLEAR_X = 320            # derived: x 320-431 is empty in capped's max_x histogram
SURVIVE = 40
STATE_X = (140, 240)
N_STATES = 12
TRIGGERS = tuple(range(244, 292, 4))
HOLDS = (1, 2, 3, 4, 6, 8, 10, 12)
ATTEMPT_FRAMES = 200
STEPS, LR, CHUNK_STEPS = 300, 1e-4, 100
N_EVAL, CHUNK_EP = 200, 20
#: capped's baseline at this threshold, same seeds
CAPPED_GOOMBA = 130


def build_states() -> list[dict]:
    if STATES.exists():
        return json.loads(STATES.read_text())["states"]
    eps = json.loads(CAPPED_TRACES.read_text())["episodes"]
    byseed = defaultdict(list)
    for e in eps:
        for i, f in enumerate(e["frames"]):
            if len(f) >= 6 and f[5] == 1 and f[4] == 8 and STATE_X[0] <= f[0] < STATE_X[1]:
                byseed[e["seed"]].append({"seed": e["seed"], "frame_index": i + 1, "x": int(f[0])})
    rng = np.random.default_rng(0)
    seeds = sorted(byseed)
    rng.shuffle(seeds)
    picked = []
    for sd in seeds[:N_STATES]:
        pool = byseed[sd]
        picked.append(pool[len(pool) // 2])       # mid-run state for that seed
    STATES.write_text(json.dumps({"threshold_x": CLEAR_X, "x_window": list(STATE_X),
                                  "n": len(picked), "states": picked}, indent=2))
    return picked


def run_attempt(session, restore, *, trigger, hold, record=None):
    obs = restore()
    left, jumped, maxx = hold, False, 0
    clear_i, prev_x, xs, died = None, None, [], False
    win84 = _resize_gray(obs.rgb, (84, 84))
    for _ in range(ATTEMPT_FRAMES):
        st = read_smb(obs.ram, obs.framecount)
        byte = RIGHT | B
        if not jumped and st.x_position >= trigger:
            if not on_ground(obs.ram):
                return {"trigger": trigger, "hold": hold, "skipped": True,
                        "grounded_at_jump": False, "max_x": int(maxx), "cleared": False}, None
            jumped = True
        if jumped and left > 0:
            byte |= A_BIT
            left -= 1
        if record is not None:
            record.append((win84.copy(), byte))
        obs = session.step(byte)
        win84 = _resize_gray(obs.rgb, (84, 84))
        st = read_smb(obs.ram, obs.framecount)
        if prev_x is not None and st.x_position - prev_x < -100:
            break
        prev_x = st.x_position
        xs.append(int(st.x_position))
        maxx = max(maxx, st.x_position)
        if clear_i is None and st.x_position > CLEAR_X:
            clear_i = len(xs) - 1
        if st.player_state in (0x06, 0x0B):
            died = True
            break
        if clear_i is not None and len(xs) - clear_i > SURVIVE:
            break
    ok = clear_i is not None and not died and (len(xs) - clear_i) >= SURVIVE
    return {"trigger": trigger, "hold": hold, "skipped": False, "grounded_at_jump": True,
            "max_x": int(maxx), "died": died, "cleared": bool(clear_i is not None),
            "survived": bool(ok)}, record


def to_runlength_samples(rec, vocab, stack=4):
    """Recorded (obs84, byte) pairs -> (stacked obs, joint class) at every run start."""
    if not rec:
        return [], []
    bytes_ = np.array([r[1] for r in rec], dtype=np.uint8)
    tokens = vocab.encode(bytes_)
    obs_list, labels = [], []
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        rows = np.clip(np.arange(i - stack + 1, i + 1), 0, len(rec) - 1)
        obs_list.append(np.stack([rec[k][0] for k in rows]))
        labels.append(encode_joint(int(tokens[i]), j - i))
        i = j
    return obs_list, labels


def resumable_eval(path: Path, n: int, make):
    if path.exists() and json.loads(path.read_text()).get("n_episodes") == n:
        return [_Ep(e) for e in json.loads(path.read_text())["episodes"]]
    partial = path.with_suffix(".partial.json")
    traces = [_Ep(e) for e in (json.loads(partial.read_text())["episodes"]
                               if partial.exists() else [])]
    while len(traces) < n:
        for i in range(len(traces), min(len(traces) + CHUNK_EP, n)):
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


def score(label: str, traces) -> dict:
    xs = [max(f[0] for f in t.frames) for t in traces]
    frames = [f for t in traces for f in t.frames]
    h2 = [h for t in traces for h in a_hold_onsets(t.frames, PIPE2_WINDOW)]
    a = np.asarray(h2, dtype=float)
    goomba_k = sum(1 for v in xs if v > CLEAR_X)
    row = {"label": label, "n": len(traces), "measurement_basis": "single_life",
           "goomba_cleared": goomba_k, "goomba_rate": goomba_k / len(traces),
           "x_median": float(np.median(xs)), "x_p99": float(np.percentile(xs, 99)),
           "x_max": int(max(xs)),
           "a_hold_pipe2": {**hold_stats(h2),
                            "p99": (float(np.percentile(a, 99)) if a.size else None)},
           "noop_runs": noop_runs(frames),
           "clearance": clearance(xs), "conditional": conditional_rates(xs),
           "vs_script": vs_script(xs), "button_marginals": button_marginals(frames),
           "behaviour": behaviour_stats(frames),
           # freshly-run traces expose `.death`; resumed ones wrap the dict as `.raw`
           "deaths_272_319": sum(1 for t in traces
                                 if (getattr(t, "death", None) or getattr(t, "raw", {}).get("death"))
                                 and 272 <= (getattr(t, "death", None)
                                             or t.raw["death"])["x"] < 320),
           "ended": {k: sum(1 for t in traces if t.ended == k) for k in ("died", "stuck")}}
    m, bh = row["button_marginals"]["rates"], row["behaviour"]
    print(f"  {label:22s} goomba {row['goomba_rate'] * 100:5.1f}%  "
          f"deaths@272-319 {row['deaths_272_319']:3d}  "
          f"p1 {row['clearance']['pipe1']['rate'] * 100:5.1f} "
          f"p2 {row['clearance']['pipe2']['rate'] * 100:5.1f} "
          f"p3 {row['clearance']['pipe3']['rate'] * 100:5.1f} "
          f"p4 {row['clearance']['pipe4']['rate'] * 100:5.1f} | "
          f"A {m['A']:.3f} airb {bh['airborne_fraction'] * 100:4.1f}% "
          f"x_med {row['x_median']:4.0f}", flush=True)
    return row


def main() -> None:
    t0 = time.time()
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    n_cls = joint_size(ctx.vocab.size)
    z = np.load(IDX)
    eidx = {k: z[k] for k in ("rows", "joints", "lengths")}
    lut = class_lengths(eidx, n_cls)
    dists = {c: eidx["lengths"][eidx["joints"] == c] for c in range(n_cls)}
    byte_of = np.array([ctx.vocab.decode_byte(c // N_BUCKETS) for c in range(n_cls)],
                       dtype=np.int64)
    blob = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig(**cfg)

    states = build_states()
    cap_bytes = {e["seed"]: [f[3] for f in e["frames"]]
                 for e in json.loads(CAPPED_TRACES.read_text())["episodes"]}
    print(f"threshold x>{CLEAR_X} (derived: x 320-431 empty in capped's max_x). "
          f"capped baseline {CAPPED_GOOMBA}/200 = {CAPPED_GOOMBA / 2:.1f}%")
    print(f"{len(states)} start states, x {min(s['x'] for s in states)}-"
          f"{max(s['x'] for s in states)}, {len(TRIGGERS)} triggers x {len(HOLDS)} holds\n",
          flush=True)

    out = {"threshold_x": CLEAR_X, "threshold_note": "derived from an empty 320-431 max_x band",
           "capped_baseline_goomba": {"k": CAPPED_GOOMBA, "n": 200, "rate": CAPPED_GOOMBA / 200},
           "n_states": len(states), "triggers": list(TRIGGERS), "holds": list(HOLDS)}

    # ---- stage 2: the sweep ----------------------------------------------------------------
    if SWEEP.exists():
        sw = json.loads(SWEEP.read_text())
        print(f"sweep resumed: {sw['cleared']} of {sw['n']} configs cleared", flush=True)
    else:
        rows = []
        s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        try:
            for si, stt in enumerate(states):
                seq = cap_bytes[stt["seed"]][:stt["frame_index"]]
                obs = s.reset(start.frame)
                for byte in seq:
                    obs = s.step(byte)
                s.save_scratch(0)
                for trig in TRIGGERS:
                    for hold in HOLDS:
                        r, _ = run_attempt(s, lambda: s.load_scratch(0),
                                           trigger=trig, hold=hold)
                        rows.append({"state": si, "seed": stt["seed"], "x": stt["x"], **r})
                print(f"  state {si + 1}/{len(states)} (seed {stt['seed']}, x={stt['x']}): "
                      f"{sum(1 for r in rows if r.get('survived'))} winners so far", flush=True)
        finally:
            s.close()
        ok = [r for r in rows if r.get("survived")]
        assert len({bool(r.get("cleared")) for r in rows}) > 1, "degenerate: one outcome only"
        sw = {"n": len(rows), "skipped": sum(1 for r in rows if r.get("skipped")),
              "cleared": sum(1 for r in rows if r.get("cleared")), "survived": len(ok),
              "rows": rows}
        SWEEP.write_text(json.dumps(sw, indent=2))
        print(f"\nsweep: {sw['n']} configs, {sw['skipped']} skipped (airborne at the trigger), "
              f"{sw['cleared']} cleared, {sw['survived']} cleared AND survived", flush=True)
    out["sweep"] = {k: v for k, v in sw.items() if k != "rows"}
    winners = [r for r in sw["rows"] if r.get("survived")]
    if winners:
        best = min(winners, key=lambda r: (r["hold"], r["trigger"]))
        out["sweep"]["minimum"] = best
        print(f"  minimum winner: hold={best['hold']} trigger={best['trigger']} "
              f"(state seed {best['seed']})", flush=True)
    if not winners:
        out["verdict"] = ("NO SWEEP SOLUTION: no (trigger, hold) cleared the Goomba and survived from "
                          "any of the start states, so there is nothing to distil.")
        OUT.write_text(json.dumps(out, indent=2, default=str))
        print("\n" + out["verdict"])
        return

    # ---- stage 3: record the winners as run-length samples ----------------------------------
    if DEMOS.exists():
        z2 = np.load(DEMOS)
        demo_obs, demo_y = z2["obs"], z2["y"]
        print(f"demos resumed: {len(demo_y):,} run-length samples", flush=True)
    else:
        obs_all, y_all = [], []
        s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
        try:
            byst = defaultdict(list)
            for r in winners:
                byst[r["state"]].append(r)
            for si, rs in byst.items():
                stt = states[si]
                seq = cap_bytes[stt["seed"]][:stt["frame_index"]]
                obs = s.reset(start.frame)
                for byte in seq:
                    obs = s.step(byte)
                s.save_scratch(0)
                # keep at most 4 winners per state: variety without near-duplicates
                for r in sorted(rs, key=lambda q: (q["hold"], q["trigger"]))[:4]:
                    rec: list = []
                    run_attempt(s, lambda: s.load_scratch(0), trigger=r["trigger"],
                                hold=r["hold"], record=rec)
                    o, y = to_runlength_samples(rec, ctx.vocab, stack=cfg.stack)
                    obs_all.extend(o)
                    y_all.extend(y)
                print(f"  state {si}: {len(y_all):,} samples", flush=True)
        finally:
            s.close()
        demo_obs = np.stack(obs_all).astype(np.uint8)
        demo_y = np.asarray(y_all, dtype=np.int64)
        np.savez_compressed(DEMOS, obs=demo_obs, y=demo_y)
    out["demos"] = {"samples": int(len(demo_y)),
                    "distinct_classes": int(len(set(demo_y.tolist()))),
                    "a_containing_share": float(np.mean(
                        [(byte_of[c] & A_BIT) > 0 for c in demo_y]))}
    print(f"demo set: {len(demo_y):,} run-length samples, "
          f"{out['demos']['distinct_classes']} classes, "
          f"{out['demos']['a_containing_share'] * 100:.1f}% A-containing\n", flush=True)

    # ---- stage 4: distil -------------------------------------------------------------------
    if NEW_CKPT.exists():
        pol = BCPolicy(cfg)
        pol.load_state_dict(torch.load(NEW_CKPT, map_location="cpu",
                                       weights_only=False)["model_state"])
        pol.eval()
        print("distilled checkpoint resumed", flush=True)
    else:
        from torch.utils.data import DataLoader, Dataset

        class Mixed(Dataset):
            """Expert run-length samples plus the Goomba demos, 1:1 by sample count."""

            def __init__(self):
                self.base = ctx.dataset(ctx.expert_train)
                self.base.label_mode = "token"
                rng = np.random.default_rng(0)
                k = min(len(demo_y), len(eidx["rows"]))
                self.pick = rng.choice(len(eidx["rows"]), size=k, replace=False)

            def __len__(self):
                return len(self.pick) + len(demo_y)

            def __getitem__(self, i):
                if i < len(self.pick):
                    j = int(self.pick[i])
                    obs, _prev, _tok = self.base[int(eidx["rows"][j])]
                    return obs, int(eidx["joints"][j])
                d = i - len(self.pick)
                obs = torch.from_numpy(demo_obs[d].astype(np.float32) / 255.0)
                return obs, int(demo_y[d])

        def coll(batch):
            return (torch.stack([b[0] for b in batch]),
                    torch.tensor([b[1] for b in batch], dtype=torch.long))

        ds = Mixed()
        part = NEW_CKPT.with_suffix(".partial.pt")
        pol = BCPolicy(cfg)
        pol.load_state_dict(blob["model_state"])
        done = 0
        if part.exists():
            pb = torch.load(part, map_location="cpu", weights_only=False)
            pol.load_state_dict(pb["model_state"])
            done = int(pb["step"])
            print(f"    resuming distillation from step {done}/{STEPS}", flush=True)
        epochs = STEPS * 128 / max(len(ds), 1)
        print(f"distilling: {len(ds):,} samples (expert {len(ds.pick):,} + demo {len(demo_y):,}), "
              f"{STEPS} steps ~ {epochs:.1f} epochs, plain CE", flush=True)
        out["training"] = {"samples": len(ds), "expert": int(len(ds.pick)),
                           "demo": int(len(demo_y)), "steps": STEPS, "lr": LR,
                           "epochs": round(epochs, 2), "loss": "plain_cross_entropy"}
        if done < STEPS:
            pol.train()
            opt = torch.optim.AdamW(pol.parameters(), lr=LR, weight_decay=1e-4)
            g = torch.Generator().manual_seed(0)
            step = done
            while step < STEPS:
                for obs, y in DataLoader(ds, batch_size=128, shuffle=True, num_workers=0,
                                         collate_fn=coll, generator=g):
                    loss = torch.nn.functional.cross_entropy(pol(obs), y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                    opt.step()
                    step += 1
                    if step % CHUNK_STEPS == 0 or step >= STEPS:
                        torch.save({"model_state": pol.state_dict(), "step": step}, part)
                        print(f"    step {step}/{STEPS} loss {float(loss.detach()):.4f}",
                              flush=True)
                    if step >= STEPS:
                        break
        pol.eval()
        torch.save({"model_state": pol.state_dict(), "policy_config": cfg,
                    "loss": "plain_cross_entropy", "base": BASE_CKPT.name}, NEW_CKPT)
        part.unlink(missing_ok=True)

    # ---- stage 5: evaluate, both bars -------------------------------------------------------
    print("\nevaluating, capped generation rule, n=200", flush=True)
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        tr = resumable_eval(TRACEDIR / "goomba_distilled_200.json", N_EVAL,
                            lambda i: rollout(s, pol, cfg, start, i, mode="capped",
                                              lut=lut, dists=dists, byte_of=byte_of))
        out["arms"] = {"capped_baseline": score("capped (before)",
                                                [_Ep(e) for e in json.loads(
                                                    CAPPED_TRACES.read_text())["episodes"]]),
                       "distilled": score("goomba-distilled", tr)}
        rates = {k: round(out["arms"]["distilled"]["button_marginals"]["rates"][k], 3)
                 for k in ("A", "B", "Right", "Down", "Left")}
        print(f"\nrate-matched control at the NEW marginals {rates}", flush=True)
        ctl = resumable_eval(TRACEDIR / "goomba_ratematched_200.json", N_EVAL,
                             lambda i: scripted_episode(s, start, i, rates))
        out["arms"]["rate_matched_new"] = score("rate-matched (new)", ctl)
    finally:
        s.close()

    # ---- the question ----------------------------------------------------------------------
    b, g = out["arms"]["capped_baseline"], out["arms"]["distilled"]
    lo, hi = diff_ci(b["goomba_cleared"], b["n"], g["goomba_cleared"], g["n"])
    dlo, dhi = diff_ci(b["deaths_272_319"], b["n"], g["deaths_272_319"], g["n"])
    out["goomba_comparison"] = {
        "capped_rate": b["goomba_rate"], "distilled_rate": g["goomba_rate"],
        "delta_pp": (g["goomba_rate"] - b["goomba_rate"]) * 100, "ci_pp": [lo * 100, hi * 100],
        "improved": bool(lo > 0),
        "deaths_272_319": {"capped": b["deaths_272_319"], "distilled": g["deaths_272_319"],
                           "delta_pp": (g["deaths_272_319"] - b["deaths_272_319"]) / 2,
                           "ci_pp": [dlo * 100, dhi * 100], "reduced": bool(dhi < 0)},
    }
    # both bars
    out["bars"] = {}
    for bar, arm in (("best_script", None), ("rate_matched_new", "rate_matched_new")):
        if arm is None:
            out["bars"][bar] = {ob: g["vs_script"]["per_obstacle"][ob]["advantage_pp"]
                                for ob in PIPE_THRESHOLDS}
        else:
            c = out["arms"][arm]
            per = {}
            for ob in PIPE_THRESHOLDS:
                x, y = c["clearance"][ob], g["clearance"][ob]
                l2, h2 = diff_ci(x["k"], x["n"], y["k"], y["n"])
                per[ob] = {"advantage_pp": (y["rate"] - x["rate"]) * 100,
                           "ci_pp": [l2 * 100, h2 * 100], "beats": bool(l2 > 0)}
            out["bars"][bar] = per
    gc = out["goomba_comparison"]
    out["verdict"] = (
        f"DISTILLATION REDUCED GOOMBA DEATHS: clearance past x>{CLEAR_X} went "
        f"{b['goomba_rate'] * 100:.1f}% -> {g['goomba_rate'] * 100:.1f}% "
        f"({gc['delta_pp']:+.1f} pp [{gc['ci_pp'][0]:+.1f}, {gc['ci_pp'][1]:+.1f}]), and deaths in "
        f"272-319 went {b['deaths_272_319']} -> {g['deaths_272_319']} of 200. The loop closes."
        if gc["improved"] else
        f"DISTILLATION DID NOT REDUCE GOOMBA DEATHS: clearance past x>{CLEAR_X} went "
        f"{b['goomba_rate'] * 100:.1f}% -> {g['goomba_rate'] * 100:.1f}% "
        f"({gc['delta_pp']:+.1f} pp [{gc['ci_pp'][0]:+.1f}, {gc['ci_pp'][1]:+.1f}]), deaths in 272-319 "
        f"{b['deaths_272_319']} -> {g['deaths_272_319']} of 200. Search-and-distil failed on the "
        f"easiest, best-specified, most-demonstrated obstacle in the level, with a representation "
        f"built to receive the answer.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
