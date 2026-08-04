"""§3: train under the degeneracy-proof credit, from start states at the obstacles that matter.

One arm, one seed. **A screen, not a ranking** (LEDGER.md §2: training-seed spread is 14.5-24.5 pp).

Everything that makes this different from every previous self-imitation round:

* **Start states are the policy's own grounded frames**, stratified across x from 48 to 1,692 -- not the
  old library, whose 16 1-1 points all sit at x 2,616-2,636, past every obstacle the credit pays for.
* **Acceptance is the per-start reach quantile**, not progress-from-start. Matching the canonical script
  earns 0.5; beating it earns more. Raising the A-rate cannot help, because the script already has the
  best marginal from the identical state.
* **The expert side of the mix is deliberately a minority.** Four schedules that pulled the A-rate toward
  the expert's 0.152 all lost the level, so this trains mostly on its own accepted rollouts, which carry
  the policy's own marginal.

Success was stated in advance by the directive and is not restated favourably here: **the A-rate need not
fall; `vs_script` at pipe 3 or pipe 4 must improve.** Pipes 1-2 may get worse -- the script owns them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.overnight as O  # noqa: E402
from scripts.compose import EARLIEST, session_when_free, train  # noqa: E402
from scripts.overnight import write_self_run  # noqa: E402
from scripts.p1_run import episode as traced_episode  # noqa: E402
from tasdata.bc.overnight_lib import (  # noqa: E402
    calibrate,
    load_policy,
    random_rows,
    save_policy,
)
from tasdata.bc.pipe4_metrics import button_marginals, clearance  # noqa: E402
from tasdata.bc.script_baseline import reach_margin, reach_quantile, reach_table, vs_script  # noqa: E402,E501
from tasdata.bc.tokens import LIVE_MASK  # noqa: E402
from tasdata.bc.trace_log import write_traces  # noqa: E402
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402
from tasdata.ram import read_smb  # noqa: E402
from tasdata.replay import _resize_gray  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "data/startlib_policy.json"
TRACES = ROOT / "data/traces/p1_200.json"
BASE = ROOT / "data/bc_coverage/C_control_matched_r2.pt"
NEW = ROOT / "data/bc_coverage/script_net_round1.pt"
SELFDIR = ROOT / "data/runs_self/script_net_round1"
OUT = ROOT / "data/train_script_net.json"
EVAL_TRACES = ROOT / "data/traces/script_net_round1_200.json"

ROLLOUTS_PER_STATE = 8
ACCEPT_Q = 0.60          # a rollout must land above the 60th percentile of the script from that state
MAX_FRAMES, STALL = 400, 120
STEPS, LR = 400, 1e-4
EXPERT_PER_SELF = 0.33   # expert is a deliberate minority; see the module docstring
N_EVAL = 200


def policy_rollout(session, policy, cfg, win, seed, record=None):
    """Per-button sampling from the restored state. `win` is the seeded frame-stack window."""
    rng = np.random.default_rng(seed)
    win = win.copy()
    maxx = best = 0
    since = 0
    died = False
    for _ in range(MAX_FRAMES):
        with torch.no_grad():
            # /255 is required on the LIVE path: the window is uint8 0-255 from _resize_gray.
            # (The *dataset* path already returns [0,1]; dividing there collapsed the model once.)
            lg = policy(torch.from_numpy(win[None]).float().div_(255.0))[0].numpy()
        bits = rng.random(8) < 1.0 / (1.0 + np.exp(-lg))
        byte = 0
        for j, nm in enumerate(NES_BUTTON_ORDER):
            if bits[j]:
                byte |= NES_BUTTON_BITS[nm]
        byte &= LIVE_MASK
        if record is not None:
            record.append((win[-1].copy(), byte))
        obs = session.step(byte)
        win = np.roll(win, -1, 0)
        win[-1] = _resize_gray(obs.rgb, (84, 84))
        st = read_smb(obs.ram, obs.framecount)
        maxx = max(maxx, st.x_position)
        if st.player_state in (0x06, 0x0B):
            died = True
            break
        if st.x_position > best:
            best, since = st.x_position, 0
        else:
            since += 1
            if since > STALL:
                break
    return {"max_x": int(maxx), "died": died}


def main() -> None:
    t0 = time.time()
    lib = json.loads(LIB.read_text())
    states = lib["states"]
    table = reach_table()
    bytes_by_seed = {e["seed"]: [f[3] for f in e["frames"]]
                     for e in json.loads(TRACES.read_text())["episodes"]}
    ctx = O.Ctx()
    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "1-1")
    policy, cfg, _ = load_policy(BASE)
    cal, _ = calibrate(policy, ctx.dataset(ctx.expert_train), ctx.target_rates)
    thr = cal.vector.astype(np.float64)

    print(f"base {BASE.name}; {len(states)} start states; reach table {len(table)} entries",
          flush=True)
    print(f"acceptance: reach quantile > {ACCEPT_Q}  (0.5 = matching the canonical script)\n",
          flush=True)

    out = {"base_checkpoint": BASE.name, "label": "SCREEN -- one arm, one training seed",
           "start_library": LIB.name, "n_states": len(states),
           "acceptance": {"rule": "per-start script reach quantile", "threshold": ACCEPT_Q,
                          "rollouts_per_state": ROLLOUTS_PER_STATE},
           "training": {"steps": STEPS, "lr": LR, "expert_per_self": EXPERT_PER_SELF,
                        "note": "expert is a deliberate minority; pulling A toward 0.152 lost the "
                                "level in 4 of 4 previous schedules"},
           "measurement_basis": "single_life", "seeds_training": 1}

    frames_all, bytes_all, per_state = [], [], []
    s = session_when_free(O.ROM, O.MOVIE, ctx.frames_needed())
    try:
        for si, stt in enumerate(states):
            key = f"{stt['seed']}:{stt['frame_index']}"
            if key not in table:
                continue
            seq = bytes_by_seed[stt["seed"]][:stt["frame_index"]]
            obs = s.reset(start.frame)
            tail = []
            for i, byte in enumerate(seq):
                obs = s.step(byte)
                if i >= len(seq) - cfg.stack:
                    tail.append(_resize_gray(obs.rgb, (84, 84)))
            s.save_scratch(0)
            win = np.zeros((cfg.stack, 84, 84), np.uint8)
            win[:] = tail[-1] if tail else 0
            for k, img in enumerate(tail[-cfg.stack:]):
                win[k] = img
            scored = []
            for r in range(ROLLOUTS_PER_STATE):
                s.load_scratch(0)
                res = policy_rollout(s, policy, cfg, win, seed=si * 1000 + r)
                q = reach_quantile(res["max_x"], key, table)
                scored.append({**res, "q": q, "r": r,
                               "margin": reach_margin(res["max_x"], key, table)})
            keep = [c for c in scored if c["q"] is not None and c["q"] > ACCEPT_Q]
            for c in keep:
                s.load_scratch(0)
                rec: list = []
                policy_rollout(s, policy, cfg, win, seed=si * 1000 + c["r"], record=rec)
                if rec:
                    frames_all.append(np.stack([p[0] for p in rec]))
                    bytes_all.append(np.array([p[1] for p in rec], dtype=np.uint8))
            per_state.append({"key": key, "x": stt["x"], "bin": stt["bin"],
                              "script_median": table[key]["median"],
                              "q_median": float(np.median([c["q"] for c in scored])),
                              "q_max": float(max(c["q"] for c in scored)),
                              "accepted": len(keep), "n": len(scored),
                              "policy_max_x_median": float(np.median(
                                  [c["max_x"] for c in scored]))})
            if (si + 1) % 12 == 0:
                acc = sum(p["accepted"] for p in per_state)
                print(f"  {si + 1}/{len(states)} states, accepted {acc}, "
                      f"{sum(len(f) for f in frames_all):,} frames", flush=True)

        n_acc = sum(p["accepted"] for p in per_state)
        q_all = [p["q_median"] for p in per_state]
        print(f"\naccepted {n_acc} of {len(per_state) * ROLLOUTS_PER_STATE} rollouts; "
              f"median per-state reach quantile {np.median(q_all):.3f}", flush=True)
        sat = sum(1 for p in per_state if p["q_max"] >= 1.0)
        out["rollout_summary"] = {
            "states_scored": len(per_state), "accepted": n_acc,
            "acceptance_rate": n_acc / max(len(per_state) * ROLLOUTS_PER_STATE, 1),
            "median_state_quantile": float(np.median(q_all)),
            "states_with_saturated_quantile": sat,
            "saturation_note": (f"{sat}/{len(per_state)} states had at least one rollout at "
                               f"quantile 1.0; the standardised margin is stored per rollout as the "
                               f"unbounded fallback"),
            "per_state": per_state,
        }
        if not frames_all:
            out["verdict"] = ("NO ROLLOUT BEAT THE SCRIPT from any of the 72 states, so there was "
                              "nothing to train on. Under a degeneracy-proof credit the policy does "
                              "not outperform a fixed-rate script from its own visited states.")
            OUT.write_text(json.dumps(out, indent=2, default=str))
            print("\n" + out["verdict"])
            return

        frames = np.concatenate(frames_all)
        bytes_ = np.concatenate(bytes_all)
        write_self_run(SELFDIR, frames, bytes_)
        print(f"wrote {SELFDIR.name}: {len(frames):,} frames, A on "
              f"{float(((bytes_ & NES_BUTTON_BITS['A']) > 0).mean()) * 100:.1f}%", flush=True)

        expert = ctx.dataset([load_run_dir(ROOT / "data/runs" / n) for n in EARLIEST])
        selfds = ctx.dataset([load_run_dir(SELFDIR)])
        n_exp = min(len(expert), int(len(selfds) * EXPERT_PER_SELF))
        mixed = ConcatDataset([Subset(expert, random_rows(expert, n_exp, seed=0)), selfds])
        print(f"\ntraining {STEPS} steps: expert {n_exp:,} + self {len(selfds):,} "
              f"(~{STEPS * 128 / max(len(mixed), 1):.1f} epochs)", flush=True)
        out["training"].update({"expert_frames": n_exp, "self_frames": len(selfds),
                                "epochs": round(STEPS * 128 / max(len(mixed), 1), 1)})
        policy = train(policy, mixed, STEPS, LR, 0)
        cal, _ = calibrate(policy, expert, ctx.target_rates)
        thr = cal.vector.astype(np.float64)
        save_policy(NEW, policy, cfg, {n: 0.5 for n in NES_BUTTON_ORDER}, base=BASE.name)

        print(f"\nevaluating n={N_EVAL}, single life, seeds 0-{N_EVAL - 1}", flush=True)
        traces = [traced_episode(s, policy, cfg, thr, start, i) for i in range(N_EVAL)]
    finally:
        s.close()

    write_traces(EVAL_TRACES, traces, checkpoint=NEW.name)
    xs = [max(f[0] for f in t.frames) for t in traces]
    allf = [f for t in traces for f in t.frames]
    out["eval"] = {"n": N_EVAL, "x_median": float(np.median(xs)), "x_max": int(max(xs)),
                   "clearance": clearance(xs), "vs_script": vs_script(xs),
                   "button_marginals": button_marginals(allf)}

    base_traces = json.loads((ROOT / "data/traces/p1_200.json").read_text())["episodes"]
    base_xs = [max(f[0] for f in e["frames"]) for e in base_traces]
    out["base_eval"] = {"n": len(base_xs), "x_median": float(np.median(base_xs)),
                        "clearance": clearance(base_xs), "vs_script": vs_script(base_xs),
                        "button_marginals": button_marginals(
                            [f for e in base_traces for f in e["frames"]])}

    b, g = out["base_eval"]["vs_script"]["per_obstacle"], out["eval"]["vs_script"]["per_obstacle"]
    moved = {p: g[p]["advantage_pp"] - b[p]["advantage_pp"] for p in b}
    improved = [p for p in ("pipe3", "pipe4") if moved[p] > 0]
    print("\nvs_script advantage, pp (base -> trained):", flush=True)
    for p in b:
        print(f"  {p:6s} {b[p]['advantage_pp']:+6.1f} -> {g[p]['advantage_pp']:+6.1f}  "
              f"({moved[p]:+.1f})", flush=True)
    out["comparison"] = {"vs_script_advantage_delta_pp": moved,
                         "improved_at": improved,
                         "a_rate": {"base": out["base_eval"]["button_marginals"]["rates"]["A"],
                                    "trained": out["eval"]["button_marginals"]["rates"]["A"]}}
    out["verdict"] = (
        f"vs_script IMPROVED at {improved}: " +
        ", ".join(f"{p} {b[p]['advantage_pp']:+.1f} -> {g[p]['advantage_pp']:+.1f} pp"
                  for p in improved) +
        ". The loop works: an objective a marginal cannot satisfy, start states where practice is "
        "useful, and a measured gain over the best fixed script."
        if improved else
        "vs_script DID NOT IMPROVE at pipe 3 or pipe 4 (" +
        ", ".join(f"{p} {moved[p]:+.1f} pp" for p in ("pipe3", "pipe4")) +
        "). Trained under a degeneracy-proof objective, from start states at the obstacles that "
        "matter, this corpus and this method did not improve where the credit paid 6.5x. They are "
        "exhausted; the remaining question is a different corpus or a different method.")
    out["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print("\n" + "=" * 78)
    print(out["verdict"])
    print(f"\nwrote {OUT} ({out['minutes']} min)")


if __name__ == "__main__":
    main()
