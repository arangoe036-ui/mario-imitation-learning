"""Retest the Stage 3 oracle with the trained arm B checkpoint as the continuation policy.

The fixed Right+B continuation asked "does jumping hurt a run-right agent?", and in SMB it
never does -- that oracle agreed with the expert at A-onsets 50.9% of the time. This drives
both branches with arm B (45.5% A-onset recall offline), which is bootstrapping, not
circularity: the continuation is a policy that already plays somewhat, and the oracle is
being asked to improve on its jump timing specifically.

Same gate as before: a held-out expert run, agreement at A-onsets. Ties are allowed; only
confident decisions are counted as labels, and the confident fraction is reported.

Eligibility uses the backward-only ground test. The symmetric test used by the savestate
library requires y to be stable going *forward* as well, which excludes A-onsets **by
construction** -- at the frame the expert first presses A, y is about to rise. On this run
the symmetric filter left 24 onsets in 68,509 frames; backward-only leaves 216.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.oracle import decide_with_policy  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.statelib import grounded_backward_mask  # noqa: E402
from tasdata.buttons import NES_BUTTON_ORDER  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402
from tasdata.ram import PLAYER_STATE_NORMAL, column  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "smb.nes"
OUT = ROOT / "data/stage3_oracle_policy.json"
CKPT = ROOT / "data/bc3/B_bernoulli_onset10x_step3000_recal.pt"

N_UNIFORM = 500
N_ONSET = 500
A_BIT = 0x01

#: A label is only emitted when the two branches differ by more than this many x units.
#: Below it the oracle is inside its own sampling noise and should abstain.
MARGIN_MIN = 8

# (horizon, jump_hold, rollouts, suppress_off). suppress_off is how many frames the
# "don't jump" branch is forbidden from pressing A; 1 makes the two branches mirror images
# ("start a jump on this frame or not"), jump_hold makes the off-branch strictly handicapped.
VARIANTS = [
    (60, 20, 1, None),
    (60, 20, 1, 1),
    (120, 20, 1, 1),
    (120, 20, 3, 1),
]


def load_policy(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig(**cfg)
    policy = BCPolicy(cfg)
    policy.load_state_dict(blob["model_state"])
    policy.eval()
    thr = blob["thresholds"]
    return policy, cfg, np.array([thr[n] for n in NES_BUTTON_ORDER], dtype=np.float64)


def pick_held_out() -> tuple[Path, Path]:
    split = json.loads((ROOT / "data/split.json").read_text())
    for run_id in split["splits"]["test"]:
        run_dir = ROOT / "data/runs" / run_id
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("category") == "warpless":
            return run_dir, ROOT / str(manifest["movie"]).replace(str(ROOT) + "/", "")
    raise SystemExit("no warpless run in the test split")


def main() -> None:
    run_dir, movie = pick_held_out()
    run = load_run_dir(run_dir)
    actions = run.actions.astype(np.int64)
    trace = np.asarray(run.trace)
    n = min(len(trace) - 1, len(actions) - 1)
    truth = (actions[1 : n + 1] & A_BIT) > 0
    prev = (actions[:n] & A_BIT) > 0

    in_control = (
        (column(trace, "pregame")[:n] == 1)
        & (column(trace, "player_state")[:n] == PLAYER_STATE_NORMAL)
        & (column(trace, "world")[:n] >= 1) & (column(trace, "world")[:n] <= 8)
        & (column(trace, "stage")[:n] >= 1) & (column(trace, "stage")[:n] <= 4)
        & (column(trace, "x_position")[:n] > 0)
        & (column(trace, "time")[:n] > 0)
        & grounded_backward_mask(run)[:n]
    )
    onsets = np.flatnonzero(truth & ~prev & in_control)
    rng = np.random.default_rng(0)
    uni = rng.choice(np.flatnonzero(in_control), size=N_UNIFORM, replace=False)
    ons = rng.choice(onsets, size=min(N_ONSET, onsets.size), replace=False)
    print(f"held-out run : {run_dir.name}")
    print(f"eligible frames (backward-only ground test): {in_control.sum():,}")
    print(f"A-onsets available: {onsets.size:,}   sampling {uni.size} uniform + {ons.size} onset")

    frames = sorted(set(uni.tolist()) | set(ons.tolist()))
    ordinal_of = {f: i for i, f in enumerate(frames)}
    policy, cfg, thresholds = load_policy(CKPT)
    print(f"continuation policy: {CKPT.name}\n")

    out = []
    with FceuxSession(ROM, movie, frames) as session:
        print(f"session up: {session.n_states} states in {session.build_seconds:.1f}s")
        for horizon, jump_hold, rollouts, suppress_off in VARIANTS:
            t0 = time.perf_counter()
            dec = {}
            for f in frames:
                dec[f] = decide_with_policy(
                    session, ordinal_of[f], f, policy, thresholds,
                    horizon=horizon, jump_hold=jump_hold, stack=cfg.stack,
                    n_rollouts=rollouts, seed=1234, suppress_off=suppress_off,
                )
            secs = time.perf_counter() - t0

            def stats(idx):
                conf = [i for i in idx if abs(dec[int(i)].margin) >= MARGIN_MIN]
                agree_all = np.mean([dec[int(i)].choose_a == bool(truth[int(i)]) for i in idx])
                agree_conf = (
                    np.mean([dec[int(i)].choose_a == bool(truth[int(i)]) for i in conf])
                    if conf else float("nan")
                )
                return float(agree_all), float(agree_conf), len(conf) / len(idx)

            a_all, a_conf, a_frac = stats(uni)
            o_all, o_conf, o_frac = stats(ons)
            jump_rate = float(np.mean([dec[int(i)].choose_a for i in uni]))
            em = sum(d.emulator_frames for d in dec.values())
            row = {
                "horizon": horizon, "jump_hold": jump_hold, "rollouts": rollouts,
                "suppress_off": suppress_off,
                "margin_min": MARGIN_MIN,
                "oracle_jump_rate": jump_rate,
                "expert_a_rate": float(np.mean(truth[uni])),
                "agreement_overall": a_all,
                "agreement_overall_confident": a_conf,
                "confident_fraction_uniform": a_frac,
                "agreement_at_a_onsets": o_all,
                "agreement_at_a_onsets_confident": o_conf,
                "confident_fraction_onsets": o_frac,
                "n_onsets": int(ons.size),
                "emulator_frames_per_label": em / len(dec),
                "seconds_per_label": secs / len(dec),
            }
            out.append(row)
            print(
                f"  h={horizon:3d} r={rollouts} off={suppress_off}  jump={jump_rate * 100:5.1f}% "
                f"(expert {row['expert_a_rate'] * 100:4.1f}%)  "
                f"onset={o_all * 100:5.1f}%  onset|conf={o_conf * 100:5.1f}% "
                f"(conf {o_frac * 100:4.1f}%)  overall={a_all * 100:5.1f}%  "
                f"{row['seconds_per_label']:.2f}s/label"
            )

    OUT.write_text(json.dumps(
        {"held_out_run": run_dir.name, "checkpoint": CKPT.name, "variants": out}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
