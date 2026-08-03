"""Stage 2 final evaluation on one persistent FCEUX, both arms, serial.

Also times the same workload on the old process-per-episode player so the speedup is a
measurement rather than a claim.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.buttons import NES_BUTTON_ORDER  # noqa: E402
from tasdata.bc.live import LivePlayer  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.session_player import evaluate_on_session, play_episode  # noqa: E402
from tasdata.bc.statelib import load_index  # noqa: E402
from tasdata.bc.tokens import ActionVocab  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "smb.nes"
MOVIE = ROOT / "data/movies/happylee_mars608-smb-warpless.fm2"
OUT = ROOT / "data/stage2_final_session.jsonl"

# Final protocol: the six level starts that the training-time evals used, so the numbers
# are comparable with the frozen table, plus 20 seeds for anything stochastic.
FINAL_LEVELS = ["1-1", "1-2", "1-3", "2-1", "3-1", "4-1"]
SEEDS = 20

ARMS = {
    "A_bernoulli_only": ROOT / "data/bc3/A_bernoulli_only_step3000_recal.pt",
    "B_bernoulli_onset10x": ROOT / "data/bc3/B_bernoulli_onset10x_step3000_recal.pt",
}
RULES = ["threshold", "sticky", "sample"]


def load_policy(path: Path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg = blob["policy_config"]
    if isinstance(cfg, dict):
        cfg = PolicyConfig(**cfg)
    policy = BCPolicy(cfg)
    policy.load_state_dict(blob["model_state"])
    policy.eval()
    thr = blob["thresholds"]
    # Stored per button name; the head emits logits in NES_BUTTON_ORDER.
    thresholds = np.array([thr[name] for name in NES_BUTTON_ORDER], dtype=np.float64)
    return policy, cfg, thresholds


def main() -> None:
    vocab = ActionVocab.load(ROOT / "data/action_vocab.json")
    expert_bytes = set(json.loads((ROOT / "data/expert_bytes.json").read_text()))
    _, points = load_index(ROOT / "data/state_index.json")
    starts = []
    for label in FINAL_LEVELS:
        cand = [p for p in points if p.label == label and p.kind == "level_start"]
        if cand:
            starts.append(cand[0])
    print(f"start points: {[f'{s.label}@{s.frame}' for s in starts]}")

    rows = []
    with FceuxSession(ROM, MOVIE, sorted({s.frame for s in starts})) as session:
        print(f"session up: {session.n_states} states in {session.build_seconds:.1f}s")
        for arm, ckpt in ARMS.items():
            policy, cfg, thresholds = load_policy(ckpt)
            row = {"kind": "final_session", "arm": arm, "checkpoint": ckpt.name,
                   "live_by_rule": {}}
            for rule in RULES:
                t0 = time.perf_counter()
                summary = evaluate_on_session(
                    session, policy, starts, vocab,
                    seeds=SEEDS, selection=rule, thresholds=thresholds,
                    head_type=cfg.head_type, stack=cfg.stack,
                    expert_bytes=expert_bytes,
                )
                summary["wall_seconds"] = round(time.perf_counter() - t0, 1)
                row["live_by_rule"][rule] = summary
                print(
                    f"  {arm:22s} {rule:9s} eps={summary['n_episodes']:3d} "
                    f"x_med={summary['furthest_x'].get('median', 0):7.1f} "
                    f"pipe1={(summary.get('cleared_pipe1_rate') or 0) * 100:5.1f}% "
                    f"Ahold_max={summary['longest_a_hold'].get('max', 0):5.1f} "
                    f"{summary['wall_seconds']:6.1f}s "
                    f"errs={len(summary.get('errors', []))}"
                )
            rows.append(row)

        # --- before/after timing on an identical small workload -----------------
        policy, cfg, thresholds = load_policy(ARMS["B_bernoulli_onset10x"])
        n_time = 4
        t0 = time.perf_counter()
        for i in range(n_time):
            play_episode(session, policy, starts[0], vocab, seed=i, selection="sample",
                         thresholds=thresholds, head_type=cfg.head_type,
                         stack=cfg.stack, max_frames=1500)
        after = time.perf_counter() - t0

    t0 = time.perf_counter()
    old_ok = 0
    for i in range(n_time):
        try:
            lp = LivePlayer(
                ROM, vocab, stack=cfg.stack, head_type=cfg.head_type,
                thresholds=thresholds, expert_movie=MOVIE, max_frames=1500,
            )
            lp.play(policy, seed=i, selection="sample", level="1-1")
            old_ok += 1
        except Exception as exc:
            print(f"  [old player] episode {i} failed: {type(exc).__name__}: {exc}"[:160])
    before = time.perf_counter() - t0

    timing = {
        "kind": "timing", "episodes": n_time, "frames_each": 1500,
        "old_process_per_episode_seconds": round(before, 1),
        "old_episodes_succeeded": old_ok,
        "session_seconds": round(after, 1),
        "speedup": round(before / after, 1) if after else None,
    }
    print(f"\nwall-clock, {n_time} x 1500-frame episodes:")
    print(f"  before (process per episode): {before:6.1f}s  ({old_ok}/{n_time} succeeded)")
    print(f"  after  (one session)       : {after:6.1f}s")
    if after:
        print(f"  speedup                    : {before / after:.1f}x")

    with OUT.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fh.write(json.dumps(timing) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
