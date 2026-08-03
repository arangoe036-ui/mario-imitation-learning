"""Stage 3 arm A: iterated self-imitation. No teacher, so nothing blocks it.

Each round: roll the current policy out from the filtered expert start points, keep the
trajectories that made the most progress *from their own start*, add them to the training
set, and retrain. Three rounds.

Progress is measured from the start point, not absolutely, because the start points are
spread across all 32 levels -- an absolute x would rank a 8-1 start above a good 1-1 run
for no reason.

The acceptance filter is the whole experiment. Too loose and the policy trains on its own
mediocre behaviour and drifts; the stated diagnostic is that round 2 comes out worse than
round 1. Two-pass scoring keeps this cheap: rollouts are seeded and start from savestates,
so scoring every episode without recording and then re-rolling only the accepted seeds
reproduces them exactly while holding one episode of frames in memory at a time.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.bc.data import FrameStackDataset  # noqa: E402
from tasdata.bc.model import BCPolicy, PolicyConfig  # noqa: E402
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.session_player import play_episode  # noqa: E402
from tasdata.bc.statelib import load_index  # noqa: E402
from tasdata.bc.tokens import ActionVocab  # noqa: E402
from tasdata.bc.stage3_train import calibrate_and_eval, finetune  # noqa: E402
from tasdata.bc.train import TrainConfig  # noqa: E402
from tasdata.buttons import NES_BUTTON_ORDER  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "smb.nes"
MOVIE = ROOT / "data/movies/happylee_mars608-smb-warpless.fm2"
SELF_ROOT = ROOT / "data/runs_self"
CKPT_DIR = ROOT / "data/bc_stage3"
OUT = ROOT / "data/stage3_arm_a.jsonl"
SEED_CKPT = ROOT / "data/bc3/B_bernoulli_onset10x_step3000_recal.pt"

ROUNDS = 3
EPISODES = 150            # rollouts scored per round
ACCEPT_FRAC = 0.25        # keep the top quarter by progress-from-start
MIN_PROGRESS = 120        # ...and require this much absolute progress regardless
MAX_FRAMES = 500
FINETUNE_STEPS = 800
LIVE_SEEDS = 60           # for the per-round pipe1 measurement from 1-1
LEVEL_BONUS = 4000        # a level advance outranks any within-level distance


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


def episode_score(ep, start) -> float:
    """Progress from this episode's own start point, penalising death."""
    gained = ep.max_x_by_level.get(start.label, start.x) - start.x
    advanced = max(0, ep.levels_reached - 1)
    return gained + LEVEL_BONUS * advanced - 2000 * ep.deaths


def write_self_run(out_dir: Path, frames: np.ndarray, bytes_: np.ndarray) -> None:
    """Persist accepted rollouts in the same on-disk shape as a captured expert run.

    The action array is shifted by one so the loader's ``label_offset=1`` convention holds:
    the action associated with observation ``i`` must live at ``actions[i + 1]``, and in a
    rollout the action taken at observation ``i`` is the byte chosen there.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    actions = np.zeros(n, dtype=np.uint8)
    actions[1:] = bytes_[: n - 1]
    np.save(out_dir / "frames.npy", frames)
    np.save(out_dir / "actions.npy", actions)
    np.save(out_dir / "trace.npy", np.zeros((n, 1), dtype=np.int64))
    np.save(out_dir / "frame_indices.npy", np.arange(n, dtype=np.int64))
    (out_dir / "manifest.json").write_text(json.dumps({
        "n_frames": n, "synced": True, "category": "self", "measured_route": "self",
        "source": "stage3_arm_a", "label": out_dir.name,
    }, indent=2))


def main() -> None:
    vocab = ActionVocab.load(ROOT / "data/action_vocab.json")
    split = json.loads((ROOT / "data/split.json").read_text())["splits"]
    expert_train = [load_run_dir(ROOT / "data/runs" / n) for n in split["train"]]
    val_runs = [load_run_dir(ROOT / "data/runs" / n) for n in split["val"]]
    _, points = load_index(ROOT / "data/state_index.json")
    traj = [p for p in points if p.kind == "trajectory"]
    one_one = next(p for p in points if p.kind == "level_start" and p.label == "1-1")
    print(f"{len(traj)} filtered trajectory start points, {len(expert_train)} expert runs")

    ckpt = SEED_CKPT
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    self_dirs: list[Path] = []
    rows = []

    for rnd in range(1, ROUNDS + 1):
        t_round = time.perf_counter()
        policy, cfg, thresholds = load_policy(ckpt)
        rng = np.random.default_rng(1000 + rnd)
        chosen = rng.choice(len(traj), size=min(EPISODES, len(traj)), replace=False)
        picks = [traj[i] for i in chosen]

        frames_needed = sorted({p.frame for p in picks} | {one_one.frame})
        with FceuxSession(ROM, MOVIE, frames_needed) as session:
            # -- pass 1: score every rollout, record nothing -------------------
            t0 = time.perf_counter()
            scored = []
            for k, start in enumerate(picks):
                ep = play_episode(session, policy, start, vocab, seed=rnd * 10_000 + k,
                                  selection="sample", thresholds=thresholds,
                                  head_type=cfg.head_type, stack=cfg.stack,
                                  max_frames=MAX_FRAMES)
                scored.append((episode_score(ep, start), k, start, ep))
            roll_s = time.perf_counter() - t0

            scores = np.array([s for s, _, _, _ in scored], dtype=float)
            cutoff = float(np.quantile(scores, 1 - ACCEPT_FRAC))
            accepted = [t for t in scored if t[0] >= max(cutoff, MIN_PROGRESS)]
            rate = len(accepted) / len(scored)
            print(f"[round {rnd}] rolled {len(scored)} in {roll_s:.0f}s  "
                  f"score med {np.median(scores):.0f} p90 {np.percentile(scores, 90):.0f}  "
                  f"cutoff {max(cutoff, MIN_PROGRESS):.0f}  accepted {len(accepted)} "
                  f"({rate * 100:.0f}%)")

            # -- pass 2: re-roll only the accepted seeds, this time recording ---
            all_f, all_b = [], []
            for _score, k, start, _ep in accepted:
                rec: list = []
                play_episode(session, policy, start, vocab, seed=rnd * 10_000 + k,
                             selection="sample", thresholds=thresholds,
                             head_type=cfg.head_type, stack=cfg.stack,
                             max_frames=MAX_FRAMES, record=rec)
                if rec:
                    all_f.append(np.stack([r[0] for r in rec]))
                    all_b.append(np.array([r[1] for r in rec], dtype=np.uint8))

            if all_f:
                d = SELF_ROOT / f"round{rnd}"
                write_self_run(d, np.concatenate(all_f), np.concatenate(all_b))
                self_dirs.append(d)
                print(f"[round {rnd}] wrote {len(np.concatenate(all_b)):,} self frames -> {d}")

            # -- retrain (fine-tune) on expert + all self data so far -----------
            train_runs = expert_train + [load_run_dir(d) for d in self_dirs]
            train_set = FrameStackDataset(train_runs, vocab, stack=cfg.stack,
                                          label_mode="buttons")
            val_set = FrameStackDataset(val_runs, vocab, stack=cfg.stack,
                                        label_mode="buttons")
            tcfg = TrainConfig(name=f"stage3_round{rnd}", head_type="bernoulli",
                               steps=FINETUNE_STEPS, onset_weight=10.0,
                               batch_size=128, lr=1e-4, num_workers=0)
            new_ckpt = finetune(policy, cfg, tcfg, train_set, val_set, rnd)

            # -- evaluate --------------------------------------------------------
            policy2, cfg2, _ = load_policy(new_ckpt)
            thr2, offline = calibrate_and_eval(policy2, cfg2, val_set, train_set)
            eps = [play_episode(session, policy2, one_one, vocab, seed=s,
                                selection="sample", thresholds=thr2,
                                head_type=cfg2.head_type, stack=cfg2.stack)
                   for s in range(LIVE_SEEDS)]

        xs = [e.furthest_x for e in eps]
        k1 = sum(e.cleared_pipe1 for e in eps)
        levels = sorted({e.furthest_level for e in eps})
        row = {
            "round": rnd,
            "acceptance_rate": rate,
            "accepted": len(accepted),
            "self_frames": int(sum(len(b) for b in all_b)),
            "score_median": float(np.median(scores)),
            "a_onset_recall": offline["onset_recall"].get("A"),
            "exact_match": offline.get("exact_match"),
            "pipe1_rate": k1 / len(eps),
            "pipe1_k": k1, "pipe1_n": len(eps),
            "x_median": float(np.median(xs)),
            "furthest_level": max(levels) if levels else "-",
            "checkpoint": str(new_ckpt.name),
            "round_seconds": round(time.perf_counter() - t_round, 1),
        }
        rows.append(row)
        with OUT.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[round {rnd}] A-onset recall {row['a_onset_recall'] * 100:.1f}%  "
              f"pipe1 {k1}/{len(eps)} = {row['pipe1_rate'] * 100:.0f}%  "
              f"x_med {row['x_median']:.0f}  furthest {row['furthest_level']}  "
              f"accept {rate * 100:.0f}%  ({row['round_seconds']:.0f}s)\n")
        ckpt = new_ckpt

    print("round  accept%  A-onset  pipe1%   x_med  furthest")
    for r in rows:
        print(f"{r['round']:5d}  {r['acceptance_rate'] * 100:6.0f}  "
              f"{r['a_onset_recall'] * 100:6.1f}  {r['pipe1_rate'] * 100:5.0f}  "
              f"{r['x_median']:6.0f}  {r['furthest_level']:>8s}")
    if len(rows) >= 2 and rows[1]["pipe1_rate"] < rows[0]["pipe1_rate"]:
        print("\nRound 2 is worse than round 1 -- by the stated diagnostic, "
              "the acceptance filter is too loose.")


if __name__ == "__main__":
    main()
