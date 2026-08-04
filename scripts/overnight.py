"""Overnight run. Tiers in priority order; one task failing never stops the rest.

Every result is appended to ``data/overnight.jsonl`` the moment it lands, and a summary
is regenerated every two minutes by a background thread, so an interrupted run still
leaves a readable record of everything that finished.

Only one FCEUX may exist at a time (enforced by a file lock in ``session.py``), so every
emulator task opens a session, uses it, and closes it before the next begins.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tasdata.bc.data import FrameStackDataset  # noqa: E402
from tasdata.bc.overnight_lib import (  # noqa: E402
    A_INDEX,
    calibrate,
    diff_ci,
    eval_live,
    expert_target_rates,
    fresh_policy,
    load_policy,
    onset_metrics,
    random_rows,
    save_policy,
    train_policy,
    wilson,
)
from tasdata.bc.session import FceuxSession  # noqa: E402
from tasdata.bc.statelib import load_index  # noqa: E402
from tasdata.bc.tokens import ActionVocab  # noqa: E402
from tasdata.buttons import NES_BUTTON_ORDER  # noqa: E402
from tasdata.dataset import load_run_dir  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "smb.nes"
MOVIE = ROOT / "data/movies/happylee_mars608-smb-warpless.fm2"
JSONL = ROOT / "data/overnight.jsonl"
SUMMARY = ROOT / "data/overnight_summary.md"
PLOTS = ROOT / "data/plots"
CKPTS = ROOT / "data/bc_overnight"

STAGE2_CKPT = ROOT / "data/bc3/B_bernoulli_onset10x_step3000_recal.pt"
ROUND1_CKPT = ROOT / "data/bc_stage3/stage3_round1.pt"

EVAL_SEEDS = 200
LEVELS = ["1-1", "2-1"]

_lock = threading.Lock()
_started = time.time()


def emit(kind: str, **payload) -> dict:
    """Append one result to the JSONL stream immediately."""
    row = {"kind": kind, "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "elapsed_min": round((time.time() - _started) / 60, 1), **payload}
    with _lock:
        with JSONL.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    return row


def log(*a) -> None:
    print(f"[{(time.time() - _started) / 60:6.1f}m]", *a, flush=True)


def task(name: str):
    """Run a task, capture any failure as a result rather than letting it propagate."""
    def wrap(fn):
        def run(*a, **kw):
            log(f"=== START {name}")
            t0 = time.time()
            try:
                out = fn(*a, **kw)
                emit("task_done", task=name, seconds=round(time.time() - t0, 1),
                     result=out)
                log(f"=== DONE {name} ({(time.time() - t0) / 60:.1f}m)")
                return out
            except Exception as exc:
                emit("task_failed", task=name, seconds=round(time.time() - t0, 1),
                     error=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc()[-2000:])
                log(f"=== FAILED {name}: {type(exc).__name__}: {exc}")
                return None
        return run
    return wrap


# ------------------------------------------------------------------ context

class Ctx:
    def __init__(self):
        self.vocab = ActionVocab.load(ROOT / "data/action_vocab.json")
        split = json.loads((ROOT / "data/split.json").read_text())["splits"]
        self.split = split
        self.expert_train = [load_run_dir(ROOT / "data/runs" / n) for n in split["train"]]
        self.val_runs = [load_run_dir(ROOT / "data/runs" / n) for n in split["val"]]
        self.expert_bytes = set(json.loads((ROOT / "data/expert_bytes.json").read_text()))
        _, points = load_index(ROOT / "data/state_index.json")
        self.points = points
        self.traj = [p for p in points if p.kind == "trajectory"]
        self.starts = [next(p for p in points if p.kind == "level_start" and p.label == lv)
                       for lv in LEVELS]
        self.target_rates = expert_target_rates(self.expert_train)
        self.val_set = self.dataset(self.val_runs)
        _, self.cfg, _ = load_policy(STAGE2_CKPT)

    def dataset(self, runs, stack: int = 4) -> FrameStackDataset:
        return FrameStackDataset(runs, self.vocab, stack=stack, label_mode="buttons")

    def frames_needed(self, extra=()):
        return sorted({p.frame for p in self.starts} | set(extra))


def full_eval(ctx, policy, cfg, tag: str, *, seeds: int = EVAL_SEEDS, train_set=None,
              extra_frames=()) -> dict:
    """Recalibrate, measure offline onset recall, then live-evaluate. Always in that order."""
    train_set = train_set if train_set is not None else ctx.dataset(ctx.expert_train)
    calibration, _ = calibrate(policy, train_set, ctx.target_rates)
    thr = calibration.vector.astype(np.float64)
    offline = onset_metrics(policy, ctx.val_set, thr)
    with FceuxSession(ROM, MOVIE, ctx.frames_needed(extra_frames)) as session:
        live = eval_live(session, policy, thr, ctx.starts, ctx.vocab, cfg,
                         seeds=seeds, expert_bytes=ctx.expert_bytes)
    return {"tag": tag, "thresholds": dict(calibration.thresholds),
            "realized_rate": dict(calibration.realized_rate),
            "target_rate": dict(calibration.target_rate),
            "offline": offline, "live": live}


# ------------------------------------------------------------------ TIER 1

@task("tier1_calibration_drift")
def tier1(ctx) -> dict:
    """Is arm A's 0.0% A-onset recall calibration drift, or a wiring bug?"""
    train_set = ctx.dataset(ctx.expert_train)
    out = {"hypothesis_tested": "calibration drift, not a defect",
           "hypothesis_holds": False,
           "root_cause": (
               "DOUBLE NORMALIZATION. FrameStackDataset already returns float32 in [0,1]; "
               "stage3_train.py divided by 255 again, so the network saw a near-black image "
               "and emitted a constant. Live play normalised correctly, which is why pipe1 "
               "stayed at 53% while offline recall read 0.0%. Fixed in stage3_train.py and "
               "overnight_lib.py; round 1's CHECKPOINT is contaminated (it was fine-tuned on "
               "the corrupted input) but its self-DATA is valid (rollouts used the correct "
               "path)."),
           }

    # The decisive probe: same weights, same frames, input scaled two ways.
    probe_policy, probe_cfg, _ = load_policy(STAGE2_CKPT)
    rows = random_rows(ctx.val_set, 2000, seed=3)
    from torch.utils.data import Subset as _Sub

    from tasdata.bc.train import make_loader as _mk
    both = {}
    for label, scale in (("correct (as given)", 1.0), ("divided by 255 again", 1 / 255.0)):
        ps = []
        with torch.no_grad():
            for obs, _p, _b, _o in _mk(_Sub(ctx.val_set, rows), batch_size=256,
                                       shuffle=False, num_workers=0):
                ps.append(torch.sigmoid(probe_policy(obs * scale))[:, A_INDEX].numpy())
        arr = np.concatenate(ps)
        both[label] = {"mean": float(arr.mean()), "std": float(arr.std()),
                       "min": float(arr.min()), "max": float(arr.max())}
        log(f"  probe p(A) {label}: mean {arr.mean():.4f} std {arr.std():.5f} "
            f"range [{arr.min():.4f}, {arr.max():.4f}]")
    out["double_normalization_probe"] = both

    for label, path in (("stage2_armB", STAGE2_CKPT), ("stage3_round1", ROUND1_CKPT)):
        if not Path(path).exists():
            out[label] = {"missing": str(path)}
            continue
        policy, cfg, stored = load_policy(Path(path))

        # 1. the stored threshold, applied as-is (what produced the 0.0%)
        if isinstance(stored, dict):
            stored_vec = np.array([stored[n] for n in NES_BUTTON_ORDER], dtype=np.float64)
            as_stored = onset_metrics(policy, ctx.val_set, stored_vec)
        else:
            stored_vec, as_stored = None, None

        # 2. recalibrated on a random TRAIN slice against the expert's press rates
        calibration, _ = calibrate(policy, train_set, ctx.target_rates)
        recal_vec = calibration.vector.astype(np.float64)
        recal = onset_metrics(policy, ctx.val_set, recal_vec)

        out[label] = {
            "stored_thresholds": stored if isinstance(stored, dict) else None,
            "recalibrated_thresholds": dict(calibration.thresholds),
            "realized_rate": dict(calibration.realized_rate),
            "target_rate": dict(calibration.target_rate),
            "A_onset_recall_stored": (as_stored or {}).get("onset_recall", {}).get("A"),
            "A_onset_recall_recalibrated": recal["onset_recall"]["A"],
            "prob_at_onset_A": recal["prob_at_onset_A"],
            "prob_elsewhere_A": recal["prob_elsewhere_A"],
            "exact_match": recal["exact_match"],
        }
        log(f"  {label}: A recall stored="
            f"{out[label]['A_onset_recall_stored']} recal="
            f"{out[label]['A_onset_recall_recalibrated']:.3f} "
            f"p(A)@onset median={recal['prob_at_onset_A'].get('median')}")
        emit("tier1_checkpoint", label=label, **out[label])

    probe = out["double_normalization_probe"]
    collapsed = probe["divided by 255 again"]["std"] < 1e-3
    out["verdict"] = (
        "WIRING BUG, not calibration drift: double normalization collapsed p(A) to a "
        f"constant (std {probe['divided by 255 again']['std']:.2e} vs "
        f"{probe['correct (as given)']['std']:.3f} when fed correctly). Recalibration is "
        "still adopted after every round as good practice, but it was not the cause."
        if collapsed else
        "Probe inconclusive -- p(A) did not collapse under double normalization; "
        "investigate further.")
    out["recovered"] = bool(collapsed)
    log(f"  VERDICT: {out['verdict']}")
    return out


# ------------------------------------------------------------------ TIER 2

def rollout_round(ctx, session, policy, cfg, thr, rnd: int, *, episodes: int = 150,
                  accept_frac: float = 0.25, min_credit: float = 0.0,
                  max_frames: int = 500):
    """Score rollouts, accept the best, re-roll those with recording."""
    from tasdata.bc.session_player import play_episode

    rng = np.random.default_rng(1000 + rnd)
    picks = [ctx.traj[i] for i in
             rng.choice(len(ctx.traj), size=min(episodes, len(ctx.traj)), replace=False)]
    scored = []
    unbaselined = 0
    for k, start in enumerate(picks):
        try:
            ep = play_episode(session, policy, start, ctx.vocab, seed=rnd * 10_000 + k,
                              selection="sample", thresholds=thr, head_type=cfg.head_type,
                              stack=cfg.stack, max_frames=max_frames)
        except Exception:
            continue
        # §3: the old score was `gained + 4000*levels - 2000*deaths`, and `gained` is maximised by
        # raising the A-rate -- a three-button script with A at p=0.85 matches or beats every learned
        # checkpoint at pipes 1 and 2. Selecting on progress-from-start therefore selected for the
        # marginal. Obstacle credit net of the best fixed-rate script makes that worth ~nothing.
        from tasdata.bc.script_baseline import rollout_credit
        max_x = ep.max_x_by_level.get(start.label, start.x)
        credit = rollout_credit(max_x, ep.deaths, label=start.label)
        if credit is None:
            # no measured script baseline for this level; dropping it beats scoring it on progress
            unbaselined += 1
            continue
        score = credit + 4.0 * max(0, ep.levels_reached - 1)
        scored.append((score, k, start))
    scores = np.array([s for s, _, _ in scored], dtype=float)
    # `min_progress=120` was 120 *pixels*; script-net credit runs 0-4, so that floor would have
    # rejected every rollout. The floor is now a credit floor.
    if not len(scores):
        return {"acceptance_rate": 0.0, "accepted": 0, "scored": 0, "cutoff": None,
                "score_median": None, "score_p90": None, "unbaselined_dropped": unbaselined,
                "scoring": "script_net_obstacle_credit"}, [], []
    cutoff = max(float(np.quantile(scores, 1 - accept_frac)), min_credit)
    accepted = [t for t in scored if t[0] >= cutoff]

    frames, bytes_ = [], []
    for _s, k, start in accepted:
        rec: list = []
        try:
            play_episode(session, policy, start, ctx.vocab, seed=rnd * 10_000 + k,
                         selection="sample", thresholds=thr, head_type=cfg.head_type,
                         stack=cfg.stack, max_frames=max_frames, record=rec)
        except Exception:
            continue
        if rec:
            frames.append(np.stack([r[0] for r in rec]))
            bytes_.append(np.array([r[1] for r in rec], dtype=np.uint8))
    return {
        "acceptance_rate": len(accepted) / max(len(scored), 1),
        "accepted": len(accepted), "scored": len(scored),
        "cutoff": cutoff, "score_median": float(np.median(scores)),
        "score_p90": float(np.percentile(scores, 90)),
        # §3 bookkeeping: what the new signal refused to score, and under which rule
        "unbaselined_dropped": unbaselined,
        "scoring": "script_net_obstacle_credit (1 - p_script per obstacle cleared)",
    }, frames, bytes_


def write_self_run(out_dir: Path, frames: np.ndarray, bytes_: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(frames)
    actions = np.zeros(n, dtype=np.uint8)
    actions[1:] = bytes_[: n - 1]   # loader convention: action for obs i lives at i+1
    np.save(out_dir / "frames.npy", frames)
    np.save(out_dir / "actions.npy", actions)
    np.save(out_dir / "trace.npy", np.zeros((n, 1), dtype=np.int64))
    np.save(out_dir / "frame_indices.npy", np.arange(n, dtype=np.int64))
    (out_dir / "manifest.json").write_text(json.dumps(
        {"n_frames": n, "synced": True, "category": "self", "measured_route": "self",
         "label": out_dir.name}, indent=2))


def mixed_dataset(ctx, self_dirs, ratio: float, seed: int = 0):
    """Expert:self at the requested ratio, by subsampling the expert side.

    Self-data comes from a policy weaker than the expert, so it is never trained on alone.
    The ratio is expert-frames per self-frame.
    """
    self_runs = [load_run_dir(d) for d in self_dirs]
    expert_set = ctx.dataset(ctx.expert_train)
    self_set = ctx.dataset(self_runs)
    n_self = len(self_set)
    n_expert = min(len(expert_set), int(n_self * ratio))
    e_rows = random_rows(expert_set, n_expert, seed=seed)
    return torch.utils.data.ConcatDataset([Subset(expert_set, e_rows), self_set]), {
        "expert_frames": n_expert, "self_frames": n_self,
        "ratio": f"{ratio:g}:1",
    }


@task("tier2_arm_a_rounds")
def tier2(ctx, recalibrate: bool) -> dict:
    rounds = []
    self_dirs = [ROOT / "data/runs_self/round1"]
    self_dirs = [d for d in self_dirs if d.exists()]
    ckpt = ROUND1_CKPT if ROUND1_CKPT.exists() else STAGE2_CKPT

    # Round 1's checkpoint was fine-tuned through the double-normalization bug, so its
    # weights are not usable. Its self-DATA is fine -- rollouts always went through the
    # correct path -- so it is kept, and every arm restarts from the Stage 2 checkpoint.
    if ROUND1_CKPT.exists():
        policy, cfg, _ = load_policy(ROUND1_CKPT)
        r1 = full_eval(ctx, policy, cfg, "round1_contaminated_for_reference", seeds=60)
        r1.update({"round": 1, "acceptance_rate": 0.26, "self_frames": 19240,
                   "note": "trained through the double-normalization bug; kept only as a "
                           "reference point, not a baseline"})
        rounds.append(r1)
        emit("tier2_round", **r1)

    baseline = full_eval(ctx, *load_policy(STAGE2_CKPT)[:2], "stage2_armB_baseline")
    baseline["round"] = 0
    rounds.append(baseline)
    emit("tier2_round", **baseline)
    log(f"  stage2 baseline: A recall {baseline['offline']['onset_recall']['A']:.3f} "
        f"pipe1 {baseline['live'].get('1-1', {}).get('pipe1_rate')}")

    for ratio in (3.0, 1.0):
        ckpt = STAGE2_CKPT
        dirs = list(self_dirs)
        for rnd in (2, 3):
            tag = f"round{rnd}_ratio{ratio:g}to1"
            policy, cfg, stored = load_policy(ckpt)
            calibration, _ = calibrate(policy, ctx.dataset(ctx.expert_train),
                                       ctx.target_rates)
            thr = calibration.vector.astype(np.float64)

            with FceuxSession(ROM, MOVIE, ctx.frames_needed(
                    p.frame for p in ctx.traj)) as session:
                stats, frames, bytes_ = rollout_round(ctx, session, policy, cfg, thr, rnd)
            log(f"  {tag}: accepted {stats['accepted']}/{stats['scored']} "
                f"({stats['acceptance_rate'] * 100:.0f}%)")
            if not frames:
                emit("tier2_round_empty", tag=tag, **stats)
                continue
            d = ROOT / f"data/runs_self/{tag}"
            write_self_run(d, np.concatenate(frames), np.concatenate(bytes_))
            dirs = dirs + [d]

            train_set, mix = mixed_dataset(ctx, dirs, ratio, seed=rnd)
            log(f"  {tag}: training on {mix}")
            policy = train_policy(policy, train_set, steps=800, lr=1e-4,
                                  onset_weight=10.0, seed=rnd, log=log)
            new_ckpt = save_policy(CKPTS / f"{tag}.pt", policy, cfg,
                                   {n: 0.5 for n in NES_BUTTON_ORDER}, round=rnd)
            row = full_eval(ctx, policy, cfg, tag)
            row.update({"round": rnd, "ratio": mix["ratio"], "mix": mix, **stats})
            rounds.append(row)
            emit("tier2_round", **row)
            log(f"  {tag}: A recall {row['offline']['onset_recall']['A']:.3f} "
                f"pipe1 {row['live'].get('1-1', {}).get('pipe1_rate')}")
            ckpt = new_ckpt
    return {"rounds": rounds, "recalibrated_every_round": recalibrate}


# ------------------------------------------------------------------ TIER 3

@task("tier3_oracle_margin")
def tier3(ctx) -> dict:
    """Final oracle attempt: calibrate a decision margin so the jump rate matches 6.0%."""
    from tasdata.bc.oracle import decide_with_policy
    from tasdata.bc.statelib import grounded_backward_mask
    from tasdata.ram import PLAYER_STATE_NORMAL, column

    split = json.loads((ROOT / "data/split.json").read_text())["splits"]
    run_dir = next(ROOT / "data/runs" / n for n in split["test"]
                   if json.loads((ROOT / "data/runs" / n / "manifest.json").read_text())
                   .get("category") == "warpless")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    movie = ROOT / str(manifest["movie"]).replace(str(ROOT) + "/", "")
    run = load_run_dir(run_dir)

    actions = run.actions.astype(np.int64)
    trace = np.asarray(run.trace)
    n = min(len(trace) - 1, len(actions) - 1)
    truth = (actions[1:n + 1] & 0x01) > 0
    prev = (actions[:n] & 0x01) > 0
    ok = ((column(trace, "pregame")[:n] == 1)
          & (column(trace, "player_state")[:n] == PLAYER_STATE_NORMAL)
          & (column(trace, "world")[:n] >= 1) & (column(trace, "world")[:n] <= 8)
          & (column(trace, "stage")[:n] >= 1) & (column(trace, "stage")[:n] <= 4)
          & (column(trace, "x_position")[:n] > 0) & (column(trace, "time")[:n] > 0)
          & grounded_backward_mask(run)[:n])
    onsets = np.flatnonzero(truth & ~prev & ok)
    rng = np.random.default_rng(0)
    uni = rng.choice(np.flatnonzero(ok), size=min(1000, int(ok.sum())), replace=False)
    ons = rng.choice(onsets, size=min(500, onsets.size), replace=False)
    frames = sorted(set(uni.tolist()) | set(ons.tolist()))
    ordinal = {f: i for i, f in enumerate(frames)}

    policy, cfg, stored = load_policy(STAGE2_CKPT)
    thr = np.array([stored[n_] for n_ in NES_BUTTON_ORDER], dtype=np.float64)

    margins = {}
    with FceuxSession(ROM, movie, frames) as session:
        for f in frames:
            d = decide_with_policy(session, ordinal[f], f, policy, thr,
                                   horizon=60, jump_hold=20, stack=cfg.stack,
                                   n_rollouts=1, seed=1234)
            margins[f] = d.margin
    m_uni = np.array([margins[int(f)] for f in uni], dtype=float)
    expert_rate = float(truth[uni].mean())

    sweep = []
    for M in [0, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]:
        jump_uni = m_uni > M
        rate = float(jump_uni.mean())
        agree_all = float((jump_uni == truth[uni]).mean())
        agree_ons = (
            float(((np.array([margins[int(f)] for f in ons], dtype=float) > M)
                   == truth[ons]).mean()) if ons.size else 0.0
        )
        sweep.append({"margin": M, "jump_rate": rate, "agreement_overall": agree_all,
                      "agreement_at_onsets": agree_ons})
        log(f"    M={M:4d} jump={rate * 100:5.1f}% (expert {expert_rate * 100:.1f}%) "
            f"overall={agree_all * 100:5.1f}% onset={agree_ons * 100:5.1f}%")
    best = min(sweep, key=lambda r: abs(r["jump_rate"] - expert_rate))
    verdict = ("ORACLE DEAD -- onset agreement below 70% at a matched jump rate; "
               "third failed teacher, do not try again"
               if best["agreement_at_onsets"] < 0.70 else
               "PASSES -- onset agreement at or above 70% at a matched jump rate")
    log(f"  best M={best['margin']} -> {verdict}")
    return {"expert_rate": expert_rate, "sweep": sweep, "matched": best,
            "n_onsets": int(ons.size), "verdict": verdict,
            "passes": bool(best["agreement_at_onsets"] >= 0.70)}


# ------------------------------------------------------------------ TIER 4

@task("tier4_glitchless_vs_glitchy")
def tier4(ctx) -> dict:
    """Matched-frame comparison of glitchless against glitch-heavy data."""
    import glob
    import os

    member = {n: b for b, ns in ctx.split.items() for n in ns}
    glitchless, glitchy = [], []
    for m in sorted(glob.glob(str(ROOT / "data/runs/*/manifest.json"))):
        name = os.path.basename(os.path.dirname(m))
        j = json.loads(Path(m).read_text())
        cat, route = str(j.get("category", "")), str(j.get("measured_route", ""))
        if not (ROOT / "data/runs" / name / "frames.npy").exists():
            continue
        if "glitchless" in cat:
            # Never train on a val run: it is the yardstick for both arms.
            if member.get(name) != "val":
                glitchless.append((name, route or cat, int(j.get("n_frames", 0))))
        elif route == "warps" and member.get(name) == "train":
            glitchy.append((name, route, int(j.get("n_frames", 0))))

    caveat = (
        "The corpus contains NO warpless-glitchless runs -- the only glitchless runs are "
        "'warps-glitchless', and one of the two is in the val split and cannot be trained "
        "on. The comparison below is therefore glitchless-WARPS against a matched-frame "
        "subsample of glitch-heavy WARPS runs, which controls the route but rests the "
        f"glitchless arm on {len(glitchless)} run(s). Run-level variance is not controlled."
    )
    if not glitchless or not glitchy:
        return {"runnable": False, "caveat": caveat,
                "glitchless": glitchless, "glitchy": glitchy}

    gl_runs = [load_run_dir(ROOT / "data/runs" / n) for n, _, _ in glitchless]
    gh_runs = [load_run_dir(ROOT / "data/runs" / n) for n, _, _ in glitchy]
    gl_set = ctx.dataset(gl_runs)
    gh_full = ctx.dataset(gh_runs)
    budget = len(gl_set)
    log(f"  glitchless {len(glitchless)} run(s), {budget:,} frames; "
        f"glitch-heavy pool {len(glitchy)} runs, {len(gh_full):,} frames")

    results = []
    for seed in (0, 1, 2):
        for arm, ds in (("glitchless", gl_set),
                        ("glitch_heavy", Subset(gh_full,
                                                random_rows(gh_full, budget, seed=seed)))):
            policy = fresh_policy(ctx.cfg, seed=seed)
            policy = train_policy(policy, ds, steps=2000, lr=3e-4, onset_weight=10.0,
                                  seed=seed, log=log)
            save_policy(CKPTS / f"t4_{arm}_seed{seed}.pt", policy, ctx.cfg,
                        {n: 0.5 for n in NES_BUTTON_ORDER})
            row = full_eval(ctx, policy, ctx.cfg, f"{arm}_seed{seed}", seeds=EVAL_SEEDS)
            row.update({"arm": arm, "seed": seed, "frames": budget})
            results.append(row)
            emit("tier4_arm", **row)
            log(f"  {arm} seed{seed}: A recall "
                f"{row['offline']['onset_recall']['A']:.3f} "
                f"pipe1 {row['live'].get('1-1', {}).get('pipe1_rate')}")
    return {"runnable": True, "caveat": caveat, "budget_frames": budget,
            "glitchless_runs": glitchless, "glitchy_pool": glitchy, "results": results}


# ------------------------------------------------------------------ TIER 5

@task("tier5_scaling_curve")
def tier5(ctx) -> dict:
    full = ctx.dataset(ctx.expert_train)
    out = []
    for frac in (0.10, 0.25, 0.50, 1.00):
        rows = random_rows(full, int(len(full) * frac), seed=7)
        ds = Subset(full, rows)
        policy = fresh_policy(ctx.cfg, seed=0)
        policy = train_policy(policy, ds, steps=2000, lr=3e-4, onset_weight=10.0,
                              seed=0, log=log)
        row = full_eval(ctx, policy, ctx.cfg, f"scale_{int(frac * 100)}pct", seeds=120)
        row.update({"fraction": frac, "frames": len(rows)})
        out.append(row)
        emit("tier5_point", **row)
        log(f"  {frac:.0%} ({len(rows):,} frames): A recall "
            f"{row['offline']['onset_recall']['A']:.3f} "
            f"pipe1 {row['live'].get('1-1', {}).get('pipe1_rate')}")
    return {"points": out}


@task("tier6_two_one_wall")
def tier6(ctx) -> dict:
    """What sits at x=530 in 2-1 -- does it block, or kill?"""
    from tasdata.bc.session_player import play_episode
    from tasdata.buttons import NES_BUTTON_BITS
    from tasdata.ram import read_smb

    start = next(p for p in ctx.points if p.kind == "level_start" and p.label == "2-1")
    policy, cfg, stored = load_policy(STAGE2_CKPT)
    thr = np.array([stored[n] for n in NES_BUTTON_ORDER], dtype=np.float64)
    run_right = NES_BUTTON_BITS["Right"] | NES_BUTTON_BITS["B"]

    out: dict = {}
    with FceuxSession(ROM, MOVIE, ctx.frames_needed()) as session:
        # 1. run right and hold: where does Mario stop, and is he alive?
        obs = session.reset(start.frame)
        trail = []
        for i in range(900):
            obs = session.step(run_right)
            st = read_smb(obs.ram, obs.framecount)
            trail.append((int(st.x_position), int(st.y_position), int(st.player_state)))
        xs = [t[0] for t in trail]
        out["run_right_max_x"] = max(xs)
        out["run_right_final_state"] = trail[-1][2]
        out["died_running_right"] = any(t[2] in (0x06, 0x0B) for t in trail)

        # 2. run right AND jump continuously: does that get past?
        obs = session.reset(start.frame)
        xs2 = []
        for i in range(900):
            byte = run_right | (NES_BUTTON_BITS["A"] if (i % 40) < 18 else 0)
            obs = session.step(byte)
            st = read_smb(obs.ram, obs.framecount)
            xs2.append(int(st.x_position))
        out["run_and_jump_max_x"] = max(xs2)

        # 3. capture what the policy's episodes actually do
        eps = [play_episode(session, policy, start, ctx.vocab, seed=s, selection="sample",
                            thresholds=thr, head_type=cfg.head_type, stack=cfg.stack)
               for s in range(30)]
        out["policy_x_median"] = float(np.median([e.furthest_x for e in eps]))
        out["policy_deaths_mean"] = float(np.mean([e.deaths for e in eps]))
        out["policy_ended"] = {}
        for e in eps:
            out["policy_ended"][e.ended] = out["policy_ended"].get(e.ended, 0) + 1

    # 4. what does the expert do around x=530 in 2-1?
    from tasdata.ram import column
    warp_runs = [r for r in ctx.expert_train
                 if r.manifest.get("measured_route") == "warps"]
    ex = {}
    for r in (warp_runs[:3] or ctx.expert_train[:1]):
        tr = np.asarray(r.trace)
        w, s_, x = column(tr, "world"), column(tr, "stage"), column(tr, "x_position")
        m = (w == 2) & (s_ == 1) & (x > 480) & (x < 600)
        idx = np.flatnonzero(m)
        if idx.size:
            a = np.asarray(r.actions, dtype=np.uint8)[idx]
            ex[r.name] = {"frames_in_window": int(idx.size),
                          "A_press_rate": float(np.mean((a & 0x01) > 0)),
                          "y_min": int(column(tr, "y_position")[idx].min()),
                          "y_max": int(column(tr, "y_position")[idx].max())}
    out["expert_at_530"] = ex
    out["interpretation"] = (
        "blocks (no deaths running right)" if not out["died_running_right"]
        else "kills (running right dies)")
    log(f"  2-1 wall: run-right max x={out['run_right_max_x']} "
        f"jump max x={out['run_and_jump_max_x']} -> {out['interpretation']}")
    return out


# ------------------------------------------------------------------ plots

@task("tier7_plots")
def tier7(ctx) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in JSONL.read_text().splitlines() if x.strip()]
    made = []

    def bars(names, ks, ns, title, path, ylabel="pipe 1 cleared"):
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = np.arange(len(names))
        rates = [k / n if n else 0 for k, n in zip(ks, ns)]
        los = [r - wilson(k, n)[0] for r, k, n in zip(rates, ks, ns)]
        his = [wilson(k, n)[1] - r for r, k, n in zip(rates, ks, ns)]
        ax.bar(xs, rates, color="#4878a8")
        ax.errorbar(xs, rates, yerr=[los, his], fmt="none", ecolor="black", capsize=4)
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(str(path))

    # Stage 2 progression (locked numbers + this run's rounds)
    names = ["categorical\n(stage 2)", "bernoulli only\n(arm A)", "bernoulli+reweight\n(arm B)"]
    bars(names, [0, 59, 119], [200, 200, 200],
         "Stage 2 progression: pipe 1 cleared, n=200 (Wilson 95%)",
         PLOTS / "stage2_progression.png")

    # Arm A rounds against the Stage 2 baseline
    r2 = [r for r in rows if r["kind"] == "tier2_round"]
    if r2:
        nm = ["stage2 armB"] + [r.get("tag", f"r{r.get('round')}") for r in r2]
        ks = [119] + [int(r["live"].get("1-1", {}).get("pipe1_k", 0)) for r in r2]
        ns = [200] + [int(r["live"].get("1-1", {}).get("n", 1)) for r in r2]
        bars(nm, ks, ns, "Stage 3 arm A rounds vs Stage 2 baseline (1-1)",
             PLOTS / "stage3_rounds.png")

    # Scaling curve
    t5 = [r for r in rows if r["kind"] == "tier5_point"]
    if t5:
        t5.sort(key=lambda r: r["fraction"])
        fig, ax1 = plt.subplots(figsize=(7, 4))
        fr = [r["fraction"] * 100 for r in t5]
        rec = [r["offline"]["onset_recall"]["A"] * 100 for r in t5]
        pipe = [r["live"].get("1-1", {}).get("pipe1_rate", 0) * 100 for r in t5]
        ax1.plot(fr, rec, "o-", color="#4878a8", label="A-onset recall")
        ax1.set_xlabel("% of training frames")
        ax1.set_ylabel("A-onset recall (%)", color="#4878a8")
        ax2 = ax1.twinx()
        ax2.plot(fr, pipe, "s--", color="#d1603d", label="pipe 1 %")
        ax2.set_ylabel("pipe 1 cleared (%)", color="#d1603d")
        ax1.set_title("Data scaling: does more data help?", fontsize=10)
        fig.tight_layout()
        fig.savefig(PLOTS / "scaling_curve.png", dpi=140)
        plt.close(fig)
        made.append(str(PLOTS / "scaling_curve.png"))

    # Glitchless vs glitch-heavy
    t4 = [r for r in rows if r["kind"] == "tier4_arm"]
    if t4:
        fig, ax = plt.subplots(figsize=(6, 4))
        for arm, colour in (("glitchless", "#4878a8"), ("glitch_heavy", "#d1603d")):
            vals = [r["live"].get("1-1", {}).get("pipe1_rate", 0) * 100
                    for r in t4 if r["arm"] == arm]
            if vals:
                ax.scatter([arm] * len(vals), vals, color=colour, s=60)
                ax.hlines(np.mean(vals), -0.3, 1.3, colors=colour, linestyles=":", alpha=0.5)
        ax.set_ylabel("pipe 1 cleared (%)")
        ax.set_title("Glitchless vs glitch-heavy, matched frames", fontsize=10)
        fig.tight_layout()
        fig.savefig(PLOTS / "glitchless_vs_glitchy.png", dpi=140)
        plt.close(fig)
        made.append(str(PLOTS / "glitchless_vs_glitchy.png"))

    # Hold-length distribution, model vs expert
    from tasdata.buttons import NES_BUTTON_BITS
    holds = []
    for r in ctx.expert_train[:4]:
        a = np.asarray(r.actions, dtype=np.uint8)
        held = (a & NES_BUTTON_BITS["A"]) > 0
        run = 0
        for v in held:
            if v:
                run += 1
            elif run:
                holds.append(run)
                run = 0
    live_rows = [r for r in rows if r["kind"] in ("tier2_round", "tier4_arm", "tier5_point")]
    model_p90 = [r["live"].get("1-1", {}).get("hold_A_p90", 0) for r in live_rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    if holds:
        ax.hist(np.clip(holds, 0, 60), bins=60, color="#4878a8", alpha=0.75,
                density=True, label=f"expert A-holds (n={len(holds):,})")
    if model_p90:
        ax.axvline(float(np.median(model_p90)), color="#d1603d", ls="--",
                   label=f"model A-hold p90 (median {np.median(model_p90):.0f})")
    ax.set_xlabel("A-hold length (frames)")
    ax.set_ylabel("density")
    ax.set_title("Taps, not holds: expert A-hold distribution", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "hold_lengths.png", dpi=140)
    plt.close(fig)
    made.append(str(PLOTS / "hold_lengths.png"))
    return {"plots": made}


# ------------------------------------------------------------------ summary

def write_summary() -> None:
    try:
        rows = [json.loads(x) for x in JSONL.read_text().splitlines() if x.strip()]
    except FileNotFoundError:
        return
    done = [r for r in rows if r["kind"] == "task_done"]
    failed = [r for r in rows if r["kind"] == "task_failed"]
    L = [
        "# Overnight run", "",
        f"Started {datetime.fromtimestamp(_started, timezone.utc).isoformat(timespec='seconds')}, "
        f"running {(time.time() - _started) / 60:.0f} min. "
        f"{len(done)} tasks finished, {len(failed)} failed.",
        "", "Regenerated every 2 minutes. Raw stream: `data/overnight.jsonl`.", "",
    ]

    t1 = next((r for r in rows if r["kind"] == "task_done"
               and r["task"] == "tier1_calibration_drift"), None)
    if t1 and t1.get("result"):
        res = t1["result"]
        L += ["## Tier 1 — the 0.0% A-onset recall", "", f"**{res.get('verdict')}**", ""]
        L += ["| checkpoint | recall, stored threshold | recall, recalibrated | p(A) at onsets (median) |",
              "| --- | --- | --- | --- |"]
        for k in ("stage2_armB", "stage3_round1"):
            d = res.get(k) or {}
            if "missing" in d:
                continue
            st = d.get("A_onset_recall_stored")
            L.append(f"| {k} | {'n/a' if st is None else f'{st * 100:.1f}%'} | "
                     f"{(d.get('A_onset_recall_recalibrated') or 0) * 100:.1f}% | "
                     f"{(d.get('prob_at_onset_A') or {}).get('median', 0):.3f} |")
        L.append("")

    r2 = [r for r in rows if r["kind"] == "tier2_round"]
    if r2:
        L += ["## Tier 2 — arm A rounds", "",
              "| tag | ratio | accept % | A-onset recall | pipe1 1-1 (95% CI) | x_med 1-1 | x_med 2-1 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in r2:
            one = r["live"].get("1-1", {})
            two = r["live"].get("2-1", {})
            ci = one.get("pipe1_ci", [0, 0])
            L.append(
                f"| {r.get('tag')} | {r.get('ratio', '-')} | "
                f"{(r.get('acceptance_rate') or 0) * 100:.0f} | "
                f"{r['offline']['onset_recall']['A'] * 100:.1f}% | "
                f"{one.get('pipe1_rate', 0) * 100:.1f}% [{ci[0] * 100:.1f}, {ci[1] * 100:.1f}] | "
                f"{one.get('x_median', 0):.0f} | {two.get('x_median', 0):.0f} |")
        L.append("")

    t3 = next((r for r in rows if r["kind"] == "task_done"
               and r["task"] == "tier3_oracle_margin"), None)
    if t3 and t3.get("result"):
        res = t3["result"]
        L += ["## Tier 3 — oracle, margin-calibrated", "", f"**{res.get('verdict')}**", "",
              f"Expert jump rate {res.get('expert_rate', 0) * 100:.1f}%. "
              f"Matched at margin M={res.get('matched', {}).get('margin')}: "
              f"onset agreement "
              f"{res.get('matched', {}).get('agreement_at_onsets', 0) * 100:.1f}%.", "",
              "| M | jump rate | agree overall | agree at A-onsets |", "| --- | --- | --- | --- |"]
        for s in res.get("sweep", []):
            L.append(f"| {s['margin']} | {s['jump_rate'] * 100:.1f}% | "
                     f"{s['agreement_overall'] * 100:.1f}% | "
                     f"{s['agreement_at_onsets'] * 100:.1f}% |")
        L.append("")

    t4 = next((r for r in rows if r["kind"] == "task_done"
               and r["task"] == "tier4_glitchless_vs_glitchy"), None)
    if t4 and t4.get("result"):
        res = t4["result"]
        L += ["## Tier 4 — glitchless vs glitch-heavy", "", f"> {res.get('caveat')}", ""]
        arms = [r for r in rows if r["kind"] == "tier4_arm"]
        if arms:
            L += ["| arm | seed | A-onset recall | pipe1 1-1 | x_med 2-1 |",
                  "| --- | --- | --- | --- | --- |"]
            for r in arms:
                L.append(f"| {r['arm']} | {r['seed']} | "
                         f"{r['offline']['onset_recall']['A'] * 100:.1f}% | "
                         f"{r['live'].get('1-1', {}).get('pipe1_rate', 0) * 100:.1f}% | "
                         f"{r['live'].get('2-1', {}).get('x_median', 0):.0f} |")
        L.append("")

    t5 = [r for r in rows if r["kind"] == "tier5_point"]
    if t5:
        L += ["## Tier 5 — data scaling", "",
              "| fraction | frames | A-onset recall | pipe1 1-1 |", "| --- | --- | --- | --- |"]
        for r in sorted(t5, key=lambda r: r["fraction"]):
            L.append(f"| {r['fraction']:.0%} | {r['frames']:,} | "
                     f"{r['offline']['onset_recall']['A'] * 100:.1f}% | "
                     f"{r['live'].get('1-1', {}).get('pipe1_rate', 0) * 100:.1f}% |")
        L.append("")

    t6 = next((r for r in rows if r["kind"] == "task_done"
               and r["task"] == "tier6_two_one_wall"), None)
    if t6 and t6.get("result"):
        res = t6["result"]
        L += ["## Tier 6 — the 2-1 wall", "",
              f"Running right and holding reaches x={res.get('run_right_max_x')}; "
              f"running and jumping reaches x={res.get('run_and_jump_max_x')}. "
              f"Died running right: {res.get('died_running_right')}. "
              f"**Interpretation: {res.get('interpretation')}**", ""]

    if failed:
        L += ["## Failures", ""]
        for f in failed:
            L.append(f"- **{f['task']}** after {f['seconds']}s — `{f['error']}`")
        L.append("")

    L += ["## Task log", "", "| task | status | minutes |", "| --- | --- | --- |"]
    for r in rows:
        if r["kind"] in ("task_done", "task_failed"):
            L.append(f"| {r['task']} | {'ok' if r['kind'] == 'task_done' else 'FAILED'} | "
                     f"{r['seconds'] / 60:.1f} |")
    SUMMARY.write_text("\n".join(L) + "\n")


def summary_loop(stop: threading.Event) -> None:
    while not stop.wait(120):
        try:
            write_summary()
        except Exception:
            pass


# ------------------------------------------------------------------ main

def main() -> None:
    JSONL.parent.mkdir(parents=True, exist_ok=True)
    CKPTS.mkdir(parents=True, exist_ok=True)
    emit("run_start", pid=None)
    stop = threading.Event()
    threading.Thread(target=summary_loop, args=(stop,), daemon=True).start()

    try:
        ctx = Ctx()
        log(f"context ready: {len(ctx.expert_train)} train runs, "
            f"{len(ctx.traj)} trajectory starts, target A rate "
            f"{ctx.target_rates['A']:.4f}")
    except Exception as exc:
        emit("fatal", error=f"{type(exc).__name__}: {exc}",
             traceback=traceback.format_exc()[-3000:])
        write_summary()
        return

    t1 = tier1(ctx)
    tier2(ctx, recalibrate=True)
    tier3(ctx)
    tier4(ctx)
    tier5(ctx)
    tier6(ctx)
    tier7(ctx)

    stop.set()
    write_summary()
    emit("run_end", minutes=round((time.time() - _started) / 60, 1))
    log("ALL DONE")


if __name__ == "__main__":
    main()
