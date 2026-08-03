"""Smoke test and overnight sweep.

The smoke test is a hard gate: 1,000 frames, 50 steps, and it must show loss
decreasing, a checkpoint that saves *and reloads*, and one complete live-play
episode. If any of that fails the long run does not start.

The sweep is several short runs rather than one long one, so a night that ends early
still yields a curve. Every result is appended to a JSONL file the moment it is
known, and one config crashing is logged and stepped over.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..dataset import LoadedRun
from .baselines import (
    ConstantPolicy,
    MarginalPolicy,
    evaluate_trivial_baselines,
    token_for_buttons,
)
from .data import FrameStackDataset, load_split
from .live import LivePlayer
from .model import pick_device
from .tokens import DEFAULT_RARE_THRESHOLD, ActionVocab, build_vocab
from .train import (
    TrainConfig,
    load_checkpoint,
    make_loader,
    save_checkpoint,
    train,
    validate,
)


def run_eval_isolated(
    checkpoint: Path | str,
    vocab_path: Path | str,
    rom_path: Path | str,
    *,
    selection: str,
    temperature: float = 1.0,
    sticky_p: float = 0.25,
    seeds: int = 20,
    levels: tuple[str, ...] = ("1-1",),
    expert_movie: Path | str | None = None,
    stall_frames: int = 300,
    max_frames: int = 3000,
    expert_bytes: str | None = None,
    workers: int = 1,
    timeout: float = 7200.0,
) -> dict:
    """Evaluate a checkpoint in a separate CPU-only process.

    Using MPS in this process would make every FCEUX child fall back to software
    OpenGL and crash, so evaluation must not share a process with training. See
    :mod:`tasdata.bc.eval_worker`.
    """
    cmd = [
        sys.executable, "-m", "tasdata.bc.eval_worker",
        "--checkpoint", str(checkpoint),
        "--vocab", str(vocab_path),
        "--rom", str(rom_path),
        "--selection", selection,
        "--temperature", str(temperature),
        "--sticky-p", str(sticky_p),
        "--seeds", str(seeds),
        "--stall-frames", str(stall_frames),
        "--max-frames", str(max_frames),
        "--levels", *levels,
    ]
    if expert_movie:
        cmd += ["--expert-movie", str(expert_movie)]
    if expert_bytes:
        cmd += ["--expert-bytes", str(expert_bytes)]
    if workers > 1:
        cmd += ["--workers", str(workers)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not proc.stdout.strip():
        return {
            "error": f"eval worker exited {proc.returncode}: "
            f"{(proc.stderr or '')[-300:]}"
        }
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"error": f"eval worker produced unparseable output: {exc}"}


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:5.1f}%"


def append_jsonl(path: Path | str, record: dict) -> None:
    """Append one record and flush, so the file is readable while the sweep runs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def environment(device: str | None = None) -> dict:
    """Provenance. Deliberately does not probe MPS -- that would poison FCEUX."""
    return {
        "torch": torch.__version__,
        "device": device or "unspecified",
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


# --------------------------------------------------------------------------- #
# Shared setup
# --------------------------------------------------------------------------- #

@dataclass
class Corpus:
    """Datasets, vocabulary and label statistics for a sweep."""

    vocab: ActionVocab
    vocab_path: Path
    train_runs: list[LoadedRun]
    val_runs: list[LoadedRun]
    train_counts: np.ndarray
    val_labels: np.ndarray
    #: Slice held out of TRAIN purely to calibrate thresholds. Val must not serve as
    #: both the calibration reference and the evaluation set, and val is stylistically
    #: unrepresentative anyway (it holds Right where train taps it).
    calib_runs: list[LoadedRun] = field(default_factory=list)

    def dataset(
        self, which: str, *, stack: int, blind: bool, prev_actions: int = 0
    ) -> FrameStackDataset:
        runs = self.train_runs if which == "train" else self.val_runs
        return FrameStackDataset(
            runs, self.vocab, stack=stack, blind=blind, prev_actions=prev_actions
        )

    def dataset_bernoulli(self, which: str, *, stack: int) -> FrameStackDataset:
        """Raw per-button labels plus onset masks; no vocabulary folding."""
        runs = {
            "train": self.train_runs,
            "val": self.val_runs,
            "calib": self.calib_runs or self.train_runs,
        }[which]
        return FrameStackDataset(
            runs, self.vocab, stack=stack, blind=False, prev_actions=0,
            label_mode="buttons",
        )


def prepare_corpus(
    runs_root: Path | str,
    split_path: Path | str,
    vocab_path: Path | str,
    *,
    threshold: int = DEFAULT_RARE_THRESHOLD,
    rebuild_vocab: bool = False,
    val_label_cap: int = 200_000,
    n_calib_runs: int = 0,
    log=print,
) -> Corpus:
    """Load the split, build or load the vocabulary, and gather label statistics."""
    split = load_split(runs_root, split_path)
    train_runs, val_runs = split["train"], split["val"]
    if not train_runs or not val_runs:
        raise ValueError("split has an empty train or val bucket")

    # Carve a calibration slice off TRAIN, deterministically (smallest runs first so
    # the training set loses as little as possible).
    calib_runs: list[LoadedRun] = []
    if n_calib_runs > 0:
        ordered = sorted(train_runs, key=lambda r: (len(r.actions), r.name))
        calib_runs = ordered[:n_calib_runs]
        keep = {r.name for r in calib_runs}
        train_runs = [r for r in train_runs if r.name not in keep]
        log(
            f"  calibration slice: {[r.name for r in calib_runs]} "
            f"({sum(len(r.actions) for r in calib_runs):,} frames) held out of train"
        )

    vocab_path = Path(vocab_path)
    if vocab_path.exists() and not rebuild_vocab:
        vocab = ActionVocab.load(vocab_path)
        log(f"  vocabulary loaded from {vocab_path} ({vocab.size} tokens)")
    else:
        vocab = build_vocab([r.actions for r in train_runs], threshold=threshold)
        vocab.save(vocab_path)
        log(f"  vocabulary built from {len(train_runs)} train runs -> {vocab_path}")

    train_counts = np.zeros(vocab.size, dtype=np.int64)
    for run in train_runs:
        tokens = vocab.encode(run.actions)
        train_counts += np.bincount(tokens, minlength=vocab.size)

    val_tokens = np.concatenate([vocab.encode(r.actions) for r in val_runs])
    if val_tokens.size > val_label_cap:
        rng = np.random.default_rng(0)
        val_tokens = val_tokens[
            np.sort(rng.choice(val_tokens.size, val_label_cap, replace=False))
        ]

    return Corpus(
        vocab=vocab,
        vocab_path=vocab_path,
        train_runs=train_runs,
        val_runs=val_runs,
        calib_runs=calib_runs,
        train_counts=train_counts,
        val_labels=val_tokens,
    )


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #

class SmokeTestFailure(RuntimeError):
    """The smoke test failed; the long run must not start."""


def smoke_test(
    corpus: Corpus,
    rom_path: Path | str,
    *,
    out_dir: Path | str,
    frames: int = 1000,
    steps: int = 50,
    live_frames: int = 600,
    device: torch.device | str | None = None,
    log=print,
) -> dict:
    """Prove the pipeline works end to end before committing to a long run.

    Checks, in order: loss decreases over ``steps``; a checkpoint saves and reloads
    to identical predictions; one live episode completes and returns metrics.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device) if device is not None else pick_device()
    report: dict = {"kind": "smoke", "device": str(device), "checks": {}}

    config = TrainConfig(
        name="smoke",
        d_model=32,
        n_layers=1,
        n_heads=2,
        cnn_channels=(8, 16, 16),
        batch_size=32,
        steps=steps,
        warmup=5,
        eval_every=steps,
        val_frames=2000,
        num_workers=0,
    )

    # -- 1. tiny slice of data ------------------------------------------------ #
    full_train = corpus.dataset(
        "train", stack=config.stack, blind=False, prev_actions=config.n_prev_actions
    )
    if len(full_train) < frames:
        raise SmokeTestFailure(
            f"train split has only {len(full_train)} frames, need {frames}"
        )
    rng = np.random.default_rng(0)
    picks = np.sort(rng.choice(len(full_train), size=frames, replace=False))
    from torch.utils.data import Subset

    tiny = Subset(full_train, picks.tolist())
    report["checks"]["data"] = {
        "ok": True,
        "train_frames_available": len(full_train),
        "frames_used": frames,
    }
    log(f"  [1/4] data: {frames} of {len(full_train):,} train frames, memory-mapped")

    # -- 2. loss decreases ---------------------------------------------------- #
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from .model import build_policy

    torch.manual_seed(0)
    policy = build_policy(config.policy_config(corpus.vocab.size)).to(device)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(tiny, batch_size=config.batch_size, shuffle=True, drop_last=True)
    losses: list[float] = []
    policy.train()
    while len(losses) < steps:
        for batch in loader:
            if len(losses) >= steps:
                break
            batch_frames, prev, target = batch
            batch_frames = batch_frames.to(device)
            target = target.to(device)
            loss = loss_fn(policy(batch_frames, prev), target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
    first = float(np.mean(losses[:10]))
    last = float(np.mean(losses[-10:]))
    decreased = last < first
    report["checks"]["loss_decreases"] = {
        "ok": decreased,
        "first10_mean": round(first, 4),
        "last10_mean": round(last, 4),
        "steps": len(losses),
        "parameters": policy.n_parameters,
    }
    log(
        f"  [2/4] loss: {first:.4f} -> {last:.4f} over {len(losses)} steps "
        f"({'decreasing' if decreased else 'NOT DECREASING'})"
    )
    if not decreased:
        raise SmokeTestFailure(
            f"loss did not decrease over {steps} steps "
            f"(first-10 mean {first:.4f}, last-10 mean {last:.4f}). "
            "Training is broken; not starting the long run."
        )

    # -- 3. checkpoint round trip -------------------------------------------- #
    ckpt = save_checkpoint(
        out_dir / "smoke.pt", policy, config, step=len(losses), vocab_path=corpus.vocab_path
    )
    try:
        reloaded, blob = load_checkpoint(ckpt, device=device)
    except Exception as exc:
        raise SmokeTestFailure(f"checkpoint reload failed: {type(exc).__name__}: {exc}") from exc
    probe = torch.rand(4, config.stack, 84, 84, device=device)
    policy.eval()
    with torch.no_grad():
        before = policy(probe).cpu().numpy()
        after = reloaded(probe).cpu().numpy()
    identical = bool(np.allclose(before, after, atol=1e-5))
    report["checks"]["checkpoint_roundtrip"] = {
        "ok": identical,
        "path": str(ckpt),
        "bytes": ckpt.stat().st_size,
        "max_abs_diff": float(np.abs(before - after).max()),
        "step_recorded": blob.get("step"),
    }
    log(
        f"  [3/4] checkpoint: saved {ckpt.stat().st_size / 1e6:.2f} MB, reloaded, "
        f"max|Δlogit| {np.abs(before - after).max():.2e}"
    )
    if not identical:
        raise SmokeTestFailure(
            "reloaded checkpoint does not reproduce the original logits "
            f"(max abs diff {np.abs(before - after).max():.3e}). Not starting the long run."
        )

    # -- 4. one live episode, through the real (isolated) evaluation path ------ #
    live = run_eval_isolated(
        ckpt,
        corpus.vocab_path,
        rom_path,
        selection="greedy",
        seeds=1,
        levels=("1-1",),
        max_frames=live_frames,
        stall_frames=0,  # do not early-stop the gate episode
    )
    if "error" in live:
        raise SmokeTestFailure(f"live play failed: {live['error']}")
    gained_control = live.get("n_episodes", 0) > 0 and (
        live.get("levels_reached", {}).get("max", 0) > 0
    )
    report["checks"]["live_play"] = {
        "ok": bool(gained_control),
        "n_episodes": live.get("n_episodes"),
        "furthest_x": live.get("furthest_x", {}).get("max"),
        "levels_reached": live.get("levels_reached", {}).get("max"),
        "flakes_retried": live.get("retried_flakes"),
        "ended": live.get("ended_histogram"),
    }
    log(
        f"  [4/4] live play (isolated CPU process): "
        f"{live.get('n_episodes')} episode, x={live.get('furthest_x', {}).get('max')}, "
        f"levels={live.get('levels_reached', {}).get('max')}, "
        f"flakes retried={live.get('retried_flakes')}"
    )
    if not gained_control:
        raise SmokeTestFailure(
            "live play completed but the policy never gained control of the game "
            f"({live.get('ended_histogram')}). The evaluation harness is not measuring "
            "anything; not starting the long run."
        )

    report["ok"] = True
    return report


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def default_configs(
    steps: int, eval_every: int, *, num_workers: int = 2
) -> list[TrainConfig]:
    """The original stage-2 sweep: size and learning rate, plus the blind control."""
    common = dict(steps=steps, eval_every=eval_every, num_workers=num_workers)
    return [
        TrainConfig(name="tiny_lr3e-4", d_model=64, n_layers=1, n_heads=2,
                    cnn_channels=(16, 32, 32), lr=3e-4, **common),
        TrainConfig(name="tiny_lr1e-3", d_model=64, n_layers=1, n_heads=2,
                    cnn_channels=(16, 32, 32), lr=1e-3, **common),
        TrainConfig(name="small_lr3e-4", d_model=128, n_layers=2, n_heads=4,
                    cnn_channels=(32, 64, 64), lr=3e-4, **common),
        TrainConfig(name="small_lr1e-4", d_model=128, n_layers=2, n_heads=4,
                    cnn_channels=(32, 64, 64), lr=1e-4, **common),
        TrainConfig(name="blind_lr3e-4", d_model=64, n_layers=1, n_heads=2,
                    cnn_channels=(16, 32, 32), lr=3e-4, blind=True, **common),
    ]


#: Steps at which to evaluate the retrain. Everything peaked at 3,000 and declined
#: last time, so the interesting region is early and log-spaced, not evenly spaced.
RETRAIN_EVAL_STEPS: tuple[int, ...] = (100, 250, 500, 1000, 2000, 3000)


def retrain_configs(*, num_workers: int = 2, steps: int = 3000) -> list[TrainConfig]:
    """One config with previous actions, plus the no-previous-action ablation.

    Same architecture and learning rate in both, so the only difference is whether the
    model can know it is mid-jump.
    """
    common = dict(
        d_model=64, n_layers=1, n_heads=2, cnn_channels=(16, 32, 32), lr=1e-3,
        steps=steps, eval_steps=RETRAIN_EVAL_STEPS, num_workers=num_workers,
    )
    return [
        TrainConfig(name="prev4_lr1e-3", n_prev_actions=4, prev_action_dropout=0.25, **common),
        TrainConfig(name="noprev_ablation_lr1e-3", n_prev_actions=0, **common),
    ]


def run_trivial_baselines(
    corpus: Corpus,
    rom_path: Path | str,
    *,
    results_path: Path | str,
    seeds: int,
    live_frames: int,
    device: str = "cpu",
    log=print,
) -> None:
    """Score and (for the constant/marginal policies) live-play the floors."""
    device = torch.device(device)
    scores = evaluate_trivial_baselines(corpus.val_labels, corpus.train_counts, corpus.vocab)
    for score in scores:
        log(score.row())

    nothing = token_for_buttons(corpus.vocab)
    right_b = token_for_buttons(corpus.vocab, "Right", "B")
    probabilities = corpus.train_counts / max(corpus.train_counts.sum(), 1)
    policies = {
        "baseline_always_nothing": ConstantPolicy(corpus.vocab.size, nothing),
        "baseline_always_right_b": ConstantPolicy(corpus.vocab.size, right_b),
        "baseline_marginal_sample": MarginalPolicy(probabilities),
    }
    by_name = {s.name: s for s in scores}
    label_for = {
        "baseline_always_nothing": "always nothing",
        "baseline_always_right_b": "always Right+B",
        "baseline_marginal_sample": "sample marginal distribution",
    }
    player = LivePlayer(
        rom_path, corpus.vocab, max_frames=live_frames, device=device
    )
    for name, policy in policies.items():
        policy.to(device)
        try:
            live = player.evaluate(policy, seeds=seeds, temperature=1.0)
        except Exception as exc:
            live = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        score = by_name.get(label_for[name])
        append_jsonl(
            results_path,
            {
                "kind": "baseline",
                "name": name,
                "timestamp": time.time(),
                "val_accuracy": score.accuracy if score else None,
                "val_macro_accuracy": score.macro_accuracy if score else None,
                "live": live,
                "environment": environment(),
            },
        )
        log(
            f"  {name}: live progress median "
            f"{live.get('total_progress', {}).get('median', 'n/a')}, "
            f"levels median {live.get('levels_reached', {}).get('median', 'n/a')}"
        )
    append_jsonl(
        results_path,
        {
            "kind": "baseline_table",
            "timestamp": time.time(),
            "scores": [
                {
                    "name": s.name,
                    "val_accuracy": s.accuracy,
                    "val_macro_accuracy": s.macro_accuracy,
                    "predicts": s.top_prediction,
                }
                for s in scores
            ],
        },
    )


def run_sweep(
    corpus: Corpus,
    rom_path: Path | str,
    configs: list[TrainConfig],
    *,
    out_dir: Path | str,
    results_path: Path | str,
    eval_seeds: int = 20,
    live_frames: int = 3000,
    expert_movie: Path | str | None = None,
    eval_levels: tuple[str, ...] = ("1-1",),
    stall_limit: int = 300,
    device: torch.device | str | None = None,
    log=print,
) -> list[dict]:
    """Train every config, evaluating live at each checkpoint. Crashes are logged.

    ``device`` matters more than it looks: training on MPS makes every FCEUX child
    fall back to broken software OpenGL, so a run that needs live evaluation must
    train on the CPU. At 3,000 steps that costs about a minute.
    """
    out_dir = Path(out_dir)
    device = torch.device(device) if device is not None else pick_device()
    summaries: list[dict] = []

    for config in configs:
        log(f"\n=== {config.name} ===")
        started = time.perf_counter()
        try:
            train_set = corpus.dataset(
                "train", stack=config.stack, blind=config.blind,
                prev_actions=config.n_prev_actions,
            )
            val_set = corpus.dataset(
                "val", stack=config.stack, blind=config.blind,
                prev_actions=config.n_prev_actions,
            )

            def on_eval(metrics: dict, ckpt: Path, policy, _cfg=config):
                """Live-play at every checkpoint, so a partial night yields a curve.

                All selection rules are reported; none is dropped. Greedy is primary,
                sticky supplies the seed distribution, and the two temperatures are
                there for comparison.
                """
                live_by_rule: dict[str, dict] = {}
                for name, kwargs in (
                    ("greedy", dict(selection="greedy")),
                    ("sticky0.25", dict(selection="sticky", sticky_p=0.25)),
                    ("temp0.1", dict(selection="temperature", temperature=0.1)),
                    ("temp0.25", dict(selection="temperature", temperature=0.25)),
                ):
                    try:
                        live_by_rule[name] = run_eval_isolated(
                            ckpt,
                            corpus.vocab_path,
                            rom_path,
                            seeds=eval_seeds,
                            levels=eval_levels,
                            expert_movie=expert_movie,
                            stall_frames=stall_limit,
                            max_frames=live_frames,
                            **kwargs,
                        )
                    except Exception as exc:
                        live_by_rule[name] = {
                            "error": f"{type(exc).__name__}: {exc}"[:300]
                        }
                live = live_by_rule["greedy"]
                record = {
                    "kind": "eval",
                    "config": _cfg.name,
                    "blind": _cfg.blind,
                    "step": metrics["step"],
                    "timestamp": time.time(),
                    "train_config": _cfg.to_dict(),
                    "val": {
                        k: v for k, v in metrics.items() if k != "per_class_accuracy"
                    },
                    "per_class_accuracy": metrics.get("per_class_accuracy"),
                    "vocab_names": corpus.vocab.names,
                    "live": live,
                    "live_by_rule": live_by_rule,
                    "environment": environment(str(device)),
                }
                append_jsonl(results_path, record)
                for name, res in live_by_rule.items():
                    if "error" in res:
                        log(f"      {name:11s} ERROR {res['error'][:60]}")
                        continue
                    log(
                        f"      {name:11s} pipe1 {_pct(res.get('cleared_pipe1_rate'))} "
                        f"pipe2 {_pct(res.get('cleared_pipe2_rate'))} "
                        f"x_med {res.get('furthest_x', {}).get('median')} "
                        f"Ahold_max {res.get('longest_a_hold', {}).get('max')} "
                        f"flakes {res.get('retried_flakes')}"
                    )

            state = train(
                config,
                train_set,
                val_set,
                corpus.vocab,
                out_dir=out_dir,
                vocab_path=corpus.vocab_path,
                device=device,
                on_eval=on_eval,
                log=log,
            )
            summary = {
                "kind": "config_done",
                "config": config.name,
                "timestamp": time.time(),
                "wall_seconds": round(time.perf_counter() - started, 1),
                "first_loss": state.first_loss,
                "last_loss": state.last_loss,
                "n_evals": len(state.evals),
                "best_val_accuracy": max(
                    (e["accuracy"] for e in state.evals), default=None
                ),
                "best_macro_accuracy": max(
                    (e["macro_accuracy"] for e in state.evals), default=None
                ),
                "loss_curve_every20": state.smoothed(20),
            }
            append_jsonl(results_path, summary)
            summaries.append(summary)
        except Exception as exc:
            failure = {
                "kind": "config_failed",
                "config": config.name,
                "timestamp": time.time(),
                "wall_seconds": round(time.perf_counter() - started, 1),
                "error": f"{type(exc).__name__}: {exc}"[:500],
                "traceback": traceback.format_exc(limit=6)[:2000],
            }
            append_jsonl(results_path, failure)
            summaries.append(failure)
            log(f"  !! {config.name} FAILED: {type(exc).__name__}: {exc}")
    return summaries
