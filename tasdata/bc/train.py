"""Training loop, checkpointing, and validation.

Checkpoints are written at every evaluation so a night that ends early still leaves
a curve and a usable model. Each checkpoint carries its own config and vocabulary
path, so it can be reloaded and played without guessing how it was built.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..buttons import NES_BUTTON_ORDER
from .baselines import score_predictions
from .bernoulli import (
    ThresholdCalibration,
    bce_with_onset_weights,
    calibrate_thresholds,
    evaluate_bernoulli,
)
from .data import FrameStackDataset
from .model import BCPolicy, PolicyConfig, build_policy, pick_device
from .tokens import ActionVocab


@dataclass
class TrainConfig:
    """One point in the sweep."""

    name: str
    d_model: int = 64
    n_layers: int = 1
    n_heads: int = 2
    cnn_channels: tuple[int, ...] = (16, 32, 32)
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 128
    steps: int = 4000
    warmup: int = 100
    stack: int = 4
    dropout: float = 0.1
    blind: bool = False
    grad_clip: float = 1.0
    #: Evaluate (val + live play) every this many steps.
    eval_every: int = 1000
    #: Explicit step numbers to evaluate at. Overrides ``eval_every`` when set --
    #: everything peaked early last time, so the useful points are not evenly spaced.
    eval_steps: tuple[int, ...] = ()
    #: How many already-applied actions to feed the model. 0 = the ablation.
    n_prev_actions: int = 0
    prev_action_dropout: float = 0.25
    #: "categorical" or "bernoulli".
    head_type: str = "categorical"
    #: Arm B: multiplier on the loss for the specific button that turns on this frame.
    onset_weight: float = 1.0
    #: Cap validation frames so evaluation stays cheap.
    val_frames: int = 40000
    seed: int = 0
    num_workers: int = 2

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cnn_channels"] = list(self.cnn_channels)
        d["eval_steps"] = list(self.eval_steps)
        return d

    def should_eval(self, step: int) -> bool:
        if self.eval_steps:
            return step in self.eval_steps
        return step % self.eval_every == 0 or step == self.steps

    def policy_config(self, n_actions: int) -> PolicyConfig:
        return PolicyConfig(
            n_actions=n_actions,
            stack=self.stack,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            cnn_channels=tuple(self.cnn_channels),
            dropout=self.dropout,
            blind=self.blind,
            n_prev_actions=self.n_prev_actions,
            prev_action_dropout=self.prev_action_dropout,
            head_type=self.head_type,
        )


def save_checkpoint(
    path: Path | str,
    policy: BCPolicy,
    config: TrainConfig,
    *,
    step: int,
    vocab_path: Path | str,
    metrics: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": policy.state_dict(),
            "policy_config": policy.config.to_dict(),
            "train_config": config.to_dict(),
            "step": step,
            "vocab_path": str(vocab_path),
            "metrics": metrics or {},
            "thresholds": (metrics or {}).get("thresholds"),
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path | str, *, device: torch.device | str = "cpu"
) -> tuple[BCPolicy, dict]:
    """Rebuild a policy from a checkpoint. Raises if the file is unusable."""
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    policy = build_policy(PolicyConfig.from_dict(blob["policy_config"]))
    policy.load_state_dict(blob["model_state"])
    policy.to(torch.device(device)).eval()
    return policy, blob


def make_loader(
    dataset: FrameStackDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int = 0,
    limit: int | None = None,
) -> DataLoader:
    data = dataset
    if limit is not None and limit < len(dataset):
        rng = np.random.default_rng(seed)
        picks = rng.choice(len(dataset), size=limit, replace=False)
        data = Subset(dataset, np.sort(picks).tolist())
    return DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle,
        persistent_workers=num_workers > 0,
        pin_memory=False,
    )


@torch.no_grad()
def validate_bernoulli(
    policy: BCPolicy,
    loader: DataLoader,
    device: torch.device,
    *,
    target_rates: dict[str, float],
    expert_bytes: set[int] | None = None,
    max_batches: int | None = None,
) -> dict:
    """Validate a Bernoulli head: BCE, calibrated thresholds, onset recall, separation."""
    policy.eval()
    total_loss = 0.0
    n_batches = 0
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    prevs: list[np.ndarray] = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames, prev, bits, onset = batch
        frames = frames.to(device, non_blocking=True)
        bits = bits.to(device, non_blocking=True)
        logits = policy(frames, prev)
        total_loss += float(bce_with_onset_weights(logits, bits))
        n_batches += 1
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(bits.cpu().numpy())
        # previous-frame bits = this frame's bits minus the onsets that just happened
        prevs.append((bits.cpu().numpy() - onset.numpy()).clip(0, 1))
    if not probs:
        return {"loss": float("nan")}
    P = np.concatenate(probs)
    Y = np.concatenate(labels)
    V = np.concatenate(prevs)
    report = evaluate_bernoulli(P, Y, V, target_rates, expert_bytes=expert_bytes)
    return {
        "loss": total_loss / max(n_batches, 1),
        "n_frames": int(Y.shape[0]),
        "exact_match": report.exact_match,
        "hamming": report.hamming,
        "onset_recall": report.onset_recall,
        "onset_counts": report.onset_counts,
        "separation": report.separation,
        "thresholds": report.calibration.thresholds,
        "realized_rate": report.calibration.realized_rate,
        "target_rate": report.calibration.target_rate,
        "novel_combo_rate": report.novel_combo_rate,
        "novel_combos": report.novel_combos,
        # accuracy-shaped fields so downstream reporting keeps working
        "accuracy": report.exact_match,
        "macro_accuracy": float(np.mean(list(report.onset_recall.values()))),
    }


@torch.no_grad()
def validate(
    policy: BCPolicy,
    loader: DataLoader,
    device: torch.device,
    n_actions: int,
    *,
    max_batches: int | None = None,
) -> dict:
    """Loss, accuracy, macro accuracy and per-class accuracy on a loader."""
    policy.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0
    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    lasts: list[np.ndarray] = []
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        frames, prev, target = batch
        frames = frames.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        lasts.append(prev[:, -1].cpu().numpy())
        logits = policy(frames, prev)
        total_loss += float(loss_fn(logits, target))
        n_batches += 1
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(target.cpu().numpy())
    if not preds:
        return {"loss": float("nan"), "accuracy": 0.0, "macro_accuracy": 0.0}
    p = np.concatenate(preds)
    y = np.concatenate(labels)
    last = np.concatenate(lasts) if lasts else np.zeros_like(p)
    acc, macro, per_class = score_predictions(p, y, n_actions)
    pred_counts = np.bincount(p, minlength=n_actions)
    # Copycat symptom: how often the model simply reproduces the previous action.
    copycat = float((p == last).mean()) if p.size else 0.0
    # The label itself repeats the previous action this often, so copycat only
    # indicates a problem when it materially exceeds this floor.
    label_repeat = float((y == last).mean()) if y.size else 0.0
    return {
        "loss": total_loss / max(n_batches, 1),
        "accuracy": acc,
        "macro_accuracy": macro,
        "copycat_rate": copycat,
        "label_repeat_rate": label_repeat,
        "n_frames": int(y.size),
        "per_class_accuracy": [
            None if not np.isfinite(v) else round(float(v), 4) for v in per_class
        ],
        "prediction_counts": pred_counts.tolist(),
        "label_counts": np.bincount(y, minlength=n_actions).tolist(),
    }


@dataclass
class TrainState:
    """Everything the sweep wants to know about a run in progress."""

    losses: list[float] = field(default_factory=list)
    evals: list[dict] = field(default_factory=list)

    @property
    def first_loss(self) -> float:
        return self.losses[0] if self.losses else float("nan")

    @property
    def last_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")

    def smoothed(self, k: int = 20) -> list[float]:
        if not self.losses:
            return []
        out = []
        for i in range(0, len(self.losses), k):
            chunk = self.losses[i : i + k]
            out.append(float(np.mean(chunk)))
        return out


def train(
    config: TrainConfig,
    train_set: FrameStackDataset,
    val_set: FrameStackDataset,
    vocab: ActionVocab,
    *,
    out_dir: Path | str,
    vocab_path: Path | str,
    device: torch.device | None = None,
    target_rates: dict[str, float] | None = None,
    expert_bytes: set[int] | None = None,
    on_eval=None,
    log=print,
) -> TrainState:
    """Train one config, evaluating and checkpointing every ``eval_every`` steps."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    policy = build_policy(config.policy_config(vocab.size)).to(device)
    log(
        f"  {config.name}: {policy.n_parameters:,} parameters, device {device}, "
        f"blind={config.blind}"
    )
    optimiser = torch.optim.AdamW(
        policy.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser,
        lambda s: min(1.0, (s + 1) / max(config.warmup, 1)),
    )
    loss_fn = nn.CrossEntropyLoss()
    bernoulli = config.head_type == "bernoulli"

    loader = make_loader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    val_loader = make_loader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=min(config.num_workers, 2),
        seed=config.seed,
        limit=config.val_frames,
    )

    state = TrainState()
    step = 0
    started = time.perf_counter()
    policy.train()
    while step < config.steps:
        for batch in loader:
            if step >= config.steps:
                break
            if bernoulli:
                frames, prev, bits, onset = batch
                logits = policy(frames.to(device, non_blocking=True), prev)
                loss = bce_with_onset_weights(
                    logits,
                    bits.to(device, non_blocking=True),
                    onset.to(device, non_blocking=True),
                    onset_weight=config.onset_weight,
                )
            else:
                frames, prev, target = batch
                frames = frames.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                logits = policy(frames, prev)
                loss = loss_fn(logits, target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
            optimiser.step()
            scheduler.step()
            state.losses.append(float(loss.detach()))
            step += 1

            if config.should_eval(step):
                metrics = (
                    validate_bernoulli(
                        policy, val_loader, device,
                        target_rates=target_rates or {},
                        expert_bytes=expert_bytes,
                    )
                    if bernoulli
                    else validate(policy, val_loader, device, vocab.size)
                )
                metrics["step"] = step
                metrics["train_loss_recent"] = float(np.mean(state.losses[-100:]))
                metrics["head_type"] = config.head_type
                metrics["onset_weight"] = config.onset_weight
                metrics["elapsed_seconds"] = round(time.perf_counter() - started, 1)
                ckpt = save_checkpoint(
                    out_dir / f"{config.name}_step{step}.pt",
                    policy,
                    config,
                    step=step,
                    vocab_path=vocab_path,
                    metrics=metrics,
                )
                metrics["checkpoint"] = str(ckpt)
                state.evals.append(metrics)
                if bernoulli:
                    log(
                        f"    step {step:6d}  train {metrics['train_loss_recent']:.4f}  "
                        f"val {metrics['loss']:.4f}  exact {metrics['exact_match'] * 100:5.2f}%  "
                        f"A-onset recall {metrics['onset_recall'].get('A', 0) * 100:5.1f}%  "
                        f"A sep {metrics['separation'].get('A', 0):.2f}x  "
                        f"A thr {metrics['thresholds'].get('A', 0):.2f} "
                        f"-> {metrics['realized_rate'].get('A', 0) * 100:.1f}%"
                    )
                else:
                    log(
                        f"    step {step:6d}  train {metrics['train_loss_recent']:.4f}  "
                        f"val {metrics['loss']:.4f}  acc {metrics['accuracy'] * 100:5.2f}%  "
                        f"macro {metrics['macro_accuracy'] * 100:5.2f}%"
                    )
                if on_eval:
                    on_eval(metrics, ckpt, policy)
                policy.train()
    return state
