"""Transition-aware accuracy: does the model know *when to act*, or only to continue?

The label repeats the previous action about 97% of the time, because SMB holds buttons
for long stretches. So a headline accuracy of 74% says almost nothing: a model that
only ever predicts "same as last frame" scores ~97% on the 97% of frames where nothing
changes, and can be entirely wrong on every frame that matters.

This module splits validation into:

**non-transition** frames, where ``a_{t+1} == a_t`` -- the model just has to continue;
**transition** frames, where ``a_{t+1} != a_t`` -- the model has to *decide* something.

and additionally reports per-button *onset* recall: of the frames where a button goes
from released to held, how often does the prediction include it. Jump onsets (A) are the
ones that matter for clearing a pipe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..buttons import NES_BUTTON_BITS
from .data import FrameStackDataset
from .tokens import ActionVocab


@dataclass
class TransitionReport:
    """Accuracy split by whether the action changes, plus per-button onsets."""

    n_frames: int
    n_transition: int
    transition_rate: float
    accuracy_overall: float
    accuracy_non_transition: float
    accuracy_transition: float
    #: Accuracy a pure "repeat the previous action" policy would get.
    copy_baseline_overall: float
    #: Per button: onset count, recall, and predicted-press rate.
    buttons: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_frames": self.n_frames,
            "n_transition": self.n_transition,
            "transition_rate": self.transition_rate,
            "accuracy_overall": self.accuracy_overall,
            "accuracy_non_transition": self.accuracy_non_transition,
            "accuracy_transition": self.accuracy_transition,
            "copy_baseline_overall": self.copy_baseline_overall,
            "buttons": self.buttons,
        }

    def text(self) -> str:
        lines = [
            f"frames evaluated      : {self.n_frames:,}",
            f"transition frames     : {self.n_transition:,} "
            f"({self.transition_rate * 100:.2f}% -- the action changes)",
            "",
            f"accuracy overall          : {self.accuracy_overall * 100:6.2f}%",
            f"  on NON-transition frames: {self.accuracy_non_transition * 100:6.2f}%   "
            f"(just continue)",
            f"  on TRANSITION frames    : {self.accuracy_transition * 100:6.2f}%   "
            f"(decide something)",
            f"'repeat previous action' baseline: {self.copy_baseline_overall * 100:6.2f}%",
            "",
            f"{'button':8s} {'onsets':>8s} {'onset recall':>13s} {'pred rate':>10s} "
            f"{'expert rate':>12s}",
        ]
        for name, d in self.buttons.items():
            lines.append(
                f"{name:8s} {d['onsets']:8,d} {d['onset_recall'] * 100:12.2f}% "
                f"{d['predicted_rate'] * 100:9.2f}% {d['expert_rate'] * 100:11.2f}%"
            )
        return "\n".join(lines)


@torch.no_grad()
def measure_transitions(
    policy: torch.nn.Module,
    dataset: FrameStackDataset,
    vocab: ActionVocab,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
    limit: int | None = 60000,
    seed: int = 0,
) -> TransitionReport:
    """Run the policy over validation frames and split accuracy by transition."""
    policy.eval()
    device = torch.device(device)

    n = len(dataset)
    if limit is not None and limit < n:
        rng = np.random.default_rng(seed)
        picks = np.sort(rng.choice(n, size=limit, replace=False))
    else:
        picks = np.arange(n)

    from torch.utils.data import Subset

    loader = DataLoader(
        Subset(dataset, picks.tolist()), batch_size=batch_size, shuffle=False
    )

    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    prevs: list[np.ndarray] = []
    for frames, prev, target in loader:
        logits = policy(frames.to(device), prev.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(target.numpy())
        prevs.append(prev[:, -1].numpy())

    pred = np.concatenate(preds)
    label = np.concatenate(labels)
    prev_token = np.concatenate(prevs)

    changed = label != prev_token
    correct = pred == label

    byte_of = np.array(vocab.token_to_byte, dtype=np.int64)
    pred_byte = byte_of[pred]
    label_byte = byte_of[label]
    prev_byte = byte_of[prev_token]

    buttons: dict[str, dict] = {}
    for name, bit in NES_BUTTON_BITS.items():
        if name == "NOOP":
            continue
        label_on = (label_byte & bit) > 0
        prev_on = (prev_byte & bit) > 0
        pred_on = (pred_byte & bit) > 0
        onset = label_on & ~prev_on
        buttons[name] = {
            "onsets": int(onset.sum()),
            # Of the frames where this button starts being held, how often does the
            # prediction include it? This is what "knows when to jump" means.
            "onset_recall": float(pred_on[onset].mean()) if onset.any() else 0.0,
            "predicted_rate": float(pred_on.mean()),
            "expert_rate": float(label_on.mean()),
        }

    return TransitionReport(
        n_frames=int(label.size),
        n_transition=int(changed.sum()),
        transition_rate=float(changed.mean()),
        accuracy_overall=float(correct.mean()),
        accuracy_non_transition=float(correct[~changed].mean()) if (~changed).any() else 0.0,
        accuracy_transition=float(correct[changed].mean()) if changed.any() else 0.0,
        copy_baseline_overall=float((prev_token == label).mean()),
        buttons=buttons,
    )


def expert_transition_stats(actions_list: list[np.ndarray]) -> dict:
    """How often the expert changes action at all, and per-button onset counts."""
    total = 0
    changes = 0
    onsets = {name: 0 for name in NES_BUTTON_BITS if name != "NOOP"}
    held = {name: 0 for name in NES_BUTTON_BITS if name != "NOOP"}
    for actions in actions_list:
        a = actions.astype(np.int64)
        total += a.size - 1
        changes += int((a[1:] != a[:-1]).sum())
        for name, bit in NES_BUTTON_BITS.items():
            if name == "NOOP":
                continue
            on = (a & bit) > 0
            held[name] += int(on.sum())
            onsets[name] += int((on[1:] & ~on[:-1]).sum())
    return {
        "frames": total,
        "action_changes": changes,
        "change_rate": changes / total if total else 0.0,
        "onsets": onsets,
        "hold_rate": {k: v / (total + 1) for k, v in held.items()},
    }
