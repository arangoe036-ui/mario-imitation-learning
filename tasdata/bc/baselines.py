"""Trivial baselines, so the learned numbers mean something.

Accuracy on this task is a trap: 40% of frames are "no buttons", so a model that
has learned nothing but the prior scores 40%. These four give the floor.

``a`` always "nothing", ``b`` always "Right+B", ``c`` sample the marginal. The
blind model is not here -- it is a real trained network (same architecture, image
zeroed) and lives in the sweep, because a learned action prior is a much stronger
floor than any of these.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..buttons import NES_BUTTON_BITS
from .tokens import ActionVocab


class ConstantPolicy(torch.nn.Module):
    """Emits one fixed token, whatever it sees. Usable in live play."""

    def __init__(self, n_actions: int, token: int) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.token = int(token)

    def forward(self, frames: torch.Tensor, prev_actions=None) -> torch.Tensor:
        logits = torch.full(
            (frames.shape[0], self.n_actions), -1e4, device=frames.device
        )
        logits[:, self.token] = 1e4
        return logits


class MarginalPolicy(torch.nn.Module):
    """Samples from the training marginal, ignoring the observation."""

    def __init__(self, probabilities: np.ndarray) -> None:
        super().__init__()
        p = np.asarray(probabilities, dtype=np.float64)
        p = np.clip(p, 1e-12, None)
        self.register_buffer(
            "log_p", torch.from_numpy(np.log(p / p.sum())).float()
        )

    def forward(self, frames: torch.Tensor, prev_actions=None) -> torch.Tensor:
        return self.log_p.to(frames.device).expand(frames.shape[0], -1)


@dataclass
class BaselineScore:
    name: str
    accuracy: float
    macro_accuracy: float
    top_prediction: str

    def row(self) -> str:
        return (
            f"  {self.name:34s} acc {self.accuracy * 100:6.2f}%  "
            f"macro {self.macro_accuracy * 100:6.2f}%  predicts {self.top_prediction}"
        )


def token_for_buttons(vocab: ActionVocab, *buttons: str) -> int:
    """Token id for an exact button combination, e.g. ``()`` or ``("Right", "B")``."""
    byte = 0
    for b in buttons:
        byte |= NES_BUTTON_BITS[b]
    return int(vocab.byte_to_token[byte])


def score_predictions(
    predictions: np.ndarray, labels: np.ndarray, n_actions: int
) -> tuple[float, float, np.ndarray]:
    """Overall accuracy, macro (per-class mean) accuracy, and per-class accuracy."""
    correct = predictions == labels
    accuracy = float(correct.mean()) if labels.size else 0.0
    per_class = np.full(n_actions, np.nan)
    for token in range(n_actions):
        mask = labels == token
        if mask.any():
            per_class[token] = float(correct[mask].mean())
    macro = float(np.nanmean(per_class)) if np.isfinite(per_class).any() else 0.0
    return accuracy, macro, per_class


def evaluate_trivial_baselines(
    val_labels: np.ndarray, train_counts: np.ndarray, vocab: ActionVocab, *, seed: int = 0
) -> list[BaselineScore]:
    """Score the three non-learned baselines on the validation labels."""
    scores: list[BaselineScore] = []
    n = vocab.size

    nothing = token_for_buttons(vocab)
    acc, macro, _ = score_predictions(np.full_like(val_labels, nothing), val_labels, n)
    scores.append(BaselineScore("always nothing", acc, macro, vocab.names[nothing]))

    right_b = token_for_buttons(vocab, "Right", "B")
    acc, macro, _ = score_predictions(np.full_like(val_labels, right_b), val_labels, n)
    scores.append(BaselineScore("always Right+B", acc, macro, vocab.names[right_b]))

    rng = np.random.default_rng(seed)
    p = np.clip(train_counts.astype(np.float64), 0, None)
    p = p / p.sum() if p.sum() else np.full(n, 1 / n)
    sampled = rng.choice(n, size=val_labels.size, p=p)
    acc, macro, _ = score_predictions(sampled, val_labels, n)
    scores.append(
        BaselineScore("sample marginal distribution", acc, macro, "(stochastic)")
    )

    # The theoretical ceiling for any observation-blind policy: always pick the
    # single most likely token. Worth stating so the blind model can be compared
    # against what it *should* reach.
    argmax_token = int(np.argmax(train_counts))
    acc, macro, _ = score_predictions(
        np.full_like(val_labels, argmax_token), val_labels, n
    )
    scores.append(
        BaselineScore(
            "always train-mode token (blind ceiling)", acc, macro, vocab.names[argmax_token]
        )
    )
    return scores
