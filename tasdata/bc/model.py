"""Policy network: per-frame CNN encoder into a transformer over the frame window.

Deliberately small. The dataset is 1.2M frames of a game whose action distribution
is 97% covered by eight tokens, so capacity is not the binding constraint --
generalisation across levels is. The smallest config here is ~0.25M parameters.

Each of the 4 stacked frames is encoded independently by a shared CNN into one
token, positional embeddings mark recency, and a transformer encoder attends over
the window. The head reads the most recent position, so the representation is
free to use the older frames purely as context for velocity.

``blind=True`` zeroes the image *inside the model* as well as in the data pipeline,
so the blind baseline cannot cheat through any path.

Previous actions
----------------
Pixels alone cannot tell the model it is *mid-jump*: whether to keep holding A depends
on how long A has already been held, which is not on screen. So the last ``k`` applied
actions are embedded and appended to the sequence.

This invites the copycat failure -- the cheapest way to fit an autocorrelated action
stream is to echo the last action, which produces a policy that never initiates
anything. Two guards: the whole previous-action block is dropped out (replaced by a
mask token) on a fraction of training samples, so the model cannot rely on it; and the
head reads the last *frame* token rather than the last action token, so copying is not
the structurally shortest path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class PolicyConfig:
    """Everything that defines the architecture."""

    n_actions: int
    stack: int = 4
    #: Side length of the square grayscale input. **Was hardcoded to 84 in three places**, so
    #: resolution could not be varied without editing the model. Checkpoints saved before this
    #: field existed omit it and `from_dict` restores the 84 default, which is what they were.
    frame_size: int = 84
    d_model: int = 64
    n_layers: int = 1
    n_heads: int = 2
    cnn_channels: tuple[int, ...] = (16, 32, 32)
    ff_mult: int = 2
    dropout: float = 0.1
    blind: bool = False
    #: "categorical" (25-way softmax) or "bernoulli" (8 independent buttons).
    #: Bernoulli avoids vote-splitting: the four A-containing tokens each lose to
    #: Right+B under argmax, so A was emitted on 0.03% of frames despite carrying
    #: real signal. It also makes the RARE token unnecessary -- buttons are predicted
    #: directly from the raw action byte.
    head_type: str = "categorical"
    n_buttons: int = 8
    #: How many already-applied actions to feed. 0 disables the input entirely.
    n_prev_actions: int = 0
    #: Probability of masking the whole previous-action block during training.
    prev_action_dropout: float = 0.25
    #: Width of one hidden layer in the action head. 0 = the original single `Linear(d_model,
    #: n_actions)`.
    #:
    #: **Why this exists (block 63/64).** The trunk's features carry wall identity at AUC 0.892-1.000
    #: and x position at R^2 0.712, but the on-top-of-pipe versus at-the-face distinction -- 17% of all
    #: failures, half of pipe-4 losses, needing OPPOSITE corrections -- is **not linearly decodable**
    #: from them (linear probe AUC 0.651, p=0.17) while a small MLP does decode it (0.743, p=0.010).
    #: The action head was one linear layer on exactly those features, so it could not read the one
    #: distinction that mattered. This is the cheapest change that could: +27,520 parameters at
    #: hidden=128, or +8% of the model.
    head_hidden: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cnn_channels"] = list(self.cnn_channels)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> PolicyConfig:
        d = dict(d)
        d["cnn_channels"] = tuple(d.get("cnn_channels", (16, 32, 32)))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class FrameEncoder(nn.Module):
    """Shared CNN: one ``size``x``size`` grayscale frame -> one ``d_model`` vector."""

    def __init__(self, channels: tuple[int, ...], d_model: int, size: int = 84) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = 1
        # Strided convs, DQN-style: 84 -> 20 -> 9 -> 7; at 128 -> 31 -> 14 -> 12
        kernels = [8, 4, 3]
        strides = [4, 2, 1]
        for i, out_ch in enumerate(channels):
            k = kernels[i] if i < len(kernels) else 3
            s = strides[i] if i < len(strides) else 1
            layers += [nn.Conv2d(in_ch, out_ch, k, s), nn.ReLU(inplace=True)]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.size = int(size)
        # Probe rather than derive: the flatten width depends on kernel/stride/padding and an
        # off-by-one here is a silent shape error at the first backward pass.
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 1, self.size, self.size)).numel()
        self.proj = nn.Linear(n_flat, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, 1, size, size)
        return self.proj(self.conv(x).flatten(1))


class BCPolicy(nn.Module):
    """Behavioural-cloning policy over a stack of frames."""

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = FrameEncoder(config.cnn_channels, config.d_model, config.frame_size)
        n_positions = config.stack + max(0, config.n_prev_actions)
        self.pos = nn.Parameter(torch.zeros(1, n_positions, config.d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        if config.n_prev_actions > 0:
            # +1 for the mask token used by previous-action dropout.
            self.action_embed = nn.Embedding(config.n_actions + 1, config.d_model)
            nn.init.trunc_normal_(self.action_embed.weight, std=0.02)
            self.mask_token = config.n_actions
        else:
            self.action_embed = None
            self.mask_token = config.n_actions
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.ff_mult,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)
        n_out = config.n_buttons if config.head_type == "bernoulli" else config.n_actions
        if config.head_hidden and config.head_hidden > 0:
            self.head = nn.Sequential(
                nn.Linear(config.d_model, config.head_hidden),
                nn.GELU(),
                nn.Linear(config.head_hidden, n_out),
            )
        else:
            self.head = nn.Linear(config.d_model, n_out)

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self, frames: torch.Tensor, prev_actions: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Logits from ``frames`` ``(B, stack, S, S)`` and optional ``prev`` ``(B, k)``."""
        b, t = frames.shape[0], frames.shape[1]
        if self.config.blind:
            # Belt and braces: the data pipeline already zeroes it.
            frames = torch.zeros_like(frames)
        s = self.config.frame_size
        if frames.shape[-1] != s or frames.shape[-2] != s:
            raise ValueError(
                f"frame_size mismatch: model expects {s}x{s}, got "
                f"{frames.shape[-2]}x{frames.shape[-1]}. A resolution change needs a re-capture "
                f"or a load-time downscale, not just a config edit."
            )
        x = self.encoder(frames.reshape(b * t, 1, s, s)).reshape(b, t, -1)

        if self.config.n_prev_actions > 0:
            k = self.config.n_prev_actions
            if prev_actions is None:
                prev = torch.full(
                    (b, k), self.mask_token, dtype=torch.long, device=frames.device
                )
            else:
                prev = prev_actions[:, -k:].to(frames.device).long()
                if prev.shape[1] < k:  # pad on the left with the mask token
                    pad = torch.full(
                        (b, k - prev.shape[1]), self.mask_token,
                        dtype=torch.long, device=frames.device,
                    )
                    prev = torch.cat([pad, prev], dim=1)
            if self.training and self.config.prev_action_dropout > 0:
                drop = (
                    torch.rand(b, 1, device=prev.device) < self.config.prev_action_dropout
                )
                prev = torch.where(drop.expand_as(prev), self.mask_token, prev)
            x = torch.cat([x, self.action_embed(prev)], dim=1)

        x = self.transformer(x + self.pos[:, : x.shape[1]])
        # Read the last FRAME position, never the last action position: copying the
        # previous action must not be the structurally shortest path to low loss.
        return self.head(self.norm(x[:, t - 1]))


def build_policy(config: PolicyConfig) -> BCPolicy:
    return BCPolicy(config)


def pick_device(prefer: str = "auto") -> torch.device:
    """Resolve a device, *without* probing MPS unless explicitly asked to.

    Do not remove the early return. On this machine, merely calling
    ``torch.backends.mps.is_available()`` initialises Metal in a way that makes every
    subsequently spawned FCEUX child fall back to Qt's software OpenGL backend --
    "known to be broken on macOS Tahoe" -- and segfault. Measured:

        before torch                exit=0    GL=Metal
        after import torch          exit=0    GL=Metal
        after mps.is_available()    exit=-11  GL=SOFTWARE   <-- here
        after empty_cache()         exit=-11  GL=SOFTWARE   <-- not reversible

    The poison is inherited by child processes, so a process that needs to run the
    emulator must never probe MPS -- not even to ask whether it exists. Pass an
    explicit device instead.
    """
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():  # poisons FCEUX; see above
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
