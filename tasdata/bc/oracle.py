"""Stage 3 arm B: a search oracle that labels "should Mario jump here?" by trying both.

The premise is that a short lookahead can answer the one question behavioural cloning got
worst -- when to press A -- without any expert data. At a state, branch: force A on, roll
forward, measure progress; restore; force A off, roll forward, measure; restore. Label the
state with whichever branch got further.

Before generating any data with it, the oracle is validated against the expert on *expert
states*: run it on frames of a held-out expert run and ask how often its choice matches
what the expert actually did. Agreement at A-onsets is the number that matters -- overall
agreement is dominated by the ~85% of frames where the answer is "no jump" and a
do-nothing oracle would look excellent.

Design choices that are themselves under test, stated explicitly because a disagreement
means one of them is wrong rather than that the expert is:

* **Horizon** (``HORIZON``, 60 frames = 1 second). Long enough that a jump completes and
  Mario lands; short enough to be affordable. Too short and a jump looks like pure loss,
  because rising costs horizontal speed before it buys anything.
* **Jump hold** (``JUMP_HOLD``, 20 frames). A single frame of A is a hop that clears
  nothing; the expert's A-holds run to a median of 2 but a p90 of 23. The oracle is
  choosing "commit to a jump" versus "stay grounded", not "set the A bit for one frame".
* **Continuation policy** (``Right+B`` held for the whole horizon). The oracle cannot use
  the expert's later inputs -- at data-generation time there are none -- so both branches
  continue under the same fixed run-right policy. This is what makes the comparison fair
  and also what makes it approximate.
* **Progress measure** (furthest absolute x reached, with death treated as the worst
  possible outcome). Final x alone would score a death mid-flight as progress.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER
from ..ram import DYING_STATES, PLAYER_STATE_NORMAL, read_smb
from ..replay import _resize_gray
from .session import FceuxSession
from .tokens import LIVE_MASK

#: Frames of lookahead per branch.
HORIZON = 60
#: How long A is held in the "jump" branch.
JUMP_HOLD = 20
#: Held for the whole horizon in both branches.
CONTINUATION = NES_BUTTON_BITS["Right"] | NES_BUTTON_BITS["B"]
A_BIT = NES_BUTTON_BITS["A"]

#: Progress penalty for dying inside the horizon, in x units. Larger than any level.
DEATH_PENALTY = 10_000


@dataclass
class Branch:
    """Outcome of rolling one branch forward."""

    pressed_a: bool
    furthest_x: int
    died: bool
    frames: int

    @property
    def score(self) -> int:
        return self.furthest_x - (DEATH_PENALTY if self.died else 0)


@dataclass
class OracleDecision:
    """What the oracle chose at one frame, and what it cost."""

    frame: int
    choose_a: bool
    margin: int          # score(A) - score(no A); sign is the decision
    a_branch: Branch
    no_branch: Branch
    emulator_frames: int


@dataclass
class OracleReport:
    """Agreement between the oracle and the expert on expert states."""

    n_frames: int
    n_decided: int
    horizon: int
    jump_hold: int
    agreement_overall: float
    n_a_onsets: int
    agreement_at_a_onsets: float
    #: Of A-onsets, how often the oracle also said jump (its recall on the thing we care about).
    oracle_says_jump_at_onsets: float
    n_expert_no_a: int
    #: How often the oracle says jump where the expert did not press A at all.
    oracle_jump_where_expert_none: float
    oracle_jump_rate: float
    expert_a_rate: float
    #: Ties, where both branches scored identically -- the oracle has no opinion.
    tie_rate: float
    emulator_frames_total: int
    emulator_frames_per_label: float
    seconds_per_label: float
    margin_stats: dict = field(default_factory=dict)

    def text(self) -> str:
        return "\n".join(
            [
                f"frames evaluated            : {self.n_frames:,}",
                f"horizon / jump hold         : {self.horizon} / {self.jump_hold} frames",
                "",
                f"oracle says JUMP            : {self.oracle_jump_rate * 100:5.1f}% of frames",
                f"expert pressed A            : {self.expert_a_rate * 100:5.1f}% of frames",
                f"ties (no opinion)           : {self.tie_rate * 100:5.1f}%",
                "",
                f"agreement, overall          : {self.agreement_overall * 100:5.1f}%",
                f"A-onsets in sample          : {self.n_a_onsets:,}",
                f"agreement at A-onsets       : {self.agreement_at_a_onsets * 100:5.1f}%",
                f"oracle jumps at A-onsets    : {self.oracle_says_jump_at_onsets * 100:5.1f}%",
                f"oracle jumps where expert did not: "
                f"{self.oracle_jump_where_expert_none * 100:5.1f}%",
                "",
                f"emulator frames total       : {self.emulator_frames_total:,}",
                f"emulator frames per label   : {self.emulator_frames_per_label:.0f}",
                f"seconds per label           : {self.seconds_per_label:.3f}",
            ]
        )

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _policy_bytes(policy, window, thresholds, rng) -> int:
    """One per-button sample from a Bernoulli policy, given the current frame stack."""
    import torch

    with torch.no_grad():
        batch = torch.from_numpy(window[None]).float().div_(255.0)
        logits = policy(batch)[0].float().cpu().numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    bits = rng.random(probs.shape) < probs
    byte = 0
    for j, name in enumerate(NES_BUTTON_ORDER):
        if bits[j]:
            byte |= NES_BUTTON_BITS[name]
    return byte & LIVE_MASK


def _roll_with_policy(
    session: FceuxSession,
    ordinal: int,
    press_a: bool,
    policy,
    thresholds: np.ndarray,
    *,
    horizon: int = HORIZON,
    jump_hold: int = JUMP_HOLD,
    measure: str = "final",
    stack: int = 4,
    seed: int = 0,
    suppress_off: int | None = None,
) -> Branch:
    """Roll a branch forward with a trained policy driving everything except the A bit.

    Fixed Right+B as the continuation asks "does jumping hurt a run-right agent over the
    next second?", and in SMB it essentially never does -- you keep horizontal momentum
    while airborne. Measured, that oracle over-jumped 4-5x and agreed with the expert at
    A-onsets 50.9% of the time, which is chance. Driving the continuation with a policy
    that already has 45.5% onset recall asks the better-posed question: "given how I would
    actually play from here, is starting a jump now better than not?"

    Both branches use the *same* rng seed, so the only difference between them is the
    forced A bit during ``jump_hold`` -- common random numbers, which removes sampling
    noise from the comparison rather than averaging it away.
    """
    rng = np.random.default_rng(seed)
    obs = session.reset_ordinal(ordinal)
    window = np.zeros((stack, 84, 84), dtype=np.uint8)
    window[:] = _resize_gray(obs.rgb, (84, 84))

    state = read_smb(obs.ram, obs.framecount)
    furthest = last = state.x_position
    died = False
    i = 0
    for i in range(horizon):
        byte = _policy_bytes(policy, window, thresholds, rng)
        if press_a:
            if i < jump_hold:
                byte |= A_BIT
        elif i < (jump_hold if suppress_off is None else suppress_off):
            # How long the "don't jump" branch is forbidden from jumping. Suppressing for
            # the full jump_hold is not the mirror image of forcing a jump -- it also
            # blocks the policy from starting its *own* jump for 20 frames, which
            # handicaps the branch rather than testing it. Measured, that asymmetry made
            # the oracle choose jump on 61% of frames against the expert's 6%.
            byte &= ~A_BIT
        obs = session.step(byte)
        window = np.roll(window, -1, axis=0)
        window[-1] = _resize_gray(obs.rgb, (84, 84))
        state = read_smb(obs.ram, obs.framecount)
        if state.player_state in DYING_STATES:
            died = True
            break
        if state.player_state == PLAYER_STATE_NORMAL and 1 <= state.world <= 8:
            furthest = max(furthest, state.x_position)
            last = state.x_position
    score = furthest if measure == "furthest" else last
    return Branch(pressed_a=press_a, furthest_x=int(score), died=died, frames=i + 1)


def decide_with_policy(
    session: FceuxSession,
    ordinal: int,
    frame: int,
    policy,
    thresholds: np.ndarray,
    *,
    horizon: int = HORIZON,
    jump_hold: int = JUMP_HOLD,
    measure: str = "final",
    stack: int = 4,
    n_rollouts: int = 1,
    seed: int = 0,
    suppress_off: int | None = None,
) -> OracleDecision:
    """Policy-continuation oracle. ``n_rollouts`` averages over sampling noise."""
    kw = {"horizon": horizon, "jump_hold": jump_hold, "measure": measure, "stack": stack,
          "suppress_off": suppress_off}
    a_scores, no_scores = [], []
    frames = 0
    a_last = no_last = None
    for r in range(n_rollouts):
        a = _roll_with_policy(session, ordinal, True, policy, thresholds, seed=seed + r, **kw)
        no = _roll_with_policy(session, ordinal, False, policy, thresholds, seed=seed + r, **kw)
        a_scores.append(a.score)
        no_scores.append(no.score)
        frames += a.frames + no.frames
        a_last, no_last = a, no
    margin = float(np.mean(a_scores) - np.mean(no_scores))
    return OracleDecision(
        frame=frame,
        choose_a=margin > 0,
        margin=int(round(margin)),
        a_branch=a_last,
        no_branch=no_last,
        emulator_frames=frames,
    )


def _roll(
    session: FceuxSession,
    ordinal: int,
    press_a: bool,
    *,
    horizon: int = HORIZON,
    jump_hold: int = JUMP_HOLD,
    measure: str = "furthest",
) -> Branch:
    """Restore the state, run the horizon under one branch, report progress.

    ``measure="furthest"`` scores the furthest x reached; ``"final"`` scores x at the end
    of the horizon. The difference matters more than it looks: in SMB you keep horizontal
    momentum while airborne, so a jump barely changes the furthest x reached inside a
    second, and "furthest" is close to blind to the very decision being made.
    """
    obs = session.reset_ordinal(ordinal)
    state = read_smb(obs.ram, obs.framecount)
    furthest = last = state.x_position
    died = False
    i = 0
    for i in range(horizon):
        byte = CONTINUATION
        if press_a and i < jump_hold:
            byte |= A_BIT
        obs = session.step(byte)
        state = read_smb(obs.ram, obs.framecount)
        if state.player_state in DYING_STATES:
            died = True
            break
        if state.player_state == PLAYER_STATE_NORMAL and 1 <= state.world <= 8:
            furthest = max(furthest, state.x_position)
            last = state.x_position
    score = furthest if measure == "furthest" else last
    return Branch(pressed_a=press_a, furthest_x=int(score), died=died, frames=i + 1)


def decide(
    session: FceuxSession,
    ordinal: int,
    frame: int,
    *,
    horizon: int = HORIZON,
    jump_hold: int = JUMP_HOLD,
    measure: str = "furthest",
) -> OracleDecision:
    """Try both branches from one savestate and pick the better."""
    kw = {"horizon": horizon, "jump_hold": jump_hold, "measure": measure}
    a = _roll(session, ordinal, True, **kw)
    no = _roll(session, ordinal, False, **kw)
    return OracleDecision(
        frame=frame,
        choose_a=a.score > no.score,
        margin=a.score - no.score,
        a_branch=a,
        no_branch=no,
        emulator_frames=a.frames + no.frames,
    )
