"""Episode playback on a persistent :class:`~tasdata.bc.session.FceuxSession`.

Same metrics as the old per-episode player, same ``EpisodeResult``, but every episode
begins with a ``savestate.load`` on an already-running emulator instead of launching a
fresh FCEUX and fast-forwarding a movie. That removes the process-per-episode behaviour
that exhausted IOSurface clients, and with it the test failures, the window flashing and
most of the evaluation cost.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER
from ..ram import DYING_STATES, PLAYER_STATE_NORMAL, read_smb
from ..replay import _resize_gray
from .live import (
    DEFAULT_STALL_LIMIT,
    PIPE1_CLEARED_X,
    PIPE2_CLEARED_X,
    EpisodeResult,
    _button_rates,
    _hold_stats,
    _novel_list,
    _novel_rate,
    summarise_episodes,
)
from .session import FceuxSession
from .statelib import StartPoint
from .tokens import LIVE_MASK, ActionVocab


def play_episode(
    session: FceuxSession,
    policy: torch.nn.Module,
    start: StartPoint,
    vocab: ActionVocab,
    *,
    seed: int = 0,
    selection: str = "threshold",
    temperature: float = 1.0,
    sticky_p: float = 0.25,
    thresholds: np.ndarray | None = None,
    head_type: str = "bernoulli",
    stack: int = 4,
    max_frames: int = 2500,
    stall_limit: int = DEFAULT_STALL_LIMIT,
    expert_bytes: set[int] | None = None,
    mask_live: bool = True,
    record: list | None = None,
) -> EpisodeResult:
    """Roll the policy forward from one savestate and score the episode.

    When ``record`` is given, each decision appends ``(obs84, byte, ram_state)`` to it --
    the observation the policy saw and the action it chose there. Rollouts are seeded and
    start from a savestate, so scoring a batch without recording and then re-rolling only
    the accepted seeds reproduces them exactly, at a fraction of the memory.
    """
    rng = np.random.default_rng(seed)
    policy.eval()

    obs0 = session.reset(start.frame)
    window = np.zeros((stack, 84, 84), dtype=np.uint8)
    first = _resize_gray(obs0.rgb, (84, 84))
    window[:] = first  # edge-pad the stack with the start frame

    deaths = 0
    prev_dying = False
    # Failure taxonomy. "Distance reached" cannot distinguish a policy that is standing
    # still against a pipe from one that is dying to a Goomba, and those need opposite
    # fixes, so every episode is classified by how it actually ended.
    death_causes: list[str] = []
    last_timer = None
    max_x_by_level: dict[str, int] = {}
    order: list[str] = []
    emitted: dict[int, int] = {}
    hold_runs = {n: [] for n in NES_BUTTON_ORDER}
    hold_open = {n: 0 for n in NES_BUTTON_ORDER}
    a_hold = longest_a_hold = a_presses = 0
    best_x = -1
    since_progress = 0
    prev_byte: int | None = None
    decisions = repeats = 0
    ended = "reached frame budget"
    frames_run = 0
    state = read_smb(obs0.ram, obs0.framecount)

    with torch.no_grad():
        for _ in range(max_frames):
            # -- decide from the current window -----------------------------
            batch = torch.from_numpy(window[None]).float().div_(255.0)
            logits = policy(batch)[0].float().cpu().numpy()
            if head_type == "bernoulli":
                probs = 1.0 / (1.0 + np.exp(-logits))
                if selection == "sample":
                    bits = rng.random(probs.shape) < probs
                else:
                    bits = probs > thresholds
                byte = 0
                for j, name in enumerate(NES_BUTTON_ORDER):
                    if bits[j]:
                        byte |= NES_BUTTON_BITS[name]
                if selection == "sticky" and prev_byte is not None and rng.random() < sticky_p:
                    byte = prev_byte
            else:
                if selection == "greedy":
                    token = int(logits.argmax())
                elif selection == "sticky" and prev_byte is not None and rng.random() < sticky_p:
                    token = int(vocab.byte_to_token[prev_byte])
                elif selection == "temperature":
                    z = logits / max(temperature, 1e-6)
                    z -= z.max()
                    p = np.exp(z)
                    token = int(rng.choice(len(p), p=p / p.sum()))
                else:
                    token = int(logits.argmax())
                byte = vocab.token_to_byte[token]
            if mask_live:
                byte &= LIVE_MASK

            decisions += 1
            if prev_byte is not None and byte == prev_byte:
                repeats += 1
            emitted[byte] = emitted.get(byte, 0) + 1
            if byte & 0x01:
                if a_hold == 0:
                    a_presses += 1
                a_hold += 1
                longest_a_hold = max(longest_a_hold, a_hold)
            else:
                a_hold = 0
            for name in NES_BUTTON_ORDER:
                if byte & NES_BUTTON_BITS[name]:
                    hold_open[name] += 1
                elif hold_open[name]:
                    hold_runs[name].append(hold_open[name])
                    hold_open[name] = 0
            prev_byte = byte
            if record is not None:
                record.append((window[-1].copy(), byte))

            # -- advance -----------------------------------------------------
            obs = session.step(byte)
            frames_run += 1
            window = np.roll(window, -1, axis=0)
            window[-1] = _resize_gray(obs.rgb, (84, 84))
            state = read_smb(obs.ram, obs.framecount)

            in_control = (
                state.pregame == 1
                and state.player_state == PLAYER_STATE_NORMAL
                and 1 <= state.world <= 8
                and 1 <= state.stage <= 4
            )
            if state.pregame == 1 and 1 <= state.world <= 8 and 1 <= state.stage <= 4:
                label = state.label()
                if label not in max_x_by_level:
                    order.append(label)
                max_x_by_level[label] = max(max_x_by_level.get(label, 0), state.x_position)

            dying = state.player_state in DYING_STATES
            if dying and not prev_dying:
                deaths += 1
                # Below the floor line means Mario fell; anything else killed him on it.
                death_causes.append("pit" if state.y_position > 200 else "enemy_contact")
            prev_dying = dying
            if state.time is not None:
                last_timer = state.time

            if in_control:
                if state.x_position > best_x:
                    best_x = state.x_position
                    since_progress = 0
                else:
                    since_progress += 1
                    if stall_limit and since_progress > stall_limit:
                        ended = f"no progress for {stall_limit} frames"
                        break
            if state.pregame == 2:
                ended = "game over"
                break
            if last_timer == 0 and in_control:
                ended = "timer expired"
                break

    furthest = order[-1] if order else "-"
    start_x = max_x_by_level.get(start.label, 0)

    # One label per episode, most specific first: a death outranks a stall, because the
    # stall detector also fires on the respawn that follows.
    if death_causes:
        end_class = death_causes[-1]
    elif ended == "timer expired":
        end_class = "timer"
    elif "no progress" in ended:
        end_class = "stuck_terrain"
    elif ended == "game over":
        end_class = "game_over"
    else:
        end_class = "budget_reached"
    return EpisodeResult(
        seed=seed,
        frames=frames_run,
        frames_survived=frames_run,
        furthest_level=furthest,
        levels_reached=len(order),
        furthest_x=max_x_by_level.get(furthest, 0),
        total_progress=int(sum(max_x_by_level.values())),
        deaths=deaths,
        ended=ended,
        start_level=start.label,
        cleared_pipe1=bool(start.label == "1-1" and start_x > PIPE1_CLEARED_X),
        cleared_pipe2=bool(start.label == "1-1" and start_x > PIPE2_CLEARED_X),
        longest_a_hold=longest_a_hold,
        a_presses=a_presses,
        hold_stats=_hold_stats(hold_runs, hold_open, max(1, frames_run)),
        button_rates=_button_rates(emitted),
        novel_combo_rate=_novel_rate(emitted, expert_bytes),
        novel_combos=_novel_list(emitted, expert_bytes),
        repeat_fraction=(repeats / decisions if decisions else 0.0),
        selection=selection,
        max_x_by_level=max_x_by_level,
        end_class=end_class,
        death_causes=death_causes,
    )


def evaluate_on_session(
    session: FceuxSession,
    policy: torch.nn.Module,
    starts: list[StartPoint],
    vocab: ActionVocab,
    *,
    seeds: int = 20,
    selection: str = "threshold",
    on_episode=None,
    **kwargs,
) -> dict:
    """Evaluate across start points and seeds on one persistent emulator."""
    deterministic = selection in ("greedy", "threshold")
    n_seeds = 1 if deterministic else seeds
    episodes: list[EpisodeResult] = []
    errors: list[str] = []
    started = time.perf_counter()
    for start in starts:
        for seed in range(n_seeds):
            try:
                ep = play_episode(
                    session, policy, start, vocab, seed=seed, selection=selection, **kwargs
                )
            except Exception as exc:
                errors.append(f"{start.label}@{start.frame} seed {seed}: {exc}"[:200])
                continue
            episodes.append(ep)
            if on_episode:
                on_episode(ep)
    summary = summarise_episodes(
        episodes, errors=errors, wall_seconds=time.perf_counter() - started
    )
    summary["selection"] = selection
    summary["seeds_per_start"] = n_seeds
    summary["n_starts"] = len(starts)
    summary["session"] = session.stats()
    return summary
