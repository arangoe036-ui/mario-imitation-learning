"""Live play in FCEUX: the policy drives the emulator, frame by frame.

Capture (stage 1) was one-way -- FCEUX replayed a movie and we recorded it. Live
play is a closed loop, so there are two FIFOs: Lua writes an observation and blocks
reading one action byte, Python decides, writes it back, and Lua applies it with
``joypad.set`` before advancing. Lua must ``flush()`` every frame or Python blocks
forever behind a buffered write.

The game is started for real, not from a savestate: the harness holds no input
through the title screen, presses Start on a fixed frame the way every SMB TAS does,
and hands control to the policy once the game reports the player has control. Left
to itself a cloned policy would never press Start -- the title screen is 40 of 1.2M
training frames -- and every episode would score zero for reasons that say nothing
about the policy.

Action selection
----------------
Per-frame independent sampling was the original approach and it was wrong: clearing the
second pipe in 1-1 needs A held for 13 consecutive frames, and independent sampling at
P(A)~8% gives that a probability of 5e-15. Every policy therefore stalled against that
pipe at x=594 and every score was identical -- the metric measured the level, not the
policy.

Three selection rules are now reported side by side:

``greedy``
    argmax. Deterministic, so one seed is all there is -- but it preserves holds, which
    makes it the primary metric.
``sticky``
    argmax, but with probability ``sticky_p`` repeat the previous action instead. The
    standard fix for deterministic games: it produces genuine variation across seeds
    *without* shredding multi-frame holds.
``temperature``
    Softmax sampling at low temperature, for comparison against the above.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..fceux_backend import (
    GD_HEADER,
    GD_HEIGHT,
    GD_LEN,
    GD_WIDTH,
    RAM_BYTES,
    FceuxError,
    _FifoReader,
    find_fceux,
)
from ..buttons import NES_BUTTON_ORDER, describe_action
from ..ram import DYING_STATES, PLAYER_STATE_NORMAL, read_smb
from ..replay import _resize_gray
from ..buttons import NES_BUTTON_BITS
from .tokens import LIVE_MASK, ActionVocab

#: Frame on which the harness presses Start. Matches the TAS convention.
START_FRAME = 40

#: Give up on an episode if the game never hands over control.
CONTROL_TIMEOUT_FRAMES = 400

#: Measured x of the two pipe faces in 1-1 where policies stall. A policy that never
#: jumps parks against pipe 1 at 435; one that hops but cannot hold A parks against the
#: taller pipe 2 at 594. Distance alone cannot tell "stuck at a pipe" from "a policy
#: that happens to score 594", so clearing each is reported as an explicit binary.
PIPE1_FACE_X = 435
PIPE2_FACE_X = 594
PIPE1_CLEARED_X = 470
PIPE2_CLEARED_X = 630

#: End an episode when x has not improved for this many in-control frames. The original
#: 5,000-frame budget burned ~4,400 frames per episode standing still.
DEFAULT_STALL_LIMIT = 300

#: Levels that can be reached by fast-forwarding an expert movie and then handing over.
DEFAULT_EVAL_LEVELS: tuple[str, ...] = ("1-1", "1-2", "2-1", "4-1")

_LUA = r"""
-- Live play: exchange one observation for one action byte, every frame.
local OBS      = %(obs)s
local ACT      = %(act)s
local N        = %(n_frames)d
local START    = %(start_frame)d
local TARGET_W = %(target_world)d   -- 0 = cold boot and press Start ourselves
local TARGET_S = %(target_stage)d

local function u32(n)
  return string.char(n %% 256,
                     math.floor(n / 256) %% 256,
                     math.floor(n / 65536) %% 256,
                     math.floor(n / 16777216) %% 256)
end

emu.speedmode("maximum")

local obs = io.open(OBS, "wb")
local act = io.open(ACT, "rb")
if obs == nil or act == nil then
  emu.print("tasdata: could not open live FIFOs")
  os.exit(1)
end

local function press(byte)
  joypad.set(1, {
    right  = bit.band(byte, 0x80) ~= 0,
    left   = bit.band(byte, 0x40) ~= 0,
    down   = bit.band(byte, 0x20) ~= 0,
    up     = bit.band(byte, 0x10) ~= 0,
    start  = bit.band(byte, 0x08) ~= 0,
    select = bit.band(byte, 0x04) ~= 0,
    B      = bit.band(byte, 0x02) ~= 0,
    A      = bit.band(byte, 0x01) ~= 0,
  })
end

local function exchange()
  local ram = memory.readbyterange(0x0000, 0x800)
  local s = gui.gdscreenshot()
  obs:write("TF", u32(emu.framecount()), ram, u32(#s), s)
  obs:flush()                         -- without this Python blocks forever
  local b = act:read(1)
  if b == nil then return false end
  press(string.byte(b))
  return true
end

if TARGET_W > 0 then
  -- An expert movie is playing (--playmov). Fast-forward to the start of the target
  -- level, then stop the movie and hand control to the policy. This gives a genuine
  -- mid-game start without needing savestate files, so one early obstacle cannot cap
  -- the whole measurement.
  local guard = 0
  while guard < 400000 do
    local w  = memory.readbyte(0x075f) + 1
    local st = memory.readbyte(0x075c) + 1
    local pg = memory.readbyte(0x0770)
    if w == TARGET_W and st == TARGET_S and pg == 1 then break end
    emu.frameadvance()
    guard = guard + 1
  end
  if guard >= 400000 then
    emu.print("tasdata: never reached target level")
    obs:close()
    os.exit(1)
  end
  movie.stop()
  local sent = 0
  while sent < N do
    if not exchange() then break end
    emu.frameadvance()
    sent = sent + 1
  end
else
  -- Cold boot. Pressing Start on a fixed frame is not reliable: if it lands in the
  -- wrong window the console ends up running the ATTRACT-MODE DEMO, which drives Mario
  -- from a canned input sequence in ROM. That looked like a competent policy -- x
  -- reached 3266 and the flagpole -- while our policy was sending literally nothing.
  -- OperMode (0x0770) alone does not catch it, so control is confirmed behaviourally:
  -- under real control with no input Mario stands still; in the demo he moves.
  local function xpos()
    return memory.readbyte(0x006d) * 256 + memory.readbyte(0x0086)
  end

  local have_control = false
  for attempt = 1, 8 do
    -- nudge Start until OperMode says a game is running
    for i = 1, 240 do
      if i %% 16 < 4 then joypad.set(1, {start = true}) end
      emu.frameadvance()
      if memory.readbyte(0x0770) == 1 then break end
    end
    -- ... then prove it is OUR game: hold nothing and require Mario to stay put.
    local x0 = xpos()
    local moved = false
    for i = 1, 30 do
      joypad.set(1, {})
      emu.frameadvance()
      if math.abs(xpos() - x0) > 2 then moved = true end
    end
    if not moved and memory.readbyte(0x0770) == 1 then
      have_control = true
      break
    end
  end
  if not have_control then
    emu.print("tasdata: never gained control (attract-mode demo?)")
    obs:close()
    os.exit(2)
  end

  local sent = 0
  while sent < N do
    if not exchange() then break end
    emu.frameadvance()
    sent = sent + 1
  end
end

obs:close()
os.exit(0)
"""


class EpisodeAborted(FceuxError):
    """The emulator failed to run the episode; retry rather than record it."""


def _send(fd: int, byte: int) -> None:
    """Write one action byte, turning an emulator death into a retryable abort.

    FCEUX occasionally loses its GPU context on launch ("software OpenGL backend")
    and dies mid-episode. That surfaced as a raw BrokenPipeError, which escaped the
    retry path and polluted the statistics with truncated episodes.
    """
    try:
        os.write(fd, bytes([byte]))
    except (BrokenPipeError, OSError) as exc:
        raise EpisodeAborted(f"emulator closed the action pipe: {exc}") from exc


@dataclass
class EpisodeResult:
    """Outcome of one live-play episode."""

    seed: int
    frames: int
    frames_survived: int
    furthest_level: str
    levels_reached: int
    furthest_x: int
    total_progress: int
    deaths: int
    ended: str
    #: Which level the episode started on ("1-1" for a cold boot).
    start_level: str = "1-1"
    #: Explicit binaries, because distance alone cannot distinguish "stuck at the
    #: pipe" from "a policy whose score happens to be 594".
    cleared_pipe1: bool = False
    cleared_pipe2: bool = False
    #: Longest run of frames on which A was actually held. The pipe needs 13.
    longest_a_hold: int = 0
    #: Number of separate A presses (onsets), not frames held.
    a_presses: int = 0
    #: Per-button hold-length distribution. Press *rate* cannot distinguish one long
    #: hold from many short taps, which is exactly the failure mode being watched:
    #: in SMB you must RELEASE A to jump again, so a stuck hold makes every later
    #: jump impossible.
    hold_stats: dict[str, dict] = field(default_factory=dict)
    #: Realized per-button press rate over the episode.
    button_rates: dict[str, float] = field(default_factory=dict)
    #: Emitted combinations that appear nowhere in the expert corpus.
    novel_combo_rate: float = 0.0
    novel_combos: list[str] = field(default_factory=list)
    #: How often the chosen action equalled the previous one (copycat symptom).
    repeat_fraction: float = 0.0
    selection: str = "greedy"
    max_x_by_level: dict[str, int] = field(default_factory=dict)
    token_counts: dict[str, int] = field(default_factory=dict)
    #: How the episode ended, as one of a fixed taxonomy: enemy_contact, pit, timer,
    #: stuck_terrain, game_over, budget_reached. Distance alone conflates these and they
    #: call for opposite fixes.
    end_class: str = "unknown"
    death_causes: list[str] = field(default_factory=list)

    def row(self) -> str:
        return (
            f"  seed {self.seed:3d} {self.start_level}->{self.furthest_level:>4s} "
            f"x={self.furthest_x:5d} pipe1={'Y' if self.cleared_pipe1 else 'n'} "
            f"pipe2={'Y' if self.cleared_pipe2 else 'n'} Ahold={self.longest_a_hold:3d} "
            f"deaths={self.deaths:2d} f={self.frames_survived:5d} ({self.ended})"
        )


class _FfmpegWriter:
    """Pipe raw RGB frames to ffmpeg. NES output is 60.0988 fps."""

    def __init__(self, path: Path | str, width: int, height: int, fps: float = 60.0988):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", f"{fps:.4f}",
                "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(self.path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, rgb: np.ndarray) -> None:
        if self.process.stdin:
            try:
                self.process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
            except BrokenPipeError:  # pragma: no cover
                pass

    def close(self) -> None:
        if self.process.stdin:
            try:
                self.process.stdin.close()
            except BrokenPipeError:  # pragma: no cover
                pass
        try:
            self.process.wait(timeout=120)
        except subprocess.TimeoutExpired:  # pragma: no cover
            self.process.kill()


def _hold_stats(
    runs: dict[str, list[int]], open_runs: dict[str, int], frames: int
) -> dict[str, dict]:
    """Presses per 1,000 frames plus median/p90/max hold length, per button."""
    out: dict[str, dict] = {}
    for name in NES_BUTTON_ORDER:
        lengths = list(runs.get(name, []))
        if open_runs.get(name):
            lengths.append(open_runs[name])  # a hold still open at episode end counts
        if not lengths:
            out[name] = {"presses": 0, "per_1000_frames": 0.0, "median": 0.0,
                         "p90": 0.0, "max": 0, "frames_held": 0}
            continue
        arr = np.array(lengths, dtype=np.int64)
        out[name] = {
            "presses": int(arr.size),
            "per_1000_frames": float(arr.size * 1000.0 / frames),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "max": int(arr.max()),
            "frames_held": int(arr.sum()),
        }
    return out


def _button_rates(emitted: dict[int, int]) -> dict[str, float]:
    """Fraction of emitted frames on which each button was held."""
    total = sum(emitted.values())
    if not total:
        return {}
    out: dict[str, float] = {}
    for name in NES_BUTTON_ORDER:
        bit = NES_BUTTON_BITS[name]
        out[name] = sum(c for b, c in emitted.items() if b & bit) / total
    return out


def _novel_rate(emitted: dict[int, int], expert: set[int] | None) -> float:
    """Fraction of frames whose emitted combination appears nowhere in expert data.

    Independent per-button outputs can invent combinations a categorical head could
    not, so this is checked rather than assumed.
    """
    if not expert:
        return 0.0
    total = sum(emitted.values()) or 1
    return sum(c for b, c in emitted.items() if b not in expert) / total


def _novel_list(emitted: dict[int, int], expert: set[int] | None) -> list[str]:
    if not expert:
        return []
    novel = [(b, c) for b, c in emitted.items() if b not in expert]
    novel.sort(key=lambda kv: -kv[1])
    return [f"{describe_action(b)} x{c}" for b, c in novel[:8]]


def _lua_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_live_lua(
    *,
    obs: Path,
    act: Path,
    n_frames: int,
    start_frame: int,
    target_world: int = 0,
    target_stage: int = 0,
) -> str:
    return _LUA % {
        "obs": _lua_string(str(obs)),
        "act": _lua_string(str(act)),
        "n_frames": n_frames,
        "start_frame": start_frame,
        "target_world": target_world,
        "target_stage": target_stage,
    }


class LivePlayer:
    """Runs episodes of a policy in FCEUX."""

    def __init__(
        self,
        rom_path: Path | str,
        vocab: ActionVocab,
        *,
        binary: str | Path = "fceux",
        stack: int = 4,
        max_frames: int = 3000,
        device: torch.device | str = "cpu",
        mask_live: bool = True,
        extra_args: tuple[str, ...] = (),
        stall_limit: int = DEFAULT_STALL_LIMIT,
        n_prev_actions: int = 0,
        head_type: str = "categorical",
        thresholds: np.ndarray | None = None,
        expert_bytes: set[int] | None = None,
        expert_movie: Path | str | None = None,
        min_episode_frames: int = 120,
    ) -> None:
        self.rom_path = Path(rom_path)
        self.vocab = vocab
        self.binary = find_fceux(binary)
        self.stack = stack
        self.max_frames = max_frames
        self.device = torch.device(device)
        self.mask_live = mask_live
        self.extra_args = list(extra_args)
        self.stall_limit = stall_limit
        self.n_prev_actions = n_prev_actions
        self.head_type = head_type
        self.thresholds = (
            np.asarray(thresholds, dtype=np.float32) if thresholds is not None else None
        )
        if head_type == "bernoulli" and self.thresholds is None:
            raise ValueError(
                "a Bernoulli policy needs calibrated per-button thresholds; 0.5 would "
                "never fire A (mass at onsets is ~19%)"
            )
        self.expert_bytes = expert_bytes
        self.expert_movie = Path(expert_movie) if expert_movie else None
        self.min_episode_frames = min_episode_frames

    # -- one episode --------------------------------------------------------- #

    def play(
        self,
        policy: torch.nn.Module,
        *,
        seed: int = 0,
        selection: str = "greedy",
        temperature: float = 1.0,
        sticky_p: float = 0.25,
        level: str = "1-1",
        max_frames: int | None = None,
        video_path: Path | str | None = None,
        obs_video_path: Path | str | None = None,
        telemetry_path: Path | str | None = None,
    ) -> EpisodeResult:
        """Play one episode, sampling actions from the policy.

        Args:
            video_path: write the emulator's own 256x240 output to this mp4.
            obs_video_path: write the 84x84 stack the *model* sees, upscaled. If the
                policy is effectively blind because of a preprocessing bug, this is
                where it shows up -- you can watch the input rather than trust it.
            telemetry_path: per-frame CSV of x, level, chosen token and state.
            selection: ``greedy``, ``sticky`` or ``temperature``.
            level: which level to start on; anything but ``1-1`` fast-forwards an expert
                movie to that level and then hands over.
        """
        categorical_rules = ("greedy", "sticky", "temperature")
        bernoulli_rules = ("threshold", "sticky", "sample")
        allowed = bernoulli_rules if self.head_type == "bernoulli" else categorical_rules
        if selection not in allowed:
            raise ValueError(
                f"unknown selection rule {selection!r} for a {self.head_type} head; "
                f"expected one of {allowed}"
            )
        # Every level, 1-1 included, is entered by fast-forwarding the expert movie to
        # that level's real first frame and then stopping the movie. The cold-boot
        # "press Start on frame 40" path is kept for reference but is NOT used when an
        # expert movie is available: mistimed Start presses leave the console running
        # the attract-mode demo, which drives Mario from canned ROM input. That produced
        # a do-nothing policy "reaching the flagpole at x=3266". Fast-forwarding is
        # deterministic and cannot be fooled.
        target_world, target_stage = 0, 0
        if self.expert_movie is not None:
            w, st = level.split("-")
            target_world, target_stage = int(w), int(st)
        elif level != "1-1":
            raise FceuxError(
                f"starting at {level} needs expert_movie= to fast-forward to it"
            )
        n_frames = max_frames or self.max_frames
        workdir = Path(tempfile.mkdtemp(prefix="tasdata-live-"))
        obs_fifo, act_fifo = workdir / "obs.fifo", workdir / "act.fifo"
        os.mkfifo(obs_fifo)
        os.mkfifo(act_fifo)
        lua = workdir / "live.lua"
        lua.write_text(
            build_live_lua(
                obs=obs_fifo,
                act=act_fifo,
                n_frames=n_frames,
                start_frame=START_FRAME,
                target_world=target_world,
                target_stage=target_stage,
            )
        )
        log = (workdir / "fceux.log").open("wb")
        # A brief pause between launches: back-to-back FCEUX starts are what provoke
        # the software-OpenGL fallback that kills an episode.
        time.sleep(0.25)

        rng = np.random.default_rng(seed)
        policy.eval()

        video = _FfmpegWriter(video_path, 256, 240) if video_path else None
        # The observation video shows the 4-frame stack side by side, nearest-scaled
        # x3, so single-pixel detail survives and stale frames are obvious.
        obs_video = (
            _FfmpegWriter(obs_video_path, 84 * self.stack * 3, 84 * 3)
            if obs_video_path
            else None
        )
        telemetry = open(telemetry_path, "w") if telemetry_path else None
        if telemetry:
            telemetry.write("frame,world,stage,x,y,player_state,pregame,token,name,byte\n")

        window = np.zeros((self.stack, 84, 84), dtype=np.uint8)
        filled = 0
        deaths = 0
        prev_dying = False
        max_x_by_level: dict[str, int] = {}
        order: list[str] = []
        token_counts = np.zeros(self.vocab.size, dtype=np.int64)
        emitted_bytes: dict[int, int] = {}
        a_presses = 0
        last_frame = 0
        first_frame = None
        ended = "reached frame budget"
        prev_tokens: list[int] = []
        prev_byte: int | None = None
        prev_token: int | None = None
        repeats = 0
        decisions = 0
        a_hold = 0
        longest_a_hold = 0
        # Completed hold runs per button, plus the run currently open.
        hold_runs: dict[str, list[int]] = {n: [] for n in NES_BUTTON_ORDER}
        hold_open: dict[str, int] = {n: 0 for n in NES_BUTTON_ORDER}
        best_x = -1
        frames_since_progress = 0

        process = subprocess.Popen(
            [
                str(self.binary),
                "--no-config", "1",
                "--sound", "0",
                "--opposite-directionals", "1",
                *(["--playmov", str(self.expert_movie)] if target_world else []),
                *self.extra_args,
                "--loadlua", str(lua),
                str(self.rom_path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        # Open the read end non-blocking first, then the write end; Lua opens them
        # in the same order, so neither side deadlocks on the other.
        reader = _FifoReader(obs_fifo, process)
        try:
            act_fd = os.open(str(act_fifo), os.O_WRONLY)
        except OSError as exc:  # pragma: no cover
            reader.close()
            process.terminate()
            raise FceuxError(f"could not open action FIFO: {exc}") from exc

        try:
            with torch.no_grad():
                for step in range(n_frames):
                    header = reader.read_exact(6, 120.0 if step == 0 else 60.0)
                    if header is None:
                        ended = "emulator closed the stream"
                        break
                    if header[:2] != b"TF":
                        raise FceuxError(f"live stream desynchronised at step {step}")
                    frame_no = int.from_bytes(header[2:6], "little")
                    ram_bytes = reader.read_exact(RAM_BYTES, 60.0)
                    if ram_bytes is None:
                        ended = "emulator closed the stream"
                        break
                    ram = np.frombuffer(ram_bytes, dtype=np.uint8)
                    screen_len = int.from_bytes(
                        reader.read_exact(4, 60.0) or b"\0\0\0\0", "little"
                    )
                    if screen_len != GD_LEN:
                        raise FceuxError(
                            f"unexpected screen payload {screen_len} at step {step}"
                        )
                    blob = reader.read_exact(screen_len, 60.0)
                    if blob is None:
                        ended = "emulator closed the stream"
                        break

                    px = np.frombuffer(blob[GD_HEADER:], dtype=np.uint8).reshape(
                        GD_HEIGHT, GD_WIDTH, 4
                    )
                    rgb = np.ascontiguousarray(px[:, :, 1:])
                    obs84 = _resize_gray(rgb, (84, 84))
                    window = np.roll(window, -1, axis=0)
                    window[-1] = obs84
                    filled = min(filled + 1, self.stack)
                    if video is not None:
                        video.write(rgb)
                    if obs_video is not None:
                        strip = np.concatenate(list(window), axis=1)  # (84, 84*stack)
                        big = np.repeat(np.repeat(strip, 3, axis=0), 3, axis=1)
                        obs_video.write(np.stack([big] * 3, axis=-1))

                    state = read_smb(ram, frame_no)
                    last_frame = frame_no

                    # Progress bookkeeping, only while the player has control.
                    if state.pregame == 1 and 1 <= state.world <= 8 and 1 <= state.stage <= 4:
                        label = state.label()
                        if label not in max_x_by_level:
                            order.append(label)
                        max_x_by_level[label] = max(
                            max_x_by_level.get(label, 0), state.x_position
                        )
                    dying = state.player_state in DYING_STATES
                    if dying and not prev_dying:
                        deaths += 1
                    prev_dying = dying

                    if state.pregame == 2:
                        ended = "game over"
                        _send(act_fd, 0)
                        break
                    if (
                        target_world == 0
                        and frame_no > START_FRAME + CONTROL_TIMEOUT_FRAMES
                        and not max_x_by_level
                    ):
                        ended = "never gained control"
                        _send(act_fd, 0)
                        break

                    # -- early stop: no forward progress for stall_limit frames ------
                    if state.pregame == 1 and state.player_state == PLAYER_STATE_NORMAL:
                        if state.x_position > best_x:
                            best_x = state.x_position
                            frames_since_progress = 0
                        else:
                            frames_since_progress += 1
                        if self.stall_limit and frames_since_progress > self.stall_limit:
                            ended = f"no progress for {self.stall_limit} frames"
                            _send(act_fd, 0)
                            break

                    # -- decide -----------------------------------------------------
                    batch = torch.from_numpy(window[None]).float().div_(255.0)
                    prev_arg = None
                    if self.n_prev_actions > 0:
                        k = self.n_prev_actions
                        hist = prev_tokens[-k:] or [self.vocab.size]
                        pad = [self.vocab.size] * (k - len(hist))
                        prev_arg = torch.tensor([pad + hist], dtype=torch.long)
                    raw = policy(batch.to(self.device), prev_arg)[0]
                    logits = raw.float().cpu().numpy()

                    if self.head_type == "bernoulli":
                        probs = 1.0 / (1.0 + np.exp(-logits))
                        if selection == "sample":
                            bits = (rng.random(probs.shape) < probs).astype(np.float32)
                        else:
                            bits = (probs > self.thresholds).astype(np.float32)
                        byte = 0
                        for j, bname in enumerate(NES_BUTTON_ORDER):
                            if bits[j] > 0:
                                byte |= NES_BUTTON_BITS[bname]
                        if selection == "sticky" and prev_byte is not None:
                            if rng.random() < sticky_p:
                                byte = prev_byte
                        decisions += 1
                        if prev_byte is not None and byte == prev_byte:
                            repeats += 1
                        emitted_bytes[byte] = emitted_bytes.get(byte, 0) + 1
                        token = -1  # not applicable to a per-button head
                    else:
                        greedy_token = int(logits.argmax())
                        if selection == "greedy":
                            token = greedy_token
                        elif selection == "sticky":
                            # Repeat the previous action with probability sticky_p.
                            # Variation across seeds without shredding held buttons.
                            if prev_token is not None and rng.random() < sticky_p:
                                token = prev_token
                            else:
                                token = greedy_token
                        else:
                            z = logits / max(temperature, 1e-6)
                            z -= z.max()
                            pdist = np.exp(z)
                            pdist /= pdist.sum()
                            token = int(rng.choice(len(pdist), p=pdist))

                        decisions += 1
                        if prev_token is not None and token == prev_token:
                            repeats += 1
                        token_counts[token] += 1
                    if self.head_type != "bernoulli":
                        byte = self.vocab.decode_byte(token, mask_live=self.mask_live)
                    elif self.mask_live:
                        byte &= LIVE_MASK
                    if byte & 0x01:
                        if a_hold == 0:
                            a_presses += 1
                        a_hold += 1
                        longest_a_hold = max(longest_a_hold, a_hold)
                    else:
                        a_hold = 0
                    for bname in NES_BUTTON_ORDER:
                        if byte & NES_BUTTON_BITS[bname]:
                            hold_open[bname] += 1
                        elif hold_open[bname]:
                            hold_runs[bname].append(hold_open[bname])
                            hold_open[bname] = 0
                    prev_token = token
                    prev_byte = byte
                    prev_tokens.append(token)
                    if len(prev_tokens) > 64:
                        del prev_tokens[:-16]
                    if first_frame is None:
                        first_frame = frame_no
                    if telemetry:
                        telemetry.write(
                            f"{frame_no},{state.world},{state.stage},{state.x_position},"
                            f"{state.y_position},{state.player_state},{state.pregame},"
                            # Token names contain commas (e.g. "RARE(43 combos, ..."),
                            # so the name field is quoted.
                            f'{token},"{self.vocab.names[token]}",{byte}\n'
                        )
                    _send(act_fd, byte)
        finally:
            try:
                os.close(act_fd)
            except OSError:
                pass
            reader.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:  # pragma: no cover
                    process.kill()
            log.close()
            if video is not None:
                video.close()
            if obs_video is not None:
                obs_video.close()
            if telemetry is not None:
                telemetry.close()
            shutil.rmtree(workdir, ignore_errors=True)

        furthest = order[-1] if order else "-"
        start_x = max_x_by_level.get(level, 0)
        # A flaky emulator start must not masquerade as a bad policy: an episode that
        # dies before the game even gets going is discarded and retried, not averaged in.
        if ended == "emulator closed the stream" and (
            len(order) == 0
            or (last_frame - (first_frame or last_frame)) < self.min_episode_frames
        ):
            raise EpisodeAborted(
                f"episode ended after {last_frame - (first_frame or last_frame)} frames "
                f"with '{ended}' (level {level}, seed {seed}); treating as a flake"
            )
        return EpisodeResult(
            seed=seed,
            frames=last_frame,
            frames_survived=max(0, last_frame - (first_frame or last_frame)),
            furthest_level=furthest,
            levels_reached=len(order),
            furthest_x=max_x_by_level.get(furthest, 0),
            total_progress=int(sum(max_x_by_level.values())),
            deaths=deaths,
            ended=ended,
            start_level=level,
            # Only meaningful for a 1-1 start; other levels begin past both pipes.
            cleared_pipe1=bool(level == "1-1" and start_x > PIPE1_CLEARED_X),
            cleared_pipe2=bool(level == "1-1" and start_x > PIPE2_CLEARED_X),
            longest_a_hold=longest_a_hold,
            a_presses=a_presses,
            hold_stats=_hold_stats(hold_runs, hold_open, max(1, last_frame - (first_frame or last_frame))),
            button_rates=_button_rates(emitted_bytes),
            novel_combo_rate=_novel_rate(emitted_bytes, self.expert_bytes),
            novel_combos=_novel_list(emitted_bytes, self.expert_bytes),
            repeat_fraction=(repeats / decisions if decisions else 0.0),
            selection=selection,
            max_x_by_level=max_x_by_level,
            token_counts={
                self.vocab.names[i]: int(c)
                for i, c in enumerate(token_counts)
                if c
            },
        )

    # -- many episodes ------------------------------------------------------- #

    def evaluate(
        self,
        policy: torch.nn.Module,
        *,
        seeds: int = 20,
        selection: str = "greedy",
        temperature: float = 1.0,
        sticky_p: float = 0.25,
        levels: tuple[str, ...] = ("1-1",),
        max_frames: int | None = None,
        retries: int = 3,
        workers: int = 1,
        on_episode=None,
    ) -> dict:
        """Play episodes across seeds and levels, retrying emulator flakes.

        ``greedy`` is deterministic, so it is run once per level rather than once per
        seed -- extra seeds would be identical trajectories. The other rules get the
        full seed sweep.
        """
        episodes: list[EpisodeResult] = []
        errors: list[str] = []
        aborts = 0
        started = time.perf_counter()
        # Deterministic rules need one episode per level; extra seeds would be identical
        # trajectories. "threshold" is the Bernoulli analogue of greedy.
        deterministic = selection in ("greedy", "threshold")
        n_seeds = 1 if deterministic else seeds
        jobs = [(lvl, sd) for lvl in levels for sd in range(n_seeds)]

        def run_job(job):
            level, seed = job
            local_aborts = 0
            for attempt in range(retries):
                try:
                    return self.play(
                        policy, seed=seed, selection=selection, temperature=temperature,
                        sticky_p=sticky_p, level=level, max_frames=max_frames,
                    ), local_aborts, None
                except EpisodeAborted as exc:
                    local_aborts += 1
                    if attempt == retries - 1:
                        return None, local_aborts, f"{level} seed {seed}: {exc}"[:200]
                    time.sleep(1.0)
                except Exception as exc:
                    return None, local_aborts, (
                        f"{level} seed {seed}: {type(exc).__name__}: {exc}"[:200]
                    )
            return None, local_aborts, None

        if workers > 1 and len(jobs) > 1:
            # Emulators are independent processes, so episodes parallelise cleanly.
            # Threads are enough: each one is mostly blocked on its FCEUX child.
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(run_job, jobs))
        else:
            results = [run_job(j) for j in jobs]

        for ep, n_ab, err in results:
            aborts += n_ab
            if err:
                errors.append(err)
            if ep is not None:
                episodes.append(ep)
                if on_episode:
                    on_episode(ep)
        summary = summarise_episodes(
            episodes, errors=errors, wall_seconds=time.perf_counter() - started
        )
        summary["selection"] = selection
        summary["temperature"] = temperature if selection == "temperature" else None
        summary["levels"] = list(levels)
        summary["seeds_per_level"] = n_seeds
        summary["retried_flakes"] = aborts
        summary["workers"] = workers
        return summary


def summarise_episodes(
    episodes: list[EpisodeResult], *, errors: list[str] | None = None, wall_seconds: float = 0.0
) -> dict:
    """Distribution over episodes -- never a single number."""

    def stats(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        arr = np.array(values, dtype=float)
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "max": float(arr.max()),
        }

    levels = [e.furthest_level for e in episodes]
    one_one = [e for e in episodes if e.start_level == "1-1"]
    return {
        "n_episodes": len(episodes),
        "wall_seconds": round(wall_seconds, 1),
        "furthest_x": stats([e.furthest_x for e in episodes]),
        "total_progress": stats([e.total_progress for e in episodes]),
        "levels_reached": stats([e.levels_reached for e in episodes]),
        "deaths": stats([e.deaths for e in episodes]),
        "frames_survived": stats([e.frames_survived for e in episodes]),
        # Rates are over 1-1 starts only: a 2-1 start begins past both pipes, so
        # counting it as "cleared" would inflate the number to no purpose.
        "cleared_pipe1_rate": (
            sum(e.cleared_pipe1 for e in one_one) / len(one_one) if one_one else None
        ),
        "cleared_pipe2_rate": (
            sum(e.cleared_pipe2 for e in one_one) / len(one_one) if one_one else None
        ),
        "n_episodes_from_1_1": len(one_one),
        "longest_a_hold": stats([e.longest_a_hold for e in episodes]),
        "a_presses": stats([e.a_presses for e in episodes]),
        "hold_stats": {
            n: {
                "presses_per_1000_frames": float(
                    np.mean([e.hold_stats.get(n, {}).get("per_1000_frames", 0.0) for e in episodes])
                ),
                "median": float(
                    np.median([e.hold_stats.get(n, {}).get("median", 0.0) for e in episodes])
                ),
                "p90": float(
                    np.mean([e.hold_stats.get(n, {}).get("p90", 0.0) for e in episodes])
                ),
                "max": int(
                    max([e.hold_stats.get(n, {}).get("max", 0) for e in episodes], default=0)
                ),
                "presses_total": int(
                    sum(e.hold_stats.get(n, {}).get("presses", 0) for e in episodes)
                ),
            }
            for n in NES_BUTTON_ORDER
        } if episodes else {},
        "novel_combo_rate": stats([e.novel_combo_rate for e in episodes]),
        "novel_combos": sorted({c for e in episodes for c in e.novel_combos})[:12],
        "button_rates": {
            n: float(np.mean([e.button_rates.get(n, 0.0) for e in episodes]))
            for n in NES_BUTTON_ORDER
        } if episodes else {},
        "repeat_fraction": stats([e.repeat_fraction for e in episodes]),
        "by_start_level": {
            lvl: {
                "n": sum(1 for e in episodes if e.start_level == lvl),
                "furthest_x_median": float(
                    np.median([e.furthest_x for e in episodes if e.start_level == lvl])
                ) if any(e.start_level == lvl for e in episodes) else None,
                "levels_reached_median": float(
                    np.median([e.levels_reached for e in episodes if e.start_level == lvl])
                ) if any(e.start_level == lvl for e in episodes) else None,
            }
            for lvl in sorted({e.start_level for e in episodes})
        },
        "furthest_level_mode": max(set(levels), key=levels.count) if levels else "-",
        "furthest_level_best": (
            max(levels, key=lambda s: tuple(int(p) for p in s.split("-")))
            if any(l != "-" for l in levels)
            else "-"
        ),
        "level_histogram": {l: levels.count(l) for l in sorted(set(levels))},
        "ended_histogram": {
            r: sum(1 for e in episodes if e.ended == r)
            for r in sorted({e.ended for e in episodes})
        },
        "episodes": [asdict(e) for e in episodes],
        "errors": errors or [],
    }
