"""Tests for the persistent-FCEUX process model.

These are the only tests that launch a real emulator. They are what the previous
process-per-episode design could not support: every episode spawned FCEUX, the suite
exhausted the macOS IOSurface client limit, and eight tests failed for environmental
reasons rather than code ones. One session per test keeps that bounded.

The bugs each test pins down are real ones that were hit:

* a savestate captured during movie playback restores the movie's *playback state*, so the
  recorded inputs kept driving Mario and the policy's inputs were ignored;
* savestates were addressed by frame number, which overflows the uint16 command argument
  at 67,117 frames;
* nothing enforced the one-emulator cap, so a parallel evaluation silently reintroduced
  the OpenGL race.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tasdata.bc.session import EmulatorLock, FceuxSession, TooManyEmulators
from tasdata.bc.statelib import frame_hash, load_index, ram_hash
from tasdata.buttons import NES_BUTTON_BITS, NES_BUTTON_ORDER
from tasdata.ram import read_smb

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "smb.nes"
MOVIE = ROOT / "data/movies/happylee_mars608-smb-warpless.fm2"
INDEX = ROOT / "data/state_index.json"

needs_fceux = pytest.mark.skipif(
    shutil.which("fceux") is None or not ROM.exists() or not MOVIE.exists(),
    reason="needs a real fceux binary, the NTSC ROM and the expert movie",
)


@pytest.fixture(scope="module")
def level_starts() -> list:
    if not INDEX.exists():
        pytest.skip("state index not built")
    _, points = load_index(INDEX)
    return [p for p in points if p.kind == "level_start"][:3]


@pytest.fixture(scope="module")
def session(level_starts):
    """One emulator for the whole module -- the entire point of the rewire."""
    frames = sorted({p.frame for p in level_starts})
    with FceuxSession(ROM, MOVIE, frames) as s:
        yield s


class TestEmulatorLock:
    def test_second_acquire_is_refused(self, tmp_path: Path):
        first = EmulatorLock(tmp_path / "lock")
        first.acquire()
        try:
            second = EmulatorLock(tmp_path / "lock")
            with pytest.raises(TooManyEmulators):
                second.acquire()
        finally:
            first.release()

    def test_release_allows_a_later_acquire(self, tmp_path: Path):
        first = EmulatorLock(tmp_path / "lock")
        first.acquire()
        first.release()
        second = EmulatorLock(tmp_path / "lock")
        second.acquire()  # must not raise
        second.release()


class TestOrdinalAddressing:
    def test_frame_numbers_would_overflow_the_argument(self):
        # The command argument is a uint16; the warpless movie is 67,117 frames. Frames are
        # therefore addressed by ordinal, and this is why.
        assert 67_117 > 0xFFFF


@needs_fceux
class TestSession:
    def test_one_process_serves_every_reset(self, session, level_starts):
        pid = session._proc.pid
        for point in level_starts:
            session.reset(point.frame)
        assert session._proc.pid == pid
        assert session._proc.poll() is None

    def test_reset_restores_the_indexed_state(self, session, level_starts):
        for point in level_starts:
            obs = session.reset(point.frame)
            state = read_smb(obs.ram, obs.framecount)
            assert state.x_position == point.x
            assert (state.world, state.stage) == (point.world, point.stage)

    def test_reset_is_deterministic_in_ram_and_pixels(self, session, level_starts):
        point = level_starts[0]
        first = session.reset(point.frame)
        # Perturb the emulator, then come back to the same state.
        for _ in range(30):
            session.step(NES_BUTTON_BITS["Right"])
        second = session.reset(point.frame)
        assert ram_hash(first.ram) == ram_hash(second.ram)
        assert frame_hash(first.rgb) == frame_hash(second.rgb)

    def test_indexed_hashes_still_match(self, session, level_starts):
        for point in level_starts:
            if point.ram_hash is None:
                continue
            obs = session.reset(point.frame)
            assert ram_hash(obs.ram) == point.ram_hash
            assert frame_hash(obs.rgb) == point.frame_hash

    def test_policy_inputs_drive_mario_not_the_movie(self, session, level_starts):
        """The movie must be stopped after a load, or the recording plays instead."""
        point = level_starts[0]

        session.reset(point.frame)
        for _ in range(40):
            idle = session.step(0)

        session.reset(point.frame)
        for _ in range(40):
            running = session.step(NES_BUTTON_BITS["Right"] | NES_BUTTON_BITS["B"])

        idle_x = read_smb(idle.ram, idle.framecount).x_position
        running_x = read_smb(running.ram, running.framecount).x_position
        assert running_x > idle_x, (
            "holding Right+B did not move Mario further than doing nothing, "
            "so the emulator is not taking our input"
        )

    def test_step_advances_one_frame(self, session, level_starts):
        obs = session.reset(level_starts[0].frame)
        before = obs.framecount
        after = session.step(0).framecount
        assert after == before + 1

    def test_observation_shapes(self, session, level_starts):
        obs = session.reset(level_starts[0].frame)
        assert obs.ram.shape == (2048,)
        assert obs.ram.dtype == np.uint8
        assert obs.rgb.ndim == 3


@needs_fceux
class TestSessionPlayer:
    def test_episode_runs_and_reports_progress(self, session, level_starts):
        import torch

        from tasdata.bc.session_player import play_episode
        from tasdata.bc.tokens import ActionVocab

        vocab_path = ROOT / "data/action_vocab.json"
        if not vocab_path.exists():
            pytest.skip("action vocabulary not built")
        vocab = ActionVocab.load(vocab_path)

        class AlwaysRunRight(torch.nn.Module):
            """Stand-in policy: logits that clear any sane threshold on Right and B.

            Indexed by name deliberately. The head's output order is NES_BUTTON_ORDER
            (Up, Down, Left, Right, Start, Select, B, A), not the controller's bit order,
            so hard-coding column 0 as "Right" presses Up instead and Mario never moves.
            """

            def forward(self, obs):
                out = torch.full((obs.shape[0], 8), -9.0)
                for name in ("Right", "B"):
                    out[:, NES_BUTTON_ORDER.index(name)] = 9.0
                return out

        start = next(p for p in level_starts if p.label == "1-1")
        result = play_episode(
            session,
            AlwaysRunRight(),
            start,
            vocab,
            thresholds=np.full(8, 0.5),
            head_type="bernoulli",
            max_frames=200,
        )
        assert result.frames > 0
        assert result.furthest_x > start.x
        assert result.button_rates["Right"] == pytest.approx(1.0)
        assert result.button_rates["A"] == 0.0
        assert result.start_level == "1-1"


@needs_fceux
def test_index_is_json_and_carries_both_hashes():
    data = json.loads(INDEX.read_text())
    points = data["points"]
    assert points, "index is empty"
    assert all(p["ram_hash"] for p in points), "a state has no RAM hash"
    assert all(p["frame_hash"] for p in points), "a state has no frame hash"
