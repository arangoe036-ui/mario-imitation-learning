"""Behavioural-cloning tests: vocabulary, dataset, model, baselines, report.

Live play needs FCEUX and a window, so it is marked ``fceux`` like the capture
tests. Everything else runs on synthetic arrays.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tasdata.bc.baselines import (
    ConstantPolicy,
    MarginalPolicy,
    evaluate_trivial_baselines,
    score_predictions,
    token_for_buttons,
)
from tasdata.bc.data import FrameStackDataset
from tasdata.bc.model import PolicyConfig, build_policy
from tasdata.bc.report import build_summary
from tasdata.bc.tokens import (
    DEFAULT_RARE_THRESHOLD,
    LIVE_MASK,
    ActionVocab,
    build_vocab,
)
from tasdata.bc.train import TrainConfig, load_checkpoint, save_checkpoint
from tasdata.buttons import NES_BUTTON_BITS
from tasdata.dataset import LoadedRun

RIGHT, LEFT, B, A = (
    NES_BUTTON_BITS["Right"],
    NES_BUTTON_BITS["Left"],
    NES_BUTTON_BITS["B"],
    NES_BUTTON_BITS["A"],
)
START, SELECT = NES_BUTTON_BITS["Start"], NES_BUTTON_BITS["Select"]


def sample_actions() -> np.ndarray:
    """Three common bytes plus a long tail below the threshold."""
    common = (
        [0] * 500
        + [RIGHT | B] * 400
        + [B] * 150
        + [RIGHT] * 120
    )
    rare = []
    for byte in (LEFT | A, RIGHT | LEFT, START, SELECT, A | B | LEFT):
        rare += [byte] * 5
    return np.array(common + rare, dtype=np.uint8)


class TestBuildVocab:
    def test_folds_the_rare_tail(self):
        vocab = build_vocab([sample_actions()], threshold=100)
        # 4 frequent bytes + RARE
        assert vocab.size == 5
        assert vocab.n_rare_members == 5
        assert vocab.names[vocab.rare_token].startswith("RARE")

    def test_frequent_tokens_ordered_by_frequency(self):
        vocab = build_vocab([sample_actions()], threshold=100)
        assert vocab.token_to_byte[0] == 0
        assert vocab.token_to_byte[1] == RIGHT | B
        assert vocab.counts[0] > vocab.counts[1] > vocab.counts[2]

    def test_rare_exists_even_with_no_rare_bytes(self):
        """Token ids must be stable whether or not a tail happens to exist."""
        vocab = build_vocab([np.zeros(500, np.uint8)], threshold=10)
        assert vocab.names[vocab.rare_token].startswith("RARE")
        assert vocab.n_rare_members == 0

    def test_encode_maps_rare_bytes_to_rare_token(self):
        vocab = build_vocab([sample_actions()], threshold=100)
        tokens = vocab.encode(np.array([0, RIGHT | B, START], np.uint8))
        assert tokens[0] == 0
        assert tokens[1] == 1
        assert tokens[2] == vocab.rare_token

    def test_unseen_byte_maps_to_rare(self):
        vocab = build_vocab([sample_actions()], threshold=100)
        assert int(vocab.encode(np.array([0x77], np.uint8))[0]) == vocab.rare_token

    def test_encode_is_vectorised_and_shape_preserving(self):
        vocab = build_vocab([sample_actions()], threshold=100)
        actions = sample_actions()
        assert vocab.encode(actions).shape == actions.shape

    def test_threshold_of_one_keeps_everything(self):
        vocab = build_vocab([sample_actions()], threshold=1)
        assert vocab.n_rare_members == 0
        assert vocab.size == 9 + 1  # 9 distinct bytes plus the empty RARE slot

    def test_default_threshold(self):
        assert DEFAULT_RARE_THRESHOLD == 100


class TestVocabPersistence:
    def test_roundtrip(self, tmp_path: Path):
        vocab = build_vocab([sample_actions()], threshold=100)
        path = vocab.save(tmp_path / "vocab.json")
        loaded = ActionVocab.load(path)
        assert loaded.size == vocab.size
        assert loaded.names == vocab.names
        assert np.array_equal(loaded.byte_to_token, vocab.byte_to_token)

    def test_loaded_vocab_encodes_identically(self, tmp_path: Path):
        vocab = build_vocab([sample_actions()], threshold=100)
        loaded = ActionVocab.load(vocab.save(tmp_path / "v.json"))
        actions = sample_actions()
        assert np.array_equal(loaded.encode(actions), vocab.encode(actions))


class TestLiveMask:
    def test_start_and_select_are_masked(self):
        vocab = build_vocab([sample_actions()], threshold=100)
        token = int(vocab.encode(np.array([START], np.uint8))[0])
        assert vocab.decode_byte(token, mask_live=True) & START == 0
        assert vocab.decode_byte(token, mask_live=True) & SELECT == 0

    def test_mask_preserves_movement_buttons(self):
        assert LIVE_MASK & RIGHT and LIVE_MASK & LEFT and LIVE_MASK & A and LIVE_MASK & B

    def test_unmasked_decode_returns_the_raw_byte(self):
        vocab = build_vocab([sample_actions()], threshold=100)
        assert vocab.decode_byte(1, mask_live=False) == RIGHT | B


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

def make_run(tmp_path: Path, name: str, n: int = 40, *, seed: int = 0) -> LoadedRun:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(n, 84, 84), dtype=np.uint8)
    np.save(d / "frames.npy", frames)
    actions = np.array(([0, RIGHT | B] * n)[:n], dtype=np.uint8)
    np.save(d / "actions.npy", actions)
    np.save(d / "frame_indices.npy", np.arange(n, dtype=np.int64))
    trace = np.zeros((n, 13), np.int32)
    np.save(d / "trace.npy", trace)
    manifest = {
        "n_frames": n, "synced": True, "category": "warpless", "chain": "",
        "label": name, "measured_route": "warpless",
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    from tasdata.dataset import load_run_dir

    return load_run_dir(d)


class TestFrameStackDataset:
    def test_length_is_total_observations(self, tmp_path: Path):
        runs = [make_run(tmp_path, "a", 40), make_run(tmp_path, "b", 25)]
        vocab = build_vocab([r.actions for r in runs], threshold=1)
        assert len(FrameStackDataset(runs, vocab)) == 65

    def test_item_shape_and_range(self, tmp_path: Path):
        runs = [make_run(tmp_path, "a", 40)]
        vocab = build_vocab([runs[0].actions], threshold=1)
        obs, _prev, token = FrameStackDataset(runs, vocab, stack=4)[10]
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == torch.float32
        assert 0.0 <= float(obs.min()) and float(obs.max()) <= 1.0
        assert isinstance(token, int)

    def test_stack_is_the_preceding_frames(self, tmp_path: Path):
        runs = [make_run(tmp_path, "a", 40, seed=3)]
        vocab = build_vocab([runs[0].actions], threshold=1)
        ds = FrameStackDataset(runs, vocab, stack=4)
        raw = np.load(tmp_path / "a" / "frames.npy")
        obs, _prev, _tok = ds[10]
        expected = raw[[7, 8, 9, 10]].astype(np.float32) / 255.0
        assert np.allclose(obs.numpy(), expected, atol=1e-6)

    def test_early_frames_are_edge_padded(self, tmp_path: Path):
        runs = [make_run(tmp_path, "a", 40, seed=5)]
        vocab = build_vocab([runs[0].actions], threshold=1)
        obs, _prev, _tok = FrameStackDataset(runs, vocab, stack=4)[0]
        # all four positions are frame 0
        assert np.allclose(obs[0].numpy(), obs[3].numpy())

    def test_stack_never_straddles_two_runs(self, tmp_path: Path):
        a = make_run(tmp_path, "a", 20, seed=1)
        b = make_run(tmp_path, "b", 20, seed=2)
        vocab = build_vocab([a.actions], threshold=1)
        ds = FrameStackDataset([a, b], vocab, stack=4)
        raw_b = np.load(tmp_path / "b" / "frames.npy")
        obs, _prev, _tok = ds[20]  # first frame of run b
        assert np.allclose(obs[-1].numpy(), raw_b[0].astype(np.float32) / 255.0)
        assert np.allclose(obs[0].numpy(), obs[-1].numpy())

    def test_blind_returns_zeros_but_the_right_label(self, tmp_path: Path):
        runs = [make_run(tmp_path, "a", 40)]
        vocab = build_vocab([runs[0].actions], threshold=1)
        sighted = FrameStackDataset(runs, vocab, blind=False)
        blind = FrameStackDataset(runs, vocab, blind=True)
        obs, _prev, token = blind[7]
        assert float(obs.abs().sum()) == 0.0
        assert token == sighted[7][2]

    def test_locate_maps_global_index_to_run(self, tmp_path: Path):
        a = make_run(tmp_path, "a", 10)
        b = make_run(tmp_path, "b", 10)
        vocab = build_vocab([a.actions], threshold=1)
        ds = FrameStackDataset([a, b], vocab)
        assert ds.locate(0) == (0, 0)
        assert ds.locate(9) == (0, 9)
        assert ds.locate(10) == (1, 0)

    def test_token_counts_sum_to_length(self, tmp_path: Path):
        runs = [make_run(tmp_path, "a", 40)]
        vocab = build_vocab([runs[0].actions], threshold=1)
        ds = FrameStackDataset(runs, vocab)
        assert int(ds.token_counts().sum()) == len(ds)

    def test_frames_are_memmapped_not_loaded(self, tmp_path: Path):
        runs = [make_run(tmp_path, "a", 40)]
        vocab = build_vocab([runs[0].actions], threshold=1)
        ds = FrameStackDataset(runs, vocab)
        _ = ds[5]
        assert isinstance(ds._frames(0), np.memmap)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class TestPolicy:
    def test_output_shape(self):
        policy = build_policy(PolicyConfig(n_actions=25))
        out = policy(torch.rand(3, 4, 84, 84))
        assert out.shape == (3, 25)

    def test_starts_small(self):
        policy = build_policy(PolicyConfig(n_actions=25))
        assert policy.n_parameters < 400_000

    def test_blind_output_is_input_independent(self):
        policy = build_policy(PolicyConfig(n_actions=25, blind=True)).eval()
        with torch.no_grad():
            a = policy(torch.rand(2, 4, 84, 84))
            b = policy(torch.rand(2, 4, 84, 84))
        assert torch.allclose(a, b)

    def test_sighted_output_is_input_dependent(self):
        policy = build_policy(PolicyConfig(n_actions=25)).eval()
        with torch.no_grad():
            a = policy(torch.rand(2, 4, 84, 84))
            b = policy(torch.rand(2, 4, 84, 84))
        assert not torch.allclose(a, b)

    def test_frame_order_matters(self):
        """If the model ignored ordering it could not represent velocity."""
        policy = build_policy(PolicyConfig(n_actions=25)).eval()
        x = torch.rand(1, 4, 84, 84)
        with torch.no_grad():
            a = policy(x)
            b = policy(x.flip(1))
        assert not torch.allclose(a, b)

    def test_config_roundtrip(self):
        cfg = PolicyConfig(n_actions=25, d_model=128, n_layers=2, n_heads=4)
        assert PolicyConfig.from_dict(cfg.to_dict()) == cfg


class TestCheckpoint:
    def test_save_and_reload_reproduces_logits(self, tmp_path: Path):
        cfg = TrainConfig(name="t", d_model=32, cnn_channels=(8, 16, 16))
        policy = build_policy(cfg.policy_config(25)).eval()
        path = save_checkpoint(
            tmp_path / "c.pt", policy, cfg, step=7, vocab_path=tmp_path / "v.json"
        )
        reloaded, blob = load_checkpoint(path)
        reloaded.eval()
        probe = torch.rand(2, 4, 84, 84)
        with torch.no_grad():
            assert torch.allclose(policy(probe), reloaded(probe), atol=1e-6)
        assert blob["step"] == 7

    def test_checkpoint_records_configs(self, tmp_path: Path):
        cfg = TrainConfig(name="t", d_model=32, lr=1e-3, cnn_channels=(8, 16, 16))
        policy = build_policy(cfg.policy_config(25))
        _, blob = load_checkpoint(
            save_checkpoint(tmp_path / "c.pt", policy, cfg, step=1, vocab_path="v.json")
        )
        assert blob["train_config"]["lr"] == 1e-3
        assert blob["policy_config"]["n_actions"] == 25


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #

class TestBaselines:
    def test_constant_policy_always_picks_its_token(self):
        policy = ConstantPolicy(25, 7)
        logits = policy(torch.rand(5, 4, 84, 84))
        assert torch.all(logits.argmax(1) == 7)

    def test_marginal_policy_ignores_input(self):
        p = np.zeros(25)
        p[3] = 1.0
        policy = MarginalPolicy(p)
        logits = policy(torch.rand(4, 4, 84, 84))
        assert torch.all(logits.argmax(1) == 3)
        assert logits.shape == (4, 25)

    def test_score_predictions(self):
        labels = np.array([0, 0, 1, 1])
        preds = np.array([0, 0, 0, 1])
        acc, macro, per_class = score_predictions(preds, labels, 3)
        assert acc == 0.75
        assert per_class[0] == 1.0 and per_class[1] == 0.5
        assert np.isnan(per_class[2])
        assert macro == pytest.approx(0.75)

    def test_macro_ignores_absent_classes(self):
        _acc, macro, _ = score_predictions(np.zeros(4, int), np.zeros(4, int), 10)
        assert macro == 1.0

    def test_token_for_buttons(self):
        vocab = build_vocab([sample_actions()], threshold=100)
        assert vocab.token_to_byte[token_for_buttons(vocab)] == 0
        assert vocab.token_to_byte[token_for_buttons(vocab, "Right", "B")] == RIGHT | B

    def test_always_nothing_accuracy_matches_label_share(self):
        vocab = build_vocab([sample_actions()], threshold=100)
        labels = vocab.encode(sample_actions())
        counts = np.bincount(labels, minlength=vocab.size)
        scores = evaluate_trivial_baselines(labels, counts, vocab)
        nothing = next(s for s in scores if s.name == "always nothing")
        assert nothing.accuracy == pytest.approx(counts[0] / labels.size)

    def test_all_four_baselines_present(self):
        vocab = build_vocab([sample_actions()], threshold=100)
        labels = vocab.encode(sample_actions())
        counts = np.bincount(labels, minlength=vocab.size)
        names = {s.name for s in evaluate_trivial_baselines(labels, counts, vocab)}
        assert "always nothing" in names
        assert "always Right+B" in names
        assert "sample marginal distribution" in names


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

class TestReport:
    def test_handles_an_empty_log(self):
        assert "Stage 2" in build_summary([])

    def test_reports_a_failed_smoke_test(self):
        out = build_summary([{"kind": "smoke", "ok": False, "error": "loss went up"}])
        assert "FAILED" in out and "loss went up" in out

    def test_renders_configs_and_failures(self):
        records = [
            {
                "kind": "eval", "config": "tiny", "blind": False, "step": 100,
                "val": {"loss": 1.0, "accuracy": 0.7, "macro_accuracy": 0.2,
                        "prediction_counts": [5, 5], "label_counts": [8, 2]},
                "vocab_names": ["-", "RARE(3 combos)"],
                "live": {"total_progress": {"median": 500.0},
                         "levels_reached": {"median": 1.0},
                         "deaths": {"median": 0.0}},
                "environment": {"torch": "2.13", "device": "mps"},
            },
            {"kind": "config_failed", "config": "small", "error": "boom",
             "wall_seconds": 3.0},
        ]
        out = build_summary(records)
        assert "tiny" in out
        assert "70.00%" in out
        assert "boom" in out
        assert "RARE predicted" in out

    def test_partial_last_line_is_skipped(self, tmp_path: Path):
        from tasdata.bc.report import read_records

        path = tmp_path / "r.jsonl"
        path.write_text('{"kind": "smoke", "ok": true}\n{"kind": "ev')
        assert len(read_records(path)) == 1
