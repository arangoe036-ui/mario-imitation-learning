"""Command-line interface: ``python -m tasdata <command>``.

Commands
--------
``parse``    inspect a .bk2/.fm2 and optionally dump its button matrix
``replay``   replay a movie through nes-py, saving frames + RAM trace
``verify``   replay and report a sync pass/fail
``run``      parse -> replay -> verify in one shot, writing a dataset directory
``fetch``    download movies from tasvideos.org
``rominfo``  print a ROM's iNES header and both fingerprints
``emuinfo``  print the FCEUX build being used
``curate``   discover + filter TASVideos movies into a capture shortlist
``bc-smoke`` gate: tiny train + checkpoint round trip + one live episode
``bc-sweep`` overnight behavioural-cloning sweep, appending JSONL results
``bc-report`` render stage2_results.jsonl into a readable markdown summary
``bc-retrain`` one config (+ ablation) with the fixed loader and rebuilt evaluation
``bc-arms``  stage 2c: bernoulli control vs bernoulli + onset reweighting
``measure``  RAM-only probe of a shortlist: measured level counts and sync status
``batch``    capture every movie in a shortlist, surviving individual failures
``stats``    action vocabulary, impossible inputs, hold lengths, overlap
``split``    write the immutable whole-run train/val/test split
``reference`` record a known-good RAM trace for frame-exact regression checks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .backends import BACKENDS, DEFAULT_BACKEND, get_replayer
from .buttons import actions_from_states, describe_action
from .formats import MovieFormatError, sniff
from .movie import parse_movie
from .ram import TRACE_COLUMNS, state_from_row
from .replay import NesReplayer, ReplayError, RomMismatchError
from .rom import load_rom
from .verify import (
    load_reference,
    parse_level_spec,
    save_reference,
    verify_smb,
)


def _shape(text: str) -> tuple[int, int]:
    """``"84x84"`` -> ``(84, 84)``."""
    try:
        h, w = text.lower().split("x")
        return int(h), int(w)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected HxW like 84x84, got {text!r}") from exc


def _progress(label: str):
    def report(done: int, total: int) -> None:
        pct = 100.0 * done / total if total else 100.0
        print(f"\r  {label}: {done}/{total} ({pct:5.1f}%)", end="", file=sys.stderr, flush=True)
        if done >= total:
            print(file=sys.stderr)

    return report


# --------------------------------------------------------------------------- #
# parse
# --------------------------------------------------------------------------- #

def cmd_parse(args: argparse.Namespace) -> int:
    movie = parse_movie(args.movie, allow_tasproj=args.allow_tasproj)
    if args.json:
        print(json.dumps(movie.to_dict(), indent=2))
    else:
        print(movie.summary())
        print(f"  button groups  : {[list(g) for g in movie.groups]}")
        print(f"  field widths   : {tuple(len(g) for g in movie.groups)}")
        print(f"  rom hashes     : {movie.rom_hashes or 'none recorded'}")
        print(f"  region         : {'PAL (50 Hz)' if movie.pal else 'NTSC'}")
        print(f"  states shape   : {movie.states.shape} dtype={movie.states.dtype}")
        if args.rom:
            check = movie.verify_rom(args.rom)
            print(f"  rom check      : {check.line()}")
        for note in movie.notes:
            print(f"  note           : {note}")
        actions = actions_from_states(movie.states, movie.button_names, args.player)
        held = int((actions != 0).sum())
        print(f"  P{args.player} action bytes: {held}/{len(actions)} frames have input")
        print("  per-button press counts:")
        for i, name in enumerate(movie.button_names):
            count = int(movie.states[:, i].sum())
            if count:
                print(f"    {name:12s} {count}")
        if args.head:
            print(f"  first {args.head} frames:")
            for f in range(min(args.head, movie.n_frames)):
                print(f"    f{f:<6d} {describe_action(int(actions[f]))}")
    if args.dump_npy:
        out = Path(args.dump_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, movie.states)
        print(f"wrote {out} {movie.states.shape} {movie.states.dtype}")
    return 0


# --------------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------------- #

def _build_replayer(args: argparse.Namespace, capture: bool | None = None):
    return get_replayer(
        args.backend,
        args.rom,
        observation_shape=args.observation,
        frame_skip=args.frame_skip,
        capture_frames=(not args.no_frames) if capture is None else capture,
        player=args.player,
        allow_rom_mismatch=args.allow_rom_mismatch,
        extra_args=args.fceux_arg or (),
    )


def cmd_replay(args: argparse.Namespace) -> int:
    movie = parse_movie(args.movie, allow_tasproj=args.allow_tasproj)
    print(movie.summary())
    replayer = _build_replayer(args)
    if not args.quiet and hasattr(replayer, "describe"):
        print(f"  backend: {replayer.describe()}")
    result = replayer.replay(
        movie,
        max_frames=args.max_frames,
        frames_path=args.frames_memmap,
        progress=None if args.quiet else _progress("replay"),
    )
    for note in result.warnings:
        print(f"  warning: {note}")
    print(f"  {result.summary()}")
    final = state_from_row(result.trace[-1]) if len(result.trace) else None
    if final:
        print(f"  final RAM state: {final}")
    if args.out:
        path = result.save(args.out)
        print(f"wrote {path}")
    return 0


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #

def _verify(args: argparse.Namespace, capture: bool) -> tuple[int, object]:
    movie = parse_movie(args.movie, allow_tasproj=args.allow_tasproj)
    if not args.quiet:
        print(movie.summary())
    replayer = _build_replayer(args, capture=capture)
    if not args.quiet and hasattr(replayer, "describe"):
        print(f"  backend: {replayer.describe()}")
    result = replayer.replay(
        movie,
        max_frames=args.max_frames,
        frames_path=getattr(args, "frames_memmap", None),
        progress=None if args.quiet else _progress("replay"),
    )
    reference = None
    ref_cols = TRACE_COLUMNS
    if getattr(args, "reference", None):
        reference, ref_cols = load_reference(args.reference)
    report = verify_smb(
        result.trace,
        movie_name=str(movie.path),
        rom_name=str(replayer.rom_path),
        expect_level=args.expect,
        min_levels=args.min_levels,
        stall_frames=args.stall_frames,
        strict_stall=args.strict_stall,
        reference=reference,
        reference_columns=ref_cols,
        rom_matches_movie=result.rom_check.matched,
        rom_check_detail=result.rom_check.line(),
        movie_is_pal=movie.pal,
        replay_warnings=result.warnings,
    )
    return 0 if report.passed else 1, (report, result, movie)


def cmd_verify(args: argparse.Namespace) -> int:
    code, (report, result, _movie) = _verify(args, capture=False)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.text())
        print()
        print(f"SYNC: {'PASS' if report.passed else 'FAIL'}")
    return code


# --------------------------------------------------------------------------- #
# run (full pipeline)
# --------------------------------------------------------------------------- #

def cmd_run(args: argparse.Namespace) -> int:
    from .dataset import dir_bytes, write_run_dataset

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.frames_memmap = out_dir / "frames.npy"
    code, (report, result, movie) = _verify(args, capture=not args.no_frames)
    write_run_dataset(out_dir, movie, result, report)
    print(report.text())
    print()
    print(f"dataset written to {out_dir} ({dir_bytes(out_dir) / (1 << 20):.1f} MiB)")
    print(f"SYNC: {'PASS' if report.passed else 'FAIL'}")
    return code


# --------------------------------------------------------------------------- #
# batch
# --------------------------------------------------------------------------- #

def _bc_corpus(args: argparse.Namespace, log):
    from .bc.sweep import prepare_corpus

    return prepare_corpus(
        args.runs,
        args.split,
        args.vocab,
        threshold=args.rare_threshold,
        rebuild_vocab=args.rebuild_vocab,
        log=log,
    )


def cmd_bc_smoke(args: argparse.Namespace) -> int:
    from .bc.sweep import SmokeTestFailure, append_jsonl, environment, smoke_test

    log = print
    print("SMOKE TEST (gate for the long run)")
    corpus = _bc_corpus(args, log)
    print(corpus.vocab.summary())
    print()
    try:
        report = smoke_test(
            corpus,
            args.rom,
            out_dir=args.out,
            frames=args.frames,
            steps=args.steps,
            live_frames=args.live_frames,
            log=log,
        )
    except SmokeTestFailure as exc:
        print(f"\nSMOKE TEST FAILED\n  {exc}", file=sys.stderr)
        append_jsonl(
            args.results,
            {"kind": "smoke", "ok": False, "error": str(exc)[:600], "environment": environment()},
        )
        return 1
    report["environment"] = environment()
    append_jsonl(args.results, report)
    print("\nSMOKE TEST PASSED - safe to start the long run")
    return 0


def cmd_bc_sweep(args: argparse.Namespace) -> int:
    from .bc.sweep import (
        SmokeTestFailure,
        append_jsonl,
        default_configs,
        environment,
        run_sweep,
        run_trivial_baselines,
        smoke_test,
    )

    log = print
    corpus = _bc_corpus(args, log)
    print(corpus.vocab.summary())

    if not args.skip_smoke:
        print("\nSMOKE TEST (gate for the long run)")
        try:
            report = smoke_test(
                corpus, args.rom, out_dir=args.out, frames=1000, steps=50,
                live_frames=600, log=log,
            )
        except SmokeTestFailure as exc:
            print(f"\nSMOKE TEST FAILED - not starting the sweep\n  {exc}", file=sys.stderr)
            append_jsonl(args.results, {"kind": "smoke", "ok": False, "error": str(exc)[:600]})
            return 1
        report["environment"] = environment()
        append_jsonl(args.results, report)
        print("SMOKE TEST PASSED\n")

    if not args.skip_baselines:
        print("TRIVIAL BASELINES")
        run_trivial_baselines(
            corpus, args.rom, results_path=args.results,
            seeds=args.baseline_seeds, live_frames=args.live_frames, log=log,
        )

    configs = default_configs(args.steps, args.eval_every, num_workers=args.num_workers)
    if args.only:
        wanted = set(args.only)
        configs = [c for c in configs if c.name in wanted]
        if not configs:
            print(f"no configs match {sorted(wanted)}", file=sys.stderr)
            return 2
    print(f"\nSWEEP: {len(configs)} configs x {args.steps} steps")
    run_sweep(
        corpus, args.rom, configs,
        out_dir=args.out, results_path=args.results,
        eval_seeds=args.eval_seeds, live_frames=args.live_frames, log=log,
    )
    print(f"\nall results appended to {args.results}")
    return 0


def cmd_bc_retrain(args: argparse.Namespace) -> int:
    from .bc.sweep import (
        SmokeTestFailure,
        append_jsonl,
        environment,
        retrain_configs,
        run_sweep,
        smoke_test,
    )

    log = print
    corpus = _bc_corpus(args, log)
    print(f"  vocabulary: {corpus.vocab.size} tokens")

    if not args.skip_smoke:
        print("\nSMOKE TEST (gate)")
        try:
            report = smoke_test(
                corpus, args.rom, out_dir=args.out, frames=1000, steps=50,
                live_frames=600, device=args.device, log=log,
            )
        except SmokeTestFailure as exc:
            print(f"\nSMOKE TEST FAILED - not retraining\n  {exc}", file=sys.stderr)
            append_jsonl(args.results, {"kind": "smoke", "ok": False, "error": str(exc)[:600]})
            return 1
        report["environment"] = environment()
        append_jsonl(args.results, report)
        print("SMOKE TEST PASSED\n")

    configs = retrain_configs(num_workers=args.num_workers, steps=args.steps)
    if args.only:
        configs = [c for c in configs if c.name in set(args.only)]
    print(f"RETRAIN: {[c.name for c in configs]} evaluating at {configs[0].eval_steps}")
    run_sweep(
        corpus, args.rom, configs,
        out_dir=args.out, results_path=args.results,
        eval_seeds=args.eval_seeds, live_frames=args.live_frames,
        expert_movie=args.expert_movie, eval_levels=tuple(args.levels),
        stall_limit=args.stall_frames, device=args.device, log=log,
    )
    print(f"\nresults appended to {args.results}")
    return 0


def cmd_bc_arms(args: argparse.Namespace) -> int:
    from .bc.arms import arm_configs, run_arms
    from .bc.sweep import SmokeTestFailure, append_jsonl, environment, smoke_test

    log = print
    corpus = _bc_corpus(args, log)
    if not args.skip_smoke:
        print("\nSMOKE TEST (gate)")
        try:
            report = smoke_test(
                corpus, args.rom, out_dir=args.out, frames=1000, steps=50,
                live_frames=600, device="cpu", log=log,
            )
        except SmokeTestFailure as exc:
            print(f"\nSMOKE TEST FAILED - not running the arms\n  {exc}", file=sys.stderr)
            append_jsonl(args.results, {"kind": "smoke", "ok": False, "error": str(exc)[:600]})
            return 1
        report["environment"] = environment("cpu")
        append_jsonl(args.results, report)
        print("SMOKE TEST PASSED")

    configs = arm_configs(steps=args.steps, num_workers=args.num_workers)
    if args.only:
        configs = [c for c in configs if c.name in set(args.only)]
    run_arms(
        corpus, args.rom, configs,
        out_dir=args.out, results_path=args.results,
        expert_movie=args.expert_movie, levels=tuple(args.levels),
        train_seeds=args.train_seeds, final_seeds=args.final_seeds,
        live_frames=args.live_frames, stall_frames=args.stall_frames,
        workers=args.workers, log=log,
    )
    print(f"\nresults appended to {args.results}")
    return 0


def cmd_bc_report(args: argparse.Namespace) -> int:
    from .bc.report import write_summary

    out = write_summary(args.results, args.out)
    print(out.read_text())
    print(f"\nwritten to {out}", file=sys.stderr)
    return 0


def cmd_measure(args: argparse.Namespace) -> int:
    from .batch import measure_batch
    from .curate import load_plan

    candidates = load_plan(args.plan)
    if args.limit:
        candidates = candidates[: args.limit]
    print(
        f"measuring {len(candidates)} runs RAM-only (no images) -> {args.report}",
        file=sys.stderr,
    )
    ms = measure_batch(
        candidates,
        args.rom,
        stall_frames=args.stall_frames,
        report_path=args.report,
        on_event=None if args.quiet else (lambda m: print(m, flush=True, file=sys.stderr)),
    )
    print()
    print(
        f"{'run':16s} {'declared category':20s} {'frames':>8s} {'duration':>8s} "
        f"{'lvl':>3s} {'far':>4s} {'measured route':12s} sync"
    )
    print("-" * 96)
    for m in ms:
        print(m.row())
    print("-" * 96)
    bad = [m for m in ms if m.declared_category.startswith("warpless") and m.route != "warpless"]
    if bad:
        print("\nMISLABELLED (declared warpless-family but does not visit ~32 levels):")
        for m in bad:
            print(f"  {m.label:16s} declared {m.declared_category:20s} measured {m.route} ({m.measured_levels} levels)")
    desync = [m for m in ms if not m.synced and not m.error]
    if desync:
        print("\nDESYNCED:")
        for m in desync:
            print(f"  {m.label:16s} {m.reason[:74]}")
    errs = [m for m in ms if m.error]
    if errs:
        print("\nERRORS:")
        for m in errs:
            print(f"  {m.label:16s} {m.error}")
    print(f"\nmeasurements written to {args.report}")
    if args.update_plan:
        from .curate import apply_measurements

        relabelled, desynced = apply_measurements(Path(args.plan), Path(args.report))
        print(
            f"plan updated: {relabelled} categories corrected from measurement, "
            f"{desynced} runs pre-flagged as desynced"
        )
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    from .batch import run_batch
    from .curate import load_plan

    candidates = load_plan(args.plan)
    if args.limit:
        candidates = candidates[: args.limit]
    est = sum(c.est_bytes for c in candidates)
    print(
        f"capturing {len(candidates)} runs, "
        f"{sum(c.n_frames for c in candidates):,} frames, "
        f"est {est / (1 << 30):.2f} GiB -> {args.out}"
    )
    report = run_batch(
        candidates,
        args.rom,
        args.out,
        observation_shape=args.observation,
        frame_skip=args.frame_skip,
        expect_level=args.expect,
        stall_frames=args.stall_frames,
        report_path=args.report,
        on_event=None if args.quiet else (lambda m: print(m, flush=True)),
    )
    print(report.summary())
    print(f"\nreport written to {args.report}")
    return 0 if not report.failed else 1


# --------------------------------------------------------------------------- #
# stats / split
# --------------------------------------------------------------------------- #

def cmd_stats(args: argparse.Namespace) -> int:
    from .analyze import build_report
    from .dataset import discover_runs

    runs = discover_runs(args.runs, synced_only=args.synced_only)
    if not runs:
        print(f"no run directories under {args.runs}", file=sys.stderr)
        return 2
    rep = build_report(runs)
    if args.json:
        print(json.dumps(rep, indent=2))
        return 0

    total = rep["total_frames"]
    print(f"runs: {rep['n_runs']}   frames: {total:,}")
    print()
    print(f"ACTION VOCABULARY: {rep['action_vocabulary_size']} distinct button combinations")
    print(f"{'combination':28s} {'count':>10s} {'share':>8s}  cumulative")
    print("-" * 62)
    cum = 0.0
    for row in rep["action_frequency"]:
        cum += row["percentage"]
        print(
            f"{row['buttons']:28s} {row['count']:10,d} {row['percentage']:7.3f}% "
            f"{cum:9.2f}%"
        )
    print()
    imp = rep["impossible_inputs"]
    print("PHYSICALLY IMPOSSIBLE INPUTS (real hardware cannot produce these)")
    for name, pct in imp["percentages"].items():
        print(f"  {name:12s} {imp['counts'][name]:9,d} frames  {pct:6.3f}%")
    print(f"  {'either':12s} {imp['either_count']:9,d} frames  {imp['either_percentage']:6.3f}%")
    print()
    print("HOLD LENGTHS (frames per contiguous press)")
    print(f"  {'button':8s} {'presses':>9s} {'held':>10s} {'mean':>7s} {'med':>5s} {'p90':>6s} {'p99':>6s} {'max':>6s} {'1f taps':>8s}")
    for row in rep["hold_lengths"]:
        if not row["presses"]:
            print(f"  {row['button']:8s} {'never pressed':>9s}")
            continue
        print(
            f"  {row['button']:8s} {row['presses']:9,d} {row['frames_held']:10,d} "
            f"{row['mean']:7.2f} {row['median']:5.0f} {row['p90']:6.0f} "
            f"{row['p99']:6.0f} {row['max']:6d} {row['one_frame_taps']:8,d}"
        )
    print()
    ov = rep["overlap"]
    print("OVERLAP WITHIN OBSOLETION CHAINS (fraction of frames with equal actions)")
    if ov["chain_pairs"]:
        for p in ov["chain_pairs"]:
            print(
                f"  {p['older']:22s} vs {p['newer']:22s} "
                f"{p['agreement'] * 100:6.2f}% of {p['n_compared']:,} frames"
            )
    else:
        print("  (no chains with two or more captured members)")
    print()
    print(f"  raw frames       : {ov['raw_frames']:,}")
    print(f"  effective frames : {ov['effective_frames']:,}")
    print(f"  redundancy       : {ov['redundancy_percentage']:.1f}%")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\nfull report written to {args.out}")
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    from .analyze import SplitExistsError, make_split, verify_split, write_split
    from .dataset import discover_runs

    if args.verify:
        ok = verify_split(args.out)
        print(f"{args.out}: checksum {'OK' if ok else 'MISMATCH - file was edited'}")
        return 0 if ok else 1

    runs = discover_runs(args.runs, synced_only=not args.include_desynced)
    if not runs:
        print(f"no usable run directories under {args.runs}", file=sys.stderr)
        return 2
    split = make_split(
        runs, val_fraction=args.val, test_fraction=args.test, seed=args.seed
    )
    try:
        path = write_split(args.out, split, force=args.force)
    except SplitExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    total = sum(split.frames.values())
    print(f"policy: {split.policy}")
    print(f"seed  : {split.seed}")
    for name in ("train", "val", "test"):
        members = getattr(split, name)
        frames = sum(split.frames[m] for m in members)
        print(f"\n{name.upper()}  {len(members)} runs, {frames:,} frames ({frames * 100 / total:.1f}%)")
        for m in members:
            print(f"    {m:28s} {split.frames[m]:8,d}")
    print(f"\nsplit written to {path} (immutable; re-running refuses to overwrite)")
    return 0


# --------------------------------------------------------------------------- #
# reference
# --------------------------------------------------------------------------- #

def cmd_reference(args: argparse.Namespace) -> int:
    movie = parse_movie(args.movie, allow_tasproj=args.allow_tasproj)
    replayer = get_replayer(
        args.backend,
        args.rom,
        capture_frames=False,
        player=args.player,
        allow_rom_mismatch=args.allow_rom_mismatch,
        extra_args=args.fceux_arg or (),
    )
    result = replayer.replay(
        movie, max_frames=args.max_frames, progress=None if args.quiet else _progress("replay")
    )
    path = save_reference(
        result.trace,
        args.out,
        meta={
            "movie": str(movie.path),
            "rom_sha1_file": result.rom.sha1_file,
            "n_frames": result.n_frames,
            "tasdata_version": __version__,
        },
    )
    print(f"wrote reference trace {path} {result.trace.shape}")
    return 0


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #

def cmd_fetch(args: argparse.Namespace) -> int:
    from .tasvideos import fetch_game_movies, search_games

    if args.game_name:
        games = search_games(args.game_name, args.system)
        if not games:
            print(f"no {args.system} game matching {args.game_name!r}", file=sys.stderr)
            return 2
        for g in games:
            print(f"  game id {g['id']}: {g['displayName']}")
        if args.game_id is None:
            args.game_id = games[0]["id"]
            print(f"using game id {args.game_id}")
    if args.game_id is None:
        print("need --game-id or --game-name", file=sys.stderr)
        return 2

    kept = fetch_game_movies(
        args.game_id,
        args.out,
        only_bk2=not args.all_formats,
        limit=args.limit,
        include_publications=not args.no_publications,
        on_progress=None if args.quiet else (lambda m: print(f"  {m}")),
    )
    print(f"kept {len(kept)} movie(s) in {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# info (format sniffing only)
# --------------------------------------------------------------------------- #

def cmd_curate(args: argparse.Namespace) -> int:
    from .curate import (
        gather_publications,
        gather_userfiles,
        select,
        write_plan,
    )

    rom = load_rom(args.rom)
    pool = Path(args.pool)
    log = (lambda m: None) if args.quiet else (lambda m: print(f"  {m}", file=sys.stderr))

    print(f"ROM: {rom.path.name} md5(prg+chr)={rom.md5_prgchr}", file=sys.stderr)
    print("discovering publications (obsoleted included)...", file=sys.stderr)
    cands = gather_publications(
        args.game_id, rom, pool, observation_shape=args.observation, on_progress=log
    )
    usable = sum(1 for c in cands if not c.rejected)
    print(f"  publications: {usable} usable of {len(cands)}", file=sys.stderr)

    if usable < args.target and not args.no_userfiles:
        print("discovering user files to fill the target...", file=sys.stderr)
        cands += gather_userfiles(
            args.game_id, rom, pool, observation_shape=args.observation, on_progress=log
        )

    selected, rejected = select(
        cands, target=args.target, max_low_coverage=args.max_warps
    )
    plan = write_plan(args.plan, selected, rejected)

    total_frames = sum(c.n_frames for c in selected)
    total_bytes = sum(c.est_bytes for c in selected)
    print()
    print(f"{'SHORTLIST':26s} {'category':20s} {'authors':26s} {'frames':>8s} {'lvls':>4s} {'est disk':>11s}")
    print("-" * 100)
    for c in selected:
        print(c.row())
    print("-" * 100)
    print(
        f"{len(selected)} runs | {total_frames:,} frames "
        f"({total_frames / 60.0988 / 3600:.1f} h of gameplay) | "
        f"est {total_bytes / (1 << 30):.2f} GiB on disk"
    )
    by_cat: dict[str, list] = {}
    for c in selected:
        by_cat.setdefault(c.category, []).append(c)
    print("\nby category:")
    for cat, group in sorted(by_cat.items(), key=lambda kv: -sum(x.n_frames for x in kv[1])):
        f = sum(x.n_frames for x in group)
        print(
            f"  {cat:22s} {len(group):3d} runs {f:9,d} frames "
            f"{f * 100 / total_frames:5.1f}%  {group[0].levels} levels each"
        )
    if args.show_rejected:
        print("\nrejected:")
        for c in sorted(rejected, key=lambda c: str(c.rejected)):
            print(f"  {c.label:26s} {str(c.rejected)[:88]}")
    print(f"\nplan written to {plan}")
    print("Nothing captured yet. Review, then run: tasdata batch --plan " + str(plan))
    return 0


def cmd_emuinfo(args: argparse.Namespace) -> int:
    from .fceux_backend import fceux_git_rev, fceux_version, find_fceux

    path = find_fceux(args.binary)
    print(f"fceux binary : {path}")
    print(f"version      : {fceux_version(path)}")
    print(f"git rev      : {fceux_git_rev(path)}")
    print(
        "L+R / U+D    : enabled per-run via --opposite-directionals 1 "
        "(config key SDL.Input.EnableOppositeDirectionals, default 0)"
    )
    return 0


def cmd_rominfo(args: argparse.Namespace) -> int:
    for path in args.roms:
        print(load_rom(path).summary())
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    rc = 0
    for path in args.paths:
        try:
            result = sniff(path)
        except MovieFormatError as exc:
            print(f"{path}: {exc}")
            rc = 1
            continue
        gz = " (gzip-wrapped)" if result.gzipped else ""
        print(f"{path}: {result.format.value} -- {result.description}{gz}")
    return rc


# --------------------------------------------------------------------------- #
# argument wiring
# --------------------------------------------------------------------------- #

def _add_movie_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("movie", help="path to a .bk2 movie (gzip-wrapped is fine)")
    p.add_argument(
        "--allow-tasproj",
        action="store_true",
        help="also accept BizHawk .tasproj projects",
    )
    p.add_argument("--player", type=int, default=1, help="controller port to read (default 1)")


def _add_replay_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--rom", required=True, help="path to the .nes ROM")
    p.add_argument(
        "--observation",
        type=_shape,
        default=(84, 84),
        metavar="HxW",
        help="downscaled grayscale frame size (default 84x84)",
    )
    p.add_argument("--frame-skip", type=int, default=1, help="capture every Nth frame")
    p.add_argument("--max-frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--no-frames", action="store_true", help="skip image capture entirely")
    p.add_argument(
        "--allow-rom-mismatch",
        action="store_true",
        help="replay even when the movie's recorded ROM fingerprint does not match "
        "the supplied ROM (default: refuse, since it is guaranteed to desync)",
    )
    p.add_argument("--quiet", action="store_true", help="suppress progress output")
    p.add_argument(
        "--backend",
        choices=BACKENDS,
        default=DEFAULT_BACKEND,
        help=f"replay backend (default {DEFAULT_BACKEND}); 'fceux' replays the "
        "movie in the emulator it was recorded in, 'nes-py' feeds inputs to nes-py",
    )
    p.add_argument(
        "--fceux-arg",
        action="append",
        default=None,
        metavar="ARG",
        help="extra raw argument passed to FCEUX (repeatable), e.g. --fceux-arg --pal --fceux-arg 1",
    )


def _add_verify_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--expect", default=None, metavar="W-S", help="level the run must reach, e.g. 8-4")
    p.add_argument(
        "--min-levels", type=int, default=2, help="distinct levels required to pass (default 2)"
    )
    p.add_argument(
        "--stall-frames",
        type=int,
        default=2000,
        help="frames without forward progress before flagging a stall (default 2000; "
        "the 2-1 vine to Coin Heaven legitimately pins x for 1,162 frames)",
    )
    p.add_argument(
        "--strict-stall", action="store_true", help="treat a stall as a failure, not an advisory"
    )
    p.add_argument("--reference", default=None, help="reference trace .npz for a frame-exact diff")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tasdata",
        description="TAS imitation-learning data pipeline: parse .bk2, replay on "
        "nes-py, verify sync.",
    )
    parser.add_argument("--version", action="version", version=f"tasdata {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("info", help="identify movie file formats")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("curate", help="build a capture shortlist from tasvideos.org")
    p.add_argument("--game-id", type=int, default=1, help="TASVideos game id (SMB is 1)")
    p.add_argument("--rom", required=True, help="movies must match this ROM")
    p.add_argument("--pool", default="data/movies/pool", help="where to download movies")
    p.add_argument("--plan", default="data/shortlist.json", help="write the shortlist here")
    p.add_argument("--target", type=int, default=40, help="max runs to shortlist")
    p.add_argument(
        "--max-warps",
        type=int,
        default=None,
        help="cap on runs that visit only a handful of levels (default: target/4); "
        "warps runs all cover the same four levels so they saturate quickly",
    )
    p.add_argument("--observation", type=_shape, default=(84, 84), metavar="HxW")
    p.add_argument("--no-userfiles", action="store_true", help="publications only")
    p.add_argument("--show-rejected", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_curate)

    def _add_bc_common(sp):
        sp.add_argument("--runs", default="data/runs")
        sp.add_argument("--split", default="data/split.json")
        sp.add_argument("--vocab", default="data/action_vocab.json")
        sp.add_argument("--rom", default="smb.nes")
        sp.add_argument("--out", default="data/bc")
        sp.add_argument("--results", default="data/stage2_results.jsonl")
        sp.add_argument("--rare-threshold", type=int, default=100)
        sp.add_argument("--rebuild-vocab", action="store_true")
        sp.add_argument("--live-frames", type=int, default=3000)

    p = sub.add_parser("bc-smoke", help="smoke test: gate before any long training run")
    _add_bc_common(p)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--steps", type=int, default=50)
    p.set_defaults(func=cmd_bc_smoke)

    p = sub.add_parser("bc-sweep", help="overnight behavioural-cloning sweep")
    _add_bc_common(p)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-seeds", type=int, default=20)
    p.add_argument("--baseline-seeds", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=2, help="dataloader workers")
    p.add_argument("--only", nargs="*", default=None, help="run only these config names")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--skip-baselines", action="store_true")
    p.set_defaults(func=cmd_bc_sweep)

    p = sub.add_parser("bc-retrain", help="retrain one config with the fixed loader")
    _add_bc_common(p)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval-seeds", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--stall-frames", type=int, default=300)
    p.add_argument(
        "--expert-movie",
        default="data/movies/happylee_mars608-smb-warpless.fm2",
        help="fast-forwarded to reach non-1-1 evaluation levels",
    )
    p.add_argument("--levels", nargs="*", default=["1-1", "1-2", "2-1", "4-1"])
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument(
        "--device",
        default="cpu",
        help="train on this device. Defaults to cpu because using MPS makes every "
        "FCEUX child fall back to broken software OpenGL and crash.",
    )
    p.set_defaults(func=cmd_bc_retrain)

    p = sub.add_parser("bc-arms", help="bernoulli control vs onset reweighting")
    _add_bc_common(p)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--train-seeds", type=int, default=5)
    p.add_argument("--final-seeds", type=int, default=20)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--workers", type=int, default=4, help="parallel emulators")
    p.add_argument("--stall-frames", type=int, default=300)
    p.add_argument("--expert-movie", default="data/movies/happylee_mars608-smb-warpless.fm2")
    p.add_argument("--levels", nargs="*", default=["1-1", "2-1"])
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--skip-smoke", action="store_true")
    p.set_defaults(func=cmd_bc_arms)

    p = sub.add_parser("bc-report", help="render the stage-2 results into markdown")
    p.add_argument("--results", default="data/stage2_results.jsonl")
    p.add_argument("--out", default="data/stage2_summary.md")
    p.set_defaults(func=cmd_bc_report)

    p = sub.add_parser("measure", help="RAM-only probe: measured levels + sync status")
    p.add_argument("--plan", default="data/shortlist.json")
    p.add_argument("--rom", required=True)
    p.add_argument("--report", default="data/measurements.json")
    p.add_argument("--stall-frames", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--update-plan",
        action="store_true",
        help="rewrite the shortlist's categories from what was measured",
    )
    p.set_defaults(func=cmd_measure)

    p = sub.add_parser("batch", help="capture every movie in a shortlist")
    p.add_argument("--plan", default="data/shortlist.json", help="shortlist from `curate`")
    p.add_argument("--rom", required=True)
    p.add_argument("--out", default="data/runs", help="root directory for run dirs")
    p.add_argument("--report", default="data/batch_report.json")
    p.add_argument("--observation", type=_shape, default=(84, 84), metavar="HxW")
    p.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="thins FRAMES only; actions and the RAM trace are always full rate",
    )
    p.add_argument("--expect", default=None, metavar="W-S")
    p.add_argument("--stall-frames", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None, help="capture only the first N")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_batch)

    p = sub.add_parser("stats", help="action vocabulary, impossible inputs, hold lengths")
    p.add_argument("--runs", default="data/runs")
    p.add_argument("--synced-only", action="store_true", help="ignore desynced runs")
    p.add_argument("--json", action="store_true")
    p.add_argument("--out", default=None, help="also write the full report here")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("split", help="write the immutable train/val/test split")
    p.add_argument("--runs", default="data/runs")
    p.add_argument("--out", default="data/split.json")
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--include-desynced", action="store_true")
    p.add_argument("--force", action="store_true", help="overwrite an existing split")
    p.add_argument("--verify", action="store_true", help="check the checksum only")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("emuinfo", help="print the FCEUX build being used")
    p.add_argument("--binary", default="fceux")
    p.set_defaults(func=cmd_emuinfo)

    p = sub.add_parser("rominfo", help="print a ROM's header and both fingerprints")
    p.add_argument("roms", nargs="+")
    p.set_defaults(func=cmd_rominfo)

    p = sub.add_parser("parse", help="parse a .bk2/.fm2 and report its inputs")
    _add_movie_args(p)
    p.add_argument("--json", action="store_true", help="emit JSON metadata")
    p.add_argument("--rom", default=None, help="also verify the movie's ROM fingerprint against this .nes")
    p.add_argument("--head", type=int, default=0, help="print the first N frames of input")
    p.add_argument("--dump-npy", default=None, help="save the (frames, buttons) bool array here")
    p.set_defaults(func=cmd_parse)

    p = sub.add_parser("replay", help="replay a .bk2 through nes-py")
    _add_movie_args(p)
    _add_replay_args(p)
    p.add_argument("--out", default=None, help="write a compressed .npz run here")
    p.add_argument("--frames-memmap", default=None, help="stream frames to this .npy instead of RAM")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("verify", help="replay and report whether the run synced")
    _add_movie_args(p)
    _add_replay_args(p)
    _add_verify_args(p)
    p.add_argument("--json", action="store_true", help="emit the report as JSON")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("run", help="parse + replay + verify, writing a dataset directory")
    _add_movie_args(p)
    _add_replay_args(p)
    _add_verify_args(p)
    p.add_argument("--out", required=True, help="output dataset directory")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("reference", help="record a known-good RAM trace")
    _add_movie_args(p)
    p.add_argument("--rom", required=True)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--allow-rom-mismatch", action="store_true")
    p.add_argument("--backend", choices=BACKENDS, default=DEFAULT_BACKEND)
    p.add_argument("--fceux-arg", action="append", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--out", required=True, help="destination .npz")
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("fetch", help="download movies from tasvideos.org")
    p.add_argument("--game-id", type=int, default=None, help="TASVideos game id (SMB is 1)")
    p.add_argument("--game-name", default=None, help="search for a game by name instead")
    p.add_argument("--system", default="NES", help="system code for --game-name (default NES)")
    p.add_argument("--out", default="data/movies", help="output directory")
    p.add_argument("--limit", type=int, default=None, help="stop after N kept movies")
    p.add_argument("--all-formats", action="store_true", help="keep non-bk2 movies too")
    p.add_argument("--no-publications", action="store_true", help="user files only")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_fetch)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except MovieFormatError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except RomMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (ReplayError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
