# Mario from a perfect teacher

**Can supervised learning alone — no policy gradient, no value bootstrapping — clear Super
Mario Bros 1-1 by imitating a flawless tool-assisted speedrun? And if not, *where exactly*
does it break?**

This repository is the measured answer. It contains a verified TAS→training-data pipeline, a
behavioural-cloning policy, and — the actual contribution — the controls that decide whether
any of it worked.

[![tests](https://github.com/arangoe036-ui/mario-imitation-learning/actions/workflows/tests.yml/badge.svg)](https://github.com/arangoe036-ui/mario-imitation-learning/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-blue)

---

## Two clips, and why one of them proves nothing

### Level 1-1, completed from the level start

![1-1 completed](gifs/01_completion_1-1.gif)

The policy reaches the flagpole and the game advances to World 1-2 (verified from the HUD in
the final frame, not inferred from a distance number).

This happens on **4 of 200 episodes — 2.0% [0.8%, 5.0%]**. A fixed-rate script that never looks
at the screen — the A-0.85 control described below — completes it **1 of 200 — 0.5%
[0.09%, 2.8%]**, Fisher **p = 0.372**, on a single training seed.

**So the completion is real, and it is not evidence of learned skill.** It is here because it is
the first thing anyone would put in a README, and because the control that empties it of meaning
is the point of the project.

### The Koopas — where learning beats its matched control

![Koopas cleared](gifs/02_koopas_cleared.gif)

Past the Koopas at x=1248. Against a **run-length script matched on the policy's own token
marginals** — identical action representation, but blind — the policy is
**+5.5 pp, 10 of 10 paired seeds, p = 0.0020**, at the design floor and surviving Bonferroni
correction **across the six measured walls: 0.0020 × 6 = 0.012 < 0.05**. (The ×6 family is the
conservative one. [`data/window_reweight_sweep.json`](data/window_reweight_sweep.json) shows the
Goomba, pipe 1 and pipe 2 are one measurement rather than three, leaving four independent
regions — but its corrected four-wall family is pipe 2, pipe 3, pipe 4 and the frontier and does
**not** include the Koopas, so ×6 is the correction actually computed for this claim.) The
matched script's mean A-hold across the ten arms is **5.5–8.3 frames**
([`data/round2_measure.json`](data/round2_measure.json)), which rules out the "long holds are
just improbable under i.i.d. sampling" objection.

**The mechanism was specified before the result: the Koopas move.** Screen-conditioning has to
pay where the obstacle moves, and a fixed token distribution cannot compete there. The early
obstacles are static geometry, which a blind baseline handles without seeing anything.

#### And a blind script that beats the policy at the same wall

**The claim is exactly "+5.5 pp against the representation-matched run-length script", and no
wider.** A cruder control does better than the policy at the Koopas. The **A-0.85** script holds
Right and B and, i.i.d. every frame and without ever looking at the screen, presses A with
probability 0.85 (Left 0.135, Down 0.086). Measured on the same basis as the policy — single life
from the level start, `STALL=6500 / CAP=12000`, n = 200 per arm, policy arm `P_84_cnn32_seed4` —
in [`data/route_audit.json`](data/route_audit.json):

| wall | policy | fixed-rate A-0.85 script |
|---|---|---|
| Goomba 320 | **82.0%** | 79.5% |
| pipe 3 735 | 47.5% | **57.5%** |
| pipe 4 975 | 29.5% | **37.5%** |
| **Koopas 1248** | 19.5% | **27.0%** |
| median furthest x | 723 | **827.5** |

**So the matched-control result is evidence about the matched comparison. It is not evidence that
the policy is the best blind-beating approach at the Koopas** — a simpler fixed-rate blind script
is 7.5 pp better there and reaches further overall. A-0.85 is not rate-matched to the policy and
is the best arm of a per-obstacle envelope, so it is a hard bar rather than a typical opponent;
it is also a real one, measured on the same basis, and it is not closed.

> **These are honest takes, not re-enactments.** A named episode cannot be re-filmed: SMB's
> pseudo-random state advances with total frames elapsed and survives a level restart, so an
> episode's outcome depends on the session's entire history. 200 episodes were filmed live and
> the takes matching each claim were kept. Full provenance in [`gifs/manifest.json`](gifs/manifest.json).

---

## The result

### The negative, with its mechanism

**Training reads 981,385 frames of flawless play.** Three corpus numbers get confused, so all
three are named: **1,684,996** frames is the whole 34-run capture, **1,223,797** is the 25 runs
that synced ([`data/stats_synced.json`](data/stats_synced.json)), and **981,385** is the 20-run
expert-train split — which is the one the training loader actually reads, per
[`data/corpus_composition.json`](data/corpus_composition.json). Those frames become 77,916
run-length training samples.

A perfect teacher never fails, so the data contains no example of getting out of trouble. The
policy therefore never observes a recovery, and the moment it leaves the expert's state
distribution it has nothing to imitate.

**"Zero deaths and zero recoveries" is withdrawn as a measured claim.** Nothing in `data/` counts
either: `stats_synced.json` has no death or recovery field, and `corpus_composition.json` cuts
the other way — the dataset does **not** filter to in-control frames, so about 3.3% of the 77,916
training samples are pregame frames, level transitions and *death animations*. What is measured
is that the teacher is a completed TAS, which is why there is no recovery to imitate; the count
of zero was prose.

The signature is visible in training: **cross-entropy falls monotonically 4.033 → 1.228 across
the steps ladder while play does not improve.** But the peak is weaker than "the measured optimum
is 1,000 steps" sounds. [`data/depth_vs_steps.json`](data/depth_vs_steps.json) is **n = 100** —
not the project's usual 200, and the artifact states its figures are therefore not comparable to
them — and is **a single training seed beyond 3,000 steps**. Its median furthest x runs 722.0 at
500 steps, **722.5 at 1,000**, 716.5 at 2,000, 698.5 at 3,000, **722.0 at 15,000**, 468.0 at
45,000 and 689.0 again at 60,000. So 1,000 steps leads the 15,000-step rung by **0.5 px of
median**, the artifact records `monotone_decline_from_first_rung: false`, and the collapse at
45,000 **recovers by 60,000**. 1,000 steps is the best rung on that ladder and the operating
point used throughout; the ladder cannot establish it as an optimum, and "optimising past it
makes the policy worse" is not what the curve shows.

Measured directly, with both of the artifact's numbers rather than the convenient one:
**imitation fidelity and task performance show no clear relationship.**
[`data/fidelity_vs_performance.json`](data/fidelity_vs_performance.json) reports
`corr_recall_pipe1 = −0.036` **and** `corr_exact_pipe1 = +0.405` over the same n = 29 points, and
its own verdict is "NO clear relationship at this sample size". Those points are unweighted and
heterogeneous — per-point n ranges from 60 to 200 — and one is tagged
`round1_contaminated_for_reference`. Copying the expert more accurately is not what makes the
policy play better; quoting r = −0.04 alone overstated how cleanly that was shown.

### Where it fails is positional, not gradual

720 episodes launched from 72 saved start positions across two independently trained networks:
**the policy stops at the same absolute positions no matter where it starts.** 650 pixels of
head start buys about 130 pixels of extra progress. (Measurement basis `conditional_on_arrival`;
[`data/reach_walls.json`](data/reach_walls.json) marks these figures as not comparable to the
n = 200 single-life numbers above.)

| started at x | median furthest x (net A) | (net B) |
|---|---|---|
| 0–200 | 701 | 723 |
| 200–350 | 716 | 722 |
| 350–500 | 707 | 722 |
| 500–650 | 819 | 864 |

Failures cluster at five named locations — pipe 3's face (720), pipe 4's face (912), the first
Goomba (288), the Koopas (1216–1248), and a fall (1504–1536) — not along a continuum. It does
not run out of competence; it arrives in good shape and fails at specific addresses.

**Two caveats the artifact carries.** First, the start library was **harvested from one policy's
own play** — `reach_walls.json` records it as "places that policy reached, not a sample of the
level". That is the confound sitting directly under "starting further along buys no extra reach":
the later start positions exist *because* a policy got there. Second,
[`scripts/hidden_area_check.py`](scripts/hidden_area_check.py) flags that stops binned at pipe 4's
face "may be pipe entries rather than stalls", since x is a different coordinate system inside a
bonus area. That one was checked and cleared —
[`data/hidden_area_check.json`](data/hidden_area_check.json) finds **0 of 720 episodes ever leave
area 1**, so the bins are genuine stalls — but the check is the reason the bins can be trusted,
not an assumption.

### Eight intervention families, closed by measurement

Observation · read-out · capacity · resolution · generation rule · corpus composition ·
objective · search-and-distil.

**This is not a failure to see, to read, to represent, to sample, or to search.** All five were
measured. Up-weighting exactly the frames where the policy fails does not help, and at the
strongest dose it hurts — but the strength matters and the whole ladder belongs in the sentence.
Past pipe 4, the pre-specified outcome: **1.5× is −1.3 pp (p = 0.373), 2.0× is −0.3 pp
(p = 0.844), 3.0× is +1.0 pp (p = 0.535), and 8.0× is −4.0 pp with 0 of 10 seeds up (p = 0.0020,
at the design floor, surviving Bonferroni)**. The mild ladder is null; only the 8.0× cell moves,
and [`data/window_reweight_sweep.json`](data/window_reweight_sweep.json) says why 8.0 was added:
"so that a flat sweep reads as 'reweighting does not work' rather than 'the manipulation was too
small to see'". The same artifact records the design limit — the intervention can move only
**295 of 77,916 samples (0.38%)**. Read it as a bounded null with one harmful extreme. The
mechanism is at least available: at a failure window the expert is executing flawless play from a
state the policy never occupies, which makes those frames the least transferable data in the
corpus.

### Two findings about the data itself, at the strength their designs support

- **The best demonstrations may not be the best teachers.** Within one obsoletion chain, at a
  matched 201,479 frames, the three oldest runs beat the three newest past pipe 1 in **3 of 3
  training seeds — 72.5 / 53.5 / 76.0% against 7.5 / 39.5 / 18.0%**
  ([`data/chain_position.jsonl`](data/chain_position.jsonl)). The "~3×" and the pooled
  `+45.7 pp [+40.5, +50.4]` in the log treat **episodes** as the independent unit while the arms
  differ **by training seed** — the same error that cost this project its largest claim, below.
  With the seed as the unit and three pairs, the smallest attainable two-sided sign-flip p is
  **0.25**: the direction is unanimous, the size is not established.
- **A quarter of the data trained the best player, on one run per subset.** Fixed-epoch scaling
  peaked at 25%: past pipe 1, 63.0% → **83.5%** → 71.5% → 70.5% at 10/25/50/100% of the corpus,
  n = 200 episodes each but **one training seed per point and no seed replication**
  ([`data/plots/scaling_curve.png`](data/plots/scaling_curve.png)). By the standard this README
  applies to everything else, that is a lead, not a result.

Both point the same way — **curate, do not accumulate** — and neither is established at the seed
level, which is the standard the rest of this document is held to.

---

## Claims withdrawn

Eight, each replaced by what the re-measurement showed:
`+80 pp` · `+6.3 pp` · "the head is the bottleneck" · "Down is the route" · "no timing
anywhere" · "the encoder collapses seed spread" · `8.9%` · the 7-wall Bonferroni family.

The largest was a statistics error made twice: an interval pooled *episodes* as the independent
unit when the two arms differed **by training seed**. Recomputed with the seed as the unit and an
exact permutation test, a "+13 pp encoder improvement" became 12.1 pp at **p = 0.175** — where
the smallest attainable p was 0.008, so the test had the power to find a separation and did not.

Full history, including the wrong turns, in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

## Seven measurement defects, and the one defence that caught them

Every one ran to completion, produced plausible numbers, and raised no error. None would have
been caught by a test asking "did it crash."

| defect | what it would have produced |
|---|---|
| stall terminator censoring reach | every reach figure understated |
| area-union bug | invented a bonus-area route |
| 11-state probe | inverted a conclusion |
| pipe-3 sweep | an action space that excluded the answer |
| local `STALL` copies | a 35-point discrepancy |
| frames-vs-samples confusion | a mis-scaled corpus |
| degenerate Bonferroni family | a correction that corrected nothing |

**What caught them was the same thing every time: cross-checking one measurement against an
*independent* measurement of the same quantity, then investigating the disagreement instead of
keeping the more convenient number.** That is the transferable finding.

---

## How it works

| Stage | Module | What it does |
|---|---|---|
| 1. Parse | `tasdata/fm2.py`, `tasdata/bk2.py` | FCEUX `.fm2` / BizHawk `.bk2` → `(n_frames, n_buttons)` bool array |
| 2. Replay | `tasdata/fceux_backend.py` | Deterministic replay; captures 84×84 grayscale frames + RAM trace |
| 3. Verify | `tasdata/verify.py` | Pass/fail per run, with the frame where it diverged |
| 4. Clone | `tasdata/bc/` | Behavioural cloning + live evaluation against scripted baselines |

Verified end to end on the primary training run: `SYNC: PASS`, all 32 levels 1-1 → 8-4, 67,117
frames. Over the whole capture set it is **25 of 34 movies synced, 9 desynced, 0 failed**
([`data/batch_report.json`](data/batch_report.json)). A desync is detected and the run is dropped
from the training split rather than used — which is why the synced corpus is 25 runs, not 34.

**Policy:** 84×84 × 4-frame stack → CNN(32,64,64) → 1-layer transformer, `d_model=64` → linear
head. 325,964 parameters. Plain cross-entropy, 1,000 steps (the best rung of the steps ladder,
not a resolvable optimum — see above), batch 64, lr 3e-4, capped run-length sampling at
temperature 0.7.

## Repository layout

**One experiment, one script, one artifact.** `scripts/<name>.py` writes `data/<name>.json` —
83 of the 107 experiment scripts pair with a same-named result file, and
[`data/README.md`](data/README.md) indexes the artifacts by the claim they support.

**Four cited artifacts have no producing script in this clone:** `reach_walls.json`,
`stats_synced.json`, `seed_variance_permutation.json` and `stats_all.json`. `stats_synced.json`
and `stats_all.json` are `tasdata stats` output (`--synced-only` and not), so their generator is
the CLI rather than a script — but it needs the uncommitted capture set. For `reach_walls.json`
and `seed_variance_permutation.json` the generator is simply not in the repository. Every number
quoted above traces to an artifact; these four are artifacts you cannot re-run from here.

```
tasdata/     the package — parse, replay, verify, clone   (46 files)
tests/       324 collected with torch, 283 without        (14 files)
scripts/     one per experiment                          (107 files)
data/        one artifact per experiment                 (158 files)
docs/        research log + the superseded findings doc
gifs/        the two clips, with provenance in manifest.json
```

**What the tests badge covers.** `pytest` collects **324 tests** with torch installed and **283**
without it — `tests/test_bc.py` skips at import when torch is missing, taking 41 tests with it.
**No test needs a ROM or an emulator to pass**; the ones that would use them skip instead.
Measured here with torch present but without FCEUX, nes-py or a ROM:
**292 passed, 32 skipped, 0 failed, ~2 s.** The 32 skips are 13 that need
the `fceux` binary, 9 that need FCEUX plus the ROM plus the expert movie, 8 that need nes-py, and
2 that need `fcntl`, which is POSIX-only. CI installs `-e ".[dev]"`, so torch is present there and
the badge is over the full 324; the FCEUX-dependent tests still skip, because FCEUX needs a real
window.

Nothing here was computed in a notebook or by hand.

## Reproducing

**You must supply your own legally-obtained ROM.** No ROM is distributed here. Place an NTSC
Super Mario Bros dump at `smb.nes`.

What the pipeline actually verifies is **each movie's own `romChecksum` header against the ROM you
supply** — `md5(prg+chr)` for `.fm2`, `sha1(file)` for `.bk2` — and it refuses to replay on a
mismatch, because a mismatch makes every SMB TAS die inside 1-1. Two consequences worth stating:
a movie whose header carries **no** fingerprint is recorded as a note ("header has no
romChecksum; ROM identity cannot be verified") and replayed anyway, and `--allow-rom-mismatch`
downgrades the refusal to a note on purpose. `8e3630186e35d477231bf8fd50e54cdd` is the NTSC dump
the warpless publication needs; it appears as a literal only in the tests, and is not a constant
the pipeline enforces.

```bash
python -m venv .venv && . .venv/bin/activate   # Python 3.11; numpy<2 is load-bearing
pip install -e ".[dev]"                        # runtime deps + pytest + torch
pytest                                         # no ROM, no emulator needed
```

There is no `environment.yml`. The dependency lists are [`pyproject.toml`](pyproject.toml)
(runtime, plus the `dev` and `bc` extras) and [`requirements.txt`](requirements.txt) — the same
set, flat, with the FCEUX notes.

**Rebuilding the capture set needs movies this repository does not distribute.**
`data/movies/pool/` is gitignored, so all 34 `selected[].path` entries in `data/shortlist.json`
point at files a fresh clone does not have — and a missing movie prints one `FAIL` line while the
run still reports overall success, so a batch over an empty pool looks like it worked. Download
the publications yourself first (IDs and provenance in
[`data/movies/README.md`](data/movies/README.md)) into `data/movies/pool/`, then:

```bash
tasdata batch --plan data/shortlist.json --rom smb.nes --out data/runs   # ~11 GB
```

The capture report defaults to `<out>/batch_report.json`, deliberately **not**
`data/batch_report.json` — that file is the committed evidence for "25 of 34 synced" and a rerun
must not be able to overwrite it.

**Platform.** Capture and live evaluation need **FCEUX 2.6.6** and a real window — headless Qt
platforms segfault — and the evaluation harness's one-emulator lock is `fcntl.flock`, so that path
is macOS/Linux only (it was developed on macOS). Parsing, verification and the whole test suite
run anywhere, Windows included. Captured run data is not committed, and is regenerable.

## Related work

Neural networks have played Mario since MarI/O (2015), and `gym-super-mario-bros` exists
because reinforcement learning on this game is well-trodden. **This project asks a different
question.** RL agents explore and receive reward; this policy only ever sees demonstrations and
cannot discover anything the teacher did not do. The comparison of interest is therefore not
against an RL agent but against *scripted baselines that also cannot learn* — which is why the
controls throughout are three blind scripts: the three-button coin-flip script, the
representation-matched run-length script, and the fixed-rate A-0.85 script, which beats the policy
at pipe 3, pipe 4 and the Koopas.

## Credits

TAS movies are the work of their authors and are used as input data, not claimed as part of this
work — in particular **happylee**, whose `happylee_mars608-smb-warpless.fm2` (TASVideos
publication 3728) is the primary training run. Provenance for every movie is in
`data/movies/README.md`.

## Status

There is no assembled result document; this README is the summary. Everything stated above is
current as of block 66 and is drawn from [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md) and
[`gifs/manifest.json`](gifs/manifest.json). [`docs/FINDINGS_2026-08-04.md`](docs/FINDINGS_2026-08-04.md)
is archived and superseded — it is kept for provenance, not as a current claim.

## License

Apache 2.0 — code only. TAS movie files belong to their respective authors.
