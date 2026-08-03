# tasdata — TAS imitation-learning data pipeline

Turns published tool-assisted speedruns into `(observation, action)` training data:
parse a movie file, replay it deterministically on an NES emulator, capture
downscaled grayscale frames and a RAM trace, and **verify the replay actually
stayed in sync** before you train on it.

No model code lives here by design. This package only produces data.

macOS / Apple Silicon, Python 3.11.

---

## The three stages

| Stage | Module | What it does |
| --- | --- | --- |
| 1. Parse | `tasdata/fm2.py`, `tasdata/bk2.py` | FCEUX `.fm2` and BizHawk `.bk2` → `(n_frames, n_buttons)` bool numpy array |
| 2. Replay | `tasdata/fceux_backend.py` (default), `tasdata/replay.py` | Two interchangeable backends; captures 84×84 grayscale frames + RAM trace |
| 3. Verify | `tasdata/verify.py` | Pass/fail per run, with the frame where it diverged |

### Backends

Both implement the same interface (constructor keywords, `.replay()`,
`ReplayResult`), selected with `--backend`, so stages 1 and 3 never change.

| Backend | How | Verdict |
| --- | --- | --- |
| **`fceux`** (default) | FCEUX replays the `.fm2` itself via `--playmov`; we only capture | **Syncs.** 67,117 frames, 1-1 → 8-4 |
| `nes-py` | Feeds action bytes to nes-py frame by frame | Clears 1-1 frame-perfectly, then loses the level transition |

The `.fm2` was recorded *in FCEUX*, so FCEUX is in sync with it by construction.
Accuracy and compatibility are different properties — swapping in a "more
accurate" core is another gamble on this specific movie, not a fix.

Supporting modules:

| Module | Purpose |
| --- | --- |
| `tasdata/movie.py` | Format-neutral `Movie` type + `parse_movie()` dispatcher |
| `tasdata/formats.py` | Content-based format sniffing; precise errors for the ~18 other TAS formats |
| `tasdata/buttons.py` | Button names ⇄ nes-py action bytes |
| `tasdata/rom.py` | iNES header parsing and both ROM fingerprints |
| `tasdata/backends.py` | Backend registry / `get_replayer()` |
| `tasdata/ram.py` | Super Mario Bros. RAM map |
| `tasdata/tasvideos.py` | Download movies from tasvideos.org |
| `tasdata/cli.py` | Command line interface |

---

## Install

nes-py needs a C compiler (Xcode CLT) and **numpy < 2** — its ROM loader indexes
numpy scalars in a way numpy 2 rejects, and the `gym` it depends on is numpy-1
only. Python 3.12+ breaks the nes-py build, so use 3.11.

```bash
conda create -y -n tas python=3.11
conda activate tas
pip install -e '.[dev]'
brew install fceux
```

### FCEUX build — sync depends on this exact version

```
fceux binary : /opt/homebrew/bin/fceux
version      : 2.6.6
git rev      : 34eb7601c415b81901fd02afbd5cfdc84b5047ac
Qt 6.11.0 / SDL 2.32.10
```

Reprint at any time with `tasdata emuinfo`; it is also written into every run's
`manifest.json` as `"backend": "fceux 2.6.6"`. If a future run desyncs, compare
this first.

#### Left+Right / Up+Down must be enabled

FCEUX filters simultaneous opposite directions **by default**, and speedruns need
them: this warpless movie holds Left+Right together on **579 frames** and Up+Down
on 1.

| | |
| --- | --- |
| config key | `SDL.Input.EnableOppositeDirectionals` |
| config file | `~/.fceux/fceux.cfg` (line ~75) |
| default | `0` — **off** |
| GUI location | Config → Input → *Allow Left+Right / Up+Down* |
| CLI flag | `--opposite-directionals 1` (works, though undocumented in `--help`) |

The backend passes `--opposite-directionals 1` explicitly on **every** run,
alongside `--no-config 1` so runs neither depend on nor mutate your
`~/.fceux/fceux.cfg`. Note the flag turns out **not** to affect movie playback —
see "Confirming the L+R setting" below for the measurement and why.

#### FCEUX needs a real window

`QT_QPA_PLATFORM=offscreen` and `minimal` both crash this build in
`QOpenGLWidget: Failed to create context` (SIGSEGV, with or without
`--opengl 0`). The backend therefore runs FCEUX windowed by default; a window
appears for the duration of a run. `show_window=False` exists if you get a
working headless Qt.

## ROMs

**You must supply your own ROM.** None is bundled and `tasdata fetch` will not
download one.

The pipeline refuses to replay a movie whose recorded ROM fingerprint does not
match the ROM you pass (`--allow-rom-mismatch` overrides). This is not pedantry:
in testing, *every* SMB movie replayed against a non-matching dump died inside
1-1, while the matching dump cleared it.

The two formats fingerprint ROMs differently, and `tasdata rominfo` prints both:

```
$ tasdata rominfo super-mario-bros.nes
super-mario-bros.nes: 40976 bytes, mapper 0, PRG 2x16K, CHR 1x8K, NTSC (per header)
  sha1(file)    ab30029efec6ccfc5d65dfda7fbc6e6489a80805   <- bk2 'SHA1'
  md5(prg+chr)  ba39dde63ab209b1bc751e0535e72b18   <- fm2 'romChecksum'
  fm2 checksum  base64:ujnd5jqyCbG8dR4FNecrGA==
```

* `.bk2` stores SHA-1 of the **whole file**, iNES header included.
* `.fm2` stores `romChecksum base64:...`, the MD5 of **PRG + CHR only**, header
  stripped.

### Careful: `gym-super-mario-bros` ships the PAL ROM

`pip install gym-super-mario-bros` bundles `_roms/super-mario-bros.nes`, which is
convenient but is **`Super Mario Bros. (Europe) (Rev 0A)`** — the 50 Hz PAL dump
(`md5(prg+chr) = ba39dde6…`). Of 233 SMB `.fm2` movies on TASVideos, only the 3
explicitly PAL runs match it; 145 use the NTSC dump `8e363018…`. Its iNES header
also *claims* NTSC, so nes-py loads it without complaint.

Consequence: NTSC TASes cannot sync against that ROM, and PAL TASes cannot sync
either, because nes-py has no PAL timing. Use it for smoke tests only.

---

## Usage

```bash
# What format is this, really? (ignores the extension)
tasdata info data/movies/*.fm2

# Inspect a movie; --rom also verifies the fingerprint
tasdata parse data/movies/happylee_mars608-smb-warpless.fm2 --rom smb.nes

# Replay and save frames + RAM trace
tasdata replay movie.fm2 --rom smb.nes --observation 84x84 --out run.npz

# Replay and report sync (FCEUX by default)
tasdata verify movie.fm2 --rom smb.nes --expect 8-4

# Force a backend; nes-py is the PAL-pair regression check
tasdata verify movie.fm2 --rom smb.nes --backend nes-py

# Which FCEUX am I using?
tasdata emuinfo

# Everything, into a dataset directory
tasdata run movie.fm2 --rom smb.nes --out data/runs/warpless

# Record a known-good trace, then use it for frame-exact regression checks
tasdata reference movie.fm2 --rom smb.nes --out ref.npz
tasdata verify movie.fm2 --rom smb.nes --reference ref.npz

# Fetch movies (SMB is game id 1)
tasdata fetch --game-id 1 --out data/movies --limit 20
```

`verify` and `run` exit `0` on sync, `1` on desync, `2` on a bad file, `3` on a
ROM mismatch — so they drop straight into a shell loop or CI.

### Dataset layout written by `tasdata run`

```
frames.npy         uint8  (n_obs, H, W)      downscaled grayscale observations
actions.npy        uint8  (n_frames,)        action byte per frame (labels)
button_states.npy  bool   (n_frames, n_btn)  raw per-button matrix
trace.npy          int32  (n_frames, 13)     decoded RAM per frame
frame_indices.npy  int64  (n_obs,)           which frame each observation is
movie.json / sync.json / manifest.json
```

Frames dominate memory: 240×256 grayscale × 20 000 frames is 1.2 GB, versus
141 MB at 84×84. `run` streams `frames.npy` as a memmap; `--frame-skip N` thins
observations without affecting the emulation (inputs are always applied to every
frame), and `--no-frames` skips capture for sync-only checks.

---

## Parsing details

### `.fm2` (FCEUX)

Plain text: `key value` header lines, then one line per frame starting with `|`.

```
version 3
palFlag 0
romFilename Super Mario Bros. (JU) [!]
romChecksum base64:jjYwGG411HcjG/j9UOVM3Q==
port0 1
|0|........|........||
|0|R......A|........||
```

Frame lines are `|commands|port0|port1|port2|`. Each controller field is eight
characters in the fixed order **`RLDUTSBA`** — Right, Left, Down, Up, s**T**art,
**S**elect, B, A — where `.` is released. That order maps MSB→LSB directly onto
nes-py's action byte. Unconnected ports are empty strings, hence the trailing
`||`. Only the first controller field is decoded by default
(`first_controller_only=False` to keep the rest).

`commands` is a bitmask: 1 reset, 2 power, 4/8 FDS, 16 VS coin.

Flagged as notes on the parsed movie: `palFlag`, `binary 1` (rejected — packed
frame data), `FDS`, `NewPPU`, `fourscore`, `RAMInitOption`, `savestate` anchors,
and input on ports that were not decoded.

### `.bk2` (BizHawk)

A zip. `Input Log.txt` holds a self-describing log key:

```
LogKey:#Power|Reset|#P1 Up|P1 Down|P1 Left|P1 Right|P1 Start|P1 Select|P1 B|P1 A|
|..|........|
```

A leading `#` opens a group; each group is one `|`-delimited field on every frame
line, one character per button. Field widths are validated per frame and errors
name the offending frame.

### Wrappers and other formats

A single outer **gzip** layer (how TASVideos serves user files) and a **zip
holding exactly one movie** (how it serves publications) are both stripped
transparently.

Anything else — `.fcm`, `.fmv`, `.smv`, `.vbm`, `.gmv`, `.m64`, `.dtm`, `.dsm`,
`.lsmv`, `.ltm`, `.bkm`, … — raises `UnsupportedMovieFormatError` naming the
detected format and, where relevant, the console it targets:

```
lagoon.smv: detected Snes9x .smv movie, but this pipeline only reads BizHawk
.bk2 and FCEUX .fm2. That format targets SNES; a replay harness built on nes-py
cannot use it at all.
```

`.tasproj` is BizHawk's editor project — structurally a `.bk2` plus editor
state. It is detected separately and accepted only with `--allow-tasproj`.

---

## How the FCEUX backend captures

FCEUX plays the movie; we only record. A generated Lua script advances one frame
at a time and writes a fixed-layout binary record per frame into a FIFO, which the
Python side consumes and downscales:

```
magic "TF" | emu.framecount() uint32 | RAM 2048 B | screen_len uint32 | screen
```

The **full 2 KB of RAM** is shipped every frame so the *same* `probe` callback the
nes-py backend uses decodes it — which is why `verify.py` and the trace layout did
not change when this backend was added. The screen arrives as FCEUX's native GD
truecolor buffer (256×240 RGBA, 245,771 B) and is converted to grayscale and
downscaled in OpenCV.

### Which capture strategy, and why

Measured on this machine, 3,000 frames of the warpless movie:

| Approach | Throughput | Verdict |
| --- | --- | --- |
| RAM only (emulation floor) | **3,200 fps** | baseline |
| **(c) GD framebuffer → FIFO → cv2** | **1,079 fps** | **chosen** |
| (b) `gui.savescreenshotas` PNG per frame | ~725 fps capture, *plus* 67k PNG decodes | rejected |
| (a) native AVI dump + Lua RAM log + ffmpeg | n/a | **not possible** |

**(a) is not available.** FCEUX 2.6.6's AVI recorder is GUI-menu only. The config
keys exist (`SDL.AviDriver`, `SDL.AviVideoFormat`, `SDL.AviFFmpegVideoCodec`) but
2.6.6 exposes no CLI flag and no Lua function to start or stop recording, so it
cannot be driven non-interactively. Nothing to pair with ffmpeg.

**(b) was rejected on frame-exactness, not just speed.** `gui.savescreenshotas`
is off by one: asking for 20 frames writes `f000001…f000019` — 19 files, no
`f000000`. Pairing PNGs to frames would mean trusting a filename convention that
is already wrong by one. It also needs a 255 MB intermediate and a second decode
pass, and one 3,000-frame attempt wedged for over 10 minutes.

**(c) wins on every axis:** ~3× faster end-to-end than (b) including the
downscale, no intermediate files, and frame-exact *by construction* rather than by
convention, because each record carries its own `emu.framecount()`.

### Frame-exactness

Three assertions, all fatal:

1. every record's magic bytes must match — catches stream desync;
2. every record's `emu.framecount()` must equal its expected index — catches a
   dropped, duplicated or misaligned frame at the exact record it happens;
3. the total record count must equal the movie's frame count (67,117), and the
   observation count must equal `ceil(n_frames / frame_skip)`.

A short or skewed capture raises `FrameCountMismatchError` with FCEUX's log tail
rather than silently writing crooked training data. Assertion 2 earned its keep
immediately: FCEUX's *first* `frameadvance()` does not increment the counter (it
moves from "loaded, nothing run" to frame 0), so after iteration `i` the counter
reads exactly `i` — the assertion caught the naive `i + 1` guess on the first run.

### Confirming the L+R setting — and what it actually affects

It is enabled on every run, and verified enabled. But *measured*, it makes no
difference to `--playmov` sync:

```
opposite_directionals=True  -> synced=True  levels=32  furthest=8-4
opposite_directionals=False -> synced=True  levels=32  furthest=8-4
```

Both reach 8-4, on the movie that holds Left+Right together on 579 frames. The
reason: `SDL.Input.EnableOppositeDirectionals` filters **live controller input**
(keyboard/gamepad) on its way into the emulated pad. Movie playback writes the
recorded button bytes straight to the pad and never passes through that filter, so
a replayed L+R survives regardless.

It stays on because it costs nothing and is required the moment anyone records or
resumes a movie interactively in this FCEUX install — but do not count it as the
reason a replay syncs. Reproduce the comparison with:

```bash
python -c "
from tasdata.movie import parse_movie
from tasdata.fceux_backend import FceuxReplayer
from tasdata.verify import verify_smb
m = parse_movie('data/movies/happylee_mars608-smb-warpless.fm2')
for flag in (True, False):
    r = FceuxReplayer('smb.nes', capture_frames=False, opposite_directionals=flag).replay(m)
    print(flag, verify_smb(r.trace, expect_level='8-4').passed)
"
```

---

## Batch collection

Four commands, in order. Each writes a file the next one reads, so the whole thing is
reproducible and resumable.

```bash
# 1. Discover and filter. Downloads nothing you will not use; captures nothing.
tasdata curate --rom smb.nes --target 40 --pool data/movies/pool --plan data/shortlist.json

# 2. Measure what each movie ACTUALLY does. RAM only, no images, ~19 s per 67k frames.
tasdata measure --plan data/shortlist.json --rom smb.nes --update-plan

# 3. Capture. One failure never aborts the batch.
tasdata batch --plan data/shortlist.json --rom smb.nes --out data/runs

# 4. Analyse and freeze the split.
tasdata stats --runs data/runs --out data/stats.json
tasdata split --runs data/runs --out data/split.json
```

### Selection rules

| Rule | Effect |
| --- | --- |
| ROM identity | `romChecksum` must match the supplied ROM. NTSC only; PAL rejected. |
| Category exclusions | game-end-glitch, ACE, minimum-presses, minimum-A-presses, walkathon, maximum-score, maximum-coins |
| Coverage priority | warpless (32 levels) → all-items (32) → warps (8) |
| Obsoletion chains | `&showObsoleted=true`, walked backwards from each current record |
| Low-coverage cap | at most `target/4` runs visiting < 16 levels, so warps cannot fill every leftover slot |
| Dedup | byte-identical movies dropped by content hash |

Publications are preferred over user files: vetted, categorised, attributed.

### Measure before you capture

**Filename and branch categories are guesses. The RAM trace is ground truth.** Skipping
step 2 on this shortlist would have written roughly 4 GiB of unusable data and mislabelled
a third of the rest:

* **9 of 34 runs desync** — all user files. Five "warpless" ones reach 1–2 levels and die
  30–40 times.
* **10 of 34 categories were wrong.** Three files named `*glitchless*` are five-minute
  *warps* runs, not 32-level warpless ones. A 19,707-frame movie cannot cover 32 levels,
  and the arithmetic says so before the emulator does.

`measure --update-plan` rewrites each entry's category from its measured route and records
`declared_category`, `measured_levels`, `measured_route`, `furthest_level` and
`premeasured_synced`. Every one of those lands in the run's `manifest.json`, so training
code can subset by category or sync status without recapturing.

Routes are named from the measured distinct-level count: 30–32 → `warpless`, 7–12 →
`warps`, anything else → `partial-N`. Note the **warps route visits eight levels**
(1-1, 1-2, 4-1, 4-2, 8-1 … 8-4), not four.

### Actions are never thinned

`--frame-skip N` thins `frames.npy` only. `actions.npy`, `button_states.npy` and
`trace.npy` are always full rate: the inputs are the labels, and they cost ~3.3 MB against
~450 MB of images. Subsample images at training time, not collection time.

### Splitting

Whole runs are held out, never frames — adjacent frames are near-identical. Whole
**obsoletion chains** are held out too, for the same reason one level up: two re-records of
the same route are near-duplicates, so putting one in train and its sibling in test leaks
just as badly. Groups are stratified by category so val/test are not all warps.

`data/split.json` is **immutable**: it carries a sha256 and `write_split` refuses to
overwrite without `force=True`. Changing a split after anything has been trained against
it invalidates every comparison. `tasdata split --verify` re-checks the checksum.

---

## Stage 2: behavioural cloning

Data-production code lives in `tasdata/`; the learning code lives in `tasdata/bc/` and
depends on it one way only.

| Module | Purpose |
| --- | --- |
| `bc/tokens.py` | action byte ⇄ token, with the rare tail folded into `RARE` |
| `bc/data.py` | memory-mapped, frame-stacked dataset |
| `bc/model.py` | per-frame CNN → transformer over the frame window |
| `bc/train.py` | training loop, validation, checkpointing |
| `bc/baselines.py` | constant / marginal floors |
| `bc/live.py` | live play in FCEUX, policy in the loop |
| `bc/session.py` | one long-lived FCEUX; savestate resets, one-emulator cap |
| `bc/session_player.py` | episode playback on a session (replaces process-per-episode) |
| `bc/statelib.py` | which movie frames are valid rollout starts, and their hashes |
| `bc/sweep.py` | smoke-test gate + overnight sweep |
| `bc/report.py` | JSONL → markdown summary |

### Selection rules: per-button sampling works, frame-level sticky does not

With a Bernoulli head, how you turn 8 probabilities into a button press matters more than
the probabilities do:

| rule | pipe 1 cleared (arm A / arm B) | longest A hold |
| --- | --- | --- |
| threshold, calibrated per button | 0% / 0% | 263 / 308 frames |
| threshold + sticky 0.25 | 0% / 0% | 364 / 476 frames |
| **per-button sampling** (n=200) | **29.5% / 59.5%** | 10 / 16 frames |

Sticky was meant to lengthen holds and it does — catastrophically. In SMB you must
**release** A to jump again, so a 476-frame hold makes every subsequent jump impossible.
Thresholding has the same failure for the same reason: it is deterministic, so wherever
p(A) sits above the threshold it stays above it. Independent per-frame sampling reproduces
the expert's *distribution* of hold lengths instead of its mean, and that is what clears
the pipe. Sticky is retained only as a reported control.

```bash
tasdata bc-smoke  --rom smb.nes                 # the gate; run this first, always
tasdata bc-sweep  --rom smb.nes --steps 30000   # runs the gate again, then the sweep
tasdata bc-report                               # data/stage2_results.jsonl -> summary.md
```

### The action space

67 raw combinations, but 43 occur fewer than 100 times in 981k training frames, so
they collapse into one `RARE` token: **25 classes**. `RARE`'s representative press is
its most common member, so emitting it live does something plausible rather than
something arbitrary. The token table is persisted to `data/action_vocab.json` —
rebuilding it from a different run set would silently relabel everything.

### Frame stacking

Four 84×84 grayscale frames. One frame has no velocity and SMB is almost entirely
momentum: from a still image you cannot tell whether Mario is sprinting right or has
just turned around. Stacks are edge-padded at run starts and **never straddle two
runs**, which is a real hazard when the dataset is a concatenation of 20 movies.

### Live play is the metric

Accuracy is a trap here — 40% of frames are "no buttons", so a model that learned only
the prior scores 40%. So the headline is live play in FCEUX with the policy in the loop:
two FIFOs, Lua sends an observation and blocks on one action byte, Python decides.

Two choices worth stating plainly:

* **Actions are sampled, not argmaxed.** SMB is deterministic, so a greedy policy plays
  the identical run on every seed and n=1 no matter how many episodes you run.
* **The harness presses Start.** The game boots for real (no savestate), the harness
  holds nothing through the title screen and presses Start on frame 40 like every SMB
  TAS, then the policy drives. Left to itself a cloned policy would never press Start —
  the title screen is 40 of 981k training frames — and every episode would score zero
  for reasons that say nothing about the policy. `Start` and `Select` are also masked at
  the emulator boundary: pressing Start mid-level only pauses.

### Baselines

`always nothing`, `always Right+B`, `sample the marginal`, plus the **blind model** —
identical architecture with the image zeroed in both the data pipeline and the forward
pass. That last one is the important floor: it learns the action prior with no vision,
so if the sighted model does not clearly beat it, it never learned to see.

### The gate

`bc-smoke` trains on 1,000 frames for 50 steps and refuses to let the long run start
unless all four checks pass: loss decreases, a checkpoint saves *and reloads to
identical logits*, and one live episode completes with the policy actually in control.
`bc-sweep` runs it again before touching the sweep.

---

## What the sync verifier proves

A frame-exact sync check needs a reference trace from the emulator that recorded
the movie. Without BizHawk/FCEUX on hand, `verify_smb` checks **game-state
invariants that a synced TAS satisfies and a desynced one essentially never
does**:

| Check | Failure means |
| --- | --- |
| `game-started` | never left the title screen (wrong ROM, savestate anchor) |
| `level-monotonic` | level ordinal went backwards (warps forward are fine) |
| `no-deaths` | Mario died — **the sharpest desync signal in SMB** |
| `timer` | timer expired |
| `forward-progress` | x stopped advancing (advisory; `--strict-stall` to fail) |
| `level-coverage` | fewer than `--min-levels` levels visited |
| `expected-level` | never reached `--expect W-S` |

A pointwise failure (death, timer, backwards level) pins divergence to a real
frame and always outranks the aggregate checks — a death at frame 877 is more
useful than "never left 1-1".

**Passing means "this run progressed as a real playthrough would", not "this
matches BizHawk frame for frame."** For the stronger claim, use `--reference`:
`compare_traces` reports the first differing frame and column, which is what you
want in CI when changing the harness.

RAM addresses (`tasdata/ram.py`) follow the community SMB disassembly and match
`gym_super_mario_bros`, so traces are comparable to that environment's `info`
dicts: world `0x075f`, stage `0x075c`, area `0x0760`, x `0x006d`×256 + `0x0086`,
player state `0x000e`, lives `0x075a`, timer `0x07f8`.

The RAM probe is injected (`replay(..., probe=...)`), so the harness itself is
game-agnostic; only `ram.py` and `verify.py` know about Mario.

---

## Tests

```bash
pytest -q                                    # 140 tests
SMB_ROM=/path/to/smb.nes pytest -q           # also runs the emulator-backed tests
```

Parser, format-sniffing, ROM-fingerprint and verifier tests are pure-python and
build their fixtures in memory. Tests that boot the emulator are marked
`emulator` and skip without nes-py or a ROM (`SMB_ROM` env var, else
`gym_super_mario_bros`'s bundled one).

---

## Findings

Measured results — perceptual aliasing, the airborne fraction, the start-point defects,
selection rules, and the two failed Stage 3 teachers — are collected in [FINDINGS.md](FINDINGS.md).

## Status

**Collected: 34 runs captured, 25 verified in sync, 0 capture failures.**

```
$ tasdata batch --plan data/shortlist.json --rom smb.nes --out data/runs
attempted 34 | captured 34 | synced 25 | desynced 9 | failed 0
1,684,996 frames (7.8 h) | 11.19 GiB on disk | 30.2 min wall clock
```

| | all 34 | synced 25 (trainable) |
| --- | --- | --- |
| frames | 1,684,996 | 1,223,797 |
| effective frames after overlap | 835,628 (50.4% redundant) | 661,005 (46.0% redundant) |
| action vocabulary | 69 combinations | 67 combinations |
| Left+Right frames | 11,605 (0.689%) | 10,102 (0.825%) |
| Up+Down frames | 95 (0.006%) | 46 (0.004%) |

The RAM-only `measure` pass predicted all 9 desyncs and every level count exactly, so
capture confirmed rather than discovered. Desynced runs are kept and tagged
(`"synced": false` in the manifest) so they can be excluded by query rather than by
deletion — their `actions.npy` is still the authentic input log; only the frames are junk.

### Action vocabulary is tiny and brutally skewed

67 distinct combinations, but **eight cover 97.4% of frames** and 43 appear fewer than
100 times each:

| combination | frames | share |
| --- | --- | --- |
| (nothing) | 493,657 | 40.34% |
| Right+B | 374,892 | 30.63% |
| B | 113,293 | 9.26% |
| Right+B+A | 100,745 | 8.23% |
| A | 40,029 | 3.27% |
| B+A | 28,060 | 2.29% |
| Right | 27,052 | 2.21% |
| Left+B | 16,726 | 0.99% |

`Right+B` is "run right"; `B` alone is holding run speed mid-air. Anything involving
Start or Select is menu noise (fewer than 500 frames total).

### Hold lengths

```
button    presses      held   mean  med   p90   p99    max  1-frame taps
B           8,943   644,221  72.04    9   159  1074   6537    3,169
Right      17,471   519,166  29.72    1    53   580   2568    8,796
A          22,502   183,830   8.17    3    23    52    269    9,978
Left       11,237    31,580   2.81    1     4    31    679    9,225
Down        6,132     7,720   1.26    1     1     8     43    5,818
Up            194       607   3.13    2     7    20     32       95
```

Note the medians: `Right` and `Left` have a **median hold of 1 frame** despite `Right`
being held for 519k frames in total. TAS input is mostly single-frame taps punctuated by
long holds (max 2,568 frames of `Right`), which is a heavy-tailed distribution rather than
anything bell-shaped. `Down` is almost entirely 1-frame taps.

### Overlap within obsoletion chains

Consecutive re-records of the same route share most of their input:

```
pub 1106  vs pub 1194   94.62% of 67,729 frames
pub 1022  vs pub 1080   89.81% of 17,891 frames
pub 1080  vs pub 1330   87.09% of 17,869 frames
pub 1194  vs pub 1331   61.18% of 67,580 frames
pub 1331  vs pub 1349   59.13% of 67,413 frames
pub 3665  vs pub 3728   53.87% of 67,117 frames
pub 1962  vs pub 3665   53.86% of 67,204 frames
pub 1349  vs pub 1962   50.03% of 67,204 frames
pub 262   vs pub 1106   42.21% of 67,779 frames
pub 3648  vs pub 4313   17.15% of 71,438 frames
```

`pub 1194` contributes only **5.4% novel** input. Least novel: pub 1194 (5.4%),
pub 1080 (10.2%), pub 1330 (12.9%). Most novel: user 4686736457 (100%),
user 6389096164 (79.5%), pub 3648 (76.6%).

So **the honest dataset size is ~661k frames, not 1.22M** — 46% is redundant. Effective
size is computed greedily: each run counts
`n_frames x (1 - max agreement with any run already counted)`, using the single closest
predecessor so overlaps are not subtracted twice.

### The split

`data/split.json`, immutable, sha256-verified (`tasdata split --verify`).

| | runs | frames | share |
| --- | --- | --- | --- |
| train | 20 | 981,385 | 80.2% |
| val | 2 | 88,394 | 7.2% |
| test | 3 | 154,018 | 12.6% |

It does not hit 80/10/10 exactly, and cannot: the indivisible unit is an obsoletion
chain, and the SMB warpless publication chain is 8 runs and 540k frames on its own — 44%
of the trainable corpus in a single group. Groups are placed largest-first into whichever
bucket has the largest remaining deficit, which puts that chain in train where it belongs.

An earlier round-robin version dealt it into test and gave test **51.5%** of the frames.
Worth remembering that a chain-grouped split needs size-aware placement, not dealing.

---

