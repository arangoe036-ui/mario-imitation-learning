# TAS imitation learning — project handoff

Written for an agent picking this up cold. Everything stated as a number here was measured
and is reproducible from a named script; where something is uncertain it says so.

Repo: `~/Desktop/tas-pipeline`. macOS / Apple Silicon, Python 3.11, conda env `tas`.

---

## 1. What the project is

Train a neural network to play Super Mario Bros by imitating published tool-assisted
speedruns (TASes), and find out where imitation breaks down.

The pipeline is: TAS movie file → deterministic emulator replay → `(84×84 grayscale frame,
8-button action)` pairs → behavioural cloning → live play in the same emulator.

The interesting question turned out not to be "can it imitate" (it can) but **"does
imitating better make it play better"** — and the answer is more complicated than expected
(§7).

### Stage structure

| Stage | Status | What it is |
| --- | --- | --- |
| 1. Data pipeline | **done** | parse → replay → verify → 34 captured runs, 11 GB |
| 2. Behavioural cloning | **done, frozen** | supervised learning from expert frames |
| 3. Beyond imitation | **in progress** | arm A self-imitation (works); arm B pseudo-expert (3 failures) |

---

## 2. Environment: the constraints that shape everything

These are not preferences. Each was learned by something breaking.

**One emulator process at a time.** FCEUX is driven headless-ish via Lua over a FIFO.
Running more than one concurrently reintroduces an OpenGL race that corrupts runs. A
`fcntl.flock` on `~/.tasdata_fceux.lock` enforces a hard cap of 1 (`tasdata/bc/session.py`).
Every emulator job must therefore run **serially**. Parallelism must come from batching
inside one session, not from more processes.

**Never probe MPS.** Calling `torch.backends.mps.is_available()` permanently poisons every
FCEUX child process launched afterwards in that session into broken software OpenGL. It is
not undone by `empty_cache()`. All training here is **CPU only** and deliberate about it
(`pick_device` will not probe unless explicitly asked). This is why training is slow
(~0.2–0.5 s/step) and why the machine is the bottleneck.

**FCEUX needs a real window.** `QT_QPA_PLATFORM=offscreen` and `minimal` both segfault this
build. A window appears during runs. Investigated, left alone.

**FCEUX version is load-bearing.** 2.6.6, git rev `34eb7601…`, from Homebrew. Sync was
established against this build. If a run desyncs, check this first. It is recorded in every
run's `manifest.json`.

**numpy < 2**, because nes-py's ROM loader breaks under numpy 2. nes-py is only kept as a
regression check; FCEUX is the real backend.

**`--opposite-directionals 1`** is passed on every run. FCEUX filters simultaneous Left+Right
by default and the warpless TAS uses it on 579 frames.

**DataLoader `num_workers>0` hangs** when launched from a heredoc (spawn re-imports
`__main__`). Run from a real file. This bit twice; all loaders now use `num_workers=0`.

**ROM is not in the repo.** `smb.nes`, md5(prg+chr) `8e3630186e35d477231bf8fd50e54cdd`. Note
`gym-super-mario-bros` bundles the **PAL** ROM, which matches only 3 of 233 SMB movies —
do not use it.

---

## 3. Module map

```
tasdata/
  formats.py        content-based movie format sniffing (not extension-based)
  fm2.py            FCEUX .fm2 parser (text; RLDUTSBA order, T=sTart S=Select)
  bk2.py            BizHawk .bk2 parser (zip + Input Log.txt)
  movie.py          format-neutral Movie type + parse_movie() dispatcher
  rom.py            iNES parsing; both fingerprints (sha1-file for bk2, md5-prgchr for fm2)
  replay.py         nes-py backend + _resize_gray (the 84x84 downscale)
  fceux_backend.py  FCEUX backend: Lua writes fixed records to a FIFO
  verify.py         sync verification (world/level progression, x, deaths, timer)
  ram.py            SMB RAM map -> decoded state per frame
  dataset.py        on-disk run format; LoadedRun; write/load
  curate.py         TASVideos discovery, obsoletion chains, .fcm conversion
  batch.py          batch capture; one failure never aborts the batch
  analyze.py        dedup, effective dataset size, the immutable split
  bc/
    tokens.py       action byte <-> token; 25-class vocab (24 frequent + RARE)
    data.py         FrameStackDataset (memmapped, frame-stacked, label_offset=1)
    model.py        BCPolicy: CNN encoder -> transformer over frame window
    bernoulli.py    8-Bernoulli head, onset reweighting, threshold calibration
    train.py        Stage 2 training loop
    live.py         legacy per-episode player + EpisodeResult (still the metric type)
    session.py      FceuxSession: ONE long-lived FCEUX, savestate resets, flock cap
    session_player.py  episode playback on a session  <-- use this, not live.py
    statelib.py     which movie frames are valid rollout starts; RAM + frame hashes
    retrieval.py    Stage 3 teacher #1 (failed)
    oracle.py       Stage 3 teacher #2/#3 (failed)
    stage3_train.py fine-tune + calibrate helpers for self-imitation rounds
    overnight_lib.py shared: calibration, onset metrics, live eval, CIs, training
scripts/            orchestrators and one-off experiments (see §9)
```

**Dependency direction is one-way**: `tasdata/bc/` may import `tasdata/`, never the reverse.

---

## 4. Data

**34 runs, 11 GB, ~981k training frames.** Captured from TASVideos publications and user
files. Every frame's action is stored — no frame-skip on actions, ever.

- `data/runs/<id>/` — `frames.npy` (uint8, n×84×84), `actions.npy` (uint8 action byte per
  frame), `trace.npy` (decoded RAM per frame), `manifest.json`
- `data/split.json` — **immutable**, sha256-stamped. Whole runs held out, never frames.
  Obsoletion chains kept together. train 20 / val 2 / test 3 (25 of 34 runs; the rest are
  partial runs excluded from the split).
- `data/action_vocab.json` — the 25-token vocabulary. Rebuilding it from a different run
  set would silently relabel everything.
- `data/state_index.json` — 532 savestate start points (32 level starts + 500 filtered
  trajectory points), each with a RAM hash **and** a frame hash.

### Label convention (easy to get wrong)

The action for the observation at frame `i` is `actions[i+1]`, i.e. `label_offset=1`. The
observation is what Mario saw *before* the action was applied. Getting this wrong produced a
model that predicted the action already taken (verified fix: 400/400 jump onsets correctly
labelled vs 0/400 before).

### Category labels are claims, not measurements

Audited (`scripts/audit_categories.py`): **4 of 34 manifests mismatch** their measured
route/level count. `warps-glitchless` means *glitchless warps*, not warpless-glitchless —
misreading it wasted an experiment. **"Glitchless" is unverifiable by this pipeline**;
nothing measures glitch use. Always check `measured_route` and `measured_levels`, never the
declared `category`.

---

## 5. The model

`BCPolicy` (`tasdata/bc/model.py`): 4 stacked 84×84 grayscale frames → small CNN encoder
(16/32/32 channels) → 1-layer transformer (d_model 64, 2 heads) over the frame window →
head. ~small by design; started smaller than felt right and never needed to grow.

**Two heads were tried:**

- **Categorical (25-way softmax).** Fails by *vote-splitting*: the four A-containing tokens
  each individually lose to `Right+B` under argmax, so A was emitted on 0.03% of frames
  despite the model carrying real signal about it.
- **Bernoulli (8 independent sigmoids)** — current. Fixes vote-splitting. Requires
  per-button **threshold calibration**: do not default to 0.5. Thresholds are swept so the
  realized press rate matches the expert's (A calibrates to 0.23–0.33, never 0.5).

**Previous actions as input: tried and rejected.** k=4 previous actions gave 95.84% accuracy
with a 98.4% copycat rate — it emitted "no buttons" on all 1,500 frames of a live episode.
Dropout 0.25 was insufficient. The input is dropped entirely.

### Selection rule matters more than the probabilities

| rule | pipe 1 cleared | longest A hold |
| --- | --- | --- |
| threshold (deterministic) | 0% | 263–308 frames |
| threshold + sticky 0.25 | 0% | 364–476 frames |
| **per-button sampling** | **59.5%** | 10–16 frames |

In SMB you must **release** A to jump again. Deterministic rules produce stuck holds that
make every subsequent jump impossible. Sampling reproduces the expert's *distribution* of
hold lengths rather than its mean. **Always evaluate with `selection="sample"`.**

---

## 6. Evaluation protocol (use this; earlier numbers are not comparable)

Live play from savestates on one persistent FCEUX, per-button sampling, **n=200 seeds**,
start points 1-1 and 2-1, Wilson intervals on rates and bootstrap on medians.

**Metrics.** `pipe1_rate` (binary, 1-1 starts only), `x_median`, `furthest_level`,
`longest_a_hold`, per-button hold stats, and the **failure taxonomy**: every episode is
classified `enemy_contact` / `pit` / `timer` / `stuck_terrain` / `game_over` /
`budget_reached`. Distance alone cannot distinguish standing still from dying, and those
need opposite fixes.

**Offline metrics.** A-onset recall at a threshold calibrated to the expert's press rate —
*not* raw accuracy. Accuracy is dominated by the ~85% of frames where the answer is "keep
holding what you were holding"; a do-nothing policy scores 36%.

**Calibration procedure (must be identical everywhere or numbers are not comparable):**
thresholds calibrated on a **random** subset of TRAIN against the expert's own per-button
press rates; onset recall measured on a **contiguous** slice of VAL (an onset needs the
previous frame). Never calibrate on val.

---

## 7. Results

### Stage 2, all re-measured under one calibration method

| model | head | A-onset recall | exact match |
| --- | --- | --- | --- |
| blind (control, image zeroed) | categorical | **0.0%** | 0.5% |
| categorical (small, lr 3e-4) | categorical | 21.9% | 67.5% |
| categorical (tiny, lr 1e-3) | categorical | 24.9% | 69.1% |
| bernoulli only (arm A) | bernoulli | 29.7% | 66.3% |
| bernoulli + onset reweight 10× (arm B) | bernoulli | **50.0%** | 64.5% |

The blind control at exactly 0.0% is the sanity check the measurement needed. Note **exact
match runs opposite to onset recall** — fitting the frames that matter costs accuracy on the
frames that don't.

Live, n=200: arm B **59.5%** pipe 1 vs arm A **29.5%**, difference **+30.0 pp
[+20.4, +38.8]** (Newcombe). Onset reweighting transfers to live play.

### Stage 3 arm A — self-imitation works

Roll out from filtered expert start points, keep the top quarter by progress-from-start, add
to training, refit. Every round recalibrated, n=200.

| round | expert:self | pipe 1 (95% CI) | A-onset recall |
| --- | --- | --- | --- |
| stage 2 baseline | — | 59.5% [53, 66] | 50.0% |
| round 2 | 3:1 | 92.0% [87, 95] | 47.3% |
| round 3 | 3:1 | 96.5% [93, 98] | 44.1% |
| round 2 | 1:1 | 96.5% [93, 98] | 38.8% |
| round 3 | 1:1 | **99.0% [96, 100]** | 39.6% |

**But `x_median` never moves off 594–595 in any round.** It solved pipe 1 and cannot touch
pipe 2. Self-imitation improved the obstacle it could already sometimes pass and made zero
progress on the next one — consistent with a filter that rewards incremental progress and
therefore can never reward a jump that never occurs.

**Do not use the acceptance rate as a health metric.** It was implemented as a fixed top-25%
quantile, so it reports ~25% every round regardless of behaviour and structurally cannot
detect the loop grading itself against a declining standard. Use the **score cutoff**, an
absolute threshold: it rose 289 → 362 → 416 across rounds (median progress 94 → 138 → 184),
so the standard got harder.

### The fidelity/performance relationship — narrower than it looks

| set | n | r(A-onset recall, pipe1) |
| --- | --- | --- |
| all checkpoints pooled | 17 | **−0.10** (nothing) |
| arm A self-imitation lineage | 5 | **−0.78** |
| data-scaling family | 4 | **+0.63** |

The anti-correlation is real **within a self-imitation lineage** and absent when pooled.
Correct statement: *successive self-imitation rounds buy task performance by drifting away
from the expert.* That is a property of the training loop, not a law about imitation
learning. Do not overclaim it.

### Data scaling (fixed compute — being redone at fixed epochs)

10% → 60.8% pipe1, 25% → 75.0%, 50% → 65.0%, 100% → 56.7%. Peaks at 25%. **Not citable as
stated**: at a fixed 2,000 steps, larger subsets get proportionally fewer passes, so this
measures "best use of a fixed budget", not "more data is harmful". A fixed-epoch rerun is
queued.

### Terrain

- **Pipe 1** (x≈435, cleared past 470): expert uses a median **2**-frame A-hold. Model can
  do this.
- **Pipe 2** (x≈594, cleared past 630): expert uses a median **18**-frame hold, p90 47, max
  72. Model's max hold is **3–8 frames**. A tap clears pipe 1; pipe 2 needs a sustained hold.
- **2-1 wall at x=530**: **blocks, does not kill.** Holding Right+B reaches x=306 and never
  dies; adding a periodic jump reaches 531. All 30 policy episodes ended in the stall
  detector with 0.1 mean deaths. The policy is standing still, not dying.

---

## 8. Three failed Stage 3 teachers (pattern, not bad luck)

All three were validated against a held-out expert **before** generating any training data.
All three failed on the same axis: **jump timing**.

1. **Retrieval pseudo-expert.** Index every expert frame by quantised
   `(world, level, x, y, player_state)`; retrieve what the expert did there. A-onset recall
   ≤11.9%; 59–87% of hit states carry contradictory expert actions.
2. **Search oracle, fixed Right+B continuation.** Force A on vs off, roll forward 60 frames,
   compare progress. Agreement at A-onsets **46.8–54.2%** — chance — across horizons
   60/120/180 and both progress measures. Over-jumped 4–5×.
3. **Search oracle, arm-B-policy continuation.** Onset agreement rose to **63.0%** (so the
   continuation policy *was* the binding constraint) but it jumped on 49–61% of frames
   against the expert's 6.0%, and overall agreement fell to 39.8%.

**Root cause, common to 2 and 3:** over a 1–2 second horizon, jumping in SMB is nearly free —
you keep horizontal momentum while airborne — so a progress-maximising comparison says
"jump" whenever it says anything. The expert jumps rarely because most jumps are
*unnecessary*, which is a fact about its own future trajectory, not about local cost.
Distance over a short horizon cannot express that.

A margin-calibrated variant (label "jump" only when it beats not-jumping by >M pixels, M
swept so the realized jump rate matches the expert's 6.0%) is queued as the final attempt.
**Hard stop agreed with the user: below 70% onset agreement at a matched jump rate, the
oracle is dead and gets written up as the third failed teacher.**

---

## 9. Silent failures — the recurring pattern (read this)

Every one ran to completion, produced plausible numbers, and raised no error.

| failure | what it looked like | what caught it |
| --- | --- | --- |
| **Double normalization** | `FrameStackDataset` already returns float32 in [0,1]; code divided by 255 again. Model emitted a **constant** p(A)=0.00710 (std 1e-5) for every frame. Calibration, thresholding, recall and exact match all computed cleanly on it and reported 0.0% | a behavioural measurement disagreeing: 53% pipe clearance from a checkpoint scoring 0% recall |
| Attract-mode contamination | a do-nothing policy "reached the flagpole" | watching the video |
| Level starts past the pipe | every arm scored 100% on "cleared pipe 1" (start was x=2616) | a test asserting the start was where it claimed |
| Start before control handover | frame 42 has `pregame=1, player_state=0x08, x=40` but is boot-time transient; 60 frames of Right+B leaves x at 0 | holding Right+B and finding x never moved |
| Symmetric ground filter | required y stable *forward* too, which excludes every A-onset by construction — 216 onsets became 24 | the onset count being implausible |
| Category labels | `warps-glitchless` read as warpless-glitchless | auditing labels against measured routes |
| Vote-splitting | 25-way argmax emitted A on 0.03% of frames | per-class recall, not accuracy |

**The pattern:** a stage that silently degrades its input produces confident, well-formed,
wrong numbers downstream. The defence that worked every single time was **not** a unit test —
it was cross-checking a metric against an independent measurement of the same thing and
investigating the disagreement rather than the more convenient number.

Practical rule for this repo: **if an offline metric and a behavioural measurement disagree,
suspect the metric.**

---

## 10. What is running right now

Two detached jobs under `caffeinate -i`, started sequentially because of the one-emulator
cap. Both stream to JSONL and isolate task failures.

| job | pid file | log | status |
| --- | --- | --- | --- |
| chain-position experiment | `data/chain_position.pid` | `data/chain_position.log` | arm 2 of 6 |
| follow-up suite | `data/followup.pid` | `data/followup.log` | waiting on the emulator |

**Chain-position experiment** (`scripts/chain_position.py`) replaces a botched
glitchless-vs-glitchy comparison. Within the `warpless/3728` obsoletion chain, position 0 is
the current publication (fastest, most glitch-dependent) and higher positions are older
records it obsoleted. Arms: latest = pub-3728/3665/1962, earliest = pub-1194/1106/262.
Matched at 201,479 frames, 3 seeds, n=200. Same route, same 32 levels, so position is the
only thing varying. Hypothesis: older/less-optimised data trains a *better* live policy,
because frame-perfect glitch execution only works from states a learned policy cannot reach.
First data point: earliest seed 0 → 72.5% pipe1 (baseline 59.5%).

**Follow-up suite** (`scripts/followup.py`), in priority order:
1. failure taxonomy for round-3 and Stage 2 baseline, n=200, 1-1 and 2-1
2. pipe-2 ceiling — emulator sweep of A-hold 1–32 frames × 5 trigger positions; settles
   "impossible vs just hard" with ground truth
3. sustain diagnosis — is p(A) on *continuation* frames below p(A) at onsets? (suspicion:
   10× onset weighting implicitly taught initiation and un-taught sustain)
4. sustain arms: (a) reweight sustain too, (b) onset 3× instead of 10×, (d) control
5. data scaling at fixed epochs
6. oracle rerun, last, non-blocking

---

## 11. Open problems, in the order they matter

1. **Sustain.** The model taps (max 3–8 frames) where pipe 2 needs an 18-frame hold. This is
   the single blocker on progress past x=594. If (a)/(b) above don't fix it, the next option
   is **hold-duration modelling**: at an onset, predict how many frames to hold and *commit*,
   instead of re-deciding A every frame. Not built — it changes the action space (needs a
   second head and a different rollout loop), so it should only be built if reweighting
   fails.
2. **Self-imitation cannot invent behaviour.** Selection over rollouts can only amplify what
   already occurs in the population. If an 18-frame hold never occurs, no amount of filtering
   produces it. This is a real limit of the method and is why §11.1 is architectural, not a
   hyperparameter.
3. **Negative examples.** The expert never dies, so training contains no collisions and no
   approach to one, and arm A *discards* failed rollouts, making it worse. Proposal: keep
   failed rollouts, mark the 30 frames before each death, train against those actions
   (negative weight, or a separate danger head). **Gated on the taxonomy** — only worth doing
   if deaths actually dominate, which the 2-1 result suggests they may not.
4. **Third teacher / RL.** If the margin-calibrated oracle fails, the pseudo-expert framing
   is exhausted (three failures). The natural next move is RL from the arm B checkpoint,
   where progress is the reward and credit assignment is the algorithm's problem rather than
   a hand-built lookahead's.
5. **Evaluation breadth.** Everything is measured on 1-1 and 2-1. 30 other levels exist and
   the savestate library already indexes all 32.

---

## 12. Reproducing anything

```bash
conda activate tas
python scripts/build_state_index.py build   # rebuild savestates, assert RAM+frame hashes
python scripts/audit_categories.py          # category audit
python scripts/remeasure_recall.py          # all recall under one calibration
python scripts/arm_ab_power.py              # arm A vs B at n=200 with CIs
python scripts/pipe2_ceiling.py             # expert A-hold analysis (offline)
python scripts/fidelity_vs_performance.py   # the anti-correlation figure
python -m pytest tests/ -q                  # 324 tests, 12 launch a real emulator
```

Artifacts: `FINDINGS.md` (all results incl. negatives), `data/stage2_summary.md`,
`data/stage3_oracle_verdict.md`, `data/overnight_summary.md`, `data/plots/`.

**Before trusting any number, check which calibration produced it.** Anything measured
before the double-normalization fix used a different method and is not comparable; the
reissued table in `FINDINGS.md` is the authority.
