# Project log — Mario from a perfect teacher

Beat Super Mario Bros using only supervised learning. No policy gradient, no value
bootstrapping; every model update is next-token prediction. Search is permitted, because
search is not a gradient method — it explores, and its results are distilled back by
supervised learning.

This file is the narrative record: what changed, what was found, and **how we got there**,
including the wrong turns. `FINDINGS_2026-08-04.md` is the technical companion with the full
tables -- archived and superseded, kept for provenance rather than as a current claim.

Entries are append-only and dated. Nothing here is rewritten when a later result overturns
it — a superseding entry is added and both are cross-referenced, because the sequence of
being wrong and then correcting it is the actual content.

**Legend:** ✅ improvement · ❌ negative result / approach killed · ⚠️ measurement found broken
· 🔁 earlier entry superseded

---

## Stage 1 — Data pipeline

### ✅ 2026-07-29 — A TAS replays frame-exactly, and the emulator choice is the reason

**What.** 67,117 frames of a warpless Super Mario Bros speedrun replay in sync from 1-1 to
8-4, producing `(84×84 grayscale frame, 8-button action)` pairs.

**How we got there.** The first attempt used nes-py, a Python NES emulator. It cleared 1-1
frame-perfectly and then desynced at the level transition. The instinct is to reach for a
"more accurate" emulator; that is the wrong instinct. Accuracy and *compatibility* are
different properties. The movie was recorded in FCEUX, so FCEUX is in sync with it by
construction, and any other core is a fresh gamble on this specific file. We drove FCEUX
itself via Lua writing fixed-size records to a FIFO, and kept nes-py only as a regression
check.

Two traps found on the way. FCEUX filters simultaneous Left+Right by default, and this run
uses it on 579 frames — `--opposite-directionals 1` is now passed on every run. And the ROM
bundled with `gym-super-mario-bros` is the **PAL** dump despite an NTSC header: of 233 SMB
movies on TASVideos, only the 3 explicitly-PAL runs match it.

**Numbers.** 67,117 frames, 1-1 → 8-4, zero divergence. ROM md5(prg+chr)
`8e3630186e35d477231bf8fd50e54cdd`.

**Cost.** ~1 day including the nes-py dead end.

**Downstream.** Everything. FCEUX 2.6.6 (git `34eb7601c`) is now a pinned, load-bearing
dependency recorded in every run's manifest.

### ⚠️ 2026-07-29 — The verifier passed on 17,868 frames of the attract-mode demo

**What.** A run was captured, verified, and declared in sync. It contained no gameplay — it
was the title-screen demo.

**How we got there.** TASVideos publications download as `.fm2.zip`. Our parser transparently
unwrapped the zip; FCEUX did not, ignored the file it was handed, and sat on the attract mode
while we recorded frames. The frame count came out *correct*, so every frame-exactness
assertion passed. What caught it was watching the video.

**Downstream.** Container unwrapping before handing anything to FCEUX, plus an
`UnplayableMovieError` guard. First entry in what became a ten-item silent-failure list.

### ✅ 2026-07-30 — 34 runs, and a split that does not leak

**What.** A 1,223,797-frame corpus from 25 synced runs, held out by whole run *and* whole
obsoletion chain.

**How we got there.** Obsoletion chains — a publication and the older records it replaced —
are near-duplicates. Measured, chain siblings agree on up to 94% of actions. Holding out
frames, or even whole runs without their chain, leaks as badly as holding out adjacent
frames. The split allocator also had to be rewritten: round-robin dealing put an 8-run chain
into test and gave test 51.5% of all frames. Largest-deficit placement achieved 80.2/7.2/12.6.

**Numbers.** train 20 runs / 981,385 frames · val 2 / 88,394 · test 3 / 154,018. Effective
size 661,005 — **46% of the corpus is redundant**. Vocabulary 67 button combinations, 8 of
which cover 97.4% of frames.

**Downstream.** `data/split.json` is immutable and sha256-stamped.

---

## Stage 2 — Behavioural cloning

### ❌ 2026-07-31 — A 25-way action softmax cannot learn to jump, for a structural reason

**What.** The categorical head emitted the A button on **0.03%** of frames despite carrying
real signal about when to jump.

**How we got there.** Live play looked broken while validation accuracy looked fine (74%).
The accuracy was the prior: two action tokens cover 71% of frames, so a policy that never
jumps scores 36%. Looking at per-class recall instead of accuracy exposed *vote-splitting* —
the four A-containing tokens each individually lose to `Right+B` under argmax, so A never
wins even when the model believes in it.

**Downstream.** Replaced by 8 independent Bernoulli outputs, one per button. Established the
house rule: **never report accuracy without the blind, always-one-action and
marginal-sampling baselines beside it.**

### ✅ 2026-07-31 — Calibrate thresholds to the expert's press rate; 0.5 is never right

**What.** A-onset recall went from 0.00% to 45.5% with no change to the weights.

**How we got there.** With independent Bernoullis you need a threshold per button. The obvious
default is 0.5. The expert presses A on 15.25% of frames, and the model's p(A) is calibrated
to that rate — the correct A threshold is **0.23–0.42**, and 0.5 fires essentially never.
Sweeping each button's threshold so its realized press rate matches the expert's fixed it.

**Downstream.** Standing check: *before believing a metric, establish that it could have come
out differently.* Thresholds are recalibrated after every training round, never carried across
a checkpoint.

### ❌ 2026-07-31 — Feeding the model its own previous actions produces a perfect copycat

**What.** 95.84% validation accuracy, 98.4% copycat rate, and it pressed nothing at all on
all 1,500 frames of a live episode.

**How we got there.** Predicted in advance by the human, and it happened exactly: given the
previous action, "repeat it" is almost always right, so the model learns that and ignores the
screen. Dropout at 0.25 was not enough. The input was removed entirely.

### ✅ 2026-07-31 — Per-button sampling works; deterministic rules and sticky actions do not

**What.** Only one action-selection rule clears an obstacle. Thresholding and sticky produce
263–476 frame A-holds; sampling produces 10–16 and clears pipe 1.

**How we got there.** In SMB you must **release** A to jump again, so a stuck hold makes every
later jump impossible. Sticky was introduced to *lengthen* holds and did — catastrophically.
Sampling reproduces the expert's *distribution* of hold lengths rather than its mean.

**Numbers.** pipe 1 cleared, n=200: threshold 0%, threshold+sticky 0%, **per-button sampling
59.5%**.

### ✅ 2026-07-31 — Upweighting the frame a button turns on beats upweighting the frame

**What.** Arm B (onset reweighting ×10) vs arm A (Bernoulli only): **+30.0 pp
[+20.4, +38.8]** on pipe 1.

**How we got there.** 85% of frames are "keep holding what you were holding". Weighting whole
transition frames also inflates the loss on buttons that did not change; weighting only the
*specific button that turns on* does not.

**Numbers.** n=200 per arm. arm A 29.5% [23.6, 36.2], arm B 59.5% [52.6, 66.1]. A-onset
recall 29.7% → 50.0%.

### ⚠️ 2026-07-31 — Three evaluation start points were fake, and each faked a result

**What.** Every arm scored 100% on "cleared pipe 1" for a while. The start point was at
x=2616 — past both pipes.

**How we got there.** The savestate library picked the first frame passing a "grounded" filter.
A TAS is airborne 61.1% of the time, so in 1-1 the first grounded frame is deep into the level.
Two more followed: seven `W-1` starts were actually the *previous* world's castle walk (the
world counter increments before Mario enters), and the 1-1 start was boot-time RAM transient —
`pregame=1, player_state=0x08, x=40`, but 60 frames of Right+B leaves x at **0**.

What caught all three was writing the first tests that launch a real emulator, and one
assertion in particular: *hold Right+B and check that x actually increases*.

**Downstream.** Level starts now require the expert's x to *increase* within 10 frames. The
arm A/B gap it had inflated (70% vs 20%) collapsed to 45% vs 40% — see the next entry.

### ⚠️ 2026-08-01 — "45% vs 40%, indistinguishable" was underpowered, not null

**What.** A comparison read as a null result at n=20 was, at n=200, a **+30 pp** effect.

**How we got there.** 45% vs 40% is 9 episodes against 8. The Wilson interval on 9/20 spans
26–66%. Two arms whose true rates differ by 30 points are simply not separable at that sample
size, and the conclusion "indistinguishable" was reported before anyone attached an interval.

**Downstream.** House rule: **never report a binary outcome without an interval.** All
subsequent live evaluation is n=200.

---

## Stage 3 — Beyond imitation

### ❌ 2026-07-31 — Teacher 1: retrieval by game state cannot decide when to jump

**What.** Indexing every expert frame by `(world, level, x, y, player_state)` and retrieving
what the expert did there gives ≤11.9% A-onset recall.

**How we got there.** Validated against a held-out run *before* generating any training data.
59–87% of matched states carry *contradictory* expert actions — the same nominal state,
different decision. The state key does not determine the jump.

**Cost.** Under an hour, versus a week of training on bad labels.

### ❌ 2026-08-01 — Teachers 2 and 3: a search oracle scored on progress cannot decide either

**What.** Force A on, roll forward, compare progress, restore. A-onset agreement 46.8–54.2%
(chance) with a fixed run-right continuation; **16.2%** with a policy continuation and a
calibrated margin.

**How we got there.** Both suspects named in advance — the lookahead horizon and the progress
measure — were varied (60/120/180 frames; furthest-x and final-x) and neither moved agreement
out of the noise. Switching the continuation to a trained policy raised onset agreement to
63%, confirming the continuation *was* the binding constraint, but the oracle then jumped on
49–61% of frames against the expert's 6%.

The mechanism is visible in the margin sweep: overall agreement and onset agreement move in
**opposite directions** as the margin rises. Every increase buys overall agreement by refusing
to jump, and refusing to jump is exactly what fails at onsets. No margin is good at both.

**Why it fails, generally.** Over a 1–2 second horizon, jumping in SMB is nearly free — you
keep horizontal momentum while airborne. A progress-maximising comparison therefore says
"jump" whenever it says anything. The expert jumps rarely because most jumps are
*unnecessary*, which is a fact about its own future trajectory, not about local cost.

**Downstream.** Three teachers, three failures, one axis. The hand-constructed-teacher framing
is closed. Pre-committed kill conditions are what stopped each of them inside an hour.

### ✅ 2026-08-01 — Self-imitation: 59.5% → 99.0% on pipe 1

**What.** Roll the policy out from filtered expert start points, keep the top quarter by
progress, add to training, refit. Three rounds.

**How we got there.** No teacher needed, so nothing blocked it. The mixing ratio matters —
self-data comes from a policy worse than the expert, so it is never trained on alone; 1:1 and
3:1 were both run.

**Numbers.** pipe 1, n=200: baseline 59.5% [53, 66] → round 2 92.0% [87, 95] → **round 3
99.0% [96, 100]** at 1:1. A-onset recall *fell* 50.0% → 39.6%.

**The caveat that matters.** `x` median never moved off 594–595 in any round. It solved the
obstacle it could already sometimes pass and made no progress on the next one.

**A design flaw worth recording.** The intended health metric — acceptance rate — was
implemented as a fixed top-25% quantile, so it reported ~25% every round regardless of
behaviour and was structurally incapable of detecting the loop grading itself against a
declining standard. The valid statistic is the **score cutoff**, an absolute threshold, and it
rose every round: 289 → 362 → 416.

### ⚠️ 2026-08-01 — A model emitted a constant probability and four metrics computed cleanly on it

**What.** A checkpoint reported **0.0%** A-onset recall while clearing pipe 1 on 53% of
episodes.

**How we got there.** `FrameStackDataset` already returns float32 in [0, 1]. The training and
evaluation code divided by 255 **again**, so the network received a near-black image and
responded with a constant: p(A) = 0.00710 with a standard deviation of 1×10⁻⁵ across every
frame. Calibration, thresholding, onset recall and exact match all ran without complaint on
that constant.

The hypothesis on the table was calibration drift — plausible, specific, and wrong. What
caught it was the contradiction: **a checkpoint cannot score 0% recall and clear an obstacle
53% of the time.** Same weights, same frames, input scaled two ways:

| input | p(A) mean | p(A) std |
|---|---|---|
| as given (correct) | 0.1604 | 0.14671 |
| divided by 255 again | 0.0071 | **0.00001** |

**Downstream.** The project's central operating rule: **when an offline metric and a
behavioural measurement disagree, suspect the metric.** Round 1's checkpoint was discarded
(trained on corrupted input) while its self-generated data was kept (produced through the
correct path).

### ✅ 2026-08-01 — Older, slower speedruns are better training data than world records

**What.** Within one obsoletion chain, the three *oldest* runs train a policy that clears pipe
1 **67.3%** of the time; the three newest manage **21.7%**. Difference **+45.7 pp
[+40.5, +50.4]**.

**How we got there.** The original plan was glitchless-vs-glitchy, which the corpus cannot
support: there are no warpless-glitchless runs, "glitchless" is not verifiable by this
pipeline at all, and the single run the pilot rested on turns out to clear 7 of 8 levels.
Obsoletion position is a cleaner proxy for the same underlying question — position 0 is the
current record, higher positions are the older records it replaced — and within a chain every
run completes the same route with the same level count, so position is the only thing varying.

**Numbers.** Matched at 201,479 frames, 3 seeds, n=200 per arm. Earliest wins every seed:
72.5 / 53.5 / 76.0 against 7.5 / 39.5 / 18.0.

**Interpretation.** Frame-perfect glitch execution only works from states a learned policy
cannot reliably reach, so the more optimised the demonstration, the less of it is
reproducible. The newest arm also has *higher* expert fidelity (42.0% vs 32.5% recall) and one
third the performance.

**Downstream.** The project's headline result, and a practical rule for anyone assembling
demonstration data: **the best demonstrations are not the best teachers.**

### ✅ 2026-08-01 — A quarter of the data trains the best player

**What.** Fixed-epoch scaling peaks at **25%** of the corpus and declines.

**How we got there.** A first version held training *steps* constant, which confounds
everything — larger subsets get proportionally fewer passes over their data. Re-run with steps
scaled to subset size, the peak survives.

**Numbers.** n=200 for pipe 1: 10% → 63.0%, **25% → 83.5%**, 50% → 71.5%, 100% → 70.5%.
A-onset recall rises monotonically to 44.4% at 100%, so the most expert-faithful model is not
the best player.

**Downstream.** Answer to "was collecting 34 runs worth it": not for task performance. It was
worth it for the split, the redundancy measurement, and the chain experiment, which needed
whole chains.

### ❌ 2026-08-01 — The policy is stuck, not dying, so negative examples were dropped

**What.** 74.5–89.5% of all episodes end **standing still against terrain**. Enemy contact is
10–25%; no pit deaths, no timer expiries.

**How we got there.** "x median 594" cannot distinguish standing still from dying, and the two
need opposite fixes. Every episode is now classified into a fixed taxonomy
(`enemy_contact` / `pit` / `timer` / `stuck_terrain` / `game_over` / `budget_reached`).

**Downstream.** Killed a planned build. Keeping failed rollouts and training against the
frames before each death targets collisions, which are a minority failure. The taxonomy is now
permanent in the evaluation output.

### 🔁 ⚠️ 2026-08-03 — "The ceiling is the run button" — an automated verdict from a pooling artifact

**What.** A diagnostic concluded the policy could not clear pipe 2 because it jumps too slowly
and needs the B (run) button. It holds B on **94%** of frames, *more* than the expert's 77%.

**How we got there.** The check averaged takeoff velocity across a window spanning both the
approach and the pipe face. 56% of the samples were jumps attempted while already touching the
pipe at zero speed, so the mean described the stall, not the takeoff. Splitting the same
samples by location: approach (x 500–575) median **+1.60 px/f**, at the wall (x 585–600)
median **+0.00**.

**Downstream.** Standing check: **always ask for the split.** A pooled statistic across two
regimes manufactures confident verdicts.

### 🔁 ❌ 2026-08-03 — "A standing jump physically cannot clear pipe 2" — inferred, not measured, and false

**What.** From the location split above, we reasoned that SMB selects initial vertical velocity
from horizontal speed at takeoff, so a standstill jump gets the short-jump table entry and no
A-hold could compensate. **Wrong.** A jump from a genuine dead stop (velocity byte 0) clears
pipe 2 with an 11-frame hold.

**How we got there.** The claim was correct game trivia applied to the wrong question, and it
entered the shared project state as if it were a measurement — where it killed two approaches
(hold-duration modelling, hysteresis) and set two queue priorities. It was overturned by
running the experiment a strict reading of it made pointless: drive to the pipe, then sweep
A-hold 1–40 against trigger position and B-state, 323 configurations, with an assertion that
at least one configuration must differ.

**Downstream.** The tenth silent failure, and the one that established a new pattern: **an
inference dropped into a chain of measurements is indistinguishable from a measurement
downstream.** Provenance tagging was added to the shared state file.

### ✅ 2026-08-03 — Clearing pipe 2 needs two things at once, not one

**What.** Ground truth, from a re-run of the standstill arm with takeoff position, contact
state and flight input logged. From a dead stop flush against pipe 2, a jump clears it if A is
held ~10–11 frames **and Right is held throughout the airborne frames**. Without horizontal
input in flight it never clears, at any hold up to 40.

**How we got there.** The advisor asked one binary question — was the jump launched in contact
with the pipe, with no horizontal input? — and predicted at 65/35 that it had been launched
from open ground. The artifact did not record enough to answer, so only that arm was re-run
with four conditions separating takeoff *position*, takeoff *velocity*, and *flight input*.

Both prior positions were wrong. The jump was launched flush against the pipe at x=593 (not
open ground), and it cleared because Right was held in flight (not because the hold was long).
SMB permits mid-air horizontal acceleration, so the jump only translates Mario over the pipe
if Right is still pressed while he is above it.

**Numbers.**

| condition | takeoff x | speed byte | horiz. in flight | min clearing A-hold |
|---|---|---|---|---|
| reproduces original | 593 | 5 | yes | 10 |
| true dead stop | 594 | **0** | yes | **11** |
| walked into pipe | 594 | 1 | yes | 10 |
| **no horizontal input** | 594 | 1 | **no** | **never** |

**The baseline that should have been there from the start.** The do-nothing control reaches
**x=594** — exactly where every failing configuration in the original sweep stopped. Those
configurations were indistinguishable from pressing no buttons.

**Downstream.** Reopens the question of *why* the policy cannot produce this. The leading
hypothesis, not yet measured: with 8 conditionally-independent Bernoulli outputs and an expert
Right-press rate of 0.4533, holding Right for 10 consecutive frames is a ~3×10⁻⁴ event, and
holding A and Right together is far rarer. If so the blocker is the **output
parameterisation** — sustained multi-button input — rather than any loss weighting, which would
explain why four sustain-reweighting arms moved pipe 1 substantially and `x` median not at all.

---

## What generalises beyond this project

**Ten silent failures, one shape.** Every one ran to completion, produced plausible numbers,
and reported no error. None would have been caught by a test asking "did it crash". A pipeline
stage that silently degrades its input produces confident, well-formed, wrong numbers
downstream, and the code never complains.

What caught them, every time, was cross-checking one measurement against an *independent*
measurement of the same thing and then investigating the disagreement instead of the more
convenient number. Specifically:

| defence | caught |
|---|---|
| an offline metric disagreeing with live behaviour | the constant-probability model |
| watching the video | attract-mode contamination |
| a trivial baseline (do-nothing, blind, always-one-action) | vote-splitting, the 594 wall |
| an assertion that at least one setting differs | two degenerate sweep harnesses |
| splitting a pooled statistic by regime | the run-button verdict |
| asserting the start state is where it claims | three fake start points |
| validating a teacher before it labels anything | three failed teachers, each inside an hour |
| attaching an interval to a null | a +30 pp effect read as no effect |

**Two research findings we did not expect.** The best demonstrations are not the best teachers
— older, slower, less-optimised speedruns train a policy three times better than world records
at matched data volume. And a quarter of the data beats all of it. Both point the same way:
**curate, do not accumulate.**

**One methodological finding.** An inference written into a shared state file becomes
indistinguishable from a measurement within one hop. Provenance tags on claims are cheap
insurance; we added them after they would have saved two killed approaches and two misdirected
priorities.

### ✅ 2026-08-03 — Clearing pipe 2 needs Right on two thirds of frames; the policy manages 44%

**What.** A sharp threshold. Holding Right on a random 45% of the airborne frames — the rate
the policy actually uses — clears pipe 2 **0 times in 60**. At 65% it clears 30% of the time,
at 80% it clears 85%. The cutoff sits between 0.60 and 0.65; the policy's measured rate is
0.4424.

**How we got there.** The previous entry established that clearing the pipe needs a long A-hold
*and* horizontal input during flight, and the builder proposed a mechanism: with 8
conditionally-independent Bernoulli outputs, holding Right for 10 consecutive frames is a
~3×10⁻⁴ event, so the output parameterisation is the blocker.

The advisor rejected that arithmetic, correctly: it substitutes the corpus-wide *marginal*
Right rate for a *per-frame conditional* that is recomputed from a new observation every frame.
If the model has learned "airborne near a pipe → Right", the conditional is near 1 and the
holds are long. It is the identical error to the pooled-velocity verdict two entries earlier —
reasoning from an average across regimes instead of measuring within one.

The advisor also spotted that the flight-input experiment had tested Right as all-or-nothing,
which cannot separate "the architecture cannot sustain a button" from "the policy is not
competent at this", and predicted at 60/40 that partial Right would clear at 15–50% because
in-air momentum accumulates and gaps should be survivable.

So Right was applied on a *fraction* of flight frames, at a range of rates. The prediction was
wrong, and 20 seeds left the interval one point too wide to settle the pre-committed gate, so
it was re-run at 60 seeds and the full curve mapped.

**Numbers.** Dead stop in contact with pipe 2, A held 11 frames, B held throughout, only Right
varying:

| Right rate | n | cleared | 95% CI |
|---|---|---|---|
| 0.45 (policy's own) | 60 | 0 | [0.0%, 6.0%] |
| 0.50 / 0.55 / 0.60 | 20 each | 0 / 0 / 0 | upper bound 16.1% |
| 0.65 | 20 | 6 (30%) | [14.5, 51.9] |
| 0.80 | 20 | 17 (85%) | [64.0, 94.8] |

**Cost.** ~15 minutes of emulator time across two runs.

**Downstream.** The architectural framing survives the gate, but only in the weak form: the
*rate* the policy uses is insufficient. Whether the policy can produce a higher rate at the
moment it matters is a separate, unmeasured question — and that distinction is precisely the
one the advisor's objection was about.

**And a mechanism withdrawn.** Two structured conditions contradict both candidate
explanations. Right on 20 *consecutive* frames covering the first half of the flight **fails**
(max_x 597), while random Right at 70% — longest run averaging 15 — **succeeds** at 55%. So run
length is not the mechanism. But alternating Right every other frame, a 50% rate with longest
run 1, also fails, so rate alone is not sufficient either. Each candidate is contradicted by
one cell. The operative variable appears to be sustained rate across the whole window including
after landing, which is a third mechanism and is not yet measured.

The "10 consecutive frames" story from the previous entry is withdrawn. The conjunction
(long A-hold **and** Right in flight) stands; the account of why Right matters does not.

### 🔁 ⚠️ 2026-08-03 — The Right-rate threshold was a walking-speed threshold, not a pipe threshold

**What.** The previous entry's dose-response — Right on 45% of frames never clears pipe 2, 65%
clears 30%, 80% clears 85% — is **void as a statement about the pipe**. On flat ground with no
pipe and no jump, Right at a random 45% of frames moves Mario **1.2 pixels in 80 frames**. At
100% he moves 144. The low-rate conditions were not failing to clear an obstacle; they were
failing to walk.

**How we got there.** The advisor asked for the trivial baseline that had been missing: run the
identical dose-response on flat ground and score distance. This is the third time in the project
that a missing baseline produced a confident wrong conclusion, and the advisor named it as the
same class of gap.

SMB applies friction the moment Right is released, so intermittent Right yields almost no net
movement at any rate below ~0.8. The pipe-2 curve's apparent cliff between 0.60 and 0.65 sits
exactly where the locomotion curve's first jump sits. The obstacle was incidental.

**The cross-check that closed it.** The real policy travels 554 px in 255 frames — **2.17
px/frame**, against a maximum steady-state of 2.5. That is only achievable by holding Right
nearly continuously. So the policy's operative Right rate while moving is ~1.0, not the 0.4424
corpus-wide average that the builder had substituted into a per-frame calculation. The advisor
had rejected exactly that substitution two rounds earlier; this quantified how large the error
was.

**Numbers.** Flat ground, 1-1 start, 80 frames, B held, no A, n=20 per rate, 0 deaths in 160
trials. Mean distance: p=0.45 → 1.2 px · 0.50 → 1.8 · 0.55 → 2.6 · 0.60 → 3.2 · 0.65 → 7.5 ·
0.70 → 9.5 · 0.80 → 32.8 · 1.00 → 144.0.

**Cost.** ~10 minutes.

**Downstream.** The architectural claim — that 8 conditionally-independent Bernoulli outputs
cannot sustain a button, and that this is why pipe 2 is blocked — is **withdrawn**, not deferred.
Its evidence was confounded, and the cross-check points the opposite way: the policy sustains
Right fine. Three candidate mechanisms were overturned in three consecutive rounds, all of them
built by reasoning about a regime that had not been measured. A pre-committed stopping rule fired:
diagnosis stops, building starts.

**A reporting defect fixed in the same pass.** The structured conditions had reported
`min(total_right_frames, 40)` as though it were a count over the first 40 frames. Reissued
honestly, `alt` is a 50% rate with a longest run of **1**, and `first_half`/`second_half` are
**6.7%** full-window rates, not 50%. The previously flagged "disagreement" between them and
i.i.d. 0.70 was two experiments of different duration in one table — not physics. With the
accounting fixed, every cell is explained by net forward travel alone, and the earlier claim
that "rate alone is not sufficient" is withdrawn.

**Also done this round:** `git init` (fourth request), 121 files, 1.2 MB, with the 11 GB corpus,
all checkpoints and the copyrighted ROM excluded.

### ✅ 2026-08-03 — The frontier, measured as a rate: pipe 2 is passed 21.5% of the time, and the real wall is x≈720

**What.** The pipe-2 clearance *rate* had never been reported in this project — every claim about
it was inferred from an `x` median of 594–595. Measured at n=200: the best model clears pipe 2 on
**21.5%** of episodes [16.4, 27.7]. Not the 0% four rounds of diagnosis assumed. Three further
facts arrived with it:

- **A hard wall at x≈720.** Across 800 episodes and four checkpoints, maximum x ever reached is
  **724**. Nothing crosses it. That is the real frontier.
- **Every death is a Goomba at x≈310.** 175 deaths across 800 episodes, 100% `enemy:goomba`,
  median death x 309–312 — *before* pipe 1, not after pipe 2. 23% of all episodes.
- **"99.0% clear pipe 1" is 81.5% on a single life.** The original harness lets Mario die and
  respawn inside one episode and scores the best of several attempts.

**How we got there.** Four consecutive mechanisms for "why the policy cannot pass pipe 2" had been
proposed and withdrawn (takeoff velocity, A-hold duration, consecutive Right frames, sustained
Right rate). When four mechanisms fail in a row the usual cause is not a subtle fifth one but a
false premise, and two independent signals said so: every high-water mark of 706–713 in the
existing artifacts turned out to be a *death* position (38/38 in one artifact), and the project
owner, watching live play, reported Mario passing pipe 2 often and dying to a Goomba.

A pre-committed stopping rule fired: mechanism-hunting ended, and the one measurement that had
been deferred four times — a rate with an interval — was finally taken.

**Numbers.** n=200 per checkpoint, single life per episode, per-button sampling.

| checkpoint | pipe 1 (x>470) | pipe 2 (x>630) | x>760 | x max | died | stuck |
|---|---|---|---|---|---|---|
| round3_ratio1to1 | 81.5% | **21.5% [16.4, 27.7]** | 0.0% | 724 | 46 | 154 |
| round2_ratio1to1 | 76.5% | 16.0% | 0.0% | 723 | 54 | 146 |
| round2_ratio3to1 | 79.0% | 8.5% | 0.0% | 723 | 36 | 164 |
| sustain_arm_a | 81.0% | 8.5% | 0.0% | 724 | 39 | 161 |

Obstacle positions, read off the x histogram rather than assumed: enemy ~296–312, pipe 1 ~432,
pipe 2 ~592, **wall ~720**.

**Cost.** ~20 minutes of emulator time, plus a free artifact read.

**Two checks that came back clean.** The failure taxonomy was suspected of mislabelling deaths as
stalls; an independent code path returns 23.0% died / 77.0% stuck against the taxonomy's 23.0% /
77.0% — exact agreement, so those figures stand. And self-imitation did **not** regress at pipe 2
while gaining at pipe 1: along the 1:1 lineage the pipe-2 rate rose 8.5% → 16.0% → 21.5%.

**The last of the architectural story, killed on direct measurement.** Realized Right rate over
*airborne* frames is **0.83–0.93** across all four checkpoints, against the ~0.62 the void
dose-response implied was needed. The claim that independent per-frame sampling cannot sustain a
button is now contradicted by measurement, not merely unsupported.

**Downstream.** Search gets scoped at x≈720 from a savestate, not at pipe 2 and not from the
level start. The Goomba at x≈310 becomes a separate and cheaper target — 23% of episodes lost to
one enemy in the first fifth of the level. And the "negative examples" kill, which rested on
enemy deaths being a minority at the *pipe-2* frontier, no longer clearly transfers.

**One discrepancy left open deliberately:** 99.0% and 81.5% are the same checkpoint on the same
level under two harnesses — multi-life versus single-life. Both are defensible measurements; they
are not the same measurement, and every pipe-1 figure in the project's history is the multi-life
one. Which is canonical is not the builder's call to make unilaterally, and it affects whether a
successor model can be said to "beat 99.0%".

### ⚠️✅ 2026-08-03 — Every clearance figure was best-of-several-lives; the headline finding survives being fixed

**What.** The evaluation harness counted deaths and let Mario respawn *inside* one episode,
scoring the best attempt across several lives. Re-measured with one life per episode, the
project's headline claim — older speedruns train better policies than world-record ones — shrinks
from **+45.7 pp to +34.2 pp [+29.0, +39.1]** and still excludes zero. The absolute numbers do
not survive: the best model's pipe-1 rate is **81.5%**, not the 99.0% published everywhere.

**How we got there.** A discrepancy surfaced while cross-checking: an independent measurement of
pipe-1 clearance gave 81.5% where the project's own figure was 99.0%, same checkpoint, same level.
The cause was a harness difference, not a bug in either number — one allows retries within an
episode, the other does not.

The advisor ruled single life canonical, on the grounds that the multi-life metric **rewards
dying**: a policy that dies respawns and gets another attempt at the same obstacle, while a policy
that gets *stuck* burns its remaining frames and gets no retry. Two policies of equal skill score
differently according to how they happen to fail.

That made re-measuring the headline urgent, because the inflation scales with how often an arm
dies and the two arms visibly died at different rates. It survived because the death counts turned
out near-identical (128 vs 137 across 600 episodes), so the inflation was common-mode.

**Numbers.** Single life, 3 seeds × n=200 per arm, matched 201,479 frames, same six publications:

| metric | earliest | latest | difference |
|---|---|---|---|
| pipe 1 | 314/600 = 52.3% | 109/600 = 18.2% | **+34.2 pp [+29.0, +39.1]** |
| pipe 2 | 17/600 = 2.8% | 0/600 = 0.0% | +2.8 pp [+1.6, +4.5] |

**Cost.** ~35 minutes of emulator time. No retraining — the checkpoints already existed.

**Downstream.** Every clearance number in the project's history is a multi-life figure and needs
reissuing: 99.0%, 95.5%, the scaling table, the arm A/B comparison. Only the chain result has been
redone so far.

**Two things found in the same run.**

*The trivial baseline reframes the Goomba.* A scripted agent holding Right+B permanently **dies to
the Goomba at x=312, every time**, and never reaches pipe 1. So the policy — which survives that
Goomba 77% of the time and reaches x=595 — is doing substantial work, and the 23% of episodes lost
there is the hard tail of a hazard it already mostly solves, not low-hanging fruit. This is the
opposite of what the previous entry recommended.

*The wall at x≈720 is a wall, not a killer.* Of the 43 episodes clearing pipe 2, **34 get stuck at
720 and 9 die** — whereas a scripted agent pushed past 720 died 38/38. Two different failures at
one position.

**A correction.** The previous entry reported "every death is a Goomba at x≈310". The advisor
showed by arithmetic that at least 9 deaths must lie past x=470 (163 clearances + 46 deaths > 200),
and the histogram confirms exactly 9, clustered at ~704. A median had been reported as a universal
— the same error that produced the pipe-2 premise and cost four rounds of diagnosis.

### ✅ 2026-08-03 — Composing four wins clears pipe 3, and breaks a ceiling that was real for every model before it

**What.** Combining four separately-measured wins — sustain+onset reweighting, earliest-in-chain
data, the ~25% subset, and self-imitation — produced the first model to get past pipe 3. Pipe 2
clearance goes from **21.5% to 54.0%** (+32.5 pp [+23.2, +41.0]), pipe 3 from a genuine **0/200 to
24/100**, and maximum x from 724 to **2227** — most of the way through 1-1.

**How we got there.** The project owner, watching live play, said Mario cleared pipe 3 sometimes.
The measurement said 0/200 with a hard ceiling at exactly x=724 across 800 episodes and four
checkpoints — and an identical number across every configuration is this project's most frequent
signature of a metric that cannot move.

Six explanations were checked against the checkpoint that had been measured, and **all six came back
clean**: arrivals at pipe 3 happen around frame 600 of a 2,500 budget (not at the end), zero episodes
time out, raising the budget to 10,000 frames changes nothing, position cross-checks exactly against
raw RAM, and no episode changes area — so it is not a pipe entry. For that checkpoint the ceiling is
completely real.

The wrong assumption was not the harness. It was that four checkpoints out of eighty-eight
generalised to the project. While those checks ran, the composed model finished training and cleared
pipe 3 immediately.

**Numbers.** Single life, n=100–200, same harness throughout:

| model | pipe 1 | pipe 2 | pipe 3 | x max | deaths |
|---|---|---|---|---|---|
| previous best | 81.5% | 21.5% | **0/200** | 724 | 46/200 |
| compose_round1 | 78.0% | 43.0% | 5/100 | 1250 | 28/100 |
| **compose_round2** | 61.0% | **50.0%** | **24/100** | **1957** | 78/100 |
| compose_round3 | 50.0% | 48.0% | 24/100 | **2227** | 86/100 |

**Cost.** ~50 minutes: base training plus three self-imitation rounds, all CPU, with evaluation
interleaved on the single permitted emulator.

**The tradeoff, stated rather than buried.** These models go further and die far more. Round 2 dies
in 78 of 100 episodes against the baseline's 46 of 200, and pipe-1 clearance *falls* from 81.5% to
61%. "Cleared pipe 2" says the composed model is much better; "survived" says it is worse. That is
the same metric hazard as an always-jump policy clearing an obstacle and then being unable to jump
again — an objective that rewards distance alone will pick the reckless model.

**Downstream.** The four wins do compose, which had been queued unbuilt for six directives. The
frontier has moved from x=724 to somewhere past x=2000 and is now unmeasured. Deaths have become the
dominant failure mode (78–86% of episodes, against 23% before), which revives the negative-examples
approach that had been killed on the grounds that enemy deaths were a minority.

**Two corrections recorded.** "Nothing in this project has ever cleared pipe 3" was generalised from
four checkpoints — the same median-as-universal error made two reports earlier with death positions.
And the 19% "past x=720" figure counted arrivals at pipe 3's face rather than clearances, because the
threshold sat on the obstacle instead of past it.

### ✅ 2026-08-03 — The best model's regression is one Goomba, and the arithmetic is exact

**What.** Composition improved pipe 2 from 21.5% to 54.0% but *lost* ground at pipe 1, 81.5% → 64.0%.
That regression is entirely extra deaths to a single Goomba at x≈288. Round 2 loses **35** pipe-1
clearances and gains **exactly 35** extra Goomba deaths; round 3 loses **63** and gains **exactly 63**.

**How we got there.** The conditional rates pointed at it before any new measurement: of episodes
clearing pipe 1, the share also clearing pipe 2 is 26.4% for the baseline and **84.4%** for the composed
model. So the composed model is dramatically better once past pipe 1 and worse at getting there — which
localises the problem to something before x=470. The death histograms were already on disk, recorded by
a summary function written for an earlier question, so answering it cost no emulator time.

**Numbers.** Single life, n=200 each.

| model | pipe 1 | Goomba-zone deaths | deaths past x=470 | ended died/stuck |
|---|---|---|---|---|
| baseline | 81.5% | 37 | 9 | 46 / 154 |
| compose_round2 | 64.0% | **72** | **81** | 153 / 47 |
| compose_round3 | 50.0% | **100** | 73 | 173 / 27 |

**Why it matters.** That Goomba is cleared by **75 of 80** scripted jump timings — it is the most
tractable obstacle in the game. Restoring baseline survival while keeping the composed model's
downstream ability projects to 0.815 × 0.844 = **~69% pipe 2**, far beyond anything measured.

**But the answer was not either/or.** Deaths past pipe 1 rose from 9 to 81 — a ninefold increase spread
across eight locations from x=640 to 1952, including Koopas. The composed models trade survival for
reach *globally* as well as regressing at the Goomba, and the dominant failure mode inverted: 153 died /
47 stuck, against the baseline's 46 died / 154 stuck.

**Cost.** Minutes, entirely from artifacts already written.

**Downstream.** The Goomba becomes the top target on value grounds rather than as a machinery test. And
"death relabelling", killed earlier because enemy deaths were a minority, is now live — deaths are 76–86%
of episodes for these models.

**A methodological note worth keeping.** Two figures in the previous report mixed n=100 and n=200
measurements of the same checkpoints as though continuous. `x max` also grows with sample size — the same
checkpoint reads 1250 at n=100 and 1957 at n=200 — so it is an anecdote, not a statistic. The rate of
reaching a given x is the statistic.

### ⚠️✅ 2026-08-03 — A survival gate on the self-imitation filter improved reach, not survival

**What.** The acceptance filter for self-imitation kept the top 25% of rollouts by
progress-from-start, which cannot distinguish "got far and survived" from "got far and died" — so it
plausibly selected for reckless play, and deaths did rise monotonically across rounds (30 → 56 → 153
→ 173). Adding a survival requirement raised pipe-2 clearance at **every** round (best: 54% → **60%**)
and **did not stop the death escalation** (30 → 85 → 144 → 189, higher at two of three rounds).

**How we got there.** The mechanism was spotted in the training loop after the same objective flaw had
already been recorded twice elsewhere — an always-jump policy that clears an obstacle and can never
jump again, and an evaluation harness that rewarded dying by granting retries. One condition was
changed and nothing else, so the comparison is clean.

**Numbers.** Single life, n=200 per round.

| round | old filter (pipe1 / pipe2 / deaths) | survival-gated |
|---|---|---|
| 1 | 79.5 / 43.5 / 56 | 75.5 / **51.5** / 85 |
| 2 | 64.0 / 54.0 / 153 | 66.0 / **60.0** / 144 |
| 3 | 50.0 / 48.0 / 173 | 58.0 / 55.5 / 189 |

`surv_round2` is the best model the project has produced: pipe 1 66.0%, pipe 2 **60.0%**
(+38.5 pp [+29.2, +46.8] over baseline), past-720 46.0%.

**Why the gate barely mattered.** It excluded almost nothing. Of 150 rollouts per round, 91–99
survived and the accepted set shrank only from ~38 to 29–30 — because inside a 500-frame window the
rollouts that travel furthest are overwhelmingly the ones that survive. **Progress and survival are
correlated at that horizon, so gating on survival is close to a no-op on selection.** The death
escalation is therefore intrinsic to self-imitation on this data, not a selection bug.

**A verdict declined.** The script printed "SELECTION BUG CONFIRMED" because the pre-committed
condition was literally satisfied — 144 deaths against 153, pipe 2 above 43.5%. That compares the best
round of one run against the best round of another and hides an unchanged trend. Reported as declined
rather than as a confirmation.

**Cost.** ~40 minutes, reusing the existing base checkpoint.

**Downstream.** The Goomba fix becomes the whole plan, and `surv_round2` is the model to freeze.

---

## Distilling 22 verified demonstrations at pipe 4 — and discovering the baseline held A on 85% of frames

**What changed.** Search-and-distil was closed for the first time: the 39 clearing (trigger, hold)
configurations found at pipe 4 were re-run against the search's own bar, the survivors were recorded
frame by frame as demonstrations, and a checkpoint was fine-tuned on them. The distilled policy was
then measured at n=200, single life, against the identical baseline seeds. **It regressed badly, and
the regression is what taught us something.**

**How we got there, including the wrong turns.**

The plan was to distil 39 demonstrations. Three things went differently:

1. **Only 22 of the 39 reproduced.** All 17 failures were seed 8 — one prefix out of three diverged
   between scripts, while seeds 12 and 16 reproduced exactly, 22 of 22. Of the 22 that were attempted,
   **zero failed to clear**; the 17 were refused because Mario was airborne at the trigger frame. The
   headline requirement survived only because hold 12 at trigger 892 also reproduced on seed 12, a
   different prefix. **A sweep result is not a result until it has been re-run from a fresh process.**
2. **The eval said "stuck at pipe 4 fell 29 → 0", which looks like total success and is the opposite.**
   Zero episodes were stuck at pipe 4 because **zero reached it.** The denominator moved. The script's
   own pre-committed verdict string caught that the A-hold had not risen, but mislabelled the outcome
   as "clearance moved without the hold" — clearance had gone to zero.
3. **The A-hold at pipe 4 was reported as `median 4.0 → None` and scored as "did not rise".** `None`
   was missing data, not a decrease. This was fixed in code rather than in prose: the audit script now
   separates `measurable: false` from a measured fall, and refuses to score a window no episode
   reached. **The fix that mattered was measuring the hold in windows both arms actually reach.**

**Numbers.** n=200 per arm, single life, seeds 0–199, identical episode function. Baseline is
`C_control_matched_r2.pt`, the checkpoint every recent figure in this project rests on.

| metric | baseline | distilled |
|---|---|---|
| arrived at x=880 | 67 | **0** |
| stuck at pipe 4 (max_x 896–928) | 29 | 0 *(no arrivals)* |
| cleared past x=975 | 38 | 0 |
| x_median | 723 | **437** |
| x_p90 / x_max | 1306 / 1939 | 596 / 696 |
| deaths at x≈256 | 54 of 142 | **65 of 66** |

A-hold length, holds counted where they begin:

| window | baseline median (frac ≥12) | distilled median (frac ≥12) |
|---|---|---|
| pipe 1, x 300–470 | 4.0 (17.1%) | 1.0 (5.7%) |
| pipe 2, x 560–640 | 4.0 (16.4%) | 1.0 (**0.0%**) |
| pipe 4, x 880–924 | 4.0 (13.9%) | unmeasurable — 0 arrivals |
| anywhere | 4.0 (18.6%), max 160 | 1.0 (2.7%), max 48 |

**The finding.** Button marginals against the expert's own press rates:

| button | expert | baseline | distilled |
|---|---|---|---|
| **A** | **0.152** | **0.852 (5.6×)** | 0.370 (2.4×) |
| Down | 0.007 | 0.281 | 0.014 |
| Left | 0.030 | 0.108 | 0.008 |

**The baseline presses A on 85.2% of every frame and spends 85.2% of its frames inside an A-hold.** It
was not clearing pipes by jumping well; it was airborne almost permanently. Fine-tuning on
demonstrations whose A-rate is 44% — near the expert's — pulled the marginal down and shortened holds
everywhere, and the policy lost the level. The demonstrations *were* absorbed. **What broke is that the
old score depended on the degeneracy they removed.**

This is the always-jump failure mode already recorded twice in this log as understood and fixed. It was
still in the checkpoint, visible in retained per-frame traces for six reports, and nobody — including
me — had looked at the A-press rate.

**Cost.** 5.0 minutes of compute for the distillation and eval; the audit is a read over files already
on disk. The expensive part was six reports of building on an unaudited baseline.

**Downstream effect.** The frontier map is demoted from a description of *the policy* to a description
of *an always-jump policy*: 29 stuck at pipe 4, 38 clearing it, 67 arrivals, x_median 723, the 17 deaths
at x≈1,216–1,248, the 11 gap falls, the pipe-3 dwell median of 161 frames. Every number is real and
every one describes a policy airborne 85% of the time. **New required field: report `a_press_rate`
beside every clearance figure.** A marginal that far from the expert's is a defect regardless of what
the clearance rate says.

---

## The trivial baseline that should have come first: a three-button script matches the policy through pipe 2

**What changed.** A control that had never been run: Right+B held on every frame, plus the A button
flipped as an i.i.d. coin at a fixed probability. No network, no observations, no learning. It matches
the learned policy through pipe 2 and loses to it past pipe 3, and that split reorganises what every
performance figure in this project means.

**How we got there, including the wrong turn.** The previous entry found that the checkpoint carrying
every recent headline presses A on 85.2% of frames against the expert's 15.2%. The obvious next question
— *is a fixed button rate enough on its own?* — had never been asked in ~35 blocks of work, because each
new result was compared against the previous model rather than against nothing at all. Two framings were
wrong along the way: the binary question "is the learned component worth anything" turned out to have
different answers at different obstacles, and the earlier claim that the demonstrations "were absorbed,
so this is not a limit of imitation" was too generous — P3 below shows absorbing them is incompatible
with keeping the level.

**Numbers.** Single life throughout. Thresholds: pipe1 x>470, pipe2 x>630, **pipe3 x>735**, pipe4 x>975.
pipe3=735 is derived, not chosen: the max_x histogram over 200 baseline episodes has a 37-episode spike
in the 720–735 bin and **nothing at all in 736–783**.

The scripted curve, n=20 per arm:

| p(A) | x median | pipe1 | pipe2 | pipe3 | pipe4 |
|---|---|---|---|---|---|
| 0.00 | 312 | 0.0 | 0.0 | 0.0 | 0.0 |
| 0.15 *(the expert's own rate)* | 436 | 45.0 | 0.0 | 0.0 | 0.0 |
| 0.50 | 595 | 85.0 | 10.0 | 0.0 | 0.0 |
| **0.85** | **722** | 70.0 | **70.0** | 10.0 | 0.0 |
| 1.00 | 316 | 0.0 | 0.0 | 0.0 | 0.0 |

A sharp interior optimum at ~0.85 — and the learned policy's A-rate is 0.852. Both extremes die at
x≈312. At n=200, paired against the policy on identical seeds:

| | script p=0.85 | policy | policy − script, pp |
|---|---|---|---|
| pipe1 | 145 (72.5%) | 146 (73.0%) | +0.5 [−8.2, +9.2] |
| **pipe2** | **137 (68.5%)** | **137 (68.5%)** | **+0.0 [−9.0, +9.0]** |
| pipe3 | 26 (13.0%) | 78 (39.0%) | **+26.0 [+17.6, +34.0]** |
| pipe4 | 8 (4.0%) | 38 (19.0%) | **+15.0 [+8.9, +21.3]** |

Identical counts at pipe 2: 137 and 137.

**Then: did composition's headline gain come from the A-rate?** Five checkpoints re-measured at n=200,
ordered by A-rate. All three archived pipe-2 figures reproduced to the decimal (21.5%, 62.0%, 60.0%), so
this is re-measurement rather than trust in the archive.

| arm | A | ×expert | pipe1 | pipe2 | pipe3 | pipe4 | x_med |
|---|---|---|---|---|---|---|---|
| `round3_ratio1to1` | 0.628 | 4.13 | 81.5 | **21.5** | 0.0 | 0.0 | 595 |
| `top20_round2` | 0.822 | 5.41 | 83.0 | **62.0** | 28.5 | 17.5 | 722 |
| *script p=0.85* | *0.850* | *5.59* | *72.5* | ***68.5*** | *13.0* | *4.0* | *722* |
| `surv_round2` | 0.865 | 5.69 | 66.0 | 60.0 | 23.0 | 10.5 | 698 |
| `compose_round2` | 0.888 | 5.84 | 64.0 | 54.0 | 31.5 | 19.5 | 688 |
| `surv_round3` | 0.926 | 6.09 | 58.0 | 55.5 | 32.0 | 24.0 | 682 |

**Yes.** The A-rate rose 0.628 → 0.822 across the sequence that took pipe 2 from 21.5% to 62.0%
[+31.2, +48.7] pp. And **no checkpoint beats the script at pipe 2; four of five are worse.** At pipe 3
and 4 every checkpoint but the first beats it, all intervals excluding zero.

A pattern nobody asked for: among learned models, as A rises 0.822 → 0.926, **pipe 1/2 fall (83.0 → 58.0,
62.0 → 55.5) while pipe 3/4 rise (28.5 → 32.0, 17.5 → 24.0)**. The A-rate trades obstacles against each
other, so no single fixed rate is right for the level — which is what a state-conditional policy is for.

**And: was the distillation collapse forgetting, or removal of a load-bearing degeneracy?** Four
schedules, one seed each, all from the same baseline (A 0.852, x_med 723):

| arm | epochs | A | x_med | pipe2 | pipe4 | reach kept? |
|---|---|---|---|---|---|---|
| `steps100_1to1` | 1.7 | 0.468 | 594 | **1.5** | 0.0 | no |
| `steps800_1to4` | 5.3 | 0.299 | 436 | 0.5 | 0.0 | no |
| `steps100_1to4` | **0.7** | 0.481 | 594 | 1.5 | 0.0 | no |
| 13-epoch 1:1 (prior entry) | 13 | 0.370 | 437 | — | 0.0 | no |

**Zero of four.** Not forgetting — **degeneracy-removal.** Even at 0.7 epochs the A-rate falls and pipe 2
collapses. Diluting with *more* expert data pushed A *lower* (0.299), which follows, since the expert's
own rate is 0.152. Tight cross-check: the distilled arm at A=0.468 clears pipe 2 at 1.5%, and the
*script* at p=0.50 clears it at 10.0% — pipe-2 performance tracks the A-rate whether that rate comes
from a network or a coin.

**Cost.** 0.6 + 18.1 + 10.0 minutes. The expensive part was ~35 blocks of comparing each model to the
previous one instead of to a coin flip.

**Downstream effect.** Every clearance figure this project has reported at or below pipe 2 — the 21.5%
baseline, composition's 54–62%, the survival gate's 60%, the frontier checkpoint's 68.5% — is a statement
about button marginals, and none of them beats a three-button script. **The learned component's only
demonstrated value is at pipes 3 and 4**: +26.0 and +15.0 pp over the best script, reproduced across four
independent checkpoints. If the project restarts from a different objective, that is the result worth
carrying forward, and the always-jump marginal is the thing to design against.

---

## The pipe-3 advantage survives every fixed-rate control, and the objective gets rebuilt around it

**What changed.** Three things. The control ladder was climbed to its last rung — scripts with Left, with
Down, and with *every* button marginal matched to a real checkpoint — and none of them closed the pipe-3
gap. The A-rate was plotted against x, showing the policy is not a fixed-rate agent. And the project's
scoring function was replaced: every clearance is now measured **net of the best fixed-rate script at that
obstacle**, in both the reporting path and the self-imitation training signal.

**How we got there, including the wrong turns.**

The previous entry found a three-button script matching the policy at pipe 2 (137/200 vs 137/200) and
losing badly at pipes 3–4. Two readings were possible: the policy has real state-conditional skill past
pipe 3, or it just presses Left and Down sometimes — which the script never did. So the ladder was
extended.

Three corrections came out of it:

1. **"Pipe 2 is a tie" was wrong.** Adding Left at the policy's own rate (0.135) took the script from
   68.5% to **82.5%** at pipe 2. The tie was an artefact of testing a weaker script; the script actually
   **wins** pipe 2 by 14.0 pp. The retraction was under-reported, not over-reported.
2. **The episode-overlap test could not work as designed.** The plan was to check whether the *same*
   seeds clear pipe 2 in both arms — coinciding sets would mean the policy is *behaving as* the script.
   But the policy draws `rng.random(8)` per frame (one uniform per button) while the script drew
   `rng.random()` once, so the streams diverge at frame 1 and **identical behaviour would still produce
   independent episode sets.** The test was structurally incapable of detecting what it was for. Repaired
   with an arm that consumes eight uniforms per frame and reads A from slot 7.
3. **`min_progress=120` was 120 pixels.** The replacement credit runs 0–4, so the old floor would have
   silently rejected every rollout and self-imitation would have accepted nothing. Caught while wiring.

**Numbers.** n=200, single life, seeds 0–199 throughout. Rates matched to `top20_round2`.

| arm | A | Left | Down | pipe1 | pipe2 | pipe3 | pipe4 | pipe-3 gap vs policy |
|---|---|---|---|---|---|---|---|---|
| script (plain) | 0.850 | 0 | 0 | 72.5 | 68.5 | 13.0 | 4.0 | +26.0 [+17.6, +34.0] |
| **left** | 0.850 | 0.135 | 0 | **87.0** | **82.5** | 22.0 | 6.0 | **+17.0 [+8.0, +25.6]** |
| down | 0.850 | 0 | 0.088 | 81.0 | 76.5 | 20.5 | 8.0 | **+18.5 [+9.6, +27.0]** |
| match_top20 | 0.848 | 0.136 | 0.086 | 82.0 | 73.0 | 21.5 | 6.5 | **+17.5 [+8.5, +26.1]** |
| rng_matched | 0.851 | 0 | 0 | 78.0 | 75.5 | 23.5 | 6.5 | **+15.5 [+6.4, +24.2]** |
| *policy* | *0.852* | *0.108* | *0.281* | *73.0* | *68.5* | ***39.0*** | ***19.0*** | — |

**No arm closes it.** Every interval excludes zero. Left and Down lift pipes 1–2 substantially and pipe 3
barely (13.0% → 20.5–23.5% against 39.0%).

**Why: the policy conditions on position.** A-rate by x, 100 px bins:

| | x 400 | **x 500** | x 600 | x 800 | **x 900** | spread across 11 bins |
|---|---|---|---|---|---|---|
| `top20_round2` | 0.816 | **0.718** | 0.750 | 0.878 | **0.737** | **0.204** |
| `surv_round3` | 0.904 | 0.903 | 0.908 | 0.936 | 0.882 | 0.082 |
| script p=0.85 | 0.853 | 0.856 | 0.852 | 0.842 | 0.838 | **0.025** *(noise floor)* |

The script's 0.025 spread over bins of 7k–27k frames is the sampling floor. `top20_round2` swings **8×**
that, and its two lowest bins are **x 500–599, immediately before pipe 2 (592–630)**, and **x 900–999,
at pipe 4**. It backs off jumping right before the tall pipes. That is the mechanism, and it needs no
appeal to Left.

**The new objective, and what it says about every model.** Best fixed-rate script per obstacle, measured
from artifacts and taking the per-obstacle maximum: pipe1 **87.0%**, pipe2 **82.5%**, pipe3 **23.5%**,
pipe4 **8.0%**. Scored as clearance minus that:

| checkpoint | pipe1 | pipe2 | pipe3 | pipe4 | beats script at |
|---|---|---|---|---|---|
| `C_control_matched_r2` | −14.0 | −14.0 | **+15.5** | **+11.0** | pipe3, pipe4 |
| `round3_ratio1to1` | −5.5 | −61.0 | −23.5 | −8.0 | **nothing** |
| `compose_round2` | −23.0 | −28.5 | +8.0 | **+11.5** | pipe4 |
| `top20_round2` | −4.0 | −20.5 | +5.0 | **+9.5** | pipe4 |
| `surv_round2` | −21.0 | −22.5 | −0.5 | +2.5 | **nothing** |
| `surv_round3` | −29.0 | −27.0 | +8.5 | **+16.0** | pipe4 |

**Not one checkpoint beats the script at pipe 2.** `surv_round2`, recorded earlier in this log as the
project's best model, beats nothing at all. **Supervised imitation on this corpus did not produce skill at
pipes 1–2.** What it did produce is robust at pipe 4 — four of six checkpoints, +9.5 to +16.0 pp — and at
pipe 3 for one.

**Cost.** 6.3 minutes of emulator time plus reads over traces already on disk. No training was run.

**Downstream effect.** The training signal changed, not just the report. `rollout_round`'s acceptance
score was `gained + 4000·levels − 2000·deaths`, and `gained` is precisely what raising the A-rate
maximises — so self-imitation had been optimising the marginal all along, which is one mechanism behind
the death escalation, the reckless models and the composition "gain." It is now
`Σ(1 − p_script(obstacle))` over obstacles cleared: past pipe 2 earns 0.305, past pipe 4 earns 1.990, so
the obstacle the script cannot do is worth **6.5×** the one it can. A known cost of the change: the
obstacle table is 1-1-only, so 484 of 500 trajectory start points now return no credit and are dropped
rather than scored on progress — **self-imitation trains on 1-1 until a per-start script *reach* table
exists.** The thesis is restated around pipes 3–4.

---

## The always-jump degeneracy had a root cause all along: the loss function

**What changed.** A start-state library built from the policy's own traces, a per-start reach table that
scores rollouts against a fixed-rate script from the identical state, in-memory savestates so restoring a
policy-visited state is O(1), and a training run under the new objective. The run failed — and chasing why
found the cause of the degeneracy that has distorted every number in this project.

**How we got there, including the wrong turns.**

The new objective had a blocker: all 16 of 1-1's start points sit at x = 2,616–2,636, past every obstacle
the credit pays for, so self-imitation had nowhere useful to practise. Fixed by mining the policy's own
retained traces — every grounded frame is a restorable start state, since replaying a recorded byte prefix
reproduces the state. That measurement came out **132,844 of 132,844 frames exact (100.000%)**, which also
bounds an older mystery: the seed-8 divergence was in prefix *generation*, not replay. It also **closed the
early-1-1 absence open since 2026-08-03** — 403 grounded candidates in x 0–120, where the expert-based
filter found none, because the expert is airborne there and the policy is not.

Then the training run produced a policy pressing **A on 0.970 of frames, Down 0.756, Left 0.398**, with x
median collapsing 723 → **311** (the first Goomba). The obvious suspects were the acceptance filter and
the data. Both were wrong: **the self-data it trained on had A on 0.871, Down 0.314, Left 0.122 — every
marginal came out above its own training data.** Supervised learning on i.i.d. targets cannot do that.

So the objective was probed directly, with no emulator: two policies from the same seed on the same expert
data, differing only in loss.

| button | expert data | plain BCE | `sustain_loss` | closed-form optimum |
|---|---|---|---|---|
| **A** | 0.147 | **0.130** | **0.403** | 0.463 |
| B | 0.509 | 0.511 | 0.745 | 0.838 |
| Right | 0.453 | 0.444 | 0.725 | 0.805 |
| Left | 0.028 | 0.023 | 0.112 | 0.126 |

`sustain_loss` — used by **every "composed recipe" run in this project** — up-weights onset frames 10× and
sustained presses 5×, and never up-weights released frames. For a Bernoulli head with weight `a` on
positives the weighted optimum is `a·p / (a·p + (1−p))`; 5× turns p=0.5 into 0.833, and the measured values
track that closed form. **Plain BCE recovers the base rate on every button. The recipe's loss inflates
every button by construction.**

**One pass inflates A from 0.147 to 0.403. Iterated self-imitation rounds compound it to 0.85–0.97.** That
is the origin of the always-jump degeneracy: not emergent selection pressure, not the data — the
objective. It explains composition's A-rate climb from 0.628 to 0.888, the frontier checkpoint's 85.2%,
and why distilling toward the expert's rate destroyed reach in 4 of 4 schedules — the reach depended on
the inflated marginal.

**Numbers for the training run itself.** 237 of 576 rollouts accepted (41.1%), median per-state reach
quantile 0.539 (0.5 = matching the script), 400 steps, 0.5 epochs.

| | base | trained |
|---|---|---|
| x median | 723 | **311** |
| pipe1 / pipe2 / pipe3 / pipe4 | 73.0 / 68.5 / 39.0 / 19.0 | 47.5 / 47.5 / 18.5 / 12.0 |
| **`vs_script` pipe3** | **+15.5** | **−5.0** |
| **`vs_script` pipe4** | **+11.0** | **+4.0** |
| A / Down / Left | 0.852 / 0.281 / 0.108 | **0.970 / 0.756 / 0.398** |

**A kill condition declined, on the record.** The pre-committed condition said that if `vs_script` failed
to improve at pipes 3–4 under a degeneracy-proof objective, the corpus and method were exhausted. It
failed — but the run had a degeneracy-proof *credit* sitting on top of a degeneracy-*producing* loss, so it
never tested the stated conditions. Declaring exhaustion here would have been a hard conclusion to walk
back from, on a confounded experiment. One re-run with plain BCE decides it cleanly.

**Cost.** ~17 minutes total: 1.6 for the library, 3.6 for the reach table, 9.9 for the training run, 1.8
for the loss probe. The in-memory savestates paid for themselves immediately — 539,040 prefix frames
avoided in the reach table alone (35,936 replayed instead of 574,976).

**Downstream effect.** `sustain_loss` is retired as a default. Every clearance figure produced by a
composed-recipe run is now known to rest on an objective that inflates button marginals, which is a
stronger statement than the earlier "the figures are marginal artefacts" — it names the mechanism and
predicts its magnitude in closed form. Reusable infrastructure that survives: `save_scratch`/`load_scratch`
on the session, the start-state library, and the per-start reach table.

---

## The clean arm: plain BCE beats the script at pipes 3–4, and stage 2's founding win was mostly a button rate

**What changed.** The previous entry found that the loss used by every "composed recipe" run inflates
button marginals by construction. This entry re-runs the training round with that one line changed, traces
which loss produced every historical result, and tests whether the project's founding win survives a
marginal-matched control. Two of the three answers are unfavourable to earlier claims.

**How we got there, including the wrong turns.**

The training round was re-run with **only** the loss changed — `sustain_loss` → plain BCE. The rollout
phase is deterministic given the base policy, the start library and the rollout seeds, so the identical
237 accepted rollouts were reused rather than re-rolled, making the comparison exactly one variable.

| | base | sustain arm | **plain arm** |
|---|---|---|---|
| x median | 723 | 311 | **723** |
| pipe1 / pipe2 / pipe3 / pipe4 | 73.0 / 68.5 / 39.0 / 19.0 | 47.5 / 47.5 / 18.5 / 12.0 | **79.5 / 77.0 / 46.5 / 17.5** |
| A / Down / Left | 0.852 / 0.281 / 0.108 | 0.970 / 0.756 / 0.398 | **0.831 / 0.256 / 0.096** |

`vs_script` for the plain arm: pipe1 −7.5 [−14.8, −0.2], pipe2 −5.5 [−13.3, +2.4], **pipe3 +23.0
[+13.7, +31.7]**, **pipe4 +9.5 [+3.0, +16.1]**. Movement from base: pipe1 +6.5, pipe2 +8.5, pipe3 +7.5,
pipe4 −1.5.

**What is solid and what is not.** Solid: the same data and seed under a press-weighted loss gives A 0.970
and x median 311, and under plain BCE gives A 0.831 and x median 723. That is a one-variable comparison and
the marginals read the objective directly. Not solid: the **+7.5 pp** pipe-3 gain comes from **one training
seed**, and this project's measured training-seed spread is **14.5–24.5 pp** — the gain sits inside the
noise band. Three obstacles also moved by a similar ~+7 to +8.5 pp, which looks more like a generally
better policy than obstacle-specific learning. Reported as a screen, not a win.

**Which loss produced every result — a code read.** Three objectives exist; the only plain-BCE arms in the
project's history are stage-2 arm A and this new one.

| loss | arms | measured A-rates |
|---|---|---|
| plain BCE (`onset_weight=1.0`) | 2 | 0.152 |
| onset 10× (**the default in `train_policy`**) | 5 | 0.219, 0.628 |
| onset 10× + sustain 5× (`compose.py`) | 6 | 0.822, 0.852, 0.865, 0.888, 0.926, 0.970 |

**The A-rate is ordered by press-weighting strength with no exceptions** — which is what makes the
mechanism a property of the objective rather than a coincidence in one run. `coverage_experiment`,
`compose_top20` and `compose_survival` all import `train` from `compose.py`, so the checkpoint every recent
figure rests on used the strongest-biased objective.

**Was the founding result real?** Stage 2's lineage descends from "bernoulli-only 29.5% → +onset-reweight
59.5% at pipe 1, +30.0 pp", and `arms.py` shows the two arms differ *only* in `onset_weight`. Re-measured
single life (the archived figures were multi-life):

| arm | A-rate | pipe 1 |
|---|---|---|
| arm A (plain BCE) | **0.152** — the expert's rate exactly | 23.0% |
| arm B (onset 10×) | 0.219 | **44.0%** |
| **arm A with A raised to 0.217** | 0.217 | **36.0%** |

**A single constant added to one logit — which cannot add state-dependent behaviour — reproduces +13.0 of
the +21.0 pp founding effect (62%).** The residual is +8.0 pp [−1.6, +17.4]: it neither excludes zero nor
is shown to be zero, and by this project's own power rule ~600 episodes per arm would be needed to resolve
8 pp. Both arms sit far below the script's 87.0%.

**Two mistakes worth recording.** The marginal intervention **overshot on the first attempt** and would
have been reported as marginal-matched when it was not: the logit offset was fitted offline on rows the
*expert* visits, but arm A visits its own states, so it realised **0.349** live against a 0.219 target.
Fixed with live bisection (δ=+0.387 → 0.218). Same failure family as calibrating on training rows. Second,
the script's verdict logic read "the residual does not exclude zero" as "onset reweighting contributed
nothing" — different claims; it now reports the decomposition and the power required instead.

**Cost.** ~20 minutes. The clean arm was 4.9 minutes because the deterministic rollouts were reused; the
loss provenance is a code read; the stage-2 test was 6.7 minutes and is now resumable from retained traces
after a restart destroyed a 10-minute run.

**Downstream effect.** Plain BCE is the default going forward, and `loss` is a required field beside every
number. The project's founding win is restated: most of it was a button rate, and the remainder is
unresolved rather than established. What survives as a genuine result is narrow and now measured under an
unbiased objective — **the policy beats the best fixed-rate script at pipes 3 and 4**, pending replication
across training seeds.

---

## Exhausted: three seeds, conditioned on arrival, show no improvement — and the loop is stable-degenerate

**What changed.** The single-seed result from the previous entry was replicated across three training
seeds and re-scored **conditional on arrival at each obstacle**, which is the form that separates
obstacle-specific learning from simply getting further up the level. It does not survive. This is the
project's terminal result.

**How we got there, including the wrong turns.**

The previous entry reported +7.5 pp unconditional at pipe 3 from one seed and flagged that the number sat
inside the known 14.5–24.5 pp training-seed band. Two things were then fixed about the measurement: the
metric was made conditional on arrival, and three seeds were spent.

Conditional advantage over the strongest fixed-rate script per obstacle:

| arm | pipe1 | pipe2 | pipe3 | pipe4 | airborne | A held while airborne | A-onsets/1k grounded |
|---|---|---|---|---|---|---|---|
| base | −14.0 | −3.0 | +25.8 | +9.7 | 79.3% | 85.2% | 3.8 |
| plain_s0 | −10.5 | −5.3 | **+28.9** | +14.5 | 79.1% | 85.5% | 3.2 |
| plain_s1 | −10.5 | +1.2 | +24.9 | +12.2 | 77.6% | 87.8% | 2.7 |
| plain_s2 | −15.0 | +2.5 | +17.8 | +16.7 | 77.3% | 87.8% | 2.6 |
| **pooled (n=600)** | −12.0 | −0.6 | **+23.8** | **+14.3** | **78.0%** | **87.0%** | **2.8** |
| *expert* | | | | | *61.1%* | | |

**The gate:** pipe 3 goes 56.9% → 55.0%, **−2.0 pp [−11.2, +7.6]**; pipe 4 goes 48.7% → 53.4%, **+4.6 pp
[−8.0, +17.1]**. Neither excludes zero.

**The previous entry's single-seed arm was an outlier in both directions.** Its +7.5 pp unconditional gain
at pipe 3 becomes −2.0 pp once conditioned and pooled, and its pipe-4 *conditional* advantage was −1.4 pp
while all three new seeds land at +12.2 to +16.7. One seed hid both errors at once.

**An honest rider:** pipe 4 improved in **3 of 3 seeds**, and the seed spread there is only 4.5 pp — the
wide interval comes from scarce *arrivals* (78 for the base, 238 pooled), not from disagreement between
seeds. Recorded as a fact, not as grounds for reopening a gate that fired.

**Why it cannot be fixed by fixing the loss.** The pooled policy's A-rate is **0.871 — exactly its
training data's 0.871.** Plain BCE faithfully reproduced a degenerate marginal, which is correct behaviour
on degenerate data. **The loss fix stopped amplification; it cannot stop inheritance.** Each round starts
from the previous round's marginal, and the self-data came from a sustain-trained base.

**The pathology, quantified for the first time.** The policy is airborne **78.0%** of frames against the
expert's **61.1%**, holds A on **87.0% of airborne frames**, and initiates only **2.8 A-onsets per 1,000
grounded frames** — roughly one jump start every 360 frames. It is not jumping often; **it is staying
airborne by never releasing the button**, which is exactly what blocks the next jump. No policy-side
clearance figure would have revealed this, which is why the A press *rate* was retired as a headline
statistic in favour of these three.

**Cost.** ~35 minutes wall, most of it spent making the run survive an environment that began killing long
jobs every 2–3 minutes. Three rounds of hardening, each a real lesson: per-arm resumption was too coarse
(a 200-episode arm exceeded the kill interval, so nothing ever completed); training had the same shape
(400 steps never reached a save, so model and optimiser state now bank every 100); and restart overhead
then dominated, until arm scores were cached, the training dataset built lazily, and **calibration dropped
from the eval path entirely — `traced_episode` samples from the sigmoid and never reads the thresholds, so
calibrating before an evaluation was pure cost.**

**Downstream effect — the project's conclusion.** Supervised imitation on this corpus is exhausted, and
the negative result is the contribution. Stated plainly: **up-weighting rare positive frames — a standard
trick for imbalanced action labels — manufactures a degenerate policy whose apparent competence is a
button rate. A three-button script (Right+B held, A flipped as a coin) matches or beats every learned
checkpoint in this project at the obstacles it was measured on. And an unbiased objective, a credit no
marginal can game, and practice states at the right obstacles do not recover the difference.** What
survives as genuine learned value is narrow: a conditional advantage at pipes 3 and 4 over the best fixed
script that is present in the base and is not improved by further training.

---

## The representational barrier: a per-frame policy never once produced the hold pipe 4 requires

**What changed.** The action space. The expert corpus was re-expressed as runs of a constant action — a
12-frame hold becomes **one** training sample with an explicit duration instead of twelve correlated ones —
and the resulting policy was compared against a per-frame control that differed in nothing else.

**Why this was the blocker.** Every solution the enumerative search finds is a macro-action: "jump at
x=892, hold A for 12 frames". A policy emitting 8 independent Bernoulli buttons per frame produces one with
probability ~p¹². That is why distilling 22 verified pipe-4 demonstrations moved the hold distribution the
*wrong* way. No teacher fixes it; the student could not represent the answer.

**How the implementation was chosen.** Two candidates existed: run-length tokens, or hold-duration counters
as observation inputs. Eight checkpoints from an old prev-action ablation sat in `data/bc2/` with no results
file, and reading their stored metrics decided it:

| arm | accuracy | **copycat rate** | label repeat rate |
|---|---|---|---|
| `noprev` | 0.726 | 0.726 | 0.970 |
| `prev4` | 0.967 | **0.995** | 0.970 |

Feeding previous actions reaches 96.7% accuracy by **copying the previous action 99.5% of the time** — the
counter approach's failure mode, already measured. Run-length tokens change the time base instead, so there
is no per-frame label to copy.

**Numbers.** Both arms: identical trunk, 3,000 steps, expert data only, unweighted loss. n=200, single life.
The expert reference was recomputed with the same windowed-onset function applied to the policies
(n=19, median 32.0, p90 70.0, max 72) and **independently reproduces the archived figure** of median 30,
p90 66, max 72 from a different segmentation.

| pipe-2 A-hold | expert | **run-length** | per-frame control |
|---|---|---|---|
| onsets measured | 19 | 237 | **2,852** |
| median | 32 | 7 | 1 |
| p90 | 70 | **30** | 2 |
| max | 72 | **86** | **6** |
| **fraction ≥ 12 frames** | 100% | **34.6%** | **0.0%** |

**The per-frame control produced 2,852 A-hold onsets at pipe 2 and not one reached 12 frames; its longest
was 6.** Pipe 4 requires ≥12. Per-frame independence did not make the macro-action merely unlikely — across
nearly three thousand attempts it never produced one.

Downstream of the representation, from the same data and budget: the control **never clears pipe 2 in 200
episodes** (0.0%) while the run-length arm clears it 25.5%, reaches pipe 3 6.0% and pipe 4 1.5%, and travels
to x=1,560 against the control's 596.

| | run-length | per-frame |
|---|---|---|
| A-hold anywhere, median / p90 / max | 7 / 31 / 101 | 1 / 2 / 11 |
| airborne | 42.6% | 77.7% |
| A held while airborne | 51.2% | 23.0% |
| A-onsets/1k grounded | 0.4 | 5.1 |

*Expert airborne fraction 61.1%.*

**What this is not.** `vs_script` is **negative at every obstacle for both arms** — run-length pipe1 −53.5,
pipe2 −57.0 pp. An expert-only policy at 3,000 steps is far weaker than the fixed-rate script, and the trap
of "finishing with a policy a tuned script also finishes" is fully in force. This result is about what the
policy can *express*, not what it can *do*. The expert reference also rests on only 19 holds, so the
comparison that does not depend on it is the ≥12 fraction and the maximum.

**Cost.** 12.4 minutes of compute, spread across many restarts. One real defect fixed along the way:
`build_index` walked ~1M rows in Python at every start, and with the environment killing this job every 2–3
minutes that rebuild consumed most of each cycle — 2,000 steps took ~20 minutes until it was cached, after
which the remaining 1,000 landed immediately.

**Downstream effect.** The representational question, open since the pipe-4 distillation failed, is closed:
**the barrier is real and removable.** Search-and-distil now has a student that can execute what search
finds. The next constraint is absolute strength, not expressiveness.

---

## The owner's sixth observation: capping no-op runs doubled x median with no retraining

**What changed.** One number in the generation rule. Runs whose button combo contains no A are capped at 4
frames; A-containing runs are left uncapped. Same trained checkpoint, same weights — only how a predicted
(combo, duration) gets executed.

| | median rule | **capped** |
|---|---|---|
| longest no-op run | **347 frames** | **12** |
| airborne | 42.6% | **66.7%** *(expert 61.1%)* |
| pipe-2 holds ≥12 frames | 34.6% | **36.8%** |
| x median | 314 | **702** |
| pipe 1 / 2 / 3 / 4 | 33.5 / 25.5 / 6.0 / 1.5% | **64.5 / 61.0 / 18.0 / 9.5%** |
| `vs_script` pipe 1 / 2 / 3 / 4 | −53.5 / −57.0 / −17.5 / −6.5 | **−22.5 / −21.5 / −5.5 / +1.5** |

**How we got there.** The owner watched the run-length policy and said *"Mario stays still for a lot of
time; in some instances he wasn't doing anything."* That is the sixth such observation and the sixth to be
right. The discriminating measurement was: when stationary, is he holding Right (pressed against terrain —
competence) or holding nothing (inside an emitted no-op block — generation)?

**The answer was both, split by where you look:**

- **In aggregate the no-op emission is correct** — zero-button fraction 17.6% against the expert's 17.0%,
  no-op run median 10 against 9.
- **The tail is fatal** — longest no-op run 347 frames against the expert's 53. Half a second is 30 frames;
  this is nearly six seconds of holding nothing.
- **Stationary frames are mostly Right** (52.4% vs 24.5% nothing), so most idling really is competence — but
  holding-nothing at 24.5% is 3.4× the expert's 7.2%, which is the long blocks showing through.

**A premise that did not survive checking.** The reasoning for this experiment held that the expert's most
common action is *nothing* at 40.3% of frames, so a run-length encoding would emit those as blocks. On 1-1
surface frames the expert's zero-button fraction is **17.0%** — the 40.3% is a whole-corpus figure including
transitions. The mechanism was still real; the magnitude behind it was not.

**Three variants, one checkpoint, no retraining** — the variants change execution, not training:

| arm | ≥12 at pipe 2 | airborne | no-op max | x med | pipe 2 |
|---|---|---|---|---|---|
| median (baseline) | 34.6% | 42.8% | 347 | 314 | 25.5% |
| (a) sample the length | 45.6% | 41.6% | 343 | 315 | 32.0% |
| **(b) cap non-A runs** | 36.8% | **66.7%** | **12** | **702** | **61.0%** |
| (c) re-decide every frame | **0.0%** | 82.4% | 4 | 436 | 2.0% |

**(a) barely helps** — the expert's own length distribution contains the long runs, so sampling from it
reproduces the tail. **(c) destroys the durations entirely** — ≥12 collapses to zero and pipe 2 to 2.0%,
which confirms the committed duration is what produces long holds. Re-measuring the baseline through this
new code reproduced phase 1 exactly (≥12 34.6%, x med 314, pipe 2 25.5%).

**A gate declined, with the reason.** The pre-committed condition asked for A-onsets per 1,000 *grounded*
frames above 2 while holding ≥12 above 20%. No variant passes: (c) gets the onsets and loses the holds; the
rest keep the holds near 1 onset/1k. **That metric is not comparable across these arms** — airborne fraction
ranges 41.6% to 82.4%, so identical jump rates divide by denominators differing threefold. On the
total-frame denominator, which has an expert reference, `capped` sits at **48.0 against the expert's 27.5**:
it jumps *too often*, not too rarely. The premise that it "almost never chooses to jump" was an artifact of
the normalisation.

**What this is not.** **No arm beats the fixed-rate script anywhere.** `capped` is 21.5 points short at pipe
2, where the best per-frame checkpoint was 14.0 short — so it is not yet better than the old lineage at pipe
2, and the script remains the bar. Its A marginal is also **0.572, 3.8× the expert's 0.152**: capping non-A
runs necessarily gives A-runs more wall-clock, so some of the reach is bought by jumping more. Airborne
66.7% against 61.1% says this is not the old 85% degeneracy, but it is the same direction.

**A measurement that could not be taken.** The expert's A-onsets per 1,000 *grounded* frames is unavailable
as a read: the trace schema has no `on_ground` column, and deriving it from y is the error that cost this
project seven failures. Replaying expert inputs against the session's savestate was tried and **rejected by
its own validation — all 20 runs mismatched on x**, because each publication has its own movie. The
substitution to a total-frame denominator is stated wherever the number appears.

**Cost.** ~40 minutes wall, most of it the four 200-episode evaluations. No training.

**Downstream effect.** Phase 2 should take the capped rule. The generation rule, not the representation, was
costing 31–36 points of script gap at every obstacle — and the fix was found by a human watching the game
rather than by any metric in the artifact.

---

## The first skill signal: at its own button rates, the policy beats the script by 54 points at pipe 2

**What changed.** Nothing was built. A control was run — the one this project existed to run. `capped`'s
reach might have been another marginal shift, since capping non-A runs raises its A-rate to 0.572 (3.8× the
expert's). So a fixed-rate script was run at **that** rate, with `capped`'s other marginals matched, on the
same 200 seeds.

**Numbers.** Two readings of "rate-matched", because they bracket the answer:

| | script, all 5 marginals matched | script, Right+B **held**, A=0.572 | **`capped`** |
|---|---|---|---|
| pipe 1 | 79.5% | **83.0%** | 64.5% |
| **pipe 2** | **7.0%** | **7.0%** | **61.0%** |
| pipe 3 | 0.0% | 0.0% | **18.0%** |
| pipe 4 | 0.0% | 0.0% | **9.5%** |
| x max | 724 | 724 | **1,559** |

Against the stronger control: pipe 2 **+54.0 pp [+45.8, +61.1]**, pipe 3 **+18.0 [+12.9, +23.9]**, pipe 4
**+9.5 [+5.7, +14.4]**. Conditional on arrival, pipe 2 is **+86.1 pp [+78.7, +90.5]**. **Neither script
clears pipe 3 or pipe 4 once**, and both stop at x=724 — pipe 3's face. The pipe-4 conditional interval is
undefined because the opponent never reaches pipe 4's gate at all.

**It loses pipe 1** by −18.5 pp [−26.7, −9.9]. This is not "better everywhere": it is better exactly where a
fixed rate cannot work, and worse on the one obstacle a fixed rate handles fine.

**How we got there, and the prediction that failed.** The expectation was that p(A)=0.572 would interpolate
between the p=0.50 arm's pipe-2 clearance of 10.0% and the p=0.85 arm's 68.5%. It came in at **7.0%**.

**A correction to how this entry first read it.** I described 7.0% as landing *below* the p=0.50 arm and
called pipe-2 clearance non-monotonic in p(A). **That comparison is not available: the p=0.50 figure is n=20,
i.e. two episodes** — 2/20 = 10% [1.7, 30.1] against 14/200 = 7% [4.2, 11.4], intervals overlapping almost
entirely. There is no non-monotonicity to explain, and comparing an n=20 screen with an n=200 measurement as
though the ordering meant something is a mistake this document's own §4 warns about.

**The real reason the prediction failed is the functional form: a p^L curve was interpolated linearly, and it
is convex by a factor of ~50 across that range.**

| A-rate | P(10 consecutive A) | P(11) | measured pipe 2 |
|---|---|---|---|
| 0.500 | 0.10% | 0.05% | 10.0% |
| **0.572** | **0.37%** | **0.21%** | **7.0%** |
| 0.850 | 19.69% | 16.73% | 68.5% |

Pipe 2 requires A held 10–11 frames. **A per-frame sampler at 0.572 produces that 0.375% of the time;
`capped` produces ≥12-frame holds on 36.8% of its pipe-2 onsets — about 170× more often at the same
marginal.** That ratio is the macro-action argument stated quantitatively, and it does not depend on the
n=20 screen at all.

**Both bars are kept, because they answer different questions.** Against the *best* script (p(A)=0.85 with
Left, pipe 2 82.5%) `capped` is still **−21.5 pp** — that is the bar `FINDINGS.md` uses and it remains unmet.
Against a script at **its own rate** it wins by +54.0. The first asks whether this is the best way to play
1-1; the second asks whether its play reduces to its button rates. **It does not.**

`capped`'s pipe-2 hold distribution, reported with tails per the rule earned by a 347-frame idle run whose
median looked fine: **median 7.0, p90 34.0, p99 68.6, max 90**, against the expert's median 32.0 / p90 70.0 /
max 72. Its p99 and max now sit at or above the expert's; its median is still well short.

**Cost.** 5 minutes. No training.

**Downstream effect.** This is the first genuine skill signal from the run-length line and the first anywhere
in this project outside pipe 3 — behaviour a marginal cannot reproduce at the rate the policy actually runs
at. The frontier has also moved: **107 of `capped`'s 151 deaths are at x≈256, the first Goomba**, which is
now the dominant single loss and the obstacle where 75 of 80 scripted timings succeed.

---

## Phase 2, the Goomba: distillation did nothing, because the Goomba's solution is a marginal

**What changed.** Search-and-distil was run end to end on the level's dominant loss for the first time: a
sweep from the policy's own start states, the winning solutions re-encoded as run-length training samples,
distilled with plain cross-entropy, and measured against both script bars. **It did not work, and the reason
reframes what "the dominant loss" meant.**

**Numbers.** Threshold derived, not chosen: `capped`'s max_x histogram piles at 272/288/304 and **x 320–431 is
completely empty**, so x>320 is the first x past the Goomba's far edge. n=200 throughout.

| arm | Goomba (x>320) | deaths 272–319 | airborne | A | pipe2 | pipe3 | pipe4 |
|---|---|---|---|---|---|---|---|
| `capped` (before) | 65.0% | 69 | 66.7% | 0.572 | 61.0% | 18.0% | 9.5% |
| **distilled** | **64.0%** | **72** | 66.1% | 0.533 | 57.0% | **21.5%** | **13.0%** |
| **rate-matched script** | **83.0%** | **34** | **88.3%** | 0.533 | 4.0% | 0.0% | 0.0% |

Goomba change: **−1.0 pp [−10.3, +8.3]**. Nothing.

**The finding: a coin-flipping script at the policy's own button rates clears the Goomba 19 points better
than the policy does.** The mechanism is one column — the script is airborne **88.3%** of frames against the
policy's **66.1%**. **The Goomba is not a timing obstacle, it is an "be in the air a lot" obstacle**, and a
script that jumps constantly is almost never on the ground to be hit. Distilling macro-actions cannot fix
that, because the solution is not a macro-action. It is a marginal.

**The sweep confirmed this before the distillation did.** From grounded start states just before the Goomba,
**1,005 of 1,152 configurations cleared and survived — 87.2%** — with the minimum winner a **single-frame**
tap at x=252. There was no needle to find.

**And that exposes the deeper mistake, which was mine.** I drew start states that were **grounded** at a
pre-Goomba x. That guarantees a good approach — which is exactly what the policy fails to achieve. **The
policy's Goomba deaths come from arriving in the wrong state, not from jumping wrongly, and the filter
excluded that by construction.** A sweep that begins after the failure cannot demonstrate its fix. This is the
same shape as conditioning on arrival: condition on being in a good state, and the obstacle looks easy.

**A second defect of mine cost an entire arm.** Run-length encoding compresses ~100 demonstration frames into
~3 run samples, so 48 winning attempts became **144 samples across 7 classes**. A 1:1 expert:demo ratio then
**capped the expert side at 144 as well** — 288 samples total, and 300 steps over them is **133 epochs**,
discarding 77,772 of 77,916 available expert samples. That arm regressed the Goomba to **53.5%**. It is the
pipe-4 distillation failure reproduced at ten times the severity, by me, one block after quoting it as the
thing not to do. Corrected to 20,000 expert samples with demos at 9% and **1.7 epochs**.

**What the demos did do.** Downstream of the Goomba the distillation helped: **pipe 3 18.0% → 21.5%**, **pipe 4
9.5% → 13.0%**. Against the best fixed-rate script, pipe 3 is now **−2.0 pp** (from −5.5) and pipe 4 is
**+5.0**. Against a script rate-matched to its own marginals it still wins pipes 2–4 (+53.0, +21.5, +13.0) and
still loses pipe 1 (−15.0).

**Cost.** ~50 minutes, two distillation arms and five 200-episode evaluations.

**Downstream effect.** **Pipe 1 and the Goomba are one problem, and it is a marginal problem:** the script
beats the policy at both by being airborne 88% of the time, while the policy beats the script everywhere a
sustained placed action is needed. Finishing 1-1 requires both halves, and search-and-distil only addresses
the second. The next lever for the first is generation and marginals — the same lever that took `capped` from
x median 314 to 702 — not another obstacle sweep.

---

## Goomba forensics: half the deaths stand on the ground next to the enemy and never jump

**What changed.** No build — a read over retained traces, answering the question the sweep was meant to
inform and could not: what does the policy actually do in the frames before it dies at the Goomba?

**Numbers.** 200 episodes: 69 died in x 272–319, 130 cleared x>320. A-onset presence per episode:

| A-onset window | died (n=69) | cleared (n=130) | difference |
|---|---|---|---|
| approach, 200–260 | **78.3%** | 63.8% | **−14.4 pp [−26.2, −0.9]** |
| **at the Goomba, 260–320** | **50.7%** | **86.2%** | **+35.4 pp [+22.0, +47.9]** |
| the expert's own peak, 272–304 | 27.5% | 68.5% | **+40.9 pp [+26.7, +52.7]** |

**Jumping early predicts death; jumping at the enemy predicts clearing.** The deaths jump *more* than the
clearers during the approach and far less at the obstacle itself.

**The finding: 34 of the 69 deaths never press A anywhere in 260–320, and in that stretch they are grounded
93% of the time (median), with none airborne throughout.** They jumped early (24 of the 34 had), landed, and
walked into the enemy on foot. They could jump and did not. The other 35 jump at the Goomba and die anyway.
**So the failure is roughly half timing, half not acting at all.**

**A mechanism I proposed and the data refuted.** I expected the non-jumpers to be airborne from an early jump
and therefore unable to jump again — SMB requires releasing A to jump. **The opposite holds: deaths are
grounded when first reaching x=272 on 47.8% of episodes against the clearers' 26.2%, −21.7 pp [−35.1, −7.7].**
Deaths are on the ground *more* than clearers. Being airborne is not what kills them; being grounded and
passive is.

| grounded fraction in x 260–300 | median |
|---|---|
| deaths that did **not** jump at the Goomba | **0.93** |
| deaths that jumped at the Goomba | 0.14 |
| clearers | 0.18 |

**A correction to the previous entry, caused by a window I chose badly.** That entry concluded the Goomba was
a marginal problem. Part of that came from measuring an "approach" window of 200–260 — and **the expert's
Goomba onsets sit at 272–304, outside it.** I measured everywhere except where the expert actually jumps. The
airborne-fraction argument remains a correct description of the *script's* exploit (88.3% airborne, 27 points
above the expert's 61.1%), but the policy's own failure is now located and it is not a marginal.

**The expert's jump positions**, read from the corpus for the first time: A-onsets in x 180–320 are
**bimodal** — an early cluster at 192–208 and a second at 272–304, the Goomba jump. n=30 across 25 runs, so
this locates the jump but supports no distributional claim.

**What this says about search.** The last sweep drew grounded start states before the enemy and found 87.2%
of configurations winning. **But for half the deaths the failure is not "wrong jump from a good state", it is
"no jump from a good state" — and every configuration in a sweep jumps by construction.** The missing
demonstration is not a better trigger; it is the decision to trigger at all.

**Cost.** A pass over files already on disk.

**Downstream effect.** The Goomba deficit is a missing *decision* from a grounded state — neither a missing
demonstration nor a marginal. If that generalises to pipe 1, then the project's two competing goods — absolute
performance, which argues for a higher airborne fraction, and defensibility, which argues for the run-length
line's expert-like behaviour — stop competing, because the fix would be to act more often *when grounded*
rather than to spend more time in the air.

---

## The policy never decides to jump: p(A) is 0.43 where it dies and 0.46 where it survives

**What changed.** A read of the policy's own output probability at the obstacle it fails, to separate two
diagnoses whose fixes do not overlap: does it fail to recognise the state, or does it recognise the state and
gamble?

**Numbers.** p(A) is the summed softmax over the 107 (of 300) classes whose button combo contains A — the
quantity the sampler draws against. Grounded frames only, x 272–304:

| group | p(A) median | p90 | p99 | max | frames |
|---|---|---|---|---|---|
| deaths that never jumped | **0.430** | 0.491 | 0.511 | **0.526** | 364 |
| deaths that jumped | 0.424 | 0.482 | 0.499 | 0.505 | 202 |
| clearers | **0.459** | 0.488 | 0.504 | **0.507** | 338 |

**Gap: +0.029.** The fraction of grounded frames with p(A) > 0.5 is 3.8% in the deaths against 2.1% in the
clearers — if anything backwards, −1.8 pp [−4.5, +0.9]. **The policy assigns the states where it dies and the
states where it survives essentially the same probability of jumping. It knows and gambles.**

**A contrast that makes it sharper.** At pipe 2 — where this same policy *succeeds*, clearing 61% — p(A) is
**lower still: median 0.288, max 0.399 over 7,254 grounded frames.**

**So the policy is never confident about jumping anywhere.** It runs at 0.29–0.46 and never exceeds 0.53. What
makes it clear pipe 2 is not a higher probability of jumping — it is the **duration attached to the A-class it
happens to sample**, since one sampled class carries a 20–70 frame hold. **The run-length representation
converted "jump" from a per-frame coin into a single coin with a long payoff, and that is the entirety of its
advantage over the per-frame line.**

It also explains the Goomba specifically: non-A runs are capped at 4 frames, so crossing the 32-px approach
gives ~4–5 decisions at p≈0.44 each. Whether an A-run starts before the enemy is close to a fair coin, and the
policy's own probability barely distinguishes the states where it matters.

**A planned build, cancelled by its own precondition.** The next step was to be the project's first properly
DAgger-shaped experiment: sweep from the 34 states where the policy actually fails, distil those solutions,
and measure whether the decision transferred. **Its premise was that the policy discriminates those states and
gets them wrong. It does not discriminate them** — so the demonstrations would have been aimed at a network
that already treats both groups alike, and this project has twice measured what distilling into the wrong
constraint produces. **Not built.**

**A defect flagged rather than left standing.** The pipe-2 contrast reused the same probe with a different
window, and its printed verdict line — *"it fails to recognise the state"* — is **invalid there**: the death
groups are defined by the Goomba death band, so at pipe 2 that group has no grounded frames, the median
returns `None`, and the gap is computed against zero. Only the `cleared` row of that run is meaningful. The
artifact records this; the log line did not.

**Cost.** ~3 minutes of emulator replay. No training.

**Downstream effect.** The Goomba deficit is now located precisely: **not a missing demonstration, not a
marginal, but a middling probability that sampling turns into a coin flip.** The indicated lever is the
generation rule — converting a middling p(A) into a decision when grounded — which is one screen over the
existing checkpoint with no retraining, the same shape as the change that doubled x median from 314 to 702.

---

## Can the network read the timing? It reads *where* the Goomba is, barely *when* to jump

**What changed.** Forward passes over expert frames — no emulator, no training — to decide whether the 84×84
observation carries the Goomba's timing at all. If it did not, no amount of imitation, search or distillation
on this input could ever fix the obstacle.

**First measurement.** For each expert A-onset in x=272–304, p(A) at the onset against its ±10 neighbours:

| offset | −10 | −5 | −1 | **0** | +1 | +5 | +10 |
|---|---|---|---|---|---|---|---|
| p(A) median | 0.402 | 0.331 | 0.448 | **0.458** | 0.459 | 0.379 | 0.308 |

Onset 0.458 against flanks 0.366, paired difference **+0.103 [+0.093, +0.124]** — a clean unimodal peak.

**Then the check that changed the reading.** ±10 frames is ~50 px at expert speed while the window is 32 px
wide, so the flanks sit at *different x*. If p(A) varies with position, the peak could be spatial rather than
temporal. It does vary:

| x bin | 180–195 | 228–243 | 260–275 | **276–291** | 292–307 | 308–323 |
|---|---|---|---|---|---|---|
| p(A) mean | 0.299 | 0.282 | 0.399 | **0.439** | 0.326 | 0.299 |

**p(A) swings 0.157 across x and peaks exactly at 276–291 — the Goomba's position.** Comparing onset frames
against non-onset frames *at the same x*:

| | value |
|---|---|
| pooled onset mean | **0.413** (n=30) |
| pooled non-onset mean | **0.345** (n=1,191) |
| difference | **+0.067** |
| per-bin | +0.082, +0.029, +0.017, +0.006, +0.031, +0.034, **−0.024**, +0.035 |

**So most of the +0.103 was the spatial gradient.** The state-conditional component — higher p(A) where the
expert actually jumps, holding position fixed — is **+0.067 pooled**, +0.006 to +0.082 per bin, negative in one
of eight. **0.157 of spatial swing against 0.067 of timing lift.**

**What this settles.** **The observation is not blind.** The pessimistic branch is refuted: the network resolves
the Goomba's position, so the obstacle is not unfixable in principle and the project's shape need not change.
**What remains open** is whether a +0.067 margin, on a policy whose p(A) never exceeds 0.53, is enough to
sharpen into a decision.

**A planned build declined, for a reason worth recording.** The follow-up was to temperature-sharpen the
policy's signal and accept the change only if the **A-rate-by-x spread** widened rather than its level. **But
sharpening a mostly-spatial signal widens the by-x spread** — that is exactly what a sharpened spatial profile
looks like — so the test would pass for "jump harder where the Goomba roughly is", which is nearer the airborne
exploit than to timing. **The discriminator that does separate them is the x-matched onset gap** (does the
+0.067 widen?), which isolates state-conditional lift from positional lift by construction. One line in the
acceptance criterion, not a different experiment.

**Sample size, stated plainly.** The specified window x=272–304 contains only **7** expert onsets, not the ~30
assumed — 30 is the count for the wider 180–320 window, which is where the x-matched control was run. Thirty
onsets across eight bins is a screen and is labelled one.

**Cost.** ~1 minute of forward passes. No emulator, no training.

**Downstream effect.** The Goomba is now characterised on both axes: **coarse spatial conditioning (0.157),
weak temporal conditioning (+0.067), and a policy that never exceeds p(A)=0.53 anywhere.** That is a complete
mechanical account of why it dies there — and it says the lever is sharpening the signal that exists, judged by
the x-matched gap rather than by positional spread.

---

## The policy has no timing anywhere — at pipe 2, where it succeeds, the timing signal is negative

**What changed.** One read, forward passes only, extending the Goomba probe to every obstacle. It closes the
question the whole run-length line rested on.

**Numbers.** Stratified estimator: within each 16-px bin, mean p(A) at expert A-onset frames minus mean at
non-onset frames, weighted across bins by onset count — position held fixed by construction. Intervals
bootstrapped over onsets.

| obstacle | onsets | **x-matched lift** | 95% CI | positional swing |
|---|---|---|---|---|
| goomba 288 | 30 | **+0.035** | [+0.022, +0.048] | 0.144 |
| pipe1 432 | 36 | **−0.011** | [−0.022, −0.001] | 0.072 |
| **pipe2 592** | 32 | **−0.012** | **[−0.023, −0.002]** | 0.138 |
| pipe3 720 | 8 | −0.014 | [−0.026, +0.005] | 0.147 |
| pipe4 912 | 12 | −0.017 | [−0.032, −0.004] | 0.000 |

**At pipe 1, pipe 2 and pipe 4 the policy assigns *lower* p(A) at the frames where the expert jumps than at
neighbouring frames at the same x.** The only obstacle with a positive lift and enough onsets to say so is the
Goomba — the one it fails. **Positional conditioning is 4–12× larger than temporal everywhere.**

**So the policy has no timing signal anywhere, and the least of it where it performs best. Its competence is
duration.** That is a one-sentence account of every result in this project: **it beats a rate-matched script
wherever a long hold suffices and loses wherever placement matters.** The +53 pp at pipe 2 is representational
— the run-length head turned "jump" from twelve coins that must all land into one coin with a long payoff —
and there is no state-conditional component underneath it.

It also explains, without further assumption: why distilling verified Goomba solutions changed nothing (the
demonstrations encode *when*, which the policy cannot represent conditionally); why p(A) never exceeds 0.53
anywhere (it never has a state-specific reason to commit); and why it loses pipe 1 and the Goomba to a script
that merely jumps a lot.

**A planned build cancelled by its own precondition, for the second block running.** Temperature sharpening
amplifies whatever signal exists. **At pipe 2 that signal points the wrong way**, so sharpening would produce a
more confident wrong answer. Not built.

**Two corrections, and the second is to my own figure.**

The **+0.103** onset-vs-flank result is **withdrawn**: it was a bootstrap of a median over **7** values, whose
distribution is discrete over those seven points, so its interval understated uncertainty rather than measuring
it.

And my **x-matched +0.067 was inflated by a milder form of the confound I had just criticised.** I computed it
as a pooled onset mean (0.413) minus a pooled non-onset mean (0.345) — but onsets and non-onsets have different
x distributions inside the window, so the pooling re-imported part of the positional gradient. Stratifying and
weighting by onset count gives **+0.035 [+0.022, +0.048]** over the identical window. **The Goomba
decomposition is 0.144 positional against +0.035 temporal**, and the conclusion strengthens — my own number
had been too generous to the timing story.

**Sample sizes, stated because three windows cannot carry a claim.** pipe3 (8 onsets), pipe4 (12), koopas (7)
and the gap (3) are too thin; the gap's +0.038 rests on three onsets and must not be cited. The load-bearing
rows are goomba (30), pipe1 (36) and pipe2 (32) — and two of those three are where the policy performs best,
both negative.

**Cost.** 0.1 minutes. No emulator, no training.

**Downstream effect.** The run-length line's advantage is now fully characterised and fully bounded: **it is a
representational win with no timing content.** Sharpening and search are both ruled out as levers by
measurement rather than by argument. What remains untested is the **observation** — resolution, or explicit
enemy-relative features from RAM, which the trace logger already records at death. That is a change of input
rather than of method, and therefore a different project than the one this document describes.

---

## Block 53 — The previous entry is overturned: it rested on a single training run

**The entry immediately above this one stays exactly as written.** It was an honest account of what we
believed and why. It is also wrong, and this entry says how we found that out.

### What changed

**"The policy has no timing anywhere" is withdrawn.** So is "its competence is duration, not timing," and so
is the elimination argument that made changing the observation the only remaining lever. All three were
inferred from the timing lift measured on **one checkpoint**, `runlength.pt`. Retrain the same recipe and the
sign flips. The policy does discriminate when to jump, including where it succeeds.

**What replaces it is narrower and more useful:** the timing signal is real, it grows sharply with input
resolution, and **behaviour does not use it.**

### How we got there, including the wrong turn

The advisor's directive opened by correcting the estimator, and the correction was legitimate. Under
run-length encoding the model only ever decides at run boundaries, but the baseline we had been comparing
against — "non-onset frames at the same x" — turned out to be **93.4% mid-run**, inputs the model is never
trained to score. So the comparison was in-distribution onsets against off-distribution neighbours. We
rebuilt it as onsets versus **non-A run starts**, which holds "this is a decision point" fixed.

The corrected estimator did not change the answer: pipe 2 stayed negative, −0.026 against the confounded
−0.012. We recorded that the finding survived and moved on to the scale-up.

**The wrong turn is visible only in hindsight, and it is a methodological one rather than a coding error.**
The fifty-first block measured one checkpoint. The correction re-measured *the same checkpoint*, more
carefully. A better instrument applied to the same single sample cannot reveal that the sample is unusual —
and our own operational ledger has said since block 39 that *one seed is a screen, not a ranking*. We had
written the rule down, cited it against other people's numbers, and then walked straight into it. Neither the
advisor who wrote the directive nor the builder who executed it caught it at the time.

What caught it was incidental. The scale-up needed an 84×84 control trained for the same 15,000 steps as the
new arms, because comparing against a 3,000-step checkpoint would confound resolution with training length.
We measured that control's timing lift as a throwaway sanity check before spending the expensive part of the
budget — and it came out **positive**. At that point the responsible move was not to celebrate but to go
backwards: re-run the *old* recipe, 3,000 steps at batch 128, through the new pipeline. That is the arm that
settles it, because it changes nothing except which particular training run you are looking at.

### The numbers, with sample size and baseline

Corrected estimator throughout; stratified over 16-px bins, weighted by onset count, bootstrapped over
onsets. Everything below is 84×84, `d_model=64`, one transformer layer, the same frozen expert-train split,
plain cross-entropy, run-length joint action classes. **Seeds vary weight initialisation as well as data
order.** Pipe-2 onsets n=32.

| arm | steps | batch | pipe-2 corrected lift |
|---|---|---|---|
| `runlength.pt` — the previous entry's checkpoint | 3,000 | 128 | **−0.026 [−0.035, −0.016]** |
| same recipe, retrained | 3,000 | 128 | **+0.028 [+0.013, +0.042]** |
| longer, seed 0 | 15,000 | 64 | **+0.063 [+0.025, +0.101]** |
| longer, seed 1 | 15,000 | 64 | **+0.177 [+0.127, +0.226]** |
| longer, seed 2 | 15,000 | 64 | **+0.130 [+0.083, +0.178]** |

Four freshly trained checkpoints, four positive lifts, every interval excluding zero. The checkpoint the
conclusion was built on is the outlier. A by-product worth keeping: the **seed spread on a lift is 0.114**,
which is the first noise floor we have ever measured for this quantity and the number future lift comparisons
have to clear.

Then the scale-up itself, n=200 episodes per arm, single life, one training seed each, evaluated identically:

| arm | resolution | transformer | pipe 1 | pipe 2 | pipe 3 | pipe 4 | pipe-2 timing lift |
|---|---|---|---|---|---|---|---|
| B | 84×84 | d64, 1 layer | 66.0% | 63.5% | 25.5% | 9.5% | +0.063 |
| R | 128×128 | d64, 1 layer | 61.5% | 55.5% | 22.5% | 6.0% | **+0.301** |
| RT | 128×128 | d128, 2 layers | 63.5% | 55.0% | 26.0% | 5.0% | **+0.324** |

**The timing lift rises roughly fivefold and clearance does not move.** At the Goomba — the obstacle this
policy actually fails — the larger network reaches **+0.366**, the strongest timing signal recorded anywhere
in this project, and clears it no better. We are explicitly *not* claiming resolution hurt performance: the
8 pp gap at pipe 2 sits well inside a documented 14.5–24.5 pp training-seed spread, and these arms are one
seed each. The supported claim is that clearance **did not improve**.

Against the standing bar, nothing has changed: no arm at any resolution beats the best fixed-rate script at
pipes 1–2 (the best, B, is 19.0 pp short at pipe 2). Against a script matched to each arm's own button rates,
all three win pipes 2–4 by +48 to +49 pp at pipe 2 and lose pipe 1 — the same shape as before.

### Cost

About four hours end to end, most of it the 26 GB re-capture at 128×128 and six sets of 200 evaluation
episodes. The training itself stopped being the expensive part: moving it to the GPU took 15,000 steps from
roughly 40 minutes to **2.9 minutes**, which is the only reason running four extra checkpoints to check a
seed was affordable at all. That is the practical lesson hiding inside the methodological one — **we had been
treating one seed as sufficient partly because a second seed used to cost most of an afternoon.**

Two incidental defects were found and fixed. Seven of the thirty-four movie paths in the capture plan still
pointed at the project's pre-move directory, so the first capture pass silently produced 27 of 34 runs — and
all seven missing runs were in the training split. And the capture command estimated 11.18 GiB of disk for a
job that wrote 25.7, because it reported a figure frozen at the old resolution rather than the one requested.

### Downstream effect

The observation is no longer the indicated next lever, and the two things the previous entry ruled out —
sharpening, and search — are ruled back in, because the premise that ruled them out has been withdrawn.

More usefully, the dissociation narrows the target. The network's probability of jumping now discriminates
the expert's jump moments substantially better at higher resolution, and the agent's behaviour is unchanged.
Whatever is binding sits *between* the probability and the action: the generation rule that samples once from
the softmax, commits to a duration, and truncates non-jump runs at four frames. That is a much smaller and
more testable object than "the observation," and it costs nothing to capture.

The broader consequence is that several earlier results need re-checking rather than re-deriving. The claim
that the policy is never confident about jumping, the decomposition of its conditioning into spatial and
temporal parts, and the per-obstacle lift table were all measured on that same single checkpoint. None of
them is refuted by this entry. All of them are now single-seed, and single-seed is a screen.

---

## Block 54 — A plausible fix that did nothing, and a spectacular result that was a sample of one

### What changed

Two things, both negative, and the second is more useful than the first.

**Capping the jump did nothing.** The policy's jump-hold had no upper bound: one sample could commit it to
over two seconds of airtime. Bounding that hold at 12, 24 or 48 frames does not improve clearance at the pipe
we measure against. The intervention is dead in the form we tried.

**And a configuration that briefly looked like the answer turned out to be a single coin flip.** Taking the
most likely action instead of sampling produced, on one checkpoint, the deepest trajectory this project has
ever recorded — past three obstacles, reaching x=916 — with button statistics closer to a human expert's than
anything we have built. Run the identical rule on two other training runs of the same recipe: one reaches x=316
and clears nothing, the other stops at x=723. It is not a policy that succeeds 92% of the time. It is one
trajectory, and we happened to look at a good one first.

### How we got there, including the wrong turn

The reasoning behind the cap was sound and is worth stating because it was wrong in an instructive way. An
earlier block had capped the *no-op* runs — the stretches where the policy holds nothing — at four frames, and
that single change doubled how far it got. The jump runs had never been capped. The policy starts about 1.6
times as many jumps as the expert and holds some of them for 130 to 304 frames, so bounding the jump looked
like the obvious symmetric fix, and it needed no retraining.

**The measurement said the mechanism was backwards.** A tight cap made the policy jump *more*, not less:
truncating a hold forces an immediate re-decision, and it simply jumps again. Jump starts per thousand frames
went *up* from 45 to 56, and airborne fraction went up too. Worse, the obstacles that need a long sustained
hold — the taller pipes — collapsed from 25% cleared to 12%. We had reasoned about the cap as though it
subtracted airtime. It redistributed it into more, shorter jumps.

**The wrong turn I want on the record is the one that didn't happen.** The advisor's directive had insisted,
as a hard requirement, that any winning configuration be confirmed on two further training runs before being
reported. When the greedy-action arm returned 91.7% at three consecutive obstacles, that requirement was the
only thing standing between us and a headline. It took about ten minutes of looking at the number before the
replication came back at 0.0% on the next seed.

Two properties of the setup made this trap unusually well concealed. The game is deterministic and the start
state is fixed, so a policy that always takes its most likely action produces **exactly one trajectory** no
matter how many episodes you run. We ran two hundred and the scorer dutifully reported a percentage, because
nothing in it knew that the two hundred episodes were the same episode. And they were *almost* the same: two
hundred runs produced two distinct trajectories, which is exactly the kind of near-uniformity that reads as
"low variance" rather than "no variance."

That second trajectory turned out to be its own small discovery. **Episode zero of a freshly started emulator
session differs from every subsequent episode**, because the initial frame buffer is filled from the first
reset's frame and that frame is not identical to later resets'. For every stochastic policy we have ever
evaluated this is invisible — the episodes all differ anyway — and at two hundred episodes it is half a
percent of the sample, so it cannot have moved any number we have published. For a deterministic policy it is
one hundred percent of the observed variation.

### The numbers, with sample size and baseline

All figures are 200 episodes, single life, one training run per checkpoint unless stated.

Jump cap, best setting per checkpoint, measured against no cap at the second pipe:

| checkpoint | best cap | no cap | capped | gain | interval |
|---|---|---|---|---|---|
| 84×84 | 48 frames | 63.5% | 64.0% | **+0.5 pp** | [−8.9, +9.8] |
| 128×128 | 24 frames | 55.5% | 59.5% | **+4.0 pp** | [−5.6, +13.5] |

Neither interval excludes zero. Replicated across three training runs of the same recipe, the gains are
**+0.5, −1.5, 0.0** — and for scale, those same three runs *without* any cap clear the pipe at **63.5%,
51.5% and 67.5%**, a 16-point spread. **The noise between training runs is larger than every effect measured
in this block.**

Greedy action selection, across six checkpoints, reported as trajectories because that is what they are:

| checkpoint | furthest x | obstacles cleared | jump-button rate | airborne |
|---|---|---|---|---|
| 84×84, run 0 | **916** | three | 0.283 | 44.8% |
| 84×84, run 1 | 316 | none | 0.320 | 77.1% |
| 84×84, run 2 | 723 | two | 0.448 | 80.9% |
| 128×128 | 300 | none | 0.357 | 58.6% |
| 128×128, larger net | **40** | none | **0.000** | 0.0% |
| short-training control | 311 | none | 0.002 | 5.5% |

Expert reference: jump-button rate 0.152, airborne 61.1%. **Two of six trajectories clear the second pipe;
one clears the third.** The last two rows are a separate finding: on those checkpoints the greedy policy never
presses the jump button at all — one of them does not leave the starting position — which is a known failure
mode we had recorded as a property of a *different* network head. It turns out to be a property of the
individual trained checkpoint, reproducing on two of six models that share an architecture.

### Cost

About forty minutes of emulator time across sixteen policy configurations and five scripted controls, plus no
training at all — every checkpoint already existed. Two small pieces of engineering came out of it: every
saved checkpoint now records its own recipe (training steps, batch size, random seed, resolution, commit hash,
and whether the working tree was clean), and the scorer now measures how many of its episodes are actually
distinct and reports an effective sample size instead of a confident percentage.

The recipe-recording exists because of the previous block. When an earlier result failed to reproduce, we
could not tell whether it had been an unlucky random seed or a different recipe nobody wrote down, and the
file itself contained no way to find out. Five fields, five minutes, and that particular ambiguity cannot
recur.

### Downstream effect

The specific intervention is closed. The broader question is not, and this entry deliberately stops short of
the conclusion the directive invited.

The directive had said: if capping fails, then the measurement we have been using to track the policy's sense
of timing is itself suspect. We are declining that inference. Two things stand in the way. The
higher-resolution model gained nine points at the third pipe under a cap while the lower-resolution one gained
nothing there — a resolution-dependent effect in the direction the theory predicted, at an obstacle we were
not looking at. And the greedy configuration is not a negative result but an **unmeasurable** one: it is the
only thing we have built whose behaviour statistics resemble an expert's, and its evaluation is a sample of
one by construction.

That last point sets up the obvious next step, and it is a change of evaluation rather than of method. More
random seeds cannot rescue a sample of one, because the trajectory does not depend on the seed. **Many
starting positions can** — and a library of seventy-two of them, harvested from the policy's own earlier
play, already sits in the repository. That would turn "two of six trajectories" into an actual rate, and it is
the cheapest remaining way to find out whether the most expert-like thing we have built is any good.

---

## Block 55 — The level was finished, twice, from halfway; and the part that sees turned out to matter

### What changed

**Super Mario Bros. level 1-1 was completed.** The policy touched the flagpole and the game advanced to the
next level. That is the goal this project was set, and it had never happened before.

**It was not completed from the beginning of the level.** Both completions started from a saved position 39%
and 46% of the way in, and they happened on two out of five hundred and seventy-six attempts. The honest
statement is that the back half of the level is now within reach, occasionally, given a good handoff. The
front half is not.

Two other things changed, both of which overturn positions this document has previously recorded. **Sharpening
the policy's action choice makes it worse, not better** — and the sharper it gets, the more its button
statistics resemble a human expert's while its actual performance declines. And **making the part of the
network that *sees* bigger is the first change in five blocks to improve anything**, which also means the
higher-resolution input we spent 26 GB capturing is now a settled loss.

### How we got there, including the wrong turn

The previous block had ended on a genuinely exciting number: choosing the single most likely action instead of
sampling produced the deepest trajectory the project had recorded. We correctly refused to call it a result,
because a deterministic policy in a deterministic game from a fixed start produces exactly one trajectory —
a sample of one. The proposed remedy was to run it from many starting positions instead of many random seeds,
which gives genuinely independent attempts.

**That remedy was right, and it was still not enough, and the gap is the lesson of this block.** Running the
greedy policy from seventy-two starting positions produces real numbers with real intervals. It produces
nothing to compare them *to*. A rate of 40% at the second pipe means nothing without knowing what ordinary
sampling does from those same positions.

So we ran that control: identical starting positions, ordinary sampling. **Sampling beat the greedy policy at
the second pipe on all four checkpoints — by 37, 67, 43 and 43 points.** The greedy policy is not the
most-expert-like-thing-we-cannot-measure that the last entry described. It is measurably worse. The x=916
trajectory was a lucky path, and the previous entry's framing of it as "unmeasurable rather than null" was
too generous to a number we wanted to be true.

The same shape appeared again from a different direction. We swept temperature — the knob that runs from
ordinary sampling at 1.0 down toward greedy at zero — and clearance falls the whole way down, in every
checkpoint. Meanwhile the policy's button statistics converge steadily on the expert's: jump-button rate from
0.54 toward 0.28 against the expert's 0.15, jump starts from 48 per thousand frames to 33 against the
expert's 27, time airborne from 67% to 48%. **Looking more like the expert and playing better are, here,
opposite directions.** This project has repeatedly used expert-like button statistics as evidence of
progress, including in the previous entry. That proxy is now retired.

The encoder result came from an instruction to stop widening the reasoning and widen the vision. Every
network this project has trained used the same three convolutional layers, sixteen channels then thirty-two
then thirty-two, while earlier "bigger model" experiments had grown the transformer behind them. Doubling the
encoder's channels is a one-line change nobody had made.

### The numbers, with sample size and baseline

Temperature, pipe-2 clearance, 200 episodes per rung, three independently trained networks:

| network | T=1.0 | T=0.7 | T=0.5 | T=0.3 | T=0.15 |
|---|---|---|---|---|---|
| seed 0 | 63.5% | 65.5% | 51.5% | 45.5% | 54.0% |
| seed 1 | 51.5% | 41.0% | 25.0% | 11.0% | 9.5% |
| seed 2 | 67.5% | 73.5% | 71.0% | 49.0% | 48.0% |

Every rung had two hundred genuinely distinct trajectories, so none of this is the sample-of-one problem.

Encoder width, 200 episodes, cell means over the seeds in each cell:

| input | encoder | parameters | pipe 2 | pipe 3 |
|---|---|---|---|---|
| 84×84 | 16/32/32 | 172,284 | 53.2% | 20.0% |
| **84×84** | **32/64/64** | **325,964** | **66.3%** | **28.5%** |
| 128×128 | 16/32/32 | 366,844 | 48.0% | 29.5% |
| 128×128 | 32/64/64 | 715,084 | 39.2% | 12.8% |

Widening the encoder at the lower resolution is worth **+13.0 points [+6.2, +19.6]** at the second pipe and
**+8.5 [+2.6, +14.4]** at the third. Widening it at the higher resolution *costs* 9 to 17 points. Comparing
the two at matched encoder width, the higher resolution is worse by 15 to 27 points.

**The most useful part of that result is not the mean.** The two networks in the narrow-encoder cell clear the
second pipe at 65.5% and 41.0% — a 24-point disagreement between two runs of the identical recipe. The two
wide-encoder networks land at 67.5% and 65.0%, a 2.5-point spread. **The wide encoder barely improves the
good network; it rescues the bad one.** Given that this project has now lost three separate claims to
single-run flukes, a change that makes independent runs agree is worth more than a change that raises an
average.

The completion itself: two of 576 attempts, both with ordinary sampling and neither with the greedy policy,
starting from x=1264 and x=1516. We verified it rather than trusting the position counter — re-running one
attempt with level logging showed the world/stage indicator advancing from 1-1 to 1-2 and the player state
sitting in the flagpole-descent value for 660 consecutive frames. Both attempts had been scored as "stuck" by
the harness, which is the flagpole freezing horizontal position, a false-positive class this project
documented long ago.

Against the standing benchmark — a script that holds three buttons at fixed random rates — the block's best
configuration is still **15.0 points short [−23.2, −6.5]** at the second pipe. That is better than the
previous block's 19.0-point deficit and it is still a loss.

### Cost

Four network trainings at roughly 4 to 10 minutes each on the GPU, twenty-four evaluation configurations, and
about ninety minutes of emulator time. One incidental piece of engineering: the previous block's discovery
that the first episode of every session behaves slightly differently was chased to its root. The emulator
restores game memory perfectly — all 2048 bytes, verified — but the *image* it hands back differs between a
session's very first restore and every later one. Crucially, that later image is *constant*: it does not vary
with what the previous episode did. So there is no episode-by-episode contamination and no published rate can
have shifted by more than one episode in two hundred. It only matters for a deterministic policy, which is
exactly how it surfaced.

### Downstream effect

Three levers are now closed by measurement rather than argument: bounding the jump duration, sharpening the
action distribution, and raising the input resolution. The resolution one is worth stating bluntly because it
cost real disk and real time: 84×84 beats 128×128, and the higher-resolution corpus should be treated as a
sunk expense rather than an asset awaiting the right experiment.

One lever is open and barely explored. The encoder was widened once, in one step, and the step helped and
halved-then-halved-again the disagreement between training runs. Nobody has tried going wider, and nobody has
tried a fourth layer.

But the completion points somewhere else entirely, and it may matter more. From x=1264 this policy finishes
the level. Every measurement in this document — hundreds of evaluations across fifty-five blocks — is
concentrated on four obstacles inside the first 975 pixels of a 3,266-pixel level. If the back half is
already passable given a decent entry, then the thing worth mapping is **where reach actually collapses as a
function of where you start**, which is a cheap sweep with a policy we already have. The difficulty may be
early rather than late, and we have been measuring the early part hardest without ever asking whether the
late part was the problem at all.

---

## Block 56 — Two of my own claims withdrawn, and a map that shrinks the problem

### What changed

**The previous entry's headline is withdrawn.** It reported that widening the network's visual encoder made
independently trained networks agree with each other — "spread 24.5 points down to 2.5" — and that it lifted
performance by 13 points. Both numbers came from comparing two training runs against two training runs. Run
five against five and the spread is 35 down to 23.5, and the performance lift, measured against the right
unit, is not statistically distinguishable from zero.

**In exchange, something more useful:** the reason this policy cannot finish the level is now known to be a
short list of specific places rather than a gradually accumulating failure. **Starting it 650 pixels further
into the level buys it about 130 pixels of extra progress and nothing more.** It stops at the same absolute
positions no matter where it begins.

### How we got there, including the wrong turn

The advisor gated the whole block on re-testing the variance claim at five seeds, on the grounds that a range
of two numbers against a range of two numbers is not a dispersion result. That was right, and the gate caught
more than it was aimed at.

**The wrong turn is mine and it is a statistics error I had already documented and then walked into.** The
previous entry's 13-point improvement carried a confidence interval computed by pooling all the episodes from
both training runs in a cell. That treats the individual episode as the independent unit. But the two arms
being compared differ *by training seed*, so the seed is the unit — and this project's own operating notes
have said so since block 39. Pooling episodes narrows precisely the interval that was in dispute. I wrote the
caveat into that report, in the report itself, and then let the headline stand on the pooled number anyway.

Recomputed with the seed as the unit, and with an exact permutation test rather than a bootstrap — the right
test when each group has five values, since it enumerates all 252 possible arrangements — the improvement is
12.1 points with p = 0.175. At this sample size the smallest p the test could have produced is 0.008, so it
had the power to find a clean separation and did not.

What survives is weaker but real and worth keeping: every summary statistic favours the wider encoder at both
sampling temperatures, and **its worst of five training runs beats the narrower encoder's median run** — 60.0
against 54.5, and 56.0 against 47.5. That is a difference in the shape of the distribution, not just noise in
its location. It is promising and unproven, and settling it would take roughly seventeen to twenty training
runs per configuration.

Because the gate did not open, the planned follow-up — widening the encoder further — was not built. The
advisor had specified that branch in advance and named the reach study as the block's result instead. That
turned out to be the better instruction.

### The numbers, with sample size and baseline

Second-pipe clearance, 200 episodes per training run, five runs per configuration:

| encoder | at T=1.0 | at T=0.7 |
|---|---|---|
| (16,32,32) | 63.5 · 51.5 · 67.5 · 48.5 · 54.5 | 65.5 · 41.0 · 73.5 · 47.5 · 38.5 |
| (32,64,64) | 62.5 · 60.5 · 60.0 · 61.5 · 72.5 | 67.5 · 65.0 · 56.0 · 58.5 · 79.5 |

Spread 19.0 → 12.5 and 35.0 → 23.5; standard deviation 8.1 → 5.2 and 15.5 → 9.2. Differences in dispersion,
bootstrapped over seeds: [−6.5, +17.5] and [−14.5, +26.0]. Difference in means: +6.3 and +12.1 points, exact
permutation p = 0.183 and 0.175.

The reach study, run from 72 saved positions with five repeats each, on two independently trained networks —
**720 episodes, and every number below is conditional on being handed that starting position**, which makes it
incomparable to the from-the-start figures above:

| started at x | median furthest x reached (run A) | (run B) |
|---|---|---|
| 0–200 | 701 | 723 |
| 200–350 | 716 | 722 |
| 350–500 | 707 | 722 |
| 500–650 | 819 | 864 |

Three of run B's bins land on 722 *exactly*. Across start positions spanning 650 pixels the ceiling moves by
118 and 142 pixels respectively.

And the places where episodes end are the same for both networks: a large pile at 672–704, which is the face
of the third pipe at x=720; a second at 896, the face of the fourth pipe at 912; a third at 288, which is the
first Goomba; then 1216–1248 at the Koopas and 1504–1536 at a known fall. Not a continuum — five named
locations.

**These runs reached the flagpole zero times in 720 attempts.** The two completions reported in the previous
entry came from different checkpoints at a different sampling temperature. They stand at two events out of 576
attempts, with the same handoff caveat, and nothing here upgrades them.

### Cost

Five network trainings at roughly three minutes each, ten evaluation configurations, and 720 start-state
episodes — about an hour and a half in total, most of it evaluation. The follow-up experiment that was gated
off would have cost another half hour and is specified and unrun.

### Downstream effect

The withdrawal matters less than the map. For fifty-six blocks the working framing has been that the policy
runs out of competence somewhere around x=900 and the question is how to extend it. That framing is wrong in a
specific and useful way: the policy does not run out of anything. **It arrives at the third pipe in good shape
whether it has travelled 50 pixels or 650, and then it fails there.** Three places — the third pipe's face,
the fourth pipe's face, and the first Goomba — account for nearly every failure in the first thousand pixels.

That is the first time this project has had a bounded, well-posed target rather than a level. It also explains
the earlier completion cleanly: handing the policy the first 39% of the level did not give it a running start,
it simply skipped the two pipes it cannot pass.

The practical consequence is that the per-obstacle study which has sat near the bottom of the queue for weeks
— restore to a saved position just before one obstacle, and measure that obstacle in isolation — is now the
highest-value thing available, because the map says the problem lives at three addresses and we finally know
which three.

---

## Block 57 — The training objective has come apart from the goal, and two of my measurements were wrong

### What changed

**Making the network train longer, and making its visual encoder bigger, both make it play worse.** Not
"no better" — measurably worse, by 300 to 800 pixels of furthest progress.

**And the reason is the finding.** Over the same range, the training loss *halves*. The network gets steadily
better at predicting what a human expert pressed, in three independent training runs, and steadily worse at
playing the game. The quantity we have been minimising for fifty-seven blocks has stopped tracking the thing
we want, and over this range it points the wrong way.

Separately: two measurement bugs in this block were mine, and both of them, uncorrected, would have produced
a more exciting result than the truth.

### How we got there, including the wrong turns

The block had three jobs: check whether the policy was slipping into a hidden bonus area (which would mean
some of last block's map was mislabelled), scale up training length and encoder width, and measure what it
actually takes to clear the third pipe.

**The first wrong turn nearly became a headline.** Checking the bonus-area question, I found four episodes
whose game state briefly entered the "entering a pipe" value. I re-ran those four with our episode terminator
switched off — a rule that ends an episode after 300 frames without new forward progress — and they went from
stopping around x=900 to reaching 2710, 2712, and 3266. The last of those is the flagpole: **it finished the
level, starting from x=158, nearly the beginning.** For a few minutes the conclusion looked like "the wall we
have been mapping for two blocks is our own stopwatch."

It is not, and the thing that showed it was pairing. Run the same starting positions with the same random
seeds and change only the terminator, and the *median* furthest position does not move at all: 899 to 900, and
900 to 916. What moves is the tail — the top tenth improves by 80 to 130 pixels, one episode in six gets
further, and level completions go from zero to four out of 432. **My four episodes were selected precisely
because they were the ones most likely to have been cut short.** Choosing the cases that show an effect and
then measuring the effect is the oldest error there is, and I walked into it because the number was thrilling.

What survives is worth knowing: about a quarter of episodes legitimately freeze their forward progress for
more than 300 frames, so the rule was too tight, every distance we have ever reported is a mild lower bound,
and last block's "zero completions in 720 attempts" was partly this.

**The second wrong turn produced a beautifully clean answer that was entirely an artifact.** The pipe-clearing
sweep reported that *nothing* clears the third pipe — zero of 350 action sequences, from all twelve arrival
positions, every single one stopping at exactly x=724. An identical number in every cell of a grid should have
been an immediate alarm and instead I nearly wrote it up as "the policy must not arrive this way."

Two faults. My probe released every button after the jump ended, so the character decelerated into the pipe and
the grid literally could not express the actual solution, which is to hold rightward movement continuously and
add the jump for a burst. And all twelve arrival positions were being saved into the same memory slot, so each
one overwrote the last and all twelve sweeps ran from a single state. **An identical result across every cell
of a grid is evidence about the grid, not about the world.**

Fixed, the pipe is clearly solvable, and the real numbers are more interesting than the false one.

### The numbers, with sample size and baseline

Furthest position reached, 200 episodes per training run, terminator held identical to the baseline so it
cannot confound the comparison:

| configuration | per-run furthest x | vs baseline | p (exact) |
|---|---|---|---|
| baseline: 15,000 steps, encoder 32/64/64 | 1800 · 2025 · 1563 · 2019 · 2227 | — | — |
| **60,000 steps** | 1663 · 1563 · 900 | **−551 px** | 0.071 |
| **encoder 48/96/96** | 1266 · 1250 · 903 | **−787 px** | **0.018** |

Three runs against five gives 56 possible arrangements, so the smallest p this test could return is 0.036 —
stated because it means the test sees clean separations and nothing subtler. The baseline also beats both
alternatives at *every* named obstacle, at both sampling temperatures, so this is not a tail artifact.

Meanwhile the training loss, recorded inside the checkpoints themselves:

| run | at 15,000 steps | at 30,000 | at 60,000 |
|---|---|---|---|
| seed 0 | 2.593 | 1.979 | **1.292** |
| seed 1 | 2.635 | 2.011 | **1.303** |
| seed 2 | 2.600 | 1.994 | **1.302** |

The wider encoder also fits better than the narrower one at equal steps — 2.267 against 2.593 — and is the
worst arm of the three on distance.

The stall-rule audit, 216 paired episodes per run: median furthest position 899→900 and 900→916; top decile
1562→1644 and 1659→1786; 14% and 17% of episodes improved; completions 0→2 each; and 23–26% of episodes freeze
for longer than 300 frames.

The third pipe, from twelve arrival positions captured live from the policy's own play, each verified to
restore correctly: **seven of twelve are solvable**, about 9% of the 350 tested sequences work, and the jump
must be held for **12 to 20 frames**. The policy puts **0.148 of its probability** on the combinations that
work, with a confidence interval of 0.038 to 0.298 computed across arrival positions. Arrival speed averaged
2.13 pixels per frame against a running maximum of 2.5 — the first time speed has been measured at one of
these walls.

One caution attached to that: three of the failures are *indistinguishable* from a success on position, height,
grounded-ness and speed. Something else — enemy positions, sub-pixel offset, an animation counter — decides it,
so "the arrival state" is not yet properly characterised.

### Cost

About four and a quarter hours unattended: six network trainings, twenty-four evaluation configurations, and
roughly 1,900 rollouts. The most expensive planned experiment — combining longer training with the wider
encoder — was skipped, correctly, by a rule written in advance: run it only if either change helps on its own.
Neither did, which saved about ninety-five minutes. A small piece of infrastructure came out of the block: a
shared wall-clock budget and per-experiment timeouts, so that one hung emulator holding the single-instance
lock cannot consume the remaining hours.

### Downstream effect

The scaling story is closed in both directions. Encoder width is a peak, not a ladder: the previous step up
helped, this one hurt, so there is no reason to try a third. Training length has a ceiling below 60,000 steps.
Neither is where the remaining progress lives.

The loss result is larger than either. Every recent decision has been justified by some version of "it fits
the expert data better," and this block shows that over the range we are working in, fitting the expert data
better makes the policy worse at the level. That indicts the objective rather than the architecture, and it
means the current operating point may simply be the accidental best of a curve nobody has plotted. Plotting it
is nearly free — the checkpoints are banked every 250 steps and the distance measurement takes minutes.

The pipe study gives the first target in this project with a *measured* requirement: hold the jump for 12 to 20
frames from a position the policy actually reaches, which it currently aims at about one time in seven. That is
a well-posed thing to teach, which "get further in the level" never was.

And the two bugs are worth carrying forward as a rule rather than as embarrassment. Both produced cleaner,
more dramatic results than the truth, and in both cases the tell was the cleanliness itself: four hand-picked
episodes all breaking a wall, and 350 action sequences all failing at exactly the same pixel. When a result
looks too good or too tidy, the measurement is the first suspect.

---

## Block 58 — The best policy has seen less than one pass over the data, and the "mistake" was the answer

### What changed

**Three things, and each one reverses a working assumption.**

**The optimum amount of training is 1,000 steps.** That is about eight tenths of a single pass over the
training data. Every network this project has ever trained ran for at least three times that, usually fifteen
times, and last block sixty times. All of them were past their own peak.

**Pressing Down — which looked like the clearest mistake the policy makes — is the only route to finishing the
level that anything here has ever found.** The policy goes *into* the pipe. The bonus area beneath skips the
back half of the level, which is the half it cannot cross on the surface.

**And level 1-1 is now completed from the level start**, not from a saved position partway through: on 2% of
episodes, verified by the game advancing to 1-2. Two such completions were already sitting on disk from earlier
blocks, mislabelled by our own episode terminator.

### How we got there, including the wrong turns

The block began from last block's finding that training loss halves while the policy plays worse. The obvious
next question was where the turning point is, and the plan was to evaluate the checkpoints already saved every
250 steps.

**Those checkpoints did not exist.** The training loop writes a resume file and *overwrites* it at every save,
so only the final step survives. The curve required re-running the same recipe with intermediate weights
retained — same architecture, seed, data and step count, nothing trained longer.

The curve, once measured, is stark. Peak performance at 1,000 steps: 79% past the second pipe, 49% past the
third. By 45,000 steps: 31% and 19%. Meanwhile the training loss falls at *every single rung*, from 4.03 to
1.23, without one exception. The thing being optimised improves monotonically; the thing we care about peaks
almost immediately and then decays. Both training runs peaked at exactly 1,000.

I want to be careful about what that does and does not show. The decline is not perfectly smooth — 15,000
steps shows a local bump, and 45,000 is worse than 60,000 — so at 100 episodes per point there is real
measurement noise between neighbouring rungs. What is solid is that both seeds' maximum is at 1,000, that the
second-pipe rate falls from 79% to around 60% across the long end, and that the loss curve has no such bump
anywhere.

**The second reversal came from testing an idea that looked obviously right.** The owner had noticed the policy
seemed to be trying to go down a pipe at the exact place it needed to jump over one, and suspected the training
data was teaching both at once. The data agreed: the human experts do press Down in that window. So we masked
the Down button — along with Up, Start and Select — at the moment of action selection, which needs no
retraining at all.

The mask worked perfectly: the Down rate went to exactly zero and pipe-entry events to zero. Surface progress
did not change at all — a fifth of a percentage point across five seeds, in both directions, thoroughly null.

**And the level completions went from four to zero.** Checking every completion this project has ever recorded
— five of them, including one by a random button script — every single one passes through the bonus area. Five
of five episodes that ever left the surface completed the level. Zero of the 395 that stayed on it did.

So the policy is not confused. It has found the one route it can actually finish, and it is not the route we
have spent fifty-eight blocks measuring. **The thing that looked like the bug was the only thing working.**

I nearly did not find this. The surface numbers were a clean null and the mask verification was perfect; the
natural write-up was "the Down mass is harmless, hypothesis disproved, move on." What caught it was that the
mask had also zeroed the rare flagpole events, and rare events were worth looking at because the completion
claim was open.

**The third reversal was smaller and entirely mine.** Fixing the episode terminator, I picked a threshold of
1,800 frames, then measured the distribution it was supposed to clear and found the 99.9th percentile was
5,374. My number was below the thing it was chosen to exceed. It is now 6,500, above the largest value ever
observed, and the discarded guess is recorded in the code so nobody restores it.

### The numbers, with sample size and baseline

Depth against training steps, 100 episodes per point, both measures from the level start:

| steps | training loss | past pipe 2 | past pipe 3 | past pipe 4 |
|---|---|---|---|---|
| 500 | 4.033 | 70% | 42% | 25% |
| **1,000** | **3.955** | **79%** | **49%** | 29% |
| 3,000 | 3.675 | 64% | 35% | 22% |
| 15,000 | 2.571 | 73% | 45% | 20% |
| 45,000 | 1.661 | 31% | 19% | 1% |
| 60,000 | **1.228** | 64% | 31% | 25% |

The button mask, five independently trained networks, 200 episodes each, paired random seeds:

| | without mask | with mask |
|---|---|---|
| past pipe 3 | 45.5 · 34.5 · 37.0 · 39.5 · 47.5 | 49.0 · 31.0 · 39.0 · 35.0 · 49.0 |
| Down button rate | 0.055 · 0.014 · 0.012 · 0.015 · 0.027 | 0.0000 × 5 |
| pipe entries | 4 · 3 · 1 · 0 · 14 | 0 × 5 |
| **flagpole reaches** | 0 · 0 · 0 · 0 · **4** | **0 × 5** |

Completions from the level start, 200 episodes each, terminator corrected: the learned policy 4, a fixed-rate
random script 1. All 400 episodes ended in death rather than by timeout. All five completions traversed the
bonus area.

And the size of the old terminator's censoring, same network, same temperature: past pipe 3 goes from 27.6% to
45.5%, past pipe 4 from 8.3% to 20.0%, purely from correcting the rule. Every distance figure in this document
predating this block is a lower bound.

### Cost

About four hours: three training runs (only to retain intermediate weights), roughly 45 evaluation
configurations and 2,700 episodes. One planned comparison was cut — the button mask at the second sampling
temperature — because the emulator is single-instance and the training-curve measurement was the block's
central question. That was a deliberate trade and it leaves a real gap.

### Downstream effect

The training-length result is the largest retraction this project has produced. It does not invalidate the
internal comparisons of earlier blocks, since every arm within a comparison shared the same length, but it does
mean all of them were measured well past the point where the policy was at its best. Every conclusion about
resolution, encoder width, temperature and seed variance was drawn in a degraded regime.

More importantly, it moves the target. If the peak arrives before one pass through the data, then stopping
earlier is a workaround and not a fix — the objective is pulling away from the behaviour almost immediately.
That makes the loss function itself the thing to change, and it makes the next experiment cheap rather than
expensive: anything worth trying should be trained for about a thousand steps, not fifteen thousand.

The route result changes what to measure. Fifty-eight blocks have been aimed at two pipes on the surface of the
level, and the only thing that has ever reached the end went underneath them. Where the pipe entrance is, what
state it requires, and how close the policy gets to it are now more interesting questions than the walls above.

And the general lesson is one this document has recorded before in a different form: **the mask produced a
clean null on the metric we were watching and destroyed the outcome we actually wanted.** A null result on the
headline number is not the same as a null result. It was worth looking at the rare events.

---

## Block 59 — Retracting the previous entry's headline: the "route" was an artifact of counting past the end of the level

### What changed

**The previous entry's central claim is withdrawn.** It reported that the policy's route to finishing level
1-1 was to go *down a pipe* into a bonus area, and that masking the Down button removed its only win. That is
wrong, and the error is a counting mistake of mine.

**There is no bonus-area route.** Every completion is an ordinary run along the surface to the flagpole.
Checked frame by frame across 400 episodes: **not one entered any area other than the main one while still in
level 1-1.**

Two other results were re-derived and both moved: the encoder-width advantage is **null** when measured at the
correct training length, and the previous "zero completions in 720 attempts" figure is **void** — the zero was
our own episode timeout, not the policy.

### How we got there, including the wrong turn

The mistake was in how an episode was summarised. Each episode recorded the set of "areas" it visited — a game
variable that distinguishes the main level from underground rooms. Five completions all showed areas 1, 2 and
3, and no non-completing episode showed anything but area 1. The correlation was perfect, and the conclusion
looked forced: the policy goes underground, and underground is how it finishes.

**But an episode does not end when the level does.** It continues into level 1-2, which is itself underground
and has several areas. So areas 2 and 3 were being entered *after* the level was already complete, in the next
level entirely. Leaving area 1 is a *consequence* of finishing 1-1, not a cause of it. The perfect correlation
was circular by construction — I had built a causal story out of counting past the finish line.

The check that settles it is trivial once framed correctly: look at the area *while the level counter still
says 1-1*. Zero of 400 episodes ever left the main area during the level. The single completing run reached
the flagpole at x=3266 with the area variable unchanged the whole way.

The cost is not just a retracted sentence. The advisor had written this block's main experiment — map the pipe
entrance, measure what it takes, measure how close the policy is — entirely on top of my claim. That
experiment has no subject and was not run. A day of direction was spent on something that does not exist.

**What this means for the button-masking result:** the part that was measured properly still stands. Masking
Down does not change surface progress at all — two tenths of a percentage point across five training runs,
thoroughly null. The part I over-read was that masking also removed four rare flagpole-reaching episodes. Four
out of a thousand against zero out of a thousand is not a distinguishable difference, and I built a mechanism
on top of it.

### The numbers, with sample size and baseline

Per-frame audit, 200 episodes each, gated on the level counter:

| | learned policy | fixed-rate script |
|---|---|---|
| entered another area *during 1-1* | **0 / 200** | **0 / 200** |
| completed 1-1 | 4 | 1 |
| median furthest x | 723 | **828** |
| past the third pipe | 47.5% | **57.5%** |
| past the fourth pipe | 29.5% | **37.5%** |

The encoder comparison, re-run at the training length the previous block identified as optimal — 1,000 steps,
five independently trained networks per side:

| measure | wider encoder | narrower | difference | p |
|---|---|---|---|---|
| furthest x | 3266 · 1797 · 3266 · 2761 · 2595 | 3267 · 2590 · 2594 · 2589 · 2018 | +125 | **0.77** |
| past third pipe | 41 · 45 · 38 · 40 · 37 | 40 · 39 · 37 · 46 · 40 | **+0.0** | **1.00** |

And the same comparison at the old training length, but with the episode timeout corrected: the advantage is
**+327 pixels at p=0.41**, where it had been **+367 pixels at p=0.03**. The size barely moved; the
significance vanished.

**That last point is the most transferable thing here.** The old timeout ended episodes early, which caps the
"furthest distance" statistic from above and squeezes the spread between training runs. A censored maximum has
artificially small variance, so every significance test computed on it was too generous. This is not specific
to the encoder — **every distance-based significance claim this project made before the timeout was fixed
should be treated as unestablished.**

Finally, the previous "0 completions in 720 attempts" figure: at the corrected timeout, **6 of 432** attempts
complete the level, verified by the game advancing to 1-2.

### Cost

About three and a half hours: ten network trainings of twenty seconds each, roughly 35 evaluation
configurations, 1,700 episodes. The block's planned main experiment was not run, because its subject does not
exist.

One small piece of engineering: episode traces now record the world, stage and area on every frame. Two
completed levels had previously sat undetected on disk because a trace could not represent "the level ended",
and this block's retraction is the same blindness one level up — an area number is meaningless without knowing
which level it belongs to.

### Downstream effect

The walls are the target again. The reach map from three blocks ago stands unchanged, and the third and fourth
pipes are still where the level is lost. The detour through the "route" cost a block and a half.

The encoder is finished as a direction. It was the only intervention that had ever appeared to help, and it
appeared to help for two separable reasons that have both now dissolved: it was measured at a training length
15× past the optimum, and its significance came from a variance artifact of the old timeout. At the correct
training length it does nothing at all, on every measure, with five runs a side.

And there is a summary sentence this project is going to have to contain, which is worth writing down now
rather than discovering later: **after fifty-nine blocks, a three-button random script gets further through
level 1-1 than the learned policy does** — 57.5% past the third pipe against 47.5%. The policy's only
advantages are at the far frontier and in completions, and neither is statistically distinguishable from the
script. That is the honest state of the result.

---

## Block 60 — The jump rate explains neither side, and the comparison table finally exists in one place

### What changed

**The obvious explanation for why a random script outperforms the learned policy has been tested and is
wrong.** The script jumps far more often — 85% of frames against the policy's 34% — so the natural theory was
that the policy simply under-jumps and could be fixed for free by biasing it toward the jump button. It can't.

The test works in both directions, and both fail:

- **Lower the script to the policy's jump rate and it collapses.** It gets past the second pipe on 3% of
  attempts, where the policy manages 82%, and never once passes the third.
- **Raise the policy to the script's jump rate and it does not improve.** It reaches 35% past the third pipe,
  against the script's 57%, and against its own unbiased 41%.

**Each agent performs best at its own operating point, and moving either toward the other transfers nothing.**
The jump rate is not the variable that explains the difference.

The block also produced, for the first time in one place and at one setting, the three-way comparison the
eventual write-up depends on.

### How we got there, including the wrong turn

The intervention was a bias added to the jump-containing actions just before the policy samples, tuned by
measurement rather than guessed: for each target jump rate, the bias was found by bisection against live
play, because an earlier attempt at fitting such an offset offline had overshot badly.

Swept across six doses and three independently trained networks, the result is flat. Progress past the third
pipe moves from 41.2% to at best 43.0%, then declines to 30.2% as the policy is pushed toward jumping on 91%
of frames.

**The wrong turn was nearly a false positive from a single training run.** The first network showed exactly
the hoped-for effect — progress past the third pipe rising from 41% to 46.5%, and past the fourth from 23.5%
to 34.5%, an eleven-point gain that looked like a clean confirmation. The second network, at the identical
dose, went the other way: 45% down to 35%. Three runs were the right number, and one would have produced a
confident and wrong headline.

Worth recording separately: the known failure mode of this kind of bias — a policy that ends up holding the
jump button permanently — appears at the top dose and is **invisible in the clearance numbers**, which decline
gently. It shows up only in the tail of the jump-hold distribution (99th percentile 51 frames rising to 131,
maximum 113 to 292) and in the median distance collapsing at the last rung. Reporting clearance alone would
have missed the mechanism entirely.

**And a measurement bug surfaced mid-build, of exactly the kind this project keeps finding.** The scripted
comparison came back showing the high-jump script reaching 22% past the third pipe, where the previous block
had measured 57.5% for identical settings. The cause: the shared episode-timeout constant, consolidated into
one file two blocks ago, still had a *local copy* in the evaluation module. So the script arms ran under the
old censoring rule while the output file confidently recorded the new one. Fixed, and the comparison re-run
from scratch. Seven other scripts still carry local copies; none were used here.

### The numbers, with sample size and baseline

Jump bias, three networks, 200 episodes per point, all from the level start:

| jump-button rate | past pipe 2 | past pipe 3 | past pipe 4 | jump-hold 99th pct | jump-hold max |
|---|---|---|---|---|---|
| 0.49 (unbiased) | 66.7% | **41.2%** | 24.2% | 51 | 113 |
| 0.59 | 67.5% | **43.0%** | 28.0% | 56 | 129 |
| 0.69 | 63.0% | 37.2% | 25.8% | 60 | 138 |
| 0.80 | 63.0% | 38.2% | 25.7% | 76 | 166 |
| 0.85 | 60.8% | 35.0% | 24.2% | 93 | 183 |
| 0.91 | 53.2% | 30.2% | 20.5% | **131** | **292** |

The three-way comparison, 200 episodes each, same timeout, same starting point:

| obstacle | learned policy | script at the policy's rates | script at 85% jumping |
|---|---|---|---|
| second pipe | **82.0%** | **1.5%** | 79.5% |
| third pipe | 47.5% | 0.0% | **57.5%** |
| fourth pipe | 29.5% | 0.0% | **37.5%** |
| the Koopas | 19.5% | 0.0% | **27.0%** |
| the frontier fall | **6.5%** | 0.0% | 4.0% |

Against a script at its own button rates the policy is **80 points better** at the second pipe. Against the
high-jumping script it is **10 points worse** at the third. Both are true and neither can be stated alone.

One correction worth making explicit: the high-jumping script is **not** a control and should never be
described as "a coin flip". It runs the jump button at 85% against the policy's 34%, and it is the
best-performing arm selected per-obstacle from a set of candidate scripts — a deliberately hard bar, but a
maximum over attempts rather than a typical opponent.

### Cost

About three and a quarter hours: 21 evaluation configurations, 15 live calibrations, roughly 4,200 episodes.
No training — every network used was already on disk, trained for a thousand steps each.

### Downstream effect

The simplest available explanation for the project's most awkward result is now closed. That is worth
something on its own: "a random script beats the learned policy and we don't know why" was the one finding
that could not be left standing, and one candidate reason has been eliminated properly rather than assumed
away.

There is a specific replacement hypothesis, and it is narrow enough to test in a single experiment. At matched
jump rates, the remaining difference between the two is **sustained running**: the script holds the
right-direction and run buttons on *every single frame*, while the policy holds them on 73% and 83% of frames
respectively. The script never stops moving. But holding those buttons is not sufficient by itself — a script
that holds them while jumping at the policy's low rate still never passes the third pipe. So the candidate is
that the script's advantage requires the high jump rate *and* the unbroken running together, which is one arm
to check.

If that arm closes the gap, the deficit was a marginal after all — just not the marginal we were testing. If
it does not, then what a memoryless coin flip has that a trained policy lacks is the *timing* of its jumps,
which would be a genuinely strange thing to be true and would need explaining rather than reporting.

---

## Block 61 — Search solved every failure it was given; the distillation learned the answer as a habit

### What changed

**The search half of the method works, and it is not marginal.** Given sixty states where the policy had
actually failed, a search guided by the policy itself found a way past **every single one** — 14,675 working
correction sequences in total. The bottleneck this project has been circling for weeks is not "can we find
better actions".

**Training on those corrections made the policy worse**, and the diagnostic built for this block says exactly
why: it learned the corrections **specifically and strongly**, and then applied them **everywhere**.

**And the project's headline positive result has been cut by more than an order of magnitude** by a control
that should have been built much earlier. What was reported as an eighty-point advantage over a scripted
opponent is, against an honest control, about six points.

### How we got there, including the wrong turn

The reasoning behind the block was that the training data — 1.2 million frames of flawless tool-assisted play
— contains no mistakes and therefore no recoveries. A policy cloned from it leaves the demonstrated path
within seconds and then has no idea what to do. The fix is to generate the missing data: take states where the
policy failed, search for actions that get past the failure, and train on those.

Stage one worked immediately. Sixty failure states, spread across every obstacle where failures actually
occur, and all sixty solvable. Most needed only that the search start half a second before the failure rather
than at it.

A deliberate ingredient turned out to matter: the owner had observed that Mario gets stuck on top of a pipe
and doesn't know he can go backwards, and the corpus turns out to contain retreating in **0.5%** of its action
tokens — so a search that samples from the policy would essentially never discover "back up and try again". We
injected sixty-four hand-built retreat manoeuvres at every state. They produced 2,885 of the corrections, and
in an earlier check one of them solved a state that no single action could.

**The wrong turn was in how I selected which corrections to train on.** Reasoning that retreats were the
scarce and valuable label, I sorted them to the front. The result was that **98% of the training corrections
were retreats**. The policy saw "press Left" as almost the entire correction signal.

What it learned is measurable and unambiguous. Its probability on the exact retreat actions rose **eight-fold**
at the states where they were the answer — the corrections were absolutely learned. But its overall rate of
pressing Left rose from 5% of frames to **55%**. It did not learn *when* to back up. It learned to back up.
Progress past the third pipe fell from 41% to 25%.

This is the failure mode that killed three previous distillation attempts in this project, and each time it
was recorded as "distillation doesn't work". The diagnostic built for this block — measure the probability
mass on the specific solutions, not just the outcome — separates "the training didn't take" from "the training
took and taught the wrong generalisation". They need opposite responses, and for three blocks the project drew
the wrong one.

**The second finding came from a control the advisor asked for.** The project's central positive claim was
that the policy beats a random script with the same button rates by eighty percentage points. But that script
picks the jump button independently on every frame, and clearing the second pipe requires holding it for
twelve consecutive frames — probability 0.338 to the twelfth power, about two in a million. **The script's
score wasn't low because it played badly; it was low because it physically could not perform the action.**

Rebuilt so the control samples *durations* the way the policy does — matched on the action representation
rather than on per-frame rates — the policy's advantage falls from **+80 points to +6.3**, and at the third
pipe the control actually beats the policy in one of three training runs.

### The numbers, with sample size and baseline

Search, from 60 failure states across 8 obstacles: **60 solved**, 14,675 correction sequences, 2,885 of them
retreats. 49 states needed a 30-frame head start, 11 needed 60, none needed 120.

Where the failures actually are, out of 200 episodes — and split by whether Mario was stuck *at* a pipe or
stuck *on top of* one:

| obstacle | failures | at the face | **on top** |
|---|---|---|---|
| third pipe | 60 | 45 | **15** |
| first Goomba | 48 | 46 | 2 |
| fourth pipe | 38 | 19 | **19** |
| the Koopas | 30 | 29 | 1 |

**Seventeen percent of all failures are Mario stranded on top of a pipe** — and at the fourth pipe it is
exactly half. These have been scored identically to "couldn't clear the pipe" for the whole project, and they
need the opposite correction.

Round one of the distillation, three training runs, 200 episodes each:

| | past pipe 3 | Left-button rate | completions |
|---|---|---|---|
| baseline | 41.0 · 45.0 · 37.5 | 0.050 · 0.062 · 0.055 | 2 · 0 · 1 |
| after distillation | 33.5 · 13.5 · 29.5 | **0.224 · 0.563 · 0.551** | 1 · 0 · 0 |

Probability on the retreat solutions, at the states where they are the answer: **0.016 → 0.11–0.17**, an
eight-fold rise, while probability on the other solution class moved by 0.003.

The honest control, three training runs, 200 episodes each:

| | pipe 2 | pipe 3 | pipe 4 |
|---|---|---|---|
| policy | 66.7% | 41.2% | 24.2% |
| run-length script | 60.3% | 35.5% | 22.4% |
| **difference** | **+6.3** | **+5.7** | **+1.8** |

### Cost

About three and a quarter hours: 2,000 evaluation episodes, roughly 26,000 search sequences, eight short
training runs. The search data is on disk and reusable — round two needs only a different selection from it.

### Downstream effect

Three things are now known that were not this morning.

**Search is not the constraint.** Every failure state handed to it was solved. Whatever is limiting this
project, it is not the ability to find better actions.

**The constraint is generalisation, and it is measurable.** Training on corrections teaches the correction —
provably, eight-fold — but teaches it as a habit rather than as a response to a situation. That reframes the
remaining problem precisely: not "can we find the right action" but "can the policy learn *when* it applies".
The immediate test is cheap, because the mix that caused it was 98% one manoeuvre and the balanced version is
a single line of selection code over data already collected.

**And the central claim needs restating before anything is built on it.** Most of what looked like learned
skill was the action representation — the ability to commit to holding a button for twelve frames, which the
comparison script could not do at all. What remains after that is corrected for is about six percentage
points. That is a real, positive, and much smaller result, and it is better to say so now than after another
five rounds of building on the larger number.

---

## Block 62 — The balanced mix fixed the pathology and taught nothing; and the real advantage is at the late obstacles

### What changed

**Three things, and the third is the one that matters.**

The emulator was cleared of suspicion: a set of checks for whether it ever renders badly while staying alive
came back clean, so corrupted or stale observations are not the explanation for anything.

Round two of the correction-training fixed the pathology from round one and produced **no learning at all**.
Rebalanced so that one manoeuvre made up a fifth of the corrections instead of 98% of them, the policy stopped
applying that manoeuvre everywhere — and stopped acquiring it entirely. Ten independently trained networks,
and every obstacle a clean null.

**And the project's central positive claim has moved.** Re-measured at ten paired training runs instead of
three, the learned policy's advantage over a properly matched scripted opponent is **not** at the early
obstacles where it has been claimed for weeks. It is at the **late** ones: at the Koopa Troopas the policy is
ahead by 5.5 percentage points, **in all ten of ten runs**, at the smallest p-value the design can produce.

### How we got there, including the wrong turn

**The graphics check nearly stopped the block on my own definition, for the third time in five blocks.**

The natural test for a stale observation is: did the picture stay byte-identical while Mario's position
changed? That fires on 2.58% of frames, comfortably over the threshold that was supposed to halt work. I was
about to report a blocking failure.

Splitting the events by *how far* Mario moved dissolved it. **Every single one occurs on a one-pixel move, and
in every one the full-resolution native frame is identical as well.** Not one occurs on a two- or three-pixel
move. The game advances Mario's position counter by a pixel without necessarily redrawing him at a new pixel;
the downscaled observation identical across that is the game being reproduced correctly, not a fault. With the
criterion corrected to "the native frame failed to redraw across a multi-pixel move", the rate is **0.297%**,
under threshold, and consistent with ordinary hardware lag frames.

That is the same shape as two earlier mistakes in this project — a search grid that couldn't express the answer
and read as "unsolvable", and a set-union taken across a level boundary that read as a discovered route. In all
three, the check that caught it was comparing the suspicious measurement against an *independent* quantity.

**Round two's result is a genuine and interpretable null.** With the manoeuvre share reduced from 98% to 21.6%,
and the corrections also balanced to match where failures actually occur and split between "clear the pipe" and
"get off the pipe", the round-one damage disappeared: progress past the third pipe went from −15.7 points to
−1.8, and the tell-tale sideways-button rate fell from 55% back to 5%.

But the diagnostic says nothing was learned. The rate of that button **inside** the regions where corrections
were collected rose by 0.0018 (p=0.44); **outside** them it fell by 0.0018 (p=0.56). Neither moves.

That required extending the diagnostic. It had two outcomes — the correction is either state-specific or a
global habit — and this is a third: **it was not acquired at all.** The two-way version would have labelled
this a "global habit" simply because it wasn't state-specific, and pointed at the wrong fix.

Taken with round one, the picture is sharp: at 98% of the corrections being one manoeuvre, the network learns
it and applies it everywhere; at 21.6% it does not learn it. Both sat at 5–7% of the total training mixture.
**The variable to sweep next is how much of the training data the corrections are, not what they contain.**

### The numbers, with sample size and baseline

Graphics: **53 render faults in 17,869 moving frames (0.297%)**; frame determinism 4 of 4 episodes
byte-identical; emulator log empty. Stale events by pixel-distance moved: 408 at one pixel, 26 at two, 27 at
three, out of 7,633 / 7,348 / 2,888 moving frames.

Round two against baseline, ten paired training runs, 200 episodes each:

| obstacle | round 2 | baseline | difference | runs improved | p |
|---|---|---|---|---|---|
| second pipe | 66.7% | 66.0% | +0.8 | 6/10 | 0.76 |
| third pipe | 37.5% | 39.3% | −1.8 | 2/10 | 0.16 |
| fourth pipe | 23.8% | 23.6% | +0.2 | 5/10 | 0.87 |
| Koopas | 18.9% | 20.4% | −1.4 | 2/10 | 0.13 |

The smallest p this design can return is 0.002, so these are real nulls rather than a lack of power.

The scripted control, ten paired runs — and this is the finding:

| obstacle | policy | matched script | difference | runs ahead | p |
|---|---|---|---|---|---|
| second pipe | 66.0% | 63.7% | +2.3 | 7/10 | 0.16 |
| third pipe | 39.3% | 36.8% | +2.5 | 8/10 | 0.18 |
| fourth pipe | 23.6% | 24.6% | −1.0 | 4/10 | 0.39 |
| **Koopa Troopas** | **20.4%** | **14.8%** | **+5.5** | **10/10** | **0.002** |
| **the frontier fall** | **3.9%** | **2.0%** | **+1.8** | 9/10 | **0.004** |

The +6.3 points at the second pipe reported last block, from three runs, becomes **+2.3 points at ten and is
not establishable**. What is establishable is the late-obstacle advantage.

### Cost

About three hours: 15 short trainings, 30 evaluation configurations, roughly 5,800 episodes. No new search —
round two reused the 14,675 corrections already collected, changing only which of them were selected.

### Downstream effect

The claim to carry forward is narrower and better founded than the one it replaces: **the action
representation buys the early obstacles, and learning buys the late ones.** It survives ten paired runs at the
smallest attainable p-value, and it has a mechanism — the Koopas are moving enemies, which is precisely where
conditioning on what is on screen should pay and where a fixed distribution over button-durations cannot. The
second-pipe number has now been revised twice and should stop being quoted.

The correction-training loop is not dead, but its failure mode is now located. Every stage works: the search
solves every failure state it is given, the corrections are learnable, and a balanced mix removes the
over-application. What is missing is a training share at which they are acquired *and* stay conditional, and
that is a one-dimensional sweep over data already on disk.

And the methodological note is worth keeping, because it has now paid for itself three times: when a check
fires alarmingly, the first hypothesis should be that the check is measuring its own definition. Comparing
against an independent quantity — the native frame here, the game's level counter before that, an action space
that could express the answer before that — has caught it every time.

---

## Block 63 — The network can see the difference; its output layer cannot read it

### What changed

**Two findings, and they fit together.**

**The rule the last two blocks were trying to teach does not exist.** Both rounds of correction-training were
attempting to make the policy back up *when backing up is required*. Measured across all sixty searched failure
states: backing up is **required at none of them** and **useful at all of them**. There is no state where it is
the only way through, and no state where it fails. So neither round failed to learn a conditional rule — there
was no conditional rule there to learn.

**And the one distinction that does matter is visible to the network but unreadable by its output layer.** Being
stuck *on top* of a pipe and stuck *at the face* of one need opposite corrections, and account for 17% of all
failures — half of the fourth pipe's. A linear read-out of the network's internal features cannot tell them
apart (AUC 0.651, p = 0.17). A small non-linear read-out can (**AUC 0.743, p = 0.010**). The policy's action
layer is a single linear map on exactly those features. **The information is sitting in front of the output
layer in a form the output layer cannot use.**

### How we got there, including the wrong turn

The plan was to check whether the network's internal representation can distinguish "a retreat is needed here"
from "it isn't", on the grounds that if it cannot, no amount of data would produce the behaviour and the line
should be closed with a precise negative.

**The probe could not be built, and finding out why was the useful part.** Constructing it needs states of both
kinds. Sorting the sixty searched states by what fraction of their working solutions involved backing up:
**two** are retreat-dominated, **five** are dominated by ordinary actions, and **fifty-three** sit in between.
The minimum fraction at any state is 0.047 and the maximum is 0.952 — backing up always helps somewhat and is
never the only option. Two positive examples cannot support a probe.

So the probes were re-pointed at distinctions that do have unambiguous ground truth. The network's features turn
out to carry an enormous amount: **which obstacle Mario is at is decodable essentially perfectly** — six of six
obstacles at AUC 0.89 to 1.00, with a permutation p of 0.0000 — and **his horizontal position** is recoverable
at R² = 0.71 out-of-fold. Whatever is limiting this project, it is not that the network cannot see where it is.

The unexpected result was the on-top-versus-at-face probe, included because those two situations need opposite
responses. Linearly it fails. Because that was the *one* case where failure would have mattered, I ran the same
protocol — same grouped cross-validation over states, same permutation null — with a small non-linear model
instead, and it succeeds. The gap between the two is the finding, and the architecture makes it concrete: the
policy chooses its action through one linear layer applied to those features.

**One honest weakness:** the linear probe has only eleven positive examples, so "not linearly separable" is the
weaker of the two claims. The non-linear result and the gap carry the interpretation.

### The numbers, with sample size and baseline

240 observations from 60 states, cross-validated with folds over *states*, every result accompanied by a
label-permutation null computed at the state level (with 64 features over 60 states, a linear probe will
separate noise without one):

| what was probed | ground truth | score | p |
|---|---|---|---|
| which obstacle (6 separate tests) | 9–11 states each | AUC 0.892 – 1.000 | 0.0000 |
| horizontal position | regression | R² 0.712 | — |
| **on-top vs at-face, linear** | 11 vs 49 | **AUC 0.651** | **0.17** |
| **on-top vs at-face, non-linear** | 11 vs 49 | **AUC 0.743** | **0.010** |

Retreat-share distribution over the 60 states: minimum 0.047, median 0.204, maximum 0.952; two states above
0.50, five below 0.15.

### Cost

Under an hour, and no new gameplay data — the probe reuses states already searched, needing only one forward
pass each plus logistic regressions.

The planned follow-up sweep was **not** run. Its gate was "only if the probe separates", and while the probe
separates on obstacle identity, the specific label the sweep would have varied does not exist. Raising the
number of training examples of a distinction that has no ground truth would not have produced conditional
behaviour, and the reasoning is recorded so the decision can be reversed cheaply.

### Downstream effect

The correction-training line now has a complete causal account rather than two unexplained failures: the search
finds solutions for every failure state, the corrections are learnable, a balanced mix stops them being
over-applied — and the specific rule being taught was never a well-defined function of the state. That is a
better negative than "distillation did not work", and it is a different one from the two hypotheses on the
table.

More usefully, the block produced the first complete chain from a measured failure to a named architectural
cause. Being stranded on top of a pipe is 17% of failures and half the fourth pipe's; the corrections for it
exist and were all found by search; the information needed to recognise it is present in the network's
features; and the single linear layer that turns features into actions cannot read it. **That points at a
smaller change than any of the scaling experiments this project has run — one hidden layer between the trunk
and the action head — and it points at it for a measured reason rather than a hopeful one.**

The headline positive result was also written up and frozen this block, so it stops being revised: **run-length
action tokens buy the early, static obstacles; learning buys the late, moving ones** — the policy beats a
representation-matched script at the Koopa Troopas by 5.5 points in all ten of ten training seeds, which
survives correction for all six obstacles measured.

---

## Block 64 — The output layer was not the problem, and the previous entry's reason was a power artifact

### What changed

**The previous entry's central mechanism is withdrawn.** It reported that the network's features contain the
distinction between being stuck on top of a pipe and stuck at its face, but that the single linear output layer
could not read it — a linear read-out scored 0.651 (not significant) where a small non-linear one scored 0.743
(significant).

**That was a sample-size artifact.** The probe used only eleven positive examples, because it looked only at the
sixty failure states that had been searched. Re-run across all two hundred recorded failures — thirty-eight
positives — the **linear** read-out scores **0.859**, overwhelmingly significant. The non-linear one still wins,
by a real but small margin (+0.056, interval [+0.012, +0.117]). The output layer reads the distinction perfectly
well.

**And the intervention it motivated does nothing.** A non-linear output layer, at two widths, ten
independently-trained networks each, moves neither the pre-specified target nor any other measured outcome.

### How we got there, including the wrong turn

The advisor's instruction had two parts, and the order mattered: fix the probe's statistical power *first*,
because the whole block rested on it. That was the correct call and it inverted the premise before the expensive
part ran.

Two things were wrong with the original probe, and they are the same mistake in different clothes. The first is
that **a difference in significance is not a difference in effect**: 0.743 being significant and 0.651 not
being significant does not establish that 0.743 is larger. The two were never compared. The second is that
eleven positive examples cannot resolve a moderate difference, and I had not checked what size of effect the
probe could actually see.

Bootstrapping the difference between the two read-outs over states — the test that had never been run — gives
+0.056 with an interval that does excludes zero. So the non-linear read-out genuinely decodes slightly better.
But "slightly better than 0.859" is a very different claim from "the linear layer cannot read it", and the
architectural story built on the second version is gone.

The behavioural experiment ran anyway, because the arms were cheap and the non-linear read-out is genuinely
(if marginally) better. Its primary outcome was fixed in writing before the runs, precisely so the result could
not be rescued by picking a different obstacle afterwards. It came back flat.

**This is the second time in three blocks that a result I liked rested on a sample size I had not checked
against the effect I was claiming.** The other was three training runs where the smallest achievable p-value was
0.25. Both were caught by the advisor rather than by me. The discipline of stating the attainable floor beside
every p-value had been applied to the gameplay arms and not inherited by the probes.

### The numbers, with sample size and baseline

The probe, 600 feature vectors from 200 failure states, cross-validated with folds over states and a
permutation null computed at state level:

| read-out | positives | AUC | p |
|---|---|---|---|
| linear | 38 | **0.859** | 0.0000 |
| non-linear (32 hidden units) | 38 | 0.915 | 0.0000 |
| difference | — | **+0.056** | interval [+0.012, +0.117] |

Against the previous entry's eleven positives: linear 0.651 (p = 0.17), non-linear 0.743 (p = 0.010),
difference never tested.

The behavioural arms, ten paired training runs each, 200 episodes each, primary outcome declared in advance:

| output layer | parameters | on-top failures at pipe 4 | past pipe 3 | past pipe 2 |
|---|---|---|---|---|
| single linear | 325,964 | 7.5 per 200 | 39.3% | 66.0% |
| 64→128→300 | 353,484 | 7.7 (p = 0.92) | 38.5% | 64.2% |
| 64→256→300 | 400,204 | 7.4 (p = 1.00) | 38.8% | 65.1% |

Every obstacle null, at both widths, with a design whose smallest attainable p-value is 0.002.

### Cost

About three and a half hours: twenty short trainings, twenty-seven evaluation configurations, 5,400 episodes,
plus the probe.

### Downstream effect

The negative is more useful than it looks, because of what it closes. The network's internal features carry
which obstacle it is at almost perfectly and its position to within about 200 pixels; the output layer reads
the distinction that matters at 0.859; and adding capacity exactly where the probing pointed changes nothing
across ten paired runs. **No future failure in this project can be blamed on the observation, the trunk, or the
output layer. All three have now been measured, and all three are adequate.**

That leaves two candidates standing, and both already have direct evidence against them rather than merely
suspicion. The training loss falls monotonically while play gets worse beyond about a thousand steps. And the
training data is 1.22 million frames of flawless play containing no deaths and no recoveries at all.

Of those two, **the objective is the one thing that has never been varied.** Every network in sixty-four blocks
has been trained by plain next-token prediction on the demonstration data. The data has been re-captured,
re-balanced, augmented with search corrections, and re-weighted; the architecture has been widened, deepened,
lengthened and given a non-linear read-out; the sampling rule has been sharpened, capped, biased and matched.
The loss function has not been touched.

The positive result is unaffected and stands as written: run-length action tokens buy the early, static
obstacles, and learning buys the late, moving ones — a 5.5-point advantage over a representation-matched script
at the Koopa Troopas, in all ten of ten training runs.

---

## Block sixty-five — training only on the level we actually test

### What changed

We tried the most obvious idea left, and it failed in a way that was worth more than success would have been.

Every network in this project is trained on demonstrations of all thirty-two levels of the game, but it is
only ever *tested* on the first one. That is an odd arrangement, and the arithmetic behind it is worse than it
looks: at the training length that works best, the network sees the first level less than once through from
beginning to end. So the question almost asks itself — what if we trained it only on the level it has to play?

We built two versions. One drew every training example from level 1-1 and nothing else. The other drew half
from 1-1 and half from everywhere else. Both were trained ten separate times so that the comparison would be
between training procedures rather than between two lucky runs.

Both were worse. Not catastrophically, and not everywhere — but worse, and worse in a place that turns out to
matter.

### How we got there, including the wrong turns

The first thing that happened was that the premise shrank three times before a single network was trained.

The brief said level 1-1 was 8.9% of the usable training signal, and that training only on it would multiply
its exposure elevenfold. Checking that against the code rather than accepting it changed all three numbers.

The training data is not stored as individual frames. It is stored as *runs* — "hold right for nine frames",
"press jump for twenty-two" — which is a compression, and it does not compress every part of the recording
equally. A long stretch of a loading screen, where nothing is pressed at all, becomes one training example.
A busy passage of play becomes hundreds. So a share measured in frames is not the share the training procedure
actually draws from. Measured properly, level 1-1 is **3.0%** of the training examples, not 8.9%.

Then a second discrepancy: the brief's figures covered the entire recorded corpus, but training only reads a
subset of it — the twenty runs reserved for the purpose. On that subset the share is smaller still.

The two corrections compounded. Restricting to level 1-1 would multiply its exposure not elevenfold but about
**thirty-four fold**, and the resulting training set would be **2,323 examples** rather than the roughly
seven thousand assumed — smaller than any of the twelve largest levels in the data. That is a very small
number of examples for a network with three hundred thousand parameters.

We wrote that down in the training script, before running anything, as a reason to expect the experiment to
fail by over-fitting. It is easy to notice a weakened premise after seeing a disappointing result. Noticing it
first is the only version that counts.

### The numbers

The comparison is against the existing full-corpus networks, ten of them, on two hundred attempts each. Before
trusting any of it, we re-scored the old networks through the new code and got the previous block's figures
back exactly — so the differences below are differences in behaviour, not in bookkeeping.

The early obstacles were unaffected, or very slightly better. The late ones got worse, in both versions, by
about the same amount:

| obstacle | full corpus | 1-1 only | half-and-half |
|---|---|---|---|
| the first Goomba | 66.0% | 68.6% | 67.0% |
| the second pipe | 66.0% | 68.4% | 67.0% |
| the third pipe | 39.3% | 41.1% | 36.6% |
| **the fourth pipe** | **23.6%** | **20.4%** | **19.5%** |
| the Koopa Troopas | 20.4% | 18.6% | 17.8% |

At the fourth pipe, nine of the ten restricted runs were worse than their paired baseline, in both versions
independently. None of these differences survives the statistical correction for having looked at seven
obstacles across two versions, and we did not name the fourth pipe in advance as the place to look, so no
individual figure here is being claimed. What is being claimed is that two differently-built versions failed
in the same place by the same amount — and that a separate experiment then explained why.

That separate experiment is the one that matters. We saved snapshots at five points during training and tested
each, which shows how behaviour tracks the training objective as learning proceeds:

| training steps | 1-1 only: error / fourth pipe | full corpus: error / fourth pipe |
|---|---|---|
| 500 | 3.93 / 17% | 4.03 / 23% |
| 1,000 | 3.11 / 18% | 3.96 / 28% |
| 2,000 | 1.98 / 11% | 3.81 / 21% |
| 5,000 | 0.98 / 5% | 3.46 / 32% |
| 15,000 | **0.70 / 0%** | 2.57 / 20% |
| 60,000 | — | 1.23 / 25% |

Read the bottom rows. Trained only on the level it is tested on, the network drives its training error down to
**0.70** — better than the full-corpus network ever achieves in sixty thousand steps — and reaches the fourth
pipe in **none of a hundred attempts**. The full-corpus network, at nearly double that error, gets there a
quarter of the time.

The prediction under test was that concentrating the data would move the best training length *later*. It did
not move at all: the best point is a thousand steps in both cases. What moved was the collapse, and it moved
**earlier** — arriving after five thousand steps instead of forty-five thousand. Thirty-four times the
exposure bought a ninefold acceleration of the failure.

Comparing the two at equal training error rather than at equal training length, the restricted networks are
worse at the fourth pipe at all ten points of comparison.

### Cost

About two and three-quarter hours: twenty-two short trainings, thirty evaluation configurations, five thousand
attempts at the level.

### Downstream effect

The 91% of the training data that is never tested on is not dead weight. It is doing something, and what it is
doing is preventing the network from memorising the small amount of data that is directly relevant. Removing
it does not focus the network; it lets it collapse.

There is a detail in *where* the damage landed that is worth more than the headline. This project's one
positive result is a claim about a boundary: that the compressed action representation is what gets the
network past the early, stationary obstacles, and that learning is what gets it past the late, moving ones.
Restricting the data left the early obstacles alone and damaged the late ones — the learning half, exactly.
That boundary was drawn in a previous block, before this experiment existed. **It is the first time a negative
result in this project has landed on the side of a line that was drawn in advance**, which is a modest form of
corroboration but a real one.

We also found a small error in our own accounting, in the direction that costs nothing. The first and second
pipes have identical clearance rates in every version ever run, including the scripted controls — no attempt
in the history of this project has ever failed in the stretch between them. They have been counted as two
independent measurements when correcting for multiple comparisons, which has made every such correction
slightly too strict. That made us under-claim rather than over-claim, but it should be six obstacles and not
seven.

What this leaves is a short list. The visual input, the network's internal representation, and its output
layer were each measured and found adequate in the two previous blocks. The training data has now been
measured too, and the answer is the opposite of the one expected: it is not too narrow, and narrowing it
further is actively harmful.

That leaves the objective — plain next-token prediction — which is the one component that has never been
changed in sixty-five blocks of work. Everything else has been varied: the data has been re-recorded,
re-balanced, augmented with search, and now re-weighted by level; the network has been widened, deepened, and
given a non-linear output; the sampling has been capped, sharpened and biased. The loss function has not been
touched once.

The positive result is unaffected and stands as written: run-length action tokens buy the early, static
obstacles, and learning buys the late, moving ones — a 5.5-point advantage over a representation-matched
script at the Koopa Troopas, in all ten of ten training runs.

---

## Block sixty-six — changing what the network is asked to minimise

### What changed

We changed the training objective, which is the one component of this system that had never been varied in
sixty-five blocks of work. It made no difference, and in most measurements it made things slightly worse.

Every network here learns by being shown a frame and asked to predict the button press a human expert made
next, scored by how much probability it puts on the correct answer. That scoring rule pushes the network to
become as certain as possible about the demonstrated action. The suspicion was that this is the problem: the
data is a single flawless run, and being trained to commit absolutely to it might produce a policy that can
only follow that exact path and falls apart the moment it drifts.

There is a standard one-line fix for exactly that — label smoothing, which forbids the network from putting
more than a set fraction of its confidence on any single answer. We swept three strengths, trained each ten
times, and tested every version on two hundred attempts.

It was worse at every strength and at every obstacle. Eighteen comparisons, all eighteen negative.

We also tried the opposite tack: keep the objective, but show the network more of the frames near where its
own attempts actually fail. That was flat at gentle strengths and clearly harmful at a strong one.

### How we got there, including the wrong turns

Two things nearly went wrong, and both were caught before they contaminated the result.

The first was a subtle one about how training examples are drawn. Our baseline networks were trained by
shuffling the whole dataset and walking through it — every example seen once per pass. Our first draft of the
new code drew examples independently at random instead, which sounds equivalent and is not: over sixty-four
thousand draws, independent sampling shows the network about forty-four thousand distinct examples where
shuffling shows sixty-four thousand. Had we left it, every label-smoothed network would have differed from its
baseline in *two* ways — the objective and the sampling — and the block's headline number would have meant
nothing. We rebuilt the reweighting as repetition inside the shuffled list, which collapses exactly onto the
baseline scheme when the reweighting is switched off, and verified that it does.

The second was about whether the second experiment could produce an answer at all. The instruction was to find
the places where the policy actually fails, from a histogram of two thousand failed attempts, and show the
network more training data from those places. The histogram was clean — four clear clusters at the first
enemy, the pipe sequence, the Koopa Troopas and the far frontier, together covering 98% of all failures. But
the positions only mean anything in the level being tested, and the training data for that level is tiny. The
frames those windows could actually reach came to **295 out of 77,916** — under four tenths of one percent.
At the strongest gentle setting we were asked for, that is about five hundred extra examples out of sixty-four
thousand.

This project has previously shipped an experiment that could not express its own answer, so we computed that
number *before* running rather than discovering it afterwards, and added one deliberately strong setting on
top of the gentle ones — not replacing them — so that a flat result could be told apart from a manipulation
too small to see.

That decision is what made the second experiment informative.

### The numbers

Success is measured as the share of two hundred attempts that get past the fourth pipe, chosen in advance as
the primary measure. Ten independently trained networks per setting.

Label smoothing:

| strength | past the fourth pipe | change | runs better than baseline |
|---|---|---|---|
| none (baseline) | 23.6% | — | — |
| 0.05 | 22.1% | −1.4 | 3 of 10 |
| 0.10 | 20.8% | −2.8 | 2 of 10 |
| 0.20 | 21.4% | −2.2 | 1 of 10 |

None of these differences is individually significant once corrected for having looked at six obstacles across
three settings. What carries the result is that all eighteen comparisons point the same way, and not one
points the other way.

The prediction being tested was specific: if over-commitment is the problem, the best training length should
move *later*, because the network should be able to keep learning without collapsing. It did not move. The
gentlest setting peaks at a thousand steps — exactly where the unmodified objective peaks — and the strongest
setting's apparent shift is a two-attempt difference out of a hundred on a single training run, which is
noise. Trained far past the peak, the smoothed networks collapse just as hard: at fifteen thousand steps they
clear the fourth pipe 12–13% of the time against the baseline's 20%.

The reweighting experiment:

| strength | past the fourth pipe | change | runs better than baseline |
|---|---|---|---|
| none | 23.6% | — | — |
| 1.5× | 22.2% | −1.3 | 4 of 10 |
| 2.0× | 23.2% | −0.3 | 4 of 10 |
| 3.0× | 24.6% | +1.0 | 4 of 10 |
| **8.0×** | **19.5%** | **−4.0** | **0 of 10** |

The gentle settings do nothing, which is what the four-tenths-of-a-percent calculation predicted. The strong
setting does something, and what it does is harm — every single one of the ten runs worse than its paired
baseline, at the smallest probability this experimental design can produce.

**Showing the network more of the demonstration data from precisely the places where it fails makes it worse.**

### Cost

About seven hours: eighty-five trainings, ninety evaluation configurations, fifteen thousand attempts.

### Downstream effect

The over-commitment explanation is now retired. It made a specific, falsifiable prediction — that the best
training length would move later — and that prediction has now failed three separate times, against augmented
data, against a concentrated training set, and now against an objective designed specifically to prevent
over-commitment. Three failures of the same prediction is enough.

That closes the last open component. The visual input, the network's internal representation, its output
layer, the composition of the training data, and now the training objective have each been measured, and none
of them is what limits this system.

We also found that our own success measure had been double-counting. The first enemy, the first pipe and the
second pipe have been reported as three separate obstacles, but no attempt in the history of this project has
ever ended between them — of ninety networks measured this block, eighty-nine have byte-identical counts for
the first and third of those, and the exception differs by one attempt. There are four genuinely independent
places to fail, not seven. This made every statistical correction we have applied *too strict* rather than too
lenient, so nothing previously claimed is weakened — but the figure should be four.

What is left is a clean statement rather than a to-do list. The training data is a single perfect run through
the game containing no mistakes and no recoveries, and supervised imitation of it stops improving after about
a thousand steps regardless of the network, the resolution, the output layer, the data mixture, or the loss
function. That is a real limit, it is now measured from five independent directions, and it sits beside a
result that does hold.

The positive result is unaffected and stands as written: run-length action tokens buy the early, static
obstacles, and learning buys the late, moving ones — a 5.5-point advantage over a representation-matched
script at the Koopa Troopas, in all ten of ten training runs.
