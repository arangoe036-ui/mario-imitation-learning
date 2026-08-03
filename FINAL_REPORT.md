# Project log — Mario from a perfect teacher

Beat Super Mario Bros using only supervised learning. No policy gradient, no value
bootstrapping; every model update is next-token prediction. Search is permitted, because
search is not a gradient method — it explores, and its results are distilled back by
supervised learning.

This file is the narrative record: what changed, what was found, and **how we got there**,
including the wrong turns. `FINDINGS.md` is the technical companion with the full tables.

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
