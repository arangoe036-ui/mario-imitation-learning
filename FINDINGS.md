# Findings

Measured results worth keeping, separate from the how-to in `README.md`. Everything here
came out of a run that was made and can be reproduced from the scripts named.

---

## Perceptual aliasing: two states render identical pixels with different RAM

Building the savestate index hashes each start point twice — once over the 2 KB of RAM and
once over the 84×84 grayscale observation the policy actually receives.

```
[build] distinct RAM hashes 532/532, distinct FRAME hashes 530/532
```

**Two of 532 states are pixel-identical while their game variables differ.** The policy
cannot tell them apart, because pixels are its entire input. Whatever distinguishes those
states — an off-screen enemy's position, a timer, a subpixel velocity — is invisible to it,
and if the correct action differs between them no amount of training can recover it.

This is not a bug to fix. It is inherent to a pixel-only policy, and it puts a ceiling on
achievable accuracy that is strictly below 100%. It also means the two hashes are not
redundant: RAM is the stronger *identity* check (532/532 distinct), the frame hash is the
only *rendering* check, and each catches drift the other misses. Both are asserted on
rebuild.

At 2/532 the measured rate is ~0.4% of sampled states, which is small — but the sample is
of deliberately-spread start points, not of the frames a policy actually visits, and
nothing here establishes that the visited distribution aliases at the same rate.

Reproduce: `scripts/build_state_index.py build`

---

## 61.1% of expert frames are airborne

Of 67,117 frames in the warpless run, **41,031 (61.1%)** have Mario's y-position moving
within a ±4-frame window. He is off the ground for most of the run.

That single number explains several things that were otherwise puzzling:

- **The start-point filter rejects 93.2% of uniform samples**, and being airborne is by far
  the largest reason — bigger than all the transition, cutscene and level-load reasons put
  together. Only 4,536 of 67,117 frames are usable rollout starts.
- **A "grounded" filter that requires stability in both directions excludes A-onsets by
  construction.** At the frame the expert first presses A, y is about to rise, so the
  forward half of the test always fails. On a held-out run that left **24** A-onsets in
  68,509 frames. The backward-only form — was Mario on the ground over the preceding 4
  frames — leaves **216**, and is the right question for "can he jump from here?".
- It is why jump timing dominates everything. A policy that gets the ground contact wrong
  is wrong for most of the run, not occasionally.

Reproduce: `scripts/build_state_index.py build` (rejection breakdown), `scripts/oracle_policy.py`
(the 24-vs-216 comparison).

---

## Three start-point defects, each of which faked a live result

Found while writing the first tests that launch a real emulator. Each had been silently
distorting live numbers.

1. **Level starts were not level starts.** Taking the first frame that passed the full
   grounded filter put the 1-1 start at **x=2616** — past both pipes. "Cleared pipe 1" was
   true before the episode began and every arm scored 100%.
2. **Seven W-1 starts were the previous world's castle walk.** The world counter increments
   during the end-of-W-4 walk, so the first frame *labelled* 2-1 was Mario at x=2430
   finishing 1-4.
3. **The 1-1 start preceded control handover.** Frame 42 reads `pregame=1`,
   `player_state=0x08`, `x=40` — but it is boot-time RAM transient with the title screen
   still up. 60 frames of Right+B from there leaves x at **0**. Real handover is frame 196.

The general lesson: RAM flags say Mario is playable well before inputs do anything. The
expert's own trace dates the handover exactly — control begins on the first frame where x
actually *increases* (not merely changes; at frame 42 it changes, from 40 to 0).

---

## The 0.0% A-onset recall was a double normalization, not calibration drift

`FrameStackDataset` already returns float32 in [0, 1]. `stage3_train.py` divided by 255
**again**, so the network received a near-black image. Same weights, same 2,000 val frames,
input scaled two ways:

| input | p(A) mean | p(A) std | range |
| --- | --- | --- | --- |
| as given (correct) | 0.1604 | 0.14671 | [0.0030, 0.7028] |
| divided by 255 again | 0.0071 | **0.00001** | [0.0071, 0.0071] |

The model emits a **constant** — every frame gets the same 0.0071, below any calibrated
threshold, so recall is exactly 0.0%. Recalibration cannot fix it: there is no threshold
that recovers information from a constant.

The tell was the contradiction that raised the question in the first place. Live play
reported 53% pipe-1 clearance from the same checkpoint, because `play_episode` normalises
correctly and never went through the broken path. **A metric that disagrees with a
behavioural measurement is usually the metric.**

Recalibrating on the corrected input restores round 1 to 48.9% A-onset recall (from 1.6%
at its stored threshold). The plausible hypothesis — that self-training pushes A
probabilities down and a fixed threshold stops firing — was tested and rejected: the
probabilities had not drifted, they had collapsed.

Two consequences: round 1's **checkpoint** is unusable (it was fine-tuned on near-black
frames), while its **self-data is valid** (rollouts always used the correct path), so
later rounds keep the data and restart the weights from the Stage 2 checkpoint.
Recalibration after every round is adopted anyway, as good practice.

---

## Reissued: all A-onset recall under one calibration method

The frozen table mixed two calibration methods and was not comparable. Every number below
is re-measured identically: thresholds calibrated on a **random** TRAIN subset against the
expert's own press rates, recall on a **contiguous** VAL slice, observations passed through
as the dataset yields them. Categorical heads are marginalised to per-button probabilities
(P(button) = total softmax mass of tokens whose byte sets that bit), which is the only way
to put a 25-way head and an 8-Bernoulli head on one axis.

| model | head | A-onset recall | threshold | realized A rate | exact match |
| --- | --- | --- | --- | --- | --- |
| blind (control) | categorical | **0.0%** | 0.16 | 0.000 | 0.5% |
| categorical (small, lr 3e-4) | categorical | 21.9% | 0.29 | 0.118 | 67.5% |
| categorical (tiny, lr 1e-3) | categorical | 24.9% | 0.30 | 0.119 | 69.1% |
| bernoulli only (arm A) | bernoulli | 29.7% | 0.23 | 0.134 | 66.3% |
| bernoulli + onset reweight (arm B) | bernoulli | **50.0%** | 0.33 | 0.158 | 64.5% |

The progression is monotonic and the blind control lands at exactly 0.0%, which is the
sanity check the measurement needed: a model that cannot see the screen recovers no onsets
at a threshold calibrated to fire 15.25% of the time.

Note that **exact match moves the opposite way to onset recall** — 69.1% for the best
categorical model against 64.5% for arm B. Fitting the frames that matter costs accuracy
on the ~85% of frames where the answer is "keep holding what you were holding".

### Arm A self-imitation rounds, same method

| round | expert:self | A-onset recall | pipe 1 cleared (n=200) |
| --- | --- | --- | --- |
| round1_contaminated_for_reference | — | 48.9% | 53.3% (n=60) |
| stage2_armB_baseline | — | 50.0% | 59.5% (n=200) |
| round2_ratio3to1 | 3:1 | 47.3% | 92.0% (n=200) |
| round3_ratio3to1 | 3:1 | 44.1% | 96.5% (n=200) |
| round2_ratio1to1 | 1:1 | 38.8% | 96.5% (n=200) |
| round3_ratio1to1 | 1:1 | 39.6% | 99.0% (n=200) |

**Recall falls as play improves.** Arm B starts at 50.0%; by round 3 at 1:1 it is 39.6%
while pipe-1 clearance has gone from 59.5% to 99.0%. Self-imitation trades expert-likeness
for task performance, and the two metrics genuinely disagree about which model is better.
Whether that is progress depends on which one the project is actually optimising.

---

## Onset reweighting transfers to live play — at n=200, not at n=20

Arm B (per-button onset reweighting, 10x) against arm A (Bernoulli only), per-button
sampling, 200 seeds per arm per start:

| start | arm A pipe 1 | arm B pipe 1 | difference (Newcombe 95%) |
| --- | --- | --- | --- |
| 1-1 | 29.5% [23.6, 36.2] | **59.5% [52.6, 66.1]** | **+30.0 pp [+20.4, +38.8]** |

The interval excludes zero, so arm B's offline advantage (A-onset recall 45.5% vs 31.5%)
**does** transfer to live play. From 2-1 there is no difference in median x (530 vs 531);
pipe 1 is a 1-1 landmark and is not scored elsewhere.

At n=20 the same comparison read 45% vs 40% and looked like a null result. It was not — 9
episodes against 8, with a Wilson interval on 9/20 spanning 26-66%. Two arms whose true
rates differ by 30 points are not separable at that sample size, and the underpowered
reading was reported as "indistinguishable" before it was rerun. Binary outcomes at n=20
should not be reported without an interval attached.

Reproduce: `scripts/arm_ab_power.py`

---

## Per-button sampling works; frame-level sticky does not

With a Bernoulli head, the rule that turns 8 probabilities into a button press matters more
than the probabilities.

Deterministic thresholding and sticky both produce **A-holds of 263–476 frames**. In SMB you
must *release* A to jump again, so a stuck hold makes every later jump impossible — sticky
was intended to lengthen holds and it does, catastrophically. Independent per-frame sampling
reproduces the expert's *distribution* of hold lengths (max 10–16 frames) rather than its
mean, and it is the only rule that clears pipe 1 at all.

---

## Two Stage 3 pseudo-experts failed validation before any training

Both on the same axis — jump timing — and both were caught by validating against a held-out
expert *before* generating data.

| teacher | metric | result |
| --- | --- | --- |
| retrieval by quantised state | A-onset recall | ≤11.9%, 59–87% of states ambiguous |
| search oracle, fixed Right+B continuation | agreement at A-onsets | 46.8–54.2% (chance) |

The oracle's failure was not the horizon or the progress measure — both were varied (60/120/180
frames; furthest-x and final-x) and neither moved onset agreement out of the noise. It was
the **continuation policy**: rolling both branches under a fixed run-right agent asks "does
jumping hurt a run-right agent over the next second?", and in SMB it never does, because you
keep horizontal momentum while airborne. That oracle over-jumped 4–5× (31% vs the expert's
6%) and had no opinion at all on 43–52% of states.

---

## Self-imitation works, and the acceptance filter did not degrade

Stage 3 arm A: roll the policy out from filtered expert start points, keep the top quarter
by progress-from-start, add to training, refit. Three rounds, two mixing ratios, every
round recalibrated, every evaluation n=200.

| round | expert:self | pipe 1 cleared (95% CI) | A-onset recall | x median 1-1 |
| --- | --- | --- | --- | --- |
| round1_contaminated_for_reference | — | 53.3% [41, 65] | 48.9% | 594 |
| stage2_armB_baseline | — | 59.5% [53, 66] | 50.0% | 594 |
| round2_ratio3to1 | 3:1 | 92.0% [87, 95] | 47.3% | 595 |
| round3_ratio3to1 | 3:1 | 96.5% [93, 98] | 44.1% | 595 |
| round2_ratio1to1 | 1:1 | 96.5% [93, 98] | 38.8% | 595 |
| round3_ratio1to1 | 1:1 | 99.0% [96, 100] | 39.6% | 595 |

**Pipe 1 goes from 59.5% to 99.0%** over three rounds at 1:1. The intervals are tight and
disjoint from the baseline's, so this is not noise.

Two caveats that matter more than the headline:

1. **Use the score cutoff, not the acceptance rate.** Acceptance was implemented as a
   fixed top-25% quantile, so it reports ~25% every round regardless of how the policy
   behaves: the intended "is it grading against its own declining standard?" alarm was
   structurally incapable of firing. That is a design flaw, not a result. The valid
   statistic is the **score cutoff**, which is an absolute progress threshold rather than a
   rank, and it rose every round: **289 → 362 → 416**, with median progress-from-start
   94 → 138 → 184. The standard got harder, so the loop was not self-congratulating. All
   future rounds report the cutoff; the acceptance rate is retired.
2. **x median is stuck at 594-595 in every round.** The policy now clears pipe 1 almost
   always and then stops at pipe 2. Self-imitation solved the obstacle it could already
   sometimes pass and made no progress on the next one — consistent with a filter that
   rewards incremental progress and therefore cannot reward a jump it never makes.

---

## More data made the policy worse

Same architecture, same 2,000 steps, fractions of the training corpus:

| fraction | frames | A-onset recall | pipe 1 (n=120) |
| --- | --- | --- | --- |
| 10% | 98,138 | 35.0% | 60.8% |
| 25% | 245,346 | 44.7% | 75.0% |
| 50% | 490,692 | 44.4% | 65.0% |
| 100% | 981,385 | 40.4% | 56.7% |

The curve **peaks at 25% and declines**. 100% of the data is worse than 10% on live play
(56.7% vs 60.8%) and worse than 25% on both metrics.

The honest caveat: this is fixed *compute*, not fixed epochs. At 2,000 steps the 100% run
sees each frame about a tenth as often as the 10% run, so this measures "best use of a
fixed training budget", not "more data is harmful in the limit". Under that reading the
answer to *was collecting 34 runs worth it* is: **not for this budget** — a quarter of the
corpus trains a better policy than all of it, which is consistent with the earlier finding
that 46% of the corpus is redundant.

---

## The 2-1 wall blocks, it does not kill

From the 2-1 start, holding Right+B reaches **x=306** and never dies. Adding a periodic
jump reaches **x=531**. Every one of 30 policy episodes ended in the stall detector ("no
progress for 300 frames"), with a mean of 0.1 deaths.

So x=530 is an obstacle that must be jumped and that punishes failure with nothing at all —
the same shape as the 594 pipe, and the reason both arms sit at exactly 530-531. A metric
that only counts distance cannot tell "stuck forever" from "died trying", and here the
policy is never dying: it is standing still.

---

## Category labels are claims, not measurements

Audited all 34 runs: declared category against the route the replay actually
took and the number of levels it actually cleared (`scripts/audit_categories.py`).
**4 mismatches.**

| run | declared | measured | problem |
| --- | --- | --- | --- |
| pub-3648 | all-items | warpless, 32 levels | declared route 'all-items' != measured 'warpless' |
| pub-4313 | all-items | warpless, 32 levels | declared route 'all-items' != measured 'warpless' |
| user-4836183441 | warps | warps, 12 levels | 12 levels measured, 8 expected for 'warps' |
| user-6384136162 | warps-glitchless | warps, 7 levels | 7 levels measured, 8 expected for 'warps' |

The label that caused real damage was `warps-glitchless`. Read as a whole it looks like a
warpless-glitchless run; it is a **glitchless warps** run. The route word and the style
qualifier are separate claims, and treating the label as atomic is what led the
glitchless-vs-glitchy experiment to believe warpless-glitchless data existed in this
corpus. It does not: there are exactly two glitchless runs, both warps, and one is in val.

Worse for that experiment specifically, `user-6384136162` — the single run its glitchless
arm was built on — **clears 7 levels, not the 8 a warps run requires**. The pilot's
treatment arm was an incomplete run.

**"Glitchless" is not verifiable by this pipeline.** Nothing measures glitch use; the
verifier checks route, level progression and sync, none of which distinguish a wall-clip
from a wall. Both glitchless-labelled runs are flagged as carrying an unverifiable claim.
Any experiment that turns on glitchiness needs a measurable proxy — which is why the
comparison was replaced with obsoletion-chain position.

---

## Older TAS data trains a much better policy than world-record data

The project's north-star question, and the largest effect measured in it.

Within the `warpless/3728` obsoletion chain, position 0 is the current publication (fastest,
most heavily optimised, most glitch-dependent) and higher positions are the older records it
obsoleted. Same route, same 32 levels, matched at 201,479 frames, 3 seeds, n=200 per arm.

| arm | runs | pipe 1 (pooled, n=600) | A-onset recall |
| --- | --- | --- | --- |
| **earliest** | pub-1194, pub-1106, pub-262 | **67.3%** | 32.5% |
| **latest** | pub-3728, pub-3665, pub-1962 | **21.7%** | 42.0% |

**Difference +45.7 pp [40.5, 50.4]**, excludes zero.
Per seed, earliest wins every time: 72.5 / 53.5 / 76.0 against 7.5 / 39.5 / 18.0.

Older, slower, less-optimised runs are **better training data** than the current world
record, by a factor of three on live task performance. The interpretation the experiment was
designed to test: frame-perfect glitch execution only works from states a learned policy
cannot reliably reach, so the more optimised the demonstration, the less of it is reproducible.

Note the inversion: the **latest** arm has *higher* A-onset recall (42.0% vs
32.5%) and *far worse* play. This is the fidelity/performance
divergence appearing across a **data** manipulation rather than a self-imitation lineage,
which is stronger evidence than the lineage result alone.

---

## Failure taxonomy: the policy is stuck, not dying

Every episode classified, n=200 per cell.

| checkpoint | level | stuck on terrain | enemy contact | x median |
| --- | --- | --- | --- | --- |
| stage 2 arm B | 1-1 | **74.5%** | 25.5% | 594 |
| stage 2 arm B | 2-1 | **89.5%** | 10.5% | 531 |
| arm A round 3 | 1-1 | **77.0%** | 23.0% | 595 |
| arm A round 3 | 2-1 | **88.0%** | 12.0% | 530 |

No pit deaths, no timer expiries, no episodes reaching the frame budget. Three-quarters to
nine-tenths of all episodes end **standing still against terrain**. Enemy contact is a
minority failure and did not grow between Stage 2 and round 3.

**This retires the negative-examples plan.** The proposal — keep failed rollouts, mark the 30
frames before each death, train against them — targets collisions, which cause at most a
quarter of failures and are not the binding constraint. The binding constraint is a jump that
does not happen.

The taxonomy is now permanent in `eval_live`, because "x median 594" cannot distinguish
standing still from dying and those need opposite fixes.

---

## Sustain: the reweighting was backwards, and fixing it was not enough

Diagnosis first, on the Stage 2 arm B checkpoint:

```
p(A) at onset 0.322    at sustain 0.272    when idle 0.135
```

Continuation frames score **below** onsets. The 10x onset weighting upweighted jump starts
and implicitly downweighted jump continuations: it taught initiation and un-taught sustain.

Four arms, trained from scratch on 300k frames for 2,000 steps, n=200:

| arm | pipe 1 | max A-hold | A-onset recall | x median |
| --- | --- | --- | --- | --- |
| (d) control, onset 10x | 76.0% | 14 | 50.3% | 594 |
| **(a) reweight sustain and onset** | **95.5%** | **21** | 35.3% | 595 |
| (b) onset 3x instead of 10x | 57.5% | 11 | 32.9% | 594 |

Arm (a) wins decisively on task performance and produces the longest holds — 21 frames,
past the expert's median 18 at pipe 2.

**And x median does not move.** 594-595 in every arm, exactly as before. The model can now
hold long enough and still does not clear pipe 2.

That relocates the problem. It is not "cannot sustain" — that is fixed. It is **cannot time**:
producing a long hold somewhere is not the same as producing it at the right x. Build the
timing diagnosis before building the hold-duration head; a head that commits to a duration
does nothing if the commitment starts in the wrong place.

Arm (c), hold-duration modelling, was deliberately not built. It changes the action space
(second head, different rollout loop) and the evidence now says duration was not the blocker.

---

## Data scaling at fixed epochs: the peak at 25% is real

Fixed *compute* could not separate "more data is worse" from "larger subsets got fewer
passes". Rerun with steps scaled to subset size, so every point gets the same number of
epochs:

| fraction | frames | steps | A-onset recall | pipe 1 (n=200) |
| --- | --- | --- | --- | --- |
| 10% | 98,138 | 200 | 19.0% | 63.0% |
| **25%** | 245,346 | 498 | 38.0% | **83.5%** |
| 50% | 490,692 | 996 | 35.6% | 71.5% |
| 100% | 981,385 | 1,993 | **44.4%** | 70.5% |

**A quarter of the corpus trains the best player.** The peak survives the correction, so it
is not a compute artifact. Doubling to 50% costs 12 points of pipe 1; using everything does
not recover it.

A-onset recall, meanwhile, rises monotonically to 44.4% at 100%, so the *most*
expert-faithful model here is not the best player. Note this family's fidelity/performance
correlation is **positive** (+0.60) — recall and performance rise together from 10% to 25%
and only diverge above that. It is a divergence at the top end, not an anti-correlation.

Answer to "was collecting 34 runs worth it": **not for task performance.** It was worth it for
measuring redundancy, for the held-out splits, and for the chain experiment, which needed
whole chains. But a quarter of the frames trains a better policy than all of them.

---

## Oracle, third and final attempt: dead

Margin-calibrated decision rule — label "jump" only when jumping beats not-jumping by more
than M pixels, with M swept so the realized jump rate matches the expert's 6.7%.

| M | jump rate | agreement overall | agreement at A-onsets |
| --- | --- | --- | --- |
| 24 | 24.4% | 74.3% | 34.7% |
| 48 | 11.0% | 85.9% | 22.2% |
| **64** | **7.9%** (matched) | 88.2% | **16.2%** |
| 96 | 2.8% | 91.7% | 7.9% |
| 128 | 0.4% | 92.9% | 0.0% |

At the matched jump rate, onset agreement is **16.2%** — far below the 70% hard stop, and
*worse* than the uncalibrated version. The mechanism is now obvious in the table: overall
agreement and onset agreement move in **opposite** directions as M rises. Every increase in M
buys overall agreement by refusing to jump, and refusing to jump is exactly what fails at
onsets. There is no M that is good at both.

**Three teachers, three failures, one axis.**

| teacher | metric | result |
| --- | --- | --- |
| retrieval by quantised state | A-onset recall | <= 11.9%, 59-87% of states ambiguous |
| oracle, fixed run-right continuation | agreement at A-onsets | 46.8-54.2% (chance) |
| oracle, policy continuation + margin | agreement at A-onsets at matched rate | **16.2%** |

Two failures is a finding; three on the same axis is a pattern. Every attempt to
hand-construct a teacher for *when to jump* failed, while a trained Bernoulli policy reaches
50% onset recall directly from the data. The pseudo-expert framing is closed.

---

## The pipe-2 ceiling is jump *placement*, not jump duration, height, or the run button

Three explanations were tested and two of the project's own earlier hypotheses were wrong.

Measured in x=500-600 of 1-1 (the run-up to pipe 2), expert against three checkpoints:

| | expert | arm A round 3 | sustain arm (a) | stage 2 arm B |
| --- | --- | --- | --- | --- |
| takeoff velocity, median (px/frame) | **2.5** | 0.0 | 0.0 | 0.0 |
| B held at takeoff | 77% | **94%** | **97%** | **91%** |
| B held across window | 68% | 94% | 97% | 91% |
| A-hold, median / max | 12 / 29 | 2 / 23 | 2 / 16 | 1 / 5 |

Pooled like that it looks like a velocity ceiling. It is not — the pooling is the artefact.
Splitting the same onsets by where they happen (arm A round 3, 12 episodes, 2,983 onsets):

| band | A-onsets | share | median takeoff velocity |
| --- | --- | --- | --- |
| approach, x 500-575 | 108 | 3.6% | **+1.60 px/frame** (max 2.60) |
| **at the wall, x 585-600** | **1,675** | **56.2%** | **+0.00 px/frame** |

**56% of every jump the model attempts happens while already touching pipe 2**, at exactly
zero horizontal speed. During the approach it reaches 2.60 px/frame, matching the expert's
2.5 — the capability is there and unused.

So:

* **Not the B button.** The model holds B *more* than the expert (94% vs 77%). Holding run
  against a wall produces no speed.
* **Not jump duration.** Arm (a) already produces 21-frame holds, longer than the expert's
  median 18 here. Confirmed separately.
* **Not jump height capability.** SMB selects initial vertical velocity from horizontal speed
  at takeoff; at v=0 the model gets the short-jump entry in that table, so no A-hold length
  can compensate. With a running takeoff the same action space would get the tall jump.

**It is placement.** The jump has to begin roughly 40-60 px before the pipe, while running.
The model runs up, stops, and then jumps from a standstill, over and over.

> **SUPERSEDED by the sweep below.** The claim that a standstill takeoff cannot be rescued by
> any A-hold is false: ground truth shows it clears with a 10-frame hold. Velocity lowers the
> required hold (10 -> 6) but is not the ceiling. The binding constraint is A-hold duration at
> the decision point. The location statistics above stand; the physics inference from them
> does not.

This also explains why three rounds of self-imitation could not fix it. Jumping earlier means
pressing A at a location where *not* jumping is locally harmless, so a filter that scores
rollouts by progress-from-start cannot distinguish the correct early jump from a wasted one —
and the correct approach may even require backing off first, which *reduces* x. A
progress-maximising selection rule will never favour that.

It is also why search is the right next move rather than more reweighting: "keep running,
press A at x~560" is a specific action sequence that search can find and imitation cannot
invent.

---

## Pipe 2 IS clearable — and the binding constraint is A-hold duration, not velocity

Ground truth, from 315 scripted configurations (A-hold x trigger position x B on/off), each
replayed from a byte-identical handover state at x=501 (the trained policy drives there
deterministically; no grounded expert frame exists in 1-1 between x=250 and 480 to savestate
from). Guard asserted: configurations returned 27 distinct max-x values, so the harness is
measuring the variable.

| takeoff | minimum A-hold that clears pipe 2 | working trigger window |
| --- | --- | --- |
| running, v >= 2.0 px/f, B held | **6 frames** | x = 540-590 |
| running, v >= 2.0 px/f, no B | 10-18 frames | x = 540-590 |
| **standstill, v = 0** | **10 frames** | n/a (clears, best max x = 713) |

Triggers at x=510-530 never cleared at any hold up to 40 — jumping *too early* fails as well
as too late. The usable window is **x = 540-590**.

### This overturns the velocity explanation

The previous section argued that SMB selects initial vertical velocity from horizontal speed,
so a standstill takeoff could not be rescued by any A-hold. **That is false, and the sweep
disproves it directly**: a standstill takeoff clears pipe 2 with a 10-frame hold. Velocity
helps -- it lowers the required hold from 10 to 6 -- but it is not the ceiling.

### What the ceiling actually is

Comparing the requirement against the policy's measured behaviour in the same window:

| | required | arm A round 3 | sustain arm (a) | stage 2 arm B |
| --- | --- | --- | --- | --- |
| A-hold at the decision point | **>= 10** (>= 6 if running) | median 2, p90 6, max 23 | median 2, p90 4, max 16 | median 1, p90 2, max 5 |
| trigger location | x 540-590 | 56% of onsets land x 585-600 | — | — |

**The policy jumps in roughly the right place and holds for a median of 2 frames where 10 are
needed.** Placement is approximately correct; duration at the decision point is not. The max
A-hold figures are misleading -- arm (a) reaches 21 frames *somewhere* in an episode, but its
p90 in the pipe-2 window is 4.

### Consequence: hold-duration modelling should be reinstated

Hysteresis and hold-duration modelling were dropped on the strength of two earlier readings:
that p(A)=0.272 during holds is "not collapsed", and that arm (a)'s 21-frame maximum showed
duration was solved. Both were about the *unconditional* hold distribution. The sweep shows
the requirement is a **conditional** one -- >= 10 frames specifically at x 540-590 -- and the
policy's conditional p90 there is 4-6 frames. Committing to a duration at an onset is exactly
the mechanism that would satisfy it.

Two independent facts now point the same way:
1. re-deciding A every frame with p(A)~0.27 gives a geometric hold distribution whose median
   is ~2, and 10 consecutive successes at p=0.27 has probability ~2e-6;
2. the requirement is >= 10 consecutive frames.

A per-frame Bernoulli cannot produce that reliably no matter how it is reweighted, because the
issue is the *independence assumption*, not the marginal probability.

---

## Retracted: the pipe-2 emulator sweep

An A-hold sweep (1-32 frames x 5 trigger positions) reported that no hold clears pipe 2.
**That result is void.** Every configuration returned max x = 314, which is short of pipe 1
at 435: the scripted agent runs into the Goomba at x~300 and dies without ever reaching pipe
2. The harness tested nothing. Recorded here rather than deleted because the failure mode is
instructive -- a sweep that returns an identical number for every setting is measuring the
harness, not the variable.

The question "is pipe 2 reachable with the model's jump?" is now partly answered from the
other direction: arm (a) produces 21-frame holds, longer than the expert's median 18 at pipe
2, and still does not clear it. That points at timing rather than height.

---

## Silent failures: the recurring pattern

Every one of these ran to completion, produced plausible numbers, and reported no error.
None would have been caught by a test that only asked "did it crash".

| failure | what it looked like | what caught it |
| --- | --- | --- |
| **Double normalization** | Model emitted a *constant* p(A)=0.00710 (std 1e-5) for every frame; every downstream metric — calibration, thresholding, onset recall, exact match — computed cleanly on it and reported 0.0% | A behavioural measurement disagreeing: 53% pipe clearance from a checkpoint scoring 0% recall |
| Attract-mode contamination | A do-nothing policy "reached the flagpole" | Watching the video |
| Level starts past the pipe | Every arm scored 100% on "cleared pipe 1" | A test asserting the start point was where it claimed |
| Start before control handover | Episodes silently wasted their first ~150 frames | Holding Right+B and finding x never moved |
| Symmetric ground filter | A-onset dataset shrank from 216 to 24 | Noticing the onset count was implausible |
| Category labels | `warps-glitchless` read as warpless-glitchless | Auditing declared labels against measured routes |

The pattern: **a pipeline stage that silently degrades its input produces confident,
well-formed, wrong numbers downstream.** The defence that actually worked, repeatedly, was
never a unit test — it was cross-checking a metric against an independent measurement of
the same thing, and investigating the disagreement rather than the more convenient number.

