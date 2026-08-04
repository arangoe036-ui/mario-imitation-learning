# Findings

Beating Super Mario Bros level 1-1 with supervised learning only — no policy gradient, no value
bootstrapping. Search was permitted because its results are distilled back by supervised learning.

**This document reports two results.** One is negative and one is positive, and the negative one is what
makes the positive one credible. Every number below names the artifact it comes from. Where a number is
unresolved it says so.

All figures are **single life** (multi-life scoring rewards dying — see §6) at **n=200 episodes per arm**
unless stated. Clearance thresholds are past each obstacle's far edge: pipe1 x>470, pipe2 x>630,
**pipe3 x>735**, pipe4 x>975. The pipe-3 threshold was derived rather than chosen: the max_x histogram
over 200 episodes has a 37-episode spike in the 720–735 bin and **nothing at all in 736–783**.

---

## 1. The two results

### 1a. Negative: a three-button script matches or beats every learned checkpoint at pipes 1–2

The control that had never been run: **Right and B held on every frame, with A flipped as an i.i.d. coin
at a fixed probability.** No network, no observations, no learning.

`data/p1_script_control.json`, n=20 per arm:

| p(A) | x median | pipe1 | pipe2 |
|---|---|---|---|
| 0.00 | 312 | 0.0% | 0.0% |
| 0.15 *(the expert's own rate)* | 436 | 45.0% | 0.0% |
| 0.50 | 595 | 85.0% | 10.0% |
| **0.85** | **722** | 70.0% | **70.0%** |
| 1.00 | 316 | 0.0% | 0.0% |

The curve has a sharp interior optimum near p=0.85, and **both extremes die at x≈312.** The learned
policy's measured A-rate is **0.852**.

At n=200, paired on identical seeds against `C_control_matched_r2` — the checkpoint every late-project
figure rested on (`data/p1_control_ladder.json`, `data/traces/p1_200.json`):

| | script p=0.85 | policy | policy − script |
|---|---|---|---|
| pipe1 | 145 (72.5%) | 146 (73.0%) | +0.5 pp [−8.2, +9.2] |
| **pipe2** | **137 (68.5%)** | **137 (68.5%)** | **+0.0 pp [−9.0, +9.0]** |
| x median | 722 | 723 | — |

**Identical counts at pipe 2 — 137 and 137.**

Adding the policy's other button rates makes the script *stronger*, not weaker. With Left at the policy's
own 0.135, the script reaches **pipe1 87.0% (174/200) and pipe2 82.5% (165/200)** — better than any learned
checkpoint. Scored against that opponent (`data/p3_script_scored.json`), **all six measured checkpoints
lose at pipe 2, every interval excluding zero:**

| checkpoint | pipe2 | vs best script (82.5%) |
|---|---|---|
| `C_control_matched_r2` | 68.5% | −14.0 pp [−22.2, −5.6] |
| `top20_round2` | 62.0% | −20.5 pp [−28.8, −11.8] |
| `surv_round2` | 60.0% | −22.5 pp [−30.8, −13.7] |
| `surv_round3` | 55.5% | −27.0 pp [−35.3, −18.1] |
| `compose_round2` | 54.0% | −28.5 pp [−36.8, −19.5] |
| `round3_ratio1to1` | 21.5% | −61.0 pp [−67.9, −52.5] |

**Every clearance figure this project reported at or below pipe 2 is a statement about button rates.**

### 1b. Positive: the policy beats every fixed-rate script at pipe 3 by ~24 points

Scored **conditional on arrival** — a policy that gets further up the level inflates its apparent
advantage at every later obstacle, so the question "did this obstacle improve" requires conditioning.
Arrived at pipe N ⟺ cleared pipe N−1. Opponent: the strongest fixed-rate script *per obstacle*.

Pooled across **three training seeds, n=600 episodes** (`data/plain_three_seeds.json`):

| obstacle | policy | arrivals | strongest script | advantage |
|---|---|---|---|---|
| pipe1 | 75.0% | 600 | 87.0% | −12.0 pp [−17.4, −5.7] |
| pipe2 | 96.2% | 450 | 96.8% | −0.6 pp [−3.4, +3.7] |
| **pipe3** | **55.0%** | **433** | **31.1%** | **+23.8 pp [+14.7, +32.1]** |
| pipe4 | 53.4% | 238 | 39.0% | +14.3 pp [−2.2, +29.1] |

**Pipe 3 is the result: +23.8 points, interval excluding zero, three seeds, unbiased loss, conditioned on
arrival.** It also holds unconditionally (+16.2 pp [+8.8, +22.8]).

**Pipe 4 is not established.** Conditionally its interval includes zero — the script arm has only 41
arrivals. Unconditionally it is +13.2 pp [+7.6, +17.7]. It improved in **3 of 3 seeds** with a seed spread
of 4.5 pp, so the direction is probably real, but see §7.

**The pair is the honest headline:** *a fixed-rate script matches or beats every learned checkpoint at
pipes 1–2 and no checkpoint beats it there — yet the policy beats every script at pipe 3 by ~24 points.
Imitation did learn something, and it is confined to the obstacle a fixed rate cannot solve.* The negative
half rules out the explanation that would otherwise apply to the positive half.

**What did not improve is the delta from base.** Pooled over three seeds, the conditional advantage moved
**pipe3 56.9% → 55.0% (−2.0 pp [−11.2, +7.6])** and **pipe4 48.7% → 53.4% (+4.6 pp [−8.0, +17.1])**. The
training loop adds nothing on top of the base checkpoint. That is a different claim from 1b and only 1b is
a project result.

---

## 2. The loss-bias mechanism — the most transferable finding here

**Up-weighting rare positive frames is standard practice for imbalanced action labels. It manufactures a
degenerate policy.**

The objective used by every "composed recipe" run in this project (`scripts/compose.py::sustain_loss`)
weights onset frames 10×, sustained presses 5×, and released frames 1×:

```
w = 1 + (ONSET_W − 1)·onset + (SUSTAIN_W − 1)·(pressed ∧ ¬onset)
```

Every term up-weights **pressed** frames; nothing up-weights released frames. For a Bernoulli head with
weight `a` on positives and 1 on negatives, the weighted optimum is not the base rate `p` but

**a·p / (a·p + (1 − p))**

so a 5× sustain weight turns a true p=0.5 into 0.833. **The optimum is above the base rate by
construction.**

Measured, from scratch on expert data, 400 steps, same seed, only the loss differing
(`data/loss_bias_probe.json`):

| button | expert data | plain BCE | `sustain_loss` | closed-form optimum |
|---|---|---|---|---|
| **A** | 0.147 | **0.130** | **0.403** | 0.463 |
| B | 0.509 | 0.511 | 0.745 | 0.838 |
| Right | 0.453 | 0.444 | 0.725 | 0.805 |
| Down | 0.007 | 0.009 | 0.053 | 0.036 |
| Left | 0.028 | 0.023 | 0.112 | 0.126 |

**Plain BCE recovers the base rate on every button. The weighted loss inflates every button, and the
measured values track the closed form.** One pass takes A from 0.147 to 0.403; iterated self-imitation
rounds compound it to 0.85–0.97.

### The dose-response across the project's entire history

`data/loss_provenance.json` — a code read mapping every arm to its objective. Three objectives exist, and
**`overnight_lib.train_policy` defaulted to `onset_weight=10.0`**, so any caller that did not override it
was biased by default (that default is now 1.0).

| loss | arms | measured A-rate, n=200 single life |
|---|---|---|
| plain BCE (`onset_weight=1.0`) | 2 | **0.152** |
| onset 10× | 5 | **0.219, 0.628** |
| onset 10× + sustain 5× | 6 | **0.822, 0.852, 0.865, 0.888, 0.926, 0.970** |

**Ordered by press-weighting strength with no exceptions** — and the plain-BCE arm sits at **0.152, the
expert's own rate to three decimals.** That ordering across independent arms is what makes this a
mechanism rather than a correlation in one run.

Only **2 of ~88 checkpoints** in this project were trained under an unbiased loss.

### The behavioural pathology it produces

The A press *rate* is the wrong statistic: a high jump rate can be correct — the expert jumps constantly
because jumping is faster than running. What is not correct is never releasing.

Measured per frame with a `grounded` flag (`data/plain_three_seeds.json`, pooled n=600):

| statistic | policy | expert |
|---|---|---|
| airborne fraction | **78.0%** | **61.1%** |
| A held while airborne | **87.0%** | — |
| **A-onsets while grounded, per 1,000 frames** | **2.8** | — |

**2.8 onsets per 1,000 grounded frames is about one jump start every 352 frames.** The policy is not
jumping often — **it is staying airborne by never releasing the button.**

**In SMB you must release A to jump again.** This is precisely the pathology that killed "frame-level
sticky actions" earlier in the project, arrived at independently from the other direction: holding A
through a descent does nothing and blocks the next jump.

---

## 3. The script control as a method point

**A trivial fixed-rate baseline should exist before any clearance figure is reported.** Its absence made
every pipe-1 and pipe-2 number in this project uninterpretable for weeks, because each new result was
compared against the previous model rather than against nothing at all.

The control is three lines: hold Right and B, flip A at probability p. It costs **0.6 minutes** for 100
episodes (`data/p1_script_control.json`). It would have shown, at any point, that pipe-2 clearance is a
function of the A-rate — and it did, once run: a distilled arm at A=0.468 clears pipe 2 at 1.5%, while the
*script* at p=0.50 clears it at 10.0%. **Pipe-2 performance tracks the marginal whether the marginal comes
from a network or a coin.**

Two extensions mattered. Adding Left and Down at the policy's own rates made the script **stronger** at
pipes 1–2 (§1a) — so the control must be given the policy's full marginal, not a convenient subset. And
matching the *RNG consumption* mattered: the policy draws 8 uniforms per frame (one per button), the script
drew 1, so an episode-set coincidence test between them was **structurally unable to detect coincidence** —
identical behaviour would still have produced independent episode sets (§4, failure 21).

---

## 4. Measurement failures

**Nineteen silent failures, none caught by 324 passing tests.** Every one ran to completion, produced
plausible, well-formed numbers, and reported no error. No test covers the Bernoulli calibration path,
`overnight_lib`, `stage3_train`, or any script in `scripts/`; **every experimental number in this project
came from untested code.**

| # | failure | caught by |
|---|---|---|
| 1 | Double normalization — constant p(A)=0.00710, every metric computed cleanly | 53% pipe clearance from a checkpoint scoring 0% recall |
| 2 | FCEUX playing nothing — 17,868 frames of attract mode with a correct frame count | The verifier; every frame-exactness assertion passed |
| 3 | Attract-mode contamination — a do-nothing policy "reached the flagpole" | **Watching the video** |
| 4 | Level starts past the pipe — every arm scored 100% on pipe 1 | A test asserting the start was where it claimed |
| 5 | Start before control handover | Holding Right+B and finding x never moved |
| 6 | Symmetric ground filter — onset dataset shrank 216 → 24 | Noticing the count was implausible |
| 7 | Category labels — `warps-glitchless` read as warpless-glitchless | Auditing declared labels against measured routes |
| 8 | Pipe-2 sweep harness — every config returned x=314 | An identical number for every setting |
| 9 | Velocity pooling verdict — "the ceiling is B" | Splitting the same onsets by location |
| 10 | **Physics inferred instead of measured** — a standstill ceiling reasoned from a jump table, killing two directions | Running the sweep the inference made look pointless |
| 11 | **A condition label not matching the condition** — "standstill v=0" was 0.4 px/f | Re-running with a genuine dead stop |
| 12 | **A sweep with no baseline** — 222 of 323 configs returned what pressing nothing returns | Running the do-nothing control |
| 13 | A median reported as a universal — "x median 595" read as "never passes pipe 2"; it passes 21.5% | Computing the rate instead of the median |
| 14 | **A metric that rewards dying** — every clearance figure was best-of-several-lives; 99.0% is really 81.5% | A second independent measurement disagreeing |
| 15 | **A threshold at the obstacle's face** — "19% clear x>720" was 19% *arriving*; x>760 could never fire | Cross-checking a crossing count against a known clearance count |
| 16 | The coin-room merge — 1,400 px of "1-1 terrain" is a different room; `area` never changes | An x histogram with a 1,400-px hole |
| 17 | **The y readout wrapping at 256** — the pit test `y > 200` could not fire; 0 pits in 600, across seven requests | An impossible reading: y "climbing" during a descent |
| 18 | **A guard clause that suppressed the only signal** — `if xs[f] > 0` excluded every x→0 transition, and a pipe transit *is* x→0 | **The owner watching footage and seeing the coin room** |
| 19 | `y == 176` as a landing test — ambiguous across three absolute heights, so 91 falls scored as crossings | Re-scoring against corrected geometry |

**Four found after that list was compiled**, in the final blocks:

| # | failure | caught by |
|---|---|---|
| 20 | **The training objective inflating every marginal** (§2) — distorted every figure since the project's founding comparison | A trained policy's marginals coming out **above its own training data's** — impossible for i.i.d. supervised learning |
| 21 | **A coincidence test that could not detect coincidence** — the two arms consumed the RNG at different rates, so identical behaviour would still give independent episode sets | Checking how each arm drew its randomness before reporting the number |
| 22 | **A threshold left on a retired scale** — `min_progress=120` was 120 *pixels*, carried into a credit function ranging 0–4; it would have rejected every rollout and the training loop would have silently accepted nothing | Grepping for thresholds after changing a scoring scale |
| 23 | **A marginal intervention calibrated on the wrong distribution** — a logit offset fitted on rows the *expert* visits realised 0.349 live against a 0.219 target, because the arm visits its own states | Measuring the realised rate instead of assuming the fitted one |

**The defence that works** is cross-checking one measurement against an **independent** measurement of the
same thing, and investigating the disagreement rather than the more convenient number. Failures 1, 14, 15
and 20 were all caught exactly this way.

### Four sub-patterns worth naming

1. **An inference inserted into a chain of measurements is indistinguishable downstream from a
   measurement.** Tag inferences as inferences. (10)
2. **A condition is what the code produced, not what the label says** — and no number means anything until
   the trivial control has run in the same harness. (11, 12, §3)
3. **A guard clause is a claim that nothing interesting happens in the excluded region.** Write the claim
   down. (18)
4. **Absence of an observation is not absence of the thing.** Four terrain measurements died on this. (17)

---

## 5. The owner's observations

**Five observations from watching footage redirected this project, and every one was right.** None came
from a metric.

1. **"It passes pipe 2 often"** — against a reported x median of 595 read as a wall. It passes 21.5%
   (failure 13).
2. **"Pipe 3 is the limit"** — later confirmed as the frontier: 34 deaths at the face, dwell median 161
   frames. And it is where the only positive result in this document lives (§1b).
3. **"The play looks glitchy"** — the first signal of the always-jump degeneracy, months of metrics before
   §2 explained it.
4. **The coin room** — 1,400 px of supposed 1-1 terrain that is a different room, invisible to every
   metric because `area` never changes (failures 16, 18).
5. **The holes and the turtles** — two of seven obstacle classes, both found by eye. Turtles at
   x≈1,216–1,248 on unbroken ground, which the enemy classifier reported as 1 in 600 because its table
   covered 5 IDs of many.

**And the reframe that produced §2's behavioural analysis:** *"I feel it jumps so much because the pro
jumps a lot as well — that's running fast. It's faster than running. So we have to adjust for that."* That
is correct, and it retired the A press rate as a headline statistic in favour of airborne fraction,
A-onsets while grounded, and A-held-while-airborne — which is how the real pathology was finally measured.

**One of the five was filed as prose and ignored for eight hours.** The process lesson: **a specific,
falsifiable observation enters the measurement queue regardless of the form it arrives in.** Turtles, gaps
and the coin room — three of seven obstacle classes — were all found by a human watching footage, and only
two of the seven were ever modelled.

---

## 6. Retractions

Retractions with mechanisms, not apologies.

| retracted | why |
|---|---|
| **The composition headline** — pipe 2 21.5% → 62% | The A-rate rose 0.628 → 0.822 across that same sequence, and **no checkpoint beats the fixed-rate script at pipe 2** (§1a, §2). The gain moves with the marginal. `data/p2_marginals.json` |
| **Stage 2's founding win** — "bernoulli-only 29.5% → +onset-reweight 59.5%, +30.0 pp" | Two arms differing *only* in `onset_weight`. Re-measured single life: 23.0% → 44.0% = **+21.0 pp**, of which **+13.0 pp (62%) is reproduced by adding one constant to one logit** — an intervention that cannot add state-dependent behaviour. Residual **+8.0 pp [−1.6, +17.4]: unresolved, not zero.** `data/stage2_marginal_test.json` |
| **Every multi-life clearance figure** | The harness counted deaths and continued, giving best-of-several-lives and **rewarding dying**. Known conversions: 99.0% → 81.5%, 67.3% → 52.3%, 21.7% → 18.2% (failure 14) |
| **All four gap geometries** — 1,475–1,531 / 1,525–1,562 / 182 px / airborne spans | Every one inferred terrain from where Mario happened to be. The gap's width is **still unmeasured** after five attempts |
| **The 91 gap "crossings"** | 0 of 188 passed the real far edge; they are falls (failure 19) |
| **"39 clearing configurations at pipe 4"** | **22 of 39 reproduce.** All 17 non-reproducers are one prefix. The minimum — hold ≥12 at trigger x≈892 — survives on a different prefix |
| **"+7.5 pp at pipe 3 from the fixed loss"** | My own, from one seed. Conditioned on arrival and pooled over three seeds it is **−2.0 pp**. That same arm's pipe-4 conditional advantage was −1.4 pp while three new seeds landed at +12.2 / +14.5 / +16.7 — an outlier in both directions at once |
| **The frontier map as a property of "the policy"** | 29 stuck at pipe 4, 38 clearing it, x median 723 — all real, all describing a checkpoint that holds A on 85.2% of frames and Right/B on 99.9% |

---

## 7. Limits

- **1-1 only.** Generality is untested; the best model's x median on 2-1 is 530.
- **Pipe 4 is not resolved.** Conditional advantage +14.3 pp [−2.2, +29.1] pooled; improved in **3 of 3
  seeds** with a 4.5 pp spread, so the direction is probably real. Resolving +4.6 pp needs ~1,900 arrivals
  per arm ≈ 9,300 episodes. **Recorded as consistent across seeds, n-limited, not resolved.**
- **2 of ~88 checkpoints were trained under an unbiased loss.** Everything else inherits §2's bias, and
  only 7 checkpoints have ever been measured at all.
- **324 tests pass and none covers the code that produced any experimental number.**
- **The loop is stable-degenerate.** The pooled policy's A-rate is **0.871 — exactly its training data's
  0.871.** Plain BCE faithfully reproduces a degenerate marginal, which is correct behaviour on degenerate
  data. **Fixing the loss stops amplification; it does not stop inheritance.** Each round starts from the
  previous round's marginal.
- **Unmeasured and no longer needed:** pipe 4's height (four attempts). **Unmeasured and still wanted:** the
  gap's width (five attempts).
- Two of seven obstacle classes were ever modelled.

### Future work, in priority order

1. **Behaviour-filtered acceptance.** Filtering self-imitation rollouts on A-releases or hold-length
   distributions is the one untried lever that addresses **inheritance** rather than amplification — the
   mechanism §7 identifies as the loop's binding constraint. It is the next hypothesis and it is not
   started; whether it is worth another day is a decision for the project owner, not a step to slide into a
   write-up.
2. **Resolve pipe 4 cheaply** by evaluating from **pipe-4 start states**, where every episode is an arrival
   — roughly 600 episodes for 3× the power of the 9,300 needed from level start. The start-state machinery
   for this exists (`data/startlib_policy.json`, `session.save_scratch`/`load_scratch`).
3. **A ROM level-data reader** for exact terrain, requested five times and never built. It is the answer
   whenever terrain matters again — four geometry attempts died on inferring terrain from sampled positions.

---

## Artifacts

| file | contents |
|---|---|
| `data/p1_script_control.json` | the fixed-rate script curve, p(A) ∈ {0, 0.15, 0.5, 0.85, 1.0} |
| `data/p1_control_ladder.json` | script arms with Left, Down, both, and RNG-matched, n=200 each |
| `data/p2_marginals.json` | five headline checkpoints re-measured with button marginals |
| `data/p3_script_scored.json` | every checkpoint scored against the best fixed-rate script |
| `data/loss_bias_probe.json` | the from-scratch loss comparison and closed-form optima |
| `data/loss_provenance.json` | which loss produced every arm, with file:line evidence |
| `data/stage2_marginal_test.json` | the founding result decomposed into marginal and residual |
| `data/plain_three_seeds.json` | three seeds, conditional-on-arrival scoring, behaviour statistics |
| `data/startlib_policy.json`, `data/reach_table.json` | policy-visited start states and the per-start script reach baseline |
| `data/traces/*.json` | per-frame retention: (x, y_absolute, speed, buttons, player_state, grounded) |
