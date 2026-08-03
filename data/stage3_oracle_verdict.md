# Stage 3 arm B sanity check — the search oracle does not agree with the expert

**Verdict: do not generate data with this oracle.** Agreement at A-onsets is at chance, and
neither of the two suspects named in advance (the horizon, the progress measure) is the
cause.

## Protocol

Run the oracle on frames of a **held-out** expert run (`user-6378829043`, warpless, 68,509
frames, in the immutable test split). At each frame: force A on, roll forward, measure
progress, restore; force A off, roll forward, measure, restore; label with the better
branch. Compare against what the expert actually did on the *next* frame (the same
`a_{i+1}` convention training uses).

Sampling is stratified and the strata are never pooled. A-onsets are ~1.5% of frames, so
1,000 uniform samples would contain ~15 and the onset number would be noise. 500 uniform
in-control frames carry the overall and false-positive figures; 216 A-onset frames (all
that exist under the eligibility filter) carry the recall figure.

Eligibility is *not* the savestate-library filter. That one requires y to be stable in both
directions, which by construction rejects every A-onset — at the frame the expert first
presses A, y is about to rise. Using it left **24** onsets in 68,509 frames. The oracle
uses a backward-only ground test (was Mario on the ground over the preceding 4 frames),
which asks the right question and yields 216.

## Results

| horizon | jump hold | progress measure | oracle jumps | expert | ties | agree overall | **agree at A-onsets** | s/label |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 20 | furthest x | 31.2% | 6.0% | 43.2% | 67.2% | **50.9%** | 0.087 |
| 60 | 20 | final x | 31.2% | 6.0% | 43.5% | 67.2% | **50.5%** | 0.087 |
| 120 | 20 | final x | 33.0% | 6.0% | 45.7% | 64.6% | **54.2%** | 0.169 |
| 180 | 20 | final x | 33.2% | 6.0% | 50.6% | 65.2% | **47.7%** | 0.256 |
| 120 | 8 | final x | 25.2% | 6.0% | 51.7% | 70.8% | **46.8%** | 0.168 |

Cost, at the specified 60-frame horizon: **120 emulator frames per label** (two 60-frame
branches), 0.087 s per label on one persistent FCEUX. That is ~11.5 labels/s, so a million
labels would cost about 24 hours of wall-clock — affordable, which is exactly why it
matters that the labels are wrong.

## Reading the numbers

**Agreement at A-onsets is 46.8–54.2% across every variant.** A coin flip. On the frames
where the expert decided to jump, the oracle is uninformative.

**Overall agreement of 67.2% is not evidence of anything.** The expert presses A on 6.0% of
eligible frames, so a policy that never jumped would score 94% overall. 67.2% is *worse*
than always saying no.

**The oracle jumps 4–5× too often** (25–33% vs the expert's 6.0%) and jumps on 30.9% of
frames where the expert pressed nothing at all.

**43–52% of states are ties** — both branches reach exactly the same score. Nearly half the
time the oracle has no opinion, and a tie currently resolves to "don't jump".

## Which design choice is at fault

The horizon and the progress measure were the two named suspects. Both were varied and
neither moves onset agreement out of the noise: 60→120→180 frames gives 50.9 → 54.2 → 47.7,
and furthest-x → final-x gives 50.9 → 50.5. Ties actually get *worse* with a longer horizon
(43.2% → 50.6%), because more of the branches converge back onto the same ground.

That leaves the third choice, the one the module docstring flags as "what makes the
comparison fair and also what makes it approximate": **the continuation policy.** Both
branches roll forward under a fixed Right+B. So the question the oracle actually answers is
"does jumping hurt a run-right agent over the next 1–3 seconds?" — and in SMB the answer is
usually no, because you keep horizontal momentum while airborne and a jump costs almost
nothing. The expert jumps when the jump is *necessary*, which is a fact about its own future
trajectory, not about a fixed continuation.

This is not fixable by tuning. Making the continuation competent requires a policy in the
loop, which is circular at the point where the oracle is supposed to bootstrap one.

## What this rules out, and what is left

Two candidate Stage 3 teachers have now failed validation before any training:

- the **retrieval pseudo-expert** — A-onset recall ≤11.9%, with 59–87% of states carrying
  contradictory expert actions;
- the **search oracle** — A-onset agreement at chance, over-jumping 4–5×.

Both failed on the same axis: jump timing. Arm B's trained Bernoulli policy reaches 45.5%
A-onset recall at a matched press rate, so it remains the strongest teacher available, and
neither pseudo-expert clears it.

Options worth considering, none started:

1. Score the branches on something jump-specific rather than distance — did the branch pass
   an obstacle the other did not, did it die.
2. Abstain instead of guessing: only emit a label when the margin is large. Ties and
   near-ties are ~half the states and currently produce noise.
3. Drop the pseudo-expert framing and go to RL from the arm B checkpoint, where the reward
   is progress and the credit assignment is the algorithm's problem rather than a
   hand-built lookahead's.


---

# Retest with the arm B checkpoint as the continuation policy

The diagnosis was right and the fix helps, but not enough to pass the gate.

Both branches are now driven by `B_bernoulli_onset10x_step3000_recal.pt` (45.5% A-onset
recall offline) instead of a fixed Right+B, with common random numbers so the only
difference between branches is the forced A bit. `suppress_off` controls how long the
"don't jump" branch is forbidden from pressing A: `None` means the full jump-hold window
(the off-branch is also blocked from starting its own jump), `1` means only the decision
frame, which makes the two branches mirror images.

| horizon | rollouts | suppress_off | oracle jumps | expert | agree at A-onsets | at onsets, confident | confident frac | agree overall | s/label |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 | 1 | full hold | 61.0% | 6.0% | **63.0%** | 67.6% | 63.0% | 39.8% | 0.18 |
| 60 | 1 | 1 frame | 54.2% | 6.0% | 51.9% | 60.4% | 71.3% | 46.6% | 0.17 |
| 120 | 1 | 1 frame | 49.4% | 6.0% | 49.1% | 56.0% | 76.9% | 49.4% | 0.34 |
| 120 | 3 | 1 frame | 56.2% | 6.0% | 51.9% | 58.2% | 76.4% | 43.8% | 1.02 |

**Agreement at A-onsets rises from 50.9% to 63.0%** (67.6% on the confident 63% of states).
That is a real gain over the fixed-continuation oracle and above chance, so the
continuation policy was indeed the binding constraint.

**But the oracle now jumps on 49-61% of frames against the expert's 6.0%**, and overall
agreement *fell* to 39.8-49.4% — worse than the 67.2% of the fixed-continuation version and
far worse than the 94% a policy that never jumped would score. A teacher this biased would
train the student to jump roughly ten times too often.

Two things this rules out as the explanation:

- **It is not the branch asymmetry.** Suppressing A in the off-branch for only the decision
  frame, rather than the whole hold window, does reduce over-jumping (61.0% → 54.2%) but it
  *lowers* onset agreement (63.0% → 51.9%). The handicap was not what was producing the
  signal.
- **It is not sampling noise.** Averaging three rollouts per branch changes nothing
  material (51.9% at onsets, 56.2% jump rate) at 6x the cost.

The remaining explanation is the progress objective itself. Over a 1-2 second horizon,
jumping is close to free in SMB and occasionally helps, so a progress-maximising comparison
says "jump" almost every time it says anything. The expert jumps rarely because most jumps
are unnecessary, not because they are locally costly. Distance over a short horizon cannot
express that, whatever drives the continuation.

**Verdict: still do not generate data with this.** 63% onset agreement paired with a 10x
over-jump bias is not a usable teacher. The next thing to change is the objective, not the
continuation or the horizon: score the branches on something jump-specific (did this branch
clear an obstacle the other did not, did it die) rather than on distance.

Cost if it were usable: 120-240 emulator frames per label, 0.17-0.34 s/label.
