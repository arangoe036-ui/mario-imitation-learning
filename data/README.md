# The evidence base

Every number in this repository traces to a file in here. Nothing was computed in a notebook,
by hand, or in a terminal that wasn't saved.

## The convention

**One experiment, one script, one artifact.** `scripts/<name>.py` writes `data/<name>.json`.
83 of the 107 experiment scripts pair with a same-named artifact; the rest either write several
files or extend an earlier one. **140 result files in total** (128 `.json`, 9 `.jsonl`, 3 `.npz`).

This is why the directory looks the way it does. It is not a dump — it is 107 experiments, most
of which can be re-run on their own.

**Four exceptions, named because the convention above would otherwise imply they can be re-run.**
`reach_walls.json`, `stats_synced.json`, `seed_variance_permutation.json` and `stats_all.json` have
no producing script in this repository. `stats_synced.json` and `stats_all.json` are the output of
the `tasdata stats` CLI (`--synced-only` and not) rather than of a script, and it needs the
uncommitted capture set; for `reach_walls.json` and `seed_variance_permutation.json` the generator
was not committed. The artifacts are the evidence; from this clone they cannot be regenerated.

## Where the published claims come from

**The one positive — learning beats its *matched* blind baseline at the Koopas**
`runlength_script_control.json` · `rate_matched_control.json` · `plain_three_seeds.json` ·
`three_way_script_table.json` · `seed_variance_5.json` · `seed_variance_permutation.json` ·
`route_audit.json`

The control is a run-length script matched on the policy's *own* token marginals — same action
representation, no vision. Ten paired seeds; the permutation test is exact.

⚠ **A different, unmatched blind script beats the policy at that same wall.** `route_audit.json`
runs a fixed-rate A-0.85 script on the same basis (single life from the level start, n = 200 per
arm, policy arm `P_84_cnn32_seed4`) and it clears the Koopas **27.0% against the policy's 19.5%**,
pipe 3 **57.5% vs 47.5%**, pipe 4 **37.5% vs 29.5%**, with median max_x **827.5 vs 723**. The
+5.5 pp is a win over the representation-matched control, not over every blind script. Read the
two together or not at all.

**The negative — a perfect teacher leaves no recoveries to learn from**
`corpus_composition.json` · `stats_synced.json` · `split.json` · `depth_vs_steps.json` ·
`lift_by_training_length.json` · `peak_rebaseline.json` · `fidelity_vs_performance.json`

Three corpus sizes, all of them real: 1,684,996 frames captured, 1,223,797 in the 25 synced runs
(`stats_synced.json`), 981,385 in the 20-run expert-train split that training actually reads
(`corpus_composition.json`). "Zero deaths and zero recoveries" is **not** measured anywhere here —
`stats_synced.json` has no such field, and `corpus_composition.json` reports that the index keeps
death animations and other out-of-control frames (3.3% of samples). The best rung of the steps
ladder is 1,000 steps / 0.82 epochs, but `depth_vs_steps.json` is n = 100 and single-seed past
3,000 steps, and its 1,000-step lead over the 15,000-step rung is 0.5 px of median x. Imitation
fidelity shows no clear relationship with task performance: `corr_recall_pipe1 = −0.036` and
`corr_exact_pipe1 = +0.405` at n = 29, verdict "NO clear relationship at this sample size".

**The reach ceiling — failures are positional, not gradual**
`reach_table.json` · `reach_walls.json` · `recheck_720.json` · `reach_curve_P_84_cnn32_seed1.json` ·
`reach_curve_P_84_cnn32_seed4.json` · `terrain_profile.json`

720 episodes from 72 saved start positions across two networks. The policy stops at the same
absolute x no matter where it begins.

**The trivial baselines that came first**
`p1_script_control.json` · `p1_control_ladder.json` · `script_baseline.json` · `single_life.json`

Right and B held with A flipped as a coin. It matches the policy through pipe 2, and it is the
reason the level-completion clip is captioned as proving nothing.

**Obstacle-level diagnosis**
`goomba_forensics.json` · `goomba_pa_probe.json` · `phase2_goomba.json` · `phase2_goomba_v2.json` ·
`pipe2_sweep.json` · `pipe2_pa_probe.json` · `pipe3_requirement.json` · `pipe3_reconcile.json` ·
`pipe4_build.json` · `pipe4_transfer_audit.json` · `partial_right.json` · `partial_right_dose.json`

**Interventions that did not work** — kept because a null is a result
`window_reweight_sweep.json` · `level_restricted_*.json` · `label_smoothing_*.json` ·
`nonlinear_head_eval.json` · `probe_ontop_power.json` · `dagger_*.json` · `objective_train.json` ·
`generation_sweep.json` · `temperature_ladder.json` · `vision_2x2.json`

**Audits that caught measurement defects** — see the README's defect table
`stall_rule_audit.json` · `hidden_area_check.json` · `graphics_integrity.json` ·
`route_audit.json` · `arrival_state_audit.json` · `category_audit.json` · `idle_audit.json` ·
`loss_provenance.json` · `mps_boundary.json` · `verify_capture128.json`

Each of these exists because a number disagreed with an independent measurement of the same
quantity and the disagreement was investigated.

**Withdrawn claims** — `claim_rewrite.md` records what was retracted and what replaced it.

## Subdirectories

- **`movies/`** — the TAS input files, with provenance in `movies/README.md`. These are the work
  of their authors, not of this project.
- **`plots/`** — published figures.

## Not in the repository

- **`traces/`** — 552 MB of raw per-episode dumps. Regenerable, and no published number cites
  one. Removed from history to keep the clone at ~30 MB.
- **Captured run data** (~11 GB) — rebuild with
  `tasdata batch --plan data/shortlist.json --rom smb.nes --out data/runs`, but see the warning
  below: the movies it reads are not distributed either.
- **Training checkpoints** — intermediate `.pt` files.
- **The ROM** — bring your own; see the main README.

⚠ **`shortlist.json` cannot be run as-is from a fresh clone.** Its 34 `selected[].path` entries
point into `data/movies/pool/`, which is gitignored, so **none of them exist here** — download the
publications yourself into that directory first. The paths were absolute paths from the author's
machine and are now repository-relative, which makes them readable but no less missing (checked:
**0 of 34 exist** in a fresh clone). `shortlist_repaired7.json` is not a corrected copy of the
whole plan — it is a 7-entry plan whose paths were repaired after the project directory moved,
with the measurement fields already filled in. Check that every `selected[].path` exists before
trusting a batch run: **a missing movie prints one `FAIL` line and the run still reports overall
success.** `tasdata batch --report` defaults to
`<out>/batch_report.json` so that such a run cannot overwrite the committed `batch_report.json`.
