# Next session: bring Leon 2026 up to the Bruns 2017 standard

Written 2026-09-01. The previous handoff is preserved as
`next_session_prompt_2026-08-30.md` — it still holds the 2027 design notes and the
standing traps, so read it second, not instead.

Branch: work directly on `v1.4.0-dev`, commit per work unit, Douglas pushes.
**13 commits are unpushed as of this writing.**

## READ FIRST, in this order

1. **`docs/STEP3_CHARTS_AND_SETTINGS.md`** — the specification this session must follow:
   the four stage-3 charts, the atmospheric-floor method with its two traps, the exact
   settings as run, and what has to be true before datasets are compared.
2. `D:\MEE2024 output\MEE_output\WHICH_REDUCTION_IS_THE_RECORD.md` — which tree is the
   record for each cell, and which are experiments.
3. `docs/MATRIX_2026.md` — cell 1 in full, including the supersessions. Trust the latest
   section; earlier ones carry marked corrections.

## THE JOB: Leon 2026, output matching Bruns 2017

Cell 1 now has a reduction of record, a complete error budget, and four charts. Leon has
a headline number and an older chart set that does not match. Bring it level.

**Cell 1 as it stands** (for the target format):

> L = 1.764 ± 0.060 (stat) ± 0.084 (scale) ± 0.15 (atmosphere) ″, total 0.182.
> Bruns 2018 published 1.752 ± 0.060. Tree `matrix_bruns2017_brunsmethod/`, charts and
> tables copied into `RECORD/bruns2017/`.

**Leon as it stands**: L = 1.98 ± 0.60 (stat) ± 0.33 (atm) ″, from the 0.6+1.2 s union,
42 stars, below-Sun anchor in, vertical-deg-2 nuisance on. Charts in `step3_s2_plots/`.

### The convention decision, and it is per-instrument on purpose

* **Bruns → moment centroiding** (`centroid_refine_window=False`), because that is how he
  did it (the design criterion) and because the estimator choice is worth only ~1.6 ppm on
  his optics — it barely matters there.
* **Leon → windowed centroiding** (`centroid_refine_window=True`), because Leon's optics
  show exactly the aberration the windowed estimator exists for: brightness-dependent
  centroid bias growing with field radius, measured at **172 and 299 mas beyond
  r = 2500 px in twelve zenith fields out of twelve**, making the fitted cubic a function
  of the magnitude limit at the 4.5 % level (`docs/LEON_2026-08-11.md` §18.3).

**The caveat that must not be lost**: "convention" has *two* axes, not one — the estimator
**and** the background mode — and the background mode is the bigger lever (19.1 ppm on
Bruns against 1.6 ppm for the estimator). The Leon headline used **windowed + annular**;
Bruns' record uses **moments + Gaussian**. So the two cells differ on both axes, and the
comparison rests on the *measured* convention sensitivity rather than on identity:
re-reducing Leon end-to-end in Bruns' convention moves its L by only **−0.08 ″**
(1.976 → 1.897, `tools/step3_leon_bruns_convention.py`), which is far inside its ±0.60.
That is why the cells are comparable. State it that way; do not claim they share a
convention.

**Open item this session should close**: the background-mode A/B has never been run on
Leon alone — the convention re-run changed both axes together. Run it, so Leon's
`annular` is a measured choice rather than an inherited default, exactly as
G ≤ 11 was measured for Bruns.

### The exposure decision

Only the **0.6 s and 1.2 s** tiers are used for the eclipse field, and the reason is
measured: the 0.1 s tier's unique annulus (1.38–1.57 R⊙) yielded nothing against the
fine-structure floor, its innermost recovery (2.29 R⊙) is no better than the 0.3 s tier's
(2.17), and its star list is a subset. **Leon had no bright star close in** — the
below-Sun anchor is G 7.71 at 2.17 R⊙ — so the short tiers had nothing to reach for. The
0.1 s tier retains insurance value only for a V ≲ 7 star inside 1.6 R⊙, which is Bruns'
configuration, not Leon's.

One nuance worth carrying: the full four-tier union was computed and gives ~2.44–2.58;
most of that excess traced to a single corrupted centroid (G 9.10), not to the shallow
tiers as a class. So the shallow tiers were dropped for adding noise without adding
reach — not because they were empty.

### Deliverables, matching cell 1

1. `RECORD/leon2026/` holding the same four charts, produced by a Leon copy of
   `tools/matrix_bruns/b17_charts_record.py`:
   deflection vs radius, **displacement vectors** (new — the tangential scatter is what
   separates noise from deflection), L-vs-plate-scale with both methods, and the
   **atmosphere night maps** from the M5/zenith fields.
2. The same error decomposition — stat / scale / atmosphere — with the atmosphere term
   from Leon's own night nulls (±0.33 ″ already measured; re-derive it with the corrected
   method in `b17_atmosphere2.py` and confirm).
3. A star table CSV beside the charts, as `bruns_method_star_table.csv` is for cell 1.
4. The chart rules from the spec: arrows are vectors not radial projections; lengths
   asserted at runtime; positions and vectors in one frame; the variant named in every
   title; every chart writes a versioned copy under `chart_versions/revNN_*` and
   superseded versions are never deleted.

### A structural question to decide, not assume

Bruns' EA and EB are the *same* 0.62 s exposure 51 s apart, so one master is natural.
Leon's 0.6 s and 1.2 s are *different* exposures, and the headline used a per-star union
across tiers instead. Either keep the union (Leon's own method, already validated) or
build a single master as Bruns did — the all-87 test showed an unweighted mean of mixed
exposures is photon-optimal. Decide it on its merits and record the reason; do not copy
Bruns' structure just because cell 1 has it.

## AFTER LEON

* **Mexico 2024** (`G:\Mexico April 2024\Station-1-Eclipse-Data`) — pure Method 2,
  quintic, frame-by-frame usability triage. Reduce it in the spec's format from the start.
* **Portland moon** (`J:\Eclipse data\Toby Portland data\2026-07-29`) — the L = 0 null.
* Then the three-dataset comparison table.

## STANDING TRAPS (all measured; details in the record)

* CLI merges `--set` over the interactive `MEE_config.txt`: pin every parameter.
* Pass each field's **own observation time** — a placeholder puts the refraction
  correction at the wrong altitude and manufactures fake systematics (it cost ±1.40 ″ of
  imaginary atmosphere once).
* Killing a driver does not kill its pipeline child; check for orphans.
* Match gates must exceed the largest physical displacement (Leon's anchor is 2.4–2.9 ″).
* Stage 3 still ignores `flag_is_outlier`; the union/vet/rematch/nuisance all live in
  `tools/`, not the program (ROADMAP F27).
* **Verify every patch you report as applied.** Multi-line string replacements failed
  silently three times this session and were reported as done; Douglas caught all three
  from the images. Grep for the change before claiming it.
* G: only for Leon 2026 and Mexico 2024 raw data; 2017 on I:, Portland moon on J:; write
  only to D: and the repo; venv python always.

## KNOWN DEFECTS THE RECORDS STILL CARRY

1. The coronal-model trench (tool-level preprocessing blurs the tier mean with the
   saturated core included). Fixed in the pipeline, not in the tools.
2. Rim artefacts reach the alignment when the pipeline mask is off (F29): 0.8 per frame
   against 12.8 real stars on Bruns EA. Moves the star sample, not the astrometry.
3. F28 blocks the pipeline path from replacing the tool chain, so these results are not
   yet reproducible from the exe.
