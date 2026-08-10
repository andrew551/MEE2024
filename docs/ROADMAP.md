# MEE2024 — findings, immediate fixes, and roadmap

**Status:** for discussion. Nothing here is implemented unless marked *done*. Every measurement in §1 was derived from the raw 2026-08-06 London dataset; several supersede earlier claims, and those are flagged where they occur.
**Date:** 2026-08-08, against v1.3.5
**Audience:** users and contributors. Please argue with it — several items below exist
because someone's field experience contradicted an assumption in the code.

Companion documents: [`UI_ROADMAP.md`](UI_ROADMAP.md) (the two-interface question in
depth), [`ARCHITECTURE.md`](ARCHITECTURE.md), [`bench/ERROR_BUDGET.md`](bench/ERROR_BUDGET.md).

---

## 1. What the data told us

Everything in this section was measured this week from real campaign data, mostly the
London 18-field zenith set of 2026-08-06 (9 positions × 10 × 10 s, repeated with an
offset; Askar 65PHQ + ASI533MM on an HEQ5 Pro, unguided).

### 1.1 Distortion repeatability — measured for the first time

Sixteen independent fits of one optical train over 42 minutes, compared as *evaluated
displacement* rather than raw coefficients (nonlinear terms only, so plate scale and roll
cannot contaminate it):

| radius | mean d(r) | scatter |
|---|---|---|
| 500 px (15.6′) | 0.0251″ | **1.91 %** |
| 1000 px (31.2′) | 0.2000″ | **1.89 %** |
| 1400 px (43.6′) | 0.5483″ | **1.88 %** |

**1.9 %, flat with radius.** The scale-invariance says this is an amplitude wobble, not a
change of shape: the distortion *form* is stable and only its overall size moves.

This prices the distortion-zone framework, which had been running on an assumed 1–2 %
drift. The assumption was right, at the top of its range. Two caveats: it is a
*within-session* lower bound (42 minutes, one focus, no disassembly), and it is inflated
by §1.3 below, so the true figure may be better.

Plate scale over the same 16 fits: **1.869110″/px ± 13 ppm** (7 ppm relative).
Stage-2 rms: mean 67.7 mas, range 50.3–83.3.

### 1.2 Two fields in eighteen failed to plate solve

`P2_Z7_col1_top` and `P2_Z9_col1_bottom`, at 00:35 and 00:39, the two lowest-altitude
fields (73.5° and 72.0°). Pass 2 is a **half-step interleave** of pass 1 (−1.44° RA,
−0.9° Dec, confirmed from the headers), so these are fresh sky, not repeats — an earlier
draft wrongly described them as positions that had solved forty minutes earlier.

Centroid counts (1200, 1262) were not low; `P2_Z8_col1_mid` solved with 1159 at a similar
altitude. Airmass differs by only 4% and their sky background is marginally *lower*. §1.9
has the actual failure mode.

**Finding it required opening 18 archives and noticing two were missing.** See F3.

### 1.3 The date guesser ran because folder mode cannot read the header

All 18 fields carry `observation_date_header: 2026-08-05`, correct and present. Header
mode *was* selected. It silently did not apply — see I0: in batch mode `build_options`
runs before any field is discovered, so `spec['lights']` is empty, `read_observation_date`
is never called, and the run logs *"no date in the FITS header"* about frames whose headers
carry one. The guesser ran anyway:

| field | guessed | error |
|---|---|---|
| P1_Z3_col1_top | 2026-12-23 | **+140 d** |
| P2_Z3_col3_bottom | 2026-11-09 | +96 d |
| P1_Z5_col2_mid | 2026-10-07 | +63 d |
| P1_Z7_col3_bottom | 2026-06-04 | −62 d |

Mean absolute error **38 days**, worst 140, against the "two to four weeks" the UI claims.
The pipeline *detected* every disagreement (`date_guess_error_days`) and proceeded.

Cost on this dataset is small — 33 days of proper motion adds ~2 mas against a 68 mas rms —
but the 16 fields were fitted at 16 different epochs spanning six months, which adds
scatter to §1.1. On a 30–40 mas precision epoch it would matter, and it is free to avoid.

### 1.4 The drift is periodic error, measured: ±7.7″ at ≈537 s

Per-frame translation measured for all 18 fields by phase correlation on dark-subtracted
frames, paired with each frame's `JD_UTC`. Fitting the signed rate along its dominant axis
against time:

| | |
|---|---|
| Best-fit period | **537 s (8.9 min)** — the HEQ5/EQ6 worm range is 479–638 s |
| Position amplitude | **±7.7″ (15.3″ peak-to-peak)** |
| Variance explained by one sinusoid | 63% |
| Constant offset | +0.14″/min → **~0.5′ polar error** |

Total excursion per 95 s burst runs 0.8–6.5 px; per-frame steps 0.09–0.77 px. The drift
direction flips sign roughly every four fields, visible by eye in the time series.

So both terms are now measured rather than inferred: **periodic error dominates, and the
polar alignment is excellent** (~0.5′, matching SharpCap's claim). Polar misalignment alone
could not produce this — 1′ gives only 0.023 px per 10 s frame, and reaching the observed
drift would need 21′ to 1.4°.

**Consequence for eclipse day:** periodic error is ±7.7″ whatever the polar alignment, so
the worse alignment expected on the day matters much less than feared. At ladder exposures
of 0.1–2 s the peak-slope trailing is ~0.18″ (0.1 px) — not a limiting term. Even 30′ of
polar error would leave 0.14 px of trailing and ~25 px of cumulative drift, the latter
removed by alignment.

**Does the drift hurt the fit? No.** Correlating per-field drift rate against stage-2 rms
over the 16 successful fits gives **r = −0.200** (n = 16, p ≈ 0.46); the high-drift half
averages 64.6 mas against 70.7 for the low-drift half. Trailing of 0.66 px on a ~2.5 px PSF
raises centroid σ by 1.7% along one axis — far below the field-to-field scatter. Knowing
the periodic error would not improve the rms; the alignment already measures and removes
the inter-frame part, and the intra-exposure part is negligible. It would start to matter
beyond **40–50 s exposures**.

> **Method note, worth recording.** The first attempt at this measurement returned ~0.00 px
> and I reported it as a null result. It was wrong: phase correlation on *raw* frames locks
> onto the 9,308 detector-fixed hot pixels and pins the answer to zero. Validation by
> injecting a shift into the whole frame cannot detect this, because it moves stars and hot
> pixels together. Subtracting a master dark first gives the true answer, which then matches
> the pipeline's own star-based `TWOD_RESIDUALS` plots. Anyone re-deriving drift from
> outside the pipeline will hit the same trap — which is also the argument for I6.

### 1.5 Alignment: the largest known removable error, already measured

From `ERROR_BUDGET.md`, on well-tracked data: a per-frame affine removes **0.0786 px of the
0.0849 px** translation-aligned floor. Translation-aligned stacking "barely helps"
(0.0895 → 0.0775 px at N = 5, against 0.0400 if it averaged as 1/√N); affine-aligned
residuals follow 1/√N cleanly (0.0339 → 0.0134 px).

**The pipeline still aligns with a pure translation.** This is the single biggest accuracy
item on the list, and the evidence for it already exists.

### 1.6 Dither: mild, free, and unevenly distributed

The periodic error acts as an unplanned dither of 0.8–6.5 px. **Six of 18 fields fall below
the 3 px floor** the dark-free hot-pixel search requires — so a session without darks would
silently lose hot-pixel rejection on a third of its fields, decided by worm phase. More
dither is weakly associated with better rms (corr(span, rms) = −0.155; 66.6 mas above 3 px
against 69.5 below), but that is suggestive, not significant.

Ideally you would want **20–50 px**, which costs 1–2% of a 3008 px field. Deliberate polar
misalignment is the wrong way to get it: reaching 20 px of drift needs ~85′ of error, which
trails each 10 s exposure by ~2 px. Dither between exposures costs no trailing at all.

**Settling is not the obstacle it was assumed to be.** After each slew plus a 10 s delay the
first inter-frame step averages 0.401 px against 0.428 px steady-state — a ratio of 0.94.
The mount is settled by the first frame, so dithering is mechanically practical.

### 1.7 Smaller measured facts

- **The data is 14-bit unshifted, so `EGAIN` is the right gain — not `EGAINSAV`.** Despite
  `BITSHIFT = -2`, stars and hot pixels saturate at exactly **16383** and nothing exceeds
  it; the flat sits at 50.6% of 16383 (exactly "peaks mid-range" as the capture script
  instructs, against an implausible 12.7% on a ×4 scale); and measured read noise
  1.38 ADU × **1.0341** = 1.43 e⁻ matches the header's `RDNOISE` 1.63, where `EGAINSAV`
  would give 0.36. *An earlier draft of this document claimed the opposite.*
- **`BIASADU` is unusable as written.** The header says 393.59; the measured bias is
  **94.66** (10 s darks) and **94.26** (0.35 s flat-darks) — a factor of ~4.16 out.
  Subtracting it as a scalar from a flat would inject a 3.6% error. Use a measured bias.
- **Darks are labelled as lights.** Both the FITS (`FRAMETYP`, `IMAGETYP`) and the SharpCap
  sidecar (`FrameType`) say `Light` on dark frames. Any header-based auto-classification
  would silently treat 50 darks as a light field. Only `OBJECT = 'DARK_10s'` distinguishes
  them. **Fixable at capture time, for free — see Q4.**
- **Flat-darks have nowhere to go.** The pipeline uses flats raw. The bias pedestal
  (393.6 ADU on a ~30000 ADU flat) dilutes the PRNU correction by 1.31 %, leaving a
  ~6.5 × 10⁻⁵ gain residual — negligible. `BIASADU` is in the header, so the scalar fix
  needs no calibration frames at all.
- **Event volume is small.** One field ≈ 38 events, 1.83 MB (1.74 MB of it preview PNGs).
  Twenty fields ≈ 760 events against a 20 000 cap — no scaling concern at campaign sizes.

### 1.8 Calibration frames, characterised from the raw data

Measured from the 50 darks, 50 flats and 50 flat-darks of 2026-08-06 (ASI533MM, uncooled,
gain 101, 20 °C).

**The darks contain almost no noise — only defects.**

| | |
|---|---|
| Dark current | **+0.020 ADU = 0.021 e⁻ in 10 s** (0.002 e⁻/s) |
| Read noise | 1.43 e⁻ |
| Random noise in a light, (L1−L2)/√2 | 6.29 ADU, against 6.62 predicted from sky shot + read |
| Background σ, raw → dark-subtracted | 7.41 → **7.03 ADU** (a 5% gain) |

The lights are **sky-shot-noise limited** and the budget closes to 5%. Dark subtraction is
not a noise-reduction tool here. What the darks are for is the defect map: **13,290 px above
the 20σ default** (0.147%), of which ~57 rail at 16383 in every frame and are therefore
unrecoverable by any subtraction. **Mask, do not subtract** — and note defect amplitudes
track temperature (the lights ran 27.2 °C → 20.0 °C against darks at 20.3 °C), which
subtraction cannot follow but a location mask does not care about.

**Defect amplitude is `pedestal + rate × t`, not `rate × t`.** Using the 10 s darks and
0.35 s flat-darks as a two-point ladder, the measured ratio is **0.0513** where pure
exposure scaling predicts 0.035 — consistent across four decades of amplitude. Solving,
P = 0.172 R: the pedestal is 1.7% of a defect at 10 s but **63% at 0.1 s**. So a defect
retains 2.7× more signal at 0.1 s than naive scaling suggests, and **tiers cannot be
extrapolated from one another in either direction**.

**Flat-darks are redundant.** Subtracting the flat-dark changes the normalised flat by
**87 ppm** (robust; 121 ppm rms), against 80 ppm predicted by the dilution argument
bias/(bias+signal) × PRNU = 94.3/8290.5 × 0.707%. That is 1.2% of the PRNU, worth ~0.1–0.25
mas on a centroid. The 10 s darks already measure the bias to within 0.4 ADU of the
flat-darks, so the information is already in hand. The argument generalises to the cooled
2600MM/6200MM, where dark current is lower still.

### 1.9 How plate solves actually fail

From a run log for the two failing fields:

```
failures:   4 triangles,   3 stars matched (need 13) — rejected
successes:  101 triangles, 99 stars matched (need 10) — accepted
```

The failure is in **candidate generation, not verification** — the solver never finds the
true pose rather than finding and rejecting it. Given the same centroids, the pose is found
and accepted with 100 stars against a threshold of 12, under **both** G<13 and G<15, so
neither catalogue depth nor the verification threshold is implicated.

It is marginal and set-sensitive: the same field succeeds or fails depending on small
differences in the centroid list (780 vs 794 hot pixels masked; 1262 vs 1265 centroids).
Since triangles are built from the brightest handful, one spurious detection in the anchor
set is enough. That is consistent with the reported higher failure rate under light
pollution, where spurious detections are more likely. Only four candidates are generated
across the whole failure ladder, which suggests the escalation gives up early.

**Resolved, once the provenance existed.** Rerun under v1.3.6, the two fields report:

```
pattern_db      : patdb_g13_t17k        layers_used : ["patdb_g13_t17k"]
verify_catalogue: gaia_offline G<13.0   noise_px_used: 0.9   anchor_rounds_used: 2
```

Only **one** pattern layer is installed on that machine. When the same fields solved
elsewhere the run used `patdb_g12_t17k` *plus* `patdb_g13_t06k` and `patdb_g12_t40k`, and
found the pose. So the escalation ladder is not giving up early by design -- it has
nothing to escalate into. `noise_px_used` of 0.9 (3x the 0.3 default) and two anchor
rounds show it exhausted what it had.

**Action:** run `mee2024 build-pattern-db` on whichever machine reduces the campaign data,
confirm the extra layers appear and that these two fields then solve, and make the full
layer set a pre-flight check. `results.txt` now records which layers were available, so it
is verifiable rather than assumed. Treat this as confirmed only after that test.

### 1.10 A caution on the acceptance test

*Raised by Douglas in his v1.3.5 verification pass, and it corrects his own earlier
prediction.*

The original acceptance test — reprocess the London 2026-07-19 set and expect global rms
0.19″ → ~0.09″ — assumed the epoch fix **and** the astrometric corrections (G2/F7) would
land together. Only the epoch half exists, so the test can no longer be run as a unit:

- **Testable now:** the residual-versus-proper-motion correlation should collapse. Clean,
  falsifiable, single-variable — a better experiment for having been isolated.
- **Testable now, separately:** defect masking on the faint bin, and now also *without*
  darks via `persistence_mask`, which did not exist when the prediction was written.
- **Not testable until corrections land:** the paired ± linear distortion coefficients
  cannot move while `enable_corrections` is False. **If they do not move, that is not a new
  error term — it is the corrections missing.** The original document said "anything that
  refuses to move is a genuinely new error term"; for this quantity that inference is
  currently wrong.
- **Therefore:** expect the global rms to land short of 0.09″, and do not read the
  shortfall as a failed prediction.

---

## 2. Immediate fixes

Small, individually verified, no design work needed. Roughly in order of consequence.

| # | Fix | Why |
|---|---|---|
| ~~**I18**~~ | ~~**Batch-level files overwrite each other**~~ | *Done.* `batch_summary.csv`, `batch_summary.json` and `activity.jsonl` were written to the *output root*, so a second batch pointed at the same output folder silently destroyed the first batch's summary and log -- the very records that say what happened. Per-field results survived; the batch-level record did not. Introduced 2026-08-09 with F3/F4, fixed by I19 on 2026-08-10. |
| ~~**I19**~~ | ~~**Create an output subfolder named after the input folder**~~ | *Done, 2026-08-10.* `batch.run_output_root` puts each run in a subfolder of the chosen output folder, named for the input folder (`.../Zenith`), with the field tree inside; the timestamp is appended **only** when a previous run's records are already there. It applies to single runs as well as batches -- the log collided the same way -- taking the name from the folder holding the first frame, and the settings file still remembers the folder the *user* picked, so runs do not nest. In the same pass, the stage-2 archive name stopped reciting the stage-1 timestamp twice: 89 characters down to 49, with a working folder of the same length beside it, which matters against the 260-character Windows limit once a field tree sits above them. The stage-1 path is now recorded inside `distortion_results.txt` instead. |
| **I0** | **Resolve the observation date per field in batch mode** | `_work` calls `build_options(spec)` once, before any field is discovered, and the header branch reads `spec['lights']` — which is empty in batch mode. So **header date mode has never worked in folder mode**: it silently falls back to guessing and logs "no date in the FITS header" about frames that have one. Folder mode is the intended default for real data, and §1.3 is the measured consequence. Resolve the date inside `_run_fields`, where the frames are known. |
| **I1** | `encoding='utf-8'` on the log `FileHandler` | Confirmed cp1252 on Windows. The log writes `str(files)` verbatim, so a `ł`, `ř`, `ğ` or CJK character in any capture path raises `UnicodeEncodeError` **mid-run**. One word. |
| **I2** | Calibration controls out of `single-input` | Darks and flats picked in single mode stay selected and **are still applied to every field** in folder mode, invisibly, with no way to see or clear them. Hidden state, and the most dangerous item here. |
| **I3** | Record the **effective options**, calibration and `source_folder` in the results JSON | Today a run's own result file cannot tell you whether it was calibrated, nor which star-selection flags were in force. `remove_double_tab2` is absent, and `n_dropped_double` is *computed and emitted on the event bus* before being discarded at the JSON boundary. Two fits that differ only in a default are indistinguishable afterwards — which is precisely what §1.1 compares. |
| **I4** | Master dark/flat headers | They record only `NCOMBINE`, `COMBTYPE`, version. No `EXPTIME`, `GAIN`, `CCD-TEMP`, `CAMID`, `BITDEPTH`. A master that cannot identify itself cannot be safely reused — which is the only reason to save one. |
| **I5** | `find_fields` must skip calibration folders | Any folder with ≥3 frames is treated as a light field, so a `DARK_10s` folder under the batch root is stacked and plate-solved, and fails. |
| **I6** | Persist per-frame shifts, rms and dither span | Computed, then sent only to a `print()` and a plot. Needed for drift (F5) and for judging a stack. |
| **I7** | Persist `deltas` (the 2-D residuals) as CSV | Currently rendered to a 600 dpi PNG and discarded. It is a picture of a measurement rather than the measurement. |
| **I8** | Widen `_field_metrics` | FWHM, ellipticity, plate scale, `n_frames` and the pointing check are all computed, pass through the event stream, and are dropped. `nn_corr` is collected and never displayed. |
| **I9** | Epoch precedence **inside the fitter**, plus a real CLI date option | The header-date fix lives in one front end's options assembly, so correctness depends on which of three interfaces was used — and the CLI, the path most likely to be scripted and left unattended, still falls back to the `2023-12-01` default. `distortion_fitter.py:240` resolves the epoch from `guess_date` alone and cannot reach a header value. Stage 1 already writes `observation_date_header` into the data that travels to stage 2, so resolving it there makes the guarantee structural rather than per-front-end. §1.3 is the consequence of not doing this. |
| **I10** | Emit `nn_r`, `max_star_mag_dist`, `distortion_fit_tol` | `nn_corr` is unreadable without its companion distance; the other two are needed to label a summary table. |
| **I11** | Release pattern-DB memory maps | Same family as the catalogue mmap fix — a mapped file cannot be deleted while the app runs. |
| **I12** | Classic UI `initial_folder` guard | Three unguarded sites fall back to the process CWD. Low priority: the classic UI is frozen. |
| **I13** | Dead `clipped` counter | The "exceeded the 16-bit container" warning can never fire. Benign — the float32 fallback protects the data — but it reads as an active check. |
| **I15** | `text_select=True` on the app window | pywebview disables text selection by default (`create_window(..., text_select=False)`), so the activity log cannot be selected or copied — bug reports currently require screenshots. Nothing in the CSS blocks it; one keyword argument in `ui/app.py`. |
| **I16** | Solve provenance in `results.txt` | Which solver ran, which pattern database, which verification catalogue, how many candidates were tried and the best match count. Three separate questions this session died for want of it, including §1.9. |
| **I17** | Make the defect threshold absolute | `dark_mask` scales its cut to the *master's* robust sigma, which shrinks as 1/√N — so 20σ is 5.3 ADU with 50 darks and 8.7 ADU with 10. **Taking more darks silently masks more pixels** and changes which stars survive. Set it in ADU or e⁻ above bias, from what actually perturbs a centroid (a 10 ADU defect 2 px from a 1000 ADU star pulls it ~37 mas); let N affect only the confidence. |
| **I14** | Close the second `logging.FileHandler` | `database_lookup2.py:40` attaches a handler to a logger that never passes through `close_logger` — the same pattern as the run-log leak, missed by our sweep because it is in the catalogue reader rather than the pipeline. Dormant today (`debug_folder` is `None` at every call site), so it locks nothing; it will bite the first person to enable debug logging, who is by definition already diagnosing something else. *Found by Douglas.* |

**Done this week:** `erfa.ld` restored on the error path; run log released when the run
ends; file dialogs remember their own folders; four star names corrected and the list
extended to 128; `gaia_dr3_g15` repaired (992 251 missing stars), published and gated
behind a size confirmation; superseded archives hidden with a one-time cleanup offer;
catalogue removal from the app; release tags corrected; `main` brought up to date.

---

## 3. Feature roadmap

### F1 — Calibration library (darks and flats batch modes)

There is no way to build a dark library or a master flat except as a side effect of a light
run. Proposal: a mode selector (Lights / Darks / Flats) rather than three peer buttons,
because catalogue, date, distortion order, tolerance and magnitude are all meaningless for
calibration frames and should disappear.

Darks and flats skip centroiding, solving and fitting — but **not** the bit-depth check,
the provenance header (I4), or a pedestal check against `BIASADU`. They should additionally
produce the **per-pixel standard deviation**, which is the only thing that finds
telegraph/RTS pixels; the mean finds only hot ones.

*Must stream.* The current `np.mean(np.array(open_images(...)))` materialises the whole
cube: **3.6 GB** for 50 × 3008², **10.4 GB** for a 2600MM. Welford needs two buffers
regardless of frame count and yields the std for free.

**Three simplifications the measurements in §1.8 allow:**

- **No flat-dark input.** Worth 87 ppm, and the darks already carry the bias. Drops one UI
  input and ~9 minutes from every session.
- **No tier synthesis / rate map.** Because defect amplitude is `pedestal + rate·t`, tiers
  cannot be extrapolated. But at eclipse-ladder exposures darks are nearly free — 50 frames
  at each of 0.1/0.3/0.6/1.2/2.4 s is ~4 minutes of integration — so **capture every tier
  directly** and delete the synthesis machinery from the design.
- **Absolute defect threshold** (I17), not one that moves with the number of darks.

**How many darks:** 3–5 for the bias, 10–20 for a defect map, **40–50** only if you also
want per-pixel σ for RTS/telegraph detection — which is the one thing many frames uniquely
buy, and the defect class that defeats subtraction entirely.

**Library key:** camera | gain | offset | binning | shape | exposure. Temperature joins the
key *only* when the header carries a setpoint (`SET-TEMP`); for an uncooled body it is
measured, not chosen, so keying on it means never matching — record it and warn instead.
Flats key additionally on `FOCUSPOS`, `TELESCOP` and filter, so a changed optical train is
*detected* rather than silently averaged.

**The payoff is larger than the feature.** Once a library exists, the lights batch points at
the library root and matching becomes automatic per field by exposure. That supersedes I2's
picker, dissolves the hidden-state problem (the match is reported per field), and solves the
eclipse ladder case. **Build the library first, then the consumer.**

### F2 — Affine per-frame alignment

Replace translation-only alignment with a per-frame affine. §1.5 is the evidence; the
mount-off dataset is the extreme validation case. Expect it to help every dataset, not just
badly tracked ones.

### F3 — Batch summary file

`batch_summary.csv` + `.json`, one row per field: source folder, output folder, status,
error text, plate-solve success, and every metric from I8. A header block carries the
run-constant parameters (distortion order, magnitude limit, fit tolerance).

This is how §1.2 should have been found. It also serves cross-run collation and the
repeatability analysis, both of which are currently manual.

### F4 — Persist the activity log

The app's event bus has one sink, `ListSink`, in memory, cleared at the start of each run.
The batch-level narrative exists nowhere on disk. `events.JsonlSink` already exists and is
used by the CLI; the app simply never wires it up.

### F5 — Drift measurement

Drift rate in arcsec/min from the per-frame shifts (I6) plus plate scale plus **per-frame
timestamps**, which the pipeline does not currently read — `read_observation_date` reads
frame 0 only, though `JD_UTC` and `DATE-AVG` are in every header.

Then: separate periodic error from polar misalignment by their signatures (sinusoid at the
worm period versus a monotonic ramp), and cross-check drift-predicted trailing
(0.7 × L/FWHM) against the measured `psf_ellipticity`. That last one turns ellipticity from
a number into a diagnosis — it distinguishes tracking error from focus.

Report `dither_px` alongside: it says both how good the alignment was *and* whether the
dark-free hot-pixel search was possible, since that declines below 3 px of dither.

### F6 — Open a previous run from disk

Reconstruct the panels — stacked preview, star labels, distortion field, advanced analysis —
from a run's output folder, all of which is already on disk. Serves the batch table
(click a row, see its graphics), single mode, and runs from weeks ago. Currently the
graphics appear for a moment during a batch and are then unreachable.

### F7 — Header harvest

`JD_UTC` (mid-exposure epoch, better than the frame-start `DATE-OBS` used now),
`OBSLAT`/`OBSLONG` (unlocks the astrometric corrections without manual entry),
`EGAINSAV` and `RDNOISE` (a measured rather than assumed noise model),
`CAMID`/`GAIN`/`OFFSET`/`CCD-TEMP` (the F1 library key), `FOCALLEN` + `XPIXSZ`
(a 0.3 %-accurate plate-scale prior — see F8).

From the SharpCap sidecar, and available nowhere else: `Subtract Dark`, `Apply Flat`,
`Background Subtraction`, `Banding Suppression` — proof the pixels are unmodified. If any
were on, the frames are already calibrated and the pipeline's assumptions break silently.

### F8 — Solve fallback from the header

Neither solver accepts a prior; both are lost-in-space. Two levels:

1. **Plate-scale prior** from `FOCALLEN` + `XPIXSZ` — 0.28 % accurate on the 65PHQ, and it
   collapses the FOV ladder that the v2 solver escalates through on failure. Cheaper, and
   robust: focal length and pixel size are reliable even when the mount is not.
2. **Pointing prior** to restrict the anchor search. More work.

Caution: the *dark* frames also carry `RA`/`DEC`, because the mount was parked somewhere.
Presence proves nothing; any prior needs a sanity gate and must fall back to a blind solve.

### F10 — Multi-select folders in batch mode

Batch mode takes one root and processes everything beneath it, so reducing an arbitrary
*subset* -- two fields out of eighteen, say, after a rerun -- means running them one at a
time. Ctrl-click selection of several folders is the standard Windows idiom and the
obvious fix.

The native dialog already supports it: pywebview's `create_file_dialog` takes
`allow_multiple`, which is not restricted to files, but `ui/app.py` currently forces it
off for directories (`allow_multiple=bool(multiple) and not directory`). Removing that is
small. The real work is elsewhere: the in-page picker tracks a single `picker.cwd` and
needs per-row selection like it already has for files; `find_fields` must accept a list of
roots; and `output_dir_for` needs a sensible `relative` when the chosen folders share no
parent -- which interacts with I19's naming.

### F9 — UI convergence

See [`UI_ROADMAP.md`](UI_ROADMAP.md). In short: the classic interface is frozen (0 commits
since the app window landed, against 19), but cannot be retired, because the app window
cannot run stage 3, cannot enable the astrometric corrections, and — as of this week's
review — has none of the blob/corona controls that eclipse-day processing depends on.
Those three are one deliverable: *the app window can process eclipse data*.

---

## 4. Open questions — for the group, not for the code

**Q1. Does the eclipse campaign need stage 3 in the GUI?** The error budget and benchmarks
already run stage 3 from the CLI. If that is acceptable on the day, F9 is a convenience
rather than a blocker and the corrections become the urgent half. This changes the priority
order more than anything else here.

**Q2. Double-star rejection is now on by default** (v1.3.5), which means a fresh install and
an upgraded machine can disagree until settings are aligned. Decided: notify by bulletin,
`mee2024 config --set remove_double_tab2=True`. Revisit if the user base grows.

Independently raised from the user side, with the sharper framing that a silent selection
difference would be *indistinguishable from* the epoch-to-epoch instability the
repeatability study exists to measure. I3 is the durable half of the answer: pin the flag,
and record it in the output so a comparison can verify rather than assume. Note also that
the release-note line "nothing here changes how a fit is computed" is in tension with this
change — reconcilable only by reading three bullets together, and worth wording better next
time.

**Q3. Are flat-darks worth taking?** Testable with data already in hand: subtract the
`FLATDARKS_matched` stack's own median and look at the residual. A few ADU rms of
unstructured noise → the scalar `BIASADU` is enough and they can leave the session chain.
Banding → keep them.

**Q4. Set the frame type at capture.** ~~Darks currently record `FRAMETYP = 'Light'`.
Setting it correctly in SharpCap costs nothing today.~~ **Answered, and the premise was
wrong:** the SharpCap Sequencer has no frame-type command, so a scripted capture cannot set
it at all (stated in `leon_darks_v1.4.scs`, 2026-08-09). The workable answer is the one the
Leon scripts adopt -- put the type in `TARGETNAME`, which lands in `OBJECT`, and key
calibration matching on `GAIN` + `EXPTIME` + `OBJECT`, never on `IMAGETYP`. That also makes
the capture folders self-labelling (`DARK_G0_1p2s`, `DARK_G101_4s`), which is what F1's
library key actually needs. See [`LEON_SCRIPT_REVIEW.md`](LEON_SCRIPT_REVIEW.md) §1.

**Q5. How spread are the zenith positions?** Tightly clustered means the polar term is
effectively constant and F5's decomposition is simple; widely spread gives better geometric
leverage but needs a joint fit.

**Q6. What is the tuned flat exposure?** The §1.6 argument scales with it. At 0.5 s the
dark contribution is unobservable; by 5 s the warm-pixel term reaches ~1 % and the
conclusion changes.

---

## 5. Reviewed and closed without change

Recorded so they are not re-raised. Each was reported in good faith and checked against the
source; the checking is the point, not the outcome.

**Picker memory / `precheck_files`.** Reported as: `write_ini` only runs when a run fails,
so a successful run never records its input folder. The source says otherwise — the
`if not good_tasks` test sits *before* `good_tasks.append(file)`, so it means "this is the
first good file", not "the run failed". The write is on the success path. Diffing the
function between the original baseline and v1.3.5 shows only whitespace and
`except:` → `except Exception:`; the logic never changed, so this was a misreading rather
than a fix that landed. (The provenance was blocked by a one-character typo in the quoted
baseline SHA — `0d294a6b**3**a…` for `0d294a6b**7**a…` — which also explains an apparent
404 on the commit. It is intact and still an ancestor of `main`.)

A *different* picker bug was real and is fixed: the file and folder dialogs shared one
last-visited folder, so choosing an output folder re-aimed the next input dialog.

**Bare `mee2024` said to exit with an argparse error.** `cli.py` does declare its
subparsers `required=True`, but argparse is never reached: the console entry point is
`mee2024.main:main`, which intercepts an empty argv and calls `run_default_interface()`
first. Documentation and behaviour agree. The report came from reading `cli.py` in
isolation, which is a reasonable thing to have done.

**Full `JD_UTC` → `DATE-AVG` → `DATE-OBS`+EXPTIME/2 chain for the catalogue epoch.**
Withdrawn by its own author on the arithmetic: on a 10 × 10 s run the first-frame versus
stack-midpoint difference is ~50 s against a proper-motion lever arm of years. Day
resolution is sufficient *for the epoch*. It is **not** sufficient for refraction, which
needs the time of day — so F7 still wants it, for a different reason.

---

## 6. Suggested order

1. **I0 first** — header date in folder mode: it silently defeats the default on the mode meant for real data. Then **I1–I5, I14–I17** — the one-liners, the hidden-state fix, and the provenance. Small, independent, protect data
   being captured now.
2. **F1** — the calibration library, because F1 supersedes part of I2 and should be built
   before its consumer.
3. **F3, F4, I6–I8** — summary file, event log, the metrics that already exist. These make
   folder mode reportable, which is what it is missing.
4. **F2** — affine alignment. The largest accuracy item, and the one to validate against the
   tracking-off dataset.
5. **F5–F8** — drift, open-from-disk, header harvest, solve fallback.
6. **F9** — UI convergence, gated on Q1.

Nothing here is urgent enough to interrupt field testing of v1.3.5. Items I1–I5 are small
enough to ride along with whatever that testing surfaces.
