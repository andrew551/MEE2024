# MEE2024 — findings, immediate fixes, and roadmap

**Status:** for discussion. Nothing here is implemented unless marked *done*. Every measurement in §1 was derived from the raw 2026-08-06 London dataset; several supersede earlier claims, and those are flagged where they occur.
**Date:** 2026-08-08, against v1.3.5; §2 and F1/F10 updated 2026-08-17 for v1.3.7;
F11–F13 and §6 updated 2026-08-19 for **v1.3.9**

> **§2 is closed entirely, plus F1, F3, F4, F10 and F11.** The whole immediate-fix list is
> done, the run log and batch summary are reachable from the app window, batch mode takes
> several folders, the calibration library exists and has been built from the real Leon
> campaign data, and SER files are read directly. See
> [`LEON_2026-08-11.md`](LEON_2026-08-11.md) for what that data says. What remains open is
> F2, F5–F9 and F12 — features needing their own validation rather than fixes. §6 now says
> which release each one lands in, and why the line is drawn where it is.
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

## 2. Immediate fixes -- all closed

Small, individually verified, no design work needed. **Every item on this list is now done**;
the table records what each was and where it landed, because the reasoning is worth keeping
and because several of them describe traps a future change could walk back into.

I0, I3, I6, I8, I15 and I16 shipped in **v1.3.6**; I18-I20 landed just after it, in source
only, and reached an executable for the first time in **v1.3.7** along with the rest.

| # | Fix | Landed | What it was |
|---|---|---|---|
| **I0** | Observation date per field in batch mode | v1.3.6 | `build_options` ran once, before any field was discovered, so the header branch read an empty `spec['lights']`. **Header date mode had never worked in folder mode**: it fell back to guessing while logging "no date in the FITS header" about frames that had one. Measured cost on the London 18-field set: 38 days mean epoch error, 140 worst, across sixteen epochs spanning six months. Resolved per field in `_run_fields`, where the frames are known. |
| **I1** | `encoding='utf-8'` on the log `FileHandler` | v1.3.7 | Confirmed cp1252 on Windows. The log writes `str(files)` verbatim, so a non-cp1252 character in any capture path raised `UnicodeEncodeError` **mid-run**, from the logging call rather than from anything to do with the data. One keyword. Done in `database_lookup2` too. |
| **I2** | Calibration controls out of `single-input` | v1.3.7 | Darks and flats picked in single mode stayed selected and **were still applied to every field** in folder mode, invisibly, with no way to see or clear them. `buildSpec` now sends neither in batch mode; folder runs take a **calibration library** instead (F1), matched per field on gain and exposure and reported per field. Hand-picked frames still win when given. |
| **I3** | Effective options, calibration and `source_folder` in the results JSON | v1.3.6 | A run's own result file could not say whether it was calibrated, nor which star-selection flags were in force -- so two fits differing only in a default were indistinguishable afterwards, which is exactly what the repeatability study compares. |
| **I4** | Master dark/flat headers | v1.3.7 | They recorded only `NCOMBINE`, `COMBTYPE` and a version. A master that cannot identify itself cannot be safely reused, and reuse is the only reason to save one. `save_calibration_stacks` now writes the same provenance block the library uses: exposure, gain, offset, both temperatures, camera, optical train, and the folder it came from. |
| **I5** | `find_fields` skips calibration folders | v1.3.7 | Any folder with frames was treated as a light field, so a `DARK_10s` folder under the batch root was stacked and plate-solved, and failed. Recognised by folder name and then by `OBJECT` -- **never** `IMAGETYP`, which reads `'Light'` on every scripted dark because the capture sequencer cannot set it. The count is reported, not silent. On the Leon tree this is 8 dark tiers and a flat set. |
| **I6** | Persist per-frame shifts, rms and dither span | v1.3.6 | Computed, then sent only to a `print()` and a plot. |
| **I7** | Persist `deltas` (the 2-D residuals) as CSV | v1.3.7 | Rendered to a 600 dpi PNG and discarded -- a picture of a measurement rather than the measurement, so it could not be re-binned, correlated against drift, or compared between epochs. Now `TWOD_RESIDUALS.csv`: position, dx/dy in pixels and arcseconds, radius, magnitude and ID. |
| **I8** | Widen `_field_metrics` | v1.3.6 | FWHM, ellipticity, plate scale, `n_frames` and the pointing check were computed, passed through the event stream, and dropped. |
| **I9** | Epoch precedence **inside the fitter**, plus a real CLI date option | v1.3.7 | The header-date fix lived in one front end's options assembly, so correctness depended on which of three interfaces was used -- and the CLI, the path most likely to be scripted and left unattended, still fell back to `2023-12-01`. `distortion_fitter.resolve_epoch` now decides it where the data is: an explicit date wins, then the header value stage 1 recorded, then the guesser. A disagreement between an explicit date and the header is reported. `--date`, `--date-from-header` and `--guess-date` join the CLI. |
| **I10** | Emit `nn_r`, `max_star_mag_dist`, `distortion_fit_tol` | v1.3.7 | `nn_corr` is unreadable without its companion distance -- 0.3 at 50 px and 0.3 at 500 px are different findings -- and the other two are needed to label a summary table. |
| **I11** | Release pattern-DB memory maps | v1.3.7 | A mapped file cannot be deleted on Windows, so a pattern database could not be removed or rebuilt in place while the app ran. `pattern_db.release_databases()` drops the maps *and* the KD-tree built over them, since the tree holds the largest array. Called before any rebuild, and from `release_catalogues`; deliberately **not** after every run, because the tree is expensive and watch mode solves field after field. |
| **I12** | Classic UI `initial_folder` guard | v1.3.7 | Nine unguarded sites fell back to the process CWD, which for a double-clicked executable is wherever Windows launched it. One helper resolves the first candidate that exists, falling back to the home directory, and returns `None` rather than `''` for "no preference". |
| **I13** | Dead `clipped` counter | v1.3.7 | The "exceeded the 16-bit container" warning could never fire: the count was taken in the branch where `max <= 65535` holds by construction. Moved to the branch it describes, and reworded -- the float32 fallback means the values were *kept*, not clipped. |
| **I14** | Close the second `logging.FileHandler` | v1.3.7 | `database_lookup2.py` attached a handler to a logger that never passed through `close_logger` -- the same leak as the run log, in the catalogue reader rather than the pipeline, which is why the earlier sweep missed it. `database_cache.release_catalogues` already called `close()` on anything that had one, so defining it was the whole fix. *Found by Douglas.* |
| **I15** | `text_select=True` on the app window | v1.3.6 | pywebview disables text selection by default, so the activity log could not be selected or copied and bug reports arrived as screenshots. |
| **I16** | Solve provenance in `results.txt` | v1.3.6 | Which solver, which pattern database, which verification catalogue, how many candidates, best match count. Three questions died for want of it, including 1.9 -- which 1.9 then answered. |
| **I17** | Absolute floor under the defect threshold | v1.3.7 | `dark_mask` scaled its cut to the *master's* robust sigma, which falls as 1/sqrt(N) -- 20 sigma measured 5.3 ADU with 50 darks against 8.7 ADU with 10, so **taking more darks silently masked more pixels** and changed which stars reached the fit. The cut is now `median + max(10 ADU, sigmas x sigma)`: 10 ADU is what moves a centroid enough to matter (~37 mas at 2 px from a 1000 ADU star), and N affects only the confidence. The threshold can only rise against the old behaviour, so this masks fewer pixels, never more. |
| **I18** | Batch-level files overwrite each other | src 2026-08-10, exe v1.3.7 | `batch_summary.csv`, `batch_summary.json` and `activity.jsonl` were written to the *output root*, so a second batch pointed at the same output folder silently destroyed the first batch's summary and log -- the very records that say what happened. Per-field results survived; the batch-level record did not. Introduced 2026-08-09 with F3/F4, fixed by I19. **This is the bug the v1.3.6 build Douglas tested still has.** |
| **I19** | Output subfolder named after the input folder | src 2026-08-10, exe v1.3.7 | `batch.run_output_root` puts each run in a subfolder of the chosen output folder, named for the input folder (`.../Zenith`), with the field tree inside; the timestamp is appended **only** when a previous run's records are already there. It applies to single runs as well -- the log collided the same way -- and the settings file still remembers the folder the *user* picked, so runs do not nest. In the same pass the stage-2 archive name stopped reciting the stage-1 timestamp twice: 89 characters down to 49, which matters against the 260-character Windows limit once a field tree sits above it. |
| **I20** | Per-field summary and log, inside each field's folder | src 2026-08-10, exe v1.3.7 | Raised from the field: a field folder that gets copied to a second drive or handed to whoever is reducing should say what produced it without the folder above it. Each field gets `field_summary.json` and its own slice of `activity.jsonl` -- its slice, because fifty copies of a fifty-field log is fifty times the bytes to say the same thing. A **failed** field gets one too, since the absence of a folder is not a diagnosis. The batch roll-up stays at the run root as well: it is the only file that can say which field is *missing*. |

### 2.1 Reaching the records at all -- new in v1.3.7

Not on the original list, and found while reviewing what v1.3.6 shipped: **every one of those
files was written to disk and none of them was reachable from the app window.** The single
"Open output folder" button pointed at `outputs.distortion_zip || outputs.centroid_zip`,
which in a batch is whichever field ran *last* -- so it opened one arbitrary field rather
than the run root where the summary and the batch log live -- and it appeared only when the
run finished `done`, which is exactly backwards: a failed or cancelled run is when someone
wants the log.

It now points at the run root, shows whatever the outcome, and every row of the batch table
is clickable and opens that field's own folder.

### 2.2 A container is not a field -- found on the Leon tree

Also new, and it would have wasted a whole session. `find_fields` claimed the first folder
holding frames and stopped descending, which is right for a capture folder (a `thumbnails`
subfolder must not become a second field) and wrong for a session root: the Leon root keeps
`actual leon site.JPG` beside `DARKS`, `Zenith`, `Horizon` and 1251 frames of data. Pointed
at that root, a batch found **one field, containing one photograph**, and processed none of
the session.

A folder whose own frames are outnumbered ten to one by what lies beneath it is now treated
as a container. That separates the two cases without knowing anything about file types or
naming: 30 frames against a 5-frame thumbnails subfolder stays a field; 1 against 1251 does
not.

**Done this week:** `erfa.ld` restored on the error path; run log released when the run
ends; file dialogs remember their own folders; four star names corrected and the list
extended to 128; `gaia_dr3_g15` repaired (992 251 missing stars), published and gated
behind a size confirmation; superseded archives hidden with a one-time cleanup offer;
catalogue removal from the app; release tags corrected; `main` brought up to date.

---

## 3. Feature roadmap

### F1 — Calibration library — **done, v1.3.7**

`mee2024/calibration.py`, `mee2024 calibrate`, and a library picker in the app window.
Built and validated against the real Leon campaign calibration data: **8 dark tiers and
1 master flat**, 449 frames of 26 megapixels. See
[`LEON_2026-08-11.md`](LEON_2026-08-11.md) for the numbers it produced.

**Keyed on gain and exposure, and nothing else.** Camera, binning and frame shape must also
agree, but they are identity rather than configuration. Temperature is **recorded and warned
about, not keyed** — for a cooled body the setpoint is in `SET-TEMP` and is what makes a dark
comparable, and for an uncooled one the temperature is measured rather than chosen, so keying
on it means never matching. The observer segregates by temperature; the library says so when
the frames disagree with the master.

**Never `IMAGETYP`.** The capture sequencer has no frame-type command, so every scripted dark
and flat in the Leon data records `FRAMETYP = IMAGETYP = 'Light'`. The scripts put the type in
`TARGETNAME`, which lands in `OBJECT` (`DARK_G0_0p1s`, `FLATS`) — and say so in their own
header comments, addressed to us. Classification uses the folder name first and `OBJECT`
second. This is also what I5 keys on.

**Nothing is scaled or interpolated.** A tier is matched to within 1% of its exposure or
reported as missing, with the tiers that *are* present listed. Defect amplitude is
`pedestal + rate·t` — the pedestal is 63% of a defect at 0.1 s and 1.7% at 10 s — so a tier
interpolated from its neighbours is wrong in a way that looks plausible. A field with no
matching tier runs uncalibrated and says why.

**The flat is deliberately not keyed on gain.** One gain-0 flat set corrects the gain-101
night data, on the stated first-order assumption that PRNU and vignetting do not depend on
gain. That is what the capture scripts assume; the library records the gain that was used and
notes the difference when it applies the flat across gains, so the assumption is visible
rather than buried. Flats key on camera, optical train, filter and focus — quantised, because
dust-donut geometry moves with focus but focuser jitter should not split one set into several
entries.

**No flat-dark input**, as decided. What makes that valid is the mid-range fill of the flat
itself: the unsubtracted offset pedestal is diluted by the signal, so at half scale it leaves
~0.3% at a vignetted corner — smooth, multiplicative, centroid-irrelevant — and five times
that at a tenth of scale. So the builder *checks the fill* and warns outside 25–75%. The Leon
flat measured **50.7% of full scale**, exactly where its capture script aimed, so the
argument holds for this campaign as a measured fact rather than an assumption.

**It streams, and it had to.** `np.mean(np.array(open_images(...)))` holds every frame at
once: 5.2 GB for fifty frames of the 26 MP ASI2600MM, and 10.4 GB while the list and the
array both exist — more than the machine has. One pass now accumulates sum, sum of squares,
running minimum and running maximum. The stacker's own dark and flat combination was changed
to the same code, so the memory ceiling is gone from the light path too.

**Sigma-clipping was tried first and is the wrong tool here** — worth recording, because it
is the obvious choice. Clipping needs a spread to measure against, and the only spread a
streaming pass has is the per-pixel standard deviation, which the outlier itself inflates:
one frame at 60 000 ADU among forty-nine at 500 gives that pixel a sigma near 17 800, so even
a 5σ cut keeps the spike. **Min-max rejection** — drop each pixel's own extreme high and low
frame — needs no scale estimate, comes free from the same pass
(`mean = (sum - min - max) / (n - 2)`), and discriminates on the right axis: a cosmic ray is
extreme in *one* frame, while a hot or telegraph pixel is high in *every* frame and survives
with 2 of 50 samples trimmed. It also halved the build time, since there is no second pass.

**The per-pixel σ map is written beside every master, and is deliberately untrimmed.** It is
the only thing that finds telegraph/RTS pixels — the defect class subtraction cannot touch —
and trimming each pixel's extremes is precisely what would hide them.

Still to do, and not blocking: the light path consumes the library but the Lights/Darks/Flats
*mode selector* was not built — building a library is a button and a CLI subcommand rather
than a mode, which turned out to be enough.

### F1a — What the library is for, once it exists

The payoff, and it lands with the library: a folder run points at one library and **each
field gets the tier matching its own frames, reported per field**. That is what supersedes
I2's picker. It also dissolves the hidden-state problem, because the match is announced in
each field's own log and recorded in its `field_summary.json`. Hand-picked darks and flats
still win when they are given — an explicit choice is an explicit choice.

Three decisions from the 2026-08-19 review, none of them built yet:

- **Clear the input fields after a *successful* build**, not after a failed one — a failure
  should not cost the user the folder selections they just made. Clearing is a courtesy
  rather than a safety measure: `build_library` is keyed by what the frames are, so an
  accidental second build **supersedes** a tier rather than adding a near-duplicate. The cost
  of the accident is minutes, not a corrupted library.
- **Guard the rebuild instead.** If a valid library already exists at the target path, say
  what is in it and require confirmation. That covers the case clearing does not — fields
  re-populated by hand — and it is the guard that actually prevents the accident.
- **Make calibration one three-way choice — none / manual / library — not two independent
  settings.** Today `-DARK-`/`-FLAT-` and `calibration_library` can both be set, which is
  exactly the shape that produces "which one won?" bugs; v1.3.7 already had to fix manual
  darks leaking invisibly into folder runs (I2). Two failure cases, deliberately treated
  differently: *library selected but none exists*, or it is empty or unreadable, is a **hard
  error before the run starts**, because it is a setup mistake and nothing useful can follow;
  *library exists but has no tier for this field* keeps the current **warn and run
  uncalibrated** behaviour, because on a batch of eighteen fields one missing tier must not
  abort the other seventeen.

### F11 — SER input, and choosing frames without a second copy — **done, v1.3.8**

Raised by a user whose 61 MP camera cannot sustain FITS-per-frame: 122 MB a frame at 3.2 fps
is 388 MB/s, and at 315 ms the frames are really a video. The only route in was a conversion
through PIPP, which costs a duplicate of a 15 GB file and — measured — destroys the
timestamps.

`mee2024/ser.py` reads the container; a frame is addressed as `capture.ser#42`, so the
pipeline's one-frame-one-path model survives untouched. `mee2024/framescan.py` measures a
sequence and *suggests* a usable range, and `--frames 50-172` applies one as a **run
parameter** rather than as a second file on disk. Both ends are trimmed symmetrically,
because the Sun can be at the end of the last file as well as the start of the first.

The same scan answers a different question: whether each frame's brightness matches the
exposure its header claims. Six frames of the Leon eclipse ladder do not — capture software
writes the new exposure into the header of a frame that still holds the previous one — and
nothing downstream could detect it. It reports rather than corrects, deliberately.

Full account, including three measures that looked right and failed, in
[`SER_INPUT.md`](SER_INPUT.md). Not done: a graphical frame selector.

### F12 — Settings that no interface can reach

Raised from the field: `MEE_config.txt` used to sit beside the executable and the source. It
now resolves through `platformdirs` to
`AppData\Local\MEE2024\MEE2024\MEE_config.txt` — doubled, because `APP_NAME` and
`APP_AUTHOR` are both `"MEE2024"` — with **no local fallback**. The copy in the repository
root is legacy and is read by nothing.

That would be a findability annoyance on its own. Measured against
`config.DEFAULT_OPTIONS`, it is more than that: of **86 options, 17 are reachable from no
interface at all** — not the classic UI, not the app window, not a named CLI flag. They are
editable only by hand-editing that file, or via `mee2024 --set`.

Thirteen of the seventeen change what a reduction does:

| | default | governs |
|---|---|---|
| `double_star_cutoff` / `double_star_mag` | 10.0″ / 17.0 | which stars are discarded from the fit |
| `hot_pixel_sigmas` / `hot_pixel_min_adu` / `hot_pixel_dark_free` | 20.0 / 10.0 / True | defect masking |
| `img_edge_distance` / `pxl_tol` / `cutoff` | 5 / 10 / 100 | centroid acceptance and stack matching |
| `sanity_check_centroids` | True | whether centroids are validated at all |
| `safety_limit_mag` | 13.0 | catalogue depth guard |
| `platesolve_noise_px` | 0.3 | the v2 solver's noise model |
| `residual_bins` | 0 | stage-2 residual binning |
| `DEFAULT_DATE` | 2020-01-01 | fallback epoch when guessing |

The remaining four — `flag_debug`, `catalogue_cleanup_dismissed`, `watch_folder`,
`default_interface` — are plumbing or UI state, though `default_interface` is the only way to
choose which window opens and it is buried with the rest.

**The fix is not seventeen new controls.** Hand-maintaining a control per option is how
seventeen came to be orphaned in the first place, and the next option added will make
eighteen. Give `DEFAULT_OPTIONS` a schema — default, type, bounds and the help text that is
already written as comments beside each entry — and generate from it:

- an **Advanced** disclosure in the app window listing every option not surfaced elsewhere,
  so the orphan count is structurally zero rather than tracked
- tooltips, from the same help strings
- `mee2024 config --path` / `--list`, and real `--help` for `--set`

Cheapest useful step, independent of the schema work: a **"reveal settings folder"** button.
One control, and the file stops being unfindable.

`migrate_config` already keys fixes on the version that wrote the file, so the schema can
land without resetting anything a user meant to keep.

### F13 — Product naming: drop the year, keep the storage key

`MEE2024` was named before the 2024 eclipse. The program now reduces 2017 Bruns data, 2024
data and 2026 data, and prepares for 2027 — so the year describes *one dataset*, not the
software. The groups.io board has already moved from MEE 2024 to MEE 2027, skipping 2026
entirely, which is the same problem seen from the other side.

**Decision: drop the year rather than chase it.** The executable becomes `MEE_v<version>.exe`
— `MEE_v1.3.9.exe` — with window titles, README, `CITATION.cff` and release titles to follow.
Not `MEE_2026`: a year in the name is a promise to rename every year, and it mislabels the
2017 re-analysis as off-label use of the wrong tool. The version number already carries the
information the year was standing in for.

**The trap, and why the rename is smaller than it looks.** `APP_NAME` does not only set the
config path — it feeds `get_data_root()`, which is where the **triangle database, the star
catalogues and the pattern databases** live. Change `APP_NAME` and every existing install
looks in a new, empty directory and re-downloads the archive. `migrate_config` keys its fixes
on the version that wrote the config file; there is no equivalent for the data root.

The way out is that **the product name and `APP_NAME` need not change together.** Rename the
user-facing surface and leave `APP_NAME = "MEE2024"` permanently, as an opaque storage key
that users never see. No migration, no re-download.

Measured surface: `MEE_2024` 16 sites (the exe filename pattern), `MEE 2024` 1 (display
text), `MEE2024` 173 — of which 111 are `MEE2024util` imports, 15 are `MEE2024.spec`, and 58
are mixed. The user-facing part is roughly **30 sites**: the exe filename, window titles,
README and docs, `CITATION.cff`, release titles. Left alone deliberately: the `MEE2024util`
imports, the `mee2024` package and CLI command, `MEE2024.spec`, `APP_NAME`, and the `MEE2024`
FITS keyword — an opaque provenance token that other people's scripts may already read.

The GitHub repository keeps its legacy name. Renaming it breaks every existing clone, link
and citation for no user-visible gain.

### F14 — Choose the reduction parameters by measuring the frames

Three historical reductions now sit side by side (`LEON_2026-08-11.md` 14, 14.2), and the
values differ by more than an order of magnitude between them:

| | zenith / calibration | eclipse 2017 (Bruns) | eclipse 2024 (Station 1) |
|---|---|---|---|
| blob removal | off | on, radius 20, gap 10 | on, radius 100, gap 2000, sat 90% |
| sensitive stacking | off | on | on |
| `centroid_gaussian_thresh` | 5.0 | 4.0 | 4.0 |
| `min_area` | 4 | **1** | 2 |
| `sigma_subtract` | 3.0 | **0.0** | **0.0** |
| `remove_edgy_centroids` | off | on | on |
| `distortion_fit_tol` | **0.2"** | -- | **20"** |
| `pxl_tol` | 10 | 10 (shown as "pixel_tolerance") | -- |

Nobody can be expected to know these. They are not documented, most are reachable only from
the classic UI, and `pxl_tol` is not reachable at all any more -- v0.3.1 exposed it under
"Advanced Parameters" and the current classic UI does not, so it has *regressed* in
reachability while staying in `DEFAULT_OPTIONS`. That is F12 with a worked example.

**The proposal: measure, then choose.** Most of the table follows from three measurements the
pipeline can make before it commits to anything.

**1. Is there a saturated blob, and how big?** Find the largest contiguous saturated region.
It sets `delete_saturated_blob`, and `blob_radius_extra` and `centroid_gap_blob` follow from
its measured radius -- which is exactly what differs between 2017 (radius 20, gap 10) and 2024
(100 / 2000), and it is a property of the data, not a preference. The reanalysed 2017 set,
where long and short exposures were stacked together, produced a strikingly irregular mask for
precisely this reason: the saturated region is much larger in the long frames, so the union is
not a disc. Measuring it handles that automatically; typing two numbers cannot.

**2. How many stars does the field yield at default settings?** This is the sensitive-mode
switch, and the classic UI already states the rule in its own label: *"use if close to sun or
moon; do not use for zenith or fields with >> 100 stars"*. A trial pass counts them.
`sigma_subtract`, `min_area` and `centroid_gaussian_thresh` then follow the same split --
3.0 / 4 / 5.0 for a rich field, 0.0 / 1-2 / 4.0 for a sparse one.

**3. What per-star scatter does the field actually achieve?** Fit once at a loose tolerance,
read the rms, set `distortion_fit_tol` to roughly 1.5-2x it, refit. That is the sequence run by
hand in `LEON_2026-08-11.md` 14.2, and it recovers 0.2" for a zenith field fitting at 0.1" and
0.5" for CAL_piLeo fitting at 0.3" -- the same numbers, derived rather than remembered.

**The exception, and measurement 1 detects it.** A field containing the Sun cannot have its
tolerance tightened onto the residual, because the deflection displaces exactly the stars the
experiment exists to measure: fit tightly and you reject the signal. So when a blob is present,
the tolerance is floored at the expected maximum deflection plus field-edge leeway -- which is
why 2024 used 20". The blob measurement that decides step 1 is the same one that decides
whether step 3 is allowed to tighten. That is a satisfying closure rather than a special case.

**4. Frames outside totality must be excluded, and the machinery is already there.**
`framescan.scan()` measures a per-frame level and `suggest()` proposes a usable range, trimming
both ends because the Sun can be at the end of the last file as well as the start of the first.
Today it *only* suggests -- its docstring says "It never edits anything". Wiring it to act by
default, with an explicit override and a line in the log saying what was dropped, closes a
mistake that is easy to make: this session's own first CAL_piLeo reduction silently mixed the
`18_29_27` and `18_29_57` blocks, and the second runs past the end of totality, which produced
a spurious "no star centroids found" and a wrong conclusion that survived until the owner
caught it.

**Scope.** Steps 1, 2 and 4 are measurements plus a lookup table and could ship without
changing any fit. Step 3 changes results and belongs in v1.4.0 with the rest. All four should
report what they chose and why, in the run log and `field_summary.json` -- an automatic choice
that cannot be inspected is worse than a documented default.

### F2 — Affine per-frame alignment

Replace translation-only alignment with a per-frame affine. §1.5 is the evidence; the
mount-off dataset is the extreme validation case. Expect it to help every dataset, not just
badly tracked ones.

### F3 — Batch summary file — **done, v1.3.6**

`batch_summary.csv` + `.json`, one row per field: source folder, output folder, status,
error text, plate-solve success, and every metric from I8. A header block carries the
run-constant parameters (distortion order, magnitude limit, fit tolerance).

This is how §1.2 should have been found. It also serves cross-run collation and the
repeatability analysis, both of which were previously manual.

v1.3.7 added the per-field roll-up beside it (I20), the calibration match per field, and —
the part that had been missed — a way to *reach* any of it from the app window (§2.1).

### F4 — Persist the activity log — **done, v1.3.6**

The app's event bus had one sink, `ListSink`, in memory, cleared at the start of each run,
so the batch-level narrative existed nowhere on disk. `events.JsonlSink` already existed and
was used by the CLI; the app simply never wired it up. It does now, minus the `IMAGE` events
— a base64 PNG each, 97% of the bytes and none of the story.

Reachable from the window as of v1.3.7 (§2.1). A library build writes one too, into the
library folder beside the masters.

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

### F10 — Multi-select folders in batch mode — **done, v1.3.7**

Batch mode took one root and processed everything beneath it, so reducing an arbitrary
*subset* — two fields out of eighteen, after a rerun — meant running them one at a time.

`allow_multiple` is honoured for the native folder dialog (it was forced off for
directories), the in-page picker gained per-row folder selection with a separate chevron to
descend — selecting a folder and entering it are different actions, and one click cannot be
both — and `find_fields_in` accepts a list of roots. Each field keeps its own root's name at
the front of its relative path, so two fields the capture software called the same thing do
not collide in the output, and the run is named after the roots' shared parent when they have
one. A nested selection does not process a field twice, and one unreadable root stops the run
rather than being silently dropped.

### F9 — UI convergence

See [`UI_ROADMAP.md`](UI_ROADMAP.md). In short: the classic interface is frozen (0 commits
since the app window landed, against 19), but cannot be retired, because the app window
cannot run stage 3, cannot enable the astrometric corrections, and — as of this week's
review — has none of the blob/corona controls that eclipse-day processing depends on.
Those three are one deliverable: *the app window can process eclipse data*.

### F15 — Centroid the stacked image under a fixed window

**Measured on the Leon zenith set 2026-08-23** ([`LEON_2026-08-11.md`](LEON_2026-08-11.md)
§18.3), and this is a general defect rather than a Leon one.

The centroider defines each star's footprint by an S/N threshold and weights it by
`max(S/N - sigma_subtract, 0)`, so **both the region summed over and the weighting inside it
scale with the star's brightness**. Where the PSF is asymmetric — and any reducer at the wrong
back focus makes it so toward the field edge — bright and faint stars are therefore measured
over different parts of the same profile and disagree with each other. On Leon that was
172 and 299 mas beyond r = 2500 px, in twelve fields out of twelve, and it made the fitted cubic
a function of `max_star_mag_dist` at the 4.5% level. `PSF_REVIEW.md` §2 predicted exactly this:
*"unbiased only if the window is symmetric about the true position (it never is — the window is
placed by the detection)"*.

The fix is not to remove the aberration but to stop its effect depending on brightness. A
fixed-width Gaussian window, iterated to convergence — SExtractor's `XWIN_IMAGE`, and
`PSF_REVIEW.md` §5(d)'s own pick — did that on the real data: bias 11–13× smaller, rms −21%,
*more* stars surviving the tolerance, and the cubic independent of the magnitude cut to 0.44%
across four magnitudes.

**Scope.** One function of ~15 lines, one call site
([`stacker_implementation.py:1233`](../mee2024/stacker_implementation.py), where the *stacked*
image is centroided — the per-frame call at :854 feeds alignment, which is differential and
integer-rounded, so it needs nothing), and two options defaulting off. It **changes measured
numbers**, so it is v1.4.0-class and needs its own validation; default-off keeps the released
path additive and reproduces `tests/test_stage2_regression.py` untouched. The real-data half of
§5(d)'s evaluation is done and passes; the synthetic-truth half is not.

### F16 — Reject saturated stars

Nothing at any stage tests a centroid's peak value. `sanity_check_centroids` only checks the
radial profile decreases monotonically, which a flat-topped star passes. On Leon, six clipped
stars per zenith field carried **454 mas** against 124 mas for everything else.

They survive `distortion_fit_tol = 1` entirely; a tolerance of 0.2 removes five of six, by luck.
That luck runs out where it matters most: §16 of the Leon document sets `distortion_fit_tol =
999` on the eclipse field, deliberately, so **a clipped star is guaranteed to reach the
deflection fit** — and CAL_piLeo, whose low-order terms propagate, has its brightest star clipped
at the longer exposure.

A peak-value flag set at stage 1, where the stamp is already in hand, honoured at stage 2
regardless of tolerance. This one is a defect rather than a tuning choice.

**Measure it per frame, not on the stack** (Douglas, 2026-08-24). CAL_piLeo's brightest star
peaks at 45958 ADU on the combined stack — 70% of full scale — because the mean of seven clipped
2 s frames and eleven unclipped 1 s frames dilutes the clip away. A stacked-image test cannot see
a clip that only the long exposures carry, so on that field a stack-based F16 is inert. The fix
is a saturation mask built as each raw frame is read, shifted by the integer offset the stacker
already applies and accumulated, giving "clipped in N of 18 frames" in stack coordinates. It
reuses the count array `add_img_to_stack` already maintains. **SCI_ladder needs this more than
CAL_piLeo**, being a ladder of exposures.

And a provenance note on this item's own justification: the claim that CAL_piLeo "has its
brightest star clipped at the longer exposure" came from the owner's prior per-frame analysis,
not from any stacked measurement — the three v1.3.6 stacks show a worst peak of 59612, below a
60000 cut. Consistent with the dilution above, but the wording should say per-frame.

### F17 — Report what the fit can actually resolve

Two numbers the pipeline has everything to compute and does not print.

**Leverage-weighted star count.** Only the outer stars constrain a cubic: with
`N_eff = sum (r/R)^6`, Leon's sensor gives `N_eff/N = 0.085` — about one star in twelve counts,
and stars inside r < 1400 px hold 0.19% of the information. The predicted precision follows:

    sigma(cubic)/cubic = k * sigma_star / ( d(R) * sqrt(N_eff) )     k ≈ 16 cubic, 28 quintic

measured to ±20% over a 25× range in N. Printing `N_eff` and the predicted fractional precision
turns "which magnitude limit should I use?" into "go deep enough to meet your requirement, and
here is whether you did" — telescope-independent, and nothing to tune. It also warns against
cropping, which looks attractive on rms and destroys the cubic.

**The median residual beside the rms.** The rms is largely a readout of `distortion_fit_tol`,
since that is the outlier threshold: on Leon, tightening it 1.0 → 0.2 halves the rms
(108 → 61 mas) and moves the median 5% (45.3 → 43.2). Runs at different tolerances are not
comparable on rms and are on the median. This mattered in practice — it is what made a claimed
15–20% pipeline improvement evaporate on re-measurement.

### F18 — Weight the deflection fit instead of cutting on magnitude

Stage 3 filters on `eclipse_limiting_mag` (default 11) and fits an unweighted sum of squares by
Nelder-Mead, with **no outlier rejection of any kind** — the magnitude cut is the only quality
filter on by default.

On an eclipse field magnitude is the wrong axis. The dominant noise is the coronal background,
which falls as R^-2 to R^-3, so between 1.2 and 4 solar radii the background moves ~100× and the
noise ~10×, against 2.5× for one magnitude. **A mag-11 star at 1.2 R☉ is worse than a mag-12 star
at 4 R☉, and the cut keeps the first and discards the second.** §16's own justification for the
cut concedes the point: fainter stars near the corona are least reliable *"and each one enters
the 1/R fit with equal weight"*.

Inverse-variance weights from an a-priori sigma — flux, measured local background, PSF width, all
already computed — need no threshold, and a star with sigma → infinity self-limits to zero
weight. Two conditions must hold first, and one currently does not:

- **sigma must be a-priori, never a star's own residual.** On a deflection field the residual
  *is* the measurement, so weighting on it biases L toward zero.
- **Something must catch mismatches.** A wrongly matched faint star is not noisy, it is wrong,
  and its a-priori sigma looks respectable. With ~30 stars a single 10-sigma blunder carries 78%
  of the total chi-squared. Today the magnitude cut is the only thing standing in the way; the
  replacement is a robust loss, not another magnitude threshold.

Related and cheap: `do_cubic_fit` and `_cubic_helper` already accept a `weights` argument, pass
it through three calls, and **never read it** — a comment at `distortion_polynomial.py:161`
records the intent as *"new Oct'24: option for weighted centroids"*. Implement it or delete it;
a signature that promises weighting and silently ignores it is worse than not having one. On the
zenith fields it would buy under 1%, because the per-star sigma there is flat to 1.2× from mag
7.5 to 12.5 — PSF-limited, not photon-limited.

### F19 — Warn when the reducer spacing is wrong

Fitted in quadrature, Leon's blur grows **linearly** with field radius — coma — where defocus or
astigmatism would grow as r². This is a back-focus error that does not allow for the filter
glass, and it is a defect every reducer user can have without knowing.

What makes it worth a warning is that its cost is invisible where people look. A wider corner
star is astrometrically harmless on its own; the damage is the one-sided flare in the wings, which
biases centroids (F15) and leaves *no signature in the rms or the residual map* because the fit
absorbs it into the cubic coefficient. The warning should say "your cubic is biased and your
residuals will not tell you", not "your corners are soft".

Detectable from a single frame with what stage 1 already measures: median star width at r < 700
against r > 2800, plus the corner radial/tangential ratio. **Which direction** the spacing is
wrong is not derivable from one frame — the sign lives in the astigmatism orientation, which flips
through focus — so the warning should point at the focus sweep rather than guess. That sweep,
five to seven exposures at ±30 steps, settles the sign and simultaneously calibrates the
focus-to-cubic sensitivity that the Leon transfer error currently rests on.

**And be careful what the warning promises.** Correcting the spacer stack removes the *baseline*
aberration; it does **not** remove the dependence of the distortion on focus. The reducer rides on
the drawtube with the camera, so reducer-to-sensor is fixed by the spacers while focusing changes
the **objective-to-reducer** distance — a different degree of freedom, which alters the reduction
factor and the residual aberration whatever the spacers are set to. That coupling is intrinsic to
having a reducer at all (`LEON_2026-08-11.md` §10.3, mechanism corrected 2026-08-23). The warning
should say the spacing is wrong, not that fixing it makes the calibration focus-independent.

While the sweep is running, **record the fitted plate scale at each step as well as the cubic
amplitude**. The mechanism above predicts the plate scale tracks focuser position, since the
effective focal length moves with the objective-to-reducer distance; the 190 ppm between the two
Leon nights is consistent with that. It costs nothing to log and would confirm the mechanism
directly.

### F20 — Record what the corrections were fed, not just that they ran

`distortion_results.txt` records `aberration/parallax correction enabled?` and `refraction
correction enabled?` — and nothing about the values behind them. Not `observation_temp`, not
`observation_pressure`, not `observation_humidity`, not `observation_wavelength`, not the site.

That is not cosmetic. The eclipse-day plate scale moves **13.5 ppm per kelvin** of assumed
temperature (`LEON_2026-08-11.md` §19.1), so a result which depends on an input that strongly
must record it. It already cost something concrete: §14.1's plate scale could not be reproduced
on another machine, and the temperature it was computed with **cannot now be recovered from the
output** — the run is gone and the number was never written down. Douglas' first question back
was "what `observation_temp` did those runs use?", and the honest answer is that the file does
not say.

Cheap, and the stage-1 pattern already exists: `results.txt` writes the whole `effective_options`
block. Stage 2 should do the same, or at minimum the correction inputs and the site. While there:
the same file should carry the **reference-projection gauge** it reports coefficients in (§18.11),
so a coefficient copied out of it cannot be compared against a TAN-gauge number by accident.

Related discipline, worth stating in the same place because it is the cheaper half of the fix:
**one assumed temperature for the whole eclipse-day chain.** The error is common-mode between the
calibration field and the eclipse field and cancels to ±1.2 ppm if they share it; it does not
cancel at all between two reductions made at different assumed temperatures (§19.2).

---

## 3a. External sources, and what they do and do not settle

Three documents in `I:\Papers` constrain this work and were not previously cited anywhere in
the repository. None is a project decision; two are external analyses whose conclusions are
suggestive rather than settled, and this section records which parts are load-bearing and
which are open.

**Bruns, "Minimizing the Effects of Cubic Optical Distortion on Wide-Field Astrometry" (V5,
2024-07-18).** Measures cubic distortion for nine telescope/camera combinations (Table 1) and
argues about where to place the Sun. What it establishes with data:

- **A reducer dominates the cubic distortion.** The same TV-85 + ASI1600 goes from -0.3" with
  no reducer to **+8.4" with a 0.8x reducer** -- 28x worse and sign-flipped, at 475 mm EFL.
- **Between-session repeatability**: two telescopes over several nights, refocused each night,
  camera sometimes removed, imaged above 80 degrees altitude across 10-15 star fields --
  **7.09" +/-0.9%** and **-1.03" +/-2.6%**. That is a stronger claim than §1.1's, which is a
  *within*-session bound over 42 minutes on one focus. Read §1.1's "measured for the first
  time" as "first from this pipeline".
- **Imaging above 80 degrees to make refraction negligible** is his method, and it is the
  origin of the zenith/horizon pairing used here. Worth citing rather than re-deriving.
- The **Askar FRA500 + 0.7x + ASI1600** he recommends measures -0.2", essentially flat. The
  Leon rig is that optical train with an **ASI2600MM** instead, so the coefficient does not
  transfer -- a larger sensor reaches further into the field -- but the optic is the clean one.

**What it does not settle: where to put the Sun.** The paper compares *centred* against *one
corner* and concludes for the corner, because with the Sun centred an error in the cubic
coefficient acts along the same direction as the deflection. It never analyses an
**edge-centre** placement, which is the live alternative here: Sun at the centre of the long
edge with a **two-panel mosaic** rather than four. That is the geometry **Leon was testing**,
and Douglas holds the analysis. Treat the corner recommendation as one input, not a decision.

### The reference-projection gauge -- the finding that changes how numbers are read

`MEE2026_Bruns_NP101is_Astrometrica_Cross_Analysis_2026-08-05.md` (analysis by Claude, not
independently reviewed) compares Astrometrica 4.13 against MEE on **four fields Astrometrica
read from MEE's own `STACKED_FLOAT` output**, so both programs saw identical pixels.

Its central claim: MEE reports distortion in an internal **angular (arc-like)** frame rather
than the tangent plane, so its coefficients differ from any TAN-gauge tool by a universal
radial theta-cubed term. Converting between them is therefore two steps:

1. **Basis rescale.** Take the linear Jacobian `J` from Astrometrica's `x'`,`y'` terms
   (radians per pixel); the nonlinear displacement in MEE's pixel gauge is `J^-1 . dxi`, and
   each coefficient maps with `W^(i+j)` where `W = max(img_shape)/2`. Mind `det J < 0`.
2. **Add the gauge term**: `k_TAN ~= k_MEE + ~0.4 "/deg^3`, following theta-cubed.

**Verified here, independently.** Step 1 was re-derived from the code and the logs and agrees.
For step 2, fitting the difference between the two distortion fields as a *single* radial cubic
collapses the residual to **0.016-0.021 px** against fields of ~0.46 px rms -- one free
parameter, four fields. The mechanism is real, and the disagreement is **additive and radial**
rather than a scale factor, which is why treating it as a scale factor gives incoherent
per-axis ratios (x3.0 in x, x4.4 in y).

**The constant is not agreed.** The note gives **+0.41 "/deg^3** (its Bruns value +0.4110);
the measurement here gives **+0.4587 +/-0.0106**, 12% higher on the same data. The theoretical
tan-minus-arc term is exactly `theta^3/3 = +0.3656 "/deg^3` (confirmed), so the unexplained
remainder is either ~0.044 (the note) or ~0.093 (here). The methods differ -- the note refits
both programs' star tables, this used the printed polynomial coefficients as fields, and MEE's
printed coefficients come from a three-pass fit that folds linear terms back into the plate
solution each pass. **Do not quote a conversion constant to better than ~10% until that is
resolved.**

Cheap and actionable regardless: **document MEE's reference projection explicitly**, and
consider a TAN-gauge export. Anyone comparing MEE output against Astrometrica, ASTAP or a
published coefficient without the gauge term will conclude the programs disagree by a factor of
three. On the evidence here they do not -- centroids agree to 0.012 px median, recovered
catalogue positions to 8 mas, per-star residuals 0.052" against 0.055".

**Does the gauge term affect the deflection measurement? No -- verified.** It is a
*field-centred radial theta-cubed* term, which is one of the basis functions the distortion
polynomial already fits, so it lands entirely in the fitted cubic coefficient and changes no
residual, no star position and no deflection. Three checks:

- **It really is pure theta-cubed.** The next term of `tan(theta) - theta` is
  `2*theta^5/15`, which is 1.2e-4 of the cubic term at 1 degree and 2.1e-4 at 1.3 degrees.
  A cubic fit absorbs the gauge completely; it does not leak into the quintic terms.
- **Deflection cannot alias with it.** Deflection is Sun-centred and goes as 1/r; the gauge is
  field-centred and goes as theta-cubed. Different centre, different radial law -- and in the
  planned geometry the Sun is not at the field centre in any case.
- **The calibration-transfer step preserves the cancellation.** Holding the cubic fixed from a
  zenith field into an eclipse field is the one place an offset could survive. It does not:
  `distortion_reference_files` (`--fix-distortion`) loads MEE's own `distortion_results.txt`
  and reads `"distortion coeffs x"`/`"distortion coeffs y"` from it, so both ends are in MEE's
  gauge.

It follows that the 12% disagreement over the constant above is also irrelevant to the
measurement. It matters only for cross-program conversion.

**The one way it bites: a coefficient typed in from outside.** A value taken from Bruns'
Table 1, an Astrometrica log or any published source is in TAN gauge, and MEE would treat it
as its own -- injecting ~0.4 "/deg^3 into the held-fixed cubic. Bruns' paper is explicit that
an error in exactly that coefficient biases the Einstein coefficient, worst with the Sun
centred. So the gauge is harmless until a number crosses a program boundary, and then it is a
first-order error. Worth a guard on the reference-file path, and worth stating the gauge
alongside any coefficient this project publishes.

### Central radial residual: tested here, and the stacker is not indicated

The note's §7 reports a "central pixel-space compression" of -8 to -65 mas across three
sensors and two telescopes, and names **stacking registration** as the leading suspect, on the
mechanism that linear-only frame alignment plus distortion plus field rotation would leave a
radius-dependent bias. Two tests were run against that, with polynomial order held fixed in
each so it cannot confound the result.

**Test 1 -- is it polynomial order?** Re-running stage 2 on Don's HIP 31096 archive, same data
and version, changing only the order:

| order | rms | nn | 0-15' | 15-25' | 25-35' | 35-45' | 45-60' |
|---|---|---|---|---|---|---|---|
| cubic | 71.7 mas | 0.185 | -1.8 | -1.8 | **-12.3** | -8.7 | +10.0 |
| quintic | 70.0 mas | 0.166 | +5.0 | +3.9 | **-5.9** | -6.0 | +7.0 |

Quintic **halves** the mid-field inward pull and turns the inner bins positive. The note treats
the cubic underfit (its §4) and the central compression (§7) as separate effects; on this
data they are not cleanly separable, and a substantial part of the reported compression is
order.

**Test 2 -- is it stacking?** One frame against a full stack, quintic in both, two datasets:

| dataset | run | stars | rms | 30-45' | 45-60' |
|---|---|---|---|---|---|
| Texas 2024, 294MM | 1 frame | 95 | 74.0 mas | +21.5 +/-13 | -14.0 +/-12 |
| Texas 2024, 294MM | 7 frames | 95 | 50.2 mas | +17.0 +/-8 | -16.4 +/-11 |
| London 2026, 533MM | 1 frame | 185 | 83.4 mas | -5.7 | +5.2 |
| London 2026, 533MM | 10 frames | 185 | 67.8 mas | -3.4 | +2.2 |

The coherent outer-field structure is **present in a single frame at the same amplitude**, and
where stacking changes anything it makes it *smaller*, consistent with noise averaging. On this
evidence stacking does not introduce the pattern.

**Caveats, and they matter.** Both datasets are star-poor -- 95 and 185 matched stars, so
per-bin standard errors of 5-25 mas against an effect of the same size. Neither is the NP101is
the note measured. And neither is expected to carry field rotation: **Don's data almost
certainly has none**, which removes the note's proposed mechanism from his fields whatever the
stacker does. `I:\Toby Portland 2026` is the dataset that plausibly does have poor polar
alignment, and is where the rotation hypothesis should actually be tested. **That test has not
been run.**

So: not confirmed, not refuted. What it does say is that **F2 should not be justified on this
basis** -- §1.5's drift measurement remains its real argument.

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

Steps 1 to 3 of the original order are **done** — I0–I20, F1, F3, F4, F10 and F11, across
v1.3.6, v1.3.7 and v1.3.8. What follows is organised by *release* rather than by priority,
split on a single criterion: **whether a change can alter a measured number.**

The reason for splitting that way is v1.3.5. Three builds — v1.3.6, v1.3.7 and v1.3.8 — were
made and field-tested and none was ever published, so general users are still running a
version that overwrites its own batch records and exits 1 on success. Getting that body of
work out matters more than getting the next feature in, and it only stays low-risk if nothing
rides along that moves a centroid.

### v1.3.9 — additive only, and therefore still the tested v1.3.8

Every item either adds information or changes a message. None changes a number, so the field
testing behind v1.3.8 carries over.

| | why it is safe |
|---|---|
| float32 light stack always written, with a real header | adds a file; the 16-bit primary output is byte-identical |
| `EXPTIME`/`GAIN`/`DATE-OBS`/calibration-applied in stacked headers | new keywords only |
| `DATAMIN`/`DATAMAX` on masters | header only |
| flat fill check on input frames | changes a warning, never the data |
| `CALIBRATION_LIBRARY` under the **output** folder | a path default; keeps the source data pristine |
| the app window gets its own settings file | fixes the leak below; no reduction path touched |
| rename to `MEE_v1.3.9.exe` (F13) | filename and display text only; `APP_NAME` unchanged |

**The settings leak this fixes.** `_remember_settings` reads the existing ini, overlays its
own keys and writes the whole file back — so the app window **writes** `sensitive_mode_stack`,
`distortionOrder`, `distortion_fit_tol`, `max_star_mag_dist`, `guess_date` and the folder
paths into `MEE_config.txt`, the file the classic UI and CLI read, while never reading that
file itself. Run the app window once on "quick", and `sensitive_mode_stack: false` is waiting
in the shared config for the next classic-UI session. One-way contamination, and in the
opposite direction from the one you would guess.

The principle settled on: **an interface should only apply settings it can show.** The classic
UI can display forty-odd options and the app window eight; if the app window inherited the
shared file it would run with values the user cannot see or change in the interface they are
actually using. Hence one settings file per interface — and hence the *site file* in v1.4.0,
because session and site data (date, time, lat/long, temperature, pressure, humidity, height,
wavelength) belong to the **observation**, not to an interface, and must stay readable by all
three. Without the site file, separating the configs would strand the astrometric corrections
in the classic UI.

### v1.4.0 — the things that change results, re-tested on Leon

Testing starts immediately and in parallel; this is not queued behind v1.3.9's distribution.

1. **F7 — header harvest**, promoted from fifth place. The Leon headers carry `OBSLAT`,
   `OBSLONG`, `SITEELEV`, `JD_UTC`, `CENTALT`/`OBJCTALT`, `AIRMASS`, `FOCTEMP` and
   `EQUINOX` on every frame, and the eclipse happened 9° above the horizon where **vertical
   refraction is the dominant error term**. Refraction needs site, time, temperature,
   pressure and humidity; four of the five are in the headers already and the fifth is in
   `leon_temp_press_humid.csv`. Nothing else on this list unlocks as much.
2. **The site file, and corrections on by default.** The other half of the settings split
   above. This turns the astrometric corrections **on**, which changes every fit — which is
   precisely why it is here and not in v1.3.9.
3. **First-file dependence.** The epoch moves to mid-sequence rather than frame 0, and the
   blob mask comes from the stack rather than from one frame. Both change results across all
   five sites.
4. **F2 — affine alignment.** The largest accuracy item, and the evidence for it already
   exists (§1.5). Validate against the tracking-off dataset.
5. **F12 — the settings schema**, plus the "reveal settings folder" button, which can land
   earlier since it adds a control and changes nothing.
6. **F8 — solve fallback from the header.** `FOCALLEN` 350 and `XPIXSZ` 3.76 give a
   2.216″/px prior on the Leon rig; the horizon fields at airmass 5.8 are exactly where a
   blind solve is most likely to need it.
7. **F13's remaining surface** — window titles, README, `CITATION.cff` — once the executable
   rename has been through a release.

### v2.0 — and beyond

Tab 3 in the app window, adaptive sensitivity, the Leon exposure-ladder threshold sweep, and
**F9 — UI convergence**, gated on Q1. Also **F5 — drift measurement**, now that `JD_UTC` is
known to be per-frame in this data, and **F6 — open a previous run from disk**.
