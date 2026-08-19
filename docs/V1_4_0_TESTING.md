# v1.4.0 — the testing line

This branch (`v1.4.0-dev`) exists so that the risky half of the work can be built and tested
**in parallel with v1.3.9 being distributed**, rather than behind it. Nothing here should ever
be handed to a general user until it has been re-tested on real data.

The version reports **`v1.4.0-dev`**, so a build from this branch is `MEE_v1.4.0-dev.exe` and
can never be confused with the eventual v1.4.0 release. That confusion has happened before:
v1.3.6 existed as a real, field-tested binary that no tag recorded. `_version_tuple` parses
the suffix to `(1, 4, 0)`, so config migration behaves exactly as it will at release.

## Why these items are here and not in v1.3.9

The split is on one criterion: **whether a change can alter a measured number.** v1.3.9 took
only what adds information or rewords a message, which is why the field testing behind v1.3.8
still covers the binary users are getting. Everything below moves a fit, so it needs its own
validation against the León campaign data before it goes anywhere.

## What has to be built and tested

Ordered as in [`ROADMAP.md`](ROADMAP.md) §6.

1. **F7 — header harvest.** The unlock. `OBSLAT`, `OBSLONG`, `SITEELEV`, `JD_UTC`,
   `CENTALT`/`OBJCTALT`, `AIRMASS`, `FOCTEMP` and `EQUINOX` are on every León frame, and the
   eclipse was 9.7° above the horizon where the curvature of vertical refraction across the
   field is 0.50″ — about 4× what those fields actually fit. Refraction needs site, time,
   temperature, pressure and humidity; four are in the headers and the fifth is in
   `leon_temp_press_humid.csv`.
2. **The site file, and corrections on by default.** The other half of the settings split
   v1.3.9 began. Session and site data belongs to the *observation*, not to an interface, so
   it must be readable by the classic UI, the app window and the CLI alike — without it,
   separating the configs strands the astrometric corrections in the classic UI. This turns
   corrections **on**, which changes every fit.
3. **First-file dependence.** The epoch moves to mid-sequence rather than frame 0, and the
   blob mask comes from the stack rather than from one frame. Both change results at all five
   sites.
4. **F2 — affine per-frame alignment.** The largest single accuracy item; the evidence is
   already measured in §1.5. Validate against the tracking-off dataset.
5. **F12 — the settings schema.** 17 of 86 options are reachable from no interface; 13 of them
   change what a reduction does. Generate the Advanced panel, the tooltips and the CLI help
   from one schema rather than hand-maintaining a control per option. The **"reveal settings
   folder"** button is separable and safe, and can land at any time.
6. **F8 — solve fallback from the header.** `FOCALLEN` and `XPIXSZ` predict the plate scale to
   0.45% on this rig; the horizon fields at airmass 5.8 are where a blind solve most needs it.
7. **F13's remaining surface** — window titles beyond the main one, and any doc text the
   v1.3.9 rename did not reach.

## The validation that decides it

The zenith/horizon pairing is the design worth protecting. At 80° altitude the refraction
curvature is 0.00 px, so **the zenith fields measure distortion cleanly**; the horizon fields
can then measure refraction with distortion held fixed. That needs
`distortion_fixed_coefficients` reachable from something other than the classic UI, which is
why F12 and the site file are on this list rather than deferred again.

Do not merge to `main` until the stage-2 regression baselines in
`tests/test_stage2_regression.py` have been re-derived deliberately, with the change that
moved them named. They are pinned precisely so that a change of this kind cannot pass quietly.
