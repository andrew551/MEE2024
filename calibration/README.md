# Frozen calibration inputs

These are **inputs to the reduction chain, not outputs of it** — which is why they are
versioned here rather than left under `D:\MEE2024 output\MEE_output`. Every downstream
number (the CAL_piLeo plate scale, and through it the deflection constant) is pinned to
these exact values, so they have to be recoverable byte for byte, independently of whether
a reduction tree still exists.

They are reproducible from the raw frames on `G:\Leon Aug 2026` — verified to all seven
digits — but that costs a stage-1 and stage-2 run per field, and the point of freezing a
calibration is that it does not get re-derived casually.

| path | what it is |
|---|---|
| `zenith_cubic/` | twelve `distortion_results.txt`, one per zenith field-night (11 and 12 August 2026), carrying the cubic-and-above distortion coefficients that steps 2 and 3 import |
| `cal_pileo_frames.txt` | the sixteen `G:` frame paths that define the canonical CAL_piLeo calibration stack |

## Which zenith files are canonical, and why not all twelve

**The chain uses the six `08-12` files only.** Carry forward **d(3000) = 3.0297 ″**.

The twelve-file mean of 3.1048 ″ is superseded and must not be used. The telescope was
dismounted, transported by car and remounted between night 1 and eclipse day, and the
change is measurable rather than assumed: the m=1 tilt dipole doubled (0.510 ″ at PA −67°
to 0.996 ″ at PA −101°), radial FWHM growth moved 1.216 → 1.335 at ~6σ, and the plate
scale stepped +197 ppm. Night 2 shares the eclipse day's mechanical state; night 1 is a
different optic. Full argument in `docs/REFRACTION_2026.md` §16.2–16.3.

The `08-11` six are kept as the pre-transport control — they are what made the transport
change detectable — and their own d(3000) is 3.1799 ″. They are not part of the chain.

The two nights' `summary.json` figures, for reference: `n1` 3.1799 ″, `n2` 3.0297 ″,
twelve-field mean 3.1048 ″, night-to-night gap 4.84 %.

## The CAL_piLeo frame list

`cal_pileo_frames.txt` holds the sixteen frames of the canonical calibration: all six of
`1.0s8_29_19`, the first three of `1.0s8_29_51` (pre-C3), and all seven of `2.0s8_29_27`.
Reduced against the six `08-12` references it gives **2.2054043 ″/px**, 74 stars,
rms 0.5318 ″, HC0 21.6 ppm (quote HC3-class ~25 ppm), at `observation_time 18:29:35`.

Two traps that cost this project time and are recorded here because this file is where
someone will meet them:

- The folder organisation on `G:` is the truth about exposure. **EXPTIME headers lie on
  the first frame after a SET EXPOSURE change** — verify by sky level when in doubt.
- `reference_files.txt` inside `zenith_cubic/` holds absolute paths from the machine that
  produced it. Regenerate it after copying; the command is in that folder's README.

Provenance for both sets, and the settings each was produced with, are in
`zenith_cubic/README.md`. The reduction they feed is `docs/CAL_PILEO_STEP2.md`.
