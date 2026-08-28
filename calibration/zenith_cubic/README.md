# Leon 2026 zenith cubic — handoff to step 2

> **THREE CONCLUSIONS BELOW ARE SUPERSEDED — read this box first.** The provenance, the
> settings tables and the step-2 invocation are still correct and are why this file is
> kept. What changed, all after 2026-08-27:
>
> 1. **The chain uses the six `08-12` files only; d(3000) = 3.0297 ″.** The twelve-field
>    mean of 3.1048 ″ below is superseded. The telescope was transported between night 1
>    and eclipse day and measurably changed — tilt dipole doubled, +197 ppm of plate scale
>    (`docs/REFRACTION_2026.md` §16.2–16.3). See `../README.md`.
> 2. **The cubic systematic is the ≥7 % class with unknown sign**, not "≥2.4 % and
>    unbounded above": the cubic stepped −7.3 % between two same-night, same-focus
>    sequences, cause unresolved (§15, §16.4). The 1/R residual-gradient diagnostic in the
>    eclipse data is the arbiter.
> 3. **The example invocation's `observation_time=18:29:19` is not the canonical value.**
>    The canonical CAL_piLeo stack is sixteen frames at **18:29:35** (true-exposure
>    weighted mid-point); the refraction scale term moves 1.78 ppm/s at this altitude, so
>    this matters. Frame list in `../cal_pileo_frames.txt`.
>
> The F16 note at the end was written before the work existed. Branch
> `f16-reject-saturated` now implements peak recording and stage-2 rejection, but on the
> *stacked* image — `docs/CAL_PILEO_STEP2.md` §8 measures that as inert where exposures
> are mixed, so the per-frame form step 3 requires is still outstanding.

Twelve `distortion_results.txt`, one per zenith field-night (11 and 12 August 2026), from the
in-pipeline windowed-centroid reduction. Self-contained JSON: each carries its own coefficients,
so nothing outside this folder is needed.

**Carry forward:** d(3000 px) = **3.1048″**, random **±0.38 %** on the twelve-field mean,
systematic **≥2.4 %** and unbounded above (see below).

## Provenance

`mee2024` on `v1.4.0-dev`, commit `4743f88` or later. Stage 1 and stage 2 both run from the
branch — no hand steps, no offline scripts.

| stage | settings |
|---|---|
| 1 | `sensitive_mode_stack=True`, `centroid_gaussian_subtract=False`, `centroid_gaussian_thresh=5.0`, `min_area=4`, `sigma_subtract=3.0`, `delete_saturated_blob=True`, `remove_edgy_centroids=True`, **`centroid_refine_window=True`**, `centroid_window_sigma=2.0` |
| 2 | `--order cubic`, `--date-from-header`, `max_star_mag_dist=13`, `distortion_fit_tol=0.2`, `rough_match_threshhold=36`, corrections **off** |

Corrections are off deliberately: at 78.5–83.4° they change the transferred cubic by +0.065 %,
and the plate scale they do move is discarded at step 2. They must be **on** for CAL_piLeo and
the eclipse field. See `docs/LEON_2026-08-11.md` §18.1.

## Using it in step 2

`reference_files.txt` holds absolute paths from the machine that produced it — **regenerate it
after copying**, from inside this folder:

```bash
python -c "import glob,os;print(';'.join(os.path.abspath(f).replace(os.sep,'/') for f in sorted(glob.glob('*.txt')) if 'reference_files' not in f))" > reference_files.txt
```

Then, per `docs/LEON_2026-08-11.md` §16:

```bash
python -m mee2024.cli distortion <CAL_piLeo_centroid_data.zip> \
  --order cubic --date-from-header \
  --fix-distortion $(cat reference_files.txt | tr ';' ' ') \
  --set distortion_fixed_coefficients=quadratic \
  --set distortion_fit_tol=0.5 --set max_star_mag_dist=13 \
  --set enable_corrections=True --set enable_corrections_ref=True \
  --set observation_lat=42.740470 --set observation_long=-5.613780 \
  --set observation_height=1101 --set observation_temp=30.5 \
  --set observation_pressure=896.6 --set observation_humidity=0.208 \
  --set observation_wavelength=0.62 --set observation_time=18:29:19 \
  -o <output>
```

`distortion_fixed_coefficients=quadratic` imports cubic-and-above from these twelve files
(averaged) and re-fits constant, linear and quadratic on the eclipse-day field. It has no named
flag, so it needs `--set`. The order must match (`cubic`) or stage 2 raises.

## Two things to know before relying on it

**The ±0.38 % is the random half only.** The two nights differ by 4.84 % at 6σ, and it is real —
identical distortion *shape* (ovality, position angle, trefoil all agree within 1σ), amplitude
differs. Both nights sit 121 and 129 focuser steps *below* the eclipse focus, on the same side,
so averaging them does **not** bracket the eclipse configuration. Quote the systematic as ≥2.4 %
and unbounded above, not as a symmetric error bar. The focus sweep in §9.3 is what converts it
to a number. Full argument in §18.6.

**Saturated stars are not excluded** (F16 is not implemented). Here it does not matter: 50
clipped stars survive across the twelve fields, 0.19 % of those used, carrying 1.0–1.7× the field
rms — against 3.7× under the old centroider, which a fixed window largely tames. **It will matter
on the eclipse field**, where `distortion_fit_tol = 999` deliberately rejects nothing, so a
clipped star is guaranteed to reach the deflection fit. Worth implementing before step 3, not
before step 2.
