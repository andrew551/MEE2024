# v1.4.0 — the work plan

This branch (`v1.4.0-dev`) holds everything that can change a measured number. Nothing here
should reach a general user until it has been re-tested on real data.

The version reports **`v1.4.0-dev`**, so a build is `MEE_v1.4.0-dev-g<sha>.exe` — the commit is
in the filename, because two people now build executables and v1.3.6 was already lost once to a
binary that no tag recorded. `-dirty` is appended if the tree had uncommitted changes.

**Read first:** [`../CLAUDE.md`](../CLAUDE.md) for the rules,
[`ONBOARDING.md`](ONBOARDING.md) for the reasoning, then §§8–17 of
[`LEON_2026-08-11.md`](LEON_2026-08-11.md), which is where most of this plan comes from.

---

## 1. What changed since this file was first written

A campaign-data review (2026-08-20) reordered the priorities and found one outright bug.

**The refraction correction never ran.** `enable_corrections` raised
`AttributeError: 'StarTable' object has no attribute 'c'` on the first star table it was
given — `StarData` exposed its SkyCoord as `.c`, `StarTable` builds one via `skycoord()`, and
the migration missed it. Off by default and classic-UI only, so nobody saw it. **Fixed on this
branch** (`refraction_correction.py`), and with it working the CAL_piLeo field reduces: a 1.27″
mean correction at 9.9° altitude, moving the fitted plate scale 3500 ppm *towards* the zenith
nights. See LEON §11.

**The eclipse was shot 121–129 focuser steps from either calibration night** (`FOCUSPOS` 17170
against 17049 and 17041, at 37 °C against 30 °C). The observer refocused, correctly — the
thermal coefficient measures 17–20 steps/°C — but the focuser and the reducer-to-sensor spacing
are the same degree of freedom, so the eclipse frames carry a different field-aberration state.
The cubic moved 5.1% between two nights 8 steps apart. Nothing bounds what 121 steps does.
See LEON §10.

**The optics have a fixed radial aberration the focuser cannot reach.** PSF grows 1.55–1.66×
centre to corner on all three runs (Portland and both Leon nights), and the growth is closer to
linear in radius than quadratic, which is coma rather than field curvature. Consistent with the
reducer sitting 0.94 mm off its design spacing because the filter thickness was not allowed
for. See PORTLAND §13.4–13.6 and LEON §9.3.

---

## 2. The work, in order

### F7 — header harvest, site and refraction *(the unlock)*

Now doubly justified: the León eclipse was 9.7° above the horizon where refraction curvature
across the field is ~0.50″, **and** the only correctly-focused calibration field (CAL_piLeo,
`FOCUSPOS 17170`) sits at 10.5° altitude, so it cannot be used for distortion without
refraction modelling. The headers carry `OBSLAT`, `OBSLONG`, `SITEELEV`, `JD_UTC`,
`CENTALT`/`OBJCTALT`, `AIRMASS`, `FOCTEMP`, `EQUINOX`; the fifth input is
`leon_temp_press_humid.csv` (UTF-16, **local time UTC+2** — LEON §4).

**Add `FOCUSPOS` to the run summary.** It is in every frame header, the pipeline never reads
it, and it turned out to be the single most diagnostic number in the campaign.

### The site file, and corrections on by default

Session and site data belongs to the *observation*, not to an interface, so all three front
ends must read it. Without it, separating the config files (done in v1.3.9) strands the
corrections in the classic UI. This turns corrections **on**, which changes every fit.

### F14 — choose the reduction parameters by measuring the frames

New. Three historical reductions use settings differing by more than an order of magnitude
(ROADMAP F14, LEON §14). Most of it is measurable: the saturated blob sets blob removal and its
geometry, a trial pass counts stars and sets the sensitive-mode group, and a loose-then-tight
refit sets `distortion_fit_tol` from the achieved scatter. **And frames outside totality must
be excluded** — `framescan` already measures and suggests a range but never acts.

### F2 — affine per-frame alignment

The alignment residual rises with accumulated shift in **12 of 12** Leon field-nights, median
correlation +0.94 — cleaner evidence than §1.5's. But it does **not** propagate to the field
rms here (correlation +0.01), so sell it as removing a known defect, not as recovering rms.

### F12 — the settings schema

17 of 86 options reachable from no interface, 13 of which change a reduction. One known
regression: `pxl_tol` was exposed as "pixel_tolerance" in v0.3.1 and is not now. Tab 3's
Sun-centred cutoff radius and centre-on-Moon were listed here as a second regression; that was
wrong -- both are live options and the classic UI still shows them (LEON §17). They are
classic-UI-only, which is an exposure gap of the ordinary kind, not a regression. Also
`--order` offers only linear/cubic/quintic/septic while the code implements quadratic, quartic
and sextic.

### F8 — solve fallback from the header

`FOCALLEN` and `XPIXSZ` predict the plate scale to 0.45% on the León rig. **But the gate is not
optional**: the Bruns rasalhague header says 135 mm against an actual 472.5 mm — wrong by 3.5×
and *plausibly* wrong, since 135 mm is a real Askar focal length. A sanity check that only
rejects absurd values would accept it (ROADMAP §3a).

---

## 3. The measurements that would settle open questions

**A focus sweep.** Five to seven exposures at spaced `FOCUSPOS`, ±30 steps in tens, on one
zenith field, measuring FWHM(r) at each. Ten minutes at the telescope. It gives the movable
fraction of the radial blur directly *and* calibrates the cubic's focus sensitivity — which is
currently the campaign's only unbounded error term.

**Re-run CAL_piLeo's cubic against both zenith nights**, once F7 lands. Replaces every
extrapolation in LEON §10.3 with a number.

**The crop test on León.** Portland's static wall fell 202 → 148 mas at r < 1400 px, and
cropping was the only lever that moved a term averaging cannot. Not yet tried on León.

**Portland for the rotation hypothesis.** `I:\Toby Portland 2026` is the dataset that
plausibly has poor polar alignment; the stacker test there has not been run.

---

## 4. Rules for this branch

- **Do not merge to `main`** until the stage-2 regression baselines in
  `tests/test_stage2_regression.py` have been re-derived deliberately, with the change that
  moved them named. They are pinned precisely so a change of this kind cannot pass quietly.
- Branch from here for each item — `f7-header-harvest`, not a personal branch — and open a
  pull request back into `v1.4.0-dev`. GitHub Actions runs the fast suite on every PR.
- Expect **876 passing, 26 skipped**.
- Say which kind of change you are making: additive-only, or capable of moving a measured
  number. That decides which release it can go in and is the first question that will be asked.

## 5. Reference: the three-step reduction method

Never written down before this review; see LEON §16 for the reasoning.

| | zenith | L/R calibration | eclipse field |
|---|---|---|---|
| `distortion_fixed_coefficients` | None | quadratic | **constant** |
| what is fitted | everything | const, linear, quadratic | the offset only |
| `distortion_fit_tol` | 0.2″ | 0.2″ | **999″** |
| `max_star_mag_dist` | 13 | 13 | 11 |
| gravitational correction | off | off | **off, deliberately** |

`tol 999` and gravity-off are the point, not shortcuts: at the final step no star may be
rejected for sitting away from its catalogue position, because that displacement *is* the
measurement, and the deflection must survive into the residuals for stage 3 to fit as 1/R.
