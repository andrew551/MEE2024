# Where the centroid error actually comes from — a measured budget

The question this answers: **are we at the limit of what the data allows, and if not,
what is the limiting factor** — optics, atmosphere, pixel noise (flat/dark), pixel
size, exposure, or the algorithm?

The method is to refuse to guess: every candidate limit leaves a distinct,
independently measurable signature, so each term is measured from the frames
themselves and the parts must add up to the measured total. Where a term cannot be
measured with the data at hand, that is stated rather than estimated silently.

| candidate limit | its signature | measured by |
|---|---|---|
| photon + pixel noise | scatter follows the Cramér–Rao curve vs flux | per-star CR bound from its own fit, background noise, measured gain |
| atmosphere (differential tip-tilt) | residuals *spatially correlated* between nearby stars; pixel noise is white | two-point correlation of per-frame residual fields |
| mount / field rotation / refraction | removed by a per-frame affine but not by translation | translation-vs-affine residual comparison |
| pixel size (undersampling) | scatter depends on subpixel phase | residuals binned by fractional pixel position |
| algorithm | different estimators disagree | three unrelated estimators benched in [CENTROIDS.md](CENTROIDS.md) |
| optics (static aberrations + model error) | absent from frame-to-frame scatter; lives in the stage-2 fit residual | stage-2 rms vs propagated stacked-centroid noise |
| flat (PRNU), dark | detector-fixed pseudo-noise as the field dithers | bounded inside the unattributed white remainder |

Tool: `tools/error_budget.py`. Runs (v1.3.2):

```
python tools/error_budget.py "tests/data/fits/00_23_49/Zenith_*.fits" \
    --out docs/bench/psf/budget_zenith
python tools/error_budget.py "tests/data/fits/example_with_darks/070424_040415/*.fits" \
    --darks "tests/data/fits/example_with_darks/070424_050036 darks 10s/*.fits" \
    --out docs/bench/psf/budget_eclipse
```

## The numbers (per-frame 2-D rms, bright-star floor)

| term | zenith (10 × 3008², FWHM 1.22 px) | eclipse field (7 × 5644×8288, FWHM 2.97 px) |
|---|---|---|
| measured floor, translation removed | **0.0849 px** | **0.0832 px** |
| Cramér–Rao bound (photon + pixel noise) | 0.0102 px (12 %) | 0.0143 px (17 %)¹ |
| background-only bound (gain-free) | 0.0056 px | 0.0065 px |
| spatially correlated (atmosphere-like) | 0.0756 px (79 % of variance) | 0.0729 px (77 %) |
| removed by affine over translation | **0.0786 px** (floor → 0.0322) | 0.0416 px (floor → 0.0721) |
| still correlated after affine (anisoplanatism) | 0.0190 px | **0.0614 px** |
| unattributed white remainder after affine | 0.026 px (×2.5 its CR bound) | 0.038 px (×2.6 its CR bound) |
| pixel-phase bias amplitude | 0.006–0.007 px | 0.007–0.010 px |
| faint stars vs their own CR bound | ×2.72 | ×2.06 |

¹ Eclipse photon transfer is unusable — the darks sit at a *higher* level than the sky
(pedestal), which corrupts the level axis and drives the fit to gain ≈ 10⁹. The tool
now refuses implausible fits and falls back to an assumed 1 e-/ADU with a printed
caveat. The conclusion is robust to this: even at the most pessimistic plausible gain
for the ASI294 the bound stays well under a third of the floor, and the gain-free
background-only bound (0.0065 px) needs no calibration at all.

The zenith gain (2.567 e-/ADU) is a single-sky-level estimate: it assumes zero read
noise and any bias pedestal inflates it. Same robustness argument applies — the
gain-free bound is 16× below the floor.

### Does the budget close?

Zenith: 0.0786² (affine term) + 0.0322² (post-affine floor) = 0.0849² exactly (the
decomposition is variance-orthogonal by construction). Inside the post-affine floor:
0.0190² correlated + 0.0260² white; the white part is ×2.5 the CR bound. Eclipse:
post-affine 0.0721² = 0.0614² correlated + 0.0378² white, ×2.6 its CR bound. So every
term is pinned except one consistent factor-2.5 white excess over the Gaussian CR
model — candidates: PRNU (no flats are used; a ~1 % flat error moves a centroid by
roughly 0.01·FWHM ≈ 0.01–0.03 px, the right order), the CR model itself (Gaussian
assumed; real profile is Moffat β≈2.8 with window truncation), and fast tracking
error white-ish at 10 s sampling. Distinguishing those needs flats — see below.

## The static term (optics + distortion model + catalogue)

Frame-to-frame scatter cannot see errors that are the same in every frame. Those live
in the stage-2 fit residual, compared against what the stacked random noise predicts:

| dataset | stage-2 rms | propagated stack noise | static excess |
|---|---|---|---|
| eclipse (3-frame stack, 95 stars, 1.15 ″/px) | 0.0804″ = 0.070 px | 0.0832/√3 = 0.048 px | **0.051 px = 0.058″** |
| rasalhague (50-frame stack, 248 stars, 1.65 ″/px) | 0.0936″ = 0.057 px | ≈0.083/√50 = 0.012 px | **0.055 px = 0.092″** |

On a 50-frame stack the random noise is negligible and the stage-2 rms *is* the
static term. Its likely identity is known from the PSF work
([PSF_REVIEW.md](../PSF_REVIEW.md) §6): rasalhague's PSF varies strongly across the
field (quadratic model explains 44 % of FWHM scatter — tilt/coma), and
aberration-induced centroid shifts are exactly the kind of smooth field-dependent
error a cubic distortion polynomial absorbs only partially. Catalogue error (Gaia,
sub-mas) is negligible in this term.

## Verdict: we are *not* at the limit, and the limiting factor differs per setup

Ranked by what fixing them buys (2-D rms on the final stacked/solved positions):

1. **Frame registration model (software, zenith-class setups).** The stacker removes a
   translation per frame; on the zenith set a per-frame affine drops the floor
   0.0849 → 0.0322 px (2.6×). Rotation/scale/refraction wobble between 10 s frames is
   the single largest term there, and it is entirely recoverable in software. After
   stacking 10 frames: 0.027 px → 0.010 px of propagated noise.
2. **Static optics / model term (~0.05 px ≈ 0.06–0.09″).** Sets the floor of the
   stage-2 rms on long stacks. Levers: focus/collimation (rasalhague's field tilt),
   PSF-aware centroiding that models the field-varying kernel, or a higher-order /
   physically-motivated distortion model. This is the binding constraint on the
   50-frame stack today.
3. **Atmosphere (eclipse-class).** After affine, 0.0614 px on the eclipse field is
   genuinely anisoplanatic — irreducible per frame at this aperture. It averages as
   1/√(total integration); at fixed total time the 10 s exposure split is close to
   neutral, so "exposure" is not an independent lever — total integration is.
4. **Flats (PRNU).** Unmeasured (no flats in these datasets) but bounded: it lives
   inside the 0.026–0.038 px white remainder. Taking flats would either shrink that
   remainder or exonerate PRNU; either way it is a ≤0.03 px item.
5. **Pixel noise (photon + read + dark).** 0.006–0.014 px — 8–17 % of the per-frame
   floor. Not the limit, robustly to the gain caveats.
6. **Pixel size.** Phase bias 0.006–0.010 px even at FWHM 1.22 px — the pipeline's
   variance-weighted COM is far less phase-sensitive than plain COM. Not the limit.
7. **Algorithm.** Bounded at ~nothing: three unrelated estimators tie at the same
   floor ([CENTROIDS.md](CENTROIDS.md)).

How we can be *sure*: each mechanism was measured through its own orthogonal
signature rather than assumed, the terms sum to the measured total (exactly at the
first level of the decomposition; to within the flagged ×2.5 white remainder at the
second), and the two datasets rank the terms differently in exactly the way their
hardware differs — which is what a real decomposition, as opposed to a fitted
narrative, should do.
