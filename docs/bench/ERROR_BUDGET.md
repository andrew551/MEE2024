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
python tools/error_budget.py "I:/65PHQ 533MM London 2026/00_23_49/Zenith_*.fits" \
    --out docs/bench/psf/budget_zenith
python tools/error_budget.py "I:/65PHQ 294MM Texas 2024/zenith 1/070424_040415/*.fits" \
    --darks "I:/65PHQ 294MM Texas 2024/070424_050036 darks 10s/*.fits" \
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

---

# Follow-up: does the scatter track the FWHM, and how does the rms stack down?

Two further questions, answered on the same footing (v1.3.3; the third dataset here
is `zwo3` — Richard Berry's ASI1600 at 420 mm, 1.87 ″/px, 9 × **0.2 s** zenith
frames plus 38 matching darks, header gain 4.96 e-/ADU, ~48 budget stars).

## 1. Does spatial variation in rms track spatial variation in FWHM?

Two different errors can track the PSF, and they need different tests.

**The random (frame-to-frame) error: no detectable tracking.** Per-star scatter
(affine-removed, so a rotation residual growing with field radius cannot masquerade
as PSF tracking) against per-star fitted FWHM, bright half only:

| dataset | PSF variation across field | Spearman ρ (scatter vs FWHM) | verdict |
|---|---|---|---|
| zenith | ~none (quadratic explains 2 %) | −0.02 (p 0.8) | null — the control behaves |
| eclipse | mild (10 %) | +0.14 (p 0.053) | a hint, not significant |
| zwo3 | unknown, 21 usable stars | −0.16 (p 0.49) | underpowered |

The frame-to-frame scatter is atmosphere/mount dominated, and neither term cares
about the local PSF — so a null here is the physically expected answer, not a
failure of the test. (`fwhm_vs_scatter.png` in each budget directory.)

**The static error: yes, clearly.** The dataset with real PSF variation
(rasalhague, 44 % FWHM variation, tilt+coma) has no per-frame data, but its
stage-2 residuals are static-dominated (50-frame stack: random ≈ 0.012 px ≪ the
0.057 px total). Matching each stage-2 star to its PSF fit
(`tools/residuals_vs_fwhm.py`, 547 stars):

- Spearman residual vs FWHM: **ρ = +0.163 (p = 1.3×10⁻⁴)**
- magnitude partialled: +0.158; **field-radius partialled: +0.186** — it is the
  *local* PSF, not a shared radial trend (partialling radius strengthens it)
- median residual, best third of FWHM: 0.065″; worst third: **0.085″** (+31 %)

So the static term (item 2 of the ranking) does live where the PSF is bad —
direct evidence for aberration-induced centroid bias, and for PSF-aware
centroiding / focus–collimation as the lever on long stacks.

## 2. What does stacking more frames buy? (rms vs N)

**Track level** (`stack_scaling.png`): disjoint groups of N frames, scatter of the
group-mean positions, bright stars, both alignment models, consecutive and
shuffled frame order:

- **zenith, translation-aligned: stacking barely helps.** 0.0895 px at N=1 →
  0.0775 px at N=5, against 0.0400 px if it averaged as 1/√N. Shuffling the frame
  order collapses it (0.0184 px at N=4) — the mount term is a temporally
  correlated *drift*, not noise; time-adjacent frames share it, so consecutive
  stacks cannot average it away.
- **affine-aligned residuals follow 1/√N cleanly** on every dataset (zenith
  0.0339 → 0.0134 px at N=5, prediction 0.0152) — after the affine, what is left
  behaves like noise and stacks properly.
- zwo3 (0.2 s frames) follows ~1/√N even translation-aligned: its floor is
  atmosphere (uncorrelated between frames), not drift.

One correction this forced to the earlier ranking: a *uniform* affine wander is
absorbed by stage-2's linear terms anyway, so per-frame affine registration mostly
buys back the per-frame scatter number and the stack's PSF sharpness — the part of
the drift that survives into final astrometry is only its non-affine residue.
The end-metric truth comes from the pipeline itself:

**Pipeline level** (`tools/stage2_vs_frames.py`): the real stack → solve → stage-2
ladder over the first N frames, fitted as rms(N)² = static² + random²/N:

| dataset | rms(N) measured | static (never stacks away) | random at N=1 |
|---|---|---|---|
| eclipse (10 s frames, 235 stars/run) | 100.8 → 91.8 → 82.6 → 78.7 mas (N=2,3,5,7) | **67.9 mas** | 105.9 mas |
| zwo3 (0.2 s frames, 69→183 stars)¹ | 301.0 → 262.0 → 202.2 → 168.2 mas (N=2,3,5,9) | **100.4 mas** | 405.4 mas |

¹ deeper stacks admit fainter stars, so the zwo3 rungs are not at constant star
population; treat its fit as indicative. The zwo3 row is also not exactly re-runnable: it was
measured from a curated 9-light, 38-dark copy that was deleted on 2026-08-21, and the source
folder holds 10 Center2 lights and 64 darks with no record of which subset was used. See
`tools/stage2_vs_frames.py`.

This is the "more information" the question hoped for, and it closes the loop on
the whole budget by an independent route: the fitted random-at-N=1 (105.9 mas =
0.092 px eclipse; 405 mas = 0.217 px zwo3) reproduces the per-frame floors the
track analysis measured (0.083 and 0.244 px), and the eclipse static asymptote
(67.9 mas = 0.059 px) reproduces the stage-2 excess estimated earlier from a
single stack (0.051 px). Stacking the eclipse field beyond ~10 frames buys almost
nothing: the predicted rms at N=100 is 68.7 mas, 1 % above the wall. **The number
of frames is not the limit; the static term is** — which section 1 just localised
to where the PSF is worst.

A pattern worth flagging: the static term is **~0.05 px on all three setups** —
0.059 px (eclipse, ASI294 + 65 mm quad), 0.055 px (rasalhague), 0.054 px (zwo3,
ASI1600 + 101 mm quad) — across different cameras, optics, platescales and
exposures. A term set by the optics alone would not land on the same *pixel*
value three times; one set in pixel units would. That points at the shared
pixel-level ingredients — the centroid estimator's response to the real
(asymmetric, field-varying) PSF, sub-pixel detector structure, or the
distortion-model truncation expressed in pixels — and it makes the PSF-aware
centroiding experiment (deliverable (d) follow-up) the sharpest next probe: it
attacks exactly the term every setup shares.

## A bug this hunt caught: fields near RA 0 could not solve

The zwo3 ladder initially failed outright: both solvers *found* the correct
solution (v2 with 259 non-redundant triangles of consensus) and then rejected it,
because `get_bbox`'s RA-wrap handling returned the 0.35°-wide sliver between the
corners nearest RA 0 instead of the field's true 3.8° extent — verification
fetched 51 catalogue stars instead of ~1000 and starved (6 matched, threshold 11).
Fixed in v1.3.3 (largest-gap wrap split; regression test with the exact corners);
the field now blind-solves with 99 matched against a threshold of 12. Every field
within ~2° of RA 0 was affected, in both solvers, silently.

## Removable vs pixel-level: the actionable split

The decomposition the two scaling laws make possible: at each dataset's current
frame count, how much of the final stage-2 rms is *removable* (keeps averaging
down with more frames) versus stuck at the static wall — and, within the wall,
how much is pixel-discreteness error that a better PSF-aware centroid method
could remove.

| dataset (current N) | total rms | removable by more frames | static wall |
|---|---|---|---|
| eclipse (N=7) | 78.7 mas | 40.0 mas — **26 %** of variance | 67.9 mas (74 %) |
| zwo3 (N=9, 0.2 s) | 168.2 mas | 135.1 mas — **65 %** of variance | 100.4 mas (36 %) |
| rasalhague (N=50) | 107 mas | 19 mas — **3 %** of variance | 106 mas (97 %) |

Reading: zwo3 should simply take more frames (still noise-dominated); the eclipse
set gains ~14 % rms at most from infinite further frames; rasalhague's 50-stack
gains nothing — it *is* the static wall.

**Inside the wall, pixel discreteness is measured ≈ zero.** The one error source
a PSF-aware fit uniquely removes — centroid bias from pixel sampling — has a
signature nothing else shares: dependence on subpixel phase, which a distortion
polynomial (period: thousands of px) cannot absorb (period: 1 px).
`tools/static_phase_bias.py` bins the signed stage-2 residuals (pixel frame,
recovered convention-free from the matched pairs) by subpixel phase, against a
permutation null:

| stack | per-frame phase bias | phase bias left in the *stacked* static error | p |
|---|---|---|---|
| eclipse n7 | 0.007–0.010 px | ≤ 0.010 px, indistinguishable from null | 0.76–0.88 |
| zwo3 n9 (FWHM 1.8 px!) | 0.034–**0.069** px | ≤ 0.018 px, excess 0.005 px | 0.42–0.96 |
| rasalhague n50 | — | ≤ 0.008 px, indistinguishable from null | 0.66–0.80 |

The undersampled set is the acid test: its per-frame discreteness bias
(0.069 px) would be ~half the static wall if it survived — and it doesn't.
**Dithered stacking already does the job an ePSF fit would do for pixel
discreteness**: each star lands at many phases, the bias averages out, and the
static wall that remains is *not* phase-locked (< 4 % of static variance,
bounded by the null sensitivity). This is also consistent with the deliverable-d
result that no estimator swap moved the random floor.

What the wall actually is, by elimination and by its measured signatures: it
tracks the local PSF (rasalhague: ρ +0.163, +31 % residual in the worst-FWHM
third) but not subpixel phase — i.e. **PSF-asymmetry bias** (an asymmetric PSF
pulls any centroid off the catalogue position in a field-dependent way that a
cubic polynomial only partly absorbs), plus whatever distortion-model truncation
and colour terms (chromatic refraction/dispersion) contribute. A PSF-aware
method can still attack the *asymmetry* part — by modelling the field-varying
kernel shape, not by fixing discreteness — with a realistic ceiling of roughly
10–15 % of the wall (the best-vs-worst-FWHM spread), not the 2× a naive reading
of "0.05 px ≈ pixel effects" would suggest. The earlier speculation that the
common ~0.05 px wall is pixel-*discreteness*-native is hereby measured down;
its pixel-unit universality across three setups still wants an explanation
(estimator response to asymmetric PSFs is itself pixel-native, as is sub-pixel
QE structure — but the phase-locked part of the latter is excluded too).

## The 0.2 s exposure lesson (zwo3)

The per-frame floor at 0.2 s is 0.244 px = 0.46″ — three times the 10 s sets —
and 70 % of its variance is spatially correlated: unaveraged atmospheric tip-tilt,
exactly as the physics predicts (tip-tilt variance ∝ 1/T once T exceeds the
turbulence timescale). Undersampled at FWHM ~1.8 px, its pixel-phase bias
(0.034–0.069 px) is the largest seen yet but still subdominant. Short exposures
also detect ~7× fewer stars (48 budget stars vs 383–455). At fixed total
integration the atmosphere term is exposure-neutral, but the detection depth,
saturation headroom and per-frame solve reliability are not — 10 s frames are the
better operating point for this pipeline unless saturation of the target stars
forces shorter.
