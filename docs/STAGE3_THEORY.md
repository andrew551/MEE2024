# Stage 3 and its ancestors: Freundlich & Ledermann 1944, Bruns 2017, and what the code computes

**Date:** 2026-08-25. Sources: F&L, *MNRAS* 104, 40 (read in full); Bruns, the 2017
measurement paper (read in full); `mee2024/eclipse_analysis.py` on `v1.4.0-dev`; Andrew's
2026-08-09 estimator analysis (`I:\MEE_transcripts\2026-08-09_mee2024-software-design-
overview_fdabe46a.md`, not independently reviewed except where marked). Both papers are in
`I:\Papers`.

The point of this document: every estimator in Tab 3 is an implementation of something in
the 1944 paper, mostly without saying so. Naming the correspondence settles several
questions that have been re-litigated from scratch — including twice this week — and it
turns two known peculiarities of the code into items with a literature answer.

---

## 1. The 1944 problem statement

F&L write the observed-minus-catalogue displacement of star *i* (origin at the Sun's
centre, r in solar radii) as a linear model — their equation (3):

    dx_i = N1 − (y_i−β)Θ + (x_i−α)S + (x_i/r_i²)L
    dy_i = N2 + (x_i−α)Θ + (y_i−β)S + (y_i/r_i²)L

Five unknowns: a pointing offset (N1, N2), a roll Θ, a scale change S, and the deflection
L. Their fuller equation (2) adds P and Q — quadratic plate-tilt terms — which they set
aside as "dealt with by other methods", i.e. by a calibration step. Everything since is a
choice about **which of these to fit from the eclipse stars and which to import**.

They analyse the two limiting cases:

**S free (their eqs. 7/12).** Fit everything from the eclipse field. The weight of L is

    w_L/w0 = n(1/h − 1/a) − asymmetry terms,     1/h = mean(1/r²),  a = mean(r²)

— the *difference between the reciprocal harmonic and arithmetic means of r²*, which is
small unless many stars sit very close to the Sun. Their verdict: the normal equations are
"critically close to [the] vanishing point" of their determinant. On the Lick field (92
stars) this route gives relative weight 1.80 where the alternative gives 3.4: **letting
the plate scale float roughly halves the weight of L**, and it costs more the narrower the
radial spread.

**S imported (their eqs. 15/20).** Determine the scale separately, fit only N1, N2, Θ, L.
The weight of L becomes n/h — no longer a difference of nearly equal numbers. But the
imported scale's error now propagates directly; their equation (23), symmetric case:

    δL = −h · δS,        h = 1/mean(1/r²)   (S in arcsec per solar radius)

"An initial error in S induces an error in L which is about r̄² times as great. Thus the
determination of S has to be carried out with the utmost accuracy." Their spec, for
σ_L ≤ 0.1″ on their three historical fields: **δS ≤ 12, 3.6 and 10 ppm** (Greenwich, Lick,
Potsdam). That is the origin of the ~10 ppm plate-scale target this project has been
quoting from memory.

Three more 1944 results that keep being rediscovered:

- **"A star whose distance from the Sun exceeds about 7.5 solar radii contributes
  practically nothing."** The leverage of a star on L goes as 1/r; wide fields add scale
  constraint, not deflection constraint.
- **The 1929 disaster is a units-free warning about calibration transfer.** Swinging the
  camera to the Hyades for a scale determination changed the focal length by 0.13 mm —
  **38 ppm** — and moved L from 2.45″ to 0.77″, a swing comparable to L itself. Their
  italicised conclusion: "*The scale of each plate must be determined in exactly the same
  position in which the exposures on the eclipse are made.*" The Potsdam fix was a grid
  exposed without moving the camera, good to 7 ppm.
- **The symmetric field is optimal, and asymmetry costs weight** through the extra terms
  in (12), (20) and (23). A one-sided star field (or a one-sided calibration) pays in
  exactly the quantity being measured.

## 2. Bruns 2017 as the F&L implementation

Bruns' reduction is the eq. 15/20 route executed as written, and his paper says so. The
mapping, with his numbers:

| F&L | Bruns 2017 |
|---|---|
| P, Q "by other methods" | night-time cubic calibration, coefficients frozen (his Table 2) |
| separate S determination | RIGHT and LEFT fields at ±7.4°, during totality, averaged — eclipse field midway |
| δS accuracy | **3.34 ppm** by his moment formula (rms·√N/Σd over 96 stars) |
| eq. 23: δL = −h·δS | 3.34 ppm × R⊙ = 0.00317″ → **1.23 %** of L |
| eq. 20: fit variance | eclipse rms 0.088″ at 1 R⊙ → **3.1 %** |
| combined | **3.4 %**, the published uncertainty |
| eq. 12 route, for comparison | same data, scale free: **L = 1.86″, ~4 %** (8 % without the two close stars) |

The last row matters most for MEE: **Bruns ran both estimators on his own data**, and the
scale-free route moved L by 6 % and roughly doubled its uncertainty — the empirical form
of F&L's degeneracy argument. His remark that omitting the L/R calibration would have
changed L "by about 1 %" is the same statement from the other side.

Two design details that are easy to miss and worth stealing: the RIGHT field was chosen at
**the same altitude** as the eclipse field so refraction largely cancels between them
(the §19.2 common-mode argument, 1917-style); and his two stars at 1.5 R⊙ sat **nearly
opposite each other** through the Sun, so the translation error of their separate
short-exposure alignment cancelled by a factor of 17 in L.

## 3. What `eclipse_analysis.py` actually computes, term by term

Three estimators of L run in one function, plus one displayed number that is not an
estimator at all.

**(a) The Nelder–Mead fit** (`error_function1`, line ~196). For each trial L: deflect the
catalogue by L/r, solve the best rotation obs→catalogue (`_find_rotation_matrix`), take
the rms. This fits F&L's N1, N2, Θ jointly with L — their **B-matrix of eq. 17**, solved
by optimiser instead of normal equations. `error_function3` adds a B·r term (A/r + B·r),
which is the **eq. 3 model with S free**. So the two historical cases are both present.

**(b) The OLS refits** (`analysis_mode_1/2`, lines ~307/335). Method 1 fits [1/r] with the
plate scale frozen — **eq. 15/20**; its `plate_covariance2` term is **eq. 23**
(`fix-platescale-covariance-units` corrects its units: the solar radius entered in pixels
where arcseconds are required, understating δL·δS by exactly the plate scale). Method 2
fits [1/r, r] — **eq. 3/7/12**. The reported covariances are the least-squares variances,
i.e. eqs. 20 and 12 respectively: *the code already computes the F&L uncertainties*, and
no separate implementation of the closed forms is needed or wanted (they require choosing
the right case by hand; the covariance derives both from the fit).

**(c) The conditioning chain, confirmed.** `deflection_obs` (line ~252) is built from the
rotation solved at the Nelder–Mead optimum of (a); the OLS fits of (b) then re-fit L on it
**without re-fitting the rotation**. The second estimate is conditioned on the first — the
circularity flagged in Andrew's 2026-08-09 analysis, verified here against the source. Its
practical size is second-order (a rotation is nearly a uniform translation over this
field, and a translation projects onto the Sun-radial direction as a dipole, largely
orthogonal to 1/r), but the estimator is not the clean one, and its covariance does not
know about the conditioning.

**(d) `naive_error`** (line ~255): `rms/√N`, printed on the Tab 3 plot as "naive
uncertainty estimate". **This is not a valid estimator of σ_L and should not be
displayed.** A star's leverage on L scales as 1/r, so the correct denominator is
√(Σ 1/r²), not √N; the ratio between them is about the field's effective solar-radius
distance — ×2.9 optimistic for Bruns' 20 stars, more for wider fields. F&L's Table I is a
table of exactly this distinction. (An earlier statement in this project that "σ/√N is not
in the code" was wrong; it is here, as display only. The OLS covariances beside it are the
valid ones.)

## 4. The sensitivity closure: h is the whole story

Three sensitivity figures have circulated this week for "% of L per ppm of plate-scale
error": ~0.99 (analytic, this project), 0.84 (Monte Carlo, Leon geometry), 0.37 (implied
by Bruns' published 1.23 %/3.34 ppm). All three are F&L's h with different star fields:

| field | h = 1/mean(1/r²) | δL/L per ppm |
|---|---|---|
| Leon annulus, uniform in r, 2–9 R⊙ | 18.0 | 0.99 % |
| Leon, uniform on the sensor, offsets absorbed (MC) | ~15 | 0.84 % |
| Bruns' 20 stars, from his Tables 5–6 | 8.2 | — |
| Bruns effective (offsets absorbed), from his published numbers | 6.8 | 0.37 % |
| 1929 Potsdam field (30 stars, mean 6.7 R⊙) | ~44 implied | 38 ppm → 1.68″ ✓ |

The factor ~2.5 between Leon and Bruns is not a disagreement about physics; it is that
**Bruns' stars sat closer to the Sun**, and h is a property of the star field. The 1929
row is the same formula reproducing the historical failure to 10 %.

Consequences, stated once:

- F&L's own spec applied to the Leon field: **δS ≤ 5.8 ppm** for σ_L ≤ 0.1″. CAL_piLeo
  currently delivers 8–13 ppm (Bruns' moment formula) to 19–20 ppm (OLS standard error),
  so the plate-scale term on L is **5–20 %** — the dominant Method 1 uncertainty, and the
  reason the combined step-2 reduction and its recorded uncertainty matter more than any
  other single number.
- The 1929 lesson lands differently on Leon than on Bruns. Bruns *swung the camera* ±7.4°
  for his scale fields — the very act F&L's italics warn about — and survived it by
  bracketing (his L and R plate scales differ by ~31 ppm in this project's re-reduction;
  their average sits at the eclipse field). CAL_piLeo involves **no swing**: same
  pointing to within a degree, same `FOCUSPOS`, minutes apart — the F&L "same position"
  ideal. Its weakness is the complementary one: a single field cannot bracket a spatial
  gradient, and the eq. 23 asymmetry terms it inherits are not zero. Neither design
  dominates the other; they fail differently, which is worth a sentence in any 2027 plan.

## 5. What follows for the code (proposals, not changes)

1. **Remove or relabel `naive_error`** on the Tab 3 output. It understates σ_L by the
   field's effective radius and sits beside two valid estimates. (§3d.)
2. **The one-linear-model refactor is F&L eq. 3.** Andrew's 2026-08-09 proposal — one
   design matrix over 2N residuals with columns L·(1/R)û, S·Rû, T, ω(ẑ×r) — is precisely
   the 1944 A-matrix solved by `lstsq`. It removes the §3c conditioning, fits both
   components with honest degrees of freedom (2N−5), and makes Method 1 vs Method 2 a
   column choice rather than two code paths. The 1944 paper is the design document.
3. **Jackknife over stars** as the quoted uncertainty: resample, refit end-to-end, take
   the spread. Captures the 1/r lever arm, the L–S degeneracy, non-Gaussian residuals and
   single-star leverage without selecting an F&L case at all; nearly free at stage-3 cost.
   Leave-one-out doubles as the diagnostic for results driven by a handful of stars.
4. **Quote Method 1 as the result wherever a calibration exists; run Method 2 alongside**
   — both papers' practice, and Bruns' own eq. 12 comparison is the demonstration of why.
   Their *gap* is diagnostic (linear in both the cubic-transfer error and the scale
   error), their agreement is not validation.

None of this is implemented here; this document is the theory record. The one code change
in flight is the eq. 23 units fix on `fix-platescale-covariance-units`.
