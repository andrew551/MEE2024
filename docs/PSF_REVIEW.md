# The point spread function: what the literature says, and what it means for this pipeline

A survey of accepted practice in PSF modelling and star centroiding, written to answer one
question: **what should MEE2024 do about the PSF?** Our regime frames every judgement below:
seeing-limited amateur refractors, platescales of 1.15–1.85 ″/px, exposures of 1–10 s,
stacks of a handful of frames, and *astrometry as the product* — centroid accuracy is the
whole game, photometry is incidental. The stage-2 baseline to beat or protect is 109.6 mas
rms (~0.06 px) on the reference field.

---

## 1. The fundamental limit, and how close simple methods get

The precision of any centroid estimate is bounded below (Cramér–Rao) by roughly

    σ_pos ≈ FWHM / (2.355 · SNR)        (bright, background-free limit)

with SNR the total signal-to-noise of the detection; in the background-dominated faint
limit the scaling worsens to FWHM²·√(background)/flux. Two consequences matter here:

* **A star at SNR 100 with FWHM 3 px carries ~0.013 px of information.** Our measured
  frame-to-frame alignment rms is ~0.1 px, so there is roughly a factor of five between
  what the photons permit and what we extract — some of it atmospheric (differential tip),
  some of it estimator inefficiency. Which, is measurable, and deliverable (d) measures it.
* **The bound is achievable by ordinary methods in the well-sampled, bright regime.**
  Vakili & Hogg ([arXiv:1610.05873](https://arxiv.org/abs/1610.05873)) compared fast
  centroiding methods against the bound directly: fitting even an *approximate* profile
  gets within a few percent of the bound at moderate-to-high SNR, while **plain
  center-of-mass does not saturate the bound**, and degrades sharply once the window
  includes noise-dominated pixels. The gap between COM and a fit is largest exactly where
  we operate: faint stars in a windowed cutout.

A useful review of the bounds from first principles: [arXiv:2512.04326](https://arxiv.org/pdf/2512.04326).

## 2. The estimator menu, in increasing order of sophistication

**Plain center of mass (what we do).** Unbiased only if the window is symmetric about the
true position (it never is — the window is placed by the detection) and the background is
exactly zero. Sensitive to the "S-curve" pixel-phase bias under discrete sampling, and its
noise grows with window radius² since edge pixels contribute variance but almost no signal.
Every comparison in the star-tracker literature finds it inferior to a Gaussian fit
([overview](https://quantumzeitgeist.com/centroiding-algorithms-compared-for-star-tracker-performance-with-gaussian-noise/));
its virtues are speed and having no model to be wrong about. Our variant operates on
background-subtracted, variance-normalised, thresholded pixels, which blunts the worst of
the background sensitivity but keeps the hard-window noise behaviour.

**Iteratively re-weighted (windowed) centroid.** A Gaussian weight centred on the current
estimate, updated until convergence — SExtractor's `XWIN_IMAGE`. The documentation and
independent tests find its accuracy "very close to that of PSF-fitting on focused and
properly sampled star images", and close to the noise limit for isolated Gaussian-like
profiles ([SExtractor docs](https://sextractor.readthedocs.io/en/latest/PositionWin.html)).
This is the classic accuracy-per-line-of-code winner: no PSF model, one tuning parameter
(the window width, from the measured FWHM), a three-line loop.

**Analytic profile fits.** Least-squares fit of a Gaussian or Moffat to the cutout. For
ground-based data the **Moffat** profile, I(r) = I₀·(1+(r/α)²)^(−β), is the standard:
turbulence theory predicts β ≈ 4.765, and real telescopes show β ≈ 2.5–4.5 because optics
add wings the Gaussian entirely lacks (Trujillo et al. 2001,
[MNRAS 328, 977](https://academic.oup.com/mnras/article/328/3/977/1247204); the Gaussian is
the β→∞ limit). Two practicalities the star-tracker literature emphasises: at FWHM ~2–3 px
the model must be **integrated over pixels**, not sampled at pixel centres, or the fit
inherits a phase-dependent bias; and closed-form/fast variants exist that reach fit-quality
accuracy at a fraction of the cost ([fast Gaussian fitting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6163372/)).
For *centroiding* (not photometry) the Gaussian core fit and the Moffat fit typically
recover indistinguishable positions on symmetric PSFs — wings carry position information
only when the PSF is asymmetric.

**Effective PSF (ePSF).** Anderson & King 2000
([PASP 112, 1360](https://iopscience.iop.org/article/10.1086/316632)): build an empirical,
oversampled model of the *pixel-sampled* PSF from many star images at different subpixel
phases, iterating between the model and the star positions. It absorbs undersampling,
intra-pixel sensitivity and PSF asymmetry, and reached **sub-millipixel** centroiding on
HST — an order of magnitude beyond analytic fits *on undersampled data*. The reference
implementation is photutils'
[`EPSFBuilder`](https://photutils.readthedocs.io/en/latest/user_guide/epsf.html), already in
our dependency tree. The caveats: it needs many bright, isolated stars at diverse pixel
phases (a dithered stack has exactly this), and its advantage over an integrated analytic
fit shrinks toward zero once the PSF is well-sampled (FWHM ≳ 2.5 px) and near-Gaussian —
recent work on undersampled ePSF strategies is explicit that undersampling is where it earns
its keep ([RASTI 2025](https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzaf063/8400331)).

**DAOPHOT-style hybrid** (Stetson 1987): analytic core plus an empirical residual lookup
table. Historically dominant in crowded-field photometry; for sparse-field astrometry it is
more machinery than the problem needs.

## 3. PSF variation across the field

The lensing surveys own this problem, and their architecture is consistent:
model the PSF per star, then **interpolate the model parameters across the field with a
low-order bivariate polynomial** — degree 2–3 per CCD in PSFEx
([Bertin 2011](https://www.researchgate.net/publication/252992913_Automated_Morphometry_with_SExtractor_and_PSFEx)),
per-exposure in Piff (Jarvis et al. 2020, used for
[DES Y3](https://par.nsf.gov/servlets/purl/10288837)), with PCA/pixel bases when the
polynomial is not enough. Validation is by held-out stars and **whisker plots** (ellipticity
sticks across the field) plus residual maps — the diagnostic style worth copying even at our
scale.

For *this* pipeline the stakes differ from lensing. Two distinct questions:

1. **Does the PSF *width* vary?** (focus gradient, field curvature, tilt). Affects detection
   completeness and per-star weight, but a symmetric size change does **not** bias a
   centroid. Diagnostic value: high. Astrometric bias: none to first order.
2. **Does the PSF *shape* (asymmetry) vary?** Coma-like asymmetry displaces every centroid
   estimator by a shape-dependent fraction of a pixel, *differently across the field*. A
   **static** asymmetry pattern is absorbed by our distortion polynomial (it is exactly a
   smooth field-dependent displacement — invisible in residuals, harmless to the eclipse
   measurement as long as calibration and eclipse fields share it). What is *not* absorbed:
   asymmetry patterns that change between nights/pointings (focus, flexure), and
   magnitude-dependent centroid shifts (saturation, nonlinearity) which the polynomial
   cannot represent at all. The literature's magnitude-dependent checks (bright/faint PSF
   consistency in the DES papers) are the model to follow.

So the zeroth-order deliverable (constant PSF: one FWHM, one ellipticity, one β) is
genuinely useful — it is the focus/sampling diagnostic — and the first-order deliverable
(low-order polynomial maps of FWHM and ellipticity) is the standard practice, with whisker
plots as the display.

## 4. Sampling: the regime decides the method

The single most decision-relevant number is **FWHM in pixels**:

| FWHM | regime | best practice |
|---|---|---|
| < 2 px | undersampled | ePSF is the only honest tool; analytic fits carry pixel-phase bias even integrated |
| 2–3 px | critically sampled | integrated analytic fit or windowed centroid ≈ ePSF; COM measurably worse |
| > 3 px | well sampled | windowed centroid ≈ Gaussian fit ≈ ePSF; differences second-order |

Our platescales (1.15–1.85 ″/px) against typical seeing (2–4″) put us anywhere from 1.3 to
3.5 px FWHM — **straddling the boundary**, possibly differently per dataset. This is why
deliverable (b) measures it per dataset before (d) picks algorithms: the literature's answer
to "which centroider?" is "it depends on exactly this number".

Saturated stars are a special case the pipeline already trips over (Rasalhague): COM on a
clipped core is biased toward whichever wing pixels survive, while a PSF fit to the
*unsaturated* pixels recovers the position — the standard trick for bright-star astrometry
and a concrete potential win for (d).

## 5. What this project should do — the plan

Ordered by value per unit risk, informed by the above:

1. **(b) Measure first.** Per dataset: FWHM (px and ″), ellipticity + angle, Moffat β from
   stacked bright-star profiles, and the constant-vs-varying comparison (polynomial maps +
   whisker plot + variance explained). No pipeline changes; a `tools/` script and figures.
2. **(c) Surface the cheap, decision-grade numbers in the UI.** Median FWHM in arcsec (the
   "seeing" number every observer knows), FWHM in px with an **undersampling warning below
   2 px** (it changes what the pipeline should do, per §4), ellipticity (focus/tracking
   diagnostic), and a PSF panel: mean radial profile with Gaussian/Moffat overlays, FWHM
   map, whisker map. All computable from cutouts already in hand during stage 1.
3. **(d) Evaluate estimators against ground truth before adopting anything.** Synthetic
   frames with pixel-integrated Moffat stars at measured β/FWHM/noise (truth known exactly),
   plus the real-data proxy: per-star centroid scatter across the 7-frame dithered set.
   Candidates, cheapest first: current COM path (baseline), Gaussian-windowed iterative COM
   (SExtractor-style — the literature's pick for our likely regime), integrated-Gaussian
   fit, photutils ePSF. Adopt into the pipeline only what beats the baseline on *both*
   synthetic truth and real-frame scatter, and only behind an option with the COM path as
   default until the stage-2 regression numbers are reproduced or beaten.

The honest expectation from the literature: if (b) finds FWHM ≥ 2.5 px, the windowed
centroid captures most of the available gain at trivial cost and complexity, and ePSF buys
little; if (b) finds FWHM < 2 px, ePSF is the correct tool and the win could be large. In
both cases the saturated-star fit is a separate, likely-worthwhile add-on.

---

## 6. Measured on our data (deliverable b)

`tools/psf_explore.py`, run on all three bundled datasets; figures and per-star JSON under
`docs/bench/psf/`. The summary table:

| dataset | FWHM (px) | FWHM (″) | ellipticity | Moffat β | constant PSF? |
|---|---|---|---|---|---|
| zenith 2026-07-19 (3008², 14-bit) | **1.22** ± 0.06 | — | 0.070 | (unreliable: undersampled) | **yes** — quadratic explains 2% |
| Rasalhague 50-stack (3520×4656) | 2.41 ± 0.27 | 3.98 | 0.152 | 2.96 | **no** — quadratic explains 44% of FWHM scatter |
| eclipse field (5644×8288, 12-bit) | 2.97 ± 0.17 | 3.42 | 0.094 | 2.77 | mostly — quadratic explains 10% |

What the measurements settle:

* **Sampling is not a property of "our data" — it is a property of each setup, and it
  spans the entire decision table in §4.** The newest dataset (the zenith set, likely the
  current instrument) sits at **FWHM 1.22 px, severely undersampled**, exactly where plain
  COM suffers pixel-phase bias and where the ePSF is the honest tool. The other two sit at
  2.4–3.0 px, where a windowed centroid ≈ a fit. The pipeline must therefore *measure*
  FWHM per run and warn — it cannot assume a regime. (This is now the strongest argument
  for deliverable (c)'s sampling metric.)
* **The wings are real and the profile is Moffat, β ≈ 2.8–3.0** on both well-sampled sets —
  heavier wings than pure seeing (β 4.77), i.e. the optics contribute, and a Gaussian
  overestimates FWHM by ~15% here. Centroid-wise the symmetric wings are harmless; for any
  future PSF *photometry* they are not.
* **PSF variation across the field is real where it matters least and absent where it
  would matter most.** The Rasalhague optic shows a textbook tilt+coma signature — FWHM
  rising from 2.3 to ~3.5 px toward one edge, ellipticity whiskers pointing radially from a
  decentre — and a quadratic surface captures 44% of the FWHM scatter. The zenith set is
  uniform to 2%. Since a *static* asymmetry pattern is absorbed by the distortion
  polynomial (§3), the immediate value of the maps is diagnostic: they tell the observer
  their focuser is tilted before a night is spent, which is exactly what a UI panel is for.
* Ellipticity medians of 0.07–0.15 are honest optics-quality numbers, worth showing beside
  FWHM as the second score-card figure.


### 6.1 Deliverable (d), real-data half: done, and the windowed centroid wins

Run on the Leon zenith set 2026-08-23 — twelve field-nights, detection held fixed so only the
position estimator changes. Full numbers in [`LEON_2026-08-11.md`](LEON_2026-08-11.md) §18.3.

**§3's predicted failure mode is the one that was actually there.** That section warned about
"magnitude-dependent centroid shifts … which the polynomial cannot represent at all". Measured:
beyond r = 2500 px the brightest stars sit 172 mas (night 1) and 299 mas (night 2) outward of the
faintest, in **twelve fields out of twelve** — and because a polynomial cannot represent it, the
fit absorbed it into the cubic coefficient instead, making the calibration a function of
`max_star_mag_dist` at the 4.5% level. It left no signature in the rms or the residual maps.

The mechanism is §2's own caveat about plain centre-of-mass, quantified: *"unbiased only if the
window is symmetric about the true position (it never is — the window is placed by the
detection)"*. Our footprint is an S/N-thresholded connected region weighted by
`max(S/N − sigma_subtract, 0)`, so its **size and its weighting both scale with brightness**.
Where the PSF is asymmetric the two populations then measure different parts of it.

**Gaussian-windowed iterative centroid, sigma 2.0 px, same detections:**

| mag 13, cubic, tol 0.2 | stars | rms | median residual | bright − faint |
|---|---|---|---|---|
| current COM path | 2034 | 83.8 mas | 54.9 / 70.8 mas | +97 / +106 mas |
| **windowed** | **2196** | **66.2 mas** | **43.2 / 52.0 mas** | **+15 / +8 mas** |

It beats the baseline on every axis at once — bias 11–13× smaller, rms −21%, median −21/−27%,
and *more* stars surviving the tolerance rather than fewer. It also removes the sensitivity that
mattered most: the cubic now moves 0.44% across mag 11–15, against 2.18% across mag 11–13 before.

Two further results bearing on §5's expectations:

- **The regime call in §6 was right.** Leon's stacks are FWHM 2.18 and 2.38 px, just inside where
  §5 expects "the windowed centroid captures most of the available gain at trivial cost and
  complexity, and ePSF buys little". It did.
- **The Moffat wings are not harmless here after all.** §6 records "centroid-wise the symmetric
  wings are harmless" — true, and the operative word is *symmetric*. Fitted in quadrature the
  Leon blur grows linearly with field radius, i.e. coma, and growing the centroiding aperture
  walks the centroid radially outward by up to 631 mas at the corner before saturating at 6–8 px.
  Asymmetric wings carry position information, and a brightness-dependent footprint reads a
  different amount of it from every star.

**Not done, and still required before this becomes a default:** the synthetic-truth half of
§5(d) — pixel-integrated Moffat stars at measured beta/FWHM/noise, where truth is known exactly.
The real-data proxy above establishes that the windowed centroid is better on this data; it
cannot establish that it is unbiased in an absolute sense, which is what synthetic frames are
for. Adopt behind an option with COM as the default, per §5(d), until both halves are in.
Tracked as F15 in [`ROADMAP.md`](ROADMAP.md).

### Sources

- [Anderson & King 2000, PASP 112, 1360 — toward high-precision astrometry with WFPC2 (the ePSF)](https://iopscience.iop.org/article/10.1086/316632)
- [photutils: building an effective PSF](https://photutils.readthedocs.io/en/latest/user_guide/epsf.html)
- [Strategies for accurate ePSF modelling on undersampled images (RASTI, 2025)](https://academic.oup.com/rasti/article/doi/10.1093/rasti/rzaf063/8400331)
- [Vakili & Hogg 2016 — do fast stellar centroiding methods saturate the Cramér–Rao bound?](https://arxiv.org/abs/1610.05873)
- [Review of fundamental bounds and estimators for photometry and astrometry (arXiv:2512.04326)](https://arxiv.org/pdf/2512.04326)
- [Trujillo et al. 2001, MNRAS 328, 977 — the Moffat PSF and seeing (β=4.765)](https://academic.oup.com/mnras/article/328/3/977/1247204)
- [SExtractor documentation — windowed positional parameters](https://sextractor.readthedocs.io/en/latest/PositionWin.html)
- [Bertin 2011 — automated morphometry with SExtractor and PSFEx](https://www.researchgate.net/publication/252992913_Automated_Morphometry_with_SExtractor_and_PSFEx)
- [DES Y3: PSF modelling (Piff in production)](https://par.nsf.gov/servlets/purl/10288837)
- [Fast Gaussian fitting for star sensors (closed-form fit accuracy at COM-like cost)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6163372/)
- [Star-tracker centroiding algorithm comparison overview](https://quantumzeitgeist.com/centroiding-algorithms-compared-for-star-tracker-performance-with-gaussian-noise/)
- [LSST DMTN-045 — PSF fitting literature overview](https://dmtn-045.lsst.io/)
- [Bruns 2018 — gravitational deflection measured at the 2017 eclipse (this experiment's closest relative)](https://arxiv.org/pdf/1802.00343)
