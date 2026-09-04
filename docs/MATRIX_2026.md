# The four-dataset matrix: one tool chain, four skies

**Started 2026-08-29.** The acceptance test for the Leon step-3 machinery
(`docs/STEP3_2026.md`): the same tools with the same defaults must reduce three solar
eclipses and one full moon with minimal hand tuning — hand tuning being what attracts
the accusation of manipulation. The convex-hull blob retires when this passes.

| cell | data | calibration geometry | known answer |
|---|---|---|---|
| 1. Bruns 2017 (Casper, WY) | `I:\2017 eclipse images Don Bruns\2017 Eclipse images\eclipse` | L/R bracket (±7.4° both sides) | Bruns 2018: GR to ~3 % |
| 2. Mexico 2024 (Station 1) | `G:\Mexico April 2024\Station-1-Eclipse-Data` | pure Method 2 — opened 2026-09-02: zenith floor 0.076″, Method-2 null ±0.10″, moment bias worth 0.011″ | GR-consistent (2024: 1.854 on 74 stars; eclipse scale −600 ppm from the zenith, Method 1 impossible) |
| 3. Leon 2026 | done — `docs/STEP3_2026.md` | one-sided CAL_piLeo | L = 1.914 ± 0.637 (stat) ± 0.675 (scale) ± 0.33 (atm) ″ — scale term added 2026-09-01, two-witness rule adopted 2026-09-02 |
| 4. Portland moon 2026-07-29 | `J:\Eclipse data\Toby Portland data\2026-07-29` | Moon-centric solver | **L = 0** (null check) |

Rule: everything that CAN be held constant from Leon IS; every departure is listed with
its reason. Tools live in `tools/matrix_bruns/` (and siblings as cells start).

## Cell 1 — Bruns 2017

### What already existed (2026-08-25 instrument-comparison week, all on `v1.4.0-dev`)

Steps 1–2 were complete before this session (`docs/STEPS12_LEON_VS_BRUNS2017.md`):
29 night fields (`bruns2017_nights/`), and the eclipse-day L/R bracket
(`bruns2017_lr/`): canonical `L/stage2` = 2.0867534 ″/px (105 stars, rms 0.220″),
`R8/stage2` = 2.0868474 ″/px (110 stars, rms 0.235″), **bracket mean 2.0868004 ″/px**,
combined HC3 ~10.3 ppm, L−R split 45.0 ppm (half-width 22.5 ppm = the measured bound on
the linear atmospheric differential, which the bracket cancels). Both froze their
above-quadratic distortion from night reference EC06 — the exact analogue of Leon's
zenith freeze. Site: 42°44′11″ N, 106°19′05″ W, 2400 m; 13.0 °C, 770.0 mb, humidity
0.4; λ 0.625 µm; refraction + aberration ON; no darks, no flats.

### S0 inventory (`tools/matrix_bruns/b17_inventory.py`, raw frames read-only on I:)

The ladder: **EA 17 × 0.62 s (mid 17:43:22 UT) → E2 11 × 0.09 s (17:43:47) → EB
17 × 0.62 s (17:44:13)**, all full-frame 3296 × 2472 (the 1400 × 1400 subframe in
Bruns' own acquisition script was not used). EXPTIME headers honest (filename = header,
every frame — Leon's trap checked, absent here). Filename timestamps are local
seconds-after-midnight (MDT = UTC−6); DATE-OBS (UTC) is the authority.

Measured saturation, PS = 2.0868 ″/px, R⊙ ≈ 948″ = 454 px:

| tier | sat radius (99th pct) | in R⊙ | sky at 2.0–2.5 R⊙ (ADU) | sat centroid (px) |
|---|---|---|---|---|
| EA 0.62 s | 901 px | **1.98** | 24 860 | (1637, 1749) |
| E2 0.09 s | 646 px | **1.42** | 4 497 | (1651, 1748) |
| EB 0.62 s | 902 px | **1.98** | 24 590 | (1644, 1748) |

Two design facts fall out immediately: (a) Bruns' fast optics saturate the deep tiers
to ~2.0 R⊙ — the forbidden-disk default (R > 2 R⊙ science cut) costs him nothing in
the 0.62 s tiers; (b) the inner annulus (1.5–2.0 R⊙, his famous near-Sun stars) is
reachable **only** through the 0.09 s series — which is exactly why his script has a
dedicated "2-star series". The E2 tier is his design's answer to the same problem
Leon's 0.1 s tier addressed, and here it carries real value.

### Held constant / departures

Held from Leon: blur-subtraction (tier mean, 10 px Gaussian, +2000 ADU pedestal);
forbidden disk max(1.25 R⊙, sat radius + 20 px); stage-1 flags verbatim; stage-2
constant-only, cubic, tol 2.0, mag-13 matching, corrections ON; two-pass rematch with
wide-gate 8″ offset pass and 4.5″ collect; per-star cross-tier medians, 3×MAD vet
(1.5″ floor); doubles 10″; mag ≤ 11; R > 2 R⊙ default; estimator
[N1, N2, Θ, L·(1/R)û + deg-2 nuisance]; 200-star bootstrap.

Departures, each with its reason:
1. **Frozen reference = the L+R8 pair** (both passed to `--fix-distortion`; the mean is
   the bracket working as designed) where Leon had only the one-sided CAL.
2. **Nuisance direction = computed local vertical** (from the field's AltAz geometry)
   instead of hardcoded sensor-y. Leon's sensor-y *was* the local vertical to 3.1°;
   same physics, dataset geometry supplied instead of assumed.
3. **Disk centre = per-tier saturated-pixel centroid** (tiers agree to 14 px) with the
   post-solve Sun ephemeris printed as the check, instead of Leon's pre-measured
   ephemeris constant.
4. **An 'inner' variant (R > 1.45 R⊙) is reported alongside the default** — E2's
   saturation is 1.42 R⊙ and its annulus is the dataset's highest-leverage region; the
   vet and the value table decide admission, per the Leon anchor doctrine.

### Results (2026-08-29, first pass — `b17_s0.py`, `b17_union.py`, `b17_check.py`)

Stage 1 → 2, constant-only against the frozen L/R bracket (imported ps 2.0868004 ″/px):
EA 60 centroids → 33 matched, rms 0.4895″; EB 50 → 31, rms 0.7123″; E2's own plate
solve FAILED (the Leon shallow-tier failure mode, by design absorbed by the union: its
detections ride EA's host model). Host round-trip gate 0.0000″. Sun ephemeris pixel
(1645, 1741) — within 4–11 px of the three saturated-pixel centroids, mask centres
validated. Local vertical is +153.5° from sensor +y (his camera roll); the nuisance
runs along it. Pipeline per-tier stage 3, first look: Method 1 L = 1.377 (EA) / 1.398
(EB) — the stage-3 flag_is_outlier gap and no vetting apply, superseded by the union.

**Pure-L injection self-test on this geometry: exact (1.7512 recovered to 4 decimals at
every nuisance order 0–3).** Stars surround the Sun here (unlike Leon's one-sided
field), so L is cleanly identifiable against the nuisance. Residual rms about the full
model: **0.219 ″/axis** (Leon: 0.28–0.35 — the wavefield is smaller at airmass 1.23,
though not gone).

| variant | N | h (R⊙²) | L base (″) | L v-deg2 (″) ± stat |
|---|---|---|---|---|
| 0.62 union, R > 2.0 (the held-constant default) | 25 | 10.6 | +1.303 ± 0.339 | **+1.556 ± 0.135** |
| FULL union, R > 2.0 (E2 adds nothing outside 2 R⊙) | 25 | 10.6 | +1.282 | +1.549 ± 0.139 |
| FULL union, R > 1.45 (both inner stars in) | 27 | 8.4 | +1.273 | +1.390 ± 0.232 |
| FULL union, R > 1.45, sans G 7.52 | 26 | 9.5 | +1.445 ± 0.306 | **+1.662 ± 0.144** |

eq-23 plate-scale term at the corrected units: h·R⊙·δS = **0.10″** at the bracket's
10.3 ppm HC3 (0.23″ at the 22.5 ppm L−R half-split, the conservative bound). Compare
Leon's ~0.4″-class term — the bracket + higher leverage cut it 4×.

**The two inner stars (E2-only, single-witness — the vet cannot arbitrate, so each is
measured individually, per the anchor doctrine):**

> **SUPERSEDED 2026-08-30.** The first-pass paragraph here blamed G 7.52's +0.04″
> reading on the painted disk edge ("mask-edge casualty"). Douglas rejected the
> exclusion and demanded the redo; the follow-up measurements cleared the suspects one
> by one: NOT saturation (raw peak 24.6 kADU, zero clipped pixels — F16 innocent), NOT
> an oversized mask (the saturated-fraction radial profile shows 2 % of pixels in its
> 650–700 px ring genuinely saturating; the disk edge was honestly placed), and NOT
> the blur-subtraction either — the per-frame path-bias measurement
> (`b17_perframe2.py`, identical measurement run on raw and on preprocessed frames)
> puts the held-constant preprocessing's centroid bias at **≤ 0.014″ on every star,
> both inner ones included**. The stacked chain's +0.04″ was a *stack-path* artifact
> (prime suspect: the 0.09 s stack's alignment quality — few usable stars, ±1″ of
> measured common frame-to-frame jitter), not a preprocessing casualty.

The verified measurements (`b17_perframe2.py`: each raw E2 frame measured
independently — quadratic local background on the window border ring, saturated pixels
excluded, Gaussian-weighted centroid σ 2.0 px — each frame referenced to the median
offset of its own R > 2 R⊙ stars; median of 11 frames, MAD/√n error):

- **G 7.09 at 1.62 R⊙: +1.632 ± 0.204″ radial** (GR: +1.079) — robust inner-annulus
  deflection detection, running +0.55″ high (local quasi-static structure suspected;
  the stacked value +1.44 agrees to 0.19″).
- **G 7.52 at 1.49 R⊙: +0.463 ± 0.229″ radial** (GR: +1.174) — a real, non-zero
  deflection, 3 σ low of GR. ~2.7 kADU of star on a ~22 kADU structured sky: the
  spread across measurement paths (~0.4″) is its honest systematic. Kept, with that
  weight understood — one noisy star among 27, not an anchor.

With both stars carried at their per-frame-verified vectors
(`b17_inner_refit.py`): **INNER union N = 27, h = 8.4 R⊙²,
L base +1.363 ± 0.274, L v-deg2 +1.519 ± 0.153 (stat) ± 0.083 (scale) ″.**

**Cell-1 verdict (revised 2026-08-30 after the per-frame redo): L = 1.556 ± 0.135
(stat) ± 0.103 (scale) ″ on the held-constant default (25 outer stars, GR at +1.2 σ),
and L = 1.519 ± 0.153 ± 0.083 ″ with the full inner annulus carried at per-frame-verified
values (27 stars, GR at +1.35 σ). The two variants agree to 0.04″. Newton (0.876″) is
excluded at ≈ 3.7–4.0 σ — the Eddington discrimination Leon's atmosphere forbade,
delivered by altitude 54° plus the L/R bracket, exactly the 2027 forecast.** (The
first-pass "1.66 sans G 7.52" variant is superseded — its exclusion rationale was
wrong.) Bruns' own published result (GR to ~3 %) remains far tighter: more stars,
fainter magnitudes, and his continuity correction working per image.

Charts: `matrix_bruns2017/deflection_b17.png`, `field_b17.png` (D:).
Union table: `matrix_bruns2017/union_full_r145.csv`.

Open items for this cell: (1) ~~per-frame E2 verification~~ **done 2026-08-30**
(`b17_perframe2.py`, `b17_inner_refit.py`); (2) the honest atmospheric systematic via
the S1-style gate run on Bruns' own 29 night fields (`bruns2017_nights/`) instead of
quoting Leon's ±0.33; (3) the 0.09 s stack-path bias: per-frame medians disagree with
the stacked centroid by up to ~0.4″ on the weakest star while the preprocessing
contributes ≤ 0.014″ — the E2 stack's alignment (11 frames, few usable stars, ±1″
measured common jitter) is the prime suspect; shallow tiers should carry per-frame
values wherever a star matters (F23-adjacent pipeline note).

### The all-45 stack (2026-08-30, `b17_all45.py` — Douglas' request: the original path)

Every preprocessed frame in one unweighted stack (the Leon all87 doctrine: a depth
probe and consistency check, NOT a union member — same photons, would double-count).
Frame order EA → EB → E2 (the stacker aligns to the first frame, F23; EA is the
best-detected tier). 78 centroids (the original-era analysis counted 255, but with the
hull blob, no blur-subtraction and that era's thresholds — not comparable); 41 matched
to mag 13, rms 0.584″; 26 stars at G ≤ 11.

| path | R cut | N | L v-deg2 ± stat (″) |
|---|---|---|---|
| tier union (default) | 2.0 | 25 | +1.556 ± 0.135 |
| **all45** | 2.0 | 24 | **+1.569 ± 0.163** |
| inner union, per-frame-verified | 1.45 | 27 | +1.519 ± 0.153 |
| **all45** | 1.45 | 26 | **+1.473 ± 0.233** |

**Every reduction path lands in [1.47, 1.57] — the cell-1 result is path-stable.**

The all-45 stack also settles the G 7.52 dispute with a third, independent path: it
reads **+0.450″** radial where the per-frame treatment read +0.463 — the two agree to
0.013″, isolating the E2-only stack's +0.04 as that stack's alignment artifact (the
all-45 stack aligns on EA's plentiful stars; the 11-frame 0.09 s stack had almost
none). G 7.09 reads +1.096 here (dead on GR's 1.078) vs +1.63 per-frame / +1.44
E2-stack — single inner stars are path-dependent at the ±0.5″ level on this structured
background, which is their honest weight; the L fit barely feels it (2 stars of 26–27).

### Procedure identity with Bruns 2018 (verified in code, 2026-08-30)

Douglas asked whether the chain reproduces Bruns' published calibration-transfer
procedure. Verified against `distortion_polynomial.py` (the freeze semantics at
`_cubic_helper`, the multi-reference averaging at `_open_distortion_files`, the
constant-only branch at line 298) and the stored reference lists:

| Bruns' published step | this chain | verified |
|---|---|---|
| cubic distortion measured on August night fields (his Table 2), then frozen | the L/R fits pass **15 night fields** (EC06–10, LC06–10, RC06–10) as `--fix-distortion` refs with `fixed_coefficients=quadratic`: the cubic terms are the coefficient-by-coefficient **average of the 15 night fits**, held fixed | ✓ |
| "leaving only linear and quadratic plate scale terms to fit these calibration images" | `quadratic` freeze frees exactly the basis columns of order ≤ 2 (constant + linear + quadratic); code partitions the basis at n_free and fits OLS on the free block only | ✓ |
| "plate constants were averaged over the RIGHT and LEFT calibration fields because the ECLIPSE field was midway" | the eclipse fits pass **both** L and R8 as references; `_open_distortion_files` averages every coefficient (`coeff += v/n`) and the plate scale (mean → 2.0868004, matching the stored EA value to 10 digits) | ✓ |
| "these plate constants were then used in polynomials used in the ECLIPSE field images" | `fixed_coefficients=constant`: the fit solves pointing, then **discards the fitted stretch/skew and overwrites the plate scale with the reference mean** (line 301) and applies the averaged polynomial verbatim | ✓ |

Disclosed differences, none of which change the transfer:
1. **Roll is refit per eclipse stack** (the constant branch keeps RA/DEC/ROLL from
   its linear helper). On a clamped mount over 90 s this is µrad-class, and the
   estimator's Θ column makes rotation L-neutral regardless.
2. **Gauge**: his Table 2 coefficients are TAN-projection (Astrometrica-convention)
   rad/px³; MEE's are in MEE's angular gauge — the two differ by the universal radial
   term (k_TAN ≈ k_MEE + ~0.4 ″/deg³, ROADMAP §gauge), so coefficient VALUES cannot be
   compared table-to-table without conversion. The procedure itself is gauge-invariant.
3. Bruns registered and measured **per image**; we stack per tier and drop to
   per-frame where a star demands it (the E2 lesson above).
4. His cubic average came from his own August night set; ours from the 15 re-reduced
   night fields of the comparison week — same instrument, same month, our reduction.

### The 2024 reanalysis compared, and the plate-scale levers measured (2026-08-30)

Douglas located the 2024-era outputs (`I:\2017 eclipse data analysis\analysis\
distortion_results{L,R,E}.txt`, produced 2024-03 by the v0.4.0-era pipeline (the UI screenshot Douglas kept shows MEE2024 v0.4.0)). They prove
the 2024 reanalysis already followed Bruns' procedure — same three-level chain: 15
Aug-19 night calibrations quadratic-frozen into L/R, eclipse field constant-frozen on
the L+R mean (its stored 2.0867322 IS the exact mean of its L and R), all 45 frames in
one stack (58 stars at tol 5.0″). So 2024, 2026, and Bruns 2018 are one procedure.

The numbers, era to era:

| quantity | 2024 (v0.4.0) | 2026 (v1.4.0-dev) | Δ |
|---|---|---|---|
| L scale (″/px) | 2.0866996 (75 stars, tol 0.2, 8 fr) | 2.0867534 (105 stars, tol 0.5, 7 fr) | **+25.8 ppm** |
| R scale (″/px) | 2.0867649 (77 stars, tol 0.2, 6 fr) | 2.0868386 (R6, like-for-like) | **+35.3 ppm** |
| imported eclipse scale | 2.0867322 | 2.0868004 | **+32.7 ppm** |
| L−R split | −31.3 ppm | −45.1 ppm (R8) / −40.8 (R6) | |

At h = 10.6 R⊙², 32.7 ppm of scale is **0.33″ of L** (eq-23): with the 2024 scale,
cell 1's headline would read ≈ 1.88 instead of 1.556 — GR (1.751) sits between the
eras. The Newton exclusion is era-proof (both values ≫ 0.876, and the 2024 scale
strengthens it); the GR-agreement precision hangs on this 33 ppm.

**The levers, measured by direct A/B on the 2026 L/R fits** (`matrix_bruns2017/
refraction_lever/`, `tol02_lever/`):

| lever | L | R8 | character |
|---|---|---|---|
| refraction correction ON → OFF | **+266.8 ppm** | **+265.9 ppm** | huge, common-mode |
| fit tolerance 0.5 → 0.2″ | +7.9 ppm | +7.9 ppm | small, common-mode, wrong sign for the gap |

> **SUPERSEDED 2026-08-30 (same day), on Douglas' question.** The paragraph that stood
> here attributed the era gap to "the broken pre-v1.4.0 refraction correction". That
> was WRONG on the history: the famous breakage (`0468e22`, fixed 2026-08-20 —
> `AttributeError: 'StarTable' object has no attribute 'c'`) was born with the
> **StarTable catalogue layer**, i.e. the v1.3.x era, and it *crashed loudly* rather
> than corrupting numbers; it was invisible only because corrections were off by
> default. The 2024 reanalysis (**v0.4.0**, March 2024 — the era note "v1.2" above is
> also corrected) predates StarTable: its catalogue object was StarData, which had
> `.c`, and **its refraction correction ran**. The 2024 results files record
> corrections enabled and computed alt/az, consistent with that.

Attribution, from the three levers measured (all A/B on the 2026 fits, everything
else pinned):

| candidate | measured effect on fitted scale | verdict |
|---|---|---|
| refraction correction ON→OFF | +266.8 / +265.9 ppm (L / R8) | the *pathway* is huge, but both eras ran it |
| fit tolerance 0.5→0.2″ | +7.9 / +7.9 ppm | wrong sign — deepens the unexplained gap to ~41 ppm |
| night references 2026→2024 (`refswap_2024/`) | **+1.0 / +1.1 ppm** | the frozen-cubic difference is NOT the owner |

**The ~33–41 ppm era gap is therefore currently UNATTRIBUTED.** The surviving
candidates act through the star positions or the correction implementation itself:
the v0.4.0 → v1.4.0-dev evolution of the correction code (`erfa.ld` error-path
restore, the 9-bug pass, astropy version) and of the centroid machinery (weighted
centroids Oct-2024, gaussian-subtract defaults). A ~12–15 % implementation difference
in a 266 ppm correction, or a ~0.1 px systematic centroid trend across the field,
would each suffice. The decisive next measurement is a **star-by-star comparison of
the 2024 stored matched tables against ours** — the 2024 `DISTORTION_OUTPUT*` folders
survive on `I:\2017 eclipse data analysis\`.

The v1.4.0-dev chain remains the reduction of record (current tested code, refraction
model validated on the Leon M-campaign, 105/110 vs 75/77 stars) — but the era gap is
now an **open measured discrepancy worth 0.33″ of L**, not an explained one, and cell
1's GR-agreement precision inherits it until the star-by-star audit closes it.

### The star-by-star audit: the gap lives in the centroids (2026-08-31)

Douglas pointed at the surviving 2024 data (`I:\2017 eclipse data analysis\`) and the
two decisive experiments ran.

**(A) 2024 centroids through 2026 code** (`xera_2024centroids/`): today's stage 2 with
today's 15-night references, run on the 2024 `dataL.zip`/`dataR.zip` centroid
archives, reproduces the **2024** scales — R to 3 ppm (2.0867674 vs the 2024 fit's
2.0867649 at like tolerance), L to −8/+6 ppm — i.e. ~35 ppm below the 2026-stack
values. **The code, the correction implementation, and the references are exonerated;
the era gap lives in the stage-1 stacks/centroids themselves.** (R6 is frame-identical
between eras, so frame selection is excluded too.)

**(B) the star-by-star affine** between the two eras' stacked centroid lists (same
fields, 104–108 matched pairs, residual scatter after affine only 0.08 px on R6):

| field | isotropic scale 2024→2026 | bright half only | faint half only |
|---|---|---|---|
| R6 | **−31.6 ppm** | −18.7 ppm | −48.9 ppm |
| L | **−37.0 ppm** | −19.1 ppm | −53.2 ppm |

The −32/−37 ppm matches the fitted-scale gap exactly — and it is **brightness-
dependent, factor ~2.6 between bright and faint halves**: the signature of the two
eras' centroiders weighting the asymmetric off-axis PSF wings differently (v0.4.0:
plain moments over the threshold area; v1.4.0-dev: Gaussian-subtracted, windowed).
The within-era arbiter (bright-vs-faint radial split against the catalogue in each
era's own fit) is **inconclusive** — the splits are noise-level (−17.5 to +17.8 ppm at
±15–30 ppm sensitivity) and change sign between fields.

**The transfer-cancellation insight, which reframes the stakes.** A *uniform* centroid
scale convention cancels exactly in the calibration→science transfer: if every pixel
position is magnified by (1+e), the calibration fit returns S/(1+e) and the science
positions carry (1+e), so the sky angles — and L — are unchanged. Each era is
internally consistent; **the absolute 33 ppm never reaches L**. What leaks is only the
**brightness-dependent part**: the calibration scale is fitted on a mag 7–13 mix
(faint-weighted) while the science stars are the bright end, so the convention
mismatch entering L is ≈ (49−19)/2 ≈ **15 ppm ≈ 0.15″ of L** — a centroid systematic
present in BOTH eras (with era-dependent sign), half the naively-feared 0.33″. The
earlier sentence "0.33″ of L hangs on this" is hereby corrected to that mechanism.

Closure path (logged, not yet run): the truth-referenced arbiter is the synthetic-PSF
centroid benchmark (`docs/bench/psf/`, which validated the v1.4.0-dev centroider), plus
a targeted A/B — re-run the 2026 stage 1 with the windowing/subtraction disabled to
emulate the v0.4.0 centroider on identical frames and measure the brightness slope
against the catalogue with full-field statistics.

### The L triangulation: 1.74 vs 1.56 is the centroid convention, not the program (2026-08-31)

Douglas: Bruns published L = 1.752″ ± 3.4 %; the 2024 rerun gave ~1.74; our chain
gives 1.47–1.57 — are we mixing old and new files inconsistently? The audit says the
chain is era-pure (every stage-1/2 product in the science path is a 2026 reduction;
2024 files entered only the labelled comparison experiments). The four-way
triangulation then splits data from method exactly:

| star table | reduction chain | L v-deg2, R > 2, G ≤ 11 (″) |
|---|---|---|
| 2024 eclipse table (58 stars) | 2024's own (as stored, ps 2.0867322) | **+1.741 ± 0.089** |
| 2024 eclipse centroids | 2026 chain (our cal, ps 2.0868004) | +1.981 ± 0.094 |
| 2026 stacks | 2026 chain | **+1.556 ± 0.135** |

Row 1: **our estimator on their table reproduces the 2024/Bruns value exactly — the
method is exonerated.** Row 2 is the deliberate cross-mix, and it shows what
inconsistency *would* look like: +0.24″ of artifact, the imported-scale mismatch
(+32.7 ppm × h·R⊙) applied to centroids of the other convention. We are NOT doing
this anywhere. Rows 1 and 3 are each internally consistent and differ by **0.18″ —
the star-dependent (brightness-dependent) part of the centroid-convention difference,
which the calibration transfer cannot cancel.**

Mechanism, sharpened: the v0.4.0 centroider (plain moments over the threshold area)
has a *structural* brightness–radius coupling on asymmetric off-axis PSFs — a bright
star's threshold area extends into the coma wings, a faint star's covers only the
core — and Bruns' own tools were moment-based too. The 2026 windowed centroider
applies the same weighting to every star and measured **mag-stable to ±3 ppm** in the
bright-cal test (`brightcal/`: bright-only calibration moves the bracket +2.4 ppm →
−0.024″ of L, refuting the cal-side leak). Their inner stars also scatter wildly
between conventions at the star level (G 7.09: +0.87″ in their table vs +1.63/+1.44/
+1.10 across our paths; G 7.52: +1.31″ vs our +0.46) — ±0.4″ single-star path
dependence, as found before.

**Agreement with GR must not arbitrate** — picking the convention that lands on
1.75 would be circular in a deflection experiment. Until the synthetic-PSF benchmark
(run with THIS instrument's measured PSF shapes, `psf_bruns2017/`) rules, cell 1
carries a **centroid-convention systematic spanning the two internally-consistent
reductions: L = 1.56 (2026 windowed) to 1.74 (2024 moments), i.e. ± ~0.09″ about
their midpoint** — quoted alongside the stat and scale terms, not hidden inside them.

### The design ruling and the rollback (2026-08-31)

**Douglas' ruling**: MEE2024's founding design constraint is that it must reproduce
Bruns 2018 on the Bruns 2017 data. The windowed-convention result (1.556 ± 0.135)
falls outside Einstein at ~1 σ and outside the constraint; the project rolls back to
the moment-based convention that Bruns' own tools shared. **The convention question is
settled by design authority, not by the (still-logged) PSF-truth benchmark; if that
benchmark someday rules against the moment convention, the finding will have to be
reconciled with Bruns' own analysis at that level.**

**The options-level rollback FAILED its checkpoints** (`b17_moment.py`,
`matrix_bruns2017_moment/`): with `sensitive_mode_stack=False`,
`centroid_gaussian_subtract=False`, `centroid_refine_window=False`, the star-by-star
affine against the 2024 centroid lists does NOT collapse (L −23.0 ppm; R8 −29.7 ppm,
now brightness-*uniform* −31.8/−31.2), the bracket mean stays at 2.0868062 (vs 2024's
2.0867322), and moment-mode science stacks drown in coronal junk (1173–1527
detections, every plate solve fails). Two conclusions: (a) **v1.4.0-dev minus its
centroid flags is not v0.4.0** — a large share of the convention gap lives in the
stack/detection layer, beyond what options reach; (b) the windowed flags are
*required* for eclipse-field detection in the current code.

**Implementation of the ruling, today**: cell 1's L-of-record is the Bruns-convention
reduction that already exists and is fully traceable — the 2024 star table (v0.4.0
centroids, the convention Bruns' tools shared) through the current estimator:

> **Cell 1 (design-constraint convention): L = 1.741 ± 0.089 (stat) ± 0.103 (scale)
> ″** — vs Bruns 2018's 1.752 ± 0.060. Constraint satisfied. The 2026-windowed
> reduction (1.556 ± 0.135) stands in the record as the measured convention
> alternative, not the number of record.

Follow-ups logged, in order of value: (1) isolate WHERE in the stack/detection layer
the remaining −25/−30 ppm lives (candidates: alignment interpolation, background
handling in detection, hot-pixel filtering) — a true `MEE2024Stacker_v0.4.4`-tag
reduction of one field would give the reference stack; (2) the synthetic-PSF
benchmark with `psf_bruns2017` PSF shapes; (3) **matrix-wide implication**: every
other cell (Mexico, Leon, Portland) currently uses the windowed convention; the
convention systematic measured here (~0.2″ of L on Bruns' optics) must be either
bounded or corrected per-instrument before cross-cell comparison — for Leon this is
now an open item against the 1.98 ± 0.60 ± 0.33 headline (likely smaller — different
optics, cleaner PSFs — but "likely" is not a measurement).

Design note the A/Bs demonstrate for free: both levers move L and R almost
identically (266.8 vs 265.9; 7.9 vs 7.9), so **the L−R split is immune to them — the
bracket cancels common-mode systematics exactly as designed**, and the split's
era-change (31 → 45 ppm) is a star-set/version effect, not atmosphere.

For reference, Bruns' own Table 4 linear terms convert to ≈ 2.08677 (X, AST) and
≈ 2.08661 (Y, AST) ″/px — same 40–90 ppm class as everything above, but his
convention embeds his own refraction handling and an axis asymmetry, so a
table-to-table comparison beyond the class level is not meaningful.

### The rollback succeeds, and the lever was the BACKGROUND, not the estimator (2026-08-31)

Douglas asked whether moment centroiding can coexist with the background subtract. The
code's answer is yes — and acting on it overturned the previous section's attribution.

`centroid_gaussian_subtract=True` with `centroid_refine_window=False` gives flux-weighted
moments over each detected footprint of the background-subtracted image: **moment
centroiding inside the sensitive detector**, which is also the config default. Rollback
attempt 1 failed because it turned the sensitive flag *off*, dropping to
`simple_get_centroids` — a different detector with a global threshold, which is why it
drowned in corona. That was a mis-designed experiment, not a property of the convention.

Three A/Bs on the calibration fields then decomposed the ~+33 ppm era gap:

| configuration | bracket mean (″/px) | vs the 2026 standard |
|---|---|---|
| windowed + annular (2026 standard) | 2.0868004 | — |
| footprint moments + annular | 2.0867970 | **−1.6 ppm** ← the *estimator* |
| windowed + Gaussian (R6, frame-identical to 2024) | — | **−19.1 ppm** ← the *background* |
| footprint moments + Gaussian | 2.0867533 | **−22.6 ppm** (both) |
| the 2024 chain | 2.0867322 | (+10.1 ppm still unattributed) |

**The centroid estimator is worth under 2 ppm. The background-subtraction mode is worth
~19.** Both act through the same physics — how much of an asymmetric off-axis PSF's wings
each measurement sees — but the 17 px annular ring reaches further into the coma than the
Gaussian kernel, and that is where the radial scale difference lives. `annular` became
the default only in July 2026 (`4bacfc4`); the 2024 archives record no detection settings
at all, which is the provenance gap now closed (F25).

**The end-to-end reduction in the 2024 convention** (`b17_like2024.py` — Gaussian
background + footprint moments applied to the calibration *and* the science field, since
changing only the calibration is the meaningless cross-mix):

| variant | N | L base (″) | L v-deg2 (″) ± stat |
|---|---|---|---|
| **0.62 union, R > 2.0** | 25 | **+1.755 ± 0.065** | **+1.720 ± 0.069** |
| FULL union, R > 2.0 | 25 | +1.721 | +1.675 ± 0.080 |
| FULL union, R > 1.45 | 27 | +1.775 | +1.667 ± 0.080 |
| *(per tier)* | | | EA +1.666, E2 +1.850, EB +1.773 |

> **Cell 1, reduction of record (revised): L = 1.720 ± 0.069 (stat) ± 0.105 (scale) ″**,
> total σ ≈ 0.126. Bruns 2018: 1.752 ± 0.060. **GR at 0.25 σ; Newton excluded at 6.7 σ.**
> The founding design criterion is met by a genuine end-to-end v1.4.0-dev reduction — the
> 2024-star-table workaround of the previous section is superseded and withdrawn.

Everything improves together, which is the sign that this is the right convention for
this instrument rather than a number chosen for agreement: calibration rms falls (L
0.2087″ vs 0.2170 windowed+annular, on more stars), the statistical error halves
(±0.069 vs ±0.135), per-tier spread narrows (0.11″ vs 0.11″ but around a higher, more
consistent mean), and E2 matches 16 stars instead of 12.

**What is still open.** +10.1 ppm of the era gap remains unattributed (candidates: 2024's
unrecorded thresholds, hot-pixel handling, alignment/reference choice — F23). And the
convention is still chosen by *design authority plus residual quality*, not by
truth: the synthetic-PSF benchmark with `psf_bruns2017` shapes remains the arbiter that
could rule independently, and it has not been run. Bruns' own §2.3 comparison — moment
versus Gaussian-fit centroids differing 0.039 px on an SNR-13 star and 0.003 px on an
SNR-48 star — is corroborating evidence that this class of difference is real and
brightness-dependent, measured in his own data.

**Matrix-wide**: cells 2–4 must use the convention cell 1 is quoted in, and Leon's
headline (1.98 ± 0.60 ± 0.33) was reduced windowed+annular — it needs re-measuring in the
Bruns-compatible convention before the cells can be compared.

### Four questions answered, and one honest failure (2026-08-31)

**Why the Leon star count fell 42 → 39 under Gaussian+moments.** Not the estimator, and
not the matching — fewer stars were *detected*. The two background models differ in how
much of a star's own light they put into its background: `annular` excludes the inner 3 px
before averaging, `Gaussian` does not, so a star is partly subtracted from itself. Peak
retained, measured on a synthetic star:

| FWHM (px) | Gaussian 17 px | annular 17/3 |
|---|---|---|
| 1.5 | 0.954 | 1.000 |
| 2.5 | 0.881 | 0.993 |
| 4.0 | 0.743 | 0.961 |

At Leon's seeing that is 10–20 % of peak SNR, and detections fell accordingly: 0.1 s
673 → 507, 0.3 s 474 → 323, 0.6 s 257 → 142, 1.2 s 101 → 79. Gaussian buys the cleaner
radial behaviour that reproduces Bruns **at a real cost in depth** — worth saying whenever
the preset is chosen.

**The magnitude limit: Douglas' 2017-era finding, reproduced.** He reported that beyond
mag 11 the error began to rise. Scanning the cut on the cell-1 union (`b17_magscan.py`,
catalogue opened to G 13):

| mag cut | N | L v-deg2 (″) | ± stat (″) | rms of the newly-admitted stars (″) |
|---|---|---|---|---|
| 10.0 | 15 | +1.765 | 0.064 | 0.084 |
| 10.5 | 20 | +1.708 | 0.057 | 0.091 |
| **11.0** | **25** | **+1.719** | **0.077** | 0.199 |
| 11.5 | 29 | +1.695 | 0.080 | 0.332 |
| 12.0 | 31 | +1.630 | **0.211** | 0.572 |

Per-bin residual scatter about the fitted model is flat to G 11 — 0.128 ″ (G 6–9), 0.156
(9–10), 0.160 (10–11) — then jumps to **0.418 ″ at G 11–12**, a factor 2.6. The
statistical error stops improving around G 10.5–11 and triples by G 12, while L itself
drifts 0.09 ″ downward. **Mag 11 is the last cut that does not hurt** — exactly where
`eclipse_limiting_mag` and every union in this matrix already sit. Inherited practice,
now measured on this data.

**Why the full union sits higher than the 0.6+1.2 union — Leon only.** Not a general
property: on Bruns the full union sits *lower* (1.675 vs 1.720). On Leon it was traced to
individual stars rather than to the shallow tiers as a class — the G 9.10 corrupted
centroid alone owned −0.36 of the −0.37 ″ correction once a deeper catalogue let the
auto-vet catch it, leaving a 0.26 ″ residual gap, inside the quoted atmospheric term. The
shallow tiers add stars whose cross-tier medians one or two bad centroids can pull; the
vet is what decides, and it works.

**The atmospheric term for cell 1: attempted twice, invalid twice — so cell 1's error bar
is still incomplete.** Leon's ±0.33 ″ is an empirical null: the estimator run on real
fields with zero true deflection, reduced the same way as the science field, with the
Sun's frame position imposed. Cell 1 was quoted with no such term
(1.720 ± 0.069 stat ± 0.105 scale), which understates it. Two attempts on Bruns' 29 night
fields both failed, and both are recorded because each looked plausible:

1. **Their existing residuals: ±0.018 ″.** Withdrawn — those come from a *free cubic* fit
   at tolerance 0.2 (rms 0.059 ″), and a free cubic absorbs the smooth atmospheric
   structure the test exists to measure. Leon's M5 fields are fitted constant-only against
   a frozen reference (rms 1.60 ″), which is what the eclipse field itself does.
2. **Refitting constant-only: ±1.04 ″, then ±1.40 ″ after pairing consecutive fields.**
   Both withdrawn. The first froze each group's field 01 for all nine others — but fields
   06–10 are from the *following night*, so it measured the documented +85 ppm
   night-to-night plate-scale gap (~0.85 ″ of L at h = 10.6). The second paired same-night
   neighbours 6–7 minutes apart and returned a *larger*, uniformly **positive** result
   (mean +1.24, all 22 pairs positive, clustered by pointing: RC ≈ +2.0, EC ≈ +1.1,
   LC ≈ +0.6). Atmosphere scatters about zero; a one-signed pointing-dependent offset does
   not. Remaining suspect: differential refraction across the 6–7 minute gap — a vertical
   compression change the design matrix has no term for, and which the science chain never
   sees (CAL_piLeo and the eclipse field are two minutes apart, and Bruns' L/R bracket
   cancels it by construction).

**What cell 1 should carry meanwhile.** The dataset already holds a direct, eclipse-day,
right-altitude measurement of its own atmospheric differential: the **L−R bracket split of
45.0 ppm**, half-width 22.5 ppm. Quoting the scale term at that bound rather than at the
10.3 ppm statistical HC3 gives **±0.23 ″ instead of ±0.105 ″** — the honest way to carry
the atmosphere until a valid null exists. On that basis:

> **Cell 1: L = 1.720 ± 0.069 (stat) ± 0.23 (scale incl. atmosphere) ″**, total σ ≈ 0.24.
> GR at 0.13 σ; **Newton excluded at 3.5 σ**. The Eddington discrimination survives; the
> earlier **6.7 σ figure is withdrawn** — it omitted the atmosphere.

A second, independent handle agrees: EA and EB are the same configuration 51 s apart and
differ by 0.107 ″ in L, a direct measure of what a couple of minutes of this atmosphere
does to the answer.

A valid null needs either a scale/compression term in the null estimator, or night pairs
~1–2 minutes apart rather than 6–7. Logged, not done.

> **Resolved the same day — the cause was mine.** The constant-only refits were run with
> `observation_time=08:00`, a placeholder I passed to every field, so the refraction
> correction was applied at the wrong altitude — differently wrong for each pointing,
> which is precisely the uniformly-positive, group-clustered signature. With each field's
> own recorded time the nulls become sane, and the nuisance now *helps* rather than hurts
> (rms 0.177 → 0.150), which is the sanity signature that was missing. See below.

### The atmosphere at the eclipse geometry, and why Leon was limited (2026-08-31)

Douglas' observation, which turns out to be the key to the whole comparison: **Bruns
rehearsed the identical three pointings on both preceding nights.** Measured from the fits:

| night field | alt / az | eclipse-day counterpart | alt / az |
|---|---|---|---|
| EC | 54.56° / 143.4° | the ECLIPSE field | 54.35° / 142.71° |
| LC | 53.56° / 131.0° | LEFT calibration | 53.47° / 130.63° |
| RC | 54.30° / 156.2° | RIGHT calibration | 54.20° / 155.58° |

0.1–0.2° in altitude, under a degree in azimuth. Same airmass, same optics, same site — a
night-time replica of the eclipse-day geometry. **Leon had nothing comparable**: its
rehearsal fields were at the zenith while the eclipse sat at 9.7°, so its atmospheric term
had to be transported across a factor of six in airmass.

**M3-style maps, built the same way Leon's were** (`b17_m3_maps.py`: cubic frozen from the
same 15-field average the L/R calibration used, quadratic free — i.e. exactly how a
calibration field is reduced; vectors rotated into alt-az through each field's own affine).
Figure: `matrix_bruns2017_m3/b17_m3_quiver_maps.png`; table: `b17_m3_stats.csv`.

| | alt | quasi-static rms (″) | alt component | az component | V/H |
|---|---|---|---|---|---|
| **Bruns EC** | 54.5° | 0.103 | 0.074 | 0.071 | 1.0 |
| **Bruns LC** | 53.5° | 0.100 | 0.072 | 0.068 | 1.1 |
| **Bruns RC** | 54.3° | 0.103 | 0.075 | 0.070 | 1.1 |
| **Leon (M3)** | 8.5–12.4° | 0.167–0.349 (mean 0.261) | 0.153–0.323 | 0.066–0.134 | ~2.3 |

**This is the answer to why Leon was limited, and it is sharper than "worse seeing".** The
*horizontal* components are nearly the same — 0.070 ″ at Bruns' altitude against
0.066–0.134 at Leon's. What explodes toward the horizon is the **vertical** component:
0.074 → 0.153–0.323 ″, a factor 2–4, turning an isotropic residual (V/H ≈ 1.0) into a
strongly polarised one (V/H ≈ 2.3). The horizontal number is the instrument-and-model
floor common to both; the vertical excess is refraction-driven atmospheric structure — and
it is the component that couples to a radial deflection signal.

**The atmospheric term for cell 1, now measured** (`b17_atmosphere2.py`, corrected):
22 constant-only nulls from consecutive same-night pairs, true L = 0 in every one —
residual rms 0.038–0.157 ″/axis, **L v-deg2 rms ±0.150 ″** (max 0.283) against a bootstrap
floor of 0.037, so it is real structure rather than noise.

Two independent routes agree on the Leon/Bruns ratio to 20 %:

* M3 quasi-static residual: 0.261 / 0.102 = **2.6×**
* null-test L systematic: 0.33 / 0.150 = **2.2×**

> **Cell 1, error budget completed: L = 1.720 ± 0.069 (stat) ± 0.105 (scale)
> ± 0.15 (atmosphere) ″ — total σ ≈ 0.20.** GR at 0.16 σ; **Newton excluded at 4.3 σ**.
> Taking instead the conservative L−R bracket bound (22.5 ppm) in place of the separate
> scale and atmosphere terms gives ±0.24 and 3.5 σ. Either way the Eddington
> discrimination holds and the earlier 6.7 σ stays withdrawn.

**One caveat on the 0.15, stated because it matters for 2027**: these are *night* fields.
Daytime convective turbulence is generally worse than night at the same altitude, so 0.15 ″
is a **best case** for the eclipse-day atmosphere at this geometry — which is exactly what
Douglas asked for, and it is the number a 2027 site should be judged against.


### Bruns 2018 against this reduction, number for number (2026-09-01)

Douglas asked for the full table. Every row is his published figure against ours from the
same variant of the same data (`tools/matrix_bruns/b17_bruns_comparison.py`):

| quantity | Bruns 2018 | this reduction | agreement |
|---|---|---|---|
| Method 1, close-in pair IN | **1.752** (± 3.4 %) | **1.777 ± 0.064 (stat)** | 0.025″ = 0.4 σ |
| Method 1, close-in pair OUT | 1.731 (± 4.1 %) | 1.705 ± 0.071 | 0.026″; both DROP without the pair |
| Method 2 (scale free), pair IN | 1.86 (± ~4 %) | 1.842 ± 0.105 | 0.018″ |
| Method 2, pair OUT | 1.711 (± ~8 %) | 1.768 ± 0.164 | 0.057″, well inside errors |
| star-fit error in L | 0.088″ = 3.1 % (his eq-20) | 0.064″ boot / 0.067″ analytic = 3.7–3.8 % | same class |
| plate-scale error in L | 3.34 ppm → 1.23 % (his moment formula; verified to 3.11–3.14 ppm on our geometry, and it omits the √2 of averaging L and R — with it, 2.36 ppm) | 10.3 ppm (HC3, √2 included) → 4.8 %; **at his 3.34 ppm we get 1.6 %** | per field the ratio is 4.24 = ×2.81 input (fit residual vs centroid precision) × ×1.51 estimator |
| effect of dropping the pair on the error | +21 % (3.4 → 4.1 %) | +7 % (0.184 → 0.197 total) | same direction |
| roll parameter | not fitted ("only … L and simple RA and Dec offsets") | fitted; removing it moves L by **−0.002″** | immaterial either way |
| stars used | 18 + 2 close-in, hand-vetted ("several stars were eliminated because of very poor fits or nearby stars") | 25 + 2, rule-based (G ≤ 11, doubles at 10″ dropped, blends dropped, R > 1.45 R⊙) | rule vs hand; see the outlier note |

Notes that came out of building the table:

* **The close-in pair pulls L up in both analyses** (ours +0.072″, his +0.021), and both
  Method-2 values land within hundredths of each other's. Four variants, two analyses
  nine years apart, one consistent picture.
* **The rotation term answers Douglas' question about roll**: Bruns did not fit one, we
  do, and the difference is −0.002″ — the constant-only stage 2 has already fixed the
  orientation, so the estimator's Θ has almost nothing to absorb.
* **The 7-star link scanned from N = 3 to 14** (14 common stars exist): the link se is
  essentially flat, 0.13″ at N = 3, 0.107″ at Bruns' 7, best 0.098″ at 14, with a bump
  at N = 9 where a discrepant faint star enters. Moving 7 → 14 shifts the pair by
  −0.05 px in x — and because the two stars sit on opposite sides of the Sun the shift
  nearly cancels in L. **The link choice is L-neutral at the 0.01″ level; 7 stays, as
  Bruns had it.**
* **The r ≈ 4.7 outlier is G 10.64 at px (3012, 108)**: radial +0.147″ vs GR's 0.374,
  tangential −0.171″. Forensics: NOT a double (no catalogue neighbour to G 13 within
  30″), not near coronal structure — it is simply faint (area 3 px, the smallest
  footprint in the table) and in the field corner where the PSF is worst. Its tangential
  error is as large as its radial one, which is the signature of noise, not of deflection
  physics. Bruns would likely have hand-eliminated it ("very poor fits"); our rule keeps
  it, which costs symmetric noise rather than bias.
* **The tangential column is why the old field chart looked wrong**: tangential rms is
  0.114″ against a radial-about-fit rms of 0.085″. The earlier arrows pointed exactly
  outward because the chart projected onto the radial direction before drawing — an
  artifact of the plotting, now fixed (full vectors drawn, lengths asserted at runtime).
* **The blur subtraction is exactly Bruns' method and still in use**: his "averaged
  without translations, then a 10-pixel wide Gaussian blur … subtracted from all of the
  individual ECLIPSE images" is our preprocessing verbatim (plus a +2000 ADU pedestal for
  unsigned output). One small difference, recorded: he blurred the 34-frame and 11-frame
  series means; our frames were preprocessed per tier (17+17+11) before the master was
  stacked. Same exposure 51 s apart — the models differ negligibly.

**Verified against the source (2026-09-01)**: Douglas pointed at the F&L 1944 paper
itself (`I:\Papers\The problem of an accurate determination of the relativistic light
deflection.pdf`), and the three equation usages in the table check out: eq (23) is the
imported-scale propagation ("its standard deviation multiplied by h does not give rise to
an uncertainty in L which surpasses the preassigned limit" — our δL = h·R⊙·δS); eq (20)
is the weight of L with the scale imported, dominated by the n·h⁻¹ term (Bruns' 3.1 %
star error and our fit covariance are the same construction); eq (12) is the weight with
the scale fitted simultaneously — Method 2 — carrying the smaller ½(κ⁻¹ − h⁻¹) term,
which is why both his and our Method-2 errors are larger. F&L's own remark under (23),
that the weight "is very sensitive to any asymmetry in the star field", is the
mathematical form of what the Leon anchor demonstrated empirically.


### Chart review round 2, and three variants (2026-09-01, later)

* **The missing vectors are explained and fixed**: arrows whose endpoints left the
  sensor area were silently clipped at the axes limits (G 7.09's ends at py ≈ 2714 on a
  2472 px axis; every "missing" outer star sat near an edge with an outward arrow). The
  axes now extend past the sensor, whose edge is drawn, and a runtime assertion fails
  the script if any endpoint leaves the axes. Both scale bars (1″ and the measured
  per-star scatter, 0.14″) sit with the legend.
* **The error-consistency point**: the covariance chart's ±0.107″ was stat+scale while
  the deflection band's ±0.18″ was the total including atmosphere. Both charts now say
  which they carry, and the covariance chart quotes the total beside its ellipses.
  One-sigma ellipses only; the y-axis is now ppm offset from the imported scale, which
  also removes the orphaned axis-offset text.
* **The r ≈ 4.7 outlier costs +0.014″ if removed** — negligible at its leverage, as
  Douglas guessed; annotated on the chart.
* **G ≤ 13 variant** (`record_deflection_g13.png`): N = 39, L = 1.728 ± 0.088 — the
  statistical error grows 37 % and L drifts down 0.05″. The mag-11 finding, now visible
  as a picture; one star beyond 3σ (G 11.60, −3.0σ), labelled.
* **The 14-star link**: L = 1.764 ± 0.060 against the 7-star 1.777 ± 0.064 — a
  −0.013″ shift, inside the link error. Bruns' 7 stays, as ruled.
* The 0.09 s master now lives beside the 0.62 s one as
  `matrix_bruns2017_brunsmethod/master009/` (a copy of the E2 stack in the convention of
  record), so the reduction's tree is self-contained.


### Chart review round 5, and a frame error in the atmosphere maps (2026-09-01)

**The maps were mixing two frames, and Douglas caught it from the instrument geometry.**
Bruns' ROLL is ~0.25–0.38°, so his sensor is aligned to RA/Dec — north up — and the
direction of increasing altitude in sensor axes is therefore the parallactic angle, which
differs per pointing. Measured: **EC +153.7°, LC +145.3°, RC +162.8°** from sensor +y. The
first version drew the same green "up" arrow on every panel, and worse, plotted the
*alt/az decomposition* as the arrow components while leaving the star positions in sensor
pixels — two different frames on one panel. Now matched to Leon's construction exactly:
positions and arrows both in sensor axes, with each panel's green arrow along its own
measured altitude direction. The alt/az decomposition remains what the **stats table**
reports (it is how the V/H ratio is computed) — that part was always right.

**Variants now on the charts** (all tables in `matrix_bruns2017_brunsmethod/`):

| chart | cut | link | N | L (″) | total σ |
|---|---|---|---|---|---|
| `record_deflection` | G ≤ 10.5 | 14-star | 22 | **1.794 ± 0.062** | 0.111 |
| `record_deflection_link14` | G ≤ 11 | 14-star | 27 | 1.764 ± 0.060 | 0.113 |
| `record_deflection_g13` | G ≤ 13 | 14-star | 39 | 1.718 ± 0.086 | 0.195 |
| `record_covariance`, `record_field` | G ≤ 11 | 14-star | 27 | 1.764 | 0.113 |

The monotone drift with depth — 1.809 at G ≤ 10.5, 1.764 at 11, 1.718 at 13 — is the
magnitude finding seen from a third angle: each fainter tranche pulls L down and widens
the error, which is why the cut sits where it does.

Also this round: `master009`/`master062` legends moved below the frame at px = 0; the
earlier greyscale master009 recovered from git as `chart_versions/rev05_*`; the field
chart's caption moved under the axis with both scale bars boxed; the covariance chart's
method text boxed at bottom-left with the imported plate scale stated, its legend moved
top-right.

### Appendix — every parameter in effect (cell 1)

The CLI merges `--set` overrides ON TOP of the operator's interactive
`MEE_config.txt` (the measured Leon trap), so this table records the *effective* value
and its provenance for every parameter that can matter. Source of truth: the merged
options dumps in `matrix_bruns2017/*/stage1.log` and the stored
`distortion_results.txt` files.

**S0 preprocessing (`b17_s0.py` — outside the pipeline):** coronal model = unshifted
per-tier mean, Gaussian blur σ 10 px, subtracted per frame; pedestal +2000 ADU; clip
[0, 65535] uint16; saturated pixels (≥ 65535) dilated 10 iterations and painted to
pedestal; forbidden disk painted at max(1.25 R⊙ = 568 px, tier 99th-pct saturation
radius + 20 px) → EA 921 px (2.03 R⊙), E2 666 px (1.47 R⊙), EB 922 px, centred on the
per-tier saturated-pixel centroid; no darks, no flats (held from the canonical L/R
chain, which used none).

**Stage 1 (stack + centroids), pinned by the tool:** `sensitive_mode_stack=True`,
`centroid_gaussian_subtract=True`, `centroid_gaussian_thresh=4.0` (σ, locally
adaptive), `min_area=2` px, `sigma_subtract=0.0`, `delete_saturated_blob=False` (the
hull blob is OFF — the forbidden disk replaces it), `remove_edgy_centroids=True`,
`centroid_refine_window=True`, `centroid_window_sigma=2.0`, `--no-scan`.
**Inherited from the interactive config (disclosed, not pinned):** `m=30`, `n=30`,
`d=100`, `cutoff=100`, `pxl_tol=10`, `img_edge_distance=5`; hot pixels: `sigmas=20`,
`min_adu=10`, `dark_free=True`; `background_subtraction_mode=annular`;
`sanity_check_centroids=True`; **`reject_saturated_stars=True` with
`saturation_fraction=0.95` (F16, active — verified harmless here: no unsaturated-tier
star clips; both inner stars peak ≤ 29.5 kADU in E2)**; solver `v2`,
`platesolve_noise_px=0.3`, `k=12`; `blob_radius_extra=500` and `centroid_gap_blob=150`
leaked in but are INACTIVE with the blob off. Frame order: lexicographic (so `*_10_of`
frames lead — the stack's alignment reference; F23 note applies).

**Stage 2 (constant-only fit):** `--order cubic`; `--date-from-header` (2017-08-21);
`--fix-distortion` = the canonical L **and** R8 results (frozen mean = the bracket,
imported plate scale 2.0868004 ″/px); `distortion_fixed_coefficients=constant` (only
the two pointing constants free); `distortion_fit_tol=2.0`″ (verified in the stored
results); `max_star_mag_dist=13`; `rough_match_threshhold=36`″; corrections ON
(refraction + aberration/parallax): 42°44′11″ N, 106°19′05″ W, 2400 m, 13.0 °C,
770.0 mb, humidity 0.4, λ 0.625 µm; observation_time 17:43:22 / 17:43:47 / 17:44:13 UT
per tier (from DATE-OBS, mid-series).

**Union + estimator (`b17_union.py`):** Gaia offline catalogue, **G ≤ 11.0** (the
magnitude limit in force), epoch 2017.64 with proper motions, refraction/aberration
corrected per tier time; doubles dropped at `is_double(10″)`; blends (two catalogue
claims on one detection) dropped; association gates 8.0″ (wide pass, sets the per-tier
constant offset) then 4.5″ (collect); per-star median across tiers; cross-tier vet at
3×MAD with a 1.5″ floor; science cuts R > 2.0 R⊙ (default) / 1.45 (inner variant),
mag ≤ 11.0; estimator columns [N1, N2, Θ, L·(R⊙/R)û] + deg-2 polynomial nuisance along
the computed local vertical (+153.5° from sensor +y); plain lstsq; 200-star-resample
bootstrap, seed 3.

**Per-frame anchor treatment (`b17_perframe2.py`, `b17_inner_refit.py`):** window
±10 px; background = quadratic fit to the 3-px border ring, saturated pixels excluded;
Gaussian-weighted centroid σ 2.0 px, 5 iterations; acceptance peak-above-background
> 250 ADU; per-frame reference = median offset of that frame's R > 2 R⊙ stars (≥ 5
required); star value = median over 11 frames, error = MAD/√n.


## Cell 3 — Leon 2026, brought to the cell-1 standard (2026-09-01/02)

The full account is the section of the same name in `docs/STEP3_2026.md`; this is the
matrix-level summary so the three cells can be read side by side.

> **Leon 2026: L = 1.98 ± 0.60 (stat) ± 0.70 (scale) ± 0.33 (atmosphere) ″, total σ ≈
> 0.97.** GR at 0.2 σ; Newton at 1.1 σ. Tree `step3_record/`, charts and tables copied
> into `RECORD/leon2026/`.

| | cell 1 (Bruns 2017) | cell 3 (Leon 2026) |
|---|---|---|
| L (Method 1) | 1.764 | 1.914 |
| stat | ±0.060 (27 stars, per-star scatter 0.14 ″) | ±0.64 (36 stars, two-witness, per-star scatter 0.73 ″) |
| scale | ±0.075 (9.23 ppm bracket, h = 8.58) | **±0.68** (25 ppm one-sided CAL, h = 25.9, leverage 0.027 ″/ppm measured by injection) |
| atmosphere | ±0.150 (22 one-sided nulls); **±0.059 with the R-E-L bracket his eclipse field actually used** (proposed) | ±0.33 (3 one-sided night windows, max; rms ±0.22), of which ≤0.10 overlaps the scale term |
| night maps | 0.100 ″ quasi-static, V/H 1.1, alt 54° | 0.260 ″, V/H 2.4, alt 8.5–12.4°. Zenith floor now on TWO instruments: Leon 0.067 ″ / ±0.12, Leakey 65PHQ 0.078 ″ / ±0.12, both isotropic |
| estimator | Method 1, no nuisance | Method 1 + vertical-deg-2 nuisance |
| structure | one 0.62 s master + linked close-in pair | per-star union of the 0.6 s and 1.2 s tiers, two-witness admission (a single master was built and rejected: it re-admits the G 9.10 corrupted centroid) |
| convention | moments + Gaussian (Bruns' own) | windowed + annular; −0.08 ″ under cell 1's convention; the 2×2 on Leon: background axis +0.14 ″, estimator axis −0.38 ″ (the estimator is the lever here, the reverse of Bruns) |

Three things came out of the levelling that the earlier Leon record did not have: the
imported-scale term (it had been dropped from the quote; on a one-sided field the free
offsets and rotation do not suppress a uniform scale error, and the term is the budget's
largest), the star table on disk, and a defect in cell 1's own field chart (arrows drawn
2.087× longer than their scale bar through nine reviewed revisions; fixed as revision 10
with a runtime round-trip assertion in both chart tools).

**The admission rule is now fixed matrix-wide** (Douglas, 2026-09-02), before cell 2 is
reduced rather than after its outliers are known. Leon's record admits every catalogue match; six of its 42 stars
are single-witness (detected in one tier only), and the cross-tier consistency vet — the
filter that removed the corrupted G 9.10 centroid — cannot act on those at all. The
visible cost is one star: G 10.00 at 6.35 R☉, a 2-px-footprint single-tier detection that
the pipeline's own stage-2 fit had already flagged, sitting +3.5 σ off the curve. A
**two-witness rule** (admit only stars seen in both tiers) removes it and everything else
beyond 2.5 σ, costs six stars and ±0.04 ″ of statistical error, and moves L by −0.06 ″.
It generalises a rule the union already applies to stars fainter than G 11. **Adopted:
it is Leon's reduction of record from 2026-09-02, and Mexico and Portland are to be
reduced under it from the start.**

The exposure decision stands as the handoff stated it: only the 0.6 s and 1.2 s tiers
carry the eclipse field; the 0.1 s tier's unique annulus yielded nothing, its innermost
recovery is no better than the 0.3 s tier's, and Leon had no bright star close in for the
short tiers to reach. The full four-tier union (2.58 ± 0.60, or 2.21 after the G 12
hygiene) is drawn as a cross-check chart, not quoted.

## Cell 2 — Mexico 2024 (Station 1), opened 2026-09-02

Full record in `docs/STEP3_2026.md` §"Cell 2 opened". Tools in `tools/matrix_station1/`;
outputs in `D:\MEE2024 output\MEE_output\station1_record\`. Measured on the 2024-era
archives; two of the seventeen raw zenith blocks were then found and put in
`I:\Mexico 2024\Station 1 Zenith\` (fields 1–2, 05:32:53Z and 05:35:48Z); the other fifteen
are not on this machine. The raw eclipse frames and their bias/dark/flat sets are on `G:`.

| measurement | result |
|---|---|
| §18.3 moment bias (17 zenith fits) | radial only, +3/+12/+22/+31/+10 mas/mag by radius bin, 17/17 same sign, bright inward, −34 mas beyond 2500 px (Leon +299) |
| worth in L through Station 1's geometry | Method 1 −0.035″, **Method 2 −0.011″** (dS −1 ppm); 5× → −0.057″ |
| quasi-static floor (quadratic free) | **0.076″** (0.070–0.085), sensor x 0.061″ / y 0.047″ |
| null, Method 1 vertical-deg-2 | ±0.30″ (16 pairs), ±0.21″ without the +45 ppm event |
| **null, Method 2** | **±0.10″** (photon floor 0.026″); the 45 ppm event alone cost 0.19″ |
| corrections mismatch (first pass) | −163 ppm self-null; Method 1 −3.9″ in every pair, Method 2 absorbed it exactly |
| eclipse field, Method 1 | +15.9″ — the eclipse scale is −600 ppm from the zenith mean (daytime refocus + 5 °C) |
| eclipse field, Method 2 isotropic | 1.677 ± 0.278″ (base), 1.815 ± 0.287″ (vertical-deg-2), S +599 ppm, 156 stars, rms 0.39″/axis |
| anisotropy test | Sy − Sx = +10 ± 16 ppm, skew +8 ± 6 ppm, ΔL −0.009″ — one isotropic S is enough |
| **2 × 2 on the raw zenith pair** (`I:\Mexico 2024\Station 1 Zenith`, fields 1–2 of 17) | outer-field bias, mas/mag: moments + annular +13–17, moments + Gaussian +1–16, **windowed + annular +1–4**, windowed + Gaussian +0–10; 2024 σ_sub-3 moments +26–32. Precision identical (rms G ≤ 12: 47–56 mas in every cell). Null pair: windowed within 0.01″ on every estimator, moments within 0.05″, 2024 within 0.09″ |
| modern stack vs its own 2024 quintic, same frames | scale residual −2.6 / −2.0 ppm, transfer rms 0.11–0.12″/axis (G ≤ 13): the right files, and the pipelines agree |
| **eclipse-field convention** (2026-09-03, `s1_eclipse_convention.py`) | re-stacking the same 123 frames windowed vs moments moves L by **−0.27″ (base) / −0.19″ (v-deg 2)**; windowed has a 26 % tighter per-star residual (0.318″ vs 0.431″) and 27 % smaller σ_L (0.235″ vs 0.320″) |
| reference convention, on the same 2 × 2 | −0.018″ / −0.032″ — the two axes are independent, and only the field's own convention matters |
| why they differ | the eclipse field's estimator bias is ~200 mas/mag against the zenith fields' 22–31: the corona's gradient biases a footprint moment inversely with star flux. **Supersedes the −0.011″ injection bound below.** |
| L, windowed re-stack, Method 2 | **1.549–1.623 ″** (moments 1.742–1.889; 2024 record 1.854; GR 1.751) — GR-consistent either way, but 0.25″ apart |
| reference field count (2026-09-03, f3 and f4 found) | three windowed fields reproduce the 2024 seventeen-field reference to **0.006 ″**; the whole reference axis spans 0.076 ″ against the field convention's 0.266 ″. **No need to wait for the missing blocks.** |
| the +50 ppm scale event | the sensor ran **12.7 °C warm** (CCD-TEMP +2.53 against −10.2), at +5.2 ± 0.7 ppm/°C. The eclipse tiers sit at −10.23 °C against the calibration's −10.19: **thermally matched**, 0.1 ppm |
| the flat across the refocus | zenith-session and post-eclipse flats agree to **0.2 % at every radius**; their difference is worth 3–5 mas. The flat cancels in the transfer |
| hot pixels in the eclipse stack | **11 % of 292 detections** land on a dark-flagged pixel (random 0.0 %); 2 of 182 matched a catalogue star; **0 reached the science set**. The dark-free search found none of them |
| **the 2024 eclipse used a dark AND a flat** | its DARK_STACK and FLAT_STACK survive and match the masters from `G:` exactly; the seventeen zenith reductions used neither. On identical stars calibration moves L by **+0.18 ″** and improves the residual 10 %, and recovers ten faint stars |
| the seventeen-field WINDOWED reference | every 2024 zenith stack re-centroided windowed: ~2000 stars per field against the moments' ~1100, mean scale within 0.2 ppm of the 2024 mean, and the coma-bin bias falls from +22/+31 to +3.2/+0.3 mas/mag |
| **the tier union** (three exposures, four blocks, all re-centroided) | 220 stars, 150 seen in all four. Three-witness + residual vet: **L = 1.795 ± 0.108 ″, residual 0.140 ″** against the 2024 single-block choice's 1.805 ± 0.193, 0.281 — the error nearly halved. Moments needs 17 stars vetted out against windowed's 6 |
| the close-in star at 1.87 R☉ | seen by the 0.25 s and both 0.3 s blocks, missed by the 0.4 s — Bruns' situation exactly — but its displacement is **−15.9 ″** against GR's +0.94: a mis-match admitted by the 20 ″ tolerance the −600 ppm scale offset forces, and made in all three blocks alike. **Multi-witness does not catch a consistent mis-identification; only a residual vet does.** Usable inner limit is ~2.0 R☉ in every block |
| block-to-block scatter | the four blocks give 1.95 / 1.71 / 1.81 / 1.43 ″ separately with formal errors ~0.20; the two 0.3 s blocks differ by 0.28 ″. A real block-level systematic, better guide to the true error than any single block |
| **the magnitude-independence test, all three cells** | apparent deflection per magnitude, which gravity requires to be zero: Bruns 2017 moments **−0.03 to −0.10 ± 0.09** (clean, and tight enough to have caught a Station-1-sized bias), Leon 2026 **±0.6 — cannot tell**, Station 1 windowed **+0.03 ± 0.06** (clean) and moments **+0.36 ± 0.07 (5.4 σ, biased)**. Station 1 is the only cell with the magnitude leverage to detect it: 7.8 mag of span and 156–173 stars against 3.2–3.5 mag and 26–40 |
| what that says about Bruns' averaging | there was no bias in his moments to average away, so averaging Astrometrica against MaxIm DL cost him nothing and was also unnecessary. It is not safe for cell 2 |
| the stacking gap (open) | the four 2024 eclipse stacks used the convex-hull `delete_saturated_blob` and **no coronal gradient subtraction**. `eclipse_mask_mode='disk'` and `coronal_subtract`, developed on 2017 and 2026, have never been applied to Mexico. Re-stacking from raw with them is the largest remaining improvement, and the 1.87 R☉ mis-match is the test of it |
| the sky at totality's edges | the 0.25 s block's frames 0000–0014 sit at exactly the 503 ADU bias — no signal — which is why the operator cut them and why re-stacking all 124 fails. Sky rate is steady at 820–870 ADU/s through three blocks, but rises 45 % to 1240 in the last quarter of the 0.4 s block and then returns: unexplained, possibly cloud, and it overlaps the 0.4 s block |
| **clouds, confirmed by a moving gradient** | seven background patches per frame, a plane fitted across them: the tilt swings 60-100 ADU/s per 1000 px within every block on a mean of ~800, and **rotates** (both components change sign within a minute). Across the frame that is up to 380 ADU/s, half the sky level. The brightening episodes coincide with the tilt excursions. Static causes cannot do this; the sky moved. A candidate for the block-to-block L spread, and a reason any coronal treatment should work **per frame** |
| what the cloud does to the data | stars dim **14 %** and the sky rises **21 %** in the steeper-gradient frames (optical depth tau ~ 0.15 in the beam). For a background-limited centroid that predicts a **+28 %** rise in per-frame position scatter; raw split says +68 %, detrended +6 %, and since the cloud thickened monotonically the truth is between. It is NOISE not bias -- the increase is identical for bright and faint stars -- so it averages down in a 123-frame stack and argues for frame weighting, not a correction to L |
| the red filter and cloud (Douglas) | a red filter cuts the blue Rayleigh clear sky but not the grey Mie scattering off cloud droplets, so the same cloud gives a larger FRACTIONAL sky rise through it. The filter buys contrast against clear sky and nothing against cloud. Not testable in single-band data; the frame headers record no filter keyword |
| **Bruns' convention on his NIGHT fields** (Douglas' point: the day-time L/R fields sit at 3400 ADU where an annular subtraction leaves its own flux-dependent residual) | 8 fields, ~900 stars each: moments minus windowed is **+28 mas/mag** in the outer bins, but bright-minus-faint is only +70 mas with the sign consistent in **3 of 8** fields, against Leon's 12/12 and Station 1's 17/17. Windowed gives a 6 % better residual (0.0740 vs 0.0781 ") and the estimator moves the plate scale **10.3 ppm** -- five times the <2 ppm the day-time fields showed, so Douglas is right that the night fields are where to look |
| **the transfer mismatch is S/N, not sky** | on the Mexico eclipse field, whose local sky spans a factor of 3.3 in one frame: windowed-minus-moments correlates with log S/N at **r = +0.28** and with log sky at **r = +0.03**; binned by S/N it runs monotonically from −191 to +129 mas. So calibration and science transfer cleanly when their S/N ranges match, not their sky levels, and the control is the magnitude cut. Moving both sides to windowed REMOVES a mismatch rather than creating one: the moment's bias runs ~25 mas/mag on the Mexico zenith to 360 on its eclipse field, a factor of 14, while windowed is near zero in both |
| why Bruns is exempt, mechanically | the window is a fixed σ = 2.0 px Gaussian, so what matters is window σ / PSF σ: **Bruns 2.87** (removes 6 % of the profile width), Leon 1.62 (15 %), Station 1 **1.26** (22 %). On his undersampled 1.64 px stars the window is nearly flat across the star and the windowed estimator collapses towards the plain moment, so the two cannot disagree — which is why they do not, and why his two programs agreed to 0.032 ″ |
| where each cell's preference actually comes from | Bruns: neither field has leverage, **no preference** (windowed 6 % better residual, nothing more). Leon: the **zenith** decides it (+299 mas, 12/12) while its eclipse field is powerless (±0.6 ″/mag). Station 1: the **eclipse field** decides it (+0.360 ± 0.066, 5.4 σ) while its zenith is only modest. No two cells are decided by the same field |
| **the residual ramp with field radius** (Douglas, 2026-09-04) | three causes, separated. In constant-only fits a scale mismatch IS a linear ramp by construction: the -163 ppm flags bug gives 1.74 " at the corner, the eclipse's -600 ppm gives 6.4 " and that is the ~4 " stage-2 rms. In the FREE quintics the ramp is 1.18-1.44 and is **isotropic scatter with a mean of zero** -- a septic does not flatten it (it helps the centre more), the PSF is *sharper* at the edge (7.55 -> 5.26 px FWHM), and the mean radial residual oscillates about zero (-46 to +55 mas) while the rms grows 81 -> 115. Vignetting explains ~1.06 of the 1.28; the rest is unattributed noise already carried in the budget |
| **the one place the ramp does harm** | it forces `distortion_fit_tol = 20 "` on the eclipse fits, and that loose gate is what admitted the -15.9 " mis-match at 1.87 R_sun. **Fix: match with the scale free, then refit constant-only on that star set** -- tight matching without letting the fit absorb deflection. Next methodological change |
| **darks: zenith no, eclipse yes** | detections on a dark-flagged hot pixel: zenith fields 0.01 % against a 0.01 % random rate (0-1 of ~2100 matched stars), eclipse 11.3 %. The cause is drift: the zenith fields move 0.3-1.0 px in x and 1-10 px in y, which smears hot pixels below detection; the eclipse is tracked to sub-pixel and stacks them coherently to ~37 sigma |
| **flats: harm the zenith fits** | with the zenith session's OWN flat, three fields lose 9 % of their stars and gain 7-13 % of residual (f1 2104 -> 1912 stars, 0.1258 -> 0.1419 "). Vignetting has already cost the photons; dividing the corners by 0.75-0.90 amplifies what noise is left. NOT separated for the eclipse field, where dark and flat were applied together |
| the tube, night to day | aluminium, 28.5 in, alpha 23.1e-6/K, effective focal length 420 mm from the plate scale: +5 K predicts **+199 ppm**. Measured **-640 ppm**, so the sign is wrong for thermal expansion and the refocus moved the sensor 268 um inward (352 um once thermal is added back) |
| **the admission rule with four blocks** | all four gives the cleanest residual (0.126 ") but LOSES the two innermost stars at 2.02 and 2.50 R_sun, and its sigma_L is no better than three-of-four (0.104 vs 0.105) because the geometry worsens (h 26.7 vs 25.4). **Three of four** keeps every star, matches the precision, still demands independent confirmation. L moves 1.781 to 1.987 across the four nested rules -- a 0.21 " sensitivity that must be fixed in advance |
| quintic vs septic | **not settled.** A septic improves the in-sample rms 5 % but mostly at the centre, which is no proof. The out-of-sample test -- a septic seventeen-field reference, refit the eclipse, see whether the null scatter falls -- has not been run |

Strategy adopted: Method 2 with the pipeline's isotropic S; **windowed + annular, now
measured on the raw zenith pair** (the estimator is the lever; the background matters only
for the moments); the 2024 moment quintic kept
until the raw zenith frames are found; corrections flags matched on import and checked by a
zenith null; the first task of the reduction is the frozen-cubic-at-a-different-focus test.
