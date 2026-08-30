# The four-dataset matrix: one tool chain, four skies

**Started 2026-08-29.** The acceptance test for the Leon step-3 machinery
(`docs/STEP3_2026.md`): the same tools with the same defaults must reduce three solar
eclipses and one full moon with minimal hand tuning — hand tuning being what attracts
the accusation of manipulation. The convex-hull blob retires when this passes.

| cell | data | calibration geometry | known answer |
|---|---|---|---|
| 1. Bruns 2017 (Casper, WY) | `I:\2017 eclipse images Don Bruns\2017 Eclipse images\eclipse` | L/R bracket (±7.4° both sides) | Bruns 2018: GR to ~3 % |
| 2. Mexico 2024 (Station 1) | `G:\Mexico April 2024\Station-1-Eclipse-Data` | pure Method 2 | GR-consistent (v1.3.x) |
| 3. Leon 2026 | done — `docs/STEP3_2026.md` | one-sided CAL_piLeo | L = 1.98 ± 0.60 ± 0.33 ″ |
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
