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
- **G 7.09 at 1.62 R⊙: raw radial displacement +1.439″ vs GR's +1.079″** — a real
  inner-annulus deflection detection, +0.36″ (≈1.6 σ) high, admitted-candidate.
- **G 7.52 at 1.49 R⊙: +0.037″ vs GR's +1.174″ — a mask-edge casualty.** It sits
  13 px outside E2's painted disk (edge at 1.47 R⊙) in a 49 kADU sky; the
  coronal-gradient residual after blur-subtraction pulls its centroid sunward and eats
  the deflection. This is precisely the regime Bruns' own continuity correction was
  built for. Excluded pending per-frame verification; admitting it un-verified would
  drag L to 1.39 for a demonstrably instrumental reason.

**Cell-1 first-pass verdict: L = 1.56 ± 0.14 (stat) ± 0.10 (scale) ″ on the
held-constant default (GR at +1.2 σ), and 1.66 ± 0.14 ± 0.09 ″ with the inner annulus
handled per the anchor doctrine (GR at +0.6 σ). Newton (0.876″) is excluded at
≈ 4.0–4.6 σ — the Eddington discrimination Leon's atmosphere forbade, delivered by
altitude 54° plus the L/R bracket, exactly the 2027 forecast.** Bruns' own published
result (GR to ~3 %) remains far tighter: he used many more stars (his analysis reached
fainter magnitudes on single well-modelled images) and his continuity correction
recovered the inner annulus this chain currently masks.

Charts: `matrix_bruns2017/deflection_b17.png`, `field_b17.png` (D:).
Union table: `matrix_bruns2017/union_full_r145.csv`.

Open items for this cell: (1) per-frame E2 verification of both inner stars (the
anchor treatment — 11 frames each); (2) the honest atmospheric systematic via the
S1-style gate run on Bruns' own 29 night fields (`bruns2017_nights/`) instead of
quoting Leon's ±0.33; (3) the mask-edge exclusion margin (disk radius + centroid
window) should become an explicit rule in the S0 tool rather than a post-hoc catch.
