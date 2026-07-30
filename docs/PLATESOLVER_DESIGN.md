# Plate solver: measured behaviour, statistical theory, and the improvement plan

Written 2026-07-30 against commit `9da4670`. Every number is measured or derived, and the
derivations are checked against the measurements. The measuring instrument is
`tools/synthetic_field.py`, which synthesizes ground-truth centroid lists from the offline
Gaia catalogue (gnomonic projection at a known pointing → cubic distortion → detection
incompleteness and magnitude-ordering scatter → Gaussian centroid noise).

Goals, from the project owner: **more reliable, wider FOV range, faster.**

---

## 1. Measured state of the current solver

Test frame: 2000×3000 px, 120 detected stars, 0.3 px centroid noise, 3 px edge
distortion, G<12 synthetic fields. Triangle DB: **113,121 anchors × 153 = 17.3 M
triangles** (169 MB), built from Tycho with `theta_pat = 1.7°`.

### FOV envelope (single draw per FOV)

| FOV | region | stars in field | solved | verified stars | time |
|---|---|---|---|---|---|
| 0.6° | mid-lat | 5 | ✗ | — | 0.3 s |
| 1.0° | mid-lat | 12 | ✗ | — | 4.8 s |
| 2.0° | mid-lat | 56 | ✓ | 33 | 17.2 s |
| 4.0° | mid-lat | 262 | ✓ | 94 | 6.0 s |
| 6.0° | mid-lat | 593 | ✓ | 93 | 5.5 s |
| 8.0° | mid-lat | 1051 | ✓ | 95 | 5.8 s |
| 10.0° | mid-lat | 1585 | ✗ | — | 12.0 s |
| 2.0° | galactic plane | 516 | ✓ | 85 | 5.9 s |
| 6.0° | galactic plane | 4654 | ✓ | 96 | 5.9 s |

### Reliability over detection draws (8 random dropout/ordering draws each)

| pointing | FOV | success |
|---|---|---|
| mid-lat (210, +35) | 2.4° | **8/8** |
| zwo3-like (356, +45) | 2.4° | **8/8** |
| sparse high-lat (30, −20) | 2.4° | **6/8** |
| mid-lat | 8.0° | **4/8** |

The dominant failure mode is **not** noise or pointing — it is *which* bright stars
happen to survive detection ordering. The anchors are a strict brightest-`f` prefix, so
an unlucky ordering draw removes every viable query triangle.

### Other limits

- **Sparse fields**: solves with 10 detected stars, fails at 7 (FOV 2.4°).
- **Noise ceiling**: solves at 4 px noise, fails at 8 px. (Derived below: the shape
  tolerance implies failure at ~`TOLERANCE·d/(2√2)` ≈ 5 px for d ≈ 2000 px. Matches.)
- **Junk rejection**: 3/3 pure-noise fields rejected. No false positives observed.
- **Timing**: 5–9 s per solve after a one-off 10.4 s DB load; failures cost 11–17 s
  (the mirror retry doubles the work).

---

## 2. The false-positive estimator: audited, one hole found and fixed

`estimate_acceptance_threshold` models chance matches as the **maximum of N Poisson
variables** (Briggs–Song–Prellberg asymptotic via Lambert-W). I compared it against the
exact quantile, `min{k : 1 − F_λ(k−1)^N ≤ 10⁻³}`:

| n_obs | θ | density | estimator | exact | verdict |
|---|---|---|---|---|---|
| 100 | 5″ | 1× mean | 12 | 10 | OK |
| 100 | 36″ | 1× | 21 | 20 | OK |
| 100 | 120″ | 1× | 48 | 46 | OK |
| 30 | 36″ | 1× | 17 | 15 | OK |
| 300 | 36″ | 1× | 29 | 28 | OK |
| 100 | 36″ | **5×** | 21 | **33** | **unsafe by 12** |
| 100 | 36″ | **10×** | 21 | **44** | **unsafe by 23** |

**Verdict: the mathematical approximation is excellent** — within 1–2 of the exact
quantile everywhere, covered by the `+3` addon. **The modelling assumption is the hole:**
it uses the all-sky mean density, and the galactic plane runs 3–10× that. A dense field
could accept a false solve with ~21 matches when 33–44 are needed.

**Fixed in this commit**: `estimate_acceptance_threshold` takes `local_density`
(stars/steradian), computed from the actual bounding-box star count that
`match_centroids` already performs — the information was already in hand and being
discarded. Locked in by `test_acceptance_threshold_tracks_local_density` and
`test_acceptance_threshold_beats_the_exact_quantile`, which compare against the exact
quantile rather than pinning constants. The real-field corpus still solves with wide
margins (85–99 matched against thresholds that rise only in dense fields).

Two residual conservatisms, both harmless: `N = C(N_cat,3)·C(g,3)·TOL²` is a vast
overestimate of the hypothesis count, but the threshold only feels `log N`, costing ~1–2
extra required stars; and mutual-nearest-neighbour matching plus the 2× confusion ratio
make true chance matches rarer than the model assumes.

---

## 3. How many stars and triangles does a solve need? (derivation, checked)

Notation: field width `x` (degrees), per-star centroid error `ε` (angular), catalogue
density `ρ` (stars/sr), matching radius `θ ≈ 3ε`, `d` = typical triangle side.

### 3.1 Verification stars

A wrong solution places `n_obs` stars at random over the candidate footprint; each has
chance-match probability `p = πρθ²`, so chance matches are `Poisson(λ = πρθ²·n_obs)`.
Against `N_eff` effective hypotheses with allowed failure probability `P_fail`:

    accept at k matches, where   N_eff · P(Poisson(λ) ≥ k) ≤ P_fail
    ⇒  k* ≈ (ln N_eff + ln 1/P_fail) / ln(k*/λ)   (+ the 3 defining stars)

Worked example (G<12 catalogue, ρ = 2.4×10⁵ sr⁻¹, ε = 1″, θ = 3″, n_obs = 100):
λ ≈ 0.016, ln N_eff ≈ 28, so k* ≈ 6 + 3 ≈ **9**. With the coarser θ = 36″ the pipeline
actually uses, the formula gives the 15–21 range the estimator produces. The formula and
the code agree.

### 3.2 Triangles: the information budget

The solver must select one cell in solution space. At the current consensus tolerances
(0.025° position, 0.025° roll, 1% scale over a ~3× scale range):

    M_sol ≈ (4π/πθ_c²) · (2π/θ_r) · (Δln s/δ_s)  ≈ 2×10⁷ · 1.4×10⁴ · 110 ≈ 3×10¹³
    bits needed  B = log₂ M_sol ≈ **45 bits**

A triangle matched at shape-space tolerance `t` (area `πt²` in (ratio, angle) space of
total area 2π) contributes

    I_tri = log₂(2π / πt²) = log₂(2/t²)      → t = 0.01 ⇒ **14.3 bits**

So the minimum consensus size is `k_tri ≥ B/I_tri ≈ 45/14.3 ≈ 3.2` — **the existing
"≥4 non-redundant triangles" rule is exactly the information budget**, derived rather
than tuned. The measured false-candidate rate confirms the model: predicted chance
matches per query triangle = 17.3M × t²/2 = **865**; observed in real solves ≈ 900–1000.

The measurement limit on `t` is set by centroid error: `δratio ≈ δangle ≈ 2√2·ε/d`. For
the zwo fields (ε ≈ 0.15″ on the stack, d ≈ 1°) that is **~2×10⁻⁴ — the current
tolerance of 0.01 is 50× coarser than the measurement supports.** It has to be, because
the *database* is the noisy party: Tycho positions carry up to 2.5″ of proper-motion
error, and unmodelled optical distortion shifts edge stars several pixels. This is the
central inefficiency, and it is fixable (§5.1, §5.2).

### 3.3 The FOV envelope, derived

Three independent ceilings, each matching the measurements:

- **Small-FOV floor — catalogue depth.** Solving needs ~10 usable stars *that exist in
  the DB star list* (700k brightest Tycho ≈ 17 stars/deg², mag ≲ 11.3). A 1° field holds
  ~12 stars to G<12 but only ~8–9 above the DB depth, and the anchor list (240k, mag
  ≲ 10.3) is sparser still: `x_min ≈ √(10/17) ≈ 0.8°`. Measured: 1.0° fails, 2.0° works.
- **Wide-FOV ceiling — the brightest-g window.** Query triangles come from the image's
  `g = 18` brightest stars; DB patterns contain the 18 brightest within 1.7° of an
  anchor. The expected number of image-top-18 stars inside one pattern disc is
  `18·π·1.7²/x²`. Needing ~2–3: marginal at `x = 8°` (2.6 expected — measured 50%
  success), dead at `x = 10°` (1.6 — measured fail).
- **Wide-FOV ceiling — projective breakdown.** The image→sky map is projective; ratio
  and angle are only *similarity* invariants. The projective residual grows as
  `(x/2 in radians)²`: ~5×10⁻³ at 8°, exceeding half the tolerance; ~8×10⁻³ at 10°. So
  even with a wide-pattern DB, similarity invariants stop working around 10–12° at
  current tolerance — this is the rigorous form of "quads win on wide fields with high
  curvature" (a 4th point buys genuinely *projective* invariants, e.g. cross-ratio-like
  quantities, exactly invariant under re-pointing).

---

## 4. Triangles vs quads vs shape space

**Shape-space triangles (the idea from your other branch) are the right upgrade, and
here is the argument.** The current (ratio, Δφ) coordinates carry a distorted metric: a
fixed Euclidean tolerance ball corresponds to wildly varying *shape* differences —
near-degenerate triangles (ratio → 0, near-collinear) get effectively looser matching,
which is where false matches concentrate. In Kendall's shape space (for triangles, a
sphere of radius ½ — equivalently normalised side-ratio coordinates with the Fubini–Study
metric), isotropic centroid noise maps to an *isotropic* shape uncertainty. Matching with
a metric ball then delivers uniform false-positive density at a uniform true-positive
rate — the estimator's assumptions become exactly true instead of approximately true, and
tolerance can be tightened toward the measurement limit. Expected gain: each factor of
`c` tighter in tolerance cuts chance matches by `c²`; even 3× tighter is ~10× fewer
candidates → faster consensus and fewer verification calls.

**Quads**: per structure, a 4D hash carries roughly twice the bits of a triangle
(≈ 4·log₂(d/3ε) vs 2·log₂(d/3ε)), so candidate lists shrink quadratically better, at the
cost of needing 4 mutually-catalogued stars — worse in sparse/narrow fields, and the DB
stores 4 floats+4 ids vs 2 floats+3 ids. Your information-density instinct is right
*per byte for narrow fields*; quads win *per matching operation* and on wide fields via
projective invariants. Conclusion: **triangles remain correct for the 1–8° regime this
instrument works in; a quad (or 5-point projective) table is the right tool if the >8°
regime ever matters.** Both can share the anchor/neighbour infrastructure.

---

## 5. Improvement plan, ranked by measured impact per effort

1. **Rebuild the pattern DB from the offline Gaia catalogue** (we now own the input:
   3.1M stars, sub-mas positions, proper epoch). Kills the 2.5″ Tycho error that forces
   the coarse tolerance, deepens the star list (fixes the 1° floor), and the build is a
   local computation. This unlocks every other gain.
2. **Multi-scale pattern DBs**: `theta_pat ∈ {0.6°, 1.7°, 4°}` with correspondingly
   deeper/shallower star lists, cascaded (or selected by platescale hint). Extends the
   envelope both directions; the 4° table pushes the top-g window ceiling from ~8° to
   ~18° (where the projective ceiling takes over).
3. **Anchor sampling instead of a strict brightest-f prefix** — draw anchors from the
   top ~2f with retries. Directly attacks the measured 6/8 and 4/8 reliability, at near
   zero cost.
4. **Kendall shape coordinates + tolerance from an ε estimate** (per-image, from the
   stacking residuals we already compute). Fewer candidates → faster; uniform statistics
   → the estimator's model becomes exact.
5. **Platescale/pointing hints** (`--platescale`, `--ra/--dec`): prefilter candidates on
   `log s` before consensus. Large speed win for the 99% case where optics are known.
6. **Local-density-aware acceptance threshold** — done in this commit (§2).
7. **Early exit + lazy/mmap DB load** — the 10.4 s load dominates single-solve latency;
   the arrays are already `.npy`-able like the starcat format.

The synthetic-field harness (`tools/synthetic_field.py` + the new slow tests) is the
regression instrument for all of these: every change must hold the junk-rejection rate at
0 false positives and push the FOV/reliability table outward.
