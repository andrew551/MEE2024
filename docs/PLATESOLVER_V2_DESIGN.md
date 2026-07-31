# Plate solver v2: theory, and a staged A/B-tested rebuild

Written 2026-07-31. Companion to `docs/PLATESOLVER_DESIGN.md`, which holds the measured
behaviour of the production solver and the statistical groundwork this design builds on.
The v2 solver lives in `mee2024/platesolve2/`, is selected with
`options['platesolver'] = 'v2'`, and the production solver stays the default until v2
dominates the bench (`tools/solver_bench.py`, results in `docs/bench/BENCH.md`).

Goals, from the project owner: rebuild from scratch — pattern DB from the offline Gaia
catalogue, Kendall shape-space invariants, quaternions where they actually help, a
conditional quad layer for >8° — developed in stages where **each stage isolates one
design change and is A/B-measured against the production solver on synthetic fields**
before the next begins.

Provenance note: the `dev-platesolve` branch already prototyped the two central ideas
(shape-space triangles, quaternion consensus). Its math was verified correct and is
adopted; its code is a spike (tip does not import, no tests, hard-coded paths, a broken
double-cover fix, an anti-conservative acceptance test) and is rewritten, not merged.

---

## 1. Theory

### 1.1 Framework: what "optimal" means

A solve selects one cell of solution space ≈ **45 bits** at current consensus tolerances
(`PLATESOLVER_DESIGN.md` §3.2, derived and confirmed against observed candidate rates).
A structure matched at shape tolerance `t` contributes `log2(A_total/A_ball(t))` bits.
The three axes to balance:

- **Space**: records × bytes. Disk is cheap; resident memory and load time are not.
- **Speed**: dominated by candidates/query = `N_db · A_ball/A_total`, which drives all
  downstream consensus and verification work. Production: ~900 candidates/query.
- **Reliability**: P(≥4 matchable triangles survive detection dropout + ordering
  scatter) — the binding constraint today (the strict brightest-f anchor prefix).

The tolerance `t` is bounded below by the sum of every unmodelled error, in shape units,
for a triangle of angular size d:

| source | magnitude | notes |
|---|---|---|
| centroid noise | 2√2·ε/d ≈ 2×10⁻⁴ (stack) … 10⁻³ (1 px single frame) | measurable per image |
| catalogue error (Tycho) | up to ~1.4×10⁻³, tail worse | **→ ~0 with Gaia (sub-mas)** |
| unmodelled optical distortion | ~3 px at edge / 2000 px ≈ 1.5×10⁻³ | dominates post-Gaia for our optics |
| projection curvature | (x_rad/2)²: 3×10⁻⁴ @2°, 1.9×10⁻³ @5°, 4.9×10⁻³ @8°, 7.6×10⁻³ @10° | the wide-field killer |

**Post-Gaia, tolerance can drop from 0.01 to ~2.5×10⁻³ below ~5–6° FOV** — ≈16× fewer
chance candidates; above ~8° curvature forces it back up, which is a different problem
(§1.4). The dev spike ran 0.001 successfully on real fields (~20 candidates/query),
consistent with this budget.

### 1.2 What Kendall shape space buys

The production invariant (side ratio, included angle) carries a distorted metric:
isotropic pixel noise maps to shape-dependent, anisotropic noise in those coordinates,
loosest exactly where chance matches concentrate (near-degenerate triangles). A fixed
tolerance must be sized for the worst shape, wasting selectivity on the typical one.

Kendall's shape space for planar triangles is a sphere; iid Gaussian vertex noise maps
to **isotropic** shape noise of magnitude ~ε/S (S = triangle size). Consequences:

1. A metric ball is the statistically correct acceptance region — uniform true-positive
   rate at uniform false-positive density; the Poisson model inside the acceptance
   estimator becomes exact rather than approximate.
2. Per-triangle, size-adaptive tolerance is principled: `t_i ≈ 3·(ε/S_i ⊕ distortion ⊕
   curvature)`. And the sphere has no coordinate singularity — the periodic-boxsize
   KD-tree hack for the angle wrap disappears.
3. **Mirror handling becomes one conjugate query.** A reflected triangle maps to the
   z-negated shape point, so querying both points in one pass replaces the full mirrored
   re-solve that doubles failure latency today (11–17 s).
4. Bits per triangle at t = 2.5×10⁻³: ~18–19 vs 14.3 today; the ≥4-triangle consensus
   rule keeps a >2× information margin.

Combined with §1.1: **~30–50× fewer false candidates per query** — the direct driver of
post-load solve time.

The verified coordinates (from the dev spike, checked algebraically and numerically on
its 13M-triangle DB — exact unit vectors):

    canonical side order: chirality flip so orientation > 0, cyclic shift so r1 largest
    D = r1² + r2² + r3²
    x = √3 (r1² − r2²) / D
    y = (r1² + r2² − 2 r3²) / D
    z = 4√3 · Area / D                     (z ≥ 0 always)

plus a 3-bit permutation code (values 0–5) recording the canonicalisation, so matched
vertices can be paired positionally with no search. z ≥ 0 means only (x, y) need be
stored — 8 bytes + 1 permutation byte per triangle.

### 1.3 Quaternions: where they help and where they do not

**Final orientation (Wahba's problem): no win.** The quaternion q-method and SVD
Procrustes solve the same least-squares problem and return the same rotation in
microseconds at n ≤ 100. Keep SVD — but with the **Kabsch determinant correction** the
production `_find_rotation_matrix` omits (it can return a reflection today). The dev
spike's batched cofactor inverse + one Newton–Schulz polar step for per-candidate
rotations is the right pattern (with a determinant floor for near-degenerate triangles).

**Consensus clustering: real win — correctness, not speed.** The production cluster key
`[log s, roll, centre]` has two latent failure modes, both verified in code:

- **roll enters the KD-tree unwrapped** (`platesolve_triangle.py`, `vector_plates`), so
  a consensus straddling roll ≈ 0/2π is split into two components and can miss the ≥4
  gate — fields at particular orientations fail for no physical reason;
- (centre, roll) is a singular chart on SO(3): at the celestial poles roll and RA
  degenerate and cluster radii are wrong.

Clustering on `[log s/τ_s, q/τ_r]` (unit quaternion, 5-D KD-tree) is singularity-free
and uniform: distance ≈ ½·(rotation-angle difference) everywhere.

**The double cover must be handled correctly.** q and −q are the same rotation. The dev
spike's handling is broken in a measured way: its near-boundary test lacks an `abs()`
(selecting ~50% of all candidates instead of ~0.03%), and the duplicated rows are not
negated — a no-op that doubles clustering work and inflates the multiple-comparison
count ~2×. Correct recipe: canonicalise the sign (largest-|component| positive) and
insert **negated** twins only for candidates within one cluster radius of the
canonicalisation boundary.

### 1.4 Wide fields (>8°): curvature, the quad folklore, and hints

Any two gnomonic projections of the sky are related by an exact plane projectivity.
Similarity invariants — triangle ratio/angle, Kendall coordinates, Astrometry.net-style
4-D quad codes alike — therefore degrade as (x_rad/2)²: ~5×10⁻³ at 8°, ~8×10⁻³ at 10°.
Two corrections to folklore:

- **A quad code is still a similarity invariant.** True projective invariants need five
  points (a projectivity has 8 DOF; 4 general points ⇒ 0 invariants, 5 ⇒ 2). Quads help
  at wide FOV for a different reason: a 4-D code carries ~2× the bits of a triangle, so
  candidate lists stay manageable even at curvature-inflated tolerance (t⁴ vs t²).
- **A platescale hint changes the game.** With a known platescale, great-circle
  separations are computable exactly from pixel positions, and spherical invariants are
  exactly rotation-invariant at any FOV. For a fixed instrument — the usual case — the
  wide-field problem reduces to the pattern-disc window, solved by a wide-θ_pat triangle
  layer. Hence: hints + wide layer first (S6); quads only if the blind >8° bench still
  shows a gap (S7, conditional).

### 1.5 Verdict: is anchor+satellite + Kendall + quaternions near-optimal?

- **Encoding.** Anchor+satellite is reliability-friendly — each DB anchor independently
  supports a solve — and its ~4–8× storage redundancy over the information-minimal
  encoding is what buys O(1) lookup. Within the ≤500 MB budget this is the right trade.
  The measured failure mode is the *query-side* brightest-f prefix, fixed by progressive
  anchor sampling at near-zero cost (S5a).
- **The dev spike's "dimmer legs only" rule** (store each star triple once, under its
  brightest member) is a real coverage-per-byte win but imports an online brightness-
  ordering assumption (query anchor must be the triple's brightest star) — risky
  precisely because ordering scatter is the dominant measured failure mode, and
  instrument-band vs G-band ordering adds more. It is a **benchable DB variant, not an
  assumption** (S2b), decided on the ordering-scatter sweep.
- **Acceptance test.** Keep the production max-of-N-Poisson estimator with the
  local-density fix. The dev spike's p-value replacement is anti-conservative — it feeds
  the matched stars' own radii into the null model (post-hoc) — and dropped local
  density. A corrected Poisson-binomial version is a possible later experiment, off the
  critical path. Keep the ≥4 non-redundant gate (it *is* the information budget) and
  MAX_MATCH = 100.
- **Bottom line.** With Gaia DB + calibrated Kendall tolerance + adaptive ε + quaternion
  consensus + progressive anchors + mmap/bucket index, the architecture sits within
  small constant factors of the practical optimum for 1–8° blind solves. Remaining
  levers (code quantisation, learned anchor selection, quads) are not worth their
  complexity until the bench says otherwise. Expected end state: **sub-second warm
  solves, ~1 s cold load, 8/8 on today's failure draws, floor below 1°, hinted solves at
  any FOV.**

---

## 2. The stages

Protocol: every stage runs the frozen bench corpus and is compared against the previous
stage's committed results. Gate to proceed: **wrong-solve rate 0 everywhere, junk fields
all rejected, both real fields solved**, and the stage's target metric moved as
predicted. Rollback is always configuration (`platesolver='triangle'`, or the previous
`pattern_db` variant) — the old solver and old DB variants are never deleted.

| stage | isolates | target metric |
|---|---|---|
| S0 | bench harness + v1 baseline + dispatch skeleton | baseline reproduces `PLATESOLVER_DESIGN.md` §1, incl. pole/roll-wrap failures |
| S1 | catalogue + DB format (Gaia, same algorithm; verification switched to the same catalogue) | 1° floor improves; parity with v1 on real fields |
| S2 | invariant (Kendall + permutation codes + conjugate-mirror single pass; tolerance *calibrated*, not tightened) | candidates/query; failure-path time ~halves |
| S2b | DB variant: dimmer-legs dedupe at equal disk budget | ordering-scatter sweep decides, pre-registered rule |
| S3 | tolerance model (per-image ε + FOV curvature + DB floor) | ~10–50× fewer candidates; no regression at 8 px noise |
| S4 | consensus metric (quaternion, correct double cover; Kabsch det fix landed in S1) | pole + roll-wrap cases flip fail→pass |
| S5a | query anchor sampling (progressive rounds from ~2f) | 8/8 on sparse and 8° reliability draws |
| S5b | latency (mmap columns, build-time shape-sphere bucket index, early exit) | cold start 10.4 s → ~1 s; identical candidate sets to S5a |
| S6 | multi-scale θ_pat layers + platescale-hint path | floor < 1°; hinted solves at any FOV |
| S7 | *(conditional)* quad layer for blind >8° | only if S6 leaves a blind 10–18° gap; go/no-go recorded either way |

Key implementation decisions (details in the stage commits):

- **v2 package** `mee2024/platesolve2/` behind `options['platesolver']`; the return
  dict, signature, and `SOLVE_CANDIDATE`/`SOLVE_RESULT` events are contract-identical to
  v1 so no pipeline call site or UI change is needed.
- **Pattern DB = directory of `.npy` + `manifest.json`** (per-file sha256,
  format_version, pinned dtypes), mirroring the starcat store; named variants under
  `<user data>/patterndb/<name>/`; shipped through the same release/`--fetch` machinery
  as the star catalogues; `mee2024 build-pattern-db` builds locally. The solver reads
  the pattern width, θ_pat, and invariant from the manifest — no compiled-in `g = 18`
  coupling. A missing DB **raises with the exact fetch/build command**; it is never
  rebuilt silently inline.
- **Prefix-sum triangle offsets** per anchor (anchors with fewer than `e` legs own fewer
  rows) — removes the fixed-153 arithmetic and the builder's "edge case handling
  unimplemented" crash.
- **No pickled KD-trees** (the dev spike's 686 MiB scipy-version-fragile sidecar is the
  cautionary tale): S1 builds the tree at load; S5b replaces it with a build-time bucket
  index that is plain `.npy`.

Known risks carried into implementation: acceptance-estimator inputs shift at S1
(re-validate junk + galactic plane there, not just S3); epoch drift of high-PM stars
matters once tolerance tightens (record build epoch; exclude extreme-PM stars from
patterns or fold worst-case drift into the DB floor); builder must chunk (~10k
anchors) to bound memory; bench must separate cold/warm timings on Windows; the batched
cofactor inverse needs a determinant floor.
