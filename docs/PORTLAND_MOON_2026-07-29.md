# Portland Moon test, 2026-07-29 — can it work next to a full Moon?

**Dataset:** `I:\Toby Portland 2026\Toby Portland Moon`, 51 fields, reduced under v1.3.6.
**Question asked:** does the plate solve and the centroid finder survive a full Moon in
frame, at every Moon position and every rung of the exposure ladder, and is ghosting a
problem? **Short answer: yes, yes, and no.** With one refinement and three findings the
summary file does not show.

Companion: [`PORTLAND_2026-07-29.md`](PORTLAND_2026-07-29.md) (the zenith set from the same
rig and night), [`LEON_SCRIPT_REVIEW.md`](LEON_SCRIPT_REVIEW.md).

---

## 0. What was captured

Three Moon positions × three block types, 51 fields:

| | ladder | ghostdeep | calfield |
|---|---|---|---|
| **CENTER** | 14 (two ladders) | 2 | 7 |
| **CORNER** | 7 | 1 | 6 |
| **MIDEDGE** | 7 | 1 | 6 |

Each `ladder` is the eclipse science ladder, one folder per rung — 0.1×24, 0.3×12, 0.6×6,
1.2×6, 0.6×6, 0.3×12, 0.1×24, confirmed against the folder timestamps. Each `calfield` is
the cal ladder (0.3×6, 1.0×6, 2.0×8). The **calfield is the control**: it is a separate
pointing ~10° away, and the Moon's projected position falls thousands of pixels outside the
frame in all three, so anything that moves with the label there is time, not the Moon.

**The Moon's position was verified rather than assumed** — projected into each frame from
its ephemeris through that field's own plate solution:

| block | Moon at | as a fraction of the frame |
|---|---|---|
| MOON_CENTER | (3194, 2141) ± (83, 127) px | (51.1%, 51.3%) |
| MOON_CORNER | (708, 657) ± (1, 6) px | (11.3%, 15.7%) |
| MOON_MIDEDGE | (2997, 422) ± (2, 4) px | (48.0%, 10.1%) |

The labels are correct, and the ±1–6 px repeatability says the projection is exact enough
to build the ghost test on. MIDEDGE puts the lunar *limb* almost exactly tangent to the
bottom edge (the Moon's radius is 421 px at this plate scale, and its centre is 422 px in).

---

## 1. The solves: not merely successful, comfortable

**51 fields, 102 solves — stage 1 and stage 2 each solve independently — and 0 failures.**
Better than that:

- **Nothing escalated.** Zero fields needed a second anchor round, zero had `noise_px_used`
  raised above the 0.3 default, and zero produced a rejected candidate. Every solve was
  first-try.
- **The worst margin was 3.23×** the acceptance threshold; the median 4.53×, the best 6.60×.
  The threshold scales with the centroids offered, so the ratio is the fair comparison —
  and for scale, the two London fields that *failed* (ROADMAP §1.9) matched 3 stars against
  a threshold of 13, a margin of 0.23×.

That holds at every rung, including the 0.1 s rungs which offered as few as **47 centroids**
for the whole frame. The thinnest field in the set solved on the first try with 42 stars
matched against a threshold of 13.

**So the claim is confirmed, and it is stronger than "it worked":** nothing in this dataset
came within a factor of three of the boundary.

---

## 2. Ghosting: a hard null

A reflection ghost lands near the point diametrically opposite the source through the
optical axis. So the test is: take every stage-1 centroid with no catalogue match within
3 px (a spurious detection — 118 of them across the 32 in-frame fields), and ask whether
they pile up at `2 × frame centre − Moon`.

| test radius | spurious found there | expected if spread uniformly | |
|---|---|---|---|
| 300 px | **0** | 1.3 | — |
| 500 px | **0** | 3.5 | — |
| 800 px | 8 | 8.4 | 0.95× |
| 1200 px | 8 | 16.6 | 0.48× |
| 1600 px | 15 | 27.2 | 0.55× |

**Zero at small radii, and at or below the uniform expectation at every larger one.** Not a
weak detection — an absence, with a 95% upper limit of about 2 spurious detections across
all 32 fields at 300 px.

The fit agrees. Matched stars within 700 px of the ghost point have a median residual of
**229 mas**; everywhere else, **227 mas**.

And the 118 spurious detections are not clustered anywhere interesting: their density per
unit area is flat at 0.7–4.4 per field from the Moon out to 11 lunar radii. The only mild
concentration is in the extreme frame corners, which §3 explains and which is not a ghost.

**Conclusion: ghosting is not a consideration for this rig at any of the three positions.**

---

## 3. But position *does* matter — through scattered light, not ghosts

This is the refinement. Ghosting is not the mechanism, and the fit does not care where the
Moon sits — but the **star yield** does, and by a lot.

| block | CENTER | MIDEDGE | CORNER |
|---|---|---|---|
| ladder — centroids | 56 | 72 | **104** |
| ladder — 90th-pct V reached | 9.58 | 9.82 | **10.06** |
| ghostdeep — centroids | 142 | 168 | **254** |
| ghostdeep — 90th-pct V | 10.27 | 10.70 | **11.00** |
| *calfield (control)* — centroids | *194* | *183* | *206* |
| *calfield (control)* — 90th-pct V | *11.22* | *11.16* | *11.20* |

**The control is flat to 0.06 mag. The in-frame blocks run monotonically.** Putting the Moon
at the frame centre costs about **45% of the detectable stars and 0.5–0.7 magnitudes of
depth** against putting it in a corner. That is the scattered-light halo: centred, it covers
the whole frame; in a corner, most of the frame is far from it.

### The halo profile, measured inside single fields

Pooling across positions would confound this with the effect above, so here it is *within*
one block at a time — 90th-percentile V of matched stars, by distance from the Moon in
lunar radii:

| block | 3–4 R | 4–6 R | 6–8 R | 8–11 R | 11–15 R |
|---|---|---|---|---|---|
| CORNER_ghostdeep | — | 10.54 | 10.86 | **11.16** | 11.01 |
| MIDEDGE_ghostdeep | — | 10.41 | 10.54 | **10.81** | — |
| CENTER_ghostdeep | 9.92 | 10.25 | **10.51** | — | — |
| CORNER_ladder | 9.31 | 9.58 | 9.81 | **10.30** | 9.92 |

**The halo costs 0.6–1.0 magnitudes within ~4 lunar radii and flattens out by ~10** — which
at 2.21″/px is 2.6°, so the halo reaches about two and a half degrees.

### And it turns over again at the frame corners

Look at the last column. `CORNER_ladder` peaks at 8–11 R and then *loses* 0.4 mag by
11–15 R; `CORNER_ghostdeep` does the same, more gently. Eleven to fifteen lunar radii from
a corner-mounted Moon is the opposite corner of the frame — and that is exactly where the
zenith set found the PSF bloated to **1.38× the centre**
([`PORTLAND_2026-07-29.md`](PORTLAND_2026-07-29.md) §5.1). The same optical defect, arriving
from a completely different dataset.

**So there are two opposing gradients across the frame, and the best place is neither the
bright object nor the frame corner.** The halo pushes you away from the Sun; the reducer
spacing pushes you away from the corners.

**This is an argument for the mid-bottom-edge science geometry that the ghost reassessment
did not make.** With the Sun 0.5° inside the bottom edge and the field extending upward, the
science stars sit in the middle of the frame — several lunar/solar radii clear of the halo,
and well inside the radius at which the corner bloat starts. The corner geometry buys the
best halo escape and spends it on the worst optics.

*Caveat worth stating.* A full Moon is not an eclipsed Sun. Their total brightnesses are
comparable (both around mag −12.7), but the Moon's is concentrated in 31′ while the corona
is spread over degrees, and during totality the photosphere is occulted so there is no
equivalent of the lunar disc at all. The *shape* of the falloff is a property of the optics
and the air, and transfers; the *amount* should be treated as an upper bound.

---

## 4. Three findings the summary file does not show

### 4.1 Half the fields could not run the dark-free hot-pixel search

| | fields | below the 3 px dither floor | search actually ran |
|---|---|---|---|
| ladder | 28 | 18 | 2 |
| calfield | 19 | 8 | 11 |
| ghostdeep | 4 | 0 | 0 |
| **total** | **51** | **26 (51%)** | **13** |

By rung:

| rung | dither span | below the floor |
|---|---|---|
| 0.1 s × 24 | 2.85–4.03 px | 2 of 8 |
| 0.3 s × 11 | 0.00–8.32 px | **13 of 15** |
| 0.6 s × 6 | 1.05–2.00 px | **8 of 8** |
| 1.0 s × 6 | 1.66–3.95 px | 2 of 6 |
| 1.2 s × 6 | 3.82–4.43 px | 0 of 4 |
| 2.0 s × 8 | 2.63–8.83 px | 1 of 6 |

The middle rungs are the short ones — six frames of 0.6 s is four seconds of wall clock, and
four seconds is not long enough for the mount to walk three pixels.

**And the dither that did exist came from the 3.7° polar error.** Fix the alignment, as
§8 of the zenith note recommends, and 0.97″/s becomes 0.004″/s: a 0.6 s × 6 rung would
dither **0.01 px** instead of 1.5. Essentially every rung of the eclipse ladder would then
fall below the floor.

**That turns the two-gain dark library from good practice into a hard requirement.** It is
already in `leon_darks_v1.4`, so nothing needs to change — but the reason is now measured
rather than argued, and the two changes interact in a way that is easy to miss: the fix for
the mount removes the accident that was covering for the missing darks.

### 4.2 The ladder wall time, which the eclipse script asks for

`leon_eclipse_v1.16` estimates 38–44 s and flags "bench cadence check to confirm". This run
is that check, four times over:

| ladder | wall | integration | duty |
|---|---|---|---|
| CENTER 01:41:47 | 36 s | 26.1 s | 72% |
| CENTER 01:52:57 | 35 s | 25.4 s | 72% |
| CORNER 01:59:54 | 34 s | 25.4 s | 74% |
| MIDEDGE 02:05:02 | 34 s | 25.4 s | 74% |

**34–36 s, not 38–44.** That is 4–10 s of margin recovered inside a 104 s totality, and it
should go into the header so the time budget stops carrying a pessimistic number.

One number moves the other way: the 0.1 s rung takes 7–8 s for 24 frames, i.e. **0.31–0.33 s
per frame against the script's 0.285 s cadence floor**. The floor is about 15% optimistic —
the 183 MB/s USB3 figure does not include the per-frame save and display overhead the same
script measures at ~0.37 s elsewhere. It makes the 0.1 s tier's duty ~31% rather than 35%,
and it is already inside the 34–36 s measured above, so nothing needs redesigning.

### 4.3 Disk: 52 MB per field, and it is not in the archive

Each field folder keeps its `CENTROID_OUTPUT.../` working folder, and inside it a
**52 MB `STACKED*.fit`** — the stacked frame. Fifty-one fields is 2.65 GB, which is
essentially the whole 2.8 GB dataset; the archives themselves are 7 KB each.

Not a bug, and it is the raw material F6 (open a previous run from disk) would need. But it
is worth knowing that a batch costs ~52 MB per field regardless of how few stars the field
had, and that nothing ever cleans the working folders up.

---

## 5. Answering the question as asked

| claim | verdict |
|---|---|
| The plate solve works next to a full Moon, at every position | **Confirmed**, with the worst margin 3.2× the threshold and no escalation anywhere |
| …and at every rung of the ladder | **Confirmed**, down to a field offering 47 centroids |
| The centroid finder works | **Confirmed** — and the spurious rate is 2–21%, spread uniformly, not clustered |
| Ghosting is not a major problem in any position | **Confirmed**, quantitatively: zero spurious detections at the ghost point against 3.5 expected, and no residual degradation there |
| Therefore any position is free for the Sun | **For solving and fitting, yes.** But not for depth: the centre costs 0.5–0.7 mag and ~45% of the stars against a corner, and the frame corners cost PSF quality. The mid-edge geometry already chosen is the right compromise, for a reason §3 supplies and the ghost reassessment did not. |
