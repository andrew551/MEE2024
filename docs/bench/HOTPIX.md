# Finding hot pixels without a dark frame

Darks are not always taken, and when they are they are not always usable — the darks in the
bundled `example_with_darks` were shot 45 minutes late and run three times hotter than the
lights, which is exactly when you want a method that does not need them.

**The idea:** a star is fixed to the *sky*, a hot pixel is fixed to the *detector*, and a
dithered sequence tells them apart. Take a bright candidate site and ask the other frames
two questions:

* is it still bright at the same **detector pixel**? → *detector persistence*
* is it still bright at the same **sky position**, undoing that frame's dither?
  → *sky persistence*

Persistence is the **weakest** appearance across frames, in units of the background noise,
so a single good frame cannot carry a bad pixel. A star answers *no, yes*; a hot pixel
answers *yes, no*.

Reproduce with:

```bash
python tools/hotpixel_explore.py "I:/65PHQ 294MM Texas 2024/zenith 1/070424_040415/*.fits" --darks "I:/65PHQ 294MM Texas 2024/070424_050036 darks 10s/*.fits" --out docs/bench/hotpix
```
```bash
python tools/hotpixel_rules.py docs/bench/hotpix
```

The darks are used **only as ground truth**, never as an input to the statistic, so the
result can be scored rather than admired.

## Result: the two populations barely overlap

7 lights, 5644×8288, 12-bit; 5237 candidate pixels above 20σ of local background; 161 of
them confirmed hot by the dark-based mask.

| | detector persistence | sky persistence |
|---|---|---|
| hot (median) | **242σ** | **−1.0σ** |
| everything else (median) | **0.2σ** | **36.2σ** |

`persistence_plane.png` shows why: the hot pixels form a tight vertical column at sky
persistence ≈ 0 spanning 20–1300σ of detector persistence, and the stars sit entirely at
sky persistence > 10σ. Two orders of magnitude apart on both axes.

## The combination matters more than the measurement

| rule | average precision | P at 95% recall | recall at **zero** false positives |
|---|---|---|---|
| `detector − sky` | 0.962 | 92.2% | 21.7% |
| `detector` alone | 0.685 | 45.8% | 7.5% |
| `−sky` alone | 0.719 | 77.3% | 0.0% |
| **`log(detector) − log(sky)`** | **0.996** | **100%** | **96.3%** |
| `detector / sky` | 0.994 | 100% | 96.3% |

The difference — the obvious first guess — conflates a faint hot pixel with a bright star,
because it is an absolute gap where the question is a relative one. **A log ratio finds
96.3% of the dark-confirmed hot pixels with no false positives at all.**

## Two things that were wrong on the way, and what they cost

**The top disagreements were not hot pixels the dark missed; they were stellar wings.**
Checked against the detected stars, 13 of the top 14 sat 3.5–8.5 px from a real star, most
of them around the brightest star in the field. The fault was in the sampling: the dither
here is sub-pixel, and rounding the sky lookup to the nearest pixel puts the weakest sample
somewhere down a steep stellar flank.

**Interpolating helped much less than predicted.** Bilinear sky sampling was expected to fix
the wings. Measured, it lifts the inferior rules substantially (`detector − sky` from 0.897
to 0.962 average precision, `min` from 0.970 to 0.987) and does raise the wings' sky
persistence as intended — but the log ratio was already robust to the effect and did not
move (0.995 → 0.996, same 96.3% at zero false positives). Recorded because the prediction
was half wrong: the sampling fix matters only if the discriminant is a poor one.

## The limit to be honest about

**This needs dither larger than the PSF.** Here the largest offset from frame 0 is only
**5.6 px**, with consecutive frames under 1 px apart, and it still separates cleanly because
the PSF is compact. With no dither the two statistics are *identical by construction* and
the method cannot work at all — it would silently call every star a hot pixel. Any
implementation must measure the dither and decline when it is too small, rather than
returning confident nonsense.

Also worth stating: 161 of the dark's 296 hot pixels were bright enough in the lights to be
candidates at all. The other 135 are the mild ones, which contaminate less; a lower
candidate threshold reaches them at the cost of more pixels to test.

## What it costs

Measured on the same 7 × 46.8 Mpixel frames:

| step | per frame |
|---|---|
| `open_image` (warm cache) | 0.10 s |
| `uniform_filter(64)` background | **0.67 s** ← nearly all of it |
| threshold + `nonzero` | 0.11 s |
| sampling the sites, all frames | ~0.00 s |
| *`get_centroids_blur`, which stage 1 already does* | *3.89 s* |

**4.8 s for the whole search against 138 s for stage 1** — about 4%, and a fifth of the
centroid pass alone. Cheap enough that the tempting shortcut of only examining centroid
neighbourhoods is not needed for cost reasons.

## Can a hot pixel reach the aligner?

In principle yes; on this data no, and the reason is worth knowing. Only **1 of 388**
centroids sat on a hot pixel, and it ranked **106th** by flux — far outside the brightest 30
that `attempt_align` uses. Centroids rank by *integrated* flux and `min_area` is 4, so a
single hot pixel, even a saturated one at 16380 ADU, integrates to 280 against 11127 for the
brightest star.

That protection is circumstantial rather than structural. It fails for **hot clusters or hot
columns** (area grows, flux ranks high) and in **sparse fields** — the aligner takes
`min(len(c1), len(c2), 30)`, so rank 106 of 388 is safe while the same pixel in a 40-centroid
field is not. The failure mode is specific: hot pixels vote for shift **(0,0)** because they
do not move, so with a large dither enough of them could pull the alignment onto zero.

## As implemented

`mee2024/hotpixels.py`, used by `do_stack`:

* **darks supplied** → `dark_mask`, applied *before* centroid finding, so a hot pixel never
  becomes a centroid at all. Unchanged.
* **no darks** → `persistence_mask` after the first alignment (it needs the shifts), then
  the existing centroid lists are **filtered** rather than re-detected, and the alignment is
  redone. Re-detection would cost more than everything else here put together; a bad
  centroid only has to be dropped.
* Either way the mask is excluded from the stack — from the *count* as well as the sum, so
  hot pixels are removed rather than diluted.

Guards, each of which exists because it can bite: fewer than three frames, or dither under
`MIN_DITHER_PX`, is **declined with the reason logged** rather than guessed at; the saturated
blob mask is excluded from candidates; the candidate list is capped. `hot_pixel_dark_free`
turns the whole path off.

Still open: using it as a **cross-check when darks are present**. It disagreed with the dark
mask in the direction of finding real hot pixels the dark's own 20σ cut had missed — e.g.
row 3219 col 4195, 27σ detector persistence, −1.3σ sky persistence, 63 px from any star.
