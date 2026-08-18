# SER input, and choosing which frames to use

**Date:** 2026-08-18, v1.3.8
**Data this was built and measured against:** `I:\Joe Izen Spain Aug 2026` (SER, 61 MP) and
`I:\FRA500 Leon Aug 2026\Eclipse` (FITS, 26 MP)

---

## 1. Why SER at all

A 61-megapixel camera at 3.2 fps is **122 MB per frame and 388 MB/s sustained**. Asking a
filesystem for 180 separate FITS files at that rate is asking a lot, and at 315 ms the frames
are really a video. One observer therefore recorded to SER, and the only route into this
pipeline was a conversion through PIPP — which costs a duplicate of a 15 GB file, adds a
manual step, and (measured, §3) destroys the timestamps on the way.

SER is easier to read than TIFF: a 178-byte header, then frame after frame of raw pixels,
then optionally one 64-bit timestamp per frame. Reading frame N is a seek, which suits a
pipeline that already works one frame at a time and never wants 22 GB resident.

There was a second reason to do this properly. The existing non-FITS path is not fit for
science: `open_image` fell back to `cv2.imread` with no flags, which is `IMREAD_COLOR` and
**silently truncates 16-bit TIFF and PNG to 8 bits** — a 16-bit TIFF written with max 60000
came back with max 234. So "convert to TIFF with PIPP first" was worse advice than it looked.

## 2. How a SER frame is addressed

**`capture.ser#42`** — the container, then the frame index.

The pipeline's unit of work is a frame path, and rewriting that assumption across
`find_fields`, the aligner, the stacker, the logs and the batch summary would have been a far
larger change than reading the format. A reference carries its index instead, so everything
downstream keeps working on "a list of frames" without knowing the difference. `find_fields`
expands a `.ser` into one reference per frame; `open_image`, `read_bit_depth`,
`read_observation_date`, `read_pointing` and the calibration library all resolve them.

```bash
mee2024 stack "capture.ser" --frames 50-172
```

A bare `.ser` on the command line expands to all its frames.

## 3. Three things measured on real files that a naive reader gets wrong

### The `LittleEndian` header field is wrong

On every file examined it reads **0** — big-endian by a literal reading — while the pixels
are unambiguously little-endian. This is a known ambiguity in the format's history and
writers disagree, so the reader **measures** the order instead.

Two obvious measures fail, both worth recording because both look right:

* **Spatial smoothness** (neighbouring pixels agree). Fails on a frame with no large-scale
  structure, where the neighbour difference *is* the whole deviation whichever way it is read.
* **The same, normalised by the range.** Fails whenever the data spans less than one step of
  the high byte — then swapping is exactly multiplication by 256 plus a constant, and *every*
  ratio is invariant under it. This one passed on real data and failed on a synthetic flat
  field, which is how it was caught.

What works is **counting distinct byte values**. Real 16-bit image data occupies a modest part
of the container, so its high byte takes few values while the low byte runs through 0–255
with the noise. Whichever byte of the pair is more repetitive is the high one. On Joe's file:
43 distinct values against 256.

### The timestamps are often missing, in two different ways

| file | header `DateTime_UTC` | trailer |
|---|---|---|
| `sun_1` (original) | 2026-08-12 18:28:45.956 | 180 slots, **all zeros** |
| `sun_2` | present | **real per-frame timestamps** |
| `left_calibration` | present | **real per-frame timestamps** |
| `PIPP` (trimmed by SER Player) | **0** | **absent entirely** |

So a trailer of the right *size* is not enough — the space can exist unwritten, which is
what `sun_1` shows for all 180 slots. The reader checks the values.

**What separates them is how the capture ended.** It is not a setting: all three sidecars say
`Timestamp Frames=Off`, including the two whose trailers are full. `sun_1` is the one capture
that terminated abnormally — its last eight frames are blank — and the trailer is written when
a capture *completes*, so an interrupted capture loses it. `sun_2` and `left_calibration` both
end on real data and both have full trailers.

The practical lesson is therefore about **stopping a capture cleanly**, not about a checkbox:
a capture that dies takes its per-frame timing with it, and per-frame UTC stamps are strictly
better than FITS gives you (one `DATE-OBS` per file).

Timing therefore falls back through: per-frame trailer → header UTC → sidecar `StartCapture`,
interpolated across the capture.

### The header strings carry leftover memory

`Observer`, `Instrument` and `Telescope` are not zero-filled — there is junk after the
terminator. A reader must cut at the first NUL, not strip trailing spaces.

## 4. The sidecar carries what a FITS header would

`<name>.CameraSettings.txt`, written alongside by the capture software, supplies everything
the pipeline reads and everything the calibration library keys on:

| sidecar | maps to |
|---|---|
| `Exposure=315.0000ms` | `EXPTIME` 0.315 (units understood) |
| `Analogue Gain=125` | `GAIN` |
| `Offset=200`, `Binning=1` | `OFFSET`, `XBINNING` |
| `[Zeus 455M PRO (IMX455)]` | `INSTRUME` |
| `CameraSerialNumber` | `CAMID` |
| `Temperature`, `Target Temperature` | `CCD-TEMP`, `SET-TEMP` |
| `StartCapture` / `MidCapture` (UTC) | `DATE-OBS` / `DATE-AVG` |
| `ASI Mount=RA=…,Dec=…` | `OBJCTRA`, `OBJCTDEC` |

`MidCapture` is **better than FITS `DATE-OBS`**: it is the mid-exposure epoch, which is what
the astrometric corrections want.

And it carries something FITS has no equivalent for: `Subtract Dark`, `Apply Flat`,
`Background Subtraction`, `Banding Suppression`. These are **proof the pixels are
unmodified** — if any were on, the frames are already calibrated and the pipeline's
assumptions break silently. Read into `_modified`, which is empty for all of Joe's captures.

Missing against a FITS header: `FOCALLEN`/`XPIXSZ` (so no plate-scale prior), site
coordinates, `EGAIN`, `RDNOISE`. The optics constants are per-rig configuration rather than
per-frame data, so they arguably belong in settings anyway.

---

## 5. Choosing which frames to use

A capture rarely starts and stops on the science. The trim used to be a manual pass through a
separate program that wrote a second copy of the file. It is now a **run parameter**:

```bash
mee2024 stack "capture.ser" --frames 50-172
```

Zero extra storage, nothing rewritten, the container's metadata preserved by construction,
and the trim recorded in the results so it is reproducible rather than an untracked manual
step. Re-trimming is free.

### The suggestion

Every run measures its frames — a strip through each, **one second for 180 frames of a 22 GB
container** — and says what looks usable:

```
frames 50-171 of 180 look usable; 50 dropped at the start, 8 at the end;
(8 blank, 39 saturated above the 4.6% floor, 11 still changing).
Set frame_range to 50-171 to use only those; nothing has been dropped automatically.
```

Nothing is dropped without being asked. `--no-scan` turns it off.

### Why the obvious rules fail

Both ends are treated identically, because **the Sun can be at either**: a sequence may open
on the uneclipsed Sun *or* run through third contact into full saturation.

* **"Drop all-black and all-white frames."** Measured against a real manual trim: it got the
  blank tail exactly right and the saturated head completely wrong. The frames either side of
  second contact are not *all* white — frame 0 was 38% saturated, and frames 22–48 decay
  smoothly through every value in between.
* **"Drop frames until the picture stops changing."** Fails silently, and this is the one to
  remember: **while a frame is saturated its median is pinned at full scale and the
  frame-to-frame difference is exactly zero**. A forward search for "settled" stops on the
  *first* frame and keeps the entire saturated run. Saturation is indistinguishable from
  perfect stability by that measure.

So a usable frame must be settled **and** not saturated far above the sequence's own floor —
a totality exposure legitimately clips a few percent (the inner corona) in every frame, so the
test is relative to that floor rather than to zero.

The change threshold **calibrates itself**. A fixed per-frame value cannot work across
exposures: at 315 ms consecutive frames are a third of a second apart and a stable stretch
changes well under 1% a frame; at 2 s they are 2.4 s apart and the same sky changes 1.3–2.6%.
A fixed 1% cut rejected *every frame* of a real 2 s calibration sequence.

### Measured against the manual trims

| sequence | shape | manual | automatic |
|---|---|---|---|
| Joe `sun_1`, 180 frames | saturated head, blank tail | 49–171 | **50–171** |
| Leon `CAL_piLeo` 2.0 s, 15 frames | **Sun returns at the end** | first 7–8 | **0–6** |
| Leon `CAL_piLeo` 1.0 s, 11 frames | Sun returns at the end | in-totality | **0–5** |

Within one frame of the human answer on the case that has one, and correct on both
Sun-at-the-end cases.

---

## 6. Does a frame's brightness match the exposure it claims?

Capture software that changes exposure mid-sequence can write the **new** exposure into the
header of a frame that still holds the **previous** exposure's pixels — the camera had not
applied the change when the frame was read out. Nothing downstream can detect it: the file is
well-formed, the header is self-consistent, and the frame simply carries half the signal it
claims. It would then be stacked into the wrong exposure tier and matched to the wrong master
dark.

Six frames of the Leon eclipse ladder are exactly this. Every run now checks and **reports**:

```
SCI_ladder_00006.fits says EXPTIME 1.2 s, but it is the first frame after a change from
0.6 s and its level (2750) looks like the 0.6 s frame before it (2768) rather than like
the other 1.2 s frames (4620). Capture software can write the new exposure into the header
of a frame that still holds the previous one. Check it before this frame is stacked or
dark-matched.
```

**It reports rather than corrects, deliberately.** Eclipse data is too scarce to drop frames
over a label, and an automatic relabel would be a silent edit of somebody's science. It also
serves a second purpose: a run on data nobody has inspected by hand says so, in the log,
before anyone trusts the output.

Two tests, because one is not enough:

* **The transition test** is the targeted one — at every stated-exposure change, does the new
  frame look like the frame *before* it rather than like its own exposure? It is local (the
  two frames are seconds apart), so a sky brightening through the sequence cannot hide the
  fault.
* **The group test** is the backstop, for a frame that is wrong without being at a transition
  — cloud, or a one-off.

A whole-group comparison alone is not sufficient and this was measured: on the pi Leo ladder
the sky rose enough *within a single exposure group* to inflate its spread until a mislabelled
member sat comfortably inside it. The transition test catches all six; the group test alone
caught three.

Running it over the Leon eclipse folders found **two more than a by-hand analysis had** —
the lag affects the first *two* frames after a change, not just the first. A 30-frame
single-exposure field stays silent.

`--no-exposure-check` turns it off.

---

## 7. What is not done

* **No graphical frame selector.** The scan reports a suggestion in the log and `--frames`
  applies one; picking frames on a plot of the level curve is not built. The decay curve is
  more useful than the images for this, so that is what such a view should show.
* **Writing SER is not supported**, and should not be — the point is to stop making copies.
* **The FITS-folder mess is not automated.** The Leon eclipse folders have frames misfiled
  across timestamp folders by the same buffer lag, and repairing that is specific to one
  capture script and one machine. Reading by header rather than by folder sidesteps it: the
  frames sort correctly by `DATE-OBS` regardless of which folder they landed in.
