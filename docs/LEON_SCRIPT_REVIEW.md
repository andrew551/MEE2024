# Leon capture scripts — review of the gain 101 → 0 change

**Reviewed:** `Leon_gain_101.zip` (six scripts) against `Leon_gain_zero.zip` (six scripts),
2026-08-10, against the findings in [`PORTLAND_2026-07-29.md`](PORTLAND_2026-07-29.md).
**Verdict:** the physics is right, the two-regime split is right, and it is what was asked
for. There are four stale comments, one of which is dangerous, and one free improvement the
change enables and has not been taken.

---

## 1. Does the gain change make sense? Yes, and it checks out numerically

Every executable `SET GAIN` line in the new set is consistent:

| script | gain | why |
|---|---|---|
| `leon_eclipse_v1.16` | 0 | bright-sky corona and π Leo — wants well depth |
| `leon_flats_v1.3` | 0 | half-scale fill, applied to both regimes |
| `leon_setup_check_v4` | 0 | throwaway frames, exercises the control path |
| `leon_darks_v1.4` | 0 **and** 101 | one block per regime |
| `leon_zenith_v1.11` | 101 | read-noise-limited night frames |
| `leon_horizon_v1.14` | 101 | read-noise-limited night frames |

### The three claims in the v1.16 header, checked independently

> *full-well 2.5×, π Leo saturation 0.35 → ~0.87 s, coronal saturation radius shrinks
> ~1.36×, read noise 1.5 → 3.3 e⁻ invisible under eclipse sky*

**The saturation arithmetic is self-consistent.** π Leo saturating a 20 000 e⁻ well in
0.35 s and a 50 000 e⁻ well in 0.87 s both give ~57 000 e⁻/s in the peak pixel. Working that
back through a V ≈ 3.7 effective magnitude, 7″ seeing and 2.13″/px gives a system
throughput of ~7×10⁵ e⁻/s for a V = 3.7 star — about what a 90 mm aperture, QE 0.8 and a red
filter should deliver. The numbers hang together.

**The 1.36× radius shrink assumes B ∝ r⁻³** (2.5^⅓ = 1.36). That is a fair mid-range slope
for the inner K-corona; a steeper r⁻⁷ near the limb would give only 1.14×. The claim is
therefore slightly optimistic close in and conservative further out. Not material.

**"Read noise is invisible under eclipse sky" is correct, comprehensively.** Taking the
corona at ~10⁻⁷ of mean disc surface brightness at 2 R☉ and ~10⁻⁸·⁵ at 4 R☉, and the
throughput above:

| position | exposure | sky (e⁻/px) | shot noise | read 1.5 | read 3.3 | SNR penalty |
|---|---|---|---|---|---|---|
| 2 R☉ | 0.1 s | 16 700 | 129 e⁻ | — | — | 0.03% |
| 4 R☉ | 1.2 s | 6 300 | 79 e⁻ | — | — | 0.09% |
| ~8 R☉ (frame corner) | 0.1 s | 83 | 9.1 e⁻ | 9.2 total | 9.7 total | **5%** |

So the worst case in the whole science ladder — the faintest corner of the shortest tier —
loses 5%, and everywhere the astrometry actually happens (0.6 s and 1.2 s, where the faint
stars are) the penalty is under 0.1%. The header's "every SNR moves by a few %" is right.

**And this is the check that matters most:** the faint field stars whose positions are the
entire measurement sit on a background of thousands of electrons. Those frames are
sky-shot-noise limited by two orders of magnitude. Read noise is not a term in their error
budget at either gain, and the 2.5× of extra headroom against the corona is free.

### The night-side reversion is the better half of the decision

`leon_zenith_v1.11` and `leon_horizon_v1.14` reverted from the v1.10/v1.12 plan (gain 0 with
the exposure doubled) back to gain 101 at the original 4 s and 6 s. That is right, and the
stated reason is the right reason. At the horizon script's own figure of ~3 e⁻/px of sky in
6 s:

- gain 101: √(3 + 1.5²) = **2.29 e⁻** per pixel
- gain 0: √(3 + 3.3²) = **3.73 e⁻** per pixel

Gain 0 is 1.63× noisier there, so recovering it needs 2.65× the exposure, not 2× — the
proposed doubling would not even have broken even, and it would have bought trailing on the
calibration centroids, which are the most precision-critical frames in the campaign.
Reverting was correct.

**Assigning each regime the gain its noise environment wants is exactly the right principle,
and it is stated in the scripts as doctrine rather than left implicit. Good.**

### Flat-darks retired — correct, and one thing gets better

This matches ROADMAP §1.8 (subtracting the flat-dark moves the normalised flat by 87 ppm,
1.2% of the PRNU). The framing in `leon_flats_v1.3` is exactly right and worth keeping:

> *THE HALF-SCALE FILL IS THEREFORE LOAD-BEARING: the mid-range histogram target below is a
> REQUIREMENT, not advice. Fractions are in ADU, so this is gain-independent.*

That is the correct dependency. The residual is the offset pedestal as a fraction of the
fill, so half-scale is what makes the retirement valid, and stating it as a requirement
rather than a preference is the difference between a justified simplification and a silent
one. A ~500 ADU pedestal on a 32 768 ADU fill leaves ~0.6% of vignetting under-correction at
a 30%-vignetted corner — smooth, multiplicative, and worth well under a mas on a centroid,
because the gradient across a 2.5 px PSF is a part in 10⁵.

**A bonus nobody claimed:** at half scale the gain-0 flat holds ~24 900 e⁻/px against
~9 800 at gain 101, so 50 frames give 0.09% per-pixel noise instead of 0.14%. The flat is
better at gain 0, not merely acceptable.

Applying it to the gain-101 night frames is sound: PRNU is a quantum-efficiency property and
vignetting is optical, and the flat is normalised, so only its shape matters. The stated
first-order assumption is the right one.

### Darks — this is what was asked for

`leon_darks_v1.4` takes six gain-0 tiers (0.1/0.3/0.6/1.0/1.2/2.0 s) and two gain-101 tiers
(4 s, 6 s), 50 frames each, `TARGETNAME` embedding the gain so the folders are
self-labelling. That answers the Portland §7 request — a dark at every ladder exposure,
because defect amplitude is `pedestal + rate × t` and tiers cannot be extrapolated
(ROADMAP §1.8) — and it answers it per gain, which the two-regime split makes necessary.

**It also corrects ROADMAP Q4, which is wrong.** Q4 says setting `FRAMETYP` at capture time
"costs nothing today". `leon_darks_v1.4` states that the SharpCap Sequencer has **no
frame-type command**, so a scripted capture cannot set it at all; the workable answer is the
one the script adopts — key calibration matching on `GAIN` + `EXPTIME` + `OBJECT` (which
carries `TARGETNAME`) and never on `IMAGETYP`. Q4 should be reworded from "fix it at
capture" to "it cannot be fixed at capture; name the folders so `OBJECT` carries the type".

---

## 2. Four stale references, one of them dangerous

None is an executable line — every `SET GAIN` is right. All four are text a tired operator
reads at night.

**(a) `leon_eclipse_v1.16.scs:207` — the eclipse-night prompt names the wrong gain.**
This is the last instruction of eclipse day:

> *(2) DARKS - cap on, cooler +10C, **gain 101**, offset 50, one tier per exposure used
> tonight: 0.1 / 0.3 / 0.6 / 1.0 / 1.2 / 2.0 s, 50 frames each*

Those six tiers were just captured at **gain 0**. `leon_darks_v1.4` gets it right, so an
operator who runs the script is fine — but this prompt tells them the settings explicitly,
which invites doing it by hand, and a dark library at the wrong gain is worthless. It cannot
be re-taken: focus, rotation and filter must not change, and by the next night they will
have. **Fix before rehearsal.**

**(b) `leon_eclipse_v1.16.scs:132` — the centring instruction now achieves the opposite of
its stated purpose.**

> *Sun centring and slew test done AT GAIN 101 so the live view does not jump when the
> script asserts gain at start-up*

The script asserts gain **0** at line 146. Centring at 101 therefore guarantees the jump it
is written to avoid: the crescent drops ~2.5× in brightness the moment the script starts —
at the armed prompt, where controls lock and exposure is deliberately left manual, and where
the crescent is described as "continuous confirmation of pointing and rotation while you
wait". Should read gain **0**. This is the one with real operational teeth.

**(c) `leon_eclipse_v1.16.scs:7`** — "Dark library must be gain 0 (`leon_darks_v1.2`)". The
file is `leon_darks_v1.4`; v1.2 is not in the set.

**(d) `leon_flats_v1.3.scs:32` and `leon_horizon_v1.14.scs:109`** — both still say "gain
101". The flats line ("SETTINGS: gain 101, offset 50, +10 C — identical to all light
frames") is doubly stale: the flats are gain 0, and there is no longer a single light-frame
gain to be identical to.

---

## 3. The free improvement the change enables, and has not been taken

The science ladder is unchanged at 0.1 / 0.3 / 0.6 / 1.2 s. For the three longer tiers that
is right — they already run near 100% duty, and lengthening them would only trade frames for
trailing.

**The 0.1 s tier is different, because it is the only one throttled by the readout floor.**
The script's own numbers: 52 MB per frame over 183 MB/s gives a 0.285 s cadence floor, so
the 0.1 s tier runs at 35% duty and spends 13.7 s of a 104 s totality delivering 4.8 s of
integration.

With 2.5× the well depth, **0.1 s → 0.25 s reproduces the gain-101 frame exactly**: 2.5×
the electrons into a 2.5× deeper well is the same saturation radius, the same star-to-corona
contrast, the same picture. What changes is that duty goes 35% → 88%, so the same 13.7 s of
wall clock delivers **12 s of integration instead of 4.8** — 2.5× the photons on the stars
those frames exist to show, for no wall-clock cost and no change in what saturates.

Trailing is not an objection: 0.25 s at even Portland's uncorrected drift is 0.24″ = 0.11 px.

The counter-argument is the script's own doctrine — *"REHEARSE THIS FILE, unedited"*, *"do
not edit the script on the day"* — and it is a good doctrine. This is a two-character change
in two places, but it changes what a tier means, so it needs a rehearsal, not just an edit.
**Worth doing if there is time to rehearse; not worth doing if there is not.** If the
measured full-well ratio is not exactly 2.5×, scale the exposure by whatever it is.

---

## 4. Two things from Portland that these scripts assume away

**(a) The focal-length freeze gate has already run, and the answer is not the coded one.**

`leon_eclipse_v1.16` carries a freeze gate: measure *f* from the first zenith frames, then
pick the matching offset pair —

- f = 363.5 mm (expected, from Carrell 2024): `OFFSET 34.3 29.3` ← coded
- f = 350.0 mm (contingency): `OFFSET 36.5 31.2`

The Portland zenith run **is** `leon_zenith_v1.11`'s grid, on an ASI2600MM, and it measured
**350.96 mm ± 5 ppm** — 1 mm from the contingency value and 12.5 mm from the coded one. If
that is the Leon rig, the coded pair is the wrong one of the two. (Only you can confirm
whether Toby's Portland train is the Leon train.)

The *pointing* consequence is mild: with the 363.5 offsets on a 350.96 mm train the Sun
lands 0.54° inside the bottom edge instead of 0.50° — 2.7′ out, well inside the design
margin. **The image-quality consequence is not mild.** Through the reducer relation
B = f_r(1 − m), 350.96 vs 363.5 mm is **≈ 5 mm of back focus**, against a typical ±0.5 mm
tolerance — and Carrell's configuration achieved 0.089″ rms where Portland achieves
0.195–0.240″ with corners 1.38× the centre. That is
[`PORTLAND_2026-07-29.md`](PORTLAND_2026-07-29.md) §5.1 arriving at the same place from the
star images alone. Two independent routes to "about 5 mm of spacing".

**(b) The exposure-versus-trailing arithmetic assumes a polar alignment the rig did not
have.**

`leon_horizon_v1.14` rejects an 8 s exposure because it would buy "~0.55 px of
refraction-drift smear plus ~0.5 px PE trailing", and prefers 6 s at "~0.28 px". That
reasoning is sound — but at Portland's measured drift of 0.97″/s the **existing** 6 s
exposure trails **2.6 px**, and the zenith script's 4 s trails **1.8 px**. Ten times the
terms being weighed, on the frames the whole method rests on.

The same applies to the horizon script's stated purpose (2), "refraction-tracking drift rate
(~0.1 arcsec/s vertical creep)". A 3.7° polar error injects 0.97″/s on top of it — a 10:1
contamination of the very quantity the field exists to measure.

Neither is an argument against anything in the scripts. Both say the same thing: **the
polar alignment is a precondition for the exposures these scripts already choose, and it was
not met at Portland.** `leon_zenith_v1.11` lists "polar aligned" under PRECONDITIONS, where
it is assumed rather than verified. A SharpCap polar-alignment routine takes twenty minutes
and takes 3.7° to under an arcminute.

*Footnote:* fixing it also removes the 55–68 px of accidental dither the drift was
supplying, which is what let the dark-free hot-pixel search work at Portland. With a real
two-gain dark library that no longer matters — but if a future zenith set is ever taken
without darks, a deliberate dither will have to replace what the bad alignment was doing by
accident.

---

## 5. Summary

| | |
|---|---|
| Gain 0 for the eclipse frames | **Right.** Read noise is invisible under that sky; the headroom is free. |
| Gain 101 for the night frames | **Right,** and the reversion from gain 0 + doubled exposure is the better call. |
| Exposures unchanged | **Right for 0.3/0.6/1.2 s.** The 0.1 s tier could take 2.5× more light for free — §3. |
| Flat-darks retired | **Right,** and the "half-scale fill is load-bearing" framing is the correct dependency. |
| Two-gain dark library | **Right,** and it is what Portland §7 asked for. |
| Four stale gain references | **Fix**, especially the eclipse-night darks prompt and the centring gain. |
| Focal length | **Check** — Portland says 350.96 mm, the script is coded for 363.5. |
| Polar alignment | **Unresolved precondition,** and every exposure choice in these scripts depends on it. |
