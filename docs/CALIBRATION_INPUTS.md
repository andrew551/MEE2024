# Frozen calibration inputs — where they are and how to check them

The Leon 2026 zenith solutions and the CAL_piLeo frame list are **inputs** to the step-2
and step-3 chain, not outputs of it. Every downstream number — the CAL_piLeo plate scale,
and through it the deflection constant — is pinned to these exact values.

**They live at `F:\MEE_output\leon2026\calibration\`.**

They were in this repository as `calibration/` from 2026-08-28 until 2026-09-17, when
Douglas moved them out: *"I am uncomfortable with output data like ... calibration ending
up in the repo. We already deleted or moved quite a bit of test data in a previous clean up
of the repo."* The reasoning stayed here; the data went to the output tree with everything
else.

What that costs, stated plainly: the output tree has no version history, so a re-run or an
edit there leaves no record, and these are values that must not move silently. **The
manifest below is the replacement guarantee** — it is metadata, not data, so it belongs
here, and it lets anyone confirm byte for byte that the files still hold what every
published number was computed from.

## Checking them

A copy of the manifest sits beside the data as `SHA256SUMS.tsv`, so a check needs nothing
from this repository:

```bash
cd "F:/MEE_output/leon2026/calibration" && awk -F'	' '{print $2"  "$1}' SHA256SUMS.tsv | sha256sum -c
```

All eighteen lines should say `OK`. The table below is the same data, kept here so the
expected values are under version control even if the copy on `F:` is altered.

## Manifest (SHA-256, taken at the move, 2026-09-17)

| file | bytes | sha256 |
|---|---|---|
| `README.md` | 6712 | `5e6024ad7bee83ae8779439e953b9966a24aa7694c29c5232a0b4464bce45f17` |
| `cal_pileo_frames.txt` | 1316 | `636a7a3503980feac12fa1a7390a75ce6e1c9221e728a31804fc2227c0124f69` |
| `zenith_cubic/08-11_Z1_base.txt` | 5008 | `c93f6dd099528ed7c954a32179de98ac7a391401794ad185cbefed47d7cb8c0f` |
| `zenith_cubic/08-11_Z2_mid_left.txt` | 5266 | `823106dea38b5f121766ee5424459d942309cbcb8732ccd949209d72f2ddbde5` |
| `zenith_cubic/08-11_Z3_top_left.txt` | 5263 | `646dc4ee0d351805637b9db275e3a129a63ec89a84d5781d010b9ebbb25d5b75` |
| `zenith_cubic/08-11_Z4_top_right.txt` | 5319 | `4403bbe83a3bc58311d9bc0b517591eca841b8855c7f76cb119cef0ea2cb09bb` |
| `zenith_cubic/08-11_Z5_mid_right.txt` | 5322 | `ce8249ac423e6bf8dbdc2561a108596e77545a49ec79fd53fb2337c316416672` |
| `zenith_cubic/08-11_Z6_bottom_right.txt` | 5501 | `31c558a74362abfc54c1edfa2af4395a57a1c482717155a708c3a392c5b6c025` |
| `zenith_cubic/08-12_Z1_base.txt` | 5013 | `8cb05744f556190211b0584ed47aa26897eed17fd4a14a8650c4f49c369ddae5` |
| `zenith_cubic/08-12_Z2_mid_left.txt` | 5255 | `8f1c3327a4ee85751da094889d332257119a5d1a82bc122be2c5b97a1aa5127c` |
| `zenith_cubic/08-12_Z3_top_left.txt` | 5257 | `2e488f6097431047fabf28272b0face60048ff4141e47a128db7140a12878bf2` |
| `zenith_cubic/08-12_Z4_top_right.txt` | 5317 | `8a5f7ad1b328a3a93080604f606587650aab20819ff5f4d367f9a66a85baa90c` |
| `zenith_cubic/08-12_Z5_mid_right.txt` | 5321 | `4ce46e1b00e0a72f816e2c662c0ef1fc58d49897e764e9c70074ae9a8546b528` |
| `zenith_cubic/08-12_Z6_bottom_right.txt` | 5502 | `6fa8eea3fc098d63110ae28d310db21f2205527c027b622a4355991a753da3ae` |
| `zenith_cubic/README.md` | 5242 | `1484095da32ea22225f107a6f39e7f2f50be51d8bbae12f7883d440a28d1182a` |
| `zenith_cubic/reference_files.txt` | 1035 | `eed8cdc980900e02ab039ff094df827dfe7efd5a54934c26d6a98f2f35e5f3ce` |
| `zenith_cubic/reference_files_relative.txt` | 267 | `737e42ed816e8795294688508b9199edfa7c8c451e96a0e4782c7430fdebaf59` |
| `zenith_cubic/summary.json` | 482 | `02a80962399059b4f67c22bb4593dcb2d5e8aa2110089ff9101dc522f69aabe0` |

## Which files are canonical

**The chain uses the six `08-12` files only.** Carry forward **d(3000) = 3.0297 ″**.

The twelve-file mean of 3.1048 ″ is superseded and must not be used. The telescope was
dismounted, transported by car and remounted between night 1 and eclipse day, and the change
is measurable rather than assumed: the m=1 tilt dipole doubled (0.510 ″ at PA −67° to
0.996 ″ at PA −101°), radial FWHM growth moved 1.216 → 1.335 at ~6σ, and the plate scale
stepped +197 ppm. Night 2 shares the eclipse day's mechanical state; night 1 is a different
optic. Full argument in `docs/REFRACTION_2026.md` §16.2–16.3.

The `08-11` six are kept as the **pre-transport control** — they are what made the transport
change detectable — and their own d(3000) is 3.1799 ″. They are not part of the chain.
Night-to-night gap 4.84 %.

The field charts drawn from both nights are in
`F:\MEE_output\tan_gauge_examples\field_charts_instruments\Leon FRA500\`, and they add a
caution to the tilt story: the distortion field's own left−right dipole is **+1.012 ″ on
both nights**, unchanged by the transport. The PSF tilt dipole above and the distortion
dipole are different measurements; do not read one as corroborating the other.

## The CAL_piLeo frame list

`cal_pileo_frames.txt` holds the sixteen frames of the canonical calibration: all six of
`1.0s8_29_19`, the first three of `1.0s8_29_51` (pre-C3), and all seven of `2.0s8_29_27`.
Reduced against the six `08-12` references it gives **2.2054043 ″/px**, 74 stars,
rms 0.5318 ″, HC0 21.6 ppm (quote HC3-class ~25 ppm), at `observation_time 18:29:35`.

**The order of the sixteen lines is part of the definition, not presentation.** The stacker
aligns every frame to the *first* one in the list, so a different first frame changes every
shift and therefore the stack. Measured 2026-08-29: the same sixteen frames in a different
order gave **112 centroids instead of 122** and **rms 0.5698 ″ instead of 0.5318 ″**,
moving the plate scale 0.3 ppm. Do not sort this file.

## Traps

* The folder organisation on `G:` is the truth about exposure. **EXPTIME headers lie on the
  first frame after a SET EXPOSURE change** — verify by sky level when in doubt.
* `reference_files.txt` holds absolute paths from the machine that produced it. Regenerate
  it after copying.
* `source_data` inside each of the twelve solutions names
  `D:\MEE_output\v1.4.0-dev_inpipe\...`, which **cannot** be repointed: those stage-1 trees
  are not in the output tree on any drive, only on `I:\MEE Project files\` under identical
  filenames. Provenance is recoverable; only the recorded path rotted.

## Other copies

Five copies of the zenith set exist and all are byte-identical (re-verified 2026-09-17 for
the two on `F:`):

| copy | role |
|---|---|
| `F:\MEE_output\leon2026\calibration\zenith_cubic\` | **the canonical one — cite this** |
| `F:\MEE_output\leon2026\HANDOFF_zenith_cubic\inpipeline_windowed\` | the working copy the campaign ran against; the 12 solutions are byte-identical to the above |
| `H:\Claude Code\HANDOFF_zenith_cubic\` | the transfer copy sent to Andrew (read-only) |
| `I:\MEE Project files\HANDOFF_zenith_cubic\` | returned from Andrew's machine |
| (this repository, `calibration/`) | removed 2026-09-17 — replaced by the manifest above |

`I:\MEE Project files\` is **not** a stray duplicate: it is the generation archive and the
only place the `v1.4.0-dev_inpipe` source trees survive.

The reduction these feed is `docs/CAL_PILEO_STEP2.md`. Provenance and the settings each set
was produced with are in `README.md` beside the data.
