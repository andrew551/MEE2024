# MEE2024 UI development roadmap

**Status:** proposal, for decision. Nothing here has been implemented.
**Date:** 2026-08-07, revised 2026-08-08 (v1.3.5)
**Question it answers:** the project has two interfaces. Do we maintain both, freeze one, or
port selectively — and in what order?

Companion documents: [`ROADMAP.md`](ROADMAP.md) (the whole picture — findings, immediate
fixes, features; **start there**), `UI_DESIGN.md` (what the app window is and why),
`ARCHITECTURE.md` (how the pipeline fits together).

**Revised 2026-08-08**, after user feedback and a review of eclipse-day working practice:

1. **The blob/corona group moves into Phase 1.** It was filed as a low-priority Advanced
   disclosure. That was wrong: `delete_saturated_blob`, `blob_saturation_level`,
   `blob_radius_extra`, `centroid_gap_blob`, `crop_circle`, `limit_radial_sun_radii` and
   `object_centre_moon` are all classic-only, and they are the reason eclipse fields are
   processed individually. Without them the app window cannot do eclipse-day *stage 1*,
   never mind stage 3.
2. **Per-field dark auto-matching is dropped.** Zenith batches share one exposure and
   eclipse ladders are processed individually, so one calibration set per batch is the
   right model. The calibration library (`ROADMAP.md` F1) covers the rest.
3. **Open-a-previous-run is promoted to a first-class feature**, not a follow-on to
   click-to-view. It serves single mode and old runs too, and the batch table's row click
   becomes just one entry point into it.

---

## 1. The recommendation in one paragraph

**Freeze the classic interface and port to the app window — but the classic interface
cannot be retired yet, and the reason is not sentiment.** The app window cannot run stage 3
at all, and cannot enable the astrometric corrections. It is also the *default* interface.
So today, a new user lands in an interface that cannot perform the experiment the project
exists to perform. That, not catalogue management or knob parity, is what should drive the
work. Port stage 3 and the corrections; keep the classic UI running untouched until they
land; do not invest in the classic UI meanwhile.

---

## 2. What the history already decided

Since the app window landed (`958e3b9`):

| | commits |
|---|---|
| `mee2024/ui/` (app window) | 19 |
| `mee2024/UI_handler.py` (classic) | **0** |

The classic interface is already frozen in practice. The only open question is whether that
is acknowledged and sequenced, or discovered by someone mid-campaign.

Size, for scale: classic UI 430 lines; app window 2,540 (frontend) + 733 (server) + 491
(runner). The app window is roughly nine times the code for a partly overlapping feature
set — most of that difference is progress reporting, result views and catalogue management,
which the classic UI simply does not attempt.

---

## 3. What each interface can actually do

Neither is a superset of the other. That is the core problem.

### Only in the classic interface

| Capability | Options involved |
|---|---|
| **Stage 3 — eclipse analysis, the entire tab** | `eclipse_limiting_mag`, `eclipse_method`, `object_centre_moon`, `limit_radial_sun_radii(_value)`, `remove_double_stars_eclipse`, `flag_display3` |
| **Astrometric corrections** | `enable_corrections`, `enable_corrections_ref`, `enable_gravitational_def`, `observation_time/lat/long/temp/pressure/humidity/height/wavelength` |
| **Distortion reference / fixed coefficients** | `distortion_reference_files`, `distortion_fixed_coefficients` |
| **Gravity sweep** | `gravity_sweep` |
| Crop circle | `crop_circle`, `crop_circle_thresh` |
| Centroid detail | `centroid_gaussian_thresh`, `min_area`, `sigma_subtract`, `background_subtraction_mode`, `remove_edgy_centroids` |
| Saturated-blob handling | `delete_saturated_blob`, `blob_saturation_level`, `blob_radius_extra`, `centroid_gap_blob` |
| Output detail | `float_fits`, `save_dark_flat`, `d` (stars drawn) |
| Match tolerance | `rough_match_threshhold` |

### Only in the app window

Batch folder processing · watch mode · catalogue download, removal and cleanup · processing
presets · date-from-header · live progress and event log · score cards · PSF panel ·
distortion-field and residual views · star labels over the stack · solve-vs-header pointing
check · `remove_missing_pm` · `distortion_field_plot` · diagnostics copy for bug reports ·
native file dialogs.

### Only from the command line

Plate-solver selection (`platesolver`, `pattern_db`, `platesolve_noise_px`) · pattern and
triangle database builds · catalogue `pack`/`install`/`merge`/`repair`/`check-remote` ·
`--set` for any option at all · JSONL event stream.

### Which interface can run which stage

| | Stage 1 stack | Stage 2 distortion | **Stage 3 eclipse** |
|---|---|---|---|
| App window | yes | yes | **no** |
| Classic | yes | yes | yes |
| CLI | yes | yes | yes |

`runner` only knows `['stack', 'distortion']`. There is no code path from the app window to
`eclipse_analysis`.

---

## 4. Two corrections to common assumptions

**"The classic UI can only use the online Gaia archive."** Not so. `options['catalogue']`
defaults to `'gaia'`, and that provider resolves to the offline archive whenever one is
installed. The classic UI never names a catalogue, so it inherits that and gets the offline
archive for free. What it lacks is *managing* catalogues — download, choose, remove.

There is one real consequence: the classic path never calls `prepare_catalogue`, so a fresh
install driven only from the classic UI silently runs against the **online** archive at
minutes per field, with no prompt. Cured permanently by one command:

```bash
mee2024 catalogue --fetch gaia_dr3_g13
```

So "port the offline catalogue to the classic UI" — the item that looked like the strongest
case for investing there — turns out to be a config fact plus a one-off CLI command. It is
the *weakest* case, not the strongest.

**"The app window is behind the classic one."** It is ahead in most respects and behind in
three that matter more than all the others combined: stage 3, the corrections, and fixed
distortion coefficients.

---

## 5. Roadmap

Ordered by consequence, not by effort.

### Phase 1 — make the default interface able to do the experiment

These three are **one deliverable**, not three items: "the app window can process eclipse
data". Shipping any two of them still leaves you in the classic interface on the day.

**1a. Stage 3 (eclipse analysis) in the app window.** The largest single item and the only
one on the critical path for the science. Needs: a distortion-zip input, the seven tab-3
options, and a results view. The existing event/score-card model already fits the outputs
(a text report plus plots), so this is mostly wiring rather than new invention.

**1c. The blob/corona group.** `delete_saturated_blob`, `blob_saturation_level`,
`blob_radius_extra`, `centroid_gap_blob`, plus `crop_circle`, `limit_radial_sun_radii` and
`object_centre_moon`. These are not expert conveniences — tuning them per field is *why*
eclipse fields are processed one at a time, and none of them exist in the app window.

**1b. The astrometric corrections — but auto-filled, not typed.** Porting the classic UI's
eight manual fields as eight manual fields would repeat the mistake that made them unused:
every real fit examined in the external review ran with corrections off, because nothing
ever fed them. Fill site and epoch from the FITS header (`OBSLAT`/`OBSLONG`, `DATE-OBS`)
with a visible override, default the corrections on when the header supplied them, and keep
pressure as a visible field. This is external-review item G2 and it lands naturally here.

*Acceptance for Phase 1:* a user who has never opened the classic interface can take frames
through stages 1–3 with corrections enabled, and get the same numbers as the CLI.

### Phase 2 — parity for the things that change results

Fixed distortion coefficients and reference files (`distortion_fixed_coefficients`,
`distortion_reference_files`) — required for cross-epoch and calibration-transfer work.
Then `rough_match_threshhold` and the centroid-detail group, behind an **Advanced**
disclosure rather than on the main panel.

### Phase 3 — retire

Delete `UI_handler.py`, drop the FreeSimpleGUI dependency, remove `mee2024 gui` and the
`default_interface` config option. Only after Phase 1 and 2, and only after asking the user
base — which is small enough to ask directly.

### Explicitly *not* ported

`flag_display`, `flag_display2`, `flag_display3` are meaningless in the app window, which
never opens a matplotlib window by design. `d` (how many stars to draw) is superseded by the
label slider. `float_fits` and `save_dark_flat` are arguably CLI-only concerns. Decide these
are dropped rather than leaving them on a backlog forever.

---

## 6. Policy, to be stated somewhere users and contributors will see it

1. **No new features in the classic interface.** Bug fixes only, and only if they block
   work that cannot move to the app window or CLI.
2. **The CLI is the contract.** Every UI action should reduce to an options dict plus a
   stage list — which `runner.build_options` already nearly does. Keeping that true makes
   each port mechanical and testable, and keeps the two interfaces from diverging in
   behaviour rather than merely in surface.
3. **Anything that changes numbers must be reachable from the CLI**, so that a result can
   be reproduced without a GUI at all.

---

## 7. Further ideas worth considering

**Generate the Advanced panel from `config.py`.** There are ~80 options and two hand-written
interfaces; that is why they diverged. Adding optional metadata to `DEFAULT_OPTIONS` (label,
help text, type, range, group) and rendering the advanced section from it would mean a new
option appears in the UI by virtue of existing. This is the only idea here that stops the
divergence recurring rather than merely repairing it once.

**Emit the exact command for a run.** Every run is an options dict; the app could show
"reproduce this run" as a CLI line or an options JSON. This directly addresses the problem
we hit this week — two machines producing different fits because one had a different
default — and it makes results self-describing when they are shared or published.

**Shareable settings profiles.** A campaign could ship one options file that every
collaborator loads, instead of a bulletin asking everyone to set a flag by hand. Same
motivation as above; more durable.

**Fix the checkboxes that do not persist.** `rm-double`, `rm-nopm` and `sensitive` are read
from the DOM and written to the config on every run, but never *restored* from it — the page
starts with the hardcoded HTML state every session. So unticking a box does not stick. This
happens to be convenient during the v1.3.5 transition (everyone converges to double-star
rejection on), but it is a real bug and should be fixed as part of any UI work.

**Rename `remove_double_tab2`.** It is named after a tab that will not exist. Worth renaming
during the port, with a config migration — the migration machinery is already there and
version-targeted.

**Test coverage is an argument in itself.** The app window's `Api` is plain methods and is
covered by ~80 headless tests. The classic UI is essentially untestable — one test asserts a
single tab-3 fix by reading source text. Every port moves a feature from untested to tested.

**Packaging.** Retiring the classic interface drops FreeSimpleGUI. Small, but real, and it
removes a Tk dependency from the frozen build.

**Know when it is safe to delete.** With a user base this size, ask. If a signal is wanted
instead, one log line at classic-UI start would answer it within a campaign.

---

## 8. Risks and open questions

**Does the eclipse campaign actually need stage 3 in the GUI?** The benchmarks and error
budget already run stage 3 from the CLI. If that is acceptable for the campaign, Phase 1a is
a convenience rather than a blocker, and the corrections (1b) become the urgent half. Worth
settling before the work is sequenced — it changes the priority by a lot.

**The port list is a control inventory, not a behavioural audit.** Some classic knobs may be
dead, or already correct at their defaults. A pass to confirm which are genuinely needed
should precede any estimate.

**`default_interface` currently points at the app window.** Until Phase 1 lands, that means
the default interface cannot do the experiment. Either accept it and document it, or point
new users at the CLI for stage 3 — but do not leave it undocumented.

**Browser mode has no native file dialogs.** The in-page picker is the fallback and works,
but any UI work should keep both paths exercised.

---

## 9. Appendix — how this inventory was produced

Control keys were read from `UI_handler.py` and `frontend.html`; the options each interface
writes were read from `interpret_UI_values{,2,3}` and `buildSpec`; stage reachability from
`runner._run_one`; churn from `git log 958e3b9..main` per path. Re-derive with:

```bash
git log --oneline 958e3b9..main -- mee2024/UI_handler.py | wc -l
git log --oneline 958e3b9..main -- mee2024/ui/ | wc -l
```
