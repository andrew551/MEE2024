# UI redesign: strategy

Written 2026-07-30. Goal: a professional, app-like interface with in-window progress and
visualisations, approachable for non-experts, with a **live mode** giving instant feedback
during data acquisition — while preserving the classic FreeSimpleGUI interface as a legacy
mode. Target platforms: Windows and macOS first-class, Linux best-effort.

## 1. The architectural insight: the UI problem is already half solved

Milestone A decoupled the pipeline from the GUI: no pipeline module imports
FreeSimpleGUI, every stage runs headlessly, and progress flows through the
`ProgressReporter` seam. **Any new UI is therefore a pure consumer.** The single missing
piece is a richer contract than "progress bar percent":

### P0 — the event bus (do this first, it is small)

Extend `mee2024/progress.py` into an event protocol. The pipeline emits typed events; any
frontend (CLI, legacy GUI, new UI, a test) subscribes:

```python
{"type": "stage_started",  "stage": "stack", "n_items": 20}
{"type": "progress",       "stage": "stack", "done": 7, "of": 20}
{"type": "frame_aligned",  "frame": 7, "shift": [1.2, -3.4], "rms": 0.06, "n_matched": 28}
{"type": "solve_candidate","triangles": 74, "accepted": true, "n_matched": 98, "thresh": 12}
{"type": "metrics",        "rms_mas": 109.6, "nn_corr": 0.166, "n_stars": 434,
                           "date_guess": "2023-10-28", "platescale": 1.8716}
{"type": "image",          "name": "stack_preview", "png": "<bytes>"}
{"type": "log",            "level": "info", "text": "..."}
```

This is worth doing even if no new UI is ever built: it makes runs machine-readable
(`--events-jsonl out.jsonl`), which the auto-calibration score (milestone B) and the
centroid benchmark (milestone E) want anyway. The existing `plt.show()` sites become
`image` events; matplotlib stays for file outputs (`Agg`).

## 2. Framework recommendation: **pywebview shell + local web frontend**

| option | look ceiling | cross-platform | packaging | risk |
|---|---|---|---|---|
| **pywebview + HTML/JS** | unlimited (canvas/CSS/SVG animation) | Win: WebView2 (ships with Win 10/11); macOS: WKWebView built-in; Linux: webkit2gtk | PyInstaller-friendly, tiny (~1 MB shell) | JS build discipline |
| PySide6 / QML | high, native | excellent | heavy (~150 MB) | steep API, licence care |
| Flet (Flutter) | high | good | moderate | young ecosystem, less control |
| Dear PyGui | medium (immediate-mode) | good | easy | utilitarian aesthetic |
| Textual | terminal only | — | — | not the ask |

Reasons for pywebview: the "fancy" ambitions (animations, dashboards, live gauges) are
exactly what browser tech is best at; the Python side stays a thin event bridge (the
`js_api` bridge or a localhost websocket); a `--browser` fallback serves the same UI in a
normal browser when no webview is available (headless Linux, remote use). The frontend
should be a **single self-contained HTML file** (inline JS/CSS, no build step, no CDN) —
keeps packaging trivial and works offline, consistent with the offline-first direction of
the project.

Legacy: `mee2024 gui` keeps launching the FreeSimpleGUI interface until the new one
reaches parity; the new one lives at `mee2024 ui` meanwhile, then they swap
(`mee2024 gui --legacy` preserved indefinitely — it is ~430 lines and costs nothing).

## 3. UX: three modes, one window

**Simple** (default): a drop zone for light frames, optional dark/flat pickers, one
**Run** button, and an **Auto** toggle (milestone B's auto-calibration chooses centroid
preset and distortion order via `nn_corr`). Results appear as large score cards —
platesolve ✓/✗ with pointing on a small sky map, RMS (mas), star count, `nn_corr` grade,
date-guess vs header date. Every number gets a plain-language caption ("113 mas ≈ the
width of a human hair at 200 m").

**Advanced**: the full option set, grouped as today's three tabs, with the config
file round-tripping unchanged (same `options` dict, same `MEE_config.txt`).

**Live**: pick a watch folder; each arriving frame is centroided and platesolved
(~5–9 s measured — comfortably within a typical exposure+download cadence), and the
dashboard updates: pointing crosshair on a sky chart, rolling per-frame FWHM/star-count
sparklines, and the metric gauges filling in as enough frames accumulate for a stack →
solve → quick distortion fit. This gives the observer "am I pointed right, is the focus
right, is tonight's data good enough" *while the telescope is still on the field* — the
quality metrics (rms, nn_corr, date-guess agreement) already exist; live mode is
plumbing, not new science.

## 4. Signature visualisations (each maps to an existing event)

1. **Stack progress**: frames fly onto a pile with their measured (dy, dx) shift vectors;
   the residual scatter tightens as alignment converges. Data: `frame_aligned` events.
2. **Platesolve animation**: detected centroids appear; candidate triangles flash;
   the winning consensus cluster lights up and the verified stars connect to their
   catalogue counterparts with name labels (the new `LabelIndex` supplies "Vega", not
   `gaia:2097892…`). Data: `solve_candidate` + matched-star list.
3. **Distortion quiver/heatmap** with the polynomial surface, replacing the 3D matplotlib
   popups.
4. **Metric gauges** for rms / nn_corr / star count / date-guess error, colour-graded by
   the milestone-B thresholds — the same component serves batch results and live mode.

## 5. Phasing

| phase | deliverable | status |
|---|---|---|
| P0 | event bus + `--events-jsonl`; CLI/legacy GUI consume it | **done** |
| P1 | app shell: Simple mode, progress, score cards | **done** |
| P2 | Advanced: distortion field, date-guess accuracy, catalogue downloads | **done** |
| P3 | Watch mode (folder watcher, settle rule, batching) | **done** |
| P4 | animations, subprocess isolation, packaging | next |

## 6. What the initial version does, and what it taught us

Built: `mee2024/events.py` (P0) and `mee2024/ui/` (P1) — `runner.py` (pipeline in a worker
thread, cooperative cancellation), `server.py` (token-guarded localhost HTTP + API),
`frontend.html` (one self-contained file), `app.py` (pywebview, browser fallback), and
`mee2024 ui`. 57 tests, none of which open a window.

Verified against real data: a stage-2 run through the UI reproduced the CLI numbers
exactly — 109.6 mas, 434 stars, `nn_corr` 0.166, date recovered 2023-10-28 — and rendered
six graded score cards with plain-language captions.

Three things the live test found that unit tests had not:

1. **`do_stack` used `os.mkdir`**, which cannot create a missing parent, so choosing a
   not-yet-existing output folder failed several layers deep with a bare `WinError 3`.
   This affected the CLI's `-o` equally. Now `os.makedirs`, plus the runner creates and
   validates the folder up front.
2. **The frontend treated any payload `error` key as a transport failure**, and the state
   payload used `error` for the run's own error — so polling broke at exactly the moment a
   run failed, which is when the user most needs feedback. The field is now `run_error`,
   and `api()` keys off the HTTP status alone.
3. **`png_event` converted a whole frame to float64 before downsampling** — 125 MiB for a
   3520×4656 stack, and more again for an RGBA copy. It now strides *then* casts, emits
   8-bit greyscale, and carries a stdlib PNG encoder so IMAGE events survive without
   Pillow.

### The open architectural question for P2

The pipeline runs in a **worker thread inside the server process**, which means the
server's cached triangle database (~1 GB resident once the KD-tree is built) coexists with
the pipeline's transient full-frame arrays. On a memory-pressured machine that is enough to
fail: a stage-1 run died in `get_centroids_blur` needing another 125 MiB, and later in
`compute_platescale` needing 89 MiB for 1.3 M triangle candidates. The CLI escapes this
because every run is a fresh process that exits.

**Recommendation: move the run into a subprocess for P2**, forwarding events over a pipe.
It reclaims all memory between runs, isolates the server from a pipeline crash, and makes
cancellation immediate rather than cooperative. The `EventBus` already serialises to JSON,
so the pipe is the natural transport and the frontend needs no change.

A cheaper complementary win: `get_centroids_blur` allocates several full-frame **float64**
arrays from **uint16** input. float32 would halve peak memory there with no meaningful loss
of centroid precision — worth measuring against the milestone-E benchmark rather than
assuming.

---

## 7. v1.0.0 additions

**Version and attribution.** `v1.0.0` throughout, authored by *Andrew Smith and Douglas
Smith*, shown in the window header and recorded in `setup.cfg`. The version bump also
carries a config migration: `migrate_config` is now **keyed on the version that wrote the
file**, so each fix runs once. Previously the `rough_match_threshhold` reset fired on
*every* version change, which would have silently discarded a deliberate setting at each
future release. It still fires once here, correctly — a value tuned against the `/33600`
units bug (200 was observed) is ~9x too large now that the bug is fixed.

**Distortion field** (Advanced, on by default). `render_distortion_field` draws the fitted
polynomial as arrows plus a magnitude map with contours, in arcseconds when the plate scale
is known. Saved as `Distortion_field.png` and emitted as an `image` event. This replaces
the three rotatable 3-D matplotlib windows the old code opened, which showed the same
information but could not be read at a glance or saved usefully.

**Date-guess accuracy.** Stage 1 now records `observation_date_header` from the first
frame's FITS `DATE-OBS`; stage 2 compares its blind guess against it and reports
`date_guess_error_days`. The UI grades it: within 21 days is green, beyond 60 red — because
the honest statistical capability is 2–4 weeks (see `progress.md`), so months apart means
something upstream is wrong rather than merely imprecise. Verified end to end: **−1 day**
on the zwo3 field. Note that centroid archives produced before v1.0.0 have no header date,
so the metric shows only for freshly stacked data.

**Watch mode** (`mee2024/ui/watcher.py`). A folder is polled; a frame is only opened once
(a) its last modification is at least `settle_seconds` old **and** (b) its size has not
changed since the previous poll. The size check matters: a writer slow enough that mtime
looks stale between two writes would defeat a pure mtime rule, and a truncated FITS either
fails or — worse — succeeds. Settled frames batch up until `batch_size` is reached, or
`quiet_seconds` passes with at least two held; a single frame is never dispatched alone
because it cannot be stacked. Frames present when the watch starts are adopted as
already-processed, so starting a watch does not reprocess the whole night. Polling rather
than OS notifications: no dependency, identical on all three platforms, and reliable on the
network shares capture software often writes to.

Verified with real frames: three dropped into a folder → settled → stacked → **plate solved
with 96 stars matched** at the correct pointing.

**Catalogue downloads.** Un-installed catalogues appear in Advanced with a download button.
Progress is reported through the same `stage_started`/`progress` events the pipeline uses,
so the existing progress bar renders it with no special case.

### Hosting: why not Google Drive

Drive is a poor fit for programmatic download and the code now says so explicitly rather
than failing obscurely:

* files over ~100 MB return a **virus-scan interstitial HTML page** instead of the file, so
  a naive download saves HTML as a `.zip` and fails much later with a baffling error;
* popular public files hit **"quota exceeded"**, which is also served as HTML with HTTP 200;
* the confirm-token mechanism has changed repeatedly, so scripts that handle it break;
* using Drive as a software CDN is against the spirit of its terms.

`_download` therefore inspects the first block and **refuses anything that looks like
HTML**, naming the likely cause. It also rewrites a Drive share link to the direct-download
endpoint as a best effort, verifies `Content-Length` and the SHA-256, and only renames the
`.part` file on success.

**Recommendation: a GitHub release asset as the interim host.** Free, 2 GB per asset, stable
direct URLs, no interstitial, no quota surprises, and the repository already exists — the
same effort as Drive and strictly better. Zenodo at publication for the citable DOI.

Either way the URL is **not hardcoded**: `catalogue_sources` in the config supplies it, set
with `mee2024 catalogue --set-source NAME --url URL --sha256 HASH`. Switching host is a
config change, not a release.

### A memory bug the watch-mode test found

The first watch run failed with `MemoryError: unable to allocate 499 MiB for shape
(3516, 4650, 4) float64`. The cause was pre-existing, not new: `plt.imshow(stacked, ...)`
in the stage-1 preview plot applies its colormap **at the resolution of the array it is
handed**, so a 3520x4656 frame becomes a full-size float64 RGBA intermediate. It now draws a
strided copy mapped back onto the original pixel grid with `extent=`, so the star overlay
still works in original coordinates: **524 MB → 33 MB**, visually identical. This also
affected the CLI, on any sufficiently large frame with tight memory.

---

## 8. Packaging: the Windows executable

`MEE2024.spec` now lives at the **repository root** and is run from there, because the code
uses absolute `from mee2024 import ...` imports and so needs the repo root on the path:

```bash
python -m PyInstaller MEE2024.spec --noconfirm
```

Produces `dist/MEE_2024_v1.0.0.exe`, **187 MB**, one file, no Python install needed.
Double-clicking it opens the classic GUI as before; `MEE_2024_v1.0.0.exe ui` opens the new
app window, and every CLI subcommand works.

Built and verified with **Python 3.9** — the interpreter that already carried the science
dependencies, and the more proven PyInstaller target than 3.14. The full test suite passes
on 3.9 as well as 3.14.

Verified end to end through the frozen exe: `distortion` on the real zwo3 data against the
offline catalogue reproduced **109.6 mas and recovered 2023-10-28**, matching the source
build exactly; `ui --browser` serves the frontend with the byline, the Watch tab and the
distortion-field card, `api/hello` reports both catalogues, and an unauthenticated request
is refused with 403.

Three things that had to be right, each of which broke the build first time:

1. **Two different bundle destinations.** `MEE2024util.resource_path()` joins
   `sys._MEIPASS` directly, so anything it looks up must land at the archive **root**
   (`resources/...`). But `ui/server.py` finds the frontend relative to its own
   `__file__`, which under PyInstaller is `_MEIPASS/mee2024/ui/`, so `frontend.html` must
   land **there**. Putting everything under `mee2024/` failed with
   `FileNotFoundError: _MEI.../resources/compressed_tycho2024epoch.npz`.
2. **`resources/*` does not recurse**, so the Hipparcos catalogue and label index are
   listed file by file — the same trap already fixed in `setup.cfg` for the wheel.
3. **stdout is buffered in a frozen build** when redirected or run without a console, so
   `ui` printed its URL nowhere. The URL is the only way to reach the interface, so those
   prints now pass `flush=True`.

Startup costs about 10 s, because a one-file build unpacks 187 MB to a temporary directory
on every launch. A `--onedir` build starts almost instantly at the cost of shipping a
folder rather than a single file; worth switching if the delay annoys people.

macOS and Linux are left for later, but the spec already has the `BUNDLE` branch for a
`.app`, and nothing in the code is Windows-specific except `os.startfile` in the reveal
helper, which already has `open`/`xdg-open` branches.
