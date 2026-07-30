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
| P1 | app shell: Simple mode, progress, score cards | **done (initial version)** |
| P2 | Advanced mode polish, subprocess isolation, packaging | next |
| P3 | Live mode (folder watcher + incremental pipeline) | not started |
| P4 | animations & polish | not started |

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
