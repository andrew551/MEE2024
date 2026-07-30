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

| phase | deliverable | size |
|---|---|---|
| P0 | event bus + `--events-jsonl`; CLI/legacy GUI consume it | small — do with milestone B |
| P1 | pywebview shell: Simple mode, progress, score cards, results browser | the core build |
| P2 | Advanced mode (full options), packaging (PyInstaller for Win/macOS) | moderate |
| P3 | Live mode (watchdog folder watcher + incremental pipeline) | moderate |
| P4 | animations & polish | open-ended, incremental |

Prerequisites from the existing roadmap: milestone B (quality score) feeds the score
cards; the `LabelIndex` feeds the annotations; nothing blocks P0/P1 today.
