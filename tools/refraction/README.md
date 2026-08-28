# Leon 2026 refraction analysis — auxiliary tooling

Experimental, probably specific to the 2026 Spanish eclipse (alt 9.5–9.9° at totality;
future eclipses are unlikely to sit this low). Everything here shells out to
`python -m mee2024.cli` — **nothing under `mee2024/` changes on this branch**, so by the
ROADMAP §6 classification none of it can alter a pipeline number. If any of it ever
graduates into the package, that is a results-changing change and takes the full
validation path.

The plan, data inventory, physics case and pilot result are in
[`docs/REFRACTION_2026.md`](../../docs/REFRACTION_2026.md). Read that first.

| script | what it does |
|---|---|
| `inventory.py` | header sweep of the horizon sets and meridian mosaic on `G:\Leon Aug 2026` → `INVENTORY.csv` |
| `drive_horizon.py` | M2: per-frame stage-1 + stage-2 (corrections ON and OFF) over the 45-frame horizon blocks; resumable; harvests `perframe_results.csv` |

Reductions land under `D:\MEE2024 output\MEE_output\refraction\`. Inputs on `G:` and `I:`
are read-only.

Run the venv, never system Python:

```bash
.venv/Scripts/python.exe tools/refraction/drive_horizon.py --windows N2,N3 --fields H1,H2,H3
```
