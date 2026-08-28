"""First M2 checkpoint: scale vs altitude on the completed N2 fields.

Corrections OFF should show the raw refraction slope (~-600 to -700 ppm/deg at alt 9-10 deg,
until now known only from the model); corrections ON should show ~zero slope if the model's
shape is right, and its mean against the same night's zenith-ON scale (2.2068874 arcsec/px,
FOCUSPOS 17041, measured 2026-08-26) is the pilot's +155 ppm tension statistic done properly.
Reads frame results directly; does not touch the running driver's CSV.
"""
import glob, json, os
import numpy as np

BASE = r"D:/MEE2024 output/MEE_output/refraction/perframe"
ZENITH_ON = 2.2068874          # arcsec/px, 08-12 Z1_base corrections ON, same focus as N2

def load(window, field):
    rows = {}
    for f in glob.glob(os.path.join(BASE, window, field, "f*", "corr_*", "**",
                                    "distortion_results.txt"), recursive=True):
        d = json.load(open(f))
        parts = f.replace("\\", "/").split("/")
        fr = [p for p in parts if p.startswith("f") and p[1:].isdigit()][0]
        corr = "on" if "corr_on" in parts else "off"
        rows.setdefault(fr, {})[corr] = d
    out = []
    for fr in sorted(rows):
        r = rows[fr]
        if "on" not in r or "off" not in r:
            continue
        alt = r["on"].get("observation alt (degrees)")
        out.append((fr, alt,
                    r["on"]["platescale (arcseconds/pixel)"],
                    r["off"]["platescale (arcseconds/pixel)"],
                    r["on"]["#stars used"], r["on"]["final rms error (arcseconds)"],
                    r["off"]["#stars used"], r["off"]["final rms error (arcseconds)"]))
    return out

def fit_slope(alt, y):
    """OLS slope of relative scale (ppm) against altitude (deg), with its standard error."""
    a = np.asarray(alt); v = np.asarray(y)
    p = (v / v.mean() - 1) * 1e6
    A = np.column_stack([np.ones_like(a), a - a.mean()])
    coef, res, *_ = np.linalg.lstsq(A, p, rcond=None)
    n = len(a)
    sig2 = float(res[0]) / (n - 2) if len(res) else np.var(p)
    se = np.sqrt(sig2 / np.sum((a - a.mean())**2))
    scat = np.std(p - A @ coef, ddof=2)
    return coef[1], se, scat

for field in ("H1", "H2"):
    rows = load("N2", field)
    if len(rows) < 40:
        print(f"N2/{field}: only {len(rows)} complete frames, skipping"); continue
    fr, alt, ps_on, ps_off, n_on, rms_on, n_off, rms_off = map(np.array, zip(*rows))
    alt = alt.astype(float)
    print(f"\n=== N2/{field}: {len(rows)} frames, alt {alt.max():.3f} -> {alt.min():.3f} deg "
          f"(span {alt.max()-alt.min():.3f} deg) ===")
    print(f"  stars/frame: ON median {np.median(n_on):.0f}, OFF {np.median(n_off):.0f}; "
          f"per-frame rms: ON median {np.median(rms_on):.3f} arcsec, OFF {np.median(rms_off):.3f} arcsec")
    for tag, ps in (("OFF", ps_off), ("ON ", ps_on)):
        sl, se, scat = fit_slope(alt, ps)
        print(f"  corrections {tag}: slope {sl:+8.1f} +/- {se:5.1f} ppm/deg   "
              f"(scatter about trend {scat:5.1f} ppm/frame)")
    print(f"  mean ON scale {ps_on.mean():.7f} arcsec/px  ->  vs zenith-ON same night: "
          f"{(ps_on.mean()/ZENITH_ON - 1)*1e6:+6.1f} ppm "
          f"(se of mean ~{np.std(ps_on,ddof=1)/np.sqrt(len(ps_on))/ps_on.mean()*1e6:.1f} ppm)")

# model expectation for the OFF slope at these altitudes: d/dh of k*sec^2(z) with k=283 ppm
for h in (9.0, 9.4, 10.7, 11.1):
    z = np.radians(90 - h)
    dvert = 2 * 283e-6 / np.cos(z)**2 * np.tan(z) * np.pi/180   # per deg, vertical term
    print(f"  [model: at alt {h:4.1f} deg, d(vertical compression)/d(alt) = {-dvert*1e6:7.0f} ppm/deg; "
          f"frame-average roughly half that]")
