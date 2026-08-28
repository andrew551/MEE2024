"""M2 checkpoint 2: all completed field-windows -- slopes, offsets, the H3-H1
differential (the Method-1-relevant number), and the N2 vs N3 repeat."""
import glob, json, os
import numpy as np

BASE = r"D:/MEE2024 output/MEE_output/refraction/perframe"
ZENITH_ON = 2.2068874   # arcsec/px, 08-12 Z1_base corrections ON (FOCUSPOS 17041)

def load(window, field):
    rows = {}
    for f in glob.glob(os.path.join(BASE, window, field, "f*", "corr_*", "**",
                                    "distortion_results.txt"), recursive=True):
        d = json.load(open(f))
        parts = f.replace("\\", "/").split("/")
        fr = [p for p in parts if p.startswith("f") and p[1:].isdigit()][0]
        rows.setdefault(fr, {})["on" if "corr_on" in parts else "off"] = d
    out = [(fr, r["on"].get("observation alt (degrees)"),
            r["on"]["platescale (arcseconds/pixel)"], r["off"]["platescale (arcseconds/pixel)"],
            r["on"]["#stars used"], r["on"]["final rms error (arcseconds)"])
           for fr, r in sorted(rows.items()) if "on" in r and "off" in r]
    return out

def slope(alt, ps):
    a = np.asarray(alt, float); p = (np.asarray(ps)/np.mean(ps) - 1) * 1e6
    A = np.column_stack([np.ones_like(a), a - a.mean()])
    c, res, *_ = np.linalg.lstsq(A, p, rcond=None)
    n = len(a); sig2 = float(res[0])/(n-2) if len(res) else np.var(p)
    return c[1], np.sqrt(sig2/np.sum((a-a.mean())**2))

stats = {}
print(f"{'window/field':14s} {'n':>3s} {'alt range (deg)':>16s} {'OFF slope':>16s} "
      f"{'ON slope (ppm/deg)':>19s} {'ON offset vs zenith (ppm)':>26s} {'rms med (arcsec)':>17s}")
for w in ("N2", "N3"):
    for f in ("H1", "H2", "H3"):
        rows = load(w, f)
        if len(rows) < 40: continue
        fr, alt, on, off, n_on, rms = map(np.array, zip(*rows))
        alt = alt.astype(float)
        s_off = slope(alt, off); s_on = slope(alt, on)
        offset = (on.mean()/ZENITH_ON - 1) * 1e6
        se_off = np.std(on, ddof=1)/np.sqrt(len(on))/on.mean()*1e6
        stats[(w, f)] = dict(alt=alt.mean(), s_on=s_on, offset=offset, se=se_off,
                             on_mean=on.mean())
    # printed after collection for alignment
for (w, f), st in stats.items():
    rows = load(w, f)
    fr, alt, on, off, n_on, rms = map(np.array, zip(*rows))
    alt = alt.astype(float)
    s_off = slope(alt, off); s_on = st['s_on']
    print(f"{w+'/'+f:14s} {len(rows):3d} {alt.max():7.3f}-{alt.min():6.3f} "
          f"{s_off[0]:+9.1f} +/-{s_off[1]:4.1f} {s_on[0]:+10.1f} +/-{s_on[1]:5.1f} "
          f"{st['offset']:+15.1f} +/-{st['se']:4.1f} {np.median(rms):13.3f}")

print("\n--- the Method-1-relevant differential: cal sightline (H3) minus eclipse pointing (H1) ---")
for w in ("N2", "N3"):
    if (w,'H3') in stats and (w,'H1') in stats:
        d = (stats[(w,'H3')]['on_mean']/stats[(w,'H1')]['on_mean'] - 1)*1e6
        se = np.hypot(stats[(w,'H3')]['se'], stats[(w,'H1')]['se'])
        da = stats[(w,'H3')]['alt'] - stats[(w,'H1')]['alt']
        print(f"  {w}: H3-H1 = {d:+7.1f} +/- {se:.1f} ppm over {da:+.2f} deg of altitude "
              f"(CAL_piLeo-to-eclipse is ~0.3-0.9 deg of the same geometry)")

print("\n--- repeatability: same field, ~2 h apart, 2.2 K cooler / +0.9 hPa ---")
for f in ("H1", "H2", "H3"):
    if ('N2',f) in stats and ('N3',f) in stats:
        a, b = stats[('N2',f)], stats[('N3',f)]
        print(f"  {f}: ON slope {a['s_on'][0]:+.1f}+/-{a['s_on'][1]:.1f} -> {b['s_on'][0]:+.1f}+/-{b['s_on'][1]:.1f} ppm/deg;"
              f"  offset {a['offset']:+.1f} -> {b['offset']:+.1f} ppm (N3 carries the 4-step focus caveat)")
