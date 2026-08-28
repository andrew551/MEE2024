"""M2 synthesis: all nine horizon field-windows -- slopes, offsets, differentials.

Zenith corrections-ON references: 08-12 measured directly (2.2068874 arcsec/px, Z1_base);
08-11 derived from its corrections-off 6-field mean via the -221.9 ppm shift of LEON 18.1
(sd 5.4 ppm across 12 fields), since no corrections-on zenith reduction exists for 08-11.
"""
import csv, glob, json, os
import numpy as np

BASE = r"D:/MEE2024 output/MEE_output/refraction/perframe"
ZREF = {'N1': 2.2077996 * (1 - 221.9e-6), 'N2': 2.2068874, 'N3': 2.2068874}
UTC  = {('N1','H1'):'23:16',('N1','H2'):'23:21',('N1','H3'):'23:26',
        ('N2','H1'):'22:31',('N2','H2'):'22:36',('N2','H3'):'22:41',
        ('N3','H1'):'00:22',('N3','H2'):'00:27',('N3','H3'):'00:32'}

def load(w, f):
    rows = {}
    for p in glob.glob(os.path.join(BASE, w, f, "f*", "corr_*", "**",
                                    "distortion_results.txt"), recursive=True):
        d = json.load(open(p))
        parts = p.replace("\\", "/").split("/")
        fr = [x for x in parts if x.startswith("f") and x[1:].isdigit()][0]
        rows.setdefault(fr, {})["on" if "corr_on" in parts else "off"] = d
    return [(fr, r["on"]["observation alt (degrees)"],
             r["on"]["platescale (arcseconds/pixel)"], r["off"]["platescale (arcseconds/pixel)"],
             r["on"]["final rms error (arcseconds)"])
            for fr, r in sorted(rows.items()) if "on" in r and "off" in r]

def slope(alt, ps):
    a = np.asarray(alt, float); p = (np.asarray(ps)/np.mean(ps) - 1)*1e6
    A = np.column_stack([np.ones_like(a), a - a.mean()])
    c, res, *_ = np.linalg.lstsq(A, p, rcond=None)
    sig2 = float(res[0])/(len(a)-2) if len(res) else np.var(p)
    return c[1], np.sqrt(sig2/np.sum((a-a.mean())**2))

out, S = [], {}
for w in ("N1", "N2", "N3"):
    for f in ("H1", "H2", "H3"):
        rows = load(w, f)
        fr, alt, on, off, rms = map(np.array, zip(*rows))
        alt = alt.astype(float)
        s_on, s_off = slope(alt, on), slope(alt, off)
        offs = (on.mean()/ZREF[w] - 1)*1e6
        se = np.std(on, ddof=1)/np.sqrt(len(on))/on.mean()*1e6
        S[(w, f)] = dict(alt=alt.mean(), on=on.mean(), offs=offs, se=se, s_on=s_on)
        out.append(dict(window=w, field=f, utc_start=UTC[(w,f)], n_frames=len(rows),
            alt_mean_deg=round(alt.mean(),3), alt_span_deg=round(alt.max()-alt.min(),3),
            off_slope_ppm_per_deg=round(s_off[0],1), off_slope_se_ppm_per_deg=round(s_off[1],1),
            on_slope_ppm_per_deg=round(s_on[0],1), on_slope_se_ppm_per_deg=round(s_on[1],1),
            on_offset_vs_zenith_ppm=round(offs,1), on_offset_se_ppm=round(se,1),
            rms_median_arcsec=round(float(np.median(rms)),3)))

with open(os.path.join(os.path.dirname(BASE), "m2_fieldwindow_summary.csv"), "w", newline="") as fp:
    wtr = csv.DictWriter(fp, fieldnames=list(out[0])); wtr.writeheader(); wtr.writerows(out)

print(f"{'win/field':10s} {'UTC':>6s} {'alt mean (deg)':>14s} {'ON slope (ppm/deg)':>19s} "
      f"{'offset vs zenith (ppm)':>23s} {'rms (arcsec)':>13s}")
for r in out:
    print(f"{r['window']+'/'+r['field']:10s} {r['utc_start']:>6s} {r['alt_mean_deg']:14.3f} "
          f"{r['on_slope_ppm_per_deg']:+10.1f} +/-{r['on_slope_se_ppm_per_deg']:5.1f} "
          f"{r['on_offset_vs_zenith_ppm']:+15.1f} +/-{r['on_offset_se_ppm']:4.1f} "
          f"{r['rms_median_arcsec']:13.3f}")

print("\nwithin-window spatial differentials (focus-free), scaled to 0.3 deg of altitude:")
for w in ("N1", "N2", "N3"):
    pairs = sorted(((S[(w,f)]['alt'], f) for f in ("H1","H2","H3")))
    (a1, f1), (a2, f2) = pairs[0], pairs[1]     # the two lowest fields bracket the band
    d = (S[(w,f2)]['on']/S[(w,f1)]['on'] - 1)*1e6
    se = np.hypot(S[(w,f2)]['se'], S[(w,f1)]['se'])
    print(f"  {w}: {f2}-{f1} = {d:+7.1f} +/- {se:4.1f} ppm over {a2-a1:+.2f} deg "
          f"-> {d*0.3/(a2-a1):+6.1f} ppm per 0.3 deg")
