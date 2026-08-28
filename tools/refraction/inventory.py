"""Inventory of the Leon refraction datasets: horizon calibrations x 3 nights + meridian mosaic.

One row per capture block: frame count, exposure, gain, UTC window, FOCUSPOS (the user's
explicit check: horizon focus should equal the same night's zenith focus), FOCTEMP.
Read-only on G: and I:.
"""
import glob, os, csv, datetime
from astropy.io import fits

G = r"G:/Leon Aug 2026"
rows = []

def block(path, tag):
    fs = sorted(glob.glob(os.path.join(path, "*.fits")))
    if not fs:
        # one level down (timestamp subfolders, CAL_piLeo-style)
        for sub in sorted(os.listdir(path)):
            p2 = os.path.join(path, sub)
            if os.path.isdir(p2):
                block(p2, tag + "/" + sub)
        return
    h0, h1 = fits.getheader(fs[0]), fits.getheader(fs[-1])
    exps, gains, focs, ftmp = set(), set(), set(), []
    for f in fs:
        h = fits.getheader(f)
        exps.add(float(h.get('EXPTIME', -1)))
        gains.add(int(h.get('GAIN', -1)))
        focs.add(int(h.get('FOCUSPOS', -99999)))
        if 'FOCTEMP' in h: ftmp.append(float(h['FOCTEMP']))
    t0 = h0['DATE-OBS']; t1 = h1['DATE-OBS']
    end = (datetime.datetime.fromisoformat(t1)
           + datetime.timedelta(seconds=float(h1.get('EXPTIME', 0)))).isoformat()
    rows.append(dict(block=tag, n_frames=len(fs),
        exptime_s=sorted(exps), gain=sorted(gains),
        utc_first=t0[:21], utc_end=end[:21],
        focuspos_steps=sorted(focs), foctemp_C_first=round(ftmp[0],1) if ftmp else None,
        foctemp_C_last=round(ftmp[-1],1) if ftmp else None,
        object=h0.get('OBJECT',''), set_temp_C=h0.get('SET-TEMP', h0.get('CCD-TEMP',''))))

for night in ("2026-08-11", "2026-08-12", "2026-08-13"):
    hz = os.path.join(G, night, "Horizon")
    if os.path.isdir(hz):
        for fld in sorted(os.listdir(hz)):
            block(os.path.join(hz, fld), f"{night}/Horizon/{fld}")

mos = os.path.join(G, "2026-08-13", "Refraction mosaic")
for fld in sorted(os.listdir(mos)):
    p = os.path.join(mos, fld)
    if os.path.isdir(p): block(p, "mosaic/" + fld)

# focus references: same-night zenith + eclipse-day CAL_piLeo
for tag, pat in [("REF zenith 08-11", r"I:/Leon 2026/2026-08-11/Zenith/Z1_base/*/*.fits"),
                 ("REF zenith 08-12", r"I:/Leon 2026/2026-08-12/Zenith/Z1_base/*/*.fits"),
                 ("REF CAL_piLeo day", r"I:/Leon 2026/2026-08-12/Eclipse/CAL_piLeo/18_29_19/*.fits")]:
    fs = sorted(glob.glob(pat))
    if fs:
        h = fits.getheader(fs[0])
        rows.append(dict(block=tag, n_frames=len(fs), exptime_s=[float(h.get('EXPTIME',-1))],
            gain=[int(h.get('GAIN',-1))], utc_first=h['DATE-OBS'][:21], utc_end='',
            focuspos_steps=[int(h.get('FOCUSPOS',-99999))],
            foctemp_C_first=round(float(h['FOCTEMP']),1) if 'FOCTEMP' in h else None,
            foctemp_C_last=None, object=h.get('OBJECT',''), set_temp_C=h.get('SET-TEMP','')))

out = r"D:/MEE2024 output/MEE_output/refraction/INVENTORY.csv"
with open(out, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} blocks -> {out}\n")
# compact print: horizon + refs in full, mosaic summarised
mos_rows = [r for r in rows if r['block'].startswith('mosaic')]
for r in rows:
    if r['block'].startswith('mosaic'): continue
    print(f"{r['block']:55s} n={r['n_frames']:3d} exp={r['exptime_s']} gain={r['gain']} "
          f"UTC {r['utc_first']}..{r['utc_end'][11:] if r['utc_end'] else ''} FOC={r['focuspos_steps']} "
          f"FT={r['foctemp_C_first']}..{r['foctemp_C_last']}")
if mos_rows:
    import collections
    nf = collections.Counter(r['n_frames'] for r in mos_rows)
    foc = sorted(set(f for r in mos_rows for f in r['focuspos_steps']))
    print(f"\nMOSAIC: {len(mos_rows)} fields, frames/field {dict(nf)}, "
          f"UTC {mos_rows[0]['utc_first']} .. {mos_rows[-1]['utc_end']}, FOCUSPOS {foc}")
    print(f"  exposures {sorted(set(e for r in mos_rows for e in r['exptime_s']))}, "
          f"gains {sorted(set(g for r in mos_rows for g in r['gain']))}")
