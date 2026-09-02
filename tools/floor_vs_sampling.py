"""Is the measured noise floor a sampling artefact? The magnitude test, on three instruments.

Douglas, 2026-09-02: "The Leakey data was also taken at a much higher sampling because of
the small pixels of the 294MM camera (at bin 1). Does the sampling factor enter into the
noise floors of the Leakey, Bruns and Leon data?"

It is the right question to ask of any cross-instrument floor comparison, because a residual
rms contains two things that behave completely differently:

  * **per-star centroid noise**, which scales as FWHM/SNR and carries an extra penalty when
    the PSF is undersampled. It is a property of the camera and the night, and it is
    magnitude-dependent -- faint stars are noisier;
  * **structure** -- model mismatch, drift, atmosphere -- which is what the floor is supposed
    to measure. It is magnitude-INDEPENDENT, because a systematic displacement field moves a
    bright star and a faint star by the same angle.

So the two separate cleanly by binning the residuals in magnitude. If a floor is
sampling-limited it rises toward faint magnitudes and its bright-end asymptote is small; if
it is structure the bins are flat and the asymptote is the floor itself.

The three instruments differ in sampling by a factor of two:

    Leakey  Askar 65PHQ + ASI294MM bin 1   2.315 um px, EFL 415 mm  ->  1.1506 "/px
    Bruns   NP101is + FLI ML8051           5.50  um px, EFL 543 mm  ->  2.0868 "/px
    Leon    FRA500 + 0.7x + ASI2600MM      3.76  um px, EFL 352 mm  ->  2.2054 "/px

A second discriminator is reported alongside: the same floors expressed in PIXELS. A floor
that is a fixed fraction of a pixel across instruments is a centroiding artefact; one that is
a fixed angle is atmosphere or model error.

Reads the quadratic-free residuals already written for the floor table, so it measures the
same fields the table's rows are built from. No re-fitting.
"""
import glob, os
import numpy as np, pandas as pd

R = r"D:/MEE2024 output/MEE_output"
OUT = os.path.join(R, 'step3_record', 'floor_vs_sampling.csv')
# FWHM medians from the campaign's own PSF surveys (psf_bruns2017 / psf_leakey /
# psf_portland). Portland stands in for Leon: same FRA500 + 0.7x + ASI2600 rig, a
# different night and site, so its FWHM is indicative rather than Leon's own.
SETS = {
    'Leakey 2024 zenith': dict(glob=os.path.join(R, 'step3_record/leakey_quadfree/*/**/TWOD_RESIDUALS.csv'),
                               ps=1.1506, px_um=2.315, efl=415, cam='ASI294MM bin1', fwhm_px=2.71, psf='psf_leakey'),
    'Leon 2026 zenith': dict(glob=os.path.join(R, 'step3_record/zenith_quadfree/*/**/TWOD_RESIDUALS.csv'),
                             ps=2.2054, px_um=3.76, efl=352, cam='ASI2600MM', fwhm_px=2.58, psf='psf_portland (same rig)'),
    'Bruns 2017 night': dict(glob=os.path.join(R, 'matrix_bruns2017_m3/*/**/TWOD_RESIDUALS.csv'),
                             ps=2.0868, px_um=5.50, efl=543, cam='FLI ML8051', fwhm_px=1.64, psf='psf_bruns2017'),
}
BINS = [(4, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13)]


def fields(pattern):
    """One entry per field, clipped exactly as the floor table clips it: 3 MAD on the vector
    magnitude with a 2.5 arcsec floor, after removing that field's own median."""
    out = []
    for f in sorted(glob.glob(pattern, recursive=True)):
        d = pd.read_csv(f)
        if not {'dx_arcsec', 'dy_arcsec', 'magV'} <= set(d.columns):
            continue
        dx = d.dx_arcsec.values - np.median(d.dx_arcsec.values)
        dy = d.dy_arcsec.values - np.median(d.dy_arcsec.values)
        m = np.hypot(dx, dy)
        lim = max(3*1.4826*np.median(np.abs(m - np.median(m))) + np.median(m), 2.5)
        k = m < lim
        out.append(pd.DataFrame(dict(magV=d.magV.values[k], dx=dx[k], dy=dy[k])))
    return out


def by_bin(flds, lo, hi, min_stars=8):
    """The floor table averages per-FIELD rms, so this does too: a field contributes one
    number to each magnitude bin, and the bins are averaged over fields with equal weight.
    Pooling all fields' stars instead lets a common clip throw away the noisier fields
    wholesale -- which understated every bright-end value by about a factor of two in the
    first version of this tool."""
    vals = []
    for d in flds:
        k = (d.magV >= lo) & (d.magV < hi)
        if k.sum() < min_stars:
            continue
        vals.append(float(np.sqrt(np.mean(d.dx[k]**2 + d.dy[k]**2))))
    return (float(np.mean(vals)) if vals else np.nan), len(vals)


print('%-20s %-14s %8s %7s %9s %9s   %s' % ('set', 'camera', 'px (um)', '"/px', 'FWHM px',
                                            'FWHM "', 'sampling'))
for name, cfg in SETS.items():
    print('%-20s %-14s %8.3f %7.4f %9.2f %9.2f   %s' % (name, cfg['cam'], cfg['px_um'], cfg['ps'],
          cfg['fwhm_px'], cfg['fwhm_px']*cfg['ps'],
          'UNDERSAMPLED' if cfg['fwhm_px'] < 2.0 else 'adequate (Nyquist >= 2 px)'))
print()
print('per-field residual rms by magnitude, averaged over fields (arcsec; fields contributing)')
print('%-20s %s' % ('set', '  '.join('%11s' % ('%.0f-%.0f' % b) for b in BINS)))
out = {}
for name, cfg in SETS.items():
    flds = fields(cfg['glob'])
    if not flds:
        print('%-20s no residuals found' % name); continue
    row = [by_bin(flds, lo, hi) for lo, hi in BINS]
    out[name] = (cfg, row, flds)
    print('%-20s %s' % (name, '  '.join('     --    ' if v != v else '%6.3f (%2d)' % (v, n)
                                        for v, n in row)))

print()
print('%-20s %9s %11s %9s %11s %8s %9s' % ('set', 'G 8-10', 'in pixels', 'G 11-13', 'in pixels',
                                           'faint/br', 'all G<=11'))
for name, (cfg, row, flds) in out.items():
    br = np.nanmean([v for (v, n), b in zip(row, BINS) if 8 <= b[0] < 10 and v == v])
    fa = np.nanmean([v for (v, n), b in zip(row, BINS) if b[0] >= 11 and v == v])
    allv, _ = by_bin(flds, 0, 11, min_stars=20)
    print('%-20s %9.3f %11.4f %9.3f %11.4f %8.2f %9.3f'
          % (name, br, br/cfg['ps'], fa, fa/cfg['ps'], fa/br, allv))

rows = []
for name, (cfg, row, flds) in out.items():
    br = float(np.nanmean([v for (v, n), b in zip(row, BINS) if 8 <= b[0] < 10 and v == v]))
    fa = float(np.nanmean([v for (v, n), b in zip(row, BINS) if b[0] >= 11 and v == v]))
    allv, _ = by_bin(flds, 0, 11, min_stars=20)
    rows.append(dict(set=name, camera=cfg['cam'], px_um=cfg['px_um'], arcsec_per_px=cfg['ps'],
                     fwhm_px=cfg['fwhm_px'], fwhm_arcsec=cfg['fwhm_px']*cfg['ps'],
                     structure_G8_10=br, structure_px=br/cfg['ps'], faint_G11_13=fa,
                     faint_px=fa/cfg['ps'], faint_over_bright=fa/br, all_G_le_11=allv))
pd.DataFrame(rows).to_csv(OUT, index=False)
print()
print('->', OUT)
print()
print('Read the two "in pixels" columns against the two arcsec columns. A floor that is a')
print('fixed fraction of a PIXEL across instruments is a centroiding or sampling artefact; one')
print('that is a fixed ANGLE is atmosphere or model error. The faint/bright ratio says how much')
print('of the floor of each instrument is photon noise: 1.0 means none of it is.')
