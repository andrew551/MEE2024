"""Cell 4's atmosphere night maps, in the construction and style of cells 1 and 3.

Douglas, 2026-09-12: "atmosphere_night_maps.png in RECORD\\leon2026: are we able to create a
chart like this one with the Husillos atmosphere and zenith fields?  Let's confine it to only
the 10 degree data (because the 9 degree was taken at twilight and so the number of stars is
small) and only the data above 10 degrees.  So just four fields, I think."

Four it is:

    23_31_59   alt 10.03 deg   gain 0     pointing A
    23_34_38   alt 10.01 deg   gain 125   pointing A, re-pointed 0.66 deg in RA to hold 10 deg
    23_44_06   alt 14.97 deg   gain 125   pointing C
    the zenith field

WHAT A PANEL SHOWS.  Cell 1's map (tools/matrix_bruns/b17_m3_maps.py) and cell 3's
(tools/step3_atmosphere_maps.py) draw every night field re-fitted the way a CALIBRATION field
is reduced -- cubic and above frozen from the reference, quadratic free -- so the residual is
what a calibration fit cannot absorb: model error above the quadratic plus quasi-static
atmosphere.  Cell 4's horizon fields are already reduced exactly that way
(`hu_horizon_reduce.py`: quintic order, cubic-and-above frozen from the night zenith,
quadratic and scale free, corrections ON), so their stage-2 residuals ARE the map with no
re-fit.  The deep-detection stacks are used: 184, 345 and 935 stars after the cuts against
71, 135 and 653 at the zenith star-field preset.

THE ZENITH PANEL IS NOT THE SAME CONSTRUCTION, and the chart says so.  Leon froze each zenith
field's model from the OTHER five of that night and re-fitted the quadratic.  **Cell 4 has one
zenith field.**  Nothing can be frozen onto it from elsewhere, and freezing its own
cubic-and-above onto itself then re-fitting the quadratic returns the fit it already had.  So
the panel shows its free-quintic residuals -- the machinery floor, honestly labelled, not a
calibration-transfer residual.

WHAT IS LEFT OUT, and why, both at Douglas' instruction and for reasons the record carries:

  * the `cal 8 deg` window (8.54 and 9.01 deg) is the eclipse altitude and the best null
    GEOMETRY in the campaign -- one tracked field 2 min 33 s apart -- but it was shot inside
    astronomical twilight and carries 47 and 34 stars after the cuts, with bootstrap floors of
    1.2 arcsec.  Too thin to draw a structure map from;
  * pointing B (5.45-5.73 deg) sits 1200-1800 ppm off in plate scale at both gains and both
    detection settings, which is where the refraction model or the transfer from an 85 deg
    reference gives way (record section 3s).  Not a measurement of the atmosphere.

CONVENTIONS TAKEN FROM THE OTHER CELLS, NOT RE-DECIDED HERE.  Positions AND arrows in sensor
axes with y down; one arrow scale, `LSCALE = 0.0018`, identical to the Leon and Bruns maps --
legitimate here because the two plate scales agree to 0.1 % (2.2028 against 2.2054 arcsec/px),
so equal arcsec draw as equal pixels; a crimson 1-arcsec reference; a green
increasing-altitude arrow on the horizon panels; per-star median removed; the 3 x MAD outlier
clip with a 2.5 arcsec floor.

THE MAGNITUDE CUT IS THIS CELL'S OWN.  Leon's maps used G <= 11; cell 4's registered window is
G <= 13 (`tools/analysis_window.py`, docs/MATRIX_2026.md), and CLAUDE.md forbids importing an
analysis parameter from another cell by analogy.  So the maps are drawn at G <= 13 and the
stats table carries a G <= 11 column beside it so the rms values can still be compared across
cells.

Writes atmosphere/atmosphere_night_maps.png (+ chart_versions/) and atmos_maps_stats.csv, then
publishes into RECORD through hu_record.publish so nothing is overwritten.

    .venv/Scripts/python.exe tools/husillos2026/hu_atmos_maps.py
"""
import glob
import json
import os
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
import astropy.units as u

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
from analysis_window import WINDOWS  # noqa: E402
from record_charts import ChartWriter  # noqa: E402
import hu_record  # noqa: E402

WIN = WINDOWS['husillos2026']
HUS = r'F:\MEE_output\husillos2026'
HOR = os.path.join(HUS, 'horizon')
OUT = os.path.join(HUS, 'atmosphere')
REV = os.environ.get('HU_REV', 'rev01')
NX, NY = 9576, 6388
#: identical to the Leon and Bruns maps: one arcsec draws as ~556 sensor px
LSCALE = 0.0018
MAGCUT = WIN.mag          # tools/analysis_window.py, this cell's registered window (G <= 13)
#: NOT this cell's cut and never used for one: the magnitude Leon's and Bruns' night maps were
#: drawn at, carried in the stats table only so the rms columns can be compared across cells.
COMPARE_MAG = 11.0        # docs/STEP3_CHARTS_AND_SETTINGS.md section 2, the other cells' maps
SITE = EarthLocation(lat=42.09293 * u.deg, lon=-4.52702 * u.deg, height=743 * u.m)
#: above this altitude the vertical direction carries no physical meaning, so no green arrow
#: and no alt/az split -- Leon's rule for its zenith panels
ZENITH_ALT = 70.0

#: (panel label, stage-2 directory the alt/az basis and epoch come from, what it is,
#:  residual table -- None means the stage-2 directory's own TWOD_RESIDUALS.csv).
#:
#: The two 10-degree captures were UNTRACKED (record section 3w: 770 px of sidereal drift over
#: 99 frames), so their stacks are smeared and the first two revisions of this chart drew
#: that smear as sky.  Their panels now come from hu_perframe.py's per-star medians over 99
#: separately solved frames -- Leon's construction -- with the alt/az basis taken from one
#: mid-capture frame's own solve.  The 15-degree capture and the zenith were tracked and
#: keep their stacks.
def _pf(tag):
    return (os.path.join(HOR, 'perframe_' + tag, 's1', 'f050', 's2'),
            os.path.join(HOR, 'perframe_' + tag, 's1_TWOD_RESIDUALS.csv'))


PANELS = [('23_31_59', *_pf('h10_g0'), 'horizon, per-frame medians'),
          ('23_34_38', *_pf('h10_g125a'), 'horizon, per-frame medians'),
          ('23_44_06', os.path.join(HOR, 's2d_h10_g125d'), None, 'horizon'),
          ('zenith', os.path.join(HUS, 'step3', 'ref'), None, 'zenith')]


def one(pattern):
    hit = sorted(glob.glob(pattern, recursive=True))
    if not hit:
        raise SystemExit('missing: ' + pattern)
    return hit[0]


def logged_altaz(d):
    """The alt/az the PIPELINE printed for this field.

    `altaz_basis` below regresses its own mean altitude out of the catalogue, and for
    23_34_38 that came out 0.14 deg from what stage 2 printed.  The record quotes the
    pipeline's number everywhere else, so the panel labels use the pipeline's and the
    regression supplies only the DIRECTION vectors -- one quantity, one source.
    """
    lg = os.path.join(d, 'stage2.log')
    m = re.search(r'sky mean position alt/az: ([\d.]+) ([\d.]+)',
                  open(lg, encoding='utf-8', errors='replace').read()) \
        if os.path.exists(lg) else None
    return (float(m.group(1)), float(m.group(2))) if m else (float('nan'), float('nan'))


def matched_and_time(d):
    """The stage-2 matched table and the epoch its fit was made at."""
    zf = zipfile.ZipFile(one(os.path.join(d, 'distortion_data*.zip')))
    name = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    t = pd.read_csv(zf.open(name))
    t.columns = [c.strip() for c in t.columns]
    j = json.load(open(one(os.path.join(d, '**', 'distortion_results.txt')), encoding='utf-8'))
    return t, j


def altaz_basis(t, j):
    """Unit vectors of increasing altitude and azimuth in sensor pixels.

    The cell-1 construction: transform the CATALOGUE positions to alt/az at the field's own
    epoch, then regress the measured pixel positions on them.  Catalogue, not observed, so
    the basis cannot be bent by the residuals it is about to decompose.
    """
    when = Time(j['observation_date'] + 'T' + j['observation_time (UTC)'], scale='utc')
    aa = SkyCoord(t['RA(catalog)'].values * u.deg, t['DEC(catalog)'].values * u.deg
                  ).transform_to(AltAz(obstime=when, location=SITE))
    alt = aa.alt.deg
    azc = aa.az.deg * np.cos(np.radians(alt.mean()))
    A = np.column_stack([azc - azc.mean(), alt - alt.mean(), np.ones(len(t))])
    cx, *_ = np.linalg.lstsq(A, t['px'].values.astype(float), rcond=None)
    cy, *_ = np.linalg.lstsq(A, t['py'].values.astype(float), rcond=None)
    v_az = np.array([cx[0], cy[0]])
    v_alt = np.array([cx[1], cy[1]])
    return (v_alt / np.linalg.norm(v_alt), v_az / np.linalg.norm(v_az),
            float(alt.mean()), float(aa.az.deg.mean()))


def residuals(d, magcut, table=None):
    """Median-removed sensor-axis residuals, 3 x MAD clipped as on the other cells' maps."""
    r = pd.read_csv(table or one(os.path.join(d, '**', 'TWOD_RESIDUALS.csv')))
    r.columns = [c.strip() for c in r.columns]
    r = r[r['magV'] <= magcut]
    px, py = r['px'].values.astype(float), r['py'].values.astype(float)
    qx = r['dx_arcsec'].values - np.median(r['dx_arcsec'])
    qy = r['dy_arcsec'].values - np.median(r['dy_arcsec'])
    m = np.hypot(qx, qy)
    lim = max(3 * 1.4826 * np.median(np.abs(m - np.median(m))) + np.median(m), 2.5)
    keep = m < lim
    return px[keep], py[keep], qx[keep], qy[keep], int((~keep).sum())


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, panels = [], []
    for label, d, table, kind in PANELS:
        t, j = matched_and_time(d)
        v_alt, v_az, _fit_alt, _fit_az = altaz_basis(t, j)
        alt, az = logged_altaz(d)
        px, py, qx, qy, nclip = residuals(d, MAGCUT, table)
        q_alt = qx * v_alt[0] + qy * v_alt[1]
        q_az = qx * v_az[0] + qy * v_az[1]
        rms = float(np.sqrt(np.mean(qx ** 2 + qy ** 2)))
        va, ha = float(np.sqrt(np.mean(q_alt ** 2))), float(np.sqrt(np.mean(q_az ** 2)))
        _, _, cx11, cy11, _ = residuals(d, COMPARE_MAG, table)
        rows.append(dict(field=label, kind=kind, source=('per-frame medians' if table
                                                          else 'stack'), alt=alt, az=az,
                         gain=j.get('analogue gain', ''), n=len(px), clipped=nclip,
                         rms=rms, vertical=va, horizontal=ha, v_over_h=va / max(ha, 1e-9),
                         n_g11=len(cx11),
                         rms_g11=float(np.sqrt(np.mean(cx11 ** 2 + cy11 ** 2))),
                         stage2_rms=j['final rms error (arcseconds)'],
                         stars_solved=j['#stars used'],
                         platescale=j['platescale (arcseconds/pixel)'],
                         rung=j.get('fixed distortion order', '?')))
        panels.append((label + (', per-frame medians' if table else ''), px, py, qx, qy, alt,
                       None if alt > ZENITH_ALT else v_alt))
        print('%-10s alt %6.2f az %6.2f  N=%4d (%d clipped)  rms %.3f " '
              '(vertical %.3f, horizontal %.3f, V/H %.2f)  [G<=11: N=%d, rms %.3f "]'
              % (label, alt, az, len(px), nclip, rms, va, ha, va / max(ha, 1e-9),
                 len(cx11), rows[-1]['rms_g11']), flush=True)

    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(OUT, 'atmos_maps_stats.csv'), index=False)

    # `ax.set_aspect(1)` on a 9576 x 6388 sensor forces each axes box to 1.5:1, and
    # `tight_layout` then places the row titles against the box ABOVE rather than the shrunk
    # box below -- the first two revisions put the bottom row's titles inside the top row's
    # panels.  The figure is sized to the aspect instead and the spacing set explicitly.
    ncol, nrow = 2, 2
    pw = 6.6
    ph = pw * NY / NX
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(pw * ncol + 0.6, (ph + 0.55) * nrow + 2.6))
    for ax, (label, px, py, qx, qy, alt, v_alt) in zip(axes.ravel(), panels):
        ax.quiver(px, py, qx, qy, angles='xy', scale_units='xy', scale=LSCALE, width=0.003,
                  color='tab:blue')
        ax.quiver([450], [450], [1.0], [0.0], angles='xy', scale_units='xy', scale=LSCALE,
                  width=0.005, color='crimson')
        ax.annotate('1"', (500, 780), fontsize=9, color='crimson')
        if v_alt is not None:
            va = v_alt * 900
            ax.annotate('', xy=(8600 + va[0], 5500 + va[1]), xytext=(8600, 5500),
                        arrowprops=dict(arrowstyle='->', color='green', lw=1.4))
            ax.text(8150, 5050, 'up', color='green', fontsize=9)
            ax.set_title('%s  (alt %.1f deg, %d stars)' % (label, alt, len(px)), fontsize=11)
        else:
            ax.set_title('%s  (alt %.1f deg, %d stars)' % (label, alt, len(px)), fontsize=11)
        ax.text(0.02, 0.04, 'rms %.2f"' % float(np.sqrt(np.mean(qx ** 2 + qy ** 2))),
                transform=ax.transAxes, fontsize=9)
        ax.set_xlim(0, NX)
        ax.set_ylim(NY, 0)
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(panels):]:
        ax.axis('off')
    fig.suptitle(
        'Husillos 2026 night fields: residual structure a calibration fit cannot absorb\n'
        'Three `10 deg` window pointings at and above 10\u00b0, the zenith quintic\u2019s'
        ' cubic-and-above frozen and the quadratic and scale free, corrections ON \u2014\nthe way'
        ' a calibration field is reduced. The two 10\u00b0 captures were UNTRACKED (770 px of'
        ' sidereal drift over 99 frames; record \u00a73w), so their\npanels are per-star MEDIANS'
        ' over 99 separately solved frames, Le\u00f3n\u2019s construction, not stacks; the 15\u00b0 capture'
        ' was tracked and is one\ndeep-detection stack. The zenith panel is its own'
        ' free-quintic residual \u2014 cell 4 has ONE zenith field, nothing can be frozen onto'
        ' it \u2014 the\nmachinery floor. G \u2264 13, this cell\u2019s registered window (Le\u00f3n\u2019s maps'
        ' used G \u2264 11; that column is in atmos_maps_stats.csv). Arrows and\npositions both'
        ' in SENSOR axes; arrow scale identical to the Le\u00f3n 2026 and Bruns 2017 maps. Left'
        ' out: the `cal 8 deg` window (8.5\u20139.0\u00b0,\nthe eclipse altitude, shot in astronomical'
        ' twilight: 34 and 47 stars) and pointing B (5.5\u00b0, plate scale 1200\u20131800 ppm out).',
        fontsize=10.5, y=0.985, va='top')
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.025, top=0.80,
                        wspace=0.06, hspace=0.12)
    ChartWriter(OUT, REV).save(fig, 'atmosphere_night_maps.png')

    H = S[S.kind.str.startswith('horizon')]
    print()
    print('horizon panels (alt %.2f-%.2f deg): rms %.3f " (%.3f-%.3f), vertical %.3f, '
          'horizontal %.3f, V/H %.2f'
          % (H.alt.min(), H.alt.max(), H.rms.mean(), H.rms.min(), H.rms.max(),
             H.vertical.mean(), H.horizontal.mean(),
             H.vertical.mean() / max(H.horizontal.mean(), 1e-9)))
    z = S[S.kind == 'zenith'].iloc[0]
    print('zenith panel: rms %.3f " on %d stars (free quintic, its own fit)' % (z.rms, z.n))
    print()
    hu_record.publish(['atmosphere_night_maps.png'], OUT)


if __name__ == '__main__':
    main()
