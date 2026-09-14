"""One displacement-vector chart per gain block, so the two can be compared star by star.

Douglas, 2026-09-11: "I think I need to examine the stars in the two blocks, gain 0 and
gain 125.  Create two charts like record_field.png ... so I can more closely look at how the
deflections are behaving for each block."

The reason this is the chart to look at: the two blocks are the same 64 stars measured 39 s
apart, and they disagree about L by more than either error bar -- 1.596 +- 0.507 against
2.782 +- 0.540 under Method 2, and 1.819 +- 0.697 against 4.322 +- 0.698 under Method 1.  The
cause has been chased through the plate scale (which explains the M1-M2 gap exactly but not the
block-to-block one) and through per-star noise (r = 0.484 between the blocks).  What has NOT
been looked at is the spatial pattern: whether gain 125's excess is spread over the field, or
sits in one region, or grows with radius.  A number cannot show that and a chart can.

IN ALT/AZ, as Leon's `record_field.png` is (Douglas, 2026-09-11).  That is the frame the
atmosphere is polarised in -- Husillos' own V/H is 1.80 (section 3h) -- and on the raw
sensor axes the same field reads y/x = 0.98, perfectly isotropic, so the polarisation is
invisible until the rotation is applied.  Altitude increases upward and azimuth to the
right, the sky as the observer sees it.

Drawn through `tools/record_charts.py` like every other chart in the set -- `field_chart`,
`scale_bars`, `bar_frame` and `SkyFrame` are imported, not re-implemented
(`tests/test_record_charts.py` fails a tool that carries its own copy).  Conventions therefore
come from the module: RA ASCENDING TO THE RIGHT, which is not the sky convention but is the
set's, and every arrow end asserted inside the axes.

WHAT THE ARROWS ARE.  Displacement = observation - catalogue, with the block's fitted pointing
offset and rotation removed -- the same three nuisances stage 3's `_find_rotation_matrix`
absorbs before it fits L.  So what is left is deflection + measurement noise: Leon's "raw
view" (`record_field_raw.png`), not its L-view, because cell 4 applies no nuisance field
(section 3h).  Both charts share one arrow scale so they can be compared directly.

    .venv/Scripts/python.exe tools/husillos2026/hu_field_charts.py
"""
import glob
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body  # noqa: E402
from astropy.time import Time  # noqa: E402
import astropy.units as u  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
from record_charts import SkyFrame, field_chart, scale_bars, bar_frame  # noqa: E402
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
from hu_record import publish  # noqa: E402

HUS = r'F:\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'charts')
RECORD = r'F:\MEE_output\RECORD\husillos2026'

NX, NY = 9576, 6388               # Zeus 455M PRO (IMX455)
R_SUN_AS = 947.1                  # docs/STAGE3_THEORY.md section 3
LAT, LON, HEIGHT = 42.09293, -4.52702, 743.0
SITE = EarthLocation(lat=LAT * u.deg, lon=LON * u.deg, height=HEIGHT * u.m)

#: (tag, stage-2 directory, mid-time, gain, frames)
BLOCKS = [('gain0', 'eclipse_gain0', '18:29:59', 0, 101),
          ('gain125', 'eclipse_gain125', '18:29:20', 125, 126)]

ARROW_DEG = 0.34     # degrees of chart per arcsec, as the Leon chart uses


def matched(d):
    z = glob.glob(os.path.join(HUS, 'step3', d, '**', 'distortion_data*.zip'), recursive=True)
    if not z:
        raise SystemExit('no stage-2 output in ' + d)
    zf = zipfile.ZipFile(z[0])
    n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    t = pd.read_csv(zf.open(n), dtype={'ID': str})
    t.columns = [c.strip() for c in t.columns]
    t['ID'] = t['ID'].astype(str).str.strip()
    res = json.load(zf.open([x for x in zf.namelist()
                             if x.endswith('distortion_results.txt')][0]))
    return t.set_index('ID'), res


def strip_pointing(ra, dec, dxi, deta):
    """Remove the offset and rotation stage 3 fits before it looks for a deflection.

    Three parameters on a 2N system: two translations and one rotation about the field
    centre. Whatever is left is deflection plus noise -- it is NOT a cleaned-up deflection,
    and a chart of it should not be read as one.
    """
    ra0, de0 = ra.mean(), dec.mean()
    X = (ra - ra0) * np.cos(np.radians(de0)) * 3600.0
    Y = (dec - de0) * 3600.0
    n = len(ra)
    Z, O = np.zeros(n), np.ones(n)
    A = np.vstack([np.column_stack([O, Z, -Y]),
                   np.column_stack([Z, O, X])])
    c, *_ = np.linalg.lstsq(A, np.concatenate([dxi, deta]), rcond=None)
    fit = A @ c
    return dxi - fit[:n], deta - fit[n:]


def main():
    os.makedirs(OUT, exist_ok=True)
    tabs = {tag: matched(d) for tag, d, _t, _g, _f in BLOCKS}
    both = set.intersection(*[set(t.index) for t, _ in tabs.values()])
    print('two-witness stars shared by both blocks: %d' % len(both))

    def altaz(ra, dec, T):
        aa = SkyCoord(np.atleast_1d(ra) * u.deg, np.atleast_1d(dec) * u.deg).transform_to(
            AltAz(obstime=T, location=SITE))
        return np.asarray(aa.az.deg), np.asarray(aa.alt.deg)

    def sky_dirs(ra_c, dec_c, T):
        """Unit vectors of increasing altitude and azimuth, expressed in the SKY frame the
        displacements are measured in -- (RA*cos(dec), Dec).  Same construction as
        `step3_charts_record.altaz_basis`, but that one returns sensor pixels because Leon's
        arrows are in sensor axes; these arrows are already on the sky."""
        fc = SkyCoord(ra_c * u.deg, dec_c * u.deg)
        aa = fc.transform_to(AltAz(obstime=T, location=SITE))
        out = {}
        for key, off in (('alt', dict(alt=aa.alt + 0.05 * u.deg, az=aa.az)),
                         ('az', dict(alt=aa.alt, az=aa.az + 0.05 * u.deg / np.cos(aa.alt)))):
            q = SkyCoord(AltAz(obstime=T, location=SITE, **off)).icrs
            v = np.array([(q.ra.deg - fc.ra.deg) * np.cos(np.radians(dec_c)),
                          q.dec.deg - fc.dec.deg])
            out[key] = v / np.linalg.norm(v)
        return out['alt'], out['az'], float(aa.alt.deg), float(aa.az.deg)

    prepared = {}
    for tag, d, tmid, gain, nfr in BLOCKS:
        t, res = tabs[tag]
        t = t.loc[sorted(both)]
        ra = t['RA(catalog)'].values
        dec = t['DEC(catalog)'].values
        cd = np.cos(np.radians(dec))
        dxi = (t['RA(obs)'].values - ra) * 3600.0 * cd
        deta = (t['DEC(obs)'].values - dec) * 3600.0
        vx, vy = strip_pointing(ra, dec, dxi, deta)

        T = Time('2026-08-12 ' + tmid)
        sun = get_body('sun', T, SITE)
        SF = SkyFrame.from_stars(ra, dec, t['px'].values, t['py'].values,
                                 res['platescale (arcseconds/pixel)'])
        cra, cdec = SF.corners(NX, NY)

        # --- into alt/az
        e_alt, e_az, alt_c, az_c = sky_dirs(ra.mean(), dec.mean(), T)
        saz, salt = altaz(ra, dec, T)
        caz, calt = altaz(cra, cdec, T)
        sunaz, sunalt = altaz(sun.ra.deg, sun.dec.deg, T)
        v_alt = vx * e_alt[0] + vy * e_alt[1]
        v_az = vx * e_az[0] + vy * e_az[1]
        tilt = np.degrees(np.arctan2(
            *SF.sky_to_px_dir(e_alt[0], e_alt[1])[[0, 1]] * np.array([1, -1])))
        print('%-8s field centre alt %.2f deg az %.2f deg | V/H %.2f (vertical %.3f, '
              'horizontal %.3f arcsec)'
              % (tag, alt_c, az_c, np.std(v_alt) / np.std(v_az), np.std(v_alt), np.std(v_az)))
        prepared[tag] = dict(ra=ra, dec=dec, az=saz, alt=salt, vx=v_az, vy=v_alt,
                             mag=t['magV'].values, corners=(caz, calt),
                             sun=(float(sunaz[0]), float(sunalt[0]), R_SUN_AS / 3600.0),
                             gain=gain, nfr=nfr, tmid=tmid, alt_c=alt_c,
                             ps=res['platescale (arcseconds/pixel)'])

    # ONE arrow scale for both charts, or they cannot be compared by eye
    allv = np.concatenate([np.hypot(p['vx'], p['vy']) for p in prepared.values()])
    print('displacement range over both blocks: %.2f to %.2f arcsec' % (allv.min(), allv.max()))

    for tag, d, tmid, gain, nfr in BLOCKS:
        p = prepared[tag]
        cosf = float(np.cos(np.radians(p['alt_c'])))
        fig, ax = plt.subplots(figsize=(11.5, 8))
        lo, hi, _, _ = field_chart(
            ax, p['az'], p['alt'], p['vx'], p['vy'], p['corners'], p['sun'], ARROW_DEG, cosf,
            groups=[(np.ones(len(p['az']), bool),
                     dict(s=24, color='tab:red' if gain else 'tab:blue',
                          label='gain %d, %d frames, %d two-witness stars'
                                % (gain, nfr, len(p['az']))))],
            arrow_color='tab:red' if gain else 'tab:blue', arrow_lw=1.5,
            point_labels=(['%.1f' % m for m in p['mag']], 0.010, 'black', 6.5),
            pad=(0.10, 0.08), sun_ellipse_ratio=cosf, sun_ring2=True,
            xlabel='azimuth (degrees, increasing to the right as seen by the observer)',
            ylabel='altitude (degrees)',
            title='Displacement vectors \u2014 Husillos 2026, gain %d block, mid %s UTC'
                  % (gain, tmid))
        fig.text(0.06, 0.020,
                 'each arrow = the star\u2019s measured shift after subtracting only the fitted '
                 'pointing offset and rotation;\ndeflection + measurement noise remain. '
                 'No nuisance field is applied (\u00a73h). Both blocks drawn at one arrow scale.\n'
                 'ALT/AZ frame: altitude upward, azimuth increasing to the right \u2014 the frame '
                 'the atmosphere is polarised in.',
                 fontsize=9)
        ax.legend(fontsize=8.5, loc='center left', bbox_to_anchor=(1.01, 0.75))
        sc = float(np.sqrt(np.mean(p['vx'] ** 2 + p['vy'] ** 2)))
        scale_bars(ax, ((0.44, 1.0, '1 arcsec of displacement'),
                        (0.34, sc, 'rms vector (%.2f")' % sc)),
                   ARROW_DEG, cosf, hi - lo)
        bar_frame(ax, (1.02, 0.29, 0.34, 0.22))
        fig.subplots_adjust(right=0.72, bottom=0.13)
        name = 'field_%s_altaz.png' % tag
        fig.savefig(os.path.join(OUT, name), dpi=130)
        plt.close(fig)
        # through publish(), so a re-run supersedes rather than overwrites (CLAUDE.md:
        # "Never overwrite a chart revision") -- this tool wrote straight into RECORD until
        # 2026-09-11 and would have destroyed a revision the first time a number changed
        publish([name], OUT)

        # the radial split, which is what a scale or a deflection error shows up in
        az0, alt0 = p['sun'][0], p['sun'][1]
        X = (p['az'] - az0) * np.cos(np.radians(alt0)) * 3600.0
        Y = (p['alt'] - alt0) * 3600.0
        R = np.hypot(X, Y)
        rad = (p['vx'] * X + p['vy'] * Y) / R
        tan = (p['vx'] * -Y + p['vy'] * X) / R
        print('%-8s rms %.3f " | radial %+.3f +- %.3f " | tangential %+.3f +- %.3f " | '
              'r(radial, 1/R) %+.2f'
              % (tag, sc, rad.mean(), rad.std(), tan.mean(), tan.std(),
                 float(np.corrcoef(rad, R_SUN_AS / R)[0, 1])))
        print('         -> %s' % os.path.join(RECORD, name))


if __name__ == '__main__':
    main()
