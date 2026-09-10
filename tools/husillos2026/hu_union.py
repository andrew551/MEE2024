"""The Leon union, applied to Husillos' two gain blocks.

Douglas, 2026-09-10: "The two exposures at the same [field] in Leon 2026 are conceptually
equivalent to the two different gains at the same exposure of Husillos.  Let's try this
method."  Leon's 0.6 s and 1.2 s tiers were not stacked together -- they were reduced
separately and combined AT THE STAR LEVEL (`tools/step3_s2_union.py`, docs/STEP3_2026.md
"S2 first pass -- the union").  The recipe, and what each step is for:

  - per block: displacement = obs - catalogue; the block's MEDIAN displacement is subtracted,
    which kills the per-block pointing constant so the blocks can be mixed;
  - per star: the MEDIAN across blocks (with two blocks that is the mean);
  - a CROSS-BLOCK CONSISTENCY VET -- a star whose blocks disagree by more than 3x the field's
    cross-block MAD (floor 1.5 ") is dropped.  This is the only automatic defence against a
    corrupted centroid, and it is why the union beats a single deeper stack: one stack gives
    one witness per star and the vet has nothing to work with (docs/STEP3_2026.md, "One
    master or the union -- decided by measurement": the master re-admitted the star the vet
    removes, worth -0.32 " of L on its own);
  - the union rides ONE HOST block's model, so the output is one consistent geometry.

Husillos needs one step Leon did not, and it is not optional.  Leon's tiers were 40 s apart
at 40 deg altitude; Husillos' blocks are 39 s apart at 8.6 deg altitude, where dR/dz is about
46 "/deg.  Measured here: the two blocks' CATALOGUE positions differ by 0.60 " with a 0.54 "
spread about that -- differential refraction over the 39 s, not a constant.  So absolute
positions must NOT be averaged.  Displacements are combined (they are epoch-independent to
first order, because the same refraction that moves the catalogue moves the observation) and
the HOST block's catalogue frame is restored.  For a star only one block saw, the frame
transfer is a quadratic in field position fitted on the shared stars: measured residual
0.005 ", against 0.724 " if the difference were treated as a constant.

Stage 3 is not re-implemented.  The union is written back as a
CATALOGUE_MATCHED_ERRORS.csv inside a copy of the host's distortion zip and run through the
CLI unchanged, exactly as `hu_step3.py witness` does.

Reported in two admissions, as Leon reports them: every star, and the two-witness rule
(matrix-wide since 2026-09-02).  No third, invented, magnitude-tier rule -- Leon's "G <= 11
on one tier, fainter only with two" used its own catalogue limit, and Husillos' registered
window (analysis_window.WINDOWS['husillos2026']) carries no such split.

    python tools/husillos2026/hu_union.py [build|stage3|report]
"""
import glob
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
from analysis_window import WINDOWS  # noqa: E402

WIN = WINDOWS['husillos2026']
PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')
HUS = r'D:\MEE2024 output\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'step3')
UNI = os.path.join(OUT, 'union')  # per-host subdirectory below, so a swap cannot overwrite

#: The host is the block whose frame the union is expressed in.  gain125 is the deeper of
#: the two -- 84 matched stars against 75 -- which is the same reason Leon hosted on its
#: best-populated tier rather than its longest exposure.
#:
#: Hosting is NOT cosmetic and must be tested rather than assumed: the union rides the
#: host's plate solution, so the host's fitted scale is adopted whole and only the
#: DISPLACEMENTS are averaged (Leon likewise put every tier on the 0.6 s fitted model).
#: With the blocks' scales 87 ppm apart, HU_UNION_HOST=gain0 re-runs the whole thing the
#: other way round and the difference is the size of that choice.
HOST = os.environ.get('HU_UNION_HOST', 'gain125')
OTHER = 'gain0' if HOST == 'gain125' else 'gain125'

#: Sensor size, READ from the stage-1 record rather than typed.  It was typed once as
#: 6248 x 4176 -- the APS-C IMX571 -- and Husillos' camera is a Zeus 455M PRO (IMX455),
#: 9576 x 6388 full frame, so a star at px 7390 looked off-sensor when it is nowhere near
#: the edge.  Nothing downstream broke (the transfer polynomial only cares about a
#: consistent scaling) but the normalisation was wrong by 1.5x.
def _img_shape():
    z = glob.glob(os.path.join(HUS, 'eclipse', 's1_sun_dark', 'centroid_data*.zip'))
    r = json.load(io.TextIOWrapper(zipfile.ZipFile(z[0]).open('results.txt'),
                                   encoding='utf-8', errors='replace'))
    ny, nx = r['img_shape']
    return int(nx), int(ny)


NX, NY = _img_shape()
W_NORM = float(NX) / 2.0  # field-normalised coordinates for the transfer polynomial


def _zip(tag):
    z = glob.glob(os.path.join(OUT, 'eclipse_%s' % tag, '**', 'distortion_data*.zip'),
                  recursive=True)
    if not z:
        raise SystemExit('no stage-2 output for %s: run hu_step3.py stage2 first' % tag)
    return z[0]


def _matched(tag):
    """The stage-2 matched table.  IDs stay STRINGS: a 19-digit Gaia id near a float is the
    trap in CLAUDE.md, and `int(float(id))` changes 73 % of them."""
    zf = zipfile.ZipFile(_zip(tag))
    n = [x for x in zf.namelist() if x.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]
    t = pd.read_csv(zf.open(n), dtype={'ID': str})
    t.columns = [c.strip() for c in t.columns]
    t['ID'] = t['ID'].astype(str).str.strip()
    assert t['ID'].str.contains(r'\.', regex=True).sum() == 0, 'a float-shaped id got through'
    return t.set_index('ID')


def _disp(t):
    """Displacement obs - catalogue, in arcsec on the sky, RA scaled by cos(dec)."""
    c = np.cos(np.radians(t['DEC(catalog)'].values))
    return (np.column_stack([(t['RA(obs)'].values - t['RA(catalog)'].values) * 3600 * c,
                             (t['DEC(obs)'].values - t['DEC(catalog)'].values) * 3600]), c)


def _quad(x, y):
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])


def _fit_transfer(src, dst, x, y, label):
    """Least-squares quadratic taking `src` columns to `dst` columns, with its residual
    printed -- the residual is the whole justification for using it."""
    M = _quad(x, y)
    cs, res = [], []
    for k in range(dst.shape[1]):
        c, *_ = np.linalg.lstsq(M, dst[:, k] - src[:, k], rcond=None)
        cs.append(c)
        res.append((dst[:, k] - src[:, k]) - M @ c)
    print('    %-28s residual %s' % (label, ' , '.join('%.4f' % np.std(r) for r in res)))
    return cs


def _apply_transfer(cs, src, x, y):
    M = _quad(x, y)
    return np.column_stack([src[:, k] + M @ cs[k] for k in range(len(cs))])


def build():
    """Combine the two blocks into one star table in the host's frame."""
    os.makedirs(UNI, exist_ok=True)
    H, O = _matched(HOST), _matched(OTHER)
    both = H.index.intersection(O.index)
    print('%s %d stars, %s %d stars, shared %d, union %d'
          % (HOST, len(H), OTHER, len(O), len(both), len(H.index.union(O.index))))

    dH, cosH = _disp(H)
    dO, _ = _disp(O)
    mH, mO = np.median(dH, axis=0), np.median(dO, axis=0)
    print('block median displacement: %s (%+.3f, %+.3f) ", %s (%+.3f, %+.3f) "'
          % (HOST, mH[0], mH[1], OTHER, mO[0], mO[1]))
    dH = dH - mH
    dO = dO - mO

    # --- the frame transfer, fitted on the shared stars only
    print('  frame transfer %s -> %s, fitted on the %d shared stars (rms, arcsec / px):'
          % (OTHER, HOST, len(both)))
    h, o = H.loc[both], O.loc[both]
    xo = (o['px'].values - NX / 2) / W_NORM
    yo = (o['py'].values - NY / 2) / W_NORM
    cat_o = np.column_stack([o['RA(catalog)'].values, o['DEC(catalog)'].values])
    cat_h = np.column_stack([h['RA(catalog)'].values, h['DEC(catalog)'].values])
    # in arcsec, so the printed residual is readable; RA scaled by cos(dec)
    sc = np.column_stack([np.cos(np.radians(cat_o[:, 1])) * 3600, np.full(len(both), 3600.0)])
    cs_cat = _fit_transfer(cat_o * sc, cat_h * sc, xo, yo, 'catalogue frame (arcsec)')
    cs_px = _fit_transfer(np.column_stack([o['px'].values, o['py'].values]),
                          np.column_stack([h['px'].values, h['py'].values]),
                          xo, yo, 'pixel frame (px)')

    # --- per-star combination
    rows = []
    posH = {k: i for i, k in enumerate(H.index)}
    posO = {k: i for i, k in enumerate(O.index)}
    for sid in H.index.union(O.index):
        inH, inO = sid in posH, sid in posO
        if inH:
            r = H.loc[sid]
            cat = np.array([r['RA(catalog)'], r['DEC(catalog)']])
            px, py = r['px'], r['py']
        else:
            r = O.loc[sid]
            x = (r['px'] - NX / 2) / W_NORM
            y = (r['py'] - NY / 2) / W_NORM
            s = np.array([np.cos(np.radians(r['DEC(catalog)'])) * 3600, 3600.0])
            cat = _apply_transfer(cs_cat, (np.array([[r['RA(catalog)'],
                                                      r['DEC(catalog)']]]) * s),
                                  np.array([x]), np.array([y]))[0] / s
            px, py = _apply_transfer(cs_px, np.array([[r['px'], r['py']]]),
                                     np.array([x]), np.array([y]))[0]
        ds = []
        if inH:
            ds.append(dH[posH[sid]])
        if inO:
            ds.append(dO[posO[sid]])
        ds = np.array(ds)
        d = np.median(ds, axis=0)
        spread = float(np.hypot(*(ds[0] - ds[1]))) if len(ds) == 2 else np.nan
        rows.append(dict(ID=sid, px=px, py=py,
                         px_dist=r['px_dist'], py_dist=r['py_dist'],
                         RA_cat=cat[0], DEC_cat=cat[1],
                         dx=d[0], dy=d[1], nblock=len(ds), spread=spread,
                         magV=r['magV'], flag_is_double=r['flag_is_double'],
                         flag_missing_pm=r['flag_missing_pm'],
                         flag_is_outlier=r['flag_is_outlier'],
                         err=r['error(")']))
    U = pd.DataFrame(rows)

    # --- the cross-block consistency vet, Leon's rule verbatim
    sp = U.loc[U.nblock >= 2, 'spread']
    lim = max(3 * 1.4826 * np.median(np.abs(sp - sp.median())) + sp.median(), 1.5)
    bad = (U.nblock >= 2) & (U.spread > lim)
    print('  cross-block vet: limit %.3f " (median spread %.3f "), %d vetted out'
          % (lim, sp.median(), int(bad.sum())))
    for _, r in U[bad].iterrows():
        print('     OUT: G %.2f at (%.0f,%.0f) spread %.3f "'
              % (r.magV, r.px, r.py, r.spread))
    U = U[~bad].reset_index(drop=True)
    U.to_csv(os.path.join(UNI, 'husillos_union_star_table.csv'), index=False)
    print('  union: %d stars (%d two-witness, %d single)'
          % (len(U), int((U.nblock == 2).sum()), int((U.nblock == 1).sum())))
    return U


def _write_zip(U, name):
    """A copy of the host's distortion zip whose matched table is the union."""
    src = _zip(HOST)
    dst = os.path.join(UNI, '%s.zip' % name)
    # the host's median displacement is added back, so a host-only star's row is
    # bit-for-bit the host's own -- the invariant that says the transfer did nothing odd
    H, _ = _matched(HOST), None
    dHm = np.median(_disp(H)[0], axis=0)
    cd = np.cos(np.radians(U['DEC_cat'].values))
    out = pd.DataFrame({
        'Unnamed: 0': np.arange(len(U)),
        'ID': U['ID'].values,
        'px': U['px'].values, 'py': U['py'].values,
        'px_dist': U['px_dist'].values, 'py_dist': U['py_dist'].values,
        'RA(catalog)': U['RA_cat'].values, 'DEC(catalog)': U['DEC_cat'].values,
        'RA(obs)': U['RA_cat'].values + (U['dx'].values + dHm[0]) / 3600 / cd,
        'DEC(obs)': U['DEC_cat'].values + (U['dy'].values + dHm[1]) / 3600,
        'magV': U['magV'].values, 'error(")': U['err'].values,
        'flag_is_double': U['flag_is_double'].values,
        'flag_missing_pm': U['flag_missing_pm'].values,
        'flag_is_outlier': U['flag_is_outlier'].values})
    zin = zipfile.ZipFile(src)
    with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith('CATALOGUE_MATCHED_ERRORS.csv'):
                buf = io.StringIO()
                out.to_csv(buf, index=False)
                data = buf.getvalue().encode('utf-8')
            zout.writestr(item, data)
    return dst


def _stage3(z, tag):
    d = os.path.join(UNI, 'method2_%s' % tag)
    os.makedirs(d, exist_ok=True)
    cmd = [PY, '-m', 'mee2024.cli', 'eclipse', z,
           '--set', 'eclipse_method=Method 2',
           '--set', 'eclipse_limiting_mag=%.1f' % WIN.mag,
           # not cropped radially: Douglas, 2026-09-10
           '--set', 'limit_radial_sun_radii=False',
           '--set', 'remove_double_stars_eclipse=False',
           '--no-display', '--quiet', '-o', d]
    with open(os.path.join(d, 'stage3.log'), 'w') as f:
        subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    fs = sorted(glob.glob(os.path.join(d, '**', 'ECLIPSE_OUTPUT*.txt'), recursive=True))
    if not fs:
        print('  %-14s stage 3 FAILED' % tag)
        return
    for line in io.open(fs[0], encoding='utf-8', errors='replace').read().splitlines():
        s = line.strip()
        if any(k in s for k in ('Method 2 results', 'number of stars',
                                'deflected star position rms')):
            print('  %-14s %s' % (tag, s.replace(u'\u00b1', ' +- ')
                                  .encode('ascii', 'replace').decode('ascii')[:130]))


def stage3():
    U = build()
    print()
    for tag, sel in (('union_all', np.ones(len(U), bool)),
                     ('union_2witness', (U.nblock == 2).values)):
        print('=== %s (%d stars)' % (tag, int(sel.sum())))
        _stage3(_write_zip(U[sel].reset_index(drop=True), tag), tag)


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'stage3'
    UNI = os.path.join(UNI, 'host_%s' % HOST)
    os.makedirs(UNI, exist_ok=True)
    print('host = %s (union expressed in its frame; its fitted scale is adopted)' % HOST)
    {'build': build, 'stage3': stage3}[cmd]()
