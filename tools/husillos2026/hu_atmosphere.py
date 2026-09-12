"""Cell 4's atmosphere term: null-test L on horizon fields at and around the eclipse altitude.

Douglas, 2026-09-12: "Reduce the horizon fields so we can get an atmosphere term."

WHAT THE TERM IS.  Every other cell carries +-0.11-0.33 " of L from NULL TESTS
(docs/STEP3_CHARTS_AND_SETTINGS.md section 2): a star field with no Sun in it is reduced
exactly as the eclipse field was, the eclipse Sun's frame position is imposed on its
residuals, and the science estimator is run.  True L is zero, so whatever comes back is what
the atmosphere, the model and the estimator manufacture.  Cell 4 has had none.

TWO CONSTRUCTIONS, both run, because the cell has two pathways:

  FIELD-TO-ZENITH -- the analogue of the cell's METHOD 2 of record.  Each horizon field fitted
  at the eclipse blocks' own rung (quadratic free, cubic and above frozen from the night
  zenith, scale free: hu_horizon_reduce.py), then the Sun imposed and L fitted.  This is what
  the pathway of record manufactures on a field with no deflection -- model transfer from
  85 deg to 9 deg, refraction with the assumed weather, and the atmosphere, all at once -- and
  it works on fields at different pointings.  "Match the null's construction to the science
  design, or it charges the wrong thing" (section 2 above).

  CONSECUTIVE PAIRS -- the matrix's standard construction and the analogue of METHOD 1.  A
  field refitted CONSTANT-ONLY against the previous capture's model, so the residual is what
  changed between two epochs with the model held.  Valid only when both captures see the same
  field; every consecutive pair is tried, and pairs that are NOT the same field are printed
  and discarded, not averaged in.  Each valid pair is also run the other way round (the
  earlier field against the later's model), which is the same epoch difference seen from the
  better-solved side.

THE INPUT IS NOT WHAT THE FOLDER NAMES SAY.  The `10 deg` window was described in the record
as one tracked field sweeping 10.4 -> 8.5 deg, from one plate solve propagated over seven
captures.  Solving them individually (hu_horizon_reduce.py h10, zenith preset):

    23_31_59  h10_g0        RA 175.72 Dec +29.60   alt 10.03   71 stars
    23_34_38  h10_g125a     RA 176.38 Dec +29.60   alt 10.01  135 stars   <- re-pointed 0.66 deg
    23_37_17  h10_g125b     19 matched of 35 centroids: no solve              in RA to HOLD 10 deg
    23_41_01  h10_g0_c      RA 177.87 Dec +23.28   alt  5.73   22 stars, scale 1350 ppm off
    23_42_43  h10_g125c     the slew from 5.7 to 15 deg is inside it (frame 48 fails to align)
    23_44_06  h10_g125d     RA 178.01 Dec +37.54   alt 14.97  685 stars

Three pointings at 10, 5.7 and 15 deg, not one field.  Fields outside 7.5-12 deg are reported
and not averaged; a reference with fewer than 40 stars is not trusted as a host.

Corrections are ON throughout (the site, the assumed weather): at 9 deg the catalogue must be
refraction-corrected per epoch or the few-hundred-ppm change in differential refraction
between epochs lands in the residuals -- the artefact of record section 3r.

The cuts are the cell's registered window (G <= 13, R > 2 R_sun, no outer crop); the Sun is
the eclipse Sun's frame position (5043, 3386) px; the estimator "L scale" is Method 2's
freedoms with no nuisance (the cell's), Leon's base and v-deg2 beside it; a bootstrap floor
from the per-star residuals; and a 63-star subsample for like-for-like with the union.

Both detection variants are reported where they exist: the zenith star-field preset
(s1_/s2_) and the eclipse blocks' deep detection (s1d_/s2d_).

    .venv/Scripts/python.exe tools/husillos2026/hu_atmosphere.py
"""
import glob
import json
import os
import re
import subprocess
import sys
import zlib

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
from analysis_window import WINDOWS  # noqa: E402
from hu_horizon_reduce import CAPTURES, midtime, SITE  # noqa: E402

PY = os.path.join(REPO, '.venv', 'Scripts', 'python.exe')
HUS = r'D:\MEE2024 output\MEE_output\husillos2026'
HOR = os.path.join(HUS, 'horizon')
OUT = os.path.join(HUS, 'atmosphere')
NULLS = os.path.join(OUT, 'nulls')

WIN = WINDOWS['husillos2026']
NX, NY = 9576, 6388
PS = 2.2028
R_SUN_AS = 947.1
SUNPX, SUNPY = 5043.0, 3386.0
W_NORM = NX / 2.0
N_UNION = 63
#: a field this far from the eclipse altitude (8.7 deg) is reported but not averaged in
ALT_LO, ALT_HI = 7.5, 12.0
#: consecutive captures are the same field only if their solved centres are within this
SAME_FIELD_DEG = 2.0
MIN_REF_STARS = 40
#: The third variant exists because the first three `10 deg` captures were UNTRACKED
#: (record section 3w): their stacks are smeared, so the '' and 'd' rows for 23_31_59,
#: 23_34_38 and 23_37_17 are contaminated, and the only valid residuals for them are
#: hu_perframe.py's per-star medians over separately solved frames.  Those have no stage-1
#: zip, so the consecutive-pair construction does not apply to them; field-to-zenith does.
VARIANTS = (('', 'zenith star-field preset'), ('d', 'deep: the eclipse blocks\' detection'),
            ('p', 'per-frame medians (hu_perframe.py; the untracked captures)'))
UNTRACKED = ('h10_g0', 'h10_g125a', 'h10_g125b')


def design(px, py, rx, ry, R, scale=False, nuis_deg=None):
    """The science estimator's design matrix (tools/step3_s2_union.py), plus an optional
    isotropic scale column for the cell's Method 2 freedoms."""
    xs, ys = (px - NX / 2) / W_NORM, (py - NY / 2) / W_NORM
    ux, uy = rx / R, ry / R
    n = len(px)
    Z = np.zeros(n)
    cols_x = [np.ones(n), Z, -(py - NY / 2) * PS, ux * R_SUN_AS / R]
    cols_y = [Z, np.ones(n), (px - NX / 2) * PS, uy * R_SUN_AS / R]
    labels = ['N1', 'N2', 'Th', 'L']
    if scale:
        cols_x.append((px - NX / 2) * PS)
        cols_y.append((py - NY / 2) * PS)
        labels.append('S')
    if nuis_deg:
        for i in range(nuis_deg + 1):
            for j in range(nuis_deg + 1 - i):
                if i == 0 and j == 0:
                    continue
                cols_x.append(Z)
                cols_y.append(xs ** i * ys ** j)
                labels.append('v%d%d' % (i, j))
    return np.vstack([np.column_stack(cols_x), np.column_stack(cols_y)]), labels


def fit_L(dx, dy, px, py, rx, ry, R, scale=False, nuis_deg=None):
    A, labels = design(px, py, rx, ry, R, scale, nuis_deg)
    c, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    return c[labels.index('L')]


def paths(tag, v):
    if v == 'p':
        pf = os.path.join(HOR, 'perframe_' + tag)
        table = os.path.join(pf, 's1_TWOD_RESIDUALS.csv')
        mid = os.path.join(pf, 's1', 'f050', 's2')
        res = glob.glob(os.path.join(mid, '**', 'distortion_results.txt'), recursive=True)
        return dict(res=res[0] if (res and os.path.exists(table)) else None,
                    resid=table if os.path.exists(table) else None,
                    log=os.path.join(mid, 'stage2.log'), s1zip=None)
    d2 = os.path.join(HOR, 's2%s_%s' % (v, tag))
    res = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    resid = glob.glob(os.path.join(d2, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    s1 = glob.glob(os.path.join(HOR, 's1%s_%s' % (v, tag), 'centroid_data*.zip'))
    return dict(res=res[0] if res else None, resid=resid[0] if resid else None,
                log=os.path.join(d2, 'stage2.log'), s1zip=s1[0] if s1 else None)


def altaz_of(log):
    if not os.path.exists(log):
        return None, None
    m = re.search(r'sky mean position alt/az: ([\d.]+) ([\d.]+)',
                  open(log, encoding='utf-8', errors='replace').read())
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def sep_deg(j1, j2):
    """Angular separation of two solves' field centres."""
    r1, d1, r2, d2 = (np.radians(v) for v in (j1['RA'], j1['DEC'], j2['RA'], j2['DEC']))
    c = np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(r1 - r2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def field_rng(label):
    """A random stream keyed to the FIELD, not to its position in the run.

    The first version drew the bootstrap floor and the 63-star subsample from one rng shared
    by every field, so adding a capture to the run advanced the stream and changed the floors
    already reported for the others -- the same 145-star pair read 0.641 " in one run and
    0.773 " in the next with identical input.  `zlib.crc32`, not `hash()`: Python salts
    string hashing per process, so `hash()` is not stable between runs either.
    """
    return np.random.default_rng(zlib.crc32(label.encode()) ^ 11)


def null_from(path, rng):
    """Impose the Sun on a residual table, apply the cell's cuts, run the estimators."""
    d = pd.read_csv(path)
    d.columns = [c.strip() for c in d.columns]
    d = d[d['magV'] <= WIN.mag]
    px, py = d['px'].values.astype(float), d['py'].values.astype(float)
    dx = d['dx_arcsec'].values - np.median(d['dx_arcsec'])
    dy = d['dy_arcsec'].values - np.median(d['dy_arcsec'])
    err = d['error_arcsec'].values
    rx, ry = (px - SUNPX) * PS, (py - SUNPY) * PS
    R = np.hypot(rx, ry)
    keep = R > WIN.rmin * R_SUN_AS
    if keep.sum() < 25:
        return None
    px, py, dx, dy, rx, ry, R, err = (v[keep] for v in (px, py, dx, dy, rx, ry, R, err))
    Lb = fit_L(dx, dy, px, py, rx, ry, R)
    Ls = fit_L(dx, dy, px, py, rx, ry, R, scale=True)
    Lv = fit_L(dx, dy, px, py, rx, ry, R, nuis_deg=2)
    # 300 draws, not 60.  At 60 the standard error on a standard deviation is ~9 %, and the
    # same 174-star field read 0.407 " and 0.522 " under two different seeds -- a 28 % swing
    # that would have been reported as a floor.  The floor decides whether a null is
    # structure or photon noise, so it has to be quieter than the thing it is judging.
    floor = float(np.std([fit_L(dx + rng.normal(0, err / np.sqrt(2)),
                                dy + rng.normal(0, err / np.sqrt(2)),
                                px, py, rx, ry, R, scale=True) for _ in range(300)], ddof=1))
    sub = []
    for _ in range(200):
        i = rng.choice(len(px), min(N_UNION, len(px)), replace=False)
        sub.append(fit_L(dx[i], dy[i], px[i], py[i], rx[i], ry[i], R[i], scale=True))
    return dict(n=int(keep.sum()), rms=float(np.sqrt(np.mean(dx ** 2 + dy ** 2) / 2)),
                L_base=Lb, L_scale=Ls, L_vdeg2=Lv, floor=floor,
                sub63_rms=float(np.sqrt(np.mean(np.array(sub) ** 2))))


def refit_constant(label, s1zip, ref_results, tmid):
    """`label`'s stage-1 centroids refitted with only the constant free, every other
    coefficient (and the scale) taken from `ref_results`."""
    d = os.path.join(NULLS, label)
    os.makedirs(d, exist_ok=True)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    if hit:
        return hit[0]
    with open(os.path.join(d, 'stage2.log'), 'w') as fh:
        subprocess.run([PY, '-m', 'mee2024.cli', 'distortion', s1zip, '--order', 'quintic',
                        '--fix-distortion', ref_results,
                        '--set', 'distortion_fixed_coefficients=constant',
                        '--set', 'distortion_fit_tol_initial=20.0',
                        '--set', 'distortion_fit_tol=3.0',
                        '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
                        *SITE, '--set', 'observation_time=' + tmid,
                        '--no-display', '--quiet', '-o', d],
                       cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    hit = glob.glob(os.path.join(d, '**', 'TWOD_RESIDUALS.csv'), recursive=True)
    return hit[0] if hit else None


def summarise(N, title):
    """Total, floor, and the structure the two imply.

    The matrix quotes the TOTAL and never subtracts the floor
    (docs/STEP3_CHARTS_AND_SETTINGS.md section 2), but it does report the decomposition
    beside it -- "Bruns' +-0.15 leaves only +-0.05 of atmosphere and Leon's +-0.33 leaves
    +-0.30".  Cell 4 needs that decomposition more than any other cell, because here the
    floor can EXCEED the total: a horizon field of 47 stars at 1 s through 6 air masses
    carries more per-star noise than it carries atmosphere, and that noise is already priced
    in the union's own statistical error bar.  structure = sqrt(max(0, total^2 - floor^2)).
    """
    print()
    print(title)
    fl = np.sqrt(np.mean(N.floor.values ** 2))
    for col, lab in (('L_scale', "THE CELL'S ESTIMATOR (Method 2 freedoms, no nuisance)"),
                     ('L_base', 'Leon base'), ('L_vdeg2', 'Leon v-deg2')):
        v = N[col].values
        tot = np.sqrt(np.mean(v ** 2))
        print('   %-52s rms %.3f "   mean %+.3f   worst %+.3f   structure %.3f "'
              % (lab, tot, v.mean(), v[np.argmax(np.abs(v))],
                 np.sqrt(max(0.0, tot ** 2 - fl ** 2))))
    print('   floor (per-star noise alone, scale-free)             %.3f "   [%s]'
          % (fl, ', '.join('%d stars' % n for n in N.n.values)))
    print('   at the union\'s 63 stars (subsample rms)               %.3f "'
          % np.sqrt(np.mean(N.sub63_rms.values ** 2)))


def solved_captures(v):
    out = []
    for tag, folder, name, gain, a, b in CAPTURES:
        p = paths(tag, v)
        if p['res']:
            alt, az = altaz_of(p['log'])
            out.append(dict(tag=tag, folder=folder, name=name, gain=gain, alt=alt, az=az,
                            j=json.load(open(p['res'], encoding='utf-8')), **p))
    return out


def run_variant(v, vname, rng):
    solved = solved_captures(v)
    print('=' * 100)
    print('DETECTION: %s  (%s)' % (vname, 's2%s_<tag>' % v))
    if not solved:
        print('   nothing solved yet')
        return None, None
    print('   solved: ' + ', '.join('%s (%.1f deg, %d stars)' % (s['tag'], s['alt'],
                                                                   s['j']['#stars used'])
                                     for s in solved))
    # ---------------------------------------------------------------- field-to-zenith
    print()
    print('FIELD-TO-ZENITH NULLS: each field at the eclipse rung against the zenith, Sun imposed')
    print('%-14s %6s %5s %5s %8s %9s %9s %9s %8s %9s  %s'
          % ('field', 'alt', 'gain', 'N', 'rms(")', 'L base', 'L scale', 'L v-deg2', 'floor',
             '63-star', ''))
    zrows = []
    for s_ in solved:
        r = null_from(s_['resid'], field_rng('z' + v + s_['tag'])) if s_['resid'] else None
        if r is None:
            print('%-14s %6.2f   too few stars after the cuts' % (s_['tag'], s_['alt']))
            continue
        ok = ALT_LO <= s_['alt'] <= ALT_HI
        note = '' if ok else '<- outside %.1f-%.1f deg: reported, not averaged' % (ALT_LO, ALT_HI)
        if v != 'p' and s_['j']['#stars used'] < MIN_REF_STARS:
            note = '<- %d stars: solve not trusted, excluded' % s_['j']['#stars used']
            ok = False
        if v != 'p' and s_['tag'] in UNTRACKED:
            note = '<- UNTRACKED capture (section 3w): a smeared stack, excluded'
            ok = False
        r.update(field=s_['tag'], alt=s_['alt'], gain=s_['gain'], used=ok, variant=vname)
        zrows.append(r)
        print('%-14s %6.2f %5d %5d %8.3f %+9.3f %+9.3f %+9.3f %8.3f %9.3f  %s'
              % (s_['tag'], s_['alt'], s_['gain'], r['n'], r['rms'], r['L_base'],
                 r['L_scale'], r['L_vdeg2'], r['floor'], r['sub63_rms'], note))
    Z = pd.DataFrame(zrows)
    if len(Z) and Z.used.any():
        summarise(Z[Z.used], 'FIELD-TO-ZENITH, %d field(s) inside %.1f-%.1f deg:'
                  % (int(Z.used.sum()), ALT_LO, ALT_HI))
        # The `cal 8 deg` captures were shot inside astronomical twilight and carry 37-53
        # stars against the dark-sky window's 184-345, so a plain rms over the band is
        # dominated by the thinnest field's photon noise rather than by any atmosphere.
        # The deep-field line below is the same construction on the fields that can
        # actually resolve structure; it is a STAR-COUNT split, stated, not a quality gate
        # applied after seeing the answers.
        deep = Z[Z.used & (Z.n >= N_UNION * 2)]
        if len(deep) and len(deep) < int(Z.used.sum()):
            summarise(deep, '   -- of those, the %d with more than %d stars after the cuts:'
                      % (len(deep), N_UNION * 2))
    # ---------------------------------------------------------------- consecutive pairs
    print()
    print('CONSECUTIVE-PAIR NULLS: constant-only against the neighbouring capture, same field only')
    prows = []
    for k in range(1, len(solved)):
        s_, p_ = solved[k], solved[k - 1]
        if s_['folder'] != p_['folder']:
            continue
        if not s_['s1zip'] or not p_['s1zip']:
            print('   %-14s vs %-14s  per-frame medians have no stage-1 zip: pairs not applicable'
                  % (s_['tag'], p_['tag']))
            continue
        if s_['tag'] in UNTRACKED or p_['tag'] in UNTRACKED:
            print('   %-14s vs %-14s  an untracked capture (section 3w): stack pairs discarded'
                  % (s_['tag'], p_['tag']))
            continue
        sep = sep_deg(s_['j'], p_['j'])
        if sep > SAME_FIELD_DEG:
            print('   %-14s vs %-14s  %.1f deg apart on the sky: NOT the same field, discarded'
                  % (s_['tag'], p_['tag'], sep))
            continue
        for fld, ref, way in ((s_, p_, 'forward'), (p_, s_, 'reversed')):
            if ref['j']['#stars used'] < MIN_REF_STARS:
                print('   %-14s vs %-14s  host has %d stars: not trusted, discarded'
                      % (fld['tag'], ref['tag'], ref['j']['#stars used']))
                continue
            label = 's2%s_%s_vs_%s' % (v, fld['tag'], ref['tag'])
            path = refit_constant(label, fld['s1zip'], ref['res'],
                                  midtime(fld['folder'], fld['name']))
            r = null_from(path, field_rng(label)) if path else None
            if r is None:
                print('   %-14s vs %-14s  constant-only refit gave no usable residuals'
                      % (fld['tag'], ref['tag']))
                continue
            ok = ALT_LO <= fld['alt'] <= ALT_HI
            r.update(field=fld['tag'], ref=ref['tag'], way=way, alt=fld['alt'], sep_deg=sep,
                     used=ok, variant=vname)
            prows.append(r)
            print('   %-14s vs %-14s %-8s %5.2f deg  N=%4d rms %.3f  L base %+.3f  '
                  'L scale %+.3f  L v-deg2 %+.3f  (floor %.3f, 63-star %.3f)%s'
                  % (fld['tag'], ref['tag'], way, fld['alt'], r['n'], r['rms'], r['L_base'],
                     r['L_scale'], r['L_vdeg2'], r['floor'], r['sub63_rms'],
                     '' if ok else '  <- outside the band, not averaged'))
    P = pd.DataFrame(prows)
    if len(P) and P.used.any():
        summarise(P[P.used], 'CONSECUTIVE PAIRS inside %.1f-%.1f deg, %d refit(s):'
                  % (ALT_LO, ALT_HI, int(P.used.sum())))
        deep = P[P.used & (P.n >= N_UNION * 2)]
        if len(deep) and len(deep) < int(P.used.sum()):
            summarise(deep, '   -- of those, the %d with more than %d stars after the cuts:'
                      % (len(deep), N_UNION * 2))
    return Z, P


def by_altitude(Z):
    """The field-to-zenith null against altitude, which is the physical axis.

    The 7.5-12 deg band exists to select fields comparable to the eclipse; it hides the trend.
    Grouping every solved field by altitude shows whether the null grows toward the horizon,
    which is the one thing an atmospheric term ought to do.
    """
    print()
    print('FIELD-TO-ZENITH AGAINST ALTITUDE (every VALID field -- per-frame medians for the '
          'untracked captures, deep stacks otherwise -- band membership ignored)')
    print('   %-18s %7s %8s %9s %9s %11s' % ('altitude', 'fields', 'stars', 'total (")',
                                             'floor (")', 'structure'))
    for lab, lo, hi in (('below %.1f deg' % ALT_LO, 0.0, ALT_LO),
                        ('%.1f-%.1f deg' % (ALT_LO, ALT_HI), ALT_LO, ALT_HI),
                        ('above %.1f deg' % ALT_HI, ALT_HI, 90.0)):
        g = Z[(Z.alt >= lo) & (Z.alt < hi)]
        if not len(g):
            continue
        tot = np.sqrt(np.mean(g.L_scale.values ** 2))
        fl = np.sqrt(np.mean(g.floor.values ** 2))
        print('   %-18s %7d %8d %9.3f %9.3f %11.3f'
              % (lab, len(g), int(g.n.sum()), tot, fl,
                 np.sqrt(max(0.0, tot ** 2 - fl ** 2))))
    print('   (the cell\'s estimator; every field, including those excluded from the averages')
    print('   above for thin solves, because the trend is the point and the floor is shown)')


def main():
    os.makedirs(NULLS, exist_ok=True)
    rng = None
    Zs, Ps = [], []
    for v, vname in VARIANTS:
        Z, P = run_variant(v, vname, rng)
        if Z is not None and len(Z):
            Zs.append(Z)
        if P is not None and len(P):
            Ps.append(P)
        print()
    if Zs:
        Z = pd.concat(Zs)
        Z.to_csv(os.path.join(OUT, 'horizon_nulls_zenith.csv'), index=False)
        # the VALID set, one row per capture: per-frame medians for the untracked captures
        # (section 3w), the deep stack for everything else -- never a smeared stack
        valid = Z[(Z.variant.str.startswith('per-frame'))
                  | (Z.variant.str.startswith('deep') & ~Z.field.isin(UNTRACKED))]
        by_altitude(valid if len(valid) else Z)
    if Ps:
        pd.concat(Ps).to_csv(os.path.join(OUT, 'horizon_nulls_pairs.csv'), index=False)
    print('Leon carries +-0.33 " (v-deg2, horizon nights), Station 1 +-0.11 " (zenith nulls),')
    print('Bruns +-0.15 ".  The field-to-zenith line is what cell 4\'s Method 2 inherits; the')
    print('consecutive pairs are the matrix construction and what a Method 1 would inherit.')


if __name__ == '__main__':
    main()
