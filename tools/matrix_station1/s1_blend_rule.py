"""A double-star cut on the companion's brightness as well as its distance (Douglas, 2026-09-09).

`s1_double_cutoff.py` measured the effect the cut exists to remove: on 1712 zenith stars the
centroid is pulled toward a close companion by +0.39 to +0.53 ", and the pull follows the
flux-weighted prediction

    b = sep x f2/(f1+f2),    f2/f1 = 10^(-0.4 dG)

with slope +1.00 inside 4 ". Douglas: then the cut should use the companion's magnitude, not only
its distance. It should, and b is the quantity to cut on -- it IS the bias, in arcsec, which a
radius alone can only proxy.

Two things are calibrated here, both on the zenith sample because it is the one with statistics:

  1. the THRESHOLD on b. Too tight and stars are thrown away for nothing; too loose and bias
     stays in. Measured as the bias left in the kept sample against the stars it costs.
  2. the MERGE RADIUS, beyond which the model does not hold because the two footprints have
     separated and there is nothing to correct. Magnitude enters here too: a bright star's wings
     clear the detection threshold further out, so its footprint merges further out.

Then the rule is applied to every cell so its cost is known before anything is changed. Nothing
here touches the pipeline: F31 must be fixed first (the offline cut has never acted at all), and
the record does not move without Douglas' agreement.

  .venv/Scripts/python.exe tools/matrix_station1/s1_blend_rule.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from mee2024.MEE2024util import get_data_root
from tools.analysis_window import WINDOWS

SEARCH_AS = 60.0
REC1 = r"F:/MEE_output/station1_record"
POOLED = os.path.join(REC1, 'pooled_fit', 'twopass')
ZEN = os.path.join(REC1, 'zenith_nulls_corr', '**', 'CATALOGUE_MATCHED_ERRORS.csv')
NX, NY, PS1 = 9576, 6388, 1.84847

_CAT = {}


def catalogue():
    """ra, dec (degrees), G, source_id. Loaded on demand so the module imports without one."""
    if not _CAT:
        C = os.path.join(get_data_root(), 'catalogues', 'gaia_dr3_g15')
        _CAT['ra'] = np.degrees(np.asarray(np.load(os.path.join(C, 'ra.npy'), mmap_mode='r')))
        _CAT['dec'] = np.degrees(np.asarray(np.load(os.path.join(C, 'dec.npy'), mmap_mode='r')))
        _CAT['mag'] = np.asarray(np.load(os.path.join(C, 'mag.npy'), mmap_mode='r'))
        _CAT['sid'] = np.asarray(np.load(os.path.join(C, 'source_id.npy'), mmap_mode='r'))
    return _CAT['ra'], _CAT['dec'], _CAT['mag'], _CAT['sid']


def neighbour(ra0, dec0, own_id):
    """Nearest catalogue neighbour inside SEARCH_AS: (sep ", dRA*cos ", dDec ", its G) or None."""
    ra_c, dec_c, mag_c, sid_c = catalogue()
    d = SEARCH_AS / 3600.0
    box = d / max(np.cos(np.radians(dec0)), 1e-6)
    idx = np.nonzero((np.abs(dec_c - dec0) < d * 1.5) & (np.abs(ra_c - ra0) < box * 1.5))[0]
    if not len(idx):
        return None
    dx = (ra_c[idx] - ra0) * np.cos(np.radians(dec0)) * 3600.0
    dy = (dec_c[idx] - dec0) * 3600.0
    sep = np.hypot(dx, dy)
    if own_id is None:
        # A guessing fallback lived here (drop anything inside 0.5 ") and it was worse than an
        # error: it turned every unreadable id into a plausible false double at ~0.6-1.0 ", which
        # is exactly where a star's own catalogue entry sits once proper motion is applied.
        raise ValueError('no usable source id: the self-match cannot be excluded')
    keep = (sep > 1e-3) & (sep < SEARCH_AS) & (sid_c[idx] != own_id)
    if not keep.any():
        return None
    j = np.argmin(sep[keep])
    return float(sep[keep][j]), float(dx[keep][j]), float(dy[keep][j]), float(mag_c[idx][keep][j])


def blend_shift(sep_as, mag_primary, mag_companion):
    """The flux-weighted centroid shift of an unresolved pair, in arcsec. Slope 1.00, measured."""
    f = 10.0 ** (-0.4 * (mag_companion - mag_primary))
    return sep_as * f / (1.0 + f)


def own_id(key):
    """The star's own Gaia source id, however the table spells it.

    Leon's union table stores a bare integer where the others store 'gaia:<id>'. Returning None
    for it, as the first version did, disabled the self-match guard: the star matched its own
    catalogue entry (~0.5 " away, since the stored position carries proper motion to a different
    epoch) and all 42 of Leon's stars were reported as doubles of themselves.

    A PLAIN PYTHON INT, and never np.uint64 or int(float(...)). Both were tried here and both
    break the guard, measured rather than reasoned: with `int(float(id))` the id changes outright
    for 73 % of cell 2's stars (2577061092921353984 -> ...354240, since 19 digits need 62 bits and
    a double carries 53) and 145 of 192 stars were reported as their own companion; with
    `np.uint64` against the int64 `source_id.npy`, Leon came out 13 of 42 where a plain Python int
    gives 0 of 42, which is the answer an independent check confirms. The mechanism of the uint64
    case was not run to ground -- a two-element reproduction did not fail -- so the rule is simply
    to compare like with like and to check the count against a second implementation.
    """
    s = str(key)
    s = s[5:] if s.startswith('gaia:') else s
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def flag(df, key, mag, ra, dec, radius=10.0, T=0.05, merge=12.0):
    """Which stars each rule removes. `df` must carry the id column as TEXT.

    pandas' iterrows() gives back a Series of one common dtype, so on an all-numeric frame an
    int64 Gaia id arrives as float64 -- 618969525396598656 becomes 6.189695253965987e+17, and
    every id then fails to parse. That is how Leon came out 13 of 42 instead of 0: with the id
    unreadable, each star matched its own catalogue entry, which sits 0.6-1.0 " away once proper
    motion is applied. Casting the column to str before iterating is the fix.
    """
    df = df.copy()
    df[key] = df[key].map(lambda v: '%d' % v if isinstance(v, (int, np.integer)) else str(v))
    rad, bl, detail = set(), set(), []
    for _, r in df.iterrows():
        n = neighbour(float(r[ra]), float(r[dec]), own_id(r[key]))
        if not n:
            continue
        s, _dx, _dy, gm = n
        b = blend_shift(s, float(r[mag]), gm)
        if s < radius:
            rad.add(r[key])
        if s < merge and b > T:
            bl.add(r[key])
        if (s < radius) != (s < merge and b > T):
            detail.append((float(r[mag]), s, gm, b, 'radius only' if s < radius else 'blend only'))
    return rad, bl, detail


# ---------------------------------------------------------------- the zenith sample
frames = []
for f in sorted(glob.glob(ZEN, recursive=True)):
    z = pd.read_csv(f)
    cd = np.cos(np.radians(z['DEC(catalog)'].values))
    frames.append(pd.DataFrame(dict(
        ID=z.ID.values, magV=z.magV.values, ra=z['RA(catalog)'].values, dec=z['DEC(catalog)'].values,
        rx=(z['RA(obs)'].values - z['RA(catalog)'].values) * cd * 3600.0,
        ry=(z['DEC(obs)'].values - z['DEC(catalog)'].values) * 3600.0)))
Z = pd.concat(frames, ignore_index=True)
RMS = float(np.sqrt(np.mean(Z.rx ** 2 + Z.ry ** 2)))
print('zenith sample: %d stars, %d rows, residual rms %.3f "' % (Z.ID.nunique(), len(Z), RMS))

info = {}
for _, r in Z.drop_duplicates('ID')[['ID', 'ra', 'dec']].iterrows():
    n = neighbour(float(r.ra), float(r['dec']), own_id(r.ID))
    if n:
        info[r.ID] = n
Z = Z[Z.ID.isin(info)].copy()
sep = np.array([info[i][0] for i in Z.ID]); dra = np.array([info[i][1] for i in Z.ID])
ddec = np.array([info[i][2] for i in Z.ID]); cg = np.array([info[i][3] for i in Z.ID])
nn = np.hypot(dra, ddec)
Z['sep'] = sep; Z['comp_G'] = cg
Z['proj'] = Z.rx.values * dra / nn + Z.ry.values * ddec / nn
g = Z.groupby('ID').agg(sep=('sep', 'first'), comp_G=('comp_G', 'first'), magV=('magV', 'first'),
                        proj=('proj', 'mean'), n=('proj', 'size')).reset_index()
g['b'] = blend_shift(g.sep.values, g.magV.values, g.comp_G.values)
print('%d of them have a neighbour inside %.0f "' % (len(g), SEARCH_AS))

# ---------------------------------------------------------------- 1. the merge radius, by brightness
print()
print('1. THE MERGE RADIUS. Beyond it the footprints have separated and the model does not hold.')
print('   Split by the PRIMARY\'s brightness, since a bright star\'s wings clear threshold further out:')
print('   %-18s %-14s %6s %16s' % ('primary', 'separation', 'stars', 'toward it (")'))
for lab, lo, hi in (('brighter than G 10', -99, 10.0), ('G 10 and fainter', 10.0, 99)):
    for a, b in ((0, 10), (10, 14), (14, 25), (25, 60)):
        m = (g.magV >= lo) & (g.magV < hi) & (g.sep >= a) & (g.sep < b)
        if m.sum() < 3:
            print('   %-18s %-14s %6d   (too few)' % (lab, '%g-%g "' % (a, b), int(m.sum()))); continue
        v = g.proj[m].values
        print('   %-18s %-14s %6d %+9.3f \u00b1 %.3f'
              % (lab, '%g-%g "' % (a, b), int(m.sum()), v.mean(), np.std(v, ddof=1) / np.sqrt(len(v))))

# ---------------------------------------------------------------- 2. the threshold on b
print()
print('2. THE THRESHOLD on the predicted shift b, inside a 12 " merge radius.')
print('   "left behind" is the mean pull still in the KEPT stars -- what the cut fails to remove.')
print('   %-12s %8s %8s %18s %14s' % ('b >', 'cut', 'kept', 'left behind (")', 'as % of rms'))
near = g[g.sep < 12]
for T in (0.00, 0.02, 0.05, 0.10, 0.20, 0.40, 99):
    cut = near[near.b > T]
    kept = near[near.b <= T]
    left = kept.proj.mean() if len(kept) else 0.0
    print('   %-12s %8d %8d %+13.3f \u00b1 %.3f %12.0f %%'
          % ('%.2f "' % T if T < 99 else 'nothing cut', len(cut), len(kept), left,
             (np.std(kept.proj, ddof=1) / np.sqrt(len(kept))) if len(kept) > 1 else np.nan,
             100 * abs(left) / RMS))
print('   (a star with a neighbour beyond 12 " is never cut: %d stars, mean pull %+.3f ")'
      % (int((g.sep >= 12).sum()), g.proj[g.sep >= 12].mean()))

# ---------------------------------------------------------------- 3. what it costs each cell
print()
print('3. WHAT THE RULE COSTS EACH CELL.  radius rule = the present 10 "; blend rule = b > 0.05 "')
print('   within 12 ", both against the same fitted-star list.')
print('   %-24s %7s %14s %14s %28s' % ('cell', 'stars', 'radius rule', 'blend rule', 'differences'))


cells = []
t1 = pd.read_csv(os.path.join(POOLED, 'pooled_rows.csv'))
s1 = t1.drop_duplicates('key')[['key', 'ra', 'dec', 'magV']]
cells.append(('Mexico 2024 Station 1', s1, 'key', 'magV', 'ra', 'dec'))
p2 = r"F:/MEE_output/station2_transfer/charts/station2_star_table.csv"
if os.path.exists(p2):
    t2 = pd.read_csv(p2)
    if 'ra' in t2.columns:
        cells.append(('Mexico 2024 Station 2', t2.drop_duplicates('ID')[['ID', 'ra', 'dec', 'magV']],
                      'ID', 'magV', 'ra', 'dec'))
pl = r"F:/MEE_output/RECORD/leon2026/leon_union_star_table.csv"
if os.path.exists(pl):
    tl = pd.read_csv(pl)
    cells.append(('Leon 2026', tl.drop_duplicates('gaia_id')[['gaia_id', 'ra_cat', 'dec_cat', 'mag']],
                  'gaia_id', 'mag', 'ra_cat', 'dec_cat'))
pb = glob.glob(r"F:/MEE_output/matrix_bruns2017_brunsmethod/master062/"
               r"**/CATALOGUE_MATCHED_ERRORS.csv", recursive=True)
if pb:
    tb = pd.read_csv(pb[0]).rename(columns={'RA(catalog)': 'ra', 'DEC(catalog)': 'dec'})
    # master062's matched stars, which is a wider list than the record's 27: those 27 give ZERO
    # under either rule (checked against the record star table by pixel match, 2026-09-09).
    cells.append(('Bruns 2017 (master062 match)', tb.drop_duplicates('ID')[['ID', 'ra', 'dec', 'magV']],
                  'ID', 'magV', 'ra', 'dec'))
alldetail = {}
for name, df, key, mag, ra, dec in cells:
    rad, bl, detail = flag(df, key, mag, ra, dec)
    alldetail[name] = detail
    print('   %-24s %7d %14d %14d %28s'
          % (name, len(df), len(rad), len(bl),
             '%d radius-only, %d blend-only' % (len(rad - bl), len(bl - rad))))
for name, detail in alldetail.items():
    if detail:
        print()
        print('   %s -- where the two rules disagree:' % name)
        for mg, s, gm, b, which in sorted(detail, key=lambda x: x[1]):
            print('      G %5.2f, companion G %5.2f at %5.2f "  predicted bias %.3f "   %s'
                  % (mg, gm, s, b, which))

# ---------------------------------------------------------------- 4. L under the blend rule
print()
print('4. CELL 2 UNDER EACH RULE (pooled Method 2, the record estimator).')
blocks = sorted(set(t1.block))


def solve(dd):
    n = len(dd); Zc = np.zeros(n)
    xs, ys = (dd.px.values - NX / 2) * PS1, (dd.py.values - NY / 2) * PS1
    ux, uy = dd.rx.values / dd.R.values, dd.ry.values / dd.R.values
    cx, cy = [], []
    for b in blocks:
        m = (dd.block.values == b).astype(float)
        cx += [m, Zc, -ys * m]; cy += [Zc, m, xs * m]
    cx += [xs, ux * dd.RS.values / dd.R.values]
    cy += [ys, uy * dd.RS.values / dd.R.values]
    A = np.vstack([np.column_stack(cx), np.column_stack(cy)])
    y = np.concatenate([dd.dx.values, dd.dy.values])
    return np.linalg.lstsq(A, y, rcond=None)[0][-1]


def boot(dd, seed=3, draws=600):
    rng = np.random.default_rng(seed); ids = dd.key.unique(); out = []
    for _ in range(draws):
        pick = rng.choice(ids, size=len(ids), replace=True)
        s = pd.concat([dd[dd.key == i] for i in pick], ignore_index=True)
        if len(set(s.block)) < len(blocks):
            continue
        try:
            out.append(solve(s))
        except Exception:
            pass
    return float(np.std(out, ddof=1))


rad1, bl1, _ = flag(s1, 'key', 'magV', 'ra', 'dec')
print('   %-34s %7s %7s %10s %8s' % ('', 'stars', 'obs', 'L (arcsec)', 'sigma'))
for lab, drop in (('as the record stands', set()),
                  ('the present 10 " radius rule', rad1),
                  ('the blend rule, b > 0.05 " in 12 "', bl1)):
    k = t1[~t1.key.isin(drop)]
    print('   %-34s %7d %7d %10.3f %8.3f'
          % (lab, k.key.nunique(), len(k), solve(k), boot(k)))
print()
print('   Nothing is applied. F31 must be fixed before any cut acts, and the record does not move')
print('   without agreement.')
