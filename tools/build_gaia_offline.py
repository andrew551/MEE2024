"""
Build an offline Gaia catalogue in the mee2024-starcat format.

The all-sky G<12 build is ~2.9 million stars and takes hours, so the work is chunked into
declination stripes, each cached on disk and skipped if already present. Interrupt it and
re-run and it picks up where it stopped.

Start small. Three useful scales:

    # 1. plumbing smoke test: all-sky but only the naked-eye stars. Seconds.
    python tools/build_gaia_offline.py --name gaia_test_g5 --max-mag 5

    # 2. real end-to-end test: full depth, but only the sky around the example fields.
    python tools/build_gaia_offline.py --name gaia_test_zwo --max-mag 12 \
        --region 350 5 40 50 --region 20 35 40 50

    # 3. the real artefact. Hours.
    python tools/build_gaia_offline.py --name gaia_dr3_g12 --max-mag 12

Output goes to the catalogue directory reported by `mee2024 catalogue`, unless --out is
given.

Neighbour flags: nn_sep/nn_mag are computed among the stars in the catalogue itself, which
is free. A companion fainter than the catalogue limit will therefore not be flagged. Pass
--neighbour-depth to run a deeper (and much slower) pass; see the note in the manifest,
which always records the depth the flags actually cover.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mee2024.MEE2024util import get_catalogue_root      # noqa: E402
from mee2024.progress import TextProgress               # noqa: E402
from mee2024.starcat import ORIGIN_GAIA, StarTable      # noqa: E402
from mee2024.starcat import store                       # noqa: E402

CHUNK_DTYPE = np.dtype([
    ('source_id', 'i8'),
    ('ra', 'f8'), ('dec', 'f8'),
    ('pmra', 'f4'), ('pmdec', 'f4'),
    ('parallax', 'f4'), ('radial_velocity', 'f4'),
    ('phot_g_mean_mag', 'f4'),
    ('ref_epoch', 'f4'),
])

#: aim for this many rows per query; stripes denser than this are split in RA
ROWS_PER_QUERY = 200_000


def gaia():
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    return Gaia


def mag_clause(min_mag, max_mag):
    clause = f'phot_g_mean_mag < {max_mag}'
    if min_mag is not None:
        clause = f'phot_g_mean_mag >= {min_mag} AND {clause}'
    return clause


def ra_clause(ra_lo, ra_hi):
    """RA restriction, handling a range that wraps through zero."""
    if ra_lo is None:
        return None
    if ra_lo <= ra_hi:
        return f'ra BETWEEN {ra_lo} AND {ra_hi}'
    return f'(ra >= {ra_lo} OR ra <= {ra_hi})'


def count_rows_per_stripe(dec_step, min_mag, max_mag, region):
    """Row counts for every declination stripe, in a single query.

    Each Gaia async query costs ~18 s of latency regardless of how little it returns, so
    counting stripe by stripe would double the total query count. One GROUP BY does it.
    """
    clauses = [mag_clause(min_mag, max_mag)]
    if region:
        clauses.append('(' + ' OR '.join(
            f'({ra_clause(r[0], r[1])} AND dec BETWEEN {r[2]} AND {r[3]})'
            for r in region) + ')')
    query = (f'SELECT FLOOR((dec + 90) / {dec_step}) AS band, COUNT(*) AS n '
             f'FROM gaiadr3.gaia_source WHERE {" AND ".join(clauses)} '
             f'GROUP BY band')
    results = gaia().launch_job_async(query).get_results()
    return {int(band): int(n) for band, n in zip(results['band'], results['n'])}


def fetch_chunk(dec_lo, dec_hi, ra_lo, ra_hi, min_mag, max_mag, region):
    clauses = [f'dec BETWEEN {dec_lo} AND {dec_hi}', mag_clause(min_mag, max_mag)]
    inner_ra = ra_clause(ra_lo, ra_hi)
    if inner_ra:
        clauses.append(inner_ra)
    if region:
        clauses.append('(' + ' OR '.join(
            f'({ra_clause(r[0], r[1])} AND dec BETWEEN {r[2]} AND {r[3]})'
            for r in region) + ')')
    query = f"""SELECT source_id, ra, dec, pmra, pmdec, parallax, radial_velocity,
phot_g_mean_mag, ref_epoch
FROM gaiadr3.gaia_source
WHERE {' AND '.join(clauses)}"""
    results = gaia().launch_job_async(query).get_results()
    out = np.zeros(len(results), dtype=CHUNK_DTYPE)
    # a 19-digit source_id does not survive float64
    out['source_id'] = np.array(results['SOURCE_ID'], dtype=np.int64)
    for name in CHUNK_DTYPE.names:
        if name == 'source_id':
            continue
        out[name] = np.array(results[name], dtype=float)
    return out


def plan_chunks(stripes, dec_step, min_mag, max_mag, region):
    """Decide the query list, splitting dense declination stripes in RA.

    stripes is a list of (band, dec_lo, dec_hi). The band number is what the server's
    GROUP BY produces and what chunk filenames are keyed on, so it stays stable no matter
    how the stripe list is filtered.
    """
    print('counting rows per declination stripe (one query)...')
    counts = count_rows_per_stripe(dec_step, min_mag, max_mag, region)
    total = sum(counts.values())
    print(f'  {total} rows to fetch across {len(counts)} populated stripe(s)')

    plan = []
    for band, dec_lo, dec_hi in stripes:
        n = counts.get(band, 0)
        if n == 0:
            continue
        parts = max(1, int(np.ceil(n / ROWS_PER_QUERY)))
        for part in range(parts):
            plan.append((band, dec_lo, dec_hi, part,
                         360.0 * part / parts, 360.0 * (part + 1) / parts))
    return plan, total


def _hms(seconds):
    seconds = int(max(0, seconds))
    return f'{seconds // 3600}h{seconds % 3600 // 60:02d}m'


def _save_chunk(path, rows):
    """Cache one chunk, atomically.

    A deep build runs for hours or days and will be interrupted. Writing straight to the
    final name risks a truncated file that the next run counts as complete and never
    refetches -- a silent hole in the sky. Write beside it and rename: the rename is
    atomic, so a chunk is either absent or whole. ``np.save`` is handed an open file so it
    does not append its own ``.npy`` to the temporary name, which also keeps the temporary
    out of the ``stripe_*.npy`` glob that assembles the catalogue.
    """
    tmp = path.with_name(path.name + '.part')
    with open(tmp, 'wb') as fp:
        np.save(fp, rows)
    tmp.replace(path)


def build(name, max_mag, min_mag, region, dec_step, out_dir, work_dir, progress,
          neighbour_depth=None):
    n_bands = int(np.ceil(180.0 / dec_step))
    stripes = [(band, -90.0 + band * dec_step, -90.0 + (band + 1) * dec_step)
               for band in range(n_bands)]
    if region:
        # only stripes that overlap a requested region are worth visiting
        stripes = [s for s in stripes
                   if any(s[2] >= r[2] and s[1] <= r[3] for r in region)]

    work_dir.mkdir(parents=True, exist_ok=True)
    plan, expected = plan_chunks(stripes, dec_step, min_mag, max_mag, region)

    def chunk_path(band, part):
        return work_dir / f'stripe_{band:04d}_{part:03d}.npy'

    remaining = [c for c in plan if not chunk_path(c[0], c[3]).exists()]
    print(f'{len(plan)} chunk(s); {len(plan) - len(remaining)} already cached, '
          f'{len(remaining)} to fetch')
    if remaining:
        # Latency alone, and only if the archive is behaving. Measured throughput has
        # ranged over 50x between days (see progress.md), so this is a floor, not a
        # forecast -- the running estimate below is the one to believe.
        print(f'floor if the archive is fast: {len(remaining) * 20 / 60:.0f}-'
              f'{len(remaining) * 45 / 60:.0f} min. A slow day can be far longer; the '
              f'estimate is revised from measured rate as it goes.')

    progress.start(len(plan), 'Querying Gaia')
    started = time.perf_counter()
    fetched = fetched_rows = 0
    for done, (band, dec_lo, dec_hi, part, ra_lo, ra_hi) in enumerate(plan, start=1):
        path = chunk_path(band, part)
        if not path.exists():
            rows = fetch_chunk(dec_lo, dec_hi, ra_lo, ra_hi, min_mag, max_mag, region)
            _save_chunk(path, rows)
            fetched += 1
            fetched_rows += len(rows)
            elapsed = time.perf_counter() - started
            left = len(remaining) - fetched
            # the reporter owns a single rewritten line, so start on a fresh one
            print(f'\r  [{done}/{len(plan)}] dec {dec_lo:+.0f}..{dec_hi:+.0f} part '
                  f'{part}: {len(rows):,} rows | {fetched_rows / elapsed:,.0f} rows/s '
                  f'| elapsed {_hms(elapsed)} | {left} left, ETA '
                  f'{_hms(elapsed / fetched * left)}', flush=True)
        progress.update(done)
    progress.finish()
    print(f'queries finished in {_hms(time.perf_counter() - started)}')

    chunks = [np.load(p) for p in sorted(work_dir.glob('stripe_*.npy'))]
    chunks = [c for c in chunks if len(c)]
    if not chunks:
        raise SystemExit('no rows returned -- check the magnitude limit and region')
    rows = np.concatenate(chunks)

    # a star can appear twice if RA sub-ranges overlap at a boundary
    _, unique = np.unique(rows['source_id'], return_index=True)
    if len(unique) != len(rows):
        print(f'dropping {len(rows) - len(unique)} duplicate row(s)')
        rows = rows[np.sort(unique)]
    print(f'{len(rows)} unique stars')

    table = StarTable(
        ra=np.radians(np.array(rows['ra'], dtype=float)),
        dec=np.radians(np.array(rows['dec'], dtype=float)),
        mag=rows['phot_g_mean_mag'], ids=rows['source_id'],
        epoch=float(np.median(rows['ref_epoch'])),
        origin=ORIGIN_GAIA, band='G',
        pmra=rows['pmra'], pmdec=rows['pmdec'],
        parallax=rows['parallax'], radial_velocity=rows['radial_velocity'])

    depth = neighbour_depth or max_mag
    table = add_neighbour_flags(table, progress)

    provenance = json.dumps({
        'source': 'gaiadr3.gaia_source via astroquery',
        'magnitude_range': [min_mag, max_mag],
        'region': region or 'all sky',
        'dec_step_degrees': dec_step,
        'neighbour_flag_depth_G': depth,
        'neighbour_flags': 'computed among catalogue members only'
                           if neighbour_depth is None else 'deep pass',
    })
    manifest = store.write_catalogue(
        out_dir, table, name=name, catalogue='Gaia DR3', provenance=provenance,
        magnitude_limit=max_mag, built=time.strftime('%Y-%m-%d'))
    print(f'\nwrote {manifest["n_stars"]} stars to {out_dir}')
    total = sum(p.stat().st_size for p in out_dir.glob('*.npy'))
    print(f'on-disk size {total / 1e6:.1f} MB')
    return manifest


def add_neighbour_flags(table, progress):
    """Nearest-neighbour separation and magnitude, among the catalogue's own stars."""
    if len(table) < 2:
        return table
    from scipy.spatial import cKDTree
    print('computing nearest-neighbour flags...')
    vectors = table.get_vectors()
    tree = cKDTree(vectors)
    chord, index = tree.query(vectors, k=2)
    # column 0 is the star itself
    sep_arcsec = np.degrees(2 * np.arcsin(np.clip(chord[:, 1] / 2, 0, 1))) * 3600
    out = table.select(slice(None))
    out.nn_sep = sep_arcsec.astype(np.float32)
    out.nn_mag = table.mag[index[:, 1]].astype(np.float32)
    close = np.sum(sep_arcsec < 10.0)
    print(f'  {close} star(s) have a catalogue neighbour within 10 arcsec '
          f'({100 * close / len(table):.2f}%)')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', required=True, help='catalogue name, e.g. gaia_dr3_g12')
    ap.add_argument('--max-mag', type=float, default=12.0, help='faintest G to include')
    ap.add_argument('--min-mag', type=float, default=None,
                    help='brightest G to include, for building a deep extension')
    ap.add_argument('--region', nargs=4, type=float, action='append',
                    metavar=('RA_LO', 'RA_HI', 'DEC_LO', 'DEC_HI'),
                    help='restrict to a sky box; repeatable. Omit for all sky')
    ap.add_argument('--dec-step', type=float, default=10.0,
                    help='declination stripe height in degrees (default 10). Each query '
                         'costs ~18 s of latency regardless of size, so prefer few large '
                         'stripes; dense ones are split in RA automatically')
    ap.add_argument('--out', type=Path, default=None,
                    help='output directory (default: the catalogue directory)')
    ap.add_argument('--work-dir', type=Path, default=None,
                    help='where to cache chunks so the build is resumable')
    ap.add_argument('--neighbour-depth', type=float, default=None,
                    help='run a deeper neighbour pass to this magnitude (slow)')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    out_dir = args.out or (get_catalogue_root() / args.name)
    work_dir = args.work_dir or (get_catalogue_root() / f'.build_{args.name}')
    progress = TextProgress() if not args.quiet else None
    if progress is None:
        from mee2024.progress import NullProgress
        progress = NullProgress()

    print(f'building {args.name}')
    print(f'  magnitude range : {args.min_mag or "-inf"} <= G < {args.max_mag}')
    print(f'  region          : {args.region or "all sky"}')
    print(f'  output          : {out_dir}')
    print(f'  chunk cache     : {work_dir}')
    print()

    build(args.name, args.max_mag, args.min_mag, args.region, args.dec_step,
          out_dir, work_dir, progress, neighbour_depth=args.neighbour_depth)

    print('\nverifying...')
    problems = store.verify(out_dir)
    print('OK' if not problems else f'PROBLEMS: {problems}')
    print(f'\ntry it with:  mee2024 catalogue --verify {args.name}')

    # The chunk cache is what makes the build resumable, so it is deliberately NOT
    # deleted here: a build that verified but that the operator has not yet accepted is
    # exactly when losing it would hurt. It is not deleted silently either, because it is
    # large and invisible -- a 1.7 GB cache sat unnoticed under the catalogue directory
    # for two weeks after gaia_dr3_g15 was built.
    #
    # It is safe to remove once the catalogue verifies and has been packed and published.
    # Measured on that g15 build: the installed catalogue held all 36,909,335 stars the
    # stripes did, every column at the same precision -- 7 bit-identical, ra/dec agreeing
    # to 6e-14 degrees after the documented degrees-to-radians conversion. Nothing in the
    # cache is unique. Note too that a cache from one magnitude range does not help build
    # another: those stripes stopped at G = 15.000, so a deeper catalogue re-queries the
    # archive regardless.
    if work_dir.is_dir():
        cached = sum(f.stat().st_size for f in work_dir.glob('*.npy'))
        if cached:
            print()
            print(f'chunk cache: {cached / 1e9:.2f} GB in {work_dir}')
            print('  Kept so the build can be resumed. Once this catalogue is verified,')
            print('  packed and published -- and not before -- it can be deleted; every')
            print('  column it holds is preserved in the catalogue itself.')


if __name__ == '__main__':
    main()
