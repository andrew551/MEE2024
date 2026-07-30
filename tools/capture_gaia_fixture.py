"""
Capture a Gaia response for one of the example fields as a test fixture, and measure
online-vs-local epoch propagation.

The fixture stores raw Gaia DR3 columns at their reference epoch (2016.0), unpropagated.
That single artefact serves three purposes:

  1. a fast, offline, deterministic stage-2 regression test,
  2. a miniature offline catalogue for exercising the GaiaOffline provider,
  3. the epoch-propagation gate -- comparing locally propagated positions against Gaia's
     server-side ESDC_EPOCH_PROP_POS, which is the number that decides whether an offline
     catalogue can reproduce today's results.

Usage:
    python tools/capture_gaia_fixture.py zwo3_zenith --date 2023-10-29
    python tools/capture_gaia_fixture.py --all --date 2023-10-29
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use('Agg')

from mee2024 import platesolve_triangle, transforms          # noqa: E402
from mee2024.MEE2024util import date_string_to_float, get_bbox  # noqa: E402
from mee2024.config import get_default_options               # noqa: E402

FIELDS_DIR = REPO / 'tests' / 'data' / 'fields'
GAIA_DIR = REPO / 'tests' / 'data' / 'gaia'

# The columns we persist. ra/dec are Gaia's own ref_epoch positions (2016.0 for DR3).
DTYPE = np.dtype([
    ('source_id', 'i8'),
    ('ra', 'f8'), ('dec', 'f8'),
    ('pmra', 'f4'), ('pmdec', 'f4'),
    ('parallax', 'f4'), ('radial_velocity', 'f4'),
    ('phot_g_mean_mag', 'f4'),
    ('ref_epoch', 'f4'),
])


def field_bbox(field_name, options):
    """The RA/Dec box that stage 2 would query for this field.

    Reproduces match_and_fit_distortion: platesolve, project the image corners onto the
    sky, take the bounding box.
    """
    field = json.loads((FIELDS_DIR / f'{field_name}.json').read_text(encoding='utf-8'))
    centroids = np.array(field['centroids'])
    image_size = field['img_shape']
    result = platesolve_triangle.platesolve(centroids, tuple(image_size), options=options)
    if not result['success']:
        raise SystemExit(f'{field_name}: platesolve failed, cannot determine the field')
    corners = transforms.to_polar(transforms.linear_transform(
        result['x'],
        np.array([[0, 0], [image_size[0] - 1., image_size[1] - 1.],
                  [0, image_size[1] - 1.], [image_size[0] - 1., 0]])
        - np.array([image_size[0] / 2, image_size[1] / 2])))
    return get_bbox(corners), result


def query_raw(ra_range, dec_range, max_mag):
    """Gaia columns at their reference epoch, with no server-side propagation."""
    from astroquery.gaia import Gaia
    query = f"""SELECT source_id, ra, dec, pmra, pmdec, parallax, radial_velocity,
phot_g_mean_mag, ref_epoch
FROM gaiadr3.gaia_source
WHERE ra BETWEEN {ra_range[0]} AND {ra_range[1]}
AND dec BETWEEN {dec_range[0]} AND {dec_range[1]}
AND phot_g_mean_mag < {max_mag}"""
    results = Gaia.launch_job_async(query).get_results()
    out = np.zeros(len(results), dtype=DTYPE)
    # source_id must stay integral: a 19-digit Gaia id does not survive float64
    out['source_id'] = np.array(results['SOURCE_ID'], dtype=np.int64)
    for name in DTYPE.names:
        if name == 'source_id':
            continue
        out[name] = np.array(results[name], dtype=float)
    return out


def propagate_locally(rows, target_epoch, regularize_parallax_min=1.0,
                      use_radial_velocity=True):
    """Propagate ref_epoch positions to target_epoch with astropy apply_space_motion."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord, Distance
    from astropy.time import Time

    parallax = np.array(rows['parallax'], dtype=float)
    parallax[np.isnan(parallax)] = 0.0
    parallax[parallax < regularize_parallax_min] = regularize_parallax_min

    pmra = np.nan_to_num(np.array(rows['pmra'], dtype=float))
    pmdec = np.nan_to_num(np.array(rows['pmdec'], dtype=float))
    rv = np.nan_to_num(np.array(rows['radial_velocity'], dtype=float))
    if not use_radial_velocity:
        rv = np.zeros_like(rv)

    coord = SkyCoord(
        ra=rows['ra'] * u.deg, dec=rows['dec'] * u.deg,
        distance=Distance(parallax=parallax * u.mas),
        pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
        radial_velocity=rv * u.km / u.s,
        obstime=Time(rows['ref_epoch'].astype(float), format='jyear', scale='tcb'))
    moved = coord.apply_space_motion(Time(target_epoch, format='jyear', scale='tcb'))
    return moved.ra.deg, moved.dec.deg


def compare(rows, ra_online, dec_online, target_epoch, label, **kwargs):
    ra_local, dec_local = propagate_locally(rows, target_epoch, **kwargs)
    dra = (ra_local - ra_online) * np.cos(np.radians(dec_online)) * 3.6e6   # mas
    ddec = (dec_local - dec_online) * 3.6e6
    sep = np.hypot(dra, ddec)
    print(f'  {label:44} median {np.median(sep):7.3f} mas   '
          f'p90 {np.percentile(sep, 90):8.3f}   max {np.max(sep):9.3f}')
    return sep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('fields', nargs='*', help='field fixture names (without .json)')
    ap.add_argument('--all', action='store_true', help='every field in tests/data/fields')
    ap.add_argument('--date', required=True, help='observation date, YYYY-MM-DD')
    ap.add_argument('--max-mag', type=float, default=12.0)
    args = ap.parse_args()

    names = sorted(p.stem for p in FIELDS_DIR.glob('*.json')) if args.all else args.fields
    if not names:
        ap.error('name at least one field, or pass --all')

    options = get_default_options()
    options.update(flag_display=False, flag_display2=False, flag_display3=False)
    target_epoch = date_string_to_float(args.date)
    GAIA_DIR.mkdir(parents=True, exist_ok=True)
    print(f'target epoch: {target_epoch:.6f}  (from {args.date})\n')

    from mee2024 import gaia_search

    for name in names:
        (ra_range, dec_range), solution = field_bbox(name, options)
        print(f'=== {name} ===')
        print(f'  field: RA {ra_range[0]:.4f}..{ra_range[1]:.4f}  '
              f'Dec {dec_range[0]:.4f}..{dec_range[1]:.4f}')

        rows = query_raw(ra_range, dec_range, args.max_mag)
        out = GAIA_DIR / f'{name}.npy'
        np.save(out, rows)
        print(f'  saved {len(rows)} rows -> {out.relative_to(REPO)} '
              f'({out.stat().st_size / 1024:.1f} KB)')

        # the online reference: server-side ESDC_EPOCH_PROP_POS, what the pipeline uses today
        online = gaia_search.select_in_box(target_epoch, ra_range, dec_range, args.max_mag)
        by_id = {int(i): k for k, i in enumerate(np.array(online['SOURCE_ID']))}
        keep = np.array([i for i, sid in enumerate(rows['source_id']) if int(sid) in by_id])
        order = np.array([by_id[int(rows['source_id'][i])] for i in keep])
        matched = rows[keep]
        ra_online = np.array(online['COORD1'])[order]
        dec_online = np.array(online['COORD2'])[order]
        print(f'  matched {len(matched)} of {len(rows)} raw rows against the online query'
              f' (online returned {len(online)}; it applies a G>3 floor)')

        print('  local propagation vs Gaia ESDC_EPOCH_PROP_POS:')
        compare(matched, ra_online, dec_online, target_epoch,
                'with RV, parallax clamped to >=1 mas (current)')
        compare(matched, ra_online, dec_online, target_epoch,
                'with RV, parallax unclamped', regularize_parallax_min=0.0)
        compare(matched, ra_online, dec_online, target_epoch,
                'no RV, parallax clamped', use_radial_velocity=False)
        print()


if __name__ == '__main__':
    main()
