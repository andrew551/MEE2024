"""
Build the Hipparcos-2 bright-star catalogue and the HIP label index.

Why Hipparcos: Gaia DR3 has **no entry at all** for the brightest stars -- they saturate
the instrument. 18,430 Hipparcos-2 stars have no Gaia DR3 counterpart, 2,629 of them
brighter than Hp=7. The bundled Tycho catalogue does not close the gap either, because
Tycho-2 moves ~120 very bright stars into a separate Supplement 1 file. Hipparcos-2 has
all of them, with good positions, proper motions and parallaxes.

One download therefore serves two purposes:

  1. the bright fill for plate solving (a starcat catalogue, ~5 MB), and
  2. the label index so plots can say "Vega" or "HIP 91262" instead of a 19-digit Gaia id.

Both are small enough to ship in the wheel.

Magnitudes: Hipparcos measures Hp, not Gaia G. Rather than cite a transformation, this
tool fits one empirically against the ~97,000 stars that have both, and reports the
residual scatter it achieved.

    python tools/build_hipparcos.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from mee2024.starcat import ORIGIN_HIPPARCOS, StarTable      # noqa: E402
from mee2024.starcat import store                            # noqa: E402

#: Hipparcos-2 positions are given at this epoch
HIP_EPOCH = 1991.25

DEFAULT_CATALOGUE_OUT = REPO / 'mee2024' / 'resources' / 'hipparcos2'
DEFAULT_LABELS_OUT = REPO / 'mee2024' / 'resources' / 'star_labels'


def gaia():
    from astroquery.gaia import Gaia
    Gaia.ROW_LIMIT = -1
    return Gaia


def column(table, name, dtype=float):
    """Fetch a column regardless of how the TAP service cased its name."""
    for candidate in (name, name.upper(), name.lower()):
        if candidate in table.colnames:
            return np.array(table[candidate], dtype=dtype)
    raise KeyError(f'{name!r} not among {table.colnames}')


def fetch_hipparcos():
    """The whole of Hipparcos-2: 117,955 rows, small enough for one query."""
    query = """SELECT hip, ra, dec, plx, pm_ra, pm_de, hp_mag, b_v
FROM public.hipparcos_newreduction"""
    return gaia().launch_job_async(query).get_results()


def fetch_gaia_crossmatch():
    """HIP number to Gaia DR3 source_id, from Gaia's own precomputed crossmatch."""
    query = """SELECT original_ext_source_id AS hip, source_id
FROM gaiadr3.hipparcos2_best_neighbour"""
    return gaia().launch_job_async(query).get_results()


def fetch_magnitude_training_set():
    """Stars with both Hp/B-V and a measured Gaia G, for fitting the transformation."""
    query = """SELECT h.hp_mag, h.b_v, g.phot_g_mean_mag AS g
FROM public.hipparcos_newreduction AS h
JOIN gaiadr3.hipparcos2_best_neighbour AS x ON x.original_ext_source_id = h.hip
JOIN gaiadr3.gaia_source AS g ON g.source_id = x.source_id
WHERE h.b_v IS NOT NULL AND g.phot_g_mean_mag IS NOT NULL AND h.hp_mag < 11"""
    return gaia().launch_job_async(query).get_results()


def fit_hp_to_g(training, order=2):
    """Fit G - Hp as a polynomial in B-V. Returns (coeffs, diagnostics).

    A quadratic wins on robust scatter; a cubic overfits the blue end. The plain rms is
    dominated by a tail of variables and of binaries that Gaia resolved but Hipparcos did
    not, so the robust sigma is the number that describes a typical star.
    """
    hp = column(training, 'hp_mag')
    bv = column(training, 'b_v')
    g = column(training, 'g')
    ok = np.isfinite(hp) & np.isfinite(bv) & np.isfinite(g) & (np.abs(bv) < 3)
    hp, bv, g = hp[ok], bv[ok], g[ok]

    coeffs = np.polyfit(bv, g - hp, order)
    resid = (g - hp) - np.polyval(coeffs, bv)
    robust_sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    return coeffs, {
        'n_training': int(len(g)),
        'order': order,
        'coeffs_highest_first': [float(c) for c in coeffs],
        'rms_mag': float(np.std(resid)),
        'robust_sigma_mag': float(robust_sigma),
    }


def estimated_g(hp_mag, b_v, coeffs):
    """Approximate Gaia G from Hp and B-V.

    Where B-V is missing, fall back to the median offset at solar colour so the star still
    gets a usable magnitude for ordering and limiting cuts.
    """
    bv = np.array(b_v, dtype=float)
    missing = ~np.isfinite(bv)
    bv = np.where(missing, 0.65, bv)          # solar-ish default
    return np.array(hp_mag, dtype=float) + np.polyval(coeffs, bv)


def build_catalogue(rows, crossmatch, coeffs, out_dir, diagnostics):
    hip = column(rows, 'hip', np.int64)
    ra = column(rows, 'ra')
    dec = column(rows, 'dec')
    good = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(column(rows, 'hp_mag'))
    dropped = int(np.sum(~good))
    if dropped:
        print(f'  dropping {dropped} row(s) with no usable position or magnitude')

    hip, ra, dec = hip[good], ra[good], dec[good]
    mag = estimated_g(column(rows, 'hp_mag')[good], column(rows, 'b_v')[good], coeffs)
    # Hipparcos pm_ra is mu_alpha* -- it already includes cos(dec), matching Gaia's pmra
    table = StarTable(
        ra=np.radians(ra), dec=np.radians(dec), mag=mag,
        ids=hip,                       # keyed on HIP number, not a Gaia source_id
        epoch=HIP_EPOCH, origin=ORIGIN_HIPPARCOS, band='G_est_from_Hp',
        pmra=column(rows, 'pm_ra')[good],
        pmdec=column(rows, 'pm_de')[good],
        parallax=column(rows, 'plx')[good])
    table = add_neighbour_flags(table)

    matched = len(set(column(crossmatch, 'hip', np.int64).tolist()) & set(hip.tolist()))
    provenance = json.dumps({
        'source': 'public.hipparcos_newreduction (van Leeuwen 2007) via the Gaia archive',
        'position_epoch': HIP_EPOCH,
        'magnitude': 'Gaia G estimated from Hp and B-V; see hp_to_g',
        'hp_to_g': diagnostics,
        'stars_with_gaia_dr3_counterpart': matched,
        'stars_without_gaia_counterpart': int(len(hip)) - matched,
    })
    manifest = store.write_catalogue(
        out_dir, table, name='hipparcos2', catalogue='Hipparcos-2',
        provenance=provenance, magnitude_limit=float(np.max(table.mag)),
        built=time.strftime('%Y-%m-%d'))
    return table, manifest


def add_neighbour_flags(table):
    from scipy.spatial import cKDTree
    vectors = table.get_vectors()
    chord, index = cKDTree(vectors).query(vectors, k=2)
    sep = np.degrees(2 * np.arcsin(np.clip(chord[:, 1] / 2, 0, 1))) * 3600
    out = table.select(slice(None))
    out.nn_sep = sep.astype(np.float32)
    out.nn_mag = table.mag[index[:, 1]].astype(np.float32)
    return out


def build_label_index(crossmatch, hip_rows, out_dir):
    """Sorted-key side tables mapping identifiers to HIP numbers and proper names.

    Sorted arrays plus np.searchsorted rather than a hash table: smaller, memory-mappable,
    and one call resolves a whole field's worth of ids.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hip_xm = column(crossmatch, 'hip', np.int64)
    source_id = column(crossmatch, 'source_id', np.int64)
    order = np.argsort(source_id, kind='stable')
    np.save(out_dir / 'gaia_id.npy', source_id[order])
    np.save(out_dir / 'gaia_hip.npy', hip_xm[order].astype(np.int32))

    all_hip = np.sort(column(hip_rows, 'hip', np.int64))
    np.save(out_dir / 'hip.npy', all_hip.astype(np.int32))

    n_named = write_name_files(out_dir, all_hip)

    manifest = {
        'format': 'mee2024-star-labels',
        'format_version': 1,
        'n_gaia_crossmatches': int(len(source_id)),
        'n_hip': int(len(all_hip)),
        'n_named': n_named,
        'sources': ['gaiadr3.hipparcos2_best_neighbour',
                    'public.hipparcos_newreduction',
                    'IAU Catalog of Star Names (bundled subset)'],
        'built': time.strftime('%Y-%m-%d'),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def write_name_files(out_dir, all_hip):
    """Write names.txt and hip_name_offset.npy for a sorted HIP key array.

    Shared by the full build and by --names-only, so the two cannot drift.
    Returns the number of names that found their HIP row.
    """
    out_dir = Path(out_dir)
    names = load_proper_names()
    name_offsets = np.full(len(all_hip), -1, dtype=np.int32)
    blob = bytearray()
    for hip_number, name in sorted(names.items()):
        position = np.searchsorted(all_hip, hip_number)
        if position < len(all_hip) and all_hip[position] == hip_number:
            name_offsets[position] = len(blob)
            blob += name.encode('utf-8') + b'\n'
    np.save(out_dir / 'hip_name_offset.npy', name_offsets)
    (out_dir / 'names.txt').write_bytes(bytes(blob))
    return int(np.sum(name_offsets >= 0))


def rebuild_label_names(labels_dir):
    """Rewrite only the proper-name files of an already-built label index.

    The name list changes far more often than the catalogue, and unlike the catalogue
    it needs no network: the sorted HIP key array is already on disk. Loading through
    LabelIndex first guarantees the target really is a label index.
    """
    from mee2024.starcat.labels import LabelIndex

    labels_dir = Path(labels_dir)
    index = LabelIndex(labels_dir)
    all_hip = np.array(index.hip, dtype=np.int64)   # a copy, not the mmap
    manifest = dict(index.manifest)
    # Windows refuses to overwrite a file that is still memory-mapped, and the
    # index holds hip_name_offset.npy mapped -- release it before rewriting
    del index
    n_named = write_name_files(labels_dir, all_hip)
    manifest['n_named'] = n_named
    manifest['built'] = time.strftime('%Y-%m-%d')
    (labels_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2),
                                              encoding='utf-8')
    return manifest


def load_proper_names():
    """HIP number to proper name (IAU Catalog of Star Names, bundled subset).

    Every entry here has been verified against the bundled Hipparcos-2 positions:
    the star found at the entry's HIP number sits within 0.3 degrees of the named
    star's J2000 position. That check exists because the original 50-entry list
    carried four names on the wrong stars -- Schedar on gamma Cas (HIP 4427, not
    3179), Castor on Miaplacidus (45238, not 36850), Alphard on Suhail (44816, not
    46390) and Thuban on Kochab (72607, not 68756). A wrong pair here labels a real
    star with another star's name, which is worse than no label at all.
    """
    return {
        677: 'Alpheratz', 746: 'Caph', 1067: 'Algenib', 2081: 'Ankaa',
        3179: 'Schedar', 3419: 'Diphda', 3821: 'Achird', 5447: 'Mirach',
        6686: 'Ruchbah', 7588: 'Achernar', 8886: 'Segin', 8903: 'Sheratan',
        9640: 'Almach', 9884: 'Hamal', 11767: 'Polaris', 13847: 'Acamar',
        14135: 'Menkar', 14576: 'Algol', 15863: 'Mirfak', 16537: 'Ran',
        18543: 'Zaurak', 21421: 'Aldebaran', 23875: 'Cursa', 24436: 'Rigel',
        24608: 'Capella', 25336: 'Bellatrix', 25428: 'Elnath', 25606: 'Nihal',
        25930: 'Mintaka', 25985: 'Arneb', 26207: 'Meissa', 26241: 'Hatysa',
        26311: 'Alnilam', 26634: 'Phact', 26727: 'Alnitak', 27366: 'Saiph',
        27989: 'Betelgeuse', 28360: 'Menkalinan', 30324: 'Mirzam', 30343: 'Tejat',
        30438: 'Canopus', 31681: 'Alhena', 32349: 'Sirius', 33579: 'Adhara',
        34444: 'Wezen', 35904: 'Aludra', 36188: 'Gomeisa', 36850: 'Castor',
        37279: 'Procyon', 37826: 'Pollux', 39429: 'Naos', 41037: 'Avior',
        42913: 'Alsephina', 44816: 'Suhail', 45238: 'Miaplacidus',
        45556: 'Aspidiske', 45941: 'Markeb', 46390: 'Alphard', 49669: 'Regulus',
        50583: 'Algieba', 53910: 'Merak', 54061: 'Dubhe', 54872: 'Zosma',
        57632: 'Denebola', 58001: 'Phecda', 59747: 'Imai', 59774: 'Megrez',
        60260: 'Ginan', 60718: 'Acrux', 61084: 'Gacrux', 61932: 'Muhlifain',
        61941: 'Porrima', 62434: 'Mimosa', 62956: 'Alioth', 63125: 'Cor Caroli',
        63608: 'Vindemiatrix', 65378: 'Mizar', 65474: 'Spica', 67301: 'Alkaid',
        68702: 'Hadar', 68756: 'Thuban', 68933: 'Menkent', 69673: 'Arcturus',
        71683: 'Rigil Kentaurus', 72607: 'Kochab', 72622: 'Zubenelgenubi',
        74785: 'Zubeneschamali', 75097: 'Pherkad', 76267: 'Alphecca',
        77070: 'Unukalhai', 78401: 'Dschubba', 78820: 'Acrab',
        80763: 'Antares', 80816: 'Kornephoros', 82273: 'Atria', 82396: 'Larawag',
        84012: 'Sabik', 84345: 'Rasalgethi', 85670: 'Rastaban', 85696: 'Lesath',
        85927: 'Shaula', 86032: 'Rasalhague', 86228: 'Sargas', 86742: 'Cebalrai',
        87833: 'Eltanin', 88635: 'Alnasl', 89931: 'Kaus Media',
        90185: 'Kaus Australis', 91262: 'Vega', 92420: 'Sheliak', 92855: 'Nunki',
        93506: 'Ascella', 95947: 'Albireo', 97278: 'Tarazed', 97649: 'Altair',
        100453: 'Sadr', 100751: 'Peacock', 102098: 'Deneb', 102488: 'Aljanah',
        105199: 'Alderamin', 106278: 'Sadalsuud', 107315: 'Enif',
        109074: 'Sadalmelik', 109268: 'Alnair', 112122: 'Tiaki', 113368: 'Fomalhaut',
        113881: 'Scheat', 113963: 'Markab',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalogue-out', type=Path, default=DEFAULT_CATALOGUE_OUT)
    ap.add_argument('--labels-out', type=Path, default=DEFAULT_LABELS_OUT)
    ap.add_argument('--order', type=int, default=2,
                    help='polynomial order for the Hp->G colour term (default 2)')
    ap.add_argument('--names-only', action='store_true',
                    help='rewrite the proper-name files of the existing label index '
                         'from load_proper_names(), without any network access')
    args = ap.parse_args()

    if args.names_only:
        manifest = rebuild_label_names(args.labels_out)
        print(f'rewrote names in {args.labels_out}: {manifest["n_named"]} proper names')
        return

    print('fetching Hipparcos-2...')
    rows = fetch_hipparcos()
    print(f'  {len(rows)} rows')

    print('fetching the Gaia DR3 crossmatch...')
    crossmatch = fetch_gaia_crossmatch()
    print(f'  {len(crossmatch)} crossmatches '
          f'({len(rows) - len(crossmatch)} HIP stars have no Gaia counterpart)')

    print('fitting Hp + B-V -> Gaia G...')
    training = fetch_magnitude_training_set()
    coeffs, diagnostics = fit_hp_to_g(training, order=args.order)
    print(f'  trained on {diagnostics["n_training"]} stars: '
          f'robust sigma {diagnostics["robust_sigma_mag"]:.4f} mag, '
          f'rms {diagnostics["rms_mag"]:.4f} mag')
    print('  (the rms tail is variables and binaries Gaia resolved but Hipparcos did not)')

    print(f'writing the catalogue to {args.catalogue_out}...')
    table, manifest = build_catalogue(rows, crossmatch, coeffs, args.catalogue_out,
                                     diagnostics)
    size = sum(p.stat().st_size for p in args.catalogue_out.glob('*.npy'))
    print(f'  {manifest["n_stars"]} stars, {size / 1e6:.1f} MB')

    print(f'writing the label index to {args.labels_out}...')
    label_manifest = build_label_index(crossmatch, rows, args.labels_out)
    size = sum(p.stat().st_size for p in args.labels_out.iterdir() if p.is_file())
    print(f'  {label_manifest["n_hip"]} HIP entries, '
          f'{label_manifest["n_gaia_crossmatches"]} Gaia crossmatches, '
          f'{label_manifest["n_named"]} proper names, {size / 1e6:.1f} MB')

    print('\nverifying the catalogue...')
    problems = store.verify(args.catalogue_out)
    print('OK' if not problems else f'PROBLEMS: {problems}')


if __name__ == '__main__':
    main()
