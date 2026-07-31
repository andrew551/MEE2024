"""
A/B bench for the plate solvers: v1 (production) vs v2 (rebuild), on synthetic fields.

Every v2 stage (docs/PLATESOLVER_V2_DESIGN.md) is gated on this harness: run the frozen
corpus, compare against the previous stage's committed JSON, and proceed only if the
wrong-solve rate stays zero, junk fields all reject, both real fields solve, and the
stage's target metric moved as predicted.

    python tools/solver_bench.py run --solver v1 --out docs/bench/s0_baseline.json
    python tools/solver_bench.py run --solver v2 --db patdb_g12_t17 --out docs/bench/s1.json
    python tools/solver_bench.py compare docs/bench/s0_baseline.json docs/bench/s1.json \
        --md docs/bench/BENCH.md
    python tools/solver_bench.py list

The corpus is deliberately frozen: cases are derived from CORPUS_VERSION and each case's
id seeds its own random draw, so two runs of the same corpus see byte-identical fields
regardless of insertion order or machine. Changing the corpus bumps CORPUS_VERSION, and
`compare` refuses to compare across versions -- re-run the older solver first.

Synthetic fields draw from the same offline Gaia catalogue the v2 pattern DB is built
from, so the two real ZWO fields (tests/data/fields/) ride along as the independent
check. Note one synthetic caveat: near the poles the generator's RA-box query misses
catalogue stars on the far side of the pole, which acts as extra detection dropout --
harmless for A/B purposes because both solvers see the identical field.
"""

import argparse
import contextlib
import io
import json
import re
import subprocess
import sys
import time
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.synthetic_field import junk_field, solution_matches_truth, synthesize_field

CORPUS_VERSION = '1'

#: named pointings (ra, dec): the measured envelope's regimes plus the two the current
#: consensus parameterisation is expected to fail (poles; roll near the 0/2pi wrap)
POINTINGS = {
    'midlat': (210.0, 35.0),
    'zwo3like': (356.0, 45.0),
    'sparse': (30.0, -20.0),       # sparse high-latitude
    'galplane': (266.0, -29.0),    # galactic plane density
    'pole_n': (85.0, 89.0),
    'pole_s': (100.0, -89.0),
}

SHAPE = (2000, 3000)
N_JUNK = 8
N_RELIABILITY_DRAWS = 8


def build_corpus():
    """The frozen case list. Every case is a dict with a unique 'id'."""
    cases = []

    def add(family, pointing=None, fov=None, seed_tag=0, **kw):
        case = {'family': family, 'pointing': pointing, 'fov': fov,
                'seed_tag': seed_tag}
        case.update(kw)
        bits = [family]
        if pointing:
            bits.append(pointing)
        if fov is not None:
            bits.append(f'fov{fov:g}')
        for key in sorted(kw):
            bits.append(f'{key}{kw[key]:g}' if isinstance(kw[key], float)
                        else f'{key}{kw[key]}')
        bits.append(f's{seed_tag}')
        case['id'] = '_'.join(bits)
        cases.append(case)

    # FOV envelope, both density regimes (PLATESOLVER_DESIGN.md section 1 shape)
    for pointing in ('midlat', 'galplane'):
        for fov in (0.6, 1.0, 2.0, 2.4, 4.0, 6.0, 8.0, 10.0, 12.0, 18.0):
            add('fov', pointing, fov)

    # reliability over detection-ordering draws: the measured 8/8, 6/8, 4/8 table
    for pointing, fov in (('midlat', 2.4), ('zwo3like', 2.4),
                          ('sparse', 2.4), ('midlat', 8.0)):
        for draw in range(N_RELIABILITY_DRAWS):
            add('reliability', pointing, fov, seed_tag=draw)

    # centroid-noise ceiling
    for noise in (0.3, 2.0, 4.0, 8.0):
        add('noise', 'midlat', 2.4, noise_px=noise)

    # magnitude-ordering scatter: the axis the dimmer-legs rule (S2b) is judged on
    for pointing in ('midlat', 'sparse'):
        for scatter in (0.0, 0.3, 0.6):
            for draw in range(2):
                add('scatter', pointing, 2.4, seed_tag=draw,
                    mag_order_scatter=scatter)

    # sparse-detection floor
    for n_detect in (7, 10, 12):
        add('sparse_detect', 'midlat', 2.4, n_detect=n_detect)

    # poles: the (centre, roll) chart is singular here; quaternion consensus (S4) target
    for pointing in ('pole_n', 'pole_s'):
        for draw in range(2):
            add('pole', pointing, 2.4, seed_tag=draw)

    # roll near the 0/2pi wrap: v1 clusters roll unwrapped, so consensus can split
    for roll in (0.0, 57.0, 359.9):
        add('rollwrap', 'midlat', 2.4, roll_deg=roll)

    # pure noise must be rejected -- the wrong-solve gate
    for draw in range(N_JUNK):
        add('junk', seed_tag=draw)

    # the real-field regression corpus: the check that is independent of the catalogue
    for name in sorted(p.stem for p in
                       (Path(__file__).parent.parent / 'tests' / 'data'
                        / 'fields').glob('*.json')):
        add('real', field=name)

    return cases


def case_seed(case_id):
    return zlib.crc32(case_id.encode('utf-8')) & 0x7fffffff


# ------------------------------------------------------------------ running

def open_synthetic_catalogue():
    from mee2024.starcat import providers
    try:
        return providers.GaiaOfflineProvider.from_installed(['gaia_dr3_g12'])
    except Exception as exc:
        raise SystemExit(
            f'the bench needs the offline Gaia catalogue to synthesize fields '
            f'({exc}). Install it with `mee2024 catalogue --fetch gaia_dr3_g12` '
            f'or build it with tools/build_gaia_offline.py.')


def make_options(solver, db):
    from mee2024.config import get_default_options
    options = get_default_options()
    options.update(flag_display=False, flag_display2=False, flag_display3=False,
                   flag_debug=False, platesolver='v2' if solver == 'v2' else 'triangle')
    if db:
        options['pattern_db'] = db
    return options


def prepare_case(case, catalogue):
    """Returns (centroids, image_shape, truth_or_None). truth None means junk."""
    if case['family'] == 'junk':
        return (junk_field(SHAPE, n=120, seed=case_seed(case['id'])), SHAPE, None)
    if case['family'] == 'real':
        field = json.loads(
            (Path(__file__).parent.parent / 'tests' / 'data' / 'fields'
             / f"{case['field']}.json").read_text(encoding='utf-8'))
        expected = field['expected']
        truth = {'ra': expected['ra'], 'dec': expected['dec'],
                 'platescale_arcsec': expected['platescale_arcsec']}
        return np.array(field['centroids']), tuple(field['img_shape']), truth
    ra, dec = POINTINGS[case['pointing']]
    centroids, truth = synthesize_field(
        catalogue, ra, dec,
        roll_deg=case.get('roll_deg', 57.0),
        fov_width_deg=case['fov'], shape=SHAPE,
        noise_px=case.get('noise_px', 0.3),
        n_detect=case.get('n_detect', 120),
        mag_order_scatter=case.get('mag_order_scatter', 0.3),
        seed=case_seed(case['id']))
    return centroids, SHAPE, truth


#: v1 prints these; the bench parses them so v1 needs no code change to be measured
RE_CANDIDATES = re.compile(r'initial triangle matches: (\d+)')
RE_THRESHOLD = re.compile(r'thresh = (\d+)')


def run_case(case, catalogue, options):
    centroids, image_shape, truth = prepare_case(case, catalogue)
    from mee2024 import platesolve_triangle

    buffer = io.StringIO()
    t0 = time.perf_counter()
    error = None
    try:
        with contextlib.redirect_stdout(buffer):
            result = platesolve_triangle.platesolve(
                centroids, image_shape, options=dict(options))
    except Exception as exc:  # a crash is a failed case, not a failed bench
        result = {'success': False}
        error = f'{type(exc).__name__}: {exc}'
    elapsed = time.perf_counter() - t0
    printed = buffer.getvalue()

    success = bool(result.get('success'))
    if truth is None:                      # junk: the only correct answer is "no"
        correct, wrong = (not success), success
    else:
        correct = solution_matches_truth(result, truth)
        wrong = success and not correct

    candidates = RE_CANDIDATES.findall(printed)
    thresholds = RE_THRESHOLD.findall(printed)
    diag = result.get('diagnostics') or {}
    row = {
        'id': case['id'], 'family': case['family'],
        'success': success, 'correct': bool(correct), 'wrong': bool(wrong),
        'time_s': round(elapsed, 3),
        'n_matched': (0 if result.get('matched_stars') is None
                      else int(len(result['matched_stars']))),
        'n_candidates': diag.get('n_candidates',
                                 sum(int(c) for c in candidates) if candidates else None),
        'threshold': diag.get('threshold',
                              int(thresholds[-1]) if thresholds else None),
        'mirror': bool(result.get('mirror')),
    }
    if error:
        row['error'] = error
    return row


def git_sha():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True,
                              cwd=Path(__file__).parent.parent,
                              check=True).stdout.strip()
    except Exception:
        return 'unknown'


def aggregate(rows):
    """Per-family and overall summary. Junk counts rejections, not solves."""
    fields = {}
    for row in rows:
        fields.setdefault(row['family'], []).append(row)
    out = {}
    for family, group in sorted(fields.items()):
        times = sorted(r['time_s'] for r in group)
        out[family] = {
            'n': len(group),
            'solved': sum(r['success'] for r in group),
            'correct': sum(r['correct'] for r in group),
            'wrong': sum(r['wrong'] for r in group),
            'median_time_s': round(times[len(times) // 2], 2),
            'p90_time_s': round(times[max(0, int(len(times) * 0.9) - 1)], 2),
        }
    solvable = [r for r in rows if r['family'] != 'junk']
    junk = [r for r in rows if r['family'] == 'junk']
    out['_overall'] = {
        'cases': len(rows),
        'correct_rate': round(sum(r['correct'] for r in solvable)
                              / max(1, len(solvable)), 4),
        'wrong_total': sum(r['wrong'] for r in rows),
        'junk_rejected': f"{sum(r['correct'] for r in junk)}/{len(junk)}",
    }
    return out


def cmd_run(args):
    corpus = build_corpus()
    if args.family:
        corpus = [c for c in corpus if c['family'] in args.family]
    if args.quick:
        corpus = [c for c in corpus if c['seed_tag'] < 2][::3]
    print(f'{len(corpus)} cases, solver={args.solver}'
          + (f', db={args.db}' if args.db else ''))

    catalogue = open_synthetic_catalogue()
    options = make_options(args.solver, args.db)

    # load the solver's database once, timed, so per-case times are pure solve time
    t0 = time.perf_counter()
    if args.solver == 'v1':
        from mee2024 import platesolve_triangle
        with contextlib.redirect_stdout(io.StringIO()):
            platesolve_triangle.load()
    db_load = round(time.perf_counter() - t0, 2)
    print(f'database load: {db_load} s')

    rows = []
    for i, case in enumerate(corpus):
        row = run_case(case, catalogue, options)
        rows.append(row)
        state = 'ok    ' if row['correct'] else ('WRONG ' if row['wrong'] else 'fail  ')
        print(f"[{i + 1:3d}/{len(corpus)}] {state} {row['time_s']:6.1f}s  {row['id']}"
              + (f"  ({row['error']})" if row.get('error') else ''))

    report = {
        'meta': {
            'corpus_version': CORPUS_VERSION,
            'solver': args.solver, 'db': args.db or '',
            'git_sha': git_sha(),
            'started': datetime.now().isoformat(timespec='seconds'),
            'db_load_s': db_load,
            'quick': bool(args.quick or args.family),
        },
        'aggregates': aggregate(rows),
        'cases': rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(f'\nwrote {out}')
    print(json.dumps(report['aggregates']['_overall'], indent=2))
    overall = report['aggregates']['_overall']
    if overall['wrong_total']:
        print('\n*** WRONG SOLVES PRESENT -- the gate for the next stage is closed ***')
    return 0


def cmd_compare(args):
    a = json.loads(Path(args.a).read_text(encoding='utf-8'))
    b = json.loads(Path(args.b).read_text(encoding='utf-8'))
    if a['meta']['corpus_version'] != b['meta']['corpus_version']:
        raise SystemExit('corpus versions differ '
                         f"({a['meta']['corpus_version']} vs "
                         f"{b['meta']['corpus_version']}); re-run the older solver "
                         'on the current corpus first')
    if a['meta'].get('quick') or b['meta'].get('quick'):
        print('note: comparing partial (quick/family-filtered) runs')

    label_a = f"{a['meta']['solver']}@{a['meta']['git_sha']}"
    label_b = f"{b['meta']['solver']}@{b['meta']['git_sha']}"
    rows_a = {r['id']: r for r in a['cases']}
    rows_b = {r['id']: r for r in b['cases']}
    shared = [i for i in rows_a if i in rows_b]

    lines = [f"## {label_a} vs {label_b} "
             f"({datetime.now().isoformat(timespec='seconds')})", '',
             f"corpus v{a['meta']['corpus_version']}, {len(shared)} shared cases. "
             f"DB load {a['meta']['db_load_s']} s -> {b['meta']['db_load_s']} s.", '',
             '| family | correct | wrong | median time (s) |',
             '|---|---|---|---|']
    for family in sorted(a['aggregates']):
        if family.startswith('_') or family not in b['aggregates']:
            continue
        fa, fb = a['aggregates'][family], b['aggregates'][family]
        lines.append(f"| {family} | {fa['correct']}/{fa['n']} -> "
                     f"{fb['correct']}/{fb['n']} | {fa['wrong']} -> {fb['wrong']} | "
                     f"{fa['median_time_s']} -> {fb['median_time_s']} |")
    oa, ob = a['aggregates']['_overall'], b['aggregates']['_overall']
    lines += ['', f"overall correct rate {oa['correct_rate']} -> "
                  f"{ob['correct_rate']}; wrong solves {oa['wrong_total']} -> "
                  f"{ob['wrong_total']}; junk rejected {oa['junk_rejected']} -> "
                  f"{ob['junk_rejected']}", '']

    flips = [(i, rows_a[i]['correct'], rows_b[i]['correct']) for i in shared
             if rows_a[i]['correct'] != rows_b[i]['correct']]
    if flips:
        lines.append('case flips:')
        lines += [f"- {'fixed' if new else 'BROKE'}: {i}" for i, _, new in flips]
        lines.append('')

    text = '\n'.join(lines)
    print(text)
    if args.md:
        md = Path(args.md)
        md.parent.mkdir(parents=True, exist_ok=True)
        header = '' if md.exists() else '# Solver bench results\n\n'
        with md.open('a', encoding='utf-8') as f:
            f.write(header + text + '\n')
        print(f'appended to {md}')
    return 0


def cmd_list(args):
    for case in build_corpus():
        print(case['id'])
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('run', help='run the corpus against one solver')
    p.add_argument('--solver', choices=['v1', 'v2'], required=True)
    p.add_argument('--db', default='', help='pattern DB variant (v2 only)')
    p.add_argument('--out', required=True, help='output JSON path')
    p.add_argument('--family', nargs='*', help='restrict to these case families')
    p.add_argument('--quick', action='store_true',
                   help='thinned corpus for a smoke run (not comparable to full runs)')
    p.set_defaults(func=cmd_run)

    p = sub.add_parser('compare', help='compare two run JSONs, append to BENCH.md')
    p.add_argument('a')
    p.add_argument('b')
    p.add_argument('--md', help='markdown file to append the comparison to')
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser('list', help='print every corpus case id')
    p.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
