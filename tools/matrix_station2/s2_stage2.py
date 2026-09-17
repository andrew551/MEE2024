"""Station 2 stage 2: the zenith reference, and the eclipse-day bracket fitted against it.

The point of this station is a plate scale Station 1 does not have. Station 2 is an NP101is with
the same reducer -- 419.8 mm effective against Station 1's 419.3, 0.12 % apart -- at the same
site on the same night, so its measured night-to-day scale change can be carried across to
Station 1, which has only a Method-2 scale fitted alongside L.

The pathway, with one step WRONG as first written -- see the correction below and
`tools/matrix_station2/s2_bracket_convention.py`:

  * zenith fields keep the ZENITH stage-1 convention, the bracket keeps the ECLIPSE one. The
    split is by day, not by pointing (mee2024/field_presets.py). Running the bracket at zenith
    settings on 2026-09-07 cost it two thirds of its stars, 28 and 36 against 83 and 83.
  * the zenith fields are fitted free, at one gate, and their results become the frozen
    reference;
  * the bracket is fitted against that reference with only the constant free and the scale let
    go (`distortion_free_scale`), two-pass at 20 " then 3 ", which is what recovers a scale the
    first fit would otherwise be dragged off by mis-matches.
    **THIS IS THE WRONG RUNG** (Douglas, 2026-09-08). The bracket is a DAYTIME CALIBRATION
    field, and `docs/V1_4_0_TESTING.md` section 5 gives the three-step ladder with a column
    headed "L/R calibration": zenith `None`, L/R calibration `quadratic`, eclipse field
    `constant`. Bruns' L and R8 and Leon's CAL_piLeo were both fitted `quadratic`; only the
    eclipse field gets `constant`. Copying Station 1's eclipse-field settings here froze the
    linear and quadratic terms that the published cells let move, and cost 0.27 " of L.
    `s2_bracket_convention.py` refits it correctly; this file is left as it ran.
  * the bracket frames where the sky is changing fast at either end of totality are dropped, as
    Leon's calibration did. Measured per frame: right 0-6 (sky falling 1.7-3.7 %/frame just after
    C2), left 47-52 (sky +150 % and tilt x65 as C3 arrives). The trim GAINS stars -- right 83 to
    98, left 83 to 94, from fewer frames -- and moves the scale by 1-3 ppm.

Both orders are run over the zenith fields, because whether the quintic is needed at all on the
ASI1600's half-size sensor is an open question (Douglas, 2026-09-07): Station 1 needed it on a
full frame, and this sensor reaches only 5450 arcsec of field radius against 10 700.

  .venv/Scripts/python.exe tools/matrix_station2/s2_stage2.py ref      # zenith, both orders
  .venv/Scripts/python.exe tools/matrix_station2/s2_stage2.py bracket  # L/R against each
  .venv/Scripts/python.exe tools/matrix_station2/s2_stage2.py report
"""
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
OUT = r"F:/MEE_output/mexico2024/station2"
ZEN = os.path.join(OUT, "zenith")                 # zenith convention, as stage 1 wrote it
LR = os.path.join(OUT, "lr_trimmed")              # eclipse convention, ends of totality trimmed

# El Salto, from Station 2's own 00README.txt -- the same site as Station 1, to the digit.
# Pressure is the Pasco weather station's 572.0 mmHg = 762.6 hPa, which is also the READMEs'
# "762.604 mbar"; Station 1's runs used 760.0, a 0.3 % difference worth nothing but no reason
# to repeat. Wavelength is the red filter's centre, 633 nm (Edmund 89819, 550-700 nm).
def site(temp, humidity):
    return ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
            '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
            '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
            '--set', 'observation_height=2400.0', '--set', 'observation_pressure=762.6',
            '--set', 'observation_wavelength=0.633',
            '--set', f'observation_temp={temp}', '--set', f'observation_humidity={humidity}']


# No sensor reaches the 04:08-04:41 zenith session -- the Pasco, both station probes and the
# Cube all start about 16:00 UTC -- so 10 C is Station 1's own assumption, carried over.
# The bracket is measured: Pasco 15.2 C, RH 24 % at 18:12.
SITE_ZENITH = site(10.0, 0.25)
SITE_BRACKET = site(15.2, 0.24)

GATE = 0.5      # the cell-2 record's reference gate
ORDERS = ('cubic', 'quintic')


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def zips(root):
    return sorted(glob.glob(os.path.join(root, '*', 'centroid_data*.zip')))


def midtime(z):
    """the field's mid time, from its stage-1 source folder name (…_HH_MM_SSZ)"""
    d = json.loads(__import__('zipfile').ZipFile(z).read('results.txt').decode('utf-8'))
    m = re.search(r'(\d{2})_(\d{2})_(\d{2})Z', d.get('source_folder', '') or '')
    if m:
        return f'{m.group(1)}:{m.group(2)}:{m.group(3)}'
    m = re.search(r'-(\d{2})(\d{2})_', d.get('source_folder', '') or '')
    return f'{m.group(1)}:{m.group(2)}:00' if m else '04:20:00'


def results_of(d):
    r = glob.glob(os.path.join(d, '**', 'distortion_results.txt'), recursive=True)
    return json.load(open(r[0], encoding='utf-8')) if r else None


def do_ref():
    for order in ORDERS:
        print(f'=== zenith reference, {order}, free fit at a {GATE} " gate')
        for z in zips(ZEN):
            stamp = os.path.basename(os.path.dirname(z))
            d = os.path.join(OUT, 'ref_' + order, stamp)
            os.makedirs(d, exist_ok=True)
            if not results_of(d):
                run([PY, '-m', 'mee2024.cli', 'distortion', z, '--order', order,
                     '--set', f'distortion_fit_tol={GATE}', '--set', 'max_star_mag_dist=15',
                     '--set', 'rough_match_threshhold=36', *SITE_ZENITH,
                     '--set', 'observation_time=' + midtime(z),
                     '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
            j = results_of(d)
            print(f'  {stamp}  ' + (f"{j['#stars used']:5d} stars  rms {j['final rms error (arcseconds)']:.4f}\"  "
                                    f"ps {j['platescale (arcseconds/pixel)']:.7f}" if j else 'FAILED'), flush=True)


def do_bracket():
    # The bracket is a DAYTIME CALIBRATION field and belongs on the ladder's middle rung,
    # `distortion_fixed_coefficients=quadratic` (docs/V1_4_0_TESTING.md section 5). This mode
    # fitted it `constant` + free scale -- the eclipse field's rung -- and the outputs it wrote
    # (bracket_cubic/, bracket_quintic/) are kept as the record of that error. It refuses to run
    # again rather than silently produce more of them: use s2_bracket_convention.py, which writes
    # bracket_quadfree/ and is what the record tools read (2026-09-09).
    raise SystemExit('s2_stage2.py bracket: wrong rung (constant); run '
                     'tools/matrix_station2/s2_bracket_convention.py instead')
    for order in ORDERS:  # noqa: unreachable -- kept so the history of the call is legible
        refs = sorted(glob.glob(os.path.join(OUT, 'ref_' + order, '*', '**', 'distortion_results.txt'), recursive=True))
        if not refs:
            print(f'no {order} reference yet'); continue
        print(f'=== bracket against the {order} reference ({len(refs)} fields)')
        for name, tm in (('right', '18:10:55'), ('left', '18:14:30')):
            z = glob.glob(os.path.join(LR, name, 'centroid_data*.zip'))
            if not z:
                print(f'  {name}: no stage-1 archive'); continue
            d = os.path.join(OUT, 'bracket_' + order, name)
            os.makedirs(d, exist_ok=True)
            if not results_of(d):
                run([PY, '-m', 'mee2024.cli', 'distortion', z[0], '--order', order,
                     '--fix-distortion', *refs,
                     '--set', 'distortion_fixed_coefficients=constant',
                     '--set', 'distortion_free_scale=True',
                     '--set', 'distortion_fit_tol_initial=20.0', '--set', f'distortion_fit_tol=3.0',
                     '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100',
                     *SITE_BRACKET, '--set', 'observation_time=' + tm,
                     '--no-display', '--quiet', '-o', d], os.path.join(d, 'stage2.log'))
            j = results_of(d)
            print(f'  {name}  ' + (f"{j['#stars used']:4d} stars  rms {j['final rms error (arcseconds)']:.4f}\"  "
                                   f"ps {j['platescale (arcseconds/pixel)']:.7f}" if j else 'FAILED'), flush=True)


def do_report():
    print(f"{'':<10} {'n':>3} {'stars/field':>12} {'rms \"':>8} {'scale as/px':>13} {'sd ppm':>8}")
    print('-' * 60)
    night = {}
    for order in ORDERS:
        js = [results_of(os.path.dirname(os.path.dirname(r))) for r in
              sorted(glob.glob(os.path.join(OUT, 'ref_' + order, '*', '**', 'distortion_results.txt'), recursive=True))]
        js = [j for j in js if j]
        if not js:
            continue
        ps = [j['platescale (arcseconds/pixel)'] for j in js]
        night[order] = (st.mean(ps), st.stdev(ps) if len(ps) > 1 else 0.0, len(ps))
        print(f"zenith {order:<9} {len(js):>3} {st.mean([j['#stars used'] for j in js]):>12.0f} "
              f"{st.mean([j['final rms error (arcseconds)'] for j in js]):>8.4f} "
              f"{st.mean(ps):>13.7f} {st.stdev(ps)/st.mean(ps)*1e6 if len(ps) > 1 else 0:>8.0f}")
    for order in ORDERS:
        rows = []
        for name in ('right', 'left'):
            j = results_of(os.path.join(OUT, 'bracket_' + order, name))
            if j:
                rows.append(j)
        if not rows:
            continue
        ps = [j['platescale (arcseconds/pixel)'] for j in rows]
        print(f"bracket {order:<8} {len(rows):>3} {st.mean([j['#stars used'] for j in rows]):>12.0f} "
              f"{st.mean([j['final rms error (arcseconds)'] for j in rows]):>8.4f} "
              f"{st.mean(ps):>13.7f} {(abs(ps[0]-ps[1])/st.mean(ps)*1e6 if len(ps) > 1 else 0):>8.0f}")
        if order in night:
            mz, sz, nz = night[order]
            md = st.mean(ps)
            print(f"    -> NIGHT TO DAY, {order}: {(md-mz)/mz*1e6:+.0f} ppm"
                  f"  (zenith se {sz/mz*1e6/nz**0.5:.0f} ppm)")


mode = sys.argv[1] if len(sys.argv) > 1 else 'report'
{'ref': do_ref, 'bracket': do_bracket, 'report': do_report}[mode]()
