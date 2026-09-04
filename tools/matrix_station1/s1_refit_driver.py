"""Constant-only eclipse refits against a chosen reference set, driven from Python.

Exists because two bash drivers for the same job failed the same way: the reference paths
live under `D:\\MEE2024 output\\...`, and an unquoted `$REFS` word-splits on the space, so
the pipeline was handed the fragment `D:\\MEE2024` and reported FileNotFoundError. Passing the
list through `subprocess.run` avoids the shell altogether.

Usage: s1_refit_driver.py septic      -- the four corona blocks against the SEPTIC reference
       s1_refit_driver.py caldecomp   -- the dark-only / flat-only / neither stacks of the 0.4 s
                                        block against the quintic reference
"""
import glob, os, subprocess, sys

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
REC = r"D:/MEE2024 output/MEE_output/station1_record"
MET = ['--set', 'enable_corrections=True', '--set', 'enable_corrections_ref=True',
       '--set', 'observation_date=2024-04-08', '--set', 'guess_date=False',
       '--set', 'observation_long=105 16 22.1 W', '--set', 'observation_lat=23 50 58.3 N',
       '--set', 'observation_temp=15.0', '--set', 'observation_pressure=760.0',
       '--set', 'observation_humidity=0.25', '--set', 'observation_height=2400.0']


def refit(cz, out, refs, order, tmid):
    os.makedirs(out, exist_ok=True)
    if glob.glob(os.path.join(out, '**', 'distortion_data*.zip'), recursive=True):
        return 'cached'
    cmd = [PY, '-m', 'mee2024.cli', 'distortion', cz, '--order', order, '--fix-distortion', *refs,
           '--set', 'distortion_fixed_coefficients=constant', '--set', 'distortion_fit_tol=20.0',
           '--set', 'max_star_mag_dist=13', '--set', 'rough_match_threshhold=100', *MET,
           '--set', 'observation_time=' + tmid, '--no-display', '--quiet', '-o', out]
    with open(os.path.join(out, 'stage2.log'), 'w') as fh:
        rc = subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode
    return 'rc %d' % rc


mode = sys.argv[1] if len(sys.argv) > 1 else 'septic'
if mode == 'septic':
    refs = sorted(glob.glob(os.path.join(REC, 'septic_test', 'ref', '*', '**', 'distortion_results.txt'), recursive=True))
    print('septic reference: %d fields' % len(refs), flush=True)
    for tag, tm in (('0p25s_1810', '18:11:12'), ('0p3s_1811', '18:11:58'),
                    ('0p4s_1812', '18:13:00'), ('0p3s_1813', '18:14:02')):
        cz = glob.glob(os.path.join(REC, 'eclipse_corona', tag, 'centroid_data*.zip'))[0]
        print('  %s %s' % (tag, refit(cz, os.path.join(REC, 'eclipse_corona', tag, 'stage2_septic'), refs, 'septic', tm)), flush=True)
else:
    refs = sorted(glob.glob(os.path.join(REC, 'zenith_recentroid', '*', 'stage2_free', '**', 'distortion_results.txt'), recursive=True))
    print('quintic reference: %d fields' % len(refs), flush=True)
    for arm in ('darkonly', 'flatonly', 'neither'):
        z = glob.glob(os.path.join(REC, 'eclipse_caldecomp', arm, 'centroid_data*.zip'))
        if not z:
            print('  %s: no stack' % arm, flush=True); continue
        print('  %s %s' % (arm, refit(z[0], os.path.join(REC, 'eclipse_caldecomp', arm, 'stage2'), refs, 'quintic', '18:13:00')), flush=True)
print('DRIVER DONE')
