"""Stage 3 rerun with outlier-flagged stars removed -- what stage 3 SHOULD have done.

Measured tonight: eclipse_analysis reads CATALOGUE_MATCHED_ERRORS.csv and filters only
flag_is_double. The mag-11 cut caught the faint fakes, but V 9.10 @ 2.29 R_sun -- outlier
in every reduction, 2.2-3.1 arcsec displaced, neither double nor missing-pm -- entered the
preliminary L with the largest single 1/r^2 weight (12.5 %). This variant drops
flag_is_outlier rows from a copy of the stage-2 archive and reruns the standard stage 3.
"""
import glob, io, os, shutil, subprocess, zipfile
import pandas as pd
REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
B = r"D:/MEE2024 output/MEE_output/step3_prelim_L"
COMMON = ['--set','enable_corrections=True','--set','enable_corrections_ref=True',
          '--set','observation_lat=42.740470','--set','observation_long=-5.613780',
          '--set','observation_height=1101','--set','observation_temp=29.2',
          '--set','observation_pressure=896.7','--set','observation_humidity=0.208',
          '--set','observation_wavelength=0.62']
MIDT = {'0p6s':'18:28:33','1p2s':'18:28:32'}
for tier in ('0p6s','1p2s'):
    src = glob.glob(os.path.join(B, tier, 'stage2_constant', 'distortion_data*.zip'))[0]
    out = os.path.join(B, tier, 'stage3_novetoutliers')
    os.makedirs(out, exist_ok=True)
    mod = os.path.join(out, 'distortion_data_filtered.zip')
    zin = zipfile.ZipFile(src)
    with zipfile.ZipFile(mod, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.namelist():
            data = zin.read(item)
            if item.endswith('CATALOGUE_MATCHED_ERRORS.csv'):
                df = pd.read_csv(io.BytesIO(data))
                df.columns = [c.strip() for c in df.columns]
                n0 = len(df)
                df = df.loc[~df['flag_is_outlier']]
                print(f'{tier}: {n0} -> {len(df)} rows after dropping flagged', flush=True)
                buf = io.StringIO(); df.to_csv(buf, index=False)
                data = buf.getvalue().encode()
            zout.writestr(item, data)
    log = os.path.join(out, 's3.log')
    with open(log, 'w') as fh:
        subprocess.run([PY,'-m','mee2024.cli','eclipse',mod,*COMMON,
                        '--set','observation_time='+MIDT[tier],
                        '--no-display','--quiet','-o',out],
                       cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    import re
    txt = open(log, encoding='utf-8', errors='replace').read()
    m1 = re.search(r'final cov mu.*?\[np\.float64\(([\d.]+)\)', txt, re.S)
    m2 = re.search(r'final cov, mu.*?\[([\d.]+)\s', txt, re.S)
    print(f'{tier}: L(M1) = {m1.group(1)[:6] if m1 else "?"}   L(M2) = {m2.group(1)[:6] if m2 else "?"}', flush=True)
print('done', flush=True)
