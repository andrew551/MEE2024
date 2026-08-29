"""Two-pass rematch: associate detections through the FITTED model, not the rough solve.

The measured need (docs/STEP3_2026.md): match_centroids associates candidates through the
rough LINEAR solve, whose error on this refraction-compressed field crosses the 36 arcsec
rough threshold at py ~ 3850 -- so the below-Sun G 7.71 anchor at 2.17 R_sun (the field's
highest-leverage confirmed star, the only one on the far side of the Sun) never matches,
at any tolerance. Widening the rough gate mismatches wholesale (43 of 44 rejected).

This tool is the matcher's own TODO done offline: take a finished constant-only stage-2
result, push EVERY detection through the fitted chain

    (px,py) -> apply_corrections (the fitted polynomial) -> linear_transform(q) -> sky

using mee2024's own functions and the zip's own stored coefficients, then match against
the AstroCorrect-corrected catalogue in sky coordinates. Nothing is reimplemented: the
reconstruction must reproduce the already-matched stars' stored RA/DEC(obs) to better
than 0.05 arcsec (a built-in round-trip test) before it is allowed to claim any new star.

New matches are appended to a copy of the archive's CATALOGUE_MATCHED_ERRORS.csv (outlier
-flagged rows dropped, per the stage-3 finding of 2026-08-29) and the standard stage 3 is
run on the result. Output: the anchor's own measured radial deflection, and L with it in.
"""
import glob, io, json, os, re, subprocess, sys, zipfile
import numpy as np, pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
sys.path.insert(0, REPO)
from mee2024 import transforms, refraction_correction
from mee2024.distortion_polynomial import apply_corrections, get_coeff_names
from mee2024.starcat import providers
from mee2024.MEE2024util import date_string_to_float

B = r"D:/MEE2024 output/MEE_output/step3_prelim_L"
V4 = r"D:/MEE2024 output/MEE_output/step3_s0_v4"
MIDT = {'0p6s': '18:28:33', '1p2s': '18:28:32'}
MATCH_AS = 3.0                      # arcsec, pass-2 association radius
OPTS = dict(observation_date='2026-08-12', observation_lat=42.740470,
            observation_long=-5.613780, observation_height=1101.0,
            observation_temp=29.2, observation_pressure=896.7,
            observation_humidity=0.208, observation_wavelength=0.62,
            enable_corrections=True, enable_corrections_ref=True,
            enable_gravitational_def=False, gravity_sweep=False,
            distortion_fit_order='cubic', guess_date=False)

def chain(qdeg, cx, cy, det_pypx, img_shape, options, variant=(1, 1, 1)):
    """(py,px) detections -> (dec,ra) degrees through the fitted model.

    `variant` = (sy, sx, sroll): axis and roll sign flips, resolved EMPIRICALLY by the
    round-trip gate below rather than asserted from reading the conventions -- the gate
    is the arbiter either way, and a wrong guess cannot slip through it."""
    where, sroll = variant
    plate = det_pypx - np.array([img_shape[0]/2, img_shape[1]/2])
    q = (qdeg[0], qdeg[1], qdeg[2], sroll*qdeg[3])
    if where == 'both':          # flip before the polynomial (poly sees flipped frame)
        plate_c = apply_corrections(q, -plate, cx, cy, img_shape, options)
    elif where == 'linear':      # poly in the original frame, flip only for the rotation
        plate_c = -apply_corrections(q, plate, cx, cy, img_shape, options)
    elif where == 'linear2':     # poly correction extracted in original frame, then flipped
        corr = apply_corrections(q, plate, cx, cy, img_shape, options) - plate
        plate_c = -plate + corr
    else:                        # no flip
        plate_c = apply_corrections(q, plate, cx, cy, img_shape, options)
    vec = transforms.linear_transform(q, plate_c, img_shape)
    return transforms.to_polar(vec)          # columns (dec, ra) in degrees

def resolve_variant(qdeg, cx, cy, df, img_shape, options):
    """Scan sign/order variants; return (variant, rt_max_arcsec) of the best."""
    best = (None, 1e9)
    for where in ('both', 'linear', 'linear2', 'none'):
        for sroll in (1, -1):
            got = chain(qdeg, cx, cy, df[['py','px']].values.astype(float),
                        img_shape, options, (where, sroll))
            dra = (got[:,1]-df['RA(obs)'].values)*np.cos(np.radians(df['DEC(obs)'].values))
            rt = float(np.hypot(dra, got[:,0]-df['DEC(obs)'].values).max()*3600)
            if rt < best[1]:
                best = ((where, sroll), rt)
    return best

for tier in ('0p6s', '1p2s'):
    src = glob.glob(os.path.join(B, tier, 'stage2_constant', 'distortion_data*.zip'))[0]
    z = zipfile.ZipFile(src)
    res = json.load(z.open([n for n in z.namelist() if n.endswith('distortion_results.txt')][0]))
    df = pd.read_csv(z.open([n for n in z.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    df.columns = [c.strip() for c in df.columns]
    img_shape = (4176, 6248)
    options = dict(OPTS, observation_time=MIDT[tier],
                   distortionOrder=res.get('distortion order', 'cubic'))

    # the fitted model, exactly as stored
    q = (np.radians(res['platescale (arcseconds/pixel)']/3600.0),
         np.radians(res['RA']), np.radians(res['DEC']), np.radians(res['ROLL']))
    names = None
    try:
        names = get_coeff_names(options)
    except Exception:
        pass
    cxd, cyd = res['distortion coeffs x'], res['distortion coeffs y']
    if isinstance(cxd, str): cxd = json.loads(cxd.replace("'", '"'))
    if isinstance(cyd, str): cyd = json.loads(cyd.replace("'", '"'))
    if names is None: names = list(cxd.keys())
    cx = np.array([cxd[k] for k in names]); cy = np.array([cyd[k] for k in names])

    # ---- round-trip self-test on the already-matched stars, resolving conventions
    variant, rtmax = resolve_variant(q, cx, cy, df, img_shape, options)
    print(f'{tier}: best variant (sy,sx,sroll)={variant}, round-trip max {rtmax:.4f} arcsec')
    if rtmax > 0.05:
        print(f'{tier}: RECONSTRUCTION FAILED the 0.05 arcsec gate -- not proceeding'); continue

    # ---- pass 2: all detections through the model, matched to the corrected catalogue
    czip = zipfile.ZipFile(glob.glob(os.path.join(V4, tier, 'centroid_data*.zip'))[0])
    det = pd.read_csv(czip.open('STACKED_CENTROIDS_DATA.csv'))
    sky = chain(q, cx, cy, det[['py','px']].values.astype(float), img_shape, options, variant)

    prov = providers.GaiaOfflineProvider.from_installed()
    ra0, de0 = res['RA'], res['DEC']
    epoch = date_string_to_float(OPTS['observation_date'])
    cat = prov.lookup((ra0-2.6, ra0+2.6), (de0-2.2, de0+2.2), 11.0, epoch=epoch)
    not_double = ~np.asarray(cat.is_double(10.0))
    corr = refraction_correction.AstroCorrect()
    cat_c, _, _ = corr.correct_ra_dec(cat, options)
    cra = np.degrees(cat_c.get_ra()); cdec = np.degrees(cat_c.get_dec())
    cmag = np.asarray(cat_c.get_mags())
    cra0 = np.degrees(cat.get_ra()); cdec0 = np.degrees(cat.get_dec())   # uncorrected, for the record

    # dedup by PIXEL distance to rows already matched -- robust against small
    # differences between this catalogue correction and the pipeline's
    used_px = df[['px','py']].values.astype(float)
    def already(j):
        return bool(np.min(np.hypot(used_px[:,0]-det['px'][j], used_px[:,1]-det['py'][j])) < 4.0)
    # the anchor, explicitly: where does the model put the nearest catalogue star?
    aj = int(np.argmin(np.hypot(det['px']-3161, det['py']-4163)))
    if np.hypot(det['px'][aj]-3161, det['py'][aj]-4163) < 6:
        d_a = np.hypot((cra-sky[aj,1])*np.cos(np.radians(sky[aj,0])), cdec-sky[aj,0])*3600
        ic = int(np.argmin(d_a))
        print(f'  ANCHOR diagnostic: detection ({det["px"][aj]:.0f},{det["py"][aj]:.0f}) -> '
              f'sky ({sky[aj,1]:.4f},{sky[aj,0]:.4f}); nearest catalogue star '
              f'G {cmag[ic]:.2f} at {d_a[ic]:.2f} arcsec', flush=True)
    # one-to-one: a detection claimed by two catalogue stars is a blend -- drop both
    claims = {}
    for i in range(len(cmag)):
        if not not_double[i]: continue
        d = np.hypot((sky[:,1]-cra[i])*np.cos(np.radians(cdec[i])), sky[:,0]-cdec[i])*3600
        j = int(np.argmin(d))
        if d[j] < MATCH_AS and not already(j):
            claims.setdefault(j, []).append((i, float(d[j])))
    blends = {j for j, cl in claims.items() if len(cl) > 1}
    if blends:
        for j in blends:
            print(f'  blend at px ({det["px"][j]:.0f},{det["py"][j]:.0f}): '
                  f'{len(claims[j])} catalogue stars claim it -- all dropped', flush=True)
    new_rows = []
    for j, cl in claims.items():
        if j in blends: continue
        i, sep = cl[0]
        d = np.array([sep]); 
        if True:
            new_rows.append(dict(px=det['px'][j], py=det['py'][j],
                                 px_dist=np.nan, py_dist=np.nan, ID=f'rematch:{i}',
                                 **{'RA(catalog)': cra[i], 'DEC(catalog)': cdec[i],
                                    'RA(obs)': sky[j,1], 'DEC(obs)': sky[j,0]},
                                 magV=cmag[i], **{'error(")': sep},
                                 flag_is_double=False, flag_missing_pm=False,
                                 flag_is_outlier=False))
            print(f"  NEW: V {cmag[i]:.2f} at px ({det['px'][j]:.0f},{det['py'][j]:.0f}), "
                  f'sep {sep:.2f} arcsec', flush=True)
    print(f'{tier}: pass 2 adds {len(new_rows)} star(s)', flush=True)

    # ---- augmented archive: flagged rows out, new matches in; then the standard stage 3
    keep = df.loc[~df['flag_is_outlier']]
    aug = pd.concat([keep, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else keep
    is_anchor = (np.hypot(aug['px']-3161, aug['py']-4163) < 6)
    variants_s3 = [('with_anchor', aug), ('sans_anchor', aug.loc[~is_anchor])]
    for tag, table in variants_s3:
        out = os.path.join(B, tier, f'stage3_rematched_{tag}')
        os.makedirs(out, exist_ok=True)
        mod = os.path.join(out, 'distortion_data_rematched.zip')
        with zipfile.ZipFile(mod, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in z.namelist():
                data = z.read(item)
                if item.endswith('CATALOGUE_MATCHED_ERRORS.csv'):
                    buf = io.StringIO(); table.to_csv(buf, index=False); data = buf.getvalue().encode()
                zout.writestr(item, data)
        log = os.path.join(out, 's3.log')
        with open(log, 'w') as fh:
            subprocess.run([PY,'-m','mee2024.cli','eclipse',mod,
                            '--set','enable_corrections=True','--set','observation_time='+MIDT[tier],
                            '--no-display','--quiet','-o',out], cwd=REPO,
                           stdout=fh, stderr=subprocess.STDOUT)
        txt = open(log, encoding='utf-8', errors='replace').read()
        m1 = re.search(r'final cov mu.*?\[np\.float64\(([\d.]+)\)', txt, re.S)
        m2 = re.search(r'final cov, mu.*?\[([\d.]+)\s', txt, re.S)
        print(f'{tier} {tag}: {len(table)} stars -> L(M1) = {m1.group(1)[:6] if m1 else "?"}  '
              f'L(M2) = {m2.group(1)[:6] if m2 else "?"}', flush=True)
print('done', flush=True)
