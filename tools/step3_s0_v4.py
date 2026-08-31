"""S0 V4: Douglas' forbidden region replaces the convex-hull blob, on subtracted frames.

The V3 record claimed a ~2 R_sun physical detection limit. Douglas challenged it twice over
and both challenges measured out: the 1.2-1.5 R_sun sigma was inflated by the blob's own
darkening step (honest MAD on un-blobbed subtracted frames: 75 ADU at 1.5-1.7, 32 at
1.7-1.9), and the MEE reduction of Bruns 2017 matched V 7.78 at 1.49 R_sun and V 7.15 at
1.63. The floor is BRIGHTNESS-dependent, not a wall: V<=8.5-class stars clear it inside
2 R_sun. V3's mask simply forbade looking.

So V4 drops the hull blob entirely and implements the forbidden region: a Sun-centred disk
painted flat at the pedestal, radius max(1.25 R_sun, that tier's measured saturation radius
+ DISK_MARGIN_PX), plus any remaining saturated pixels (streamer tips) dilated 10 px and
painted too. The margin was 20 px through the Leon reduction of record; Douglas cut it to
10 on 2026-08-31 after the Bruns cell showed an inner-annulus star (G 7.52 at 1.49 R_sun)
clearing E2's disk edge by only 11 px. The Leon numbers in docs/STEP3_2026.md were
measured at 20 and are NOT re-derived by this change: Leon's innermost admitted star sits
at 2.07 R_sun, 570 px outside its tier's disk, so the margin cannot reach it. Nothing else is masked; detection's own locally-adaptive threshold (4 sigma of local
variance) is left to price the fine-structure band, which the honest numbers say it can.

Sun centre: (3171, 3232), the astropy ephemeris projected through the V3 affine -- the
clipped-region centroid used before was 67 px off (streamers bias it) and the record's
(3208, 3293) was 71 px off the other way. Good to ~25 px, fine for a mask.

Also in this run:
  all87   -- every subtracted frame of the ladder in ONE stack (24.7 s of photons; an
             unweighted mean of mixed exposures is photon-optimal, since matched-filter
             weights are S/sigma^2 = exp/exp = 1) -- the depth ceiling probe.
  discard -- the single ~50 ms default-setting frame, corona removed with the 0.1 s tier
             model scaled by the measured sky ratio: what does a very short exposure give
             back? (a 2027 design datum).

Inputs: the FROZEN step3_s0_blursub preprocessed frames (read only). Outputs to
step3_s0_v4/. No mee2024/ change.
"""
import datetime as dt, glob, json, os, subprocess, zipfile
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, binary_dilation

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
FROZEN = r"D:/MEE2024 output/MEE_output/step3_s0_blursub"
RAW = r"G:/Leon Aug 2026/2026-08-12/Eclipse/SCI_ladder"
OUT = r"D:/MEE2024 output/MEE_output/step3_s0_v4"
REFS = sorted(glob.glob(os.path.join(REPO, "calibration", "zenith_cubic", "08-12_Z*.txt")))
SUN = (3171.0, 3232.0)                 # (x, y) px, ephemeris through the V3 affine
PS, RSUN, PED = 2.2054043, 947.1, 2000.0
RSAT_PX = {'0p1s': 612, '0p3s': 679, '0p6s': 736, '1p2s': 801}   # measured 99th-pct radii
DISK_MARGIN_PX = 10                            # was 20; see the module docstring
assert len(REFS) == 6

STAGE1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
          '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
          '--set','sigma_subtract=0.0','--set','delete_saturated_blob=False',
          '--set','remove_edgy_centroids=True','--set','centroid_refine_window=True',
          '--set','centroid_window_sigma=2.0']
S2COMMON = ['--order','cubic','--date-from-header','--fix-distortion',*REFS,
            '--set','distortion_fixed_coefficients=quadratic',
            '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
            '--set','rough_match_threshhold=36',
            '--set','enable_corrections=True','--set','enable_corrections_ref=True',
            '--set','observation_lat=42.740470','--set','observation_long=-5.613780',
            '--set','observation_height=1101','--set','observation_temp=29.2',
            '--set','observation_pressure=896.7','--set','observation_humidity=0.208',
            '--set','observation_wavelength=0.62']

def paint(img):
    """Return a copy with the forbidden disk and residual saturated tips painted flat."""
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    out = img.copy()
    sat = binary_dilation(img >= 65535, iterations=10)
    out[sat] = PED
    return out, (yy, xx)

def forbid(img, grids, radius_px):
    yy, xx = grids
    out = img
    out[np.hypot(xx-SUN[0], yy-SUN[1]) <= radius_px] = PED
    return out

def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode

def stages(name, pframes, obstime):
    out = os.path.join(OUT, name)
    os.makedirs(out, exist_ok=True)
    zips = glob.glob(os.path.join(out, 'centroid_data*.zip'))
    if not zips:
        run([PY,'-m','mee2024.cli','stack',*pframes,*STAGE1,
             '--no-scan','--no-display','--quiet','-o',out], os.path.join(out,'stage1.log'))
        zips = glob.glob(os.path.join(out, 'centroid_data*.zip'))
    if not zips:
        print(f'{name}: STAGE 1 FAILED', flush=True); return
    n = json.load(zipfile.ZipFile(zips[0]).open('results.txt'))['n_centroids']
    d2 = os.path.join(out, 'stage2')
    if not glob.glob(os.path.join(d2,'**','distortion_results.txt'), recursive=True):
        os.makedirs(d2, exist_ok=True)
        run([PY,'-m','mee2024.cli','distortion',zips[0],*S2COMMON,
             '--set','observation_time='+obstime,'--no-display','--quiet','-o',d2],
            os.path.join(d2,'stage2.log'))
    r = glob.glob(os.path.join(d2,'**','distortion_results.txt'), recursive=True)
    if r:
        j = json.load(open(r[0], encoding='utf-8'))
        print(f"{name}: {n} centroids -> {j['#stars used']} matched, "
              f"rms {j['final rms error (arcseconds)']:.4f} arcsec, "
              f"ps {j['platescale (arcseconds/pixel)']:.7f}", flush=True)
    else:
        print(f'{name}: {n} centroids -> STAGE 2 FAILED', flush=True)

MIDT = {'0p1s':'18:28:32','0p3s':'18:28:34','0p6s':'18:28:33','1p2s':'18:28:32'}
all_frames = []
for tier in ('0p1s','0p3s','0p6s','1p2s'):
    radius = max(1.25*RSUN/PS, RSAT_PX[tier] + DISK_MARGIN_PX)
    pre = os.path.join(OUT, tier, 'preprocessed')
    os.makedirs(pre, exist_ok=True)
    src = sorted(glob.glob(os.path.join(FROZEN, tier, 'preprocessed', '*.fits')))
    if len(glob.glob(os.path.join(pre, '*.fits'))) != len(src):
        print(f'{tier}: painting forbidden disk r={radius:.0f} px '
              f'({radius*PS/RSUN:.2f} R_sun) on {len(src)} frames', flush=True)
        for f in src:
            with fits.open(f) as hd:
                img = hd[0].data.astype(np.float64); hdr = hd[0].header.copy()
            out_img, grids = paint(img)
            out_img = forbid(out_img, grids, radius)
            hdr['HISTORY'] = f'V4 forbidden disk r={radius:.0f}px at ({SUN[0]:.0f},{SUN[1]:.0f}); sat tips dilated 10px; all painted to {PED:.0f} ADU'
            fits.writeto(os.path.join(pre, os.path.basename(f)), out_img.astype(np.uint16),
                         hdr, overwrite=True)
    pframes = sorted(glob.glob(os.path.join(pre, '*.fits')))
    all_frames += pframes
    stages(tier, pframes, MIDT[tier])

# the depth probe: every painted frame in one stack
stages('all87', all_frames, '18:28:32')

# the discard frame: corona removed with the scaled 0.1 s model
dfits = glob.glob(os.path.join(RAW, 'discard', '*.fits'))
if dfits:
    pre = os.path.join(OUT, 'discard', 'preprocessed')
    os.makedirs(pre, exist_ok=True)
    if not glob.glob(os.path.join(pre, '*.fits')):
        frames01 = sorted(glob.glob(os.path.join(RAW, '0.1s', '*', '*.fits')))
        acc = None
        for f in frames01:
            d = fits.getdata(f).astype(np.float64)
            acc = d if acc is None else acc + d
        model01 = gaussian_filter(acc/len(frames01), 10.0)
        with fits.open(dfits[0]) as hd:
            raw = hd[0].data.astype(np.float64); hdr = hd[0].header.copy()
        ny, nx = raw.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        ann = (np.hypot(xx-SUN[0], yy-SUN[1])*PS/RSUN > 2.5) & (np.hypot(xx-SUN[0], yy-SUN[1])*PS/RSUN < 6)
        scale = np.median(raw[ann]) / np.median(model01[ann])
        sub = np.clip(raw - scale*model01 + PED, 0, 65535)
        out_img, grids = paint(sub)
        out_img = forbid(out_img, grids, max(1.25*RSUN/PS, 272))
        hdr['HISTORY'] = f'V4 discard: 0.1s model scaled x{scale:.3f} (sky-ratio), forbidden disk'
        fits.writeto(os.path.join(pre, 'discard.fits'), out_img.astype(np.uint16), hdr,
                     overwrite=True)
        print(f'discard: model scale {scale:.3f}', flush=True)
    stages('discard', sorted(glob.glob(os.path.join(pre, '*.fits'))), '18:28:13')
print('done', flush=True)
