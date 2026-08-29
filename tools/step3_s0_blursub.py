"""S0: Bruns-style coronal subtraction, prototyped ahead of any pipeline change.

Bruns 2017 (the paper's ECLIPSE pre-step): average the series WITHOUT translations, apply a
10-pixel Gaussian blur -- hides the stars, preserves the local coronal shape -- and subtract
that model from every individual frame. The centroids then sit on a flat background.

Why this beats the blob margin here, measured on this data: the margin cannot win. At
500/150 px (the config-leak values) the exclusion reached ~2.5 R_sun and the 2-4 R_sun
annulus held 2 detections of its 36 catalogue stars. At the 100/30 defaults the 0.1 s tier's
smaller hull exposed the steep unsaturated rim, 107 of 160 detections landed in the
1.5-2.0 R_sun ring, the solver fed on them brightest-first, and the plate solve failed. The
blob deletes a REGION; Bruns deletes the CORONA. Once the corona is gone the exclusion can
hug the saturated core -- the Bruns-2017 MEE reduction ran at radius 20 / gap 10 and found
255 stars to the rim.

Mechanics, per tier (the folder is the truth about exposure):
  model    = gaussian_blur(mean of raw frames, unshifted; sigma 10 px)
             Drift within a tier is a few px, far under the blur, so unshifted is fine.
             A star's own imprint in the model is flux/(2*pi*sigma^2) -- about 0.16 % of
             its flux at its peak pixel: negligible self-subtraction, no centroid bias.
  out      = raw - model + 2000 ADU pedestal, clipped to [0, 65535], uint16
  saturated pixels of the raw frame are RE-IMPOSED at 65535 in the output, so the
             blob machinery still finds and masks the informationless core.
  then the standard chain, blob at radius 30 / gap 10 (Bruns-2017-analysis class).

Writes preprocessed frames to D: (never G:). ~4.5 GB; delete after S0 if wanted.
No mee2024/ change: if this wins it becomes a ROADMAP feature proposal with these numbers.
"""
import datetime as dt, glob, json, os, subprocess, zipfile
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, shift as nd_shift
from skimage.registration import phase_cross_correlation

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
SRC = r"G:/Leon Aug 2026/2026-08-12/Eclipse/SCI_ladder"
OUT = r"D:/MEE2024 output/MEE_output/step3_s0_blursub"
REFS = sorted(glob.glob(os.path.join(REPO, "calibration", "zenith_cubic", "08-12_Z*.txt")))
SIGMA, PED = 10.0, 2000.0
assert len(REFS) == 6

# V3. The V1/V2 subtracted stacks measure where the junk actually lives: coronal FINE
# structure (streamer rays, the clip-boundary kink -- the filaments surviving in Bruns'
# own fig 8c/d) confined to r < ~1.8-2.0 R_sun, with the field beyond flat to 4-8 ADU.
# Registration changed nothing because the residual is not a displacement error; no smooth
# model removes structure at star scale. So the inner limit for detection on this corona is
# ~2 R_sun -- the design's own inner cut -- and the mask should sit there per tier:
# margin ~ (2.0 R_sun = 858 px) minus that tier's saturated-hull radius.
MARGINS = {'0.1s': (200, 30), '0.3s': (150, 30), '0.6s': (90, 20), '1.2s': (40, 10)}
def stage1_for(tier):
    r, g = MARGINS[tier]
    return ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
            '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
            '--set','sigma_subtract=0.0','--set','delete_saturated_blob=True',
            '--set','remove_edgy_centroids=True','--set','centroid_refine_window=True',
            '--set','centroid_window_sigma=2.0',
            '--set',f'blob_radius_extra={r}','--set',f'centroid_gap_blob={g}',
            '--set','blob_saturation_level=100']

def midtime(frames, exp):
    ts = [dt.datetime.fromisoformat(fits.getheader(f)['DATE-OBS']) + dt.timedelta(seconds=exp/2)
          for f in frames]
    t0 = min(ts)
    return (t0 + dt.timedelta(seconds=sum((t-t0).total_seconds() for t in ts)/len(ts))).strftime('%H:%M:%S')

for tier in ('0.1s','0.3s','0.6s','1.2s'):
    frames = sorted(glob.glob(os.path.join(SRC, tier, '*', '*.fits')))
    exp = float(tier.rstrip('s'))
    out = os.path.join(OUT, tier.replace('.','p'))
    pre = os.path.join(out, 'preprocessed')
    os.makedirs(pre, exist_ok=True)

    done = sorted(glob.glob(os.path.join(pre, '*.fits')))
    if len(done) != len(frames):
        # V2, after the unregistered attempt failed on the shallow tiers: the model must be
        # REGISTERED to each frame's corona before subtracting. The mount drifted 2.04 px
        # between the 0.1 s tier's two blocks (measured from run 2's alignment shifts), and
        # the rim gradient runs ~200 ADU/px, so even a ~1 px mismatch leaves ~200 ADU of
        # STRUCTURED ripple against ~30 ADU of frame noise -- the unregistered subtraction
        # produced 1190 detections on the 0.1 s tier (1146 of them inside 2 R_sun), worse
        # than no subtraction at all. Registration is by phase correlation of the blurred
        # frame against the blurred model, on a crop around the corona (the blur removes
        # the stars, so the corona -- fixed to the Sun -- is what registers; that is the
        # correct frame for a background model). Residual after registration:
        # gradient x ~0.05 px ~ 10 ADU, at the noise.
        print(f'{tier}: building coronal model from {len(frames)} frames', flush=True)
        acc = None
        for f in frames:
            d = fits.getdata(f).astype(np.float64)
            acc = d if acc is None else acc + d
        model = gaussian_filter(acc/len(frames), SIGMA)
        Y0, Y1, X0, X1 = 2200, 4176, 2000, 4300     # crop around the corona (Sun ~ (3110, 3204))
        mcrop = model[Y0:Y1, X0:X1]
        for f in frames:
            with fits.open(f) as hd:
                raw = hd[0].data.astype(np.float64); hdr = hd[0].header.copy()
            fcrop = gaussian_filter(raw[Y0:Y1, X0:X1], SIGMA)
            off, _, _ = phase_cross_correlation(fcrop, mcrop, upsample_factor=20)
            m_shifted = nd_shift(model, off, order=1, mode='nearest')
            sub = np.clip(raw - m_shifted + PED, 0, 65535)
            sub[raw >= 65535] = 65535           # keep the core visibly saturated for the blob
            hdr['HISTORY'] = ('coronal model subtracted: tier mean, gauss sigma='
                              f'{SIGMA} px, registered shift ({off[0]:+.2f},{off[1]:+.2f}) px, '
                              f'pedestal {PED} ADU')
            base = os.path.basename(os.path.dirname(f)) + '_' + os.path.basename(f).replace(' ', '_')
            fits.writeto(os.path.join(pre, base), sub.astype(np.uint16), hdr, overwrite=True)
        print(f'{tier}: {len(frames)} frames preprocessed (registered)', flush=True)

    pframes = sorted(glob.glob(os.path.join(pre, '*.fits')))
    zips = glob.glob(os.path.join(out, 'centroid_data*.zip'))
    if not zips:
        subprocess.run([PY,'-m','mee2024.cli','stack',*pframes,*stage1_for(tier),
                        '--no-scan','--no-display','--quiet','-o',out], cwd=REPO,
                       stdout=open(os.path.join(out,'stage1.log'),'w'), stderr=subprocess.STDOUT)
        zips = glob.glob(os.path.join(out, 'centroid_data*.zip'))
    if not zips:
        print(f'{tier}: STAGE 1 FAILED', flush=True); continue
    n = json.load(zipfile.ZipFile(zips[0]).open('results.txt'))['n_centroids']
    print(f'{tier}: {n} centroids detected', flush=True)

    d2 = os.path.join(out, 'stage2')
    if not glob.glob(os.path.join(d2,'**','distortion_results.txt'), recursive=True):
        os.makedirs(d2, exist_ok=True)
        subprocess.run([PY,'-m','mee2024.cli','distortion',zips[0],
                        '--order','cubic','--date-from-header','--fix-distortion',*REFS,
                        '--set','distortion_fixed_coefficients=quadratic',
                        '--set','distortion_fit_tol=2.0','--set','max_star_mag_dist=13',
                        '--set','rough_match_threshhold=36',
                        '--set','enable_corrections=True','--set','enable_corrections_ref=True',
                        '--set','observation_lat=42.740470','--set','observation_long=-5.613780',
                        '--set','observation_height=1101','--set','observation_temp=29.2',
                        '--set','observation_pressure=896.7','--set','observation_humidity=0.208',
                        '--set','observation_wavelength=0.62',
                        '--set','observation_time='+midtime(frames, exp),
                        '--no-display','--quiet','-o',d2], cwd=REPO,
                       stdout=open(os.path.join(d2,'stage2.log'),'w'), stderr=subprocess.STDOUT)
    r = glob.glob(os.path.join(d2,'**','distortion_results.txt'), recursive=True)
    if r:
        j = json.load(open(r[0], encoding='utf-8'))
        print(f"   -> {j['#stars used']} stars matched, rms {j['final rms error (arcseconds)']:.4f} arcsec, "
              f"ps {j['platescale (arcseconds/pixel)']:.7f}", flush=True)
    else:
        print(f'   -> STAGE 2 FAILED', flush=True)
print('done', flush=True)
