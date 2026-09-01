"""Re-reduce both eclipse cells with the coronal model computed from unsaturated pixels,
and produce the program's own graphical output.

The change, and how it was found. Douglas noted that fake centroids always cluster on the
perimeter of the exclusion circle and proposed replacing the painted region with a logical
gate. The pipeline now does exactly that (`mask_bright_object`, F26). But testing it on
these reductions turned up something more specific, and the diagnosis reversed twice:

  1. Forcing the pipeline gate onto tool-preprocessed frames COLLAPSED detection --
     61 centroids to 15 on Bruns EA. Not the gate's fault: to let the pipeline find the
     Sun the frames had to keep their saturated core at full scale, and a 1.3 Mpx region
     of 65535 mixed with subtracted sky is a huge high-variance patch that the detector
     then has to mask, degrading a ring around it.
  2. **The painted disk was never an edge here.** Measured on the frames of record: the
     paint value is 2000 ADU and the surrounding subtracted sky is 1963-1977. It blends
     in. Douglas' concern is right in general -- on a RAW frame the pipeline's old blob
     painted 65535 down to a 5th-percentile floor, which is a violent edge, and that path
     is now gated instead -- but it does not describe these reductions.
  3. What DID put structure on the perimeter is the coronal model. Blurring the tier mean
     with the saturated core included puts a 65535 plateau into the model, so for a few
     tens of pixels outside the core the model is still near full scale while the frame
     has dropped to the corona. Subtracting it cuts a ring of over-subtracted pixels right
     where the inner stars are -- ~3 sigma = 30 px wide, against a painted disk that
     extends only 10-20 px past the core, so part of that ring was always left exposed
     just outside the mask. Stacked sky sigma: 2385 ADU that way against 66 with the fix.

So this run keeps the architecture of the reduction of record -- tool-level coronal
subtraction, forbidden disk painted flat at the pedestal, pipeline mask off -- and changes
exactly one thing: the model is a MASKED blur, estimated from unsaturated pixels only,

    model = gaussian_filter(mean * valid) / gaussian_filter(valid)

so near the rim it follows the true coronal trend instead of the plateau. Everything else
is the cell's convention of record (Gaussian background, footprint moments, same frozen
calibrations, same cuts).

Also produced, which the analysis tools have been skipping: stage 2's error graphs and
distortion field plot, and stage 3's own deflection analysis and plots, for every tier.
"""
import glob, json, os, subprocess, sys, zipfile
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter, binary_dilation

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
PEDESTAL, BLUR_SIGMA, SAT = 2000.0, 10.0, 65535
DISK_MARGIN = 10

BRUNS_RAW = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images/eclipse"
BRUNS_OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_modelfix"
BRUNS_RSAT = {'EA': 901, 'E2': 646, 'EB': 902}          # measured, b17_inventory.py
BRUNS_RSUN_PX = 948.7/2.0868004
LEON_RAW = r"G:/Leon Aug 2026/2026-08-12/Eclipse/SCI_ladder"
LEON_OUT = r"D:/MEE2024 output/MEE_output/step3_modelfix"
LEON_RSAT = {'0p1s': 612, '0p3s': 679, '0p6s': 736, '1p2s': 801}
LEON_RSUN_PX = 947.1/2.2054043

S1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
      '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
      '--set','sigma_subtract=0.0','--set','centroid_refine_window=False',
      '--set','background_subtraction_mode=Gaussian',
      '--set','delete_saturated_blob=False','--set','remove_edgy_centroids=True',
      '--set','distortion_field_plot=True']
BRUNS_SITE = ['--set','observation_lat=42 44 11 N','--set','observation_long=106 19 05 W',
              '--set','observation_height=2400','--set','observation_temp=13.0',
              '--set','observation_pressure=770.0','--set','observation_humidity=0.4',
              '--set','observation_wavelength=0.625']
LEON_SITE = ['--set','observation_lat=42.740470','--set','observation_long=-5.613780',
             '--set','observation_height=1101','--set','observation_temp=29.2',
             '--set','observation_pressure=896.7','--set','observation_humidity=0.208',
             '--set','observation_wavelength=0.62']
BRUNS_MIDT = {'EA': '17:43:22', 'E2': '17:43:47', 'EB': '17:44:13'}
LEON_MIDT = {'0p1s': '18:28:32', '0p3s': '18:28:34', '0p6s': '18:28:33', '1p2s': '18:28:32'}


def run(cmd, log):
    with open(log, 'w') as fh:
        return subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT).returncode


def preprocess(sources, outdir, radius_px):
    """Masked-blur coronal subtraction, then the forbidden disk painted at the pedestal."""
    os.makedirs(outdir, exist_ok=True)
    made = sorted(glob.glob(os.path.join(outdir, '*.fits')))
    if len(made) == len(sources):
        return made
    acc, sat_any = None, None
    for f in sources:
        d = fits.getdata(f).astype(np.float64)
        acc = d if acc is None else acc + d
        s = d >= SAT
        sat_any = s if sat_any is None else (sat_any | s)
    mean = acc/len(sources)
    valid = mean < SAT
    num = gaussian_filter(np.where(valid, mean, 0.0), BLUR_SIGMA)
    den = gaussian_filter(valid.astype(np.float64), BLUR_SIGMA)
    model = np.where(den > 0.05, num/np.maximum(den, 1e-9), SAT)
    yy, xx = np.nonzero(sat_any)
    cy, cx = float(yy.mean()), float(xx.mean())
    ny, nx = mean.shape
    gy = (np.arange(ny, dtype=np.float32) - np.float32(cy))[:, None]
    gx = (np.arange(nx, dtype=np.float32) - np.float32(cx))[None, :]
    disk = (gy*gy + gx*gx) <= np.float32(radius_px*radius_px)
    print('  disk r=%.0f px at (%.0f, %.0f); masked-blur model' % (radius_px, cx, cy),
          flush=True)
    out = []
    for f in sources:
        with fits.open(f) as hd:
            raw = hd[0].data.astype(np.float64)
            hdr = hd[0].header.copy()
        sub = np.clip(raw - model + PEDESTAL, 0, SAT)
        sub[binary_dilation(raw >= SAT, iterations=10)] = PEDESTAL
        sub[disk] = PEDESTAL
        hdr['HISTORY'] = ('coronal model from UNSATURATED pixels (masked blur %.0f px), '
                          '+%.0f ADU; disk r=%.0f px painted at the pedestal'
                          % (BLUR_SIGMA, PEDESTAL, radius_px))
        base = os.path.basename(f)
        p = os.path.join(outdir, base if base.endswith('.fits') else base[:-4] + '.fits')
        fits.writeto(p, sub.astype(np.uint16), hdr, overwrite=True)
        out.append(p)
    return sorted(out)


def reduce_tier(name, frames, outroot, site, obstime, ref, tol='2.0'):
    d = os.path.join(outroot, name)
    os.makedirs(d, exist_ok=True)
    z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        run([PY,'-m','mee2024.cli','stack',*frames,*S1,'--no-scan','--quiet','-o',d],
            os.path.join(d, 'stage1.log'))
        z = glob.glob(os.path.join(d, 'centroid_data*.zip'))
    if not z:
        print('%s: STAGE 1 FAILED' % name, flush=True)
        return None
    n = json.load(zipfile.ZipFile(z[0]).open('results.txt'))['n_centroids']
    d2 = os.path.join(d, 'stage2')
    os.makedirs(d2, exist_ok=True)
    if not glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True):
        run([PY,'-m','mee2024.cli','distortion',z[0],'--order','cubic','--date-from-header',
             '--fix-distortion',*ref,'--set','distortion_fixed_coefficients=constant',
             '--set','distortion_fit_tol=' + tol,'--set','max_star_mag_dist=13',
             '--set','rough_match_threshhold=36','--set','enable_corrections=True',
             '--set','enable_corrections_ref=True','--set','distortion_field_plot=True',
             *site,'--set','observation_time=' + obstime,'--quiet','-o',d2],
            os.path.join(d2, 'stage2.log'))
    r = glob.glob(os.path.join(d2, '**', 'distortion_results.txt'), recursive=True)
    if not r:
        print('%s: %d centroids -> stage 2 FAILED' % (name, n), flush=True)
        return None
    j = json.load(open(r[0], encoding='utf-8'))
    print('%s: %d centroids -> %d matched, rms %.4f arcsec'
          % (name, n, j['#stars used'], j['final rms error (arcseconds)']), flush=True)
    dz = glob.glob(os.path.join(d2, 'distortion_data*.zip'))
    d3 = os.path.join(d, 'stage3')
    os.makedirs(d3, exist_ok=True)
    if dz and not glob.glob(os.path.join(d3, '*.log')):
        run([PY,'-m','mee2024.cli','eclipse',dz[0],'--set','enable_corrections=True',
             '--set','enable_corrections_ref=True',*site,
             '--set','observation_time=' + obstime,'--quiet','-o',d3],
            os.path.join(d3, 'stage3.log'))
    return r[0]


which = sys.argv[1] if len(sys.argv) > 1 else 'both'

if which in ('both', 'bruns'):
    print('=== Bruns 2017, masked-blur coronal model ===', flush=True)
    refs = [glob.glob(r'D:/MEE2024 output/MEE_output/matrix_bruns2017_like2024/%s/stage2/'
                      r'**/distortion_results.txt' % t, recursive=True)[0]
            for t in ('L', 'R8')]
    for tier in ('EA', 'E2', 'EB'):
        src = sorted(glob.glob(os.path.join(BRUNS_RAW, '%s_*.fit' % tier)))
        rad = max(1.25*BRUNS_RSUN_PX, BRUNS_RSAT[tier] + DISK_MARGIN)
        frames = preprocess(src, os.path.join(BRUNS_OUT, tier, 'preprocessed'), rad)
        reduce_tier(tier, frames, BRUNS_OUT, BRUNS_SITE, BRUNS_MIDT[tier], refs)

if which in ('both', 'leon'):
    print('\n=== Leon 2026, masked-blur coronal model ===', flush=True)
    cal = glob.glob(r'D:/MEE2024 output/MEE_output/step3_bruns_convention/cal_pileo/'
                    r'stage2/**/distortion_results.txt', recursive=True)
    if not cal:
        print('no Bruns-convention CAL_piLeo; run tools/step3_leon_bruns_convention.py first')
    else:
        for tier, sub in (('0p1s', '0.1s'), ('0p3s', '0.3s'), ('0p6s', '0.6s'),
                          ('1p2s', '1.2s')):
            src = sorted(glob.glob(os.path.join(LEON_RAW, sub, '*', '*.fits')))
            if not src:
                print('%s: no raw frames' % tier, flush=True)
                continue
            rad = max(1.25*LEON_RSUN_PX, LEON_RSAT[tier] + DISK_MARGIN)
            frames = preprocess(src, os.path.join(LEON_OUT, tier, 'preprocessed'), rad)
            reduce_tier(tier, frames, LEON_OUT, LEON_SITE, LEON_MIDT[tier], cal[:1])
print('\ndone', flush=True)
