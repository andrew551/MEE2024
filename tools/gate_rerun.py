"""Re-reduce both eclipse cells with the mask as a LOGICAL GATE, and produce the
program's own graphical output.

Douglas, 2026-08-31: the fake centroids always cluster on the perimeter of the exclusion
circle, so the region should not be blanked out physically at all -- just gated.
`mask_bright_object` now works that way in the pipeline, but neither reduction of record
went through it: both `tools/step3_s0_v4.py` (Leon) and `tools/matrix_bruns/b17_s0.py`
(Bruns) paint the forbidden disk themselves, at tool level, and then run stage 1 with the
pipeline's mask switched off. So the painted edge is still in both sets of numbers.

This driver rebuilds both from the raw frames with no painting anywhere:

  * the coronal background is subtracted exactly as before (tier mean, 10 px Gaussian
    blur, +2000 ADU pedestal), so the flattening that makes inner stars measurable is
    unchanged;
  * pixels that were saturated in the RAW frame are written back at full scale instead of
    being painted over. That is the one thing the pipeline needs in order to find the
    Sun for itself -- its disk geometry is read from saturation -- and it leaves no dark
    edge for a detector to mistake for structure;
  * stage 1 then runs with the pipeline's own disk gate on, which excludes detections
    inside the disk without touching a pixel.

Everything else is held at the cell's reduction of record: the Bruns-compatible
convention (Gaussian background, footprint moments), the same frozen calibrations, the
same tolerances and cuts. The only variable is the painted edge.

Also produces what the program normally draws and the analysis tools have been skipping:
stage 2's error graphs and distortion field plot, and stage 3's covariance ellipse and
deflection scatter, for every tier of both datasets.
"""
import glob, json, os, subprocess, sys, zipfile
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
PEDESTAL, BLUR_SIGMA = 2000.0, 10.0

BRUNS_RAW = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images/eclipse"
BRUNS_OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_gate"
LEON_RAW = r"G:/Leon Aug 2026/2026-08-12/Eclipse/SCI_ladder"
LEON_OUT = r"D:/MEE2024 output/MEE_output/step3_gate"

# the convention of record for both cells: Gaussian background + footprint moments,
# plus the pipeline disk gate (delete_saturated_blob on, nothing painted)
S1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
      '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
      '--set','sigma_subtract=0.0','--set','centroid_refine_window=False',
      '--set','background_subtraction_mode=Gaussian',
      '--set','delete_saturated_blob=True','--set','eclipse_mask_mode=disk',
      '--set','eclipse_disk_margin_px=10','--set','remove_edgy_centroids=True',
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


def gate_ready(sources, outdir, saturation=65535):
    """Coronal subtraction with the saturated core preserved, and nothing painted.

    The tier mean is blurred and subtracted from every frame, then the pixels that were
    saturated in the RAW frame are restored to full scale. Restoring them is what lets the
    pipeline locate the Sun on its own -- it reads the disk geometry from saturation -- and
    unlike painting it creates no dark edge: the blur straddling the rim is pulled up
    rather than down, which suppresses spurious detections instead of generating them.
    """
    os.makedirs(outdir, exist_ok=True)
    made = sorted(glob.glob(os.path.join(outdir, '*.fits')))
    if len(made) == len(sources):
        return made
    acc = None
    for f in sources:
        d = fits.getdata(f).astype(np.float64)
        acc = d if acc is None else acc + d
    mean = acc/len(sources)
    # A MASKED blur: the coronal model is estimated from unsaturated pixels only.
    # Blurring the mean as it stands puts the 65535 plateau into the model, so a few tens
    # of pixels outside the core the model is still near full scale while the frame has
    # dropped to the corona -- the subtraction then cuts a ring of zeros right where the
    # inner stars are. Measured: stacked sky sigma 2385 ADU that way against 66 with this.
    # The old tools hid that ring by painting over it, which is very likely the mechanism
    # behind the fake centroids Douglas saw clustering on the mask perimeter.
    valid = mean < saturation
    num = gaussian_filter(np.where(valid, mean, 0.0), BLUR_SIGMA)
    den = gaussian_filter(valid.astype(np.float64), BLUR_SIGMA)
    model = np.where(den > 0.05, num/np.maximum(den, 1e-9), saturation)
    out = []
    for f in sources:
        with fits.open(f) as hd:
            raw = hd[0].data.astype(np.float64)
            hdr = hd[0].header.copy()
        sub = np.clip(raw - model + PEDESTAL, 0, saturation)
        sub[raw >= saturation] = saturation          # keep the core findable, unpainted
        hdr['HISTORY'] = ('gate rerun: tier-mean blur %.0f px subtracted, +%.0f ADU; '
                          'saturated core preserved; NOTHING painted'
                          % (BLUR_SIGMA, PEDESTAL))
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
    # stage 3: the program's own deflection analysis and its plots
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
    print('=== Bruns 2017, gate rerun ===', flush=True)
    refs = [glob.glob(r'D:/MEE2024 output/MEE_output/matrix_bruns2017_like2024/%s/stage2/'
                      r'**/distortion_results.txt' % t, recursive=True)[0]
            for t in ('L', 'R8')]
    for tier in ('EA', 'E2', 'EB'):
        src = sorted(glob.glob(os.path.join(BRUNS_RAW, '%s_*.fit' % tier)))
        frames = gate_ready(src, os.path.join(BRUNS_OUT, tier, 'preprocessed'))
        reduce_tier(tier, frames, BRUNS_OUT, BRUNS_SITE, BRUNS_MIDT[tier], refs)

if which in ('both', 'leon'):
    print('\n=== Leon 2026, gate rerun ===', flush=True)
    cal = glob.glob(r'D:/MEE2024 output/MEE_output/step3_bruns_convention/cal_pileo/'
                    r'stage2/**/distortion_results.txt', recursive=True)
    if not cal:
        print('no Bruns-convention CAL_piLeo; run tools/step3_leon_bruns_convention.py first')
    else:
        for tier, sub in (('0p1s', '0.1s'), ('0p3s', '0.3s'), ('0p6s', '0.6s'), ('1p2s', '1.2s')):
            src = sorted(glob.glob(os.path.join(LEON_RAW, sub, '*', '*.fits')))
            if not src:
                print('%s: no raw frames under %s' % (tier, os.path.join(LEON_RAW, sub)),
                      flush=True)
                continue
            frames = gate_ready(src, os.path.join(LEON_OUT, tier, 'preprocessed'))
            reduce_tier(tier, frames, LEON_OUT, LEON_SITE, LEON_MIDT[tier], cal[:1])
print('\ndone', flush=True)
