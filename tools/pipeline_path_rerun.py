"""Re-reduce both eclipse cells through the PIPELINE's own eclipse path, and produce the
program's graphical output.

Douglas, 2026-09-01: "a lot of fake stars at the circular boundary are being used for
stacking. That does not look like a good idea." He is right, and the mechanism is worse
than it first sounds.

Both reductions of record preprocess outside the program -- coronal subtraction and a
painted forbidden disk in `tools/step3_s0_v4.py` and `tools/matrix_bruns/b17_s0.py` --
and then run stage 1 with the pipeline's mask OFF. With the mask off, `mask2` is all
zeros, so `filter_bad_centroids` has nothing to filter with and **every rim artefact
enters the per-frame centroid list that drives the alignment**. And because the tool
paints the disk at one fixed pixel position for the whole tier, those artefacts sit at
fixed DETECTOR coordinates while real stars dither with the sky -- so they behave exactly
like hot pixels, pulling the fitted shift toward zero. The stacker already knows this
danger for hot pixels (`hot_pixel_dark_free`); nothing was protecting it here.

The fix is to stop preprocessing outside the program. Everything the tools were doing now
exists in the pipeline (F26), so this runs the raw frames straight through:

    coronal_subtract=True      -- Bruns' 1917 background method, masked blur (the model is
                                 estimated from unsaturated pixels only, so it does not
                                 carve a trench outside the core)
    delete_saturated_blob=True -- with eclipse_mask_mode='disk'
    eclipse_mask_mode='disk'   -- a logical gate: no pixel is modified, and `mask2` is
                                 real, so per-frame centroids inside the disk are
                                 filtered out BEFORE they reach the alignment

Held at each cell's convention of record otherwise: Gaussian background, footprint
moments, the same frozen calibrations, the same tolerances and cuts. Stage 2's error
graphs and field plot and stage 3's own analysis and plots are written for every tier.
"""
import glob, json, os, subprocess, sys, zipfile

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
BRUNS_RAW = r"I:/2017 eclipse images Don Bruns/2017 Eclipse images/eclipse"
BRUNS_OUT = r"D:/MEE2024 output/MEE_output/matrix_bruns2017_pipeline"
LEON_RAW = r"G:/Leon Aug 2026/2026-08-12/Eclipse/SCI_ladder"
LEON_OUT = r"D:/MEE2024 output/MEE_output/step3_pipeline"

S1 = ['--set','sensitive_mode_stack=True','--set','centroid_gaussian_subtract=True',
      '--set','centroid_gaussian_thresh=4.0','--set','min_area=2',
      '--set','sigma_subtract=0.0','--set','centroid_refine_window=False',
      '--set','background_subtraction_mode=Gaussian',
      '--set','coronal_subtract=True','--set','coronal_subtract_sigma_px=10.0',
      '--set','coronal_pedestal_adu=2000.0',
      '--set','delete_saturated_blob=True','--set','eclipse_mask_mode=disk',
      '--set','eclipse_disk_margin_px=10','--set','centroid_gap_blob=30',
      '--set','remove_edgy_centroids=True','--set','distortion_field_plot=True']
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
    res1 = json.load(zipfile.ZipFile(z[0]).open('results.txt'))
    al = res1.get('alignment', {})
    nm = [x for x in al.get('n_matched', []) if x]
    rms = [x for x in al.get('rms_px', []) if x]
    print('%s: %d centroids on the stack; alignment used %d-%d stars/frame, rms %.2f px'
          % (name, res1['n_centroids'], (min(nm) if nm else 0), (max(nm) if nm else 0),
             (sum(rms)/len(rms) if rms else float('nan'))), flush=True)
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
        print('   -> stage 2 FAILED', flush=True)
        return None
    j = json.load(open(r[0], encoding='utf-8'))
    print('   -> %d matched, rms %.4f arcsec' % (j['#stars used'],
                                                 j['final rms error (arcseconds)']),
          flush=True)
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
    print('=== Bruns 2017 through the pipeline path ===', flush=True)
    refs = [glob.glob(r'D:/MEE2024 output/MEE_output/matrix_bruns2017_like2024/%s/stage2/'
                      r'**/distortion_results.txt' % t, recursive=True)[0]
            for t in ('L', 'R8')]
    for tier in ('EA', 'E2', 'EB'):
        src = sorted(glob.glob(os.path.join(BRUNS_RAW, '%s_*.fit' % tier)))
        reduce_tier(tier, src, BRUNS_OUT, BRUNS_SITE, BRUNS_MIDT[tier], refs)

if which in ('both', 'leon'):
    print('\n=== Leon 2026 through the pipeline path ===', flush=True)
    cal = glob.glob(r'D:/MEE2024 output/MEE_output/step3_bruns_convention/cal_pileo/'
                    r'stage2/**/distortion_results.txt', recursive=True)
    if not cal:
        print('no Bruns-convention CAL_piLeo available')
    else:
        for tier, sub in (('0p1s', '0.1s'), ('0p3s', '0.3s'), ('0p6s', '0.6s'),
                          ('1p2s', '1.2s')):
            src = sorted(glob.glob(os.path.join(LEON_RAW, sub, '*', '*.fits')))
            if not src:
                print('%s: no raw frames' % tier, flush=True)
                continue
            reduce_tier(tier, src, LEON_OUT, LEON_SITE, LEON_MIDT[tier], cal[:1])
print('\ndone', flush=True)
