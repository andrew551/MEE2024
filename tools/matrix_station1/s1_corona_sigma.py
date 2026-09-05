"""Station 1: does a more gradual coronal model (blur sigma 15 px instead of 10) steady the tiers?

Douglas, 2026-09-05: the per-block L moves noticeably when the corona is subtracted (the 0.4 s
block by -0.19"), so the blocks may be sensitive to the coronal model; would a wider blur --
15 px rather than 10 -- make the model more gradual and the four tiers more consistent?

The record's stacking, with one number changed: coronal_subtract_sigma_px = 15.0. The four
blocks are re-stacked from raw into eclipse_corona_s15/<tag>/, fitted two-pass against the
0.5" reference, and tabulated block by block beside the record (sigma 10) with
`s1_blocks_alone.blocks_table`. What to read: the spread of the four block L's and the pooled
sigma. A model that is too sharp leaves coronal structure under the stars near the Sun; one that
is too wide cannot follow the corona's gradient and leaves a residual slope. Sigma is in
pixels: 10 px = 18", 15 px = 28".

Usage: s1_corona_sigma.py [sigma_px]      (default 15)
"""
import glob, os, subprocess, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_blocks_alone import blocks_table, BLOCKS, REC, PY, REPO  # noqa: E402

SIG = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0
TAG = 's%g' % SIG
OUT = os.path.join(REC, 'eclipse_corona_' + TAG)
G = r"G:/Mexico April 2024/Station-1-Eclipse-Data"
RAW = {'0p25s_1810': ('2024-04-08_18_10_26Z', 'dark-250ms', 15),
       '0p3s_1811':  ('2024-04-08_18_11_28Z', 'dark-300ms', 0),
       '0p4s_1812':  ('2024-04-08_18_12_30Z', 'dark-400ms', 0),
       '0p3s_1813':  ('2024-04-08_18_13_31Z', 'dark-300ms', 0)}
S1 = ['--set', 'sensitive_mode_stack=True', '--set', 'centroid_gaussian_subtract=True',
      '--set', 'centroid_gaussian_thresh=4.0', '--set', 'min_area=2',
      '--set', 'sigma_subtract=0.0', '--set', 'background_subtraction_mode=annular',
      '--set', 'centroid_window_sigma=2.0', '--set', 'centroid_refine_window=True',
      '--set', 'delete_saturated_blob=True', '--set', 'blob_saturation_level=95',
      '--set', 'blob_radius_extra=200', '--set', 'centroid_gap_blob=100',
      '--set', 'eclipse_mask_mode=disk', '--set', 'eclipse_disk_margin_px=10',
      '--set', 'coronal_subtract=True', '--set', 'coronal_subtract_sigma_px=%.1f' % SIG,
      '--set', 'coronal_pedestal_adu=2000.0']


def restack(tag):
    block, darkset, first = RAW[tag]
    d = os.path.join(OUT, tag); os.makedirs(d, exist_ok=True)
    if not glob.glob(os.path.join(d, 'centroid_data*.zip')):
        frames = sorted(glob.glob(os.path.join(G, 'CapObj', block, '*.FIT')))[first:]
        print('  %s: stacking %d raw frames, coronal blur sigma %g px...' % (tag, len(frames), SIG), flush=True)
        with open(os.path.join(d, 'stage1.log'), 'w') as fh:
            subprocess.run([PY, '-m', 'mee2024.cli', 'stack', *frames,
                            '--dark', os.path.join(G, darkset, 'CapObj', '*', '*.FIT'),
                            '--flat', os.path.join(G, 'flat', 'CapObj', '2024-04-08*', '*.FIT'),
                            *S1, '--no-scan', '--no-display', '--quiet', '-o', d],
                           cwd=REPO, stdout=fh, stderr=subprocess.STDOUT)
    return d


os.makedirs(OUT, exist_ok=True)
for tag, _ in BLOCKS:
    restack(tag)
arms = [('corona sigma 10 px (record)', lambda t: os.path.join(REC, 'eclipse_corona', t)),
        ('corona sigma %g px' % SIG, lambda t: os.path.join(OUT, t))]
out = blocks_table(arms, label='coronal blur sigma, against the 0.5" reference, 2-10 R_sun, G<=13')
out.to_csv(os.path.join(REC, 'corona_sigma_%s.csv' % TAG), index=False)
print('->', os.path.join(REC, 'corona_sigma_%s.csv' % TAG))
