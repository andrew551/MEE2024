"""Station 1: the pooled fit across the admission rule, the magnitude cut and the vet.

Douglas' two questions of 2026-09-04: pooling every observation was meant to remove the
two/three/four-block admission rule -- does it, and what does the rule cost or buy? And the
science set is G <= 12 while stage 2 matches to G 13 -- what do G 11, 12 and 13 do to L?
Plus the vet, whose median + 4 MAD needed explaining: what do no vet, 3 MAD and 5 MAD do?

Runs `s1_pooled_fit.py` over the grid (bootstrap shortened to 300 samples per cell, which
puts +-0.005" on each sigma) and tabulates. Writes station1_record/pooled_fit/grid.csv.
"""
import json, os, subprocess, sys
import pandas as pd

REPO = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024"
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
TOOL = os.path.join(REPO, 'tools', 'matrix_station1', 's1_pooled_fit.py')
REC = r"D:/MEE2024 output/MEE_output/station1_record"
REF = sys.argv[1] if len(sys.argv) > 1 else 'twopass'

cells = []
for mb in (1, 2, 3, 4):
    for mag in (11.0, 12.0, 13.0):
        cells.append(dict(min_blocks=mb, magcut=mag, vet=1, vet_k=4.0))
for vet, k in ((0, 4.0), (1, 3.0), (1, 5.0), (3, 4.0)):
    cells.append(dict(min_blocks=1, magcut=12.0, vet=vet, vet_k=k))

rows = []
for c in cells:
    tag = 'grid_b%d_g%.0f_v%d_k%.0f' % (c['min_blocks'], c['magcut'], c['vet'], c['vet_k'])
    cmd = [PY, TOOL, '--ref', REF, '--min-blocks', str(c['min_blocks']), '--magcut', str(c['magcut']),
           '--vet', str(c['vet']), '--vet-k', str(c['vet_k']), '--boot', '300', '--tag', tag]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        print(tag, 'FAILED\n', r.stdout[-800:], r.stderr[-800:]); continue
    j = json.load(open(os.path.join(REC, 'pooled_fit', REF + '_' + tag, 'pooled_summary.json')))
    rows.append(dict(min_blocks=c['min_blocks'], magcut=c['magcut'], vet=c['vet'], vet_k=c['vet_k'],
                     observations=j['observations'], stars=j['stars'], L=j['L'],
                     sigma_boot=j['sigma_bootstrap'], sigma_formal=j['sigma_formal'], residual=j['residual']))
    print('  blocks>=%d  G<=%2.0f  vet %d x %.0f MAD:  %3d obs / %3d stars   L = %.3f +- %.3f   residual %.3f"'
          % (c['min_blocks'], c['magcut'], c['vet'], c['vet_k'], j['observations'], j['stars'], j['L'], j['sigma_bootstrap'], j['residual']), flush=True)
G = pd.DataFrame(rows)
G.to_csv(os.path.join(REC, 'pooled_fit', 'grid_%s.csv' % REF), index=False)
print('\n->', os.path.join(REC, 'pooled_fit', 'grid_%s.csv' % REF))
