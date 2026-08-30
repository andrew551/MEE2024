"""Which stars does the G-12 catalogue remove from the FULL union, and why?"""
import os, sys
lim = sys.argv[1]
os.environ['S2_LIMIT_MAG'] = lim
HERE = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024/tools"
src = open(os.path.join(HERE, 'step3_s2_union.py'), encoding='utf-8').read()
exec(src.split("print()\nfor t in (")[0])
import numpy as np
U, rx, ry, R = build_union(('0p1s','0p3s','0p6s','1p2s'))
print(f'--- limit {lim}: FULL union N={len(U)}')
for _, r in U.sort_values('mag').iterrows():
    print(f'STAR G {r.mag:5.2f} px ({r.px:6.0f},{r.py:6.0f}) ntier {int(r.ntier)} '
          f'spread {r.spread:5.2f}" dx {r.dx:+6.2f}" dy {r.dy:+6.2f}"')
# doubles diagnostic: which bright stars are now flagged double?
dbl = np.asarray(cat.is_double(10.0))
mags = np.asarray(cat.get_mags())
print(f'doubles in catalogue: {dbl.sum()} of {len(dbl)}; bright (G<=11) doubles:')
for i in np.where(dbl & (mags <= 11.0))[0]:
    print(f'DBL G {mags[i]:5.2f} at RA {np.degrees(cat.get_ra()[i]):9.4f} '
          f'DEC {np.degrees(cat.get_dec()[i]):+8.4f}')
