"""S0 definitive scoring: pipeline-matched lists where the solve worked, the inherited
cubic solution where it did not; union keyed on catalogue position; h with the ephemeris
Sun centre. Mag to 12 (the pipeline matches to 13; the design field is G<=11)."""
import glob, json, zipfile
import numpy as np, pandas as pd
from mee2024.starcat import providers
B = r'D:/MEE2024 output/MEE_output/step3_s0_v4'
SUN = (3171.0, 3232.0); PS, RSUN = 2.2054043, 947.1

ds = [pd.read_csv(glob.glob(B+f'/{t}/stage2/**/CATALOGUE_MATCHED_ERRORS.csv', recursive=True)[0])
      for t in ('0p6s','1p2s')]
for d in ds: d.columns=[c.strip() for c in d.columns]
anchor = pd.concat(ds).drop_duplicates('ID')
ra, dec, px, py = anchor['RA(catalog)'].values, anchor['DEC(catalog)'].values, anchor['px'].values, anchor['py'].values
ra0, de0, dec0 = ra.mean(), dec.mean(), dec.mean()
def design(rr, dd):
    X=(rr-ra0)*np.cos(np.radians(dec0)); Y=dd-de0
    return np.c_[np.ones_like(X),X,Y,X*X,X*Y,Y*Y,X**3,X*X*Y,X*Y*Y,Y**3]
A=design(ra,dec)
cx,*_=np.linalg.lstsq(A,px,rcond=None); cy,*_=np.linalg.lstsq(A,py,rcond=None)
print(f'cubic from {len(anchor)} anchors, residual {np.hypot(A@cx-px,A@cy-py).mean():.2f} px')

prov = providers.GaiaOfflineProvider.from_installed()
t = prov.lookup((ra0-2.7,ra0+2.7),(de0-2.2,de0+2.2), 12.0)
cra,cdec,cmag = np.degrees(t.ra), np.degrees(t.dec), np.asarray(t.mag)
Ac=design(cra,cdec); cpx,cpy = Ac@cx, Ac@cy
on=(cpx>10)&(cpx<6238)&(cpy>10)&(cpy<4166)
cr=np.hypot(cpx-SUN[0],cpy-SUN[1])*PS/RSUN

union={}
def add(key, mag, r, src):
    if key not in union or r < union[key][1]: union[key]=(mag,r,src)
# pipeline matches (full model) for the solved tiers
for tname in ('0p6s','1p2s'):
    d=pd.read_csv(glob.glob(B+f'/{tname}/stage2/**/CATALOGUE_MATCHED_ERRORS.csv',recursive=True)[0])
    d.columns=[c.strip() for c in d.columns]
    for _,s in d.iterrows():
        r=np.hypot(s['px']-SUN[0],s['py']-SUN[1])*PS/RSUN
        add(s['ID'], s['magV'], r, tname)
# inherited-solution matches for the failed solves
for tname in ('0p1s','0p3s','all87','discard'):
    z=zipfile.ZipFile(glob.glob(B+f'/{tname}/centroid_data*.zip')[0])
    det=pd.read_csv(z.open('STACKED_CENTROIDS_DATA.csv'))
    hits=[]
    for i in np.where(on)[0]:
        dd=np.hypot(det['px']-cpx[i],det['py']-cpy[i])
        if dd.min()<3.0:
            hits.append((cmag[i],cr[i]))
            add(f'g{i}', cmag[i], cr[i], tname)
    hits.sort(key=lambda h:h[1])
    inner=', '.join(f'V{m:.1f}@{r:.2f}' for m,r in hits[:3])
    print(f'{tname:8} {len(det):4d} det -> {len(hits):3d} catalogue matches; innermost {inner}')

m=np.array([v[0] for v in union.values()]); r=np.array([v[1] for v in union.values()])
print(f'\n=== S0 FINAL UNION: {len(union)} stars, V {m.min():.2f}-{m.max():.2f} ===')
print(f'innermost {r.min():.2f} R_sun;  h = {1/np.mean(1/r**2):.1f} R_sun^2  (design 19.8; raw-frame S0 33.0)')
print(f"{'annulus':>9} {'cat G<=12':>9} {'union':>6}")
for lo,hi in [(1.4,2),(2,3),(3,4),(4,6),(6,8),(8,11)]:
    print(f'{lo:4.1f}-{hi:<4.0f} {((cr[on]>=lo)&(cr[on]<hi)).sum():9d} {((r>=lo)&(r<hi)).sum():6d}')
