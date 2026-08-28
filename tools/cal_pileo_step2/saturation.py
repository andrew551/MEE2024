"""Which CAL_piLeo stars would F16 actually reject?

ROADMAP F16 specifies a saturation mask built as each RAW frame is read, shifted by the
integer offset the stacker applies and accumulated -- "clipped in N of 17 frames" -- because
a stacked-image test cannot see a clip that only the long exposures carry (the mean of 8
clipped 2 s frames and 9 unclipped 1 s frames dilutes it away).

This implements that measurement, so the leverage test in section 8 can use F16's real star
set instead of a magnitude cut standing in for it.
"""
import glob, json, zipfile
import numpy as np, pandas as pd
from astropy.io import fits

SAT = 65535          # 16-bit full scale; the ASI2600 writes clipped pixels here exactly
NEAR = 60000         # "near-clip" -- the cut the v1.3.6 stacks were checked against
BOX = 4

B = r"D:/MEE2024 output/MEE_output/cal_pileo_step2"
z = zipfile.ZipFile(glob.glob(f"{B}/s1_combined17/centroid_data*.zip")[0])
meta = json.load(z.open('results.txt'))
shifts = [(round(s[0]), round(s[1])) for s in meta['alignment']['shifts_px']]
files = eval(meta['source_files'])
assert len(files) == len(shifts) == 17

stars = pd.read_csv(glob.glob(f"{B}/definitive_tol1.0/**/TWOD_RESIDUALS.csv", recursive=True)[0])
peak = np.zeros((len(stars), 17))
exptime = []
for k, (f, sh) in enumerate(zip(files, shifts)):
    with fits.open(f) as hd:
        img = hd[0].data
        exptime.append(float(hd[0].header['EXPTIME']))
    ny, nx = img.shape
    # stack coords = raw coords + shift  =>  raw = stack - shift
    for i, s in stars.iterrows():
        r = int(round(s.py)) - sh[0]
        c = int(round(s.px)) - sh[1]
        if BOX <= r < ny-BOX and BOX <= c < nx-BOX:
            peak[i, k] = img[r-BOX:r+BOX+1, c-BOX:c+BOX+1].max()
        else:
            peak[i, k] = np.nan

exptime = np.array(exptime)
is1s, is2s = exptime == 1.0, exptime == 2.0
n_clip = np.nansum(peak >= SAT, axis=1)
n_near = np.nansum(peak >= NEAR, axis=1)
stars = stars.assign(peak_max=np.nanmax(peak, axis=1),
                     peak_1s=np.nanmax(np.where(is1s, peak, np.nan), axis=1),
                     peak_2s=np.nanmax(np.where(is2s, peak, np.nan), axis=1),
                     n_clip=n_clip, n_near=n_near)

print(f'frames: {is1s.sum()} x 1 s, {is2s.sum()} x 2 s')
print(f'\nstacked-image peak of the brightest star: '
      f'{fits.getdata(glob.glob(f"{B}/s1_combined17/CENTROID_OUTPUT*/STACKED_FLOAT*.fit")[0]).max():.0f}')
print(f'\nstars clipped at {SAT} in >=1 raw frame: {(n_clip>0).sum()} of {len(stars)}')
print(f'stars above {NEAR} in >=1 raw frame:   {(n_near>0).sum()} of {len(stars)}')
top = stars.nlargest(8, 'peak_max')[['magV','peak_max','peak_1s','peak_2s','n_clip','n_near','error_arcsec']]
print('\nbrightest eight by raw-frame peak:')
print(top.to_string(index=False, float_format=lambda v: f'{v:9.1f}'))
stars.to_csv(f'{B}/saturation_per_frame.csv', index=False)
print(f'\n-> {B}/saturation_per_frame.csv')
