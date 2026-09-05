"""Station 1: the PSF's shape across the frame -- which way the tail points, and which way it blurs.

Douglas, 2026-09-05: if the moment estimator's magnitude bias is "the estimator on this PSF",
that implies an optical asymmetry in the telescope. Coma's teardrop can point inward or
outward, and the blur can be tangential rather than radial. Which is it here?

Rather than infer the PSF from the estimator's behaviour, measure it. For every bright,
unsaturated, isolated star on a stack, take a cutout, remove the local background, and
compute the light distribution's second and third moments in the RADIAL/TANGENTIAL frame about
the frame centre:

    sigma_r, sigma_t   the widths along and across the radius     -> radial or tangential blur
    skew_r             the third moment along the radius, signed  -> which way the tail points
                       (+ = tail outward, away from the frame centre; - = tail inward)
    skew_t             the tangential third moment, which should be ~0 for any axisymmetric
                       aberration and is the control

and, on the same cutout, the two centroids the argument is about: the footprint moment and the
windowed (Gaussian sigma 2 px) centroid, and their radial difference. A comatic PSF with the
tail pointing outward puts the moment centroid outward of the windowed one, more so for bright
stars whose footprint reaches further into the tail; inward tail, the reverse. This is the
direct check on the sign the zenith bias diagnostic found ("bright inward").

Done on three zenith stacks (3 s, dark sky, the calibration) and the four corona-subtracted
eclipse stacks (0.25-0.4 s, daylight, after the refocus), so the two PSFs can be compared at
the same frame radius. Writes station1_record/psf_shape.csv.
"""
import glob, os, zipfile
import numpy as np, pandas as pd
from astropy.io import fits

REC = r"D:/MEE2024 output/MEE_output/station1_record"
Z24 = r"D:/MEE2024 output/Station 1/zenith fields"
NX, NY, PS = 9576, 6388, 1.84847
HALF = np.hypot(NX/2, NY/2)
EDGES = [0, 0.3, 0.5, 0.7, 0.85, 1.0]
GMIN, GMAX = 8.0, 11.5
BOX, RAP, RIN, ROUT = 15, 8.0, 11.0, 15.0     # half-size of the cutout, aperture, background annulus (px)


def matched(zp):
    zf = zipfile.ZipFile(zp)
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    return d[(d.flag_is_outlier == False) & (d.magV >= GMIN) & (d.magV <= GMAX)]


def shape(img, x0, y0, full_scale):
    """Second and third moments of the background-subtracted light within RAP of the peak, in
    the radial/tangential frame about the frame centre; plus the moment and windowed centroids."""
    xi, yi = int(round(x0)), int(round(y0))
    if not (BOX + ROUT < xi < NX - BOX - ROUT and BOX + ROUT < yi < NY - BOX - ROUT):
        return None
    cut = img[yi-BOX:yi+BOX+1, xi-BOX:xi+BOX+1].astype(float)
    if cut.max() >= 0.9*full_scale:
        return None
    yy, xx = np.mgrid[-BOX:BOX+1, -BOX:BOX+1]
    rr = np.hypot(xx, yy)
    ann = (rr >= RIN) & (rr <= ROUT)
    bg = np.median(cut[ann]); noise = 1.4826*np.median(np.abs(cut[ann]-bg))
    f = cut - bg
    ap = rr <= RAP
    w = np.where(ap & (f > 3*noise), f, 0.0)          # the footprint: pixels above 3 sigma
    if w.sum() <= 0 or (w > 0).sum() < 5:
        return None
    mx, my = (w*xx).sum()/w.sum(), (w*yy).sum()/w.sum()
    dx, dy = xx-mx, yy-my
    cxx, cyy, cxy = (w*dx*dx).sum()/w.sum(), (w*dy*dy).sum()/w.sum(), (w*dx*dy).sum()/w.sum()
    # radial / tangential unit vectors about the frame centre
    ux, uy = (x0-NX/2)/np.hypot(x0-NX/2, y0-NY/2), (y0-NY/2)/np.hypot(x0-NX/2, y0-NY/2)
    tx, ty = -uy, ux
    dr, dt = dx*ux + dy*uy, dx*tx + dy*ty
    var_r, var_t = (w*dr*dr).sum()/w.sum(), (w*dt*dt).sum()/w.sum()
    if var_r <= 0 or var_t <= 0:
        return None
    skew_r = (w*dr**3).sum()/w.sum()/var_r**1.5
    skew_t = (w*dt**3).sum()/w.sum()/var_t**1.5
    # the windowed centroid: Gaussian window sigma 2 px, iterated from the moment centroid
    wx, wy = mx, my
    for _ in range(6):
        g = np.exp(-((xx-wx)**2 + (yy-wy)**2)/(2*2.0**2))
        ww = np.clip(f, 0, None)*g*ap
        if ww.sum() <= 0:
            return None
        wx, wy = (ww*xx).sum()/ww.sum(), (ww*yy).sum()/ww.sum()
    d_rad = ((mx-wx)*ux + (my-wy)*uy)*PS*1000        # moment minus windowed, radial, mas (+ = moment outward)
    return dict(sig_r=np.sqrt(var_r), sig_t=np.sqrt(var_t), skew_r=skew_r, skew_t=skew_t,
                mom_minus_win_rad_mas=d_rad, peak=cut.max()-bg, sn=(cut.max()-bg)/max(noise, 1e-6))


def run(label, img_path, zp, full_scale):
    img = fits.getdata(img_path)
    d = matched(zp)
    out = []
    for _, s in d.iterrows():
        r = shape(img, s.px, s.py, full_scale)
        if r:
            r.update(field=label, magV=s.magV, r_frac=np.hypot(s.px-NX/2, s.py-NY/2)/HALF); out.append(r)
    return pd.DataFrame(out)


sets = []
zen = sorted(glob.glob(os.path.join(REC, 'zenith_recentroid_tol', 'tol0p5', '*', 'distortion_data*.zip')))[:3]
for zp in zen:
    stamp = os.path.basename(os.path.dirname(zp))
    img = glob.glob(os.path.join(Z24, 'CENTROID_OUTPUT' + stamp, 'STACKED%s.fit' % stamp))
    if img:
        sets.append(('zenith ' + stamp[-6:], img[0], zp, 65535.0))
for tag in ('0p25s_1810', '0p3s_1811', '0p4s_1812', '0p3s_1813'):
    img = sorted(glob.glob(os.path.join(REC, 'eclipse_corona', tag, 'CENTROID_OUTPUT*', 'STACKED_FLOAT*.fit')))
    zp = sorted(glob.glob(os.path.join(REC, 'eclipse_corona', tag, 'stage2_twopass_reftol0p5', '**', 'distortion_data*.zip'), recursive=True))
    if img and zp:
        sets.append(('eclipse ' + tag, img[-1], zp[-1], 1e9))    # the float stack has no hard saturation

frames = []
for label, img, zp, fs in sets:
    df = run(label, img, zp, fs); frames.append(df)
    print('%-18s %4d stars G %.1f-%.1f' % (label, len(df), GMIN, GMAX), flush=True)
allf = pd.concat(frames, ignore_index=True)
allf['kind'] = np.where(allf.field.str.startswith('zenith'), 'zenith', 'eclipse')
allf.to_csv(os.path.join(REC, 'psf_shape.csv'), index=False)


def med(x):
    return np.median(x) if len(x) else np.nan


print('\nby frame radius (fraction of the half-diagonal): medians; sigma in px, skew dimensionless, dR in mas')
print('%-8s %-10s %5s %7s %7s %8s %8s %8s %10s' % ('kind', 'r', 'n', 'sig_r', 'sig_t', 'sig_r/t', 'skew_r', 'skew_t', 'mom-win_R'))
for kind in ('zenith', 'eclipse'):
    a = allf[allf.kind == kind]
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        k = (a.r_frac >= lo) & (a.r_frac < hi)
        b = a[k]
        print('%-8s %.2f-%.2f %5d %7.2f %7.2f %8.3f %+8.3f %+8.3f %+10.0f'
              % (kind, lo, hi, len(b), med(b.sig_r), med(b.sig_t), med(b.sig_r/b.sig_t), med(b.skew_r), med(b.skew_t), med(b.mom_minus_win_rad_mas)))
print('\nthe moment-minus-windowed radial offset against magnitude, outer half of the frame (r > 0.5):')
for kind in ('zenith', 'eclipse'):
    a = allf[(allf.kind == kind) & (allf.r_frac > 0.5)]
    for lo, hi in ((8.0, 9.5), (9.5, 10.5), (10.5, 11.5)):
        b = a[(a.magV >= lo) & (a.magV < hi)]
        print('  %-8s G %.1f-%.1f  n=%4d  moment minus windowed, radial: %+5.0f mas (+ = moment centroid outward)'
              % (kind, lo, hi, len(b), med(b.mom_minus_win_rad_mas)))
print('\nreading: skew_r > 0 = the tail points outward (away from the frame centre); sig_r/sig_t > 1 = radial blur, < 1 = tangential')
print('->', os.path.join(REC, 'psf_shape.csv'))
