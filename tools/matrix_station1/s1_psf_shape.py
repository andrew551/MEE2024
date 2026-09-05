"""The PSF's shape across the frame, on any of the matrix's stacks -- which way the tail points,
and which way it blurs.

Douglas, 2026-09-05: if the moment estimator's magnitude bias is "the estimator on this PSF",
that implies an optical asymmetry in the telescope. Coma's teardrop can point inward or
outward, and the blur can be tangential rather than radial. Which is it here -- and, for the
question of what the program's DEFAULT centroid estimator should be, what do the other
instruments in the matrix look like: the FRA500 + reducer at Leon 2026, the Askar 65PHQ at
Leakey 2024?

Rather than infer the PSF from the estimator's behaviour, measure it. For every bright,
unsaturated star on a stack, take a cutout, remove the local background, and compute the light
distribution's second and third moments in the RADIAL/TANGENTIAL frame about the frame centre:

    sigma_r, sigma_t   the widths along and across the radius     -> radial or tangential blur
    skew_r             the third moment along the radius, signed  -> which way the tail points
                       (+ = tail outward, away from the frame centre; - = tail inward)
    skew_t             the tangential third moment, ~0 for any axisymmetric aberration: the control

and, on the same cutout, the two centroids the argument is about -- the footprint moment
(pixels above 3 sigma within the aperture) and the windowed (Gaussian sigma 2 px, iterated) --
and their radial difference, by magnitude. A comatic PSF with the tail pointing outward puts the
moment centroid outward of the windowed one, more so for bright stars whose footprint reaches
further into the tail; inward tail, the reverse. The magnitude dependence of that offset IS the
estimator bias that leaks into L; the windowed estimator's own offset is what the distortion
model absorbs, if it is the same on the calibration and the science field.

Sets: Station 1 zenith (3 s, night) and the four corona-subtracted eclipse stacks; Leon 2026
zenith12 (three fields); Leakey 2024 zenith1 (three blocks). Frame size is read from each stack,
the plate scale from its stage-2 results. sigma is the second moment of the clipped footprint --
smaller than the full PSF sigma -- so read ratios and signs, not absolute widths.

Writes station1_record/psf_shape.csv.
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from astropy.io import fits

REC = r"D:/MEE2024 output/MEE_output/station1_record"
Z24 = r"D:/MEE2024 output/Station 1/zenith fields"
LEON = r"D:/MEE2024 output/MEE_output/refraction/zenith12"
LEAKEY = r"D:/MEE2024 output/MEE_output/leakey_zenith/zenith1"
EDGES = [0, 0.3, 0.5, 0.7, 0.85, 1.0]
GMIN, GMAX = 8.0, 11.5
BOX, RAP, RIN, ROUT = 15, 8.0, 11.0, 15.0     # cutout half-size, aperture, background annulus (px)


def matched(zp):
    zf = zipfile.ZipFile(zp)
    d = pd.read_csv(zf.open([n for n in zf.namelist() if n.endswith('CATALOGUE_MATCHED_ERRORS.csv')][0]))
    d.columns = [c.strip() for c in d.columns]
    ps = json.load(zf.open('distortion_results.txt'))['platescale (arcseconds/pixel)']
    return d[(d.flag_is_outlier == False) & (d.magV >= GMIN) & (d.magV <= GMAX)], ps


def shape(img, x0, y0, NX, NY, PS, full_scale):
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
    w = np.where(ap & (f > 3*noise), f, 0.0)
    if w.sum() <= 0 or (w > 0).sum() < 5:
        return None
    mx, my = (w*xx).sum()/w.sum(), (w*yy).sum()/w.sum()
    dx, dy = xx-mx, yy-my
    rad = np.hypot(x0-NX/2, y0-NY/2)
    ux, uy = (x0-NX/2)/rad, (y0-NY/2)/rad
    tx, ty = -uy, ux
    dr, dt = dx*ux + dy*uy, dx*tx + dy*ty
    var_r, var_t = (w*dr*dr).sum()/w.sum(), (w*dt*dt).sum()/w.sum()
    if var_r <= 0 or var_t <= 0:
        return None
    skew_r = (w*dr**3).sum()/w.sum()/var_r**1.5
    skew_t = (w*dt**3).sum()/w.sum()/var_t**1.5
    wx, wy = mx, my
    for _ in range(6):
        g = np.exp(-((xx-wx)**2 + (yy-wy)**2)/(2*2.0**2))
        ww = np.clip(f, 0, None)*g*ap
        if ww.sum() <= 0:
            return None
        wx, wy = (ww*xx).sum()/ww.sum(), (ww*yy).sum()/ww.sum()
    d_rad = ((mx-wx)*ux + (my-wy)*uy)*PS*1000
    return dict(sig_r=np.sqrt(var_r), sig_t=np.sqrt(var_t), skew_r=skew_r, skew_t=skew_t,
                mom_minus_win_rad_mas=d_rad, sn=(cut.max()-bg)/max(noise, 1e-6))


def measure(kind, label, img_path, zp, full_scale):
    img = fits.getdata(img_path)
    NY, NX = img.shape
    d, PS = matched(zp)
    half = np.hypot(NX/2, NY/2)
    out = []
    for _, s in d.iterrows():
        r = shape(img, s.px, s.py, NX, NY, PS, full_scale)
        if r:
            r.update(kind=kind, field=label, magV=s.magV, r_frac=np.hypot(s.px-NX/2, s.py-NY/2)/half, PS=PS, NX=NX, NY=NY)
            out.append(r)
    print('%-26s %4d stars G %.1f-%.1f, %dx%d px, %.4f "/px' % (label, len(out), GMIN, GMAX, NX, NY, PS), flush=True)
    return pd.DataFrame(out)


def first_zip(*parts):
    hit = sorted(glob.glob(os.path.join(*parts), recursive=True))
    return hit[-1] if hit else None


sets = []
for zp in sorted(glob.glob(os.path.join(REC, 'zenith_recentroid_tol', 'tol0p5', '*', 'distortion_data*.zip')))[:3]:
    stamp = os.path.basename(os.path.dirname(zp))
    img = glob.glob(os.path.join(Z24, 'CENTROID_OUTPUT' + stamp, 'STACKED%s.fit' % stamp))
    if img:
        sets.append(('Station 1 zenith', 'S1 zenith ' + stamp[-6:], img[0], zp, 65535.0))
for tag in ('0p25s_1810', '0p3s_1811', '0p4s_1812', '0p3s_1813'):
    img = sorted(glob.glob(os.path.join(REC, 'eclipse_corona', tag, 'CENTROID_OUTPUT*', 'STACKED_FLOAT*.fit')))
    zp = first_zip(REC, 'eclipse_corona', tag, 'stage2_twopass_reftol0p5', '**', 'distortion_data*.zip')
    if img and zp:
        sets.append(('Station 1 eclipse', 'S1 eclipse ' + tag, img[-1], zp, 1e9))
for fld in sorted(glob.glob(os.path.join(LEON, '*')))[:3]:
    img = sorted(glob.glob(os.path.join(fld, 'CENTROID_OUTPUT*', 'STACKED*.fit')))
    zp = first_zip(fld, 'stage2', '**', 'distortion_data*.zip')
    if img and zp:
        sets.append(('Leon 2026 zenith', 'Leon ' + os.path.basename(fld), img[-1], zp, 65535.0))
for blk in sorted(glob.glob(os.path.join(LEAKEY, '*')))[:3]:
    img = sorted(glob.glob(os.path.join(blk, 'CENTROID_OUTPUT*', 'STACKED*.fit')))
    zp = first_zip(blk, 'stage2', '**', 'distortion_data*.zip')
    if img and zp:
        sets.append(('Leakey 2024 zenith', 'Leakey ' + os.path.basename(blk), img[-1], zp, 65535.0))

frames = [measure(*s) for s in sets]
allf = pd.concat([f for f in frames if len(f)], ignore_index=True)
allf.to_csv(os.path.join(REC, 'psf_shape.csv'), index=False)


def med(x):
    return np.median(x) if len(x) else np.nan


print('\nby frame radius (fraction of the half-diagonal): medians; sigma in px, skew dimensionless, dR in mas')
print('%-20s %-10s %5s %7s %7s %8s %8s %8s %10s' % ('instrument', 'r', 'n', 'sig_r', 'sig_t', 'sig_r/t', 'skew_r', 'skew_t', 'mom-win_R'))
for kind in allf.kind.unique():
    a = allf[allf.kind == kind]
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        b = a[(a.r_frac >= lo) & (a.r_frac < hi)]
        print('%-20s %.2f-%.2f %5d %7.2f %7.2f %8.3f %+8.3f %+8.3f %+10.0f'
              % (kind, lo, hi, len(b), med(b.sig_r), med(b.sig_t), med(b.sig_r/b.sig_t), med(b.skew_r), med(b.skew_t), med(b.mom_minus_win_rad_mas)))
print('\nthe moment-minus-windowed radial offset against magnitude, outer half of the frame (r > 0.5); + = moment centroid outward')
for kind in allf.kind.unique():
    a = allf[(allf.kind == kind) & (allf.r_frac > 0.5)]
    cells = []
    for lo, hi in ((8.0, 9.5), (9.5, 10.5), (10.5, 11.5)):
        b = a[(a.magV >= lo) & (a.magV < hi)]
        cells.append('G %.1f-%.1f: %+5.0f mas (n=%3d)' % (lo, hi, med(b.mom_minus_win_rad_mas), len(b)))
    # the slope in mas per magnitude, a straight line through the individual stars
    if len(a) > 10:
        k = np.polyfit(a.magV.values, a.mom_minus_win_rad_mas.values, 1)[0]
        cells.append('slope %+5.0f mas/mag' % k)
    print('  %-20s %s' % (kind, '   '.join(cells)))
print('\nreading: skew_r > 0 = tail outward (away from the frame centre); sig_r/sig_t > 1 = radial blur, < 1 = tangential')
print('->', os.path.join(REC, 'psf_shape.csv'))
