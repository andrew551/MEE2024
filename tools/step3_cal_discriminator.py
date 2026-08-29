import glob, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
HERE = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024/tools"
exec(open(os.path.join(HERE, 'step3_s2_union.py'), encoding='utf-8').read().split("print()\nfor t in (")[0])
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u
import pandas as pd

OUTP = r"D:/MEE2024 output/MEE_output/step3_s2_plots"
SITE = EarthLocation(lat=42.740470*u.deg, lon=-5.613780*u.deg, height=1101*u.m)
T = Time("2026-08-12T18:28:32", scale="utc")
U, rx, ry, R = build_union(('0p6s','1p2s'))
cra0, cdec0 = np.degrees(cat.get_ra()), np.degrees(cat.get_dec())
sc = SkyCoord(cra0[U.cat_i.values]*u.deg, cdec0[U.cat_i.values]*u.deg)
aa = sc.transform_to(AltAz(obstime=T, location=SITE))
sun_aa = SkyCoord(142.107*u.deg, 14.909*u.deg).transform_to(AltAz(obstime=T, location=SITE))
fig, ax = plt.subplots(figsize=(10, 7))
ax.scatter(aa.az.deg, aa.alt.deg, s=30, color='tab:blue', label='catalog', zorder=3)
ax.scatter(aa.az.deg, aa.alt.deg, marker='+', s=52, color='orange', label='observation (used)', zorder=4)
for i in range(len(U)):
    if U.mag.values[i] <= 8.6:
        ax.annotate(f" {U.mag.values[i]:.1f}", (aa.az.deg[i], aa.alt.deg[i]), fontsize=8)
ax.add_patch(Circle((sun_aa.az.deg, sun_aa.alt.deg), 947.1/3600, color='black', zorder=5))
ax.annotate('Sun', (sun_aa.az.deg+0.35, sun_aa.alt.deg-0.05), fontsize=10)
ax.set_xlabel('Azimuth (degrees)', fontsize=13); ax.set_ylabel('Altitude (degrees)', fontsize=13)
ax.set_title(f'Leon 2026 eclipse field in ALT/AZ at 18:28:32 UTC — {len(U)} stars, V {U.mag.min():.1f}-{U.mag.max():.1f}')
ax.set_aspect(1.0); ax.legend()
ax.invert_xaxis()   # az increases to the left = looking at the sky
fig.tight_layout(); fig.savefig(os.path.join(OUTP, 'field_altaz.png'), dpi=140); plt.close(fig)
print('alt-az field written; field spans alt', f'{aa.alt.deg.min():.1f}-{aa.alt.deg.max():.1f} deg')

# ---- CAL_piLeo as the daytime nuisance discriminator
calres = glob.glob(r'D:/MEE2024 output/MEE_output/cal_pileo_step2/canonical_16f_night2refs/**/TWOD_RESIDUALS.csv', recursive=True)[0]
dc = pd.read_csv(calres)
cdx, cdy = dc['dx_arcsec'].values, dc['dy_arcsec'].values
cdx, cdy = cdx - np.median(cdx), cdy - np.median(cdy)
print(f"\nCAL_piLeo residual field ({len(dc)} stars, after its own quadratic-free fit):")
print(f"  rms horizontal {np.std(cdx):.3f} arcsec, vertical {np.std(cdy):.3f} (V/H {np.std(cdy)/np.std(cdx):.2f})")
# fit the vertical deg-3 polynomial to CAL's residuals (its linear+quadratic are already
# absorbed by its own fit, so only the deg-3 vertical terms carry content)
xs, ys = (dc['px'].values-NX/2)/W_NORM, (dc['py'].values-NY/2)/W_NORM
terms3 = [(i, j) for i in range(4) for j in range(4-i) if i+j == 3]
M = np.column_stack([xs**i * ys**j for i, j in terms3])
c3, *_ = np.linalg.lstsq(M, cdy, rcond=None)
pred = M@c3
print(f"  deg-3 vertical fit absorbs {100*(1-np.var(cdy-pred)/np.var(cdy)):.0f} % of the vertical variance; "
      f"coefficients {np.round(c3, 3)} arcsec at frame edge")

# apply CAL's deg-3 vertical field to the science stars, then fit base + free v-deg2
xs_s, ys_s = (U.px.values-NX/2)/W_NORM, (U.py.values-NY/2)/W_NORM
Ms = np.column_stack([xs_s**i * ys_s**j for i, j in terms3])
dy_corr = U.dy.values - Ms@c3
U2 = U.copy(); U2['dy'] = dy_corr
L_free = fit_L(U, rx, ry, R, nuis_deg=2)
L_anch = fit_L(U2, rx, ry, R, nuis_deg=2)
L_anch_base = fit_L(U2, rx, ry, R)
print(f"\nscience fit (0.6+1.2 union, anchor in):")
print(f"  free v-deg2 nuisance:                 L = {L_free:+.3f} arcsec")
print(f"  CAL deg-3 field subtracted, + v-deg2: L = {L_anch:+.3f}")
print(f"  CAL deg-3 field subtracted, base:     L = {L_anch_base:+.3f}")
print(f"  shift from the CAL field: {L_anch-L_free:+.3f} arcsec -- the DISCRIMINATOR readout:")
print(f"  near zero = the two sightlines' high-order fields are decorrelated (Douglas'")
print(f"  patch hypothesis); large = a shared component exists and CAL constrains it")
