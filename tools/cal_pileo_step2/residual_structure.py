"""Is the CAL_piLeo stage-2 residual noise, or an unmodelled field term?

Weighting can only help the first. The magnitude table says the residual is flat with
brightness, which is not what photon-limited centroids look like -- so this asks whether
what is left is spatially structured, and in particular whether it carries the radial
cubic signature that a mis-transferred zenith cubic would leave (LEON 18.9).
"""
import numpy as np, pandas as pd

W, CX, CY, PS = 3124.0, 3124.0, 2088.0, 2.2054

def radial_decomposition(d, label):
    x, y = (d.px.values - CX) / W, (d.py.values - CY) / W
    r = np.hypot(x, y)
    dx, dy = d.dx_arcsec.values, d.dy_arcsec.values
    # radial and tangential components of the residual about the field centre
    ur, ut = np.array([x, y]) / r, np.array([-y, x]) / r
    rad = dx * ur[0] + dy * ur[1]
    tan = dx * ut[0] + dy * ut[1]
    print(f'\n=== {label}  N={len(d)} ===')
    print(f'  rms radial = {np.sqrt(np.mean(rad**2)):.4f}"   rms tangential = {np.sqrt(np.mean(tan**2)):.4f}"')

    # a leftover cubic shows as radial displacement growing like r^3
    for name, basis in (('r   (scale)', r), ('r^3 (cubic)', r**3)):
        c = np.dot(basis, rad) / np.dot(basis, basis)
        resid = rad - c * basis
        # bootstrap the coefficient so "detected" means something
        rng = np.random.default_rng(7)
        bs = [np.dot(basis[i], rad[i]) / np.dot(basis[i], basis[i])
              for i in (rng.integers(0, len(r), (400, len(r))))]
        sd = np.std(bs)
        edge = c * (1.0 if name.startswith('r ') else 1.0)   # value of the term at r=1
        print(f'  {name}: coeff {c:+.4f} +/- {sd:.4f}" at r=1  ({abs(c)/sd:.1f} sigma), '
              f'residual rms {np.sqrt(np.mean(resid**2)):.4f}"')

    # radial bins -- structure the polynomials might miss
    b = pd.cut(r, [0, .3, .5, .7, .85, 1.0, 1.3])
    g = pd.DataFrame({'r': r, 'rad': rad, 'tan': tan}).groupby(b, observed=True).agg(
        n=('r', 'size'), mean_rad=('rad', 'mean'), rms_rad=('rad', lambda v: np.mean(v**2)**.5),
        rms_tan=('tan', lambda v: np.mean(v**2)**.5))
    print('  by radius:   n   <radial>   rms_rad   rms_tan')
    for k, rw in g.iterrows():
        print(f'    {str(k):12s} {int(rw.n):3d}   {rw.mean_rad:+7.4f}"  {rw.rms_rad:7.4f}" {rw.rms_tan:7.4f}"')

    # x versus y anisotropy, which drives the y-heavy standard error
    print(f'  sigma_x = {dx.std(ddof=0):.4f}"  sigma_y = {dy.std(ddof=0):.4f}"  '
          f'ratio = {dy.std(ddof=0)/dx.std(ddof=0):.2f}')

base = r"D:/MEE2024 output/MEE_output/step2_ladder"
for tag, run in [("tol 0.5", "tolsweep_0.5/DISTORTION_OUTPUT20260825143416__20260825033557"),
                 ("tol 1.0", "tolsweep_1.0/DISTORTION_OUTPUT20260825143422__20260825033557"),
                 ("tol 5.0", "tolsweep_5.0/DISTORTION_OUTPUT20260825143434__20260825033557")]:
    radial_decomposition(pd.read_csv(f"{base}/{run}/distortion/TWOD_RESIDUALS.csv"), tag)
