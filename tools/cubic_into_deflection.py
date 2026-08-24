"""How much of a cubic-amplitude error reaches the deflection constant.

`docs/LEON_2026-08-11.md` section 9.2 item 4 asks for this and records that it has never
been computed: "5% of the cubic at the radii the eclipse stars actually occupy is a
computable contribution to the Einstein coefficient, and it has not been computed."

The question is not "how big is the cubic error" -- section 18.6 measured that -- but how
much of it survives the three-step transfer of section 16 and lands on L. Those are very
different numbers, because two stages of the transfer remove part of it:

  step 2  CAL_piLeo re-fits constant, linear and quadratic with the cubic frozen, so the
          best degree-<=2 approximation of the error over CAL_piLeo's stars is absorbed
          into the coefficients step 3 then imports;
  step 3  the eclipse field fits its own constant offset, removing the mean;
  stage 3 Method 2 additionally fits a term linear in radius, which removes more still --
          and, as it turns out, makes matters worse rather than better.

What is left is projected onto the Sun-radial direction (which is what `eclipse_analysis`
measures: `deflection_obs = radial_distance_obs - radial_distance_catalog`) and fitted the
way stage 3 fits it, so the output is directly comparable with the other entries in
`docs/bench/ERROR_BUDGET.md`.

Geometry is the Leon rig, read from `SCI_ladder_00001.fits`: ASI2600MM Pro 6248 x 4176
unbinned, 3.76 um pixels, FOCALLEN 350 -> 2.216 "/px, a 3.85 x 2.57 degree field. The
cubic amplitude d(3000 px) = 3.1048" is the handed-forward value of section 18.8.

Two assumptions are worth stating because they are not measured here. Both star fields are
taken as uniform over the sensor -- the real CAL_piLeo and eclipse distributions are not,
and a lopsided distribution absorbs the error less well. And the injected error is a pure
change of cubic *amplitude* with the shape held fixed, which is what section 18.6 measured
(ovality, position angle and trefoil all agree within 1 sigma between the two nights; only
|B| differs). A shape change would not propagate the same way.

Run:  .venv/Scripts/python.exe tools/cubic_into_deflection.py
"""

import argparse

import numpy as np

# --- the Leon eclipse geometry, from the frame headers ------------------------------
NX, NY = 6248, 4176           # SCI_ladder NAXIS1, NAXIS2, XBINNING 1
SCALE = 2.216                 # arcsec/px, from XPIXSZ 3.76 and FOCALLEN 350
D3 = 3.1048                   # arcsec, cubic radial displacement at r = 3000 px (18.8)
SUN_RADIUS_ARCSEC = 961.0
R_SUN = SUN_RADIUS_ARCSEC / SCALE          # solar radius in pixels
ALPHA_LIMB = 1.7512                        # arcsec; L, the quantity being measured

K = D3 / 3000.0 ** 3          # arcsec per px^3

# Where the Sun sat in the frame is the single biggest lever on the answer, which is
# Bruns' corner argument (ROADMAP 3a, LEON 10.5/11.4) made quantitative. Leon was
# "horizon-parallel with the Sun low in the frame" (section 5), hence the third row.
GEOMETRIES = {
    'centre': (0.0, 0.0),                    # no geometric protection at all
    'Leon-like': (0.0, -1400.0),             # "horizon-parallel, Sun low in the frame"
    'mid-edge': (0.0, -NY / 2),              # midpoint of a short edge
    'corner': (-2400.0, -1600.0),            # Bruns' placement
}

# The error sizes that section 18 actually establishes, so nothing here is invented.
CASES = [
    (0.0038, 'random, 12-field mean (18.8)'),
    (0.0240, 'systematic floor (18.8)'),
    (0.0484, 'night to night (18.6)'),
    (0.1000, '18.9 extrapolation, low'),
    (0.2400, '18.9 extrapolation, high'),
]


def cubic_error_field(x, y, eps):
    """Vector displacement, arcsec, of a fractional cubic-amplitude error `eps`.

    A radial displacement of magnitude k*r^3 is the vector k*r^2*(x, y), i.e. degree 3 in
    each component. That it is *odd* is why a quadratic fit removes so little of it: the
    constant and quadratic terms are even and therefore orthogonal to it over a symmetric
    star distribution, leaving only the linear (plate-scale) term to absorb anything.
    """
    r2 = x * x + y * y
    return eps * K * r2 * x, eps * K * r2 * y


def design_deg2(x, y):
    """Every 2-D polynomial term of degree <= 2 -- what step 2 is free to re-fit."""
    return np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])


def _sample(rng, n):
    return rng.uniform(-NX / 2, NX / 2, n), rng.uniform(-NY / 2, NY / 2, n)


def _annulus(rng, sun, n, rmin, rmax):
    """`n` stars uniform on the sensor, kept between rmin and rmax solar radii."""
    sx, sy = sun
    xs, ys = _sample(rng, n * 60)
    rho = np.hypot(xs - sx, ys - sy)
    keep = (rho >= rmin * R_SUN) & (rho <= rmax * R_SUN)
    return xs[keep][:n], ys[keep][:n], rho[keep][:n]


def propagate(eps, sun, rng, n_cal=94, n_ecl=100, rmin=2.0, rmax=9.0, trials=400):
    """Fractional bias on L, for Method 1 and Method 2.

    Returns (mean bias method 1, mean bias method 2), each as a fraction of L.
    """
    out1, out2 = [], []
    for _ in range(trials):
        # step 2: CAL_piLeo absorbs whatever a degree-<=2 fit can, over its own stars
        cx, cy = _sample(rng, n_cal)
        dxc, dyc = cubic_error_field(cx, cy, eps)
        Xc = design_deg2(cx, cy)
        coef_x = np.linalg.lstsq(Xc, dxc, rcond=None)[0]
        coef_y = np.linalg.lstsq(Xc, dyc, rcond=None)[0]

        xs, ys, rho = _annulus(rng, sun, n_ecl, rmin, rmax)
        if len(xs) < 20:
            continue

        # step 3 imports the frozen model, so the absorbed part is already subtracted...
        dx, dy = cubic_error_field(xs, ys, eps)
        Xe = design_deg2(xs, ys)
        dx = dx - Xe @ coef_x
        dy = dy - Xe @ coef_y
        # ...and then fits its own constant offset, and nothing else
        dx -= dx.mean()
        dy -= dy.mean()

        sx, sy = sun
        ux, uy = (xs - sx) / rho, (ys - sy) / rho
        contam = dx * ux + dy * uy                 # the Sun-radial component, arcsec

        r_solar = rho / R_SUN                      # stage 3 works in solar radii
        meas = ALPHA_LIMB / r_solar + contam

        A1 = np.linalg.lstsq(np.c_[1 / r_solar], meas, rcond=None)[0][0]
        A2 = np.linalg.lstsq(np.c_[1 / r_solar, r_solar], meas, rcond=None)[0][0]
        out1.append(A1 / ALPHA_LIMB - 1)
        out2.append(A2 / ALPHA_LIMB - 1)
    return float(np.mean(out1)), float(np.mean(out2))


def statistical(sigma_mas, sun, rng, n=100, rmin=2.0, rmax=9.0, trials=2000):
    """Scatter on L from independent per-star centroid noise, for comparison."""
    out1, out2 = [], []
    for _ in range(trials):
        _, _, rho = _annulus(rng, sun, n, rmin, rmax)
        if len(rho) < 20:
            continue
        r_solar = rho / R_SUN
        meas = ALPHA_LIMB / r_solar + rng.normal(0, sigma_mas / 1000.0, len(r_solar))
        out1.append(np.linalg.lstsq(np.c_[1 / r_solar], meas, rcond=None)[0][0] / ALPHA_LIMB - 1)
        out2.append(np.linalg.lstsq(np.c_[1 / r_solar, r_solar], meas,
                                    rcond=None)[0][0] / ALPHA_LIMB - 1)
    return float(np.std(out1)), float(np.std(out2))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--trials', type=int, default=400)
    p.add_argument('--stars', type=int, default=100, help='eclipse stars in the annulus')
    p.add_argument('--rmin', type=float, default=2.0, help='inner radius, solar radii')
    p.add_argument('--rmax', type=float, default=9.0, help='outer radius, solar radii')
    p.add_argument('--seed', type=int, default=20260824)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    diag = np.hypot(NX / 2, NY / 2)
    print(f'sensor {NX} x {NY} px at {SCALE}"/px = '
          f'{NX * SCALE / 3600:.2f} x {NY * SCALE / 3600:.2f} deg')
    print(f'solar radius {R_SUN:.0f} px; annulus {args.rmin}-{args.rmax} R_sun = '
          f'{args.rmin * R_SUN:.0f}-{args.rmax * R_SUN:.0f} px')
    print(f'cubic displacement at the frame corner (r = {diag:.0f} px): '
          f'{K * diag ** 3:.3f}"\n')

    print(f'bias on L, per cent, Method 1 / Method 2, {args.stars} stars\n')
    print('Sun placement: centre = field centre, Leon-like = 1400 px off centre,')
    print("               mid-edge = short-edge midpoint, corner = Bruns' placement\n")
    print(f'{"cubic error":<37s}' + ''.join(f'{g:>17s}' for g in GEOMETRIES))
    for eps, label in CASES:
        cells = []
        for sun in GEOMETRIES.values():
            b1, b2 = propagate(eps, sun, rng, n_ecl=args.stars, rmin=args.rmin,
                               rmax=args.rmax, trials=args.trials)
            cells.append(f'{b1 * 100:+7.2f} /{b2 * 100:+7.2f} ')
        print(f'{eps * 100:6.2f}%  {label:<28s}' + ''.join(cells))

    print(f'\nfor comparison, scatter on L from per-star centroid noise '
          f'({args.stars} stars, Leon-like placement):')
    sun = GEOMETRIES['Leon-like']
    for sigma in (68, 120, 200, 300, 440):
        s1, s2 = statistical(sigma, sun, rng, n=args.stars,
                             rmin=args.rmin, rmax=args.rmax)
        print(f'   per-star {sigma:3d} mas   Method 1 +-{s1 * 100:5.2f}%   '
              f'Method 2 +-{s2 * 100:5.2f}%')


if __name__ == '__main__':
    main()
