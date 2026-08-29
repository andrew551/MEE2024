"""Injection self-test + inner-star value table for the Bruns union."""
import os, sys
sys.argv = ['x']
HERE = r"C:/Users/dpesm/OneDrive/Documents/GitHub/MEE2024/tools/matrix_bruns"
src = open(os.path.join(HERE, 'b17_union.py'), encoding='utf-8').read()
exec(src.split("print()\nfor t in (")[0])
import numpy as np

# pure-L injection: every variant must return the injected value on THIS geometry
U, rx, ry, R = build_union(('EA','EB'), 2.0)
inj_x, inj_y = L_REF*(R_SUN_AS/R)*(rx/R), L_REF*(R_SUN_AS/R)*(ry/R)
for nd in (None, 1, 2, 3):
    A, labels = design(U.px.values, U.py.values, rx, ry, R, nd)
    b = np.concatenate([inj_x, inj_y])
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    got = c[labels.index('L')]
    print(f'pure-L self-test, nuisance deg {nd}: recovered {got:.4f} / {L_REF}'
          + ('   <-- IDENTIFIABILITY LOST' if abs(got-L_REF) > 0.02*L_REF else ''))

# per-star radial deflections (the value table), R>2 union, v-deg2 residual view
A, labels = design(U.px.values, U.py.values, rx, ry, R, 2)
b = np.concatenate([U.dx.values, U.dy.values])
c, *_ = np.linalg.lstsq(A, b, rcond=None)
iL = labels.index('L')
other = A@c - A[:, iL]*c[iL]
clean = b - other
n = len(U)
ddx, ddy = clean[:n], clean[n:]
defl = ddx*(rx/R) + ddy*(ry/R)
resid = b - A@c
print(f'\nresidual rms about the full model: {np.sqrt(np.mean(resid**2)):.3f} arcsec/axis')
print(f'{"G":>6} {"px":>6} {"py":>6} {"R/Rsun":>7} {"defl_rad (as)":>13} {"GR pred (as)":>12}')
for k in np.argsort(R):
    print(f'{U.mag.values[k]:6.2f} {U.px.values[k]:6.0f} {U.py.values[k]:6.0f} '
          f'{R[k]/R_SUN_AS:7.2f} {defl[k]:13.3f} {L_REF*R_SUN_AS/R[k]:12.3f}')

# the two inner stars, radially, from the R>1.45 union
U2, rx2, ry2, R2 = build_union(('EA','E2','EB'), 1.45)
inner = R2 < 2.0*R_SUN_AS
print('\ninner stars (E2 only):')
for k in np.where(inner)[0]:
    ur, vr = rx2[k]/R2[k], ry2[k]/R2[k]
    dr = U2.dx.values[k]*ur + U2.dy.values[k]*vr
    print(f'  G {U2.mag.values[k]:.2f} R={R2[k]/R_SUN_AS:.2f} Rsun: raw radial displacement '
          f'{dr:+.3f} as (GR predicts {L_REF*R_SUN_AS/R2[k]:+.3f}; E2 disk edge at 1.47 Rsun)')
