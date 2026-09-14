"""L against the plate scale for cell 4: the covariance chart cells 1 and 3 draw.

Douglas, 2026-09-11: make a `record_covariance.png` like Leon's.

WHY THIS CHART IS THE RIGHT ONE FOR CELL 4.  Everything difficult about Husillos has turned
out to be the plate scale rather than the deflection: the Method 1 - Method 2 gap is the scale
difference times this field's lever to within a thousandth of an arcsec, hosting the union on
either block moves the scale 70 ppm and L not at all, and CalibS' own scale has walked 18.6 ppm
-- about its own sigma -- under two corrections.  A chart with L on one axis and the scale on
the other is the picture of exactly that.

METHOD 1 AND METHOD 2 COME FROM DIFFERENT RUNS, AND MUST.  The first version of this chart
took both ellipses from ONE stage-3 run, because `eclipse_method='Method 1 & 2'` prints both
and they sat next to each other in the log.  That run's stage 2 was the METHOD 1 pathway --
`constant` with `distortion_free_scale=False`, so the plate scale was IMPORTED FROM CalibS and
the linear and quadratic were frozen from it too.  Running Method 2's estimator on those
residuals does not make it Method 2: Method 2 for this cell is the eclipse field fitted
STRAIGHT AGAINST THE ZENITH at `quadratic`, with the scale FITTED ON THE FIELD
(`plate scale source: fitted on this field`).  Husillos has no reduced calibration field in
that pathway at all -- it SKIPS the ladder's middle rung, as Station 1 does.  Douglas,
2026-09-11: "Method 2 should not use an imported platescale."  Quite so, and the mistake was
taking two numbers from one log because they were adjacent rather than because each was the
right object -- the same error as the earlier wrong-rung ones.

So: Method 1 from the m1s_ union run (scale imported, correct for Method 1), Method 2 from
the union run fitted against the zenith with a free scale.

("zenith rung" appeared in an earlier version of this file and in the eclipse record.  It was
my coinage, it is in no document, and it is misleading: the ladder's rungs are the zenith
reference, the daytime L/R calibration and the eclipse field, so "the zenith rung" would mean
the FIRST one.  What is meant here is the eclipse field skipping the middle rung.)  They are different pathways by definition, and a chart
that draws them from one run is drawing one pathway twice.

A CONSEQUENCE WORTH STATING ON THE CHART: because the two ellipses now come from different
DISTORTION MODELS and not merely different scale choices, the gap between them is NOT simply
the scale difference times the lever.  Within one run it was, to a thousandth of an arcsec.
Across these two it is not, and reading it that way would be wrong.

THE NUMBERS COME FROM STAGE 3'S OWN COVARIANCE, not from a re-fit.  `eclipse_analysis`
builds both matrices (`analysis_mode_1`, `analysis_mode_2`) and prints them, but writes them
to no machine-readable file, so they are parsed out of the run's stdout log.  Method 1's cov1
already carries the imported-scale term added in quadrature
(`cov2[0,0] = cov[0,0] + plate_covariance2**2`), which is what makes its ellipse taller and
tilted; Method 2's is the free-scale fit's own.

THE VERTICAL AXIS IS ppm FROM THE IMPORTED SCALE, and BOTH ellipses are put on that one base.
Stage 3 reports each method's scale in absolute "/px, so both are converted here against the
SAME reference -- Method 1's imported value.  Station 2's chart was drawn once with Method 2's
scale on the wrong base and had to be withdrawn and redrawn (`docs/STEP3_2026.md`); this is
the trap that produced that, and the reason the conversion is done in one place with the
reference named.

Drawn through `tools/record_charts.covariance_chart`, not re-implemented.

    .venv/Scripts/python.exe tools/husillos2026/hu_covariance.py
"""
import glob
import io
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
from record_charts import (covariance_chart, covariance_chart_single,  # noqa: E402
                           packed_box)
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
from hu_record import publish  # noqa: E402

HUS = r'F:\MEE_output\husillos2026'
RECORD = r'F:\MEE_output\RECORD\husillos2026'
OUT = os.path.join(HUS, 'charts')

#: Method 1: the two-gain union, two-witness, importing the SETTLED CalibS (frames 21-81,
#: eclipse detection settings). Its stage 2 is `constant` + free_scale off, so the scale is
#: imported -- which is what Method 1 means.
LOG_M1 = os.path.join(HUS, 'step3', 'union', 'm1s_host_gain125',
                      'method2_union_2witness', 'stage3.log')
#: Method 2: the same union on the ZENITH RUNG -- `quadratic`, scale fitted on the field.
#: A different run, necessarily.
LOG_M2 = os.path.join(HUS, 'step3', 'union', 'host_gain125',
                      'method2_union_2witness', 'stage3.log')

#: this field's own lever, h = 1/mean(1/r^2) = 34.24 R_sun^2 on its 63 stars (hu_step3 compare)
LEVER = 0.0324          # arcsec of L per ppm of scale
ATM_ERR = None          # cell 4 has no atmospheric term yet: see section 3i


def parse(log):
    """Both covariance matrices and means, out of stage 3's own printed output."""
    # The two blocks do NOT print alike: Method 1 writes "final cov mu" with a
    # comma-separated list wrapped in np.float64(...), Method 2 writes "final cov, mu" with a
    # bare space-separated list. A regex written for one silently finds only one block, which
    # is how the first version of this parser failed. So the numbers are pulled out
    # positionally instead: six floats after each header, whatever punctuation surrounds them.
    s = io.open(log, encoding='utf-8', errors='replace').read()
    num = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    out = []
    for m in re.finditer(r'final cov[, ]+mu', s):
        # `np.float64(` CONTAINS THE DIGITS 64, and a bare number-finder happily returns it
        # as a value: the first version of this read L = 64.000 "/px and carried on. The
        # wrapper is stripped before any number is looked for.
        tail = s[m.end():m.end() + 400].replace('np.float64(', '').replace('np.', '')
        g = [float(x) for x in re.findall(num, tail)[:6]]
        if len(g) < 6:
            continue
        out.append((np.array([[g[0], g[1]], [g[2], g[3]]]), np.array([g[4], g[5]])))
    if not out:
        raise SystemExit('no covariance block in %s' % log)
    # A parse that lands on the wrong numbers is far more dangerous than one that fails, so
    # the two things we know about these values are asserted: the plate scale is ~2.2 "/px on
    # this rig, and L is an arcsec-scale quantity.
    for C, mu in out:
        assert 2.0 < mu[1] < 2.5, 'parsed plate scale %.4f is not a Husillos scale' % mu[1]
        assert -10 < mu[0] < 20, 'parsed L %.4f is not an arcsec-scale deflection' % mu[0]
    return out


def to_ppm(C, mu, base):
    """(arcsec, arcsec/px) -> (arcsec, ppm from `base`).  One reference for both methods."""
    f = 1e6 / base
    Cp = np.array([[C[0, 0], C[0, 1] * f], [C[1, 0] * f, C[1, 1] * f * f]])
    return Cp, np.array([mu[0], (mu[1] - base) * f])


def main():
    got1 = parse(LOG_M1)
    got2 = parse(LOG_M2)
    # the Method 1 run prints BOTH estimators; take only its first block, which is Method 1's
    (C1, mu1) = got1[0]
    # the Method 2 run was asked for Method 2 alone, so it prints one block
    (C2, mu2) = got2[-1]
    base = mu1[1]                       # the imported scale IS Method 1's mean
    C1p, mu1p = to_ppm(C1, mu1, base)
    C2p, mu2p = to_ppm(C2, mu2, base)
    assert abs(mu1p[1]) < 1e-9, 'Method 1 must sit at 0 ppm by construction'

    sL1, sS1 = np.sqrt(C1p[0, 0]), np.sqrt(C1p[1, 1])
    sL2, sS2 = np.sqrt(C2p[0, 0]), np.sqrt(C2p[1, 1])
    print('Method 1  L = %.3f +- %.3f "   scale imported %.7f "/px +- %.1f ppm'
          % (mu1p[0], sL1, base, sS1))
    print('Method 2  L = %.3f +- %.3f "   scale %+.1f +- %.1f ppm from imported '
          '(fitted on the field)' % (mu2p[0], sL2, mu2p[1], sS2))
    print('gap %.3f " ; the lever alone would give %.1f ppm x %.4f = %.3f " -- these two come '
          'from DIFFERENT distortion models, so they need not agree'
          % (mu1p[0] - mu2p[0], mu2p[1], LEVER, mu2p[1] * LEVER))
    print('correlation: Method 1 %.2f, Method 2 %.2f'
          % (C1p[0, 1] / (sL1 * sS1), C2p[0, 1] / (sL2 * sS2)))

    # Douglas, 2026-09-11: keep only the two methods' own lines. The caveats that used to sit
    # here in black (the lever, the missing atmospheric term, CalibS' one-sidedness and its
    # 18.6 ppm walk) belong in the record document, not on the face of the chart -- they made
    # the box big enough to cover an ellipse, which is how Method 2 came to be invisible.
    lines = [
        ('Method 1:  L = %.3f $\\pm$ %.3f"' % (mu1p[0], sL1), 'darkred'),
        ('      scale imported from CalibS, %.7f "/px $\\pm$ %.1f ppm' % (base, sS1),
         'darkred'),
        ('Method 2:  L = %.3f $\\pm$ %.3f"' % (mu2p[0], sL2), 'tab:blue'),
        ('      scale fitted on the field, %+.1f $\\pm$ %.1f ppm from CalibS\u2019'
         % (mu2p[1], sS2), 'tab:blue'),
    ]
    fig, ax = covariance_chart(
        C1p, mu1p, C2p, mu2p, lines,
        'Husillos 2026 \u2014 L against the plate scale, two-gain union, '
        '63 two-witness stars',
        # no Newton line (Douglas, 2026-09-11)
        newton=False,
        name2='Method 2 (scale fitted on the field)')
    # THE BOX WAS COVERING METHOD 2'S ELLIPSE. Method 1 sits at 0 ppm and Method 2 at -162,
    # so covariance_chart's default lower-left box lands exactly on top of it: the ellipse was
    # drawn and then hidden, which is worse than omitting it. covariance_chart takes no
    # placement argument (an earlier edit here invented a `box_loc` it does not have), so the
    # box it made is removed and replaced. Done in this tool, not in record_charts, because
    # this is cell 4's geometry rather than a property of the chart type.
    for art in list(ax.artists):
        art.remove()

    # PLACE THE BOX WHERE IT CANNOT COVER AN ELLIPSE, and check that it does not.
    # Method 1 sits near 0 ppm to the right, Method 2 near -162 ppm to the left, so the free
    # quadrant is upper-left. Rather than trust that, each candidate corner is tested against
    # both ellipses' bounding boxes in axes coordinates and the first clear one is used.
    def ell_bbox(C, mu):
        """1-sigma ellipse extent (x0, x1, y0, y1): the ellipse never exceeds +-sqrt(var)."""
        return (mu[0] - C[0, 0] ** 0.5, mu[0] + C[0, 0] ** 0.5,
                mu[1] - C[1, 1] ** 0.5, mu[1] + C[1, 1] ** 0.5)

    xl, yl = ax.get_xlim(), ax.get_ylim()

    def to_axes(b):
        return ((b[0] - xl[0]) / (xl[1] - xl[0]), (b[1] - xl[0]) / (xl[1] - xl[0]),
                (b[2] - yl[0]) / (yl[1] - yl[0]), (b[3] - yl[0]) / (yl[1] - yl[0]))

    ells = [to_axes(ell_bbox(C1p, mu1p)), to_axes(ell_bbox(C2p, mu2p))]
    fig.canvas.draw()                      # so the box's size is known before it is judged
    for loc, anchor in (('upper left', (0.02, 0.98)), ('lower right', (0.98, 0.02)),
                        ('lower left', (0.02, 0.02)), ('upper right', (0.98, 0.98))):
        box = packed_box(ax, lines, loc=loc, anchor=anchor)
        fig.canvas.draw()
        bb = box.get_window_extent().transformed(ax.transAxes.inverted())
        clash = any(not (bb.x1 < e[0] or bb.x0 > e[1] or bb.y1 < e[2] or bb.y0 > e[3])
                    for e in ells)
        if not clash:
            print('   text box at %-12s clear of both ellipses' % loc)
            break
        box.remove()
        print('   text box at %-12s would overlap an ellipse; trying the next corner' % loc)
    else:
        raise SystemExit('no corner is clear of both ellipses -- widen the axes')
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'record_covariance.png'), dpi=130)
    # --- and the Method 2 alone variant, built to match cell 2's record_covariance.png
    #
    # Douglas, 2026-09-11: "was record_covariance_method2.png done the same way as
    # record_covariance.png in mexico2024?  Can we make it look more similar."  It was not --
    # it carried only the two blue lines.  Cell 2's chart
    # (`tools/matrix_station1/s1_charts_record.py` through `covariance_chart_single`) puts the
    # plate scale with its ABSOLUTE error, the L-vs-scale correlation and the sample size in
    # the box, on an absolute plate-scale axis with no Newton line, and titles it "L and plate
    # scale -- <cell>, <what was fitted>".  All of that is matched here.
    #
    # The one line that CANNOT be matched is cell 2's second blue line, "+- 0.139" with the
    # atmosphere term 0.11".  Cell 4 has no atmosphere term (section 3i), so that slot says so
    # rather than being quietly dropped: an absent term should be visible on the chart, not
    # inferred from its absence.
    ps_abs_err = sS2 * 1e-6 * mu2[1]
    n_stars = 63
    lines2 = [
        ('Method 2:  L = %.3f $\\pm$ %.3f" (stat)' % (mu2[0], sL2), 'tab:blue'),
        ('      NO atmosphere term yet \u2014 every other cell carries \u00b10.11\u20130.33"',
         'tab:blue'),
        ('Plate scale: %.7f $\\pm$ %.1f$\\times$10$^{-5}$ "/px'
         % (mu2[1], ps_abs_err * 1e5), 'black'),
        ('      (plate scale found by fitting S with L, %.1f ppm)' % sS2, 'black'),
        ('correlation L vs plate scale = %.2f' % (C2[0, 1] / (sL2 * np.sqrt(C2[1, 1]))),
         'black'),
        ('from %d stars, each measured in BOTH gain blocks (%d observations)'
         % (n_stars, 2 * n_stars), 'black'),
    ]
    fig2, ax2 = covariance_chart_single(
        C2, mu2, lines2,
        'L and plate scale \u2014 Husillos 2026, two-gain union, one fitted scale',
        name='Method 2 (scale fitted with L)')
    fig2.savefig(os.path.join(OUT, 'record_covariance_method2.png'), dpi=130)
    publish(['record_covariance.png', 'record_covariance_method2.png'], OUT)


if __name__ == '__main__':
    main()
