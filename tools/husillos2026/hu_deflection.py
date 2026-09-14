"""The deflection chart for cell 4: radial deflection against radius, cell 2's construction.

Douglas, 2026-09-11: make one like `RECORD/mexico2024/record_deflection.png`.

Two charts, because cell 4 has a question cell 2 does not:

  record_deflection.png            the reduction of record -- the two-gain union under the
                                   two-witness rule, 63 stars, one colour
  record_deflection_by_block.png   the SAME stars, coloured by which gain block measured them,
                                   which is the picture of the disagreement that has run through
                                   this whole cell (gain 0 reads 1.596 +- 0.507, gain 125 reads
                                   2.782 +- 0.540 on identical stars)

Drawn through `tools/record_charts.reference_curves` and published through
`hu_record.publish()`, so a re-run supersedes rather than overwrites.

THE BAND IS STATISTICAL ONLY, and says so on its own label.  Cell 2's band is
sqrt(stat^2 + atmosphere^2); cell 4 HAS NO ATMOSPHERIC TERM (section 3i -- the horizon fields
that would supply one are not reduced), so drawing a total band here would be inventing the
missing piece.  The chart shows what is measured and the caption names what is absent.

Data comes from stage 3's own printed arrays -- `radial distances` in R_sun and
`deflection (arcsec)` -- not from a re-fit, so the points on the chart are the points the
quoted L was fitted to.

    .venv/Scripts/python.exe tools/husillos2026/hu_deflection.py
"""
import glob
import io
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'tools'))
sys.path.insert(0, os.path.join(REPO, 'tools', 'husillos2026'))
from record_charts import reference_curves  # noqa: E402
from hu_record import publish  # noqa: E402

HUS = r'F:\MEE_output\husillos2026'
OUT = os.path.join(HUS, 'charts')

#: the reduction of record: two-gain union, two-witness, fitted straight against the zenith
#: with a free scale (Method 2). NOT the Method 1 pathway -- its imported scale has walked
#: 18.6 ppm under two corrections (section 3n) and is not quotable.
UNION = os.path.join(HUS, 'step3', 'union', 'host_gain125', 'method2_union_2witness')
BLOCKS = [('gain 0', os.path.join(HUS, 'step3', 'method2_gain0_witness'), 'tab:blue'),
          ('gain 125', os.path.join(HUS, 'step3', 'method2_gain125_witness'), 'tab:red')]

L_REC, SE_REC = 2.129, 0.430          # the union's Method 2 value


def arrays(d):
    """(radius in R_sun, radial deflection in arcsec) from stage 3's own report."""
    f = glob.glob(os.path.join(d, 'ECLIPSE_OUTPUT*.txt'))
    if not f:
        raise SystemExit('no stage-3 report in ' + d)
    s = io.open(f[0], encoding='utf-8', errors='replace').read()

    def arr(key):
        m = re.search(re.escape(key) + r':\s*\[(.*?)\]', s, re.S)
        return np.array([float(x) for x in m.group(1).split()])
    return arr('radial distances'), arr('deflection (arcsec)')


def chart(groups, fname, title, note, legend_title):
    fig, ax = plt.subplots(figsize=(10, 6.8))
    ax.axhline(0, color='black', lw=1)
    dots, allr, alld = [], [], []
    for lab, R, D, colr in groups:
        dots.append(ax.scatter(R, D, s=26, alpha=0.85, color=colr, zorder=4,
                               label='%s (%d stars)' % (lab, len(R))))
        allr.append(R)
        alld.append(D)
    R = np.concatenate(allr)
    D = np.concatenate(alld)
    xx = np.linspace(1.85, R.max() + 0.4, 300)
    band, ln1, ln2, ln3 = reference_curves(
        ax, xx, L_REC, SE_REC, 'Method 2 union:  L = %.3f"' % L_REC,
        'statistical $\\pm$%.3f" ONLY \u2014 cell 4 has no atmospheric term yet' % SE_REC)
    ax.set_xlabel('radial position (solar radii)', fontsize=13)
    ax.set_ylabel('radial deflection (arcsec, outward positive)', fontsize=13)
    ax.set_title(title, fontsize=12)
    lo, hi = np.percentile(D, [0.5, 99.5])
    ax.set_ylim(min(lo, -0.35) - 0.15, max(hi, L_REC / 1.9) + 0.35)
    first = ax.legend(handles=dots, fontsize=8.5, loc='lower left',
                      title=legend_title, title_fontsize=8.5)
    ax.add_artist(first)
    ax.legend(handles=[ln1, ln2, ln3, band], fontsize=9, loc='upper right')
    fig.text(0.06, 0.012, note, fontsize=8.5)
    # two-line captions need the room; a one-line rect clipped the first version
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, fname), dpi=130)
    plt.close(fig)
    resid = D - L_REC / R
    print('%-32s N=%3d  rms about the fit %.3f "' % (fname, len(R), float(np.sqrt(np.mean(resid ** 2)))))
    return fname


def main():
    names = []
    R, D = arrays(UNION)
    names.append(chart(
        [('two-gain union, two-witness', R, D, 'tab:blue')],
        'record_deflection.png',
        'Husillos 2026 \u2014 radial deflection, two-gain union, 63 two-witness stars',
        'Method 2, fitted straight against the zenith with a free scale; no radial crop.\n'
        'Band is STATISTICAL ONLY: cell 4 has no atmospheric term (\u00a73i), where every other '
        'cell carries \u00b10.11\u20130.33".',
        'reduction of record'))

    groups = []
    for lab, d, colr in BLOCKS:
        r, dd = arrays(d)
        groups.append((lab, r, dd, colr))
        print('   %-9s N=%d  mean deflection %+.3f "  at mean radius %.2f R_sun'
              % (lab, len(r), dd.mean(), r.mean()))
    names.append(chart(
        groups, 'record_deflection_by_block.png',
        'Husillos 2026 \u2014 the same 64 stars, measured by each gain block separately',
        'The two blocks are the same field 39 s apart. On identical stars they fit '
        'L = 1.596 \u00b1 0.507" (gain 0) and 2.782 \u00b1 0.540" (gain 125); they correlate at only '
        'r = 0.484,\nand gain 125\u2019s excess is vertically polarised (V/H 1.89 against 1.43). '
        'The curve and band are the UNION\u2019s, drawn for reference.',
        'measured by'))
    publish(names, OUT)


if __name__ == '__main__':
    main()
