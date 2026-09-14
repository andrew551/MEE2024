"""The chart constructions every record set shares, in one place.

Until 2026-09-09 each cell's chart tool -- `matrix_bruns/b17_charts_record.py`,
`step3_charts_record.py`, `matrix_station1/s1_charts_record.py`,
`matrix_station2/s2_charts_record.py` -- carried its own copy of the sky frame, the field chart,
the covariance chart and the joint-scale arithmetic. Four copies diverged in four ways during
one week: an RA axis drawn reversed, a field chart in pixels beside three in RA/DEC, a
joint scale computed on the wrong base with the wrong sign, and a Method 1 drawn as a point
beside three drawn as ellipses. Each was a re-implementation of something a neighbouring tool
already did correctly.

So the constructions live here and the tools import them. What is shared is the GEOMETRY and the
ARITHMETIC -- the affine between sky and sensor, the arrow and ellipse drawing, the axis
conventions, the joint-scale formula -- with every cosmetic knob (arrow length, paddings, bar
positions, marker sizes) passed in, so that each tool reproduces its existing output exactly.
`tests/test_record_charts.py` pins the arithmetic and the conventions.

Conventions fixed here, and where they come from:

  * RA ascends to the RIGHT on every field chart. It is not the sky convention; it is what cells
    1 and 2 drew and what the set keeps (`tests/test_chart_conventions.py`).
  * A unit sensor displacement round-trips to one arcsec of sky, asserted at construction
    (the check the Bruns chart lacked until its revision 10, when arrows were 2.09x too long).
  * Every arrow end is asserted inside the axes.
  * The joint plate scale is `stage-2 scale - S * PS`: S is the residual-convention scale term
    the stage-3 fit finds, and a NEGATIVE S means a LARGER scale in arcsec per pixel. Verified
    against cell 2's four recorded joint scales to seven digits (2026-09-09), after Station 2's
    chart had used PS * (1 + S) with PS the zenith reference and sat 840 ppm from the truth.
  * The covariance chart's vertical axis, on a cell that has a Method 1, is S in ppm relative
    to the imported scale -- residual convention, so "+ppm" is a physically smaller scale. That
    is what cells 1 and 3 drew first; the box text should state both readings.
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, VPacker
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon

DPI = 140
GR, NEWTON = 1.7512, 0.8756


# ---------------------------------------------------------------- writing
class ChartWriter:
    """Every chart goes to `out/<name>` and to `out/chart_versions/<rev>_<name>`; superseded
    revisions are never deleted (the Bruns rule, seventh revision)."""

    def __init__(self, out, rev, ver=None, dpi=DPI):
        self.out, self.rev, self.dpi = out, rev, dpi
        self.ver = ver or os.path.join(out, 'chart_versions')
        os.makedirs(self.out, exist_ok=True)
        os.makedirs(self.ver, exist_ok=True)

    def save(self, fig, name):
        fig.savefig(os.path.join(self.out, name), dpi=self.dpi)
        fig.savefig(os.path.join(self.ver, self.rev + '_' + name), dpi=self.dpi)
        plt.close(fig)


# ---------------------------------------------------------------- the sky frame
class SkyFrame:
    """The affine between sky (RA, Dec in degrees) and sensor pixels, and the two maps the field
    charts need: a pixel position to RA/Dec, and a sensor-axis displacement in arcsec to a sky
    displacement in arcsec of RA*cos(dec) and Dec.

    Fit it on the matched stars of the reduction (`from_stars`), or rebuild it from stored
    coefficients (`from_affine`, Leon's `leon_union_meta.json`). Either way a unit sensor
    displacement is asserted to come back as one arcsec of sky.
    """

    def __init__(self, ax_, ay_, ra0, de0, ps):
        self.ax_, self.ay_ = np.asarray(ax_, float), np.asarray(ay_, float)
        self.ra0, self.de0, self.ps = float(ra0), float(de0), float(ps)
        self.cos0 = float(np.cos(np.radians(self.de0)))
        self.Minv = np.linalg.inv(np.array([[self.ax_[0], self.ax_[1]], [self.ay_[0], self.ay_[1]]]))
        rt = float(np.hypot(*self.sensor_vec_to_sky(np.array([1.0]), np.array([0.0])))[0])
        assert abs(rt - 1.0) < 0.01, 'a 1 arcsec sensor displacement maps to %.3f arcsec' % rt

    @classmethod
    def from_stars(cls, ra, dec, px, py, ps):
        ra, dec = np.asarray(ra, float), np.asarray(dec, float)
        ra0, de0 = ra.mean(), dec.mean()
        Xa = (ra - ra0) * np.cos(np.radians(de0))
        Ya = dec - de0
        Aa = np.c_[Xa, Ya, np.ones_like(Xa)]
        ax_, *_ = np.linalg.lstsq(Aa, np.asarray(px, float), rcond=None)
        ay_, *_ = np.linalg.lstsq(Aa, np.asarray(py, float), rcond=None)
        return cls(ax_, ay_, ra0, de0, ps)

    @classmethod
    def from_affine(cls, ax_, ay_, ra0, de0, ps):
        return cls(ax_, ay_, ra0, de0, ps)

    def px_to_sky(self, px, py):
        v = self.Minv @ np.vstack([np.asarray(px, float) - self.ax_[2],
                                   np.asarray(py, float) - self.ay_[2]])
        return self.ra0 + v[0] / self.cos0, self.de0 + v[1]

    def sensor_vec_to_sky(self, dx_as, dy_as):
        """Sensor-axis displacement (arcsec) -> (arcsec of RA*cos(dec), arcsec of Dec).

        Divides by the plate scale to get pixels, maps through the inverse affine to degrees,
        and returns arcsec. The Bruns chart multiplied by the scale instead through its ninth
        revision and drew every arrow 2.087x too long against its own bar.
        """
        v = self.Minv @ np.vstack([np.asarray(dx_as, float) / self.ps,
                                   np.asarray(dy_as, float) / self.ps])
        return v[0] * 3600, v[1] * 3600

    def sky_to_px_dir(self, dra_deg, ddec_deg):
        """A sky offset in degrees (RA*cos(dec), Dec) -> the same direction in sensor pixels."""
        v = np.array([dra_deg, ddec_deg], float)
        return np.array([v @ self.ax_[:2], v @ self.ay_[:2]])

    def corners(self, nx, ny):
        return self.px_to_sky(np.array([0, nx, nx, 0], float), np.array([0, 0, ny, ny], float))


# ---------------------------------------------------------------- the arithmetic
def joint_plate_scale(stage2_scale, S, ps):
    """The plate scale the stage-3 fit implies: stage 2's fitted scale corrected by the residual
    scale term S found alongside L.

    `joint = stage2 - S * ps`. S is in the residual convention of the design matrix every cell
    uses (dx = measured - model against a column (px - NX/2) * PS): a POSITIVE S means the
    model's scale was too large, so the physical scale is SMALLER. Reproduces cell 2's four
    recorded joint scales to seven digits (`tests/test_record_charts.py`).

    Station 2's chart computed `PS * (1 + S)` with PS the zenith reference instead -- wrong base
    (the residuals are against the stage-2 model, not the zenith), wrong sign -- and placed
    Method 2's scale 840 ppm from where it belongs (2026-09-09).
    """
    return float(stage2_scale) - float(S) * float(ps)


def ppm_from(value, reference):
    return 1e6 * (float(value) - float(reference)) / float(reference)


# ---------------------------------------------------------------- drawing pieces
def draw_ellipse(ax, cov, mu, color, name, lw=1.6, marker_size=110):
    """A 1-sigma ellipse of a 2x2 covariance about `mu`, with a '+' at the centre."""
    vals, vecs = np.linalg.eigh(np.asarray(cov, float))
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    ax.add_patch(Ellipse(mu, 2 * np.sqrt(vals[1]), 2 * np.sqrt(vals[0]), angle=ang,
                         fill=False, color=color, lw=lw, label='1$\\sigma$ — %s' % name))
    ax.scatter(*mu, marker='+', s=marker_size, color=color, zorder=5)


def packed_box(ax, lines, loc='lower left', anchor=(0.0, 0.0), size=9.5, alpha=None):
    """A text box packed around its own lines (VPacker), pinned in axes coordinates, so it can
    neither leave the axes nor be hand-sized too small for its contents (Leon, revision 2)."""
    pack = VPacker(children=[TextArea(t, textprops=dict(color=c, size=size)) for t, c in lines],
                   pad=0, sep=3, align='left')
    box = AnchoredOffsetbox(loc=loc, child=pack, pad=0.45, borderpad=0.6, frameon=True,
                            bbox_to_anchor=anchor, bbox_transform=ax.transAxes)
    kw = dict(facecolor='white', edgecolor='gray', linewidth=0.9)
    if alpha is not None:
        kw['alpha'] = alpha
    box.patch.set(**kw)
    box.set_zorder(6)
    ax.add_artist(box)
    return box


def reference_curves(ax, xx, L, tot, fit_label, band_label, fit_lw=2.2, band_color='black'):
    """The fitted 1/R curve with its total band, and Einstein and Newton beside it."""
    band = ax.fill_between(xx, (L - tot) / xx, (L + tot) / xx, color=band_color, alpha=0.10,
                           label=band_label)
    ln1, = ax.plot(xx, L / xx, color='black', lw=fit_lw, label=fit_label)
    ln2, = ax.plot(xx, GR / xx, color='green', lw=1.5, label='Einstein  1.751"')
    ln3, = ax.plot(xx, NEWTON / xx, color='orange', lw=1.5, ls='--', label='Newton  0.876"')
    return band, ln1, ln2, ln3


def arcsinh_stretch(img, pct=(5, 99.5), gain=30.0):
    """The display stretch every annotated master uses."""
    lo, hi = np.percentile(img, list(pct))
    return np.arcsinh((np.clip(img, lo, hi) - lo) / max(hi - lo, 1) * gain)


# ---------------------------------------------------------------- the field chart
def field_chart(ax, x, y, vx, vy, corners, sun, arrow_deg, cosf, *,
                groups, arrow_color, arrow_lw=1.5, arrow_style='-|>,head_width=0.22,head_length=0.45',
                point_labels=None, sun_ring2=False, sun_ring2_style=None,
                include_sun_in_limits=False, pad=(0.06, 0.05), footprint_label='sensor footprint',
                xlabel='RA (degrees)', ylabel='DEC (degrees)', title=None, title_size=12,
                sun_ellipse_ratio=None):
    """Displacement vectors on a chart whose axes are angles in degrees.

    Frame-agnostic: `x`, `y` are the stars' chart positions in degrees, `vx`, `vy` their
    displacements in arcsec ALONG THE CHART AXES (RA*cos(dec) and Dec for an RA/DEC chart, az*cos
    and alt for Leon's alt/az one), `corners` the sensor footprint as (xs, ys), `sun` as
    (x, y, radius_deg) or None. `cosf` stretches the horizontal axis: an arrow of a arcsec is
    drawn `a * arrow_deg / cosf` wide and `a * arrow_deg` tall, and the aspect is 1/cosf.

    `groups` is a list of (mask, dict(scatter kwargs incl. label)) drawn in order; `arrow_color`
    is one colour or one per point; `point_labels`, if given, is (texts, dx, colors, fontsize)
    drawn beside each point inside the arrow loop, as the Bruns and Leon charts do.

    The horizontal axis ASCENDS to the right, whatever the frame -- the set's convention
    (`tests/test_chart_conventions.py`). Every arrow end is asserted inside the axes. Returns the
    limits so the caller can place its scale bars.
    """
    x, y, vx, vy = (np.asarray(a, float) for a in (x, y, vx, vy))
    cx, cy = np.asarray(corners[0], float), np.asarray(corners[1], float)
    ax.add_patch(Polygon(np.c_[cx, cy], fill=False, color='gray', lw=1.2, label=footprint_label))
    colors = [arrow_color] * len(x) if isinstance(arrow_color, str) else list(arrow_color)
    ends_x, ends_y = [], []
    for k in range(len(x)):
        x1 = x[k] + vx[k] * arrow_deg / cosf
        y1 = y[k] + vy[k] * arrow_deg
        ends_x.append(x1); ends_y.append(y1)
        ax.annotate('', xy=(x1, y1), xytext=(x[k], y[k]),
                    arrowprops=dict(arrowstyle=arrow_style, color=colors[k], lw=arrow_lw,
                                    shrinkA=0, shrinkB=0))
        if point_labels is not None:
            texts, dxs, lcols, fs = point_labels
            dxk = dxs[k] if np.ndim(dxs) else dxs
            lck = lcols[k] if not isinstance(lcols, str) else lcols
            ax.annotate(texts[k], (x[k] + dxk, y[k]), fontsize=fs, color=lck)
    for mask, kw in groups:
        mask = np.asarray(mask, bool)
        if mask.any():
            ax.scatter(x[mask], y[mask], zorder=5, **kw)
    if sun is not None:
        sx, sy, sr = sun
        if sun_ellipse_ratio:
            ax.add_patch(Ellipse((sx, sy), 2 * sr / sun_ellipse_ratio, 2 * sr, color='black',
                                 zorder=3, label='the Sun, 1 R$_\\odot$ to scale'))
        else:
            ax.add_patch(Circle((sx, sy), sr, color='black', zorder=3,
                                label='the Sun, 1 R$_\\odot$ to scale'))
        if sun_ring2:
            st = dict(fill=False, color='gray', ls='--', lw=1.0, zorder=3, label='2 R$_\\odot$')
            st.update(sun_ring2_style or {})
            ax.add_patch(Circle((sx, sy), 2 * sr, **st))
    lo_x, hi_x = min(cx.min(), min(ends_x)), max(cx.max(), max(ends_x))
    lo_y, hi_y = min(cy.min(), min(ends_y)), max(cy.max(), max(ends_y))
    if include_sun_in_limits and sun is not None:
        sx, sy, sr = sun
        lo_x, hi_x = min(lo_x, sx - 2 * sr), max(hi_x, sx + 2 * sr)
        lo_y, hi_y = min(lo_y, sy - 2 * sr), max(hi_y, sy + 2 * sr)
    lo_x -= pad[0]; hi_x += pad[0]; lo_y -= pad[1]; hi_y += pad[1]
    for x1, y1 in zip(ends_x, ends_y):
        assert lo_x < x1 < hi_x and lo_y < y1 < hi_y, 'an arrow leaves the axes'
    ax.set_xlim(lo_x, hi_x)          # ascending: the set's convention, not the sky's
    ax.set_ylim(lo_y, hi_y)
    ax.set_aspect(1 / cosf)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    return lo_x, hi_x, lo_y, hi_y


def scale_bars(ax, bars, arrow_deg, cosf, x_span, x_frac=1.04, text_dy=0.028, fontsize=8):
    """The '1 arcsec of displacement' bar and its companion, outside the axes to the right.
    `bars` is a list of (y_axes_fraction, length_arcsec, text)."""
    bar_deg = arrow_deg / cosf
    for y_fr, ln, txt in bars:
        ax.annotate('', xy=(x_frac + ln * bar_deg / x_span, y_fr), xytext=(x_frac, y_fr),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='-', color='black', lw=3))
        ax.annotate(txt, (x_frac, y_fr + text_dy), xycoords='axes fraction', fontsize=fontsize)


def bar_frame(ax, rect=(1.02, 0.29, 0.30, 0.22)):
    """The rounded frame the Bruns and Leon charts draw around their scale bars."""
    x0, y0, w, h = rect
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.012',
                                transform=ax.transAxes, fill=False, color='gray', lw=0.9,
                                clip_on=False))


# ---------------------------------------------------------------- the covariance chart
def covariance_chart(C1, mu1, C2, mu2, lines, title, *, newton=True, newton_lw=1.2,
                     name1='Method 1 (scale imported)', name2='Method 2 (scale free)',
                     C3=None, mu3=None, name3='Method 3 (quadratic imported, scale free)',
                     colour3='tab:purple',
                     ylabel='Plate scale (ppm difference from imported value)',
                     figsize=(9.5, 7), margins=0.15, box_alpha=None):
    """L against the plate scale: two or three 1-sigma ellipses, Einstein (and Newton), a box.

    The cells-1-and-3 construction. `mu1 = (L1, 0)` with C1 carrying the statistical AND the
    imported-scale terms; `mu2 = (L2, S in ppm)` with C2 the free-scale fit's covariance in
    (arcsec, ppm). The vertical axis is S in the residual convention -- see the module docstring.

    `C3`/`mu3` add an optional THIRD ellipse for Method 3 -- the quadratic and above imported
    from a calibration field, the constant and linear refitted on the eclipse field, the scale
    fitted (2026-09-14, Douglas). It is plotted on the same axis as Method 2 and for the same
    reason: once the linear terms are free the plate scale is fitted rather than imported
    (`distortion_polynomial` replaces the scale only at order_free == 0), so Method 3 has a
    fitted scale to place, not a pinned one. Omit it and the chart is exactly as it was, which
    is what the other three cells still draw.

    Returns (fig, ax); the caller saves it.
    """
    fig, ax = plt.subplots(figsize=figsize)
    draw_ellipse(ax, C1, np.asarray(mu1, float), 'darkred', name1)
    draw_ellipse(ax, C2, np.asarray(mu2, float), 'tab:blue', name2)
    if C3 is not None:
        draw_ellipse(ax, C3, np.asarray(mu3, float), colour3, name3)
    ax.axvline(GR, color='green', lw=1.5, label='Einstein 1.751"')
    if newton:
        ax.axvline(NEWTON, color='orange', lw=newton_lw, ls='--', label='Newton 0.876"')
    packed_box(ax, lines, alpha=box_alpha)
    ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.autoscale_view()
    ax.margins(margins)
    return fig, ax


def covariance_chart_single(C, mu, lines, title, *, name='Method 2 (scale fitted with L)',
                            ylabel='fitted plate scale (arcsec per pixel)', figsize=(9.5, 7),
                            margins=0.25, lw=1.8, marker_size=140):
    """Cell 2's variant: one ellipse on an absolute plate-scale axis, because there is no
    imported scale to measure from. Returns (fig, ax)."""
    fig, ax = plt.subplots(figsize=figsize)
    draw_ellipse(ax, C, np.asarray(mu, float), 'tab:blue', name, lw=lw, marker_size=marker_size)
    ax.axvline(GR, color='green', lw=1.5, label='Einstein 1.751"')
    packed_box(ax, lines)
    ax.set_xlabel('L (arcsec at the solar limb)', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.ticklabel_format(axis='y', useOffset=False, style='plain')
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.autoscale_view()
    ax.margins(margins)
    return fig, ax
