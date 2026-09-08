"""An analysis window must be cited, not chosen at the keyboard.

On 2026-09-08 Station 2's eclipse field was fitted at 2-5.5 R_sun. No document contains that
bound; it dropped three of seventeen stars per tier and moved the answer. Prose in CLAUDE.md had
already failed to prevent the same class of error once that week (the daytime calibration was
given the eclipse field's `distortion_fixed_coefficients`), so this is a test rather than a rule.

Two things are enforced:

  1. `tools/analysis_window.py` holds each cell's window with its source, and the values are
     pinned here the way the stage-2 regression baselines are. Moving one is a deliberate act
     that edits this file and names what changed.
  2. No tool may introduce a NEW window literal without a same-line comment naming where the
     number comes from. Sites that predate the registry are listed in PRE_REGISTRY below,
     unaudited and quarantined: the list may shrink, never grow.
"""
import ast
import os
import re

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, 'tools')
WINDOW_NAMES = {'MAGCUT', 'MAG', 'MAGLIM', 'MAG_ECL', 'LIMIT_MAG', 'RCUT', 'RMIN', 'RMAX',
                'RMIN_RSUN', 'RMAX_RSUN'}

# Window literals written before this test existed. Not audited -- each is a claim about a
# convention that nobody has checked against a document. Shrink this list by citing them; never
# add to it. Path is relative to tools/, with the constant names on that line.
PRE_REGISTRY = {
    ('matrix_bruns/b17_atmosphere.py', 'RCUT,MAGCUT'),
    ('matrix_bruns/b17_atmosphere2.py', 'RCUT,MAGCUT'),
    ('matrix_bruns/b17_bracket_null.py', 'RCUT,MAGCUT'),
    ('matrix_bruns/b17_bruns_method.py', 'LIMIT_MAG'),
    ('matrix_bruns/b17_lr_bracket_null.py', 'RCUT,MAGCUT'),
    ('matrix_bruns/b17_night_estimator.py', 'RMAX'),
    ('matrix_bruns/b17_union.py', 'LIMIT_MAG'),
    ('matrix_station1/s1_atmosphere_maps.py', 'MAGCUT'),
    ('matrix_station1/s1_bias_leverage.py', 'MAGCUT,RCUT,GREF'),
    ('matrix_station1/s1_blocks_alone.py', 'RCUT,RMAX,MAG'),
    ('matrix_station1/s1_caldecomp_fit.py', 'MAGCUT,RCUT,RMAX'),
    ('matrix_station1/s1_eclipse_calibrated.py', 'MAGCUT,RCUT,RMAX'),
    ('matrix_station1/s1_eclipse_convention.py', 'MAGCUT,RCUT,RMAX'),
    ('matrix_station1/s1_eclipse_corona.py', 'MAGCUT,RCUT,RMAX'),
    ('matrix_station1/s1_eclipse_tiers.py', 'MAGCUT,RCUT,RMAX'),
    ('matrix_station1/s1_estimator_arbiter.py', 'MAGCUT,RCUT,RMAX,GREF'),
    ('matrix_station1/s1_moments_on_corona.py', 'RCUT,RMAX,MAG'),
    ('matrix_station1/s1_pooled_fit.py', 'MAGCUT,RCUT,RMAX'),
    ('matrix_station1/s1_reference_convention_test.py', 'MAGCUT,RCUT,RMAX'),
    ('matrix_station1/s1_septic_test.py', 'MAGCUT,RCUT,RMAX'),
    ('matrix_station1/s1_zenith_floor.py', 'MAGCUT,RCUT'),
    ('matrix_station2/s2_zenith_null.py', 'MAGCUT,RCUT'),
    ('leakey_zenith_floor.py', 'R_SUN_AS,MAGCUT,RCUT'),
    ('refraction/m5_projection.py', 'MAG_ECL'),
    ('refraction/m5_projection.py', 'RMIN_RSUN'),
    ('step3_atmosphere.py', 'RCUT,MAGCUT'),
    ('step3_atmosphere_maps.py', 'MAGCUT'),
    ('step3_null_gap.py', 'RCUT,MAGCUT'),
    ('step3_s2_union.py', 'LIMIT_MAG'),
    ('step3_zenith_floor.py', 'RCUT,MAGCUT'),
}

# a comment that names a source: a docs path, a .md file, a cell of the matrix, or the registry
# cited = a comment naming a source, OR a value taken straight from the registry
CITED = re.compile(r'(#.*(docs/|\.md|MATRIX_2026|STEP3_2026|V1_4_0_TESTING|LEON_2026|'
                   r'analysis_window|WINDOW:|Bruns 2018|the record))|(_W\.|WINDOWS\[)', re.I)


def _window_sites():
    """Every module-level assignment to a window constant in tools/, with its source line."""
    out = []
    for root, _dirs, files in os.walk(TOOLS):
        if '__pycache__' in root:
            continue
        for fn in files:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, TOOLS).replace('\\', '/')
            src = open(path, encoding='utf-8').read()
            try:
                tree = ast.parse(src)
            except SyntaxError:                       # not ours to police
                continue
            lines = src.splitlines()
            for node in tree.body:                    # module level only
                if not isinstance(node, ast.Assign):
                    continue
                names = []
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
                    elif isinstance(t, ast.Tuple):
                        names += [e.id for e in t.elts if isinstance(e, ast.Name)]
                hit = [n for n in names if n in WINDOW_NAMES]
                if hit:
                    out.append((rel, ','.join(names), lines[node.lineno - 1]))
    return out


def test_documented_windows_are_pinned():
    """The registry's numbers are the record's numbers. Changing one is deliberate."""
    from tools.analysis_window import WINDOWS
    w = WINDOWS['mexico2024_station1']
    assert (w.mag, w.rmin, w.rmax) == (13.0, 2.0, 10.0), (
        'cell 2 is quoted at G <= 13, 2-10 R_sun in docs/MATRIX_2026.md; the registry disagrees')
    assert WINDOWS['mexico2024_station2'][:3] == w[:3], (
        "Station 2 is reduced by the Station 1 technique verbatim; its window follows cell 2's")
    for name, win in WINDOWS.items():
        assert win.source.strip(), '%s has no citation' % name
        assert win.rmin < win.rmax and win.mag > 0


def test_no_uncited_window_literal():
    """A new window constant must say where its number comes from."""
    bad = []
    for rel, names, line in _window_sites():
        if (rel, names) in PRE_REGISTRY:
            continue
        if not CITED.search(line):
            bad.append('%s: %s' % (rel, line.strip()))
    assert not bad, (
        'these window constants name no source. Put the value in tools/analysis_window.py with '
        'its citation and import it, or add a same-line comment naming the document it comes '
        'from. An analysis cut that no document contains is an invented cut:\n  '
        + '\n  '.join(bad))


def test_quarantine_list_only_shrinks():
    """Every PRE_REGISTRY entry must still exist; a stale one means the list was not maintained."""
    live = {(rel, names) for rel, names, _ in _window_sites()}
    gone = sorted(PRE_REGISTRY - live)
    assert not gone, ('PRE_REGISTRY names sites that no longer exist -- delete them from the list '
                      'rather than leaving it stale: %s' % (gone,))


@pytest.mark.parametrize('tool', ['s2_charts_record.py', 's2_eclipse_fit.py',
                                  's2_method1_per_tier.py', 's2_bracket_lr_split.py'])
def test_station2_tools_use_the_registry(tool):
    """The tools that produced the 2-5.5 R_sun mistake now take their window from the registry."""
    src = open(os.path.join(TOOLS, 'matrix_station2', tool), encoding='utf-8').read()
    assert 'analysis_window' in src, (
        '%s must import its window from tools/analysis_window.py, not declare one' % tool)
