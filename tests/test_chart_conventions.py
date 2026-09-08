"""A chart axis must follow the set it sits beside, not a remembered convention.

On 2026-09-08 two new RA/DEC charts were drawn with RA descending to the right -- the sky
convention, applied from habit -- while every record chart in the project draws it ascending
(`s1_charts_record.py` line 272, `b17_charts_record.py` line 264). The reversed line even carried
a comment saying "as cell 2 draws it", which is what cell 2 does not do. A citation written from
memory reads exactly like a checked one, so this is a test.

The rule enforced is narrow and mechanical: in a tool that labels an axis "RA", the RA limits go
low-to-high. If a chart ever genuinely needs the reversed axis, the reversal gets a same-line
comment saying so and naming what it follows.
"""
import ast
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, 'tools')
# a deliberate reversal must say so on the line, e.g. "# reversed: follows <x>"
DELIBERATE = re.compile(r'#.*reversed', re.I)


def _ra_chart_files():
    for root, _dirs, files in os.walk(TOOLS):
        if '__pycache__' in root:
            continue
        for fn in files:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(root, fn)
            src = open(path, encoding='utf-8').read()
            if "'RA (degrees)'" in src or '"RA (degrees)"' in src:
                yield os.path.relpath(path, TOOLS).replace('\\', '/'), src


def test_ra_axis_is_not_reversed():
    """set_xlim on an RA axis runs low to high, as every record chart in the project does."""
    bad = []
    for rel, src in _ra_chart_files():
        for line in src.splitlines():
            if 'set_xlim(' not in line or DELIBERATE.search(line):
                continue
            m = re.search(r'set_xlim\(\s*([A-Za-z_][A-Za-z_0-9]*)\s*,\s*([A-Za-z_][A-Za-z_0-9]*)',
                          line)
            if not m:
                continue
            first, second = m.group(1), m.group(2)
            if first.startswith(('hi', 'max')) or second.startswith(('lo', 'min')):
                bad.append('%s: %s' % (rel, line.strip()))
    assert not bad, (
        'RA axis reversed. Every record chart set in this project draws RA ascending to the '
        'right (s1_charts_record.py line 272, b17_charts_record.py line 264); a chart that sits '
        'beside them must match, whatever the sky convention says. If the reversal is genuinely '
        'wanted, say "reversed" in a comment on the line and name what it follows:\n  '
        + '\n  '.join(bad))


def test_the_reference_charts_still_define_the_convention():
    """If cells 1 and 2 ever change direction, this test is what tells the next reader."""
    for rel in ('matrix_station1/s1_charts_record.py', 'matrix_bruns/b17_charts_record.py'):
        src = open(os.path.join(TOOLS, rel), encoding='utf-8').read()
        assert 'ax.set_xlim(lo_ra, hi_ra)' in src, (
            '%s no longer draws RA ascending -- the convention this test enforces came from it, '
            'so decide deliberately and update both together' % rel)
