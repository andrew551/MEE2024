"""A Gaia source id must survive being read, or the self-match guard silently fails.

Every double-star check in this project asks the same question of a star: is the nearest catalogue
entry a real companion, or the star's own entry? Once proper motion is applied the star's own
entry sits 0.6-1.0 " from its stored position -- squarely inside any plausible cut -- so getting
the id comparison wrong does not produce an error, it produces a plausible false double.

On 2026-09-09 that guard was broken three different ways in one afternoon, each giving a
confident wrong answer:

  * `int(float(id))` -- 19 digits need 62 bits and a double carries 53, so 73 % of cell 2's ids
    changed outright (2577061092921353984 -> ...354240) and 145 of its 192 stars were reported as
    their own companion;
  * `np.uint64(...)` compared against the int64 `source_id.npy` -- Leon came out 13 of 42 where a
    plain Python int gives 0;
  * `df.iterrows()` on an all-numeric frame, which upcasts the row to float64 and delivers
    618969525396598656 as 6.189695253965987e+17 -- again Leon 13 of 42.

The first two are covered by the exactness test below. The third is covered by the dtype test:
a numeric id column must be converted to text before it is read row by row.

These are pure functions with no catalogue behind them, so the tests run anywhere.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, 'tools', 'matrix_station1', 's1_blend_rule.py')

# the tool loads data at module level, so take only its function definitions
_src = open(TOOL, encoding='utf-8').read()
_head = _src.split('# ---------------------------------------------------------------- the zenith sample')[0]
_ns = {'__name__': 'blend_rule_probe', '__file__': TOOL}
exec(compile(_head, TOOL, 'exec'), _ns)
own_id, blend_shift, flag = _ns['own_id'], _ns['blend_shift'], _ns['flag']

REAL_IDS = ['2577061092921353984', '618450384109429376', '618969525396598656',
            '2577673177300873216', '630926851787181056']


@pytest.mark.parametrize('digits', REAL_IDS)
def test_own_id_is_exact_for_both_spellings(digits):
    """'gaia:<id>' and a bare id must both come back as the same exact integer."""
    want = int(digits)
    assert own_id('gaia:' + digits) == want
    assert own_id(digits) == want
    assert own_id(np.int64(want)) == want


@pytest.mark.parametrize('digits', REAL_IDS)
def test_a_float_shaped_id_is_refused_not_rounded(digits):
    """The float form has already lost the id. Refusing it is right; rounding it is not."""
    lossy = repr(float(int(digits)))                 # e.g. '6.189695253965987e+17'
    assert own_id(lossy) is None, 'a float-shaped id must not be silently parsed'


def test_the_float_round_trip_really_does_corrupt_ids():
    """The premise of the test above, so it cannot quietly stop being true."""
    changed = [d for d in REAL_IDS if int(d) != int(float(d))]
    assert changed, 'if a double now holds 19 digits, these tests can be relaxed'


def test_flag_reads_ids_exactly_from_an_all_numeric_frame(monkeypatch):
    """iterrows() upcasts an all-numeric row to float64; flag() must defeat that."""
    seen = []

    def fake_neighbour(ra0, dec0, oid):
        seen.append(oid)
        return None                                   # no companion: we only check the id

    monkeypatch.setitem(_ns, 'neighbour', fake_neighbour)
    df = pd.DataFrame({'gaia_id': np.array([int(d) for d in REAL_IDS], dtype='int64'),
                       'ra_cat': np.linspace(10.0, 10.4, len(REAL_IDS)),
                       'dec_cat': np.linspace(5.0, 5.4, len(REAL_IDS)),
                       'mag': np.linspace(8.0, 12.0, len(REAL_IDS))})
    assert df.dtypes.map(lambda t: np.issubdtype(t, np.number)).all(), 'frame must be all-numeric'
    flag(df, 'gaia_id', 'mag', 'ra_cat', 'dec_cat')
    assert seen == [int(d) for d in REAL_IDS], (
        'flag() lost or rounded a source id: %r' % (seen,))


def test_blend_shift_is_the_flux_weighted_centroid():
    """b = sep x f2/(f1+f2): the measured slope against this was +1.00 inside 4 arcsec."""
    assert blend_shift(4.0, 10.0, 10.0) == pytest.approx(2.0)          # equal pair: halfway
    assert blend_shift(4.0, 10.0, 15.0) == pytest.approx(4.0 * 0.01 / 1.01, rel=1e-6)
    assert blend_shift(10.0, 10.0, 99.0) == pytest.approx(0.0, abs=1e-9)
    # monotonic in separation, and falling as the companion fades
    assert blend_shift(2.0, 10.0, 12.0) < blend_shift(4.0, 10.0, 12.0)
    assert blend_shift(4.0, 10.0, 14.0) < blend_shift(4.0, 10.0, 12.0)
