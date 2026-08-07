"""The choices the app window starts with.

A default is the setting almost every run will actually use, so each of these is a
decision about what the pipeline does by default -- not cosmetics. They live in the
HTML as attributes, where nothing else would notice them changing, hence this file.
"""

import re
from pathlib import Path

import pytest

from mee2024.config import DEFAULT_OPTIONS

FRONTEND = Path(__file__).parent.parent / 'mee2024' / 'ui' / 'frontend.html'


@pytest.fixture(scope='module')
def html():
    return FRONTEND.read_text(encoding='utf-8')


def _select_options(html, select_id):
    """The <option> tags of one <select>, in order."""
    block = re.search(rf'<select id="{select_id}".*?</select>', html, re.S)
    assert block, f'no <select id="{select_id}"> in the page'
    return re.findall(r'<option value="([^"]+)"([^>]*)>', block.group(0))


def _checkbox(html, box_id):
    match = re.search(rf'<input type="checkbox" id="{box_id}"([^>]*)>', html)
    assert match, f'no checkbox id="{box_id}" in the page'
    return match.group(1)


def test_the_observation_date_is_read_from_the_fits_header_by_default(html):
    """The header is the measurement; recovering the date from proper motions is the
    fallback for frames that carry no date, and is honest to two to four weeks."""
    options = _select_options(html, 'datemode')
    selected = [value for value, attrs in options if 'selected' in attrs]
    assert selected == ['header'], f'expected header to be the default, got {selected}'
    assert options[0][0] == 'header', 'the default should also be listed first'


def test_double_stars_are_dropped_by_default(html):
    """A blended pair gives a systematically wrong position, not a noisy one, so it is
    worth less than the star count it costs."""
    assert 'checked' in _checkbox(html, 'rm-double')
    assert DEFAULT_OPTIONS['remove_double_tab2'] is True, \
        'the CLI and the window must not disagree about what a fit of the same frames is'


def test_stars_without_proper_motion_are_kept_by_default(html):
    """Deliberately off: the motion is borrowed from Hipparcos where Gaia lacks it, and
    the stars this drops are disproportionately the bright ones worth keeping."""
    assert 'checked' not in _checkbox(html, 'rm-nopm')
    assert DEFAULT_OPTIONS['remove_missing_pm'] is False


def test_the_sensitive_centroid_finder_stays_on(html):
    """Unchanged, but pinned beside the others so a stray edit here is visible."""
    assert 'checked' in _checkbox(html, 'sensitive')
