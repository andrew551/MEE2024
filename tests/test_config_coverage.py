"""Every option the code reads must exist in DEFAULT_OPTIONS, and vice versa.

Removing an unused option is easy; noticing that some far-away module still reads it is
not. This test cross-checks the two directions statically, so a KeyError cannot reach
the user at GUI start-up.
"""

import re
from pathlib import Path

import pytest

from mee2024.config import DEFAULT_OPTIONS

PACKAGE = Path(__file__).parent.parent / 'mee2024'

# Keys injected at runtime rather than configured by the user.
RUNTIME_ONLY = {
    'no_plot',        # gravity_sweep sets this to silence plotting during its sweep
    '__version__',    # stamped into the saved config by read_ini
}

# Options that are read only by the GUI/CLI plumbing, or reserved for a stage that
# reads them out of a results file rather than from options.
UNREAD_BUT_KEPT = {
    'workDir', 'workDir2', '-DARK-', '-FLAT-',  # GUI remembers these between runs
}

KEY_PATTERN = re.compile(r"""options\[\s*['"]([A-Za-z_][\w\-]*)['"]\s*\]""")
DICT_UPDATE_PATTERN = re.compile(r"""\{\s*['"]([A-Za-z_][\w\-]*)['"]\s*:""")


def _python_sources():
    return sorted(p for p in PACKAGE.glob('*.py'))


def options_keys_read_by_the_code():
    found = {}
    for path in _python_sources():
        for match in KEY_PATTERN.finditer(path.read_text(encoding='utf-8')):
            found.setdefault(match.group(1), set()).add(path.name)
    return found


def test_every_option_read_by_the_code_has_a_default():
    missing = {key: sorted(files)
               for key, files in options_keys_read_by_the_code().items()
               if key not in DEFAULT_OPTIONS and key not in RUNTIME_ONLY}
    assert not missing, (
        'these options are read but have no default (KeyError at runtime): '
        f'{missing}')


def test_every_default_option_is_actually_used():
    read = set(options_keys_read_by_the_code())
    unused = sorted(set(DEFAULT_OPTIONS) - read - UNREAD_BUT_KEPT)
    assert not unused, (
        f'these defaults are never read -- remove them or wire them up: {unused}')


def test_ui_handler_only_writes_known_options():
    """The GUI assigns straight into the options dict; every target must be known."""
    source = (PACKAGE / 'UI_handler.py').read_text(encoding='utf-8')
    assigned = re.findall(r"""options\[\s*['"]([A-Za-z_][\w\-]*)['"]\s*\]\s*=""", source)
    unknown = sorted(set(assigned) - set(DEFAULT_OPTIONS))
    assert not unknown, f'UI_handler writes unknown options: {unknown}'


def test_defaults_are_json_round_trippable():
    """The config file is JSON, so every default must survive a round trip unchanged."""
    import json
    assert json.loads(json.dumps(DEFAULT_OPTIONS)) == DEFAULT_OPTIONS


@pytest.mark.parametrize('key', sorted(DEFAULT_OPTIONS))
def test_no_default_is_none(key):
    """None defeats the CLI's type coercion, which infers the type from the default."""
    assert DEFAULT_OPTIONS[key] is not None
