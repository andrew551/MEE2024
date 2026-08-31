"""The two standard field configurations (F25).

The presets exist because the difference between a zenith and an eclipse-day reduction
used to live in operators' heads and in shell drivers, so a reduction could silently run
the wrong one. These tests pin the two properties that makes them trustworthy: the
measured values are what they say they are, and a preset the user then edited stops
claiming to be that preset.
"""

import pytest

from mee2024 import field_presets as fp
from mee2024.config import DEFAULT_OPTIONS, get_default_options


def test_the_two_presets_exist_and_are_labelled():
    assert set(fp.FIELD_PRESETS) == {'zenith', 'eclipse'}
    for preset in fp.FIELD_PRESETS.values():
        assert preset['label'] and preset['blurb']


@pytest.mark.parametrize('name', sorted(fp.FIELD_PRESETS))
def test_every_preset_key_is_a_real_option(name):
    """A preset that sets a key nothing reads would apply silently and do nothing."""
    for key in fp.FIELD_PRESETS[name]['options']:
        assert key in DEFAULT_OPTIONS, f'{name} sets unknown option {key!r}'


def test_the_background_mode_is_the_measured_split():
    """19.1 ppm of plate scale rides on this one; if it ever flips, say why in the diff."""
    assert fp.FIELD_PRESETS['zenith']['options']['background_subtraction_mode'] == 'annular'
    assert fp.FIELD_PRESETS['eclipse']['options']['background_subtraction_mode'] == 'Gaussian'


def test_the_eclipse_preset_is_the_bruns_reproducing_convention():
    """Gaussian background + footprint moments is what reproduces Bruns 2018 end to end
    (L = 1.720 +- 0.069 against his 1.752 +- 0.060) -- docs/MATRIX_2026.md."""
    ecl = fp.FIELD_PRESETS['eclipse']['options']
    assert ecl['centroid_refine_window'] is False
    assert ecl['background_subtraction_mode'] == 'Gaussian'


def test_applying_a_preset_sets_every_value_and_names_itself():
    options = get_default_options()
    fp.apply_field_preset(options, 'eclipse')
    for key, value in fp.FIELD_PRESETS['eclipse']['options'].items():
        assert options[key] == value
    assert options['field_preset'] == 'eclipse'
    assert fp.matching_preset(options) == 'eclipse'


def test_an_edited_preset_no_longer_claims_to_be_one():
    options = get_default_options()
    fp.apply_field_preset(options, 'zenith')
    options['min_area'] = 9
    assert fp.matching_preset(options) == 'custom'


def test_custom_is_accepted_and_unknown_names_are_not():
    options = get_default_options()
    assert fp.apply_field_preset(options, 'custom')['field_preset'] == 'custom'
    assert fp.apply_field_preset(options, None)['field_preset'] == 'custom'
    with pytest.raises(ValueError):
        fp.apply_field_preset(options, 'deep')


def test_the_app_window_offers_both_field_presets():
    from mee2024.ui.runner import PipelineRunner
    for name in fp.FIELD_PRESETS:
        assert name in PipelineRunner.PRESETS


def test_the_app_runner_applies_a_field_preset():
    from mee2024.ui.runner import PipelineRunner
    options = PipelineRunner().build_options({'preset': 'eclipse'})
    assert options['background_subtraction_mode'] == 'Gaussian'
    assert options['centroid_refine_window'] is False
    assert options['coronal_subtract'] is True
