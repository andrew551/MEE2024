"""The two standard stage-1 configurations, defined once for every interface.

A zenith field has thousands of stars on a flat sky; an eclipse-day field has tens of
stars on a steep bright gradient beside a saturated object. The same detector settings
cannot serve both, and until now the difference lived in operators' heads and in this
project's shell drivers -- which is how a reduction could silently run the wrong one.

Every value here is one the reductions of record actually used (docs/FIELD_PRESETS.md
lists each difference with its reason). The two that are not simply "sensitivity" are
worth stating outright, because they were measured rather than chosen:

  * `background_subtraction_mode` moves the fitted plate scale by 19.1 ppm between its
    two settings on frame-identical stacks -- roughly 0.2 arcsec of deflection constant
    on Bruns' optics. `Gaussian` on eclipse fields is what reproduces Bruns 2018 end to
    end; `annular` is inherited practice at zenith, where the same A/B has not been run.
  * `centroid_refine_window` chooses the estimator, and is worth under 2 ppm -- much less
    than the background mode, which is the opposite of what this project assumed for two
    days (docs/MATRIX_2026.md).

A preset is a starting point, not a lock: an interface applies it and the user may then
change anything it set. What must not happen is a preset silently setting something the
interface cannot show, so `PRESET_KEYS` is the contract a UI can check itself against.
"""

FIELD_PRESETS = {
    'zenith': {
        'label': 'Zenith / night calibration',
        'blurb': 'Thousands of stars on a flat sky. Plain per-frame detection, '
                 'ring background, tighter thresholds.',
        'options': {
            'sensitive_mode_stack': True,
            'centroid_gaussian_subtract': False,
            'centroid_gaussian_thresh': 5.0,
            'min_area': 4,
            'sigma_subtract': 3.0,
            'background_subtraction_mode': 'annular',
            'centroid_refine_window': True,
            'centroid_window_sigma': 2.0,
            'delete_saturated_blob': False,
            'coronal_subtract': False,
        },
    },
    'eclipse': {
        'label': 'Eclipse day (Sun or Moon)',
        'blurb': 'Tens of stars on a steep gradient beside a saturated object. '
                 'Sensitive detection, smooth background, the Sun/Moon masked.',
        'options': {
            'sensitive_mode_stack': True,
            'centroid_gaussian_subtract': True,
            'centroid_gaussian_thresh': 4.0,
            'min_area': 2,
            'sigma_subtract': 0.0,
            'background_subtraction_mode': 'Gaussian',
            'centroid_refine_window': False,
            'centroid_window_sigma': 2.0,
            'delete_saturated_blob': True,
            'eclipse_mask_mode': 'disk',
            'coronal_subtract': True,
        },
    },
}

# every key any preset sets: the set an interface must be able to display
PRESET_KEYS = sorted({k for p in FIELD_PRESETS.values() for k in p['options']})


def apply_field_preset(options, name):
    """Apply a preset in place and record which one. Returns `options`.

    The name is written to `field_preset` so `results.txt` states which standard produced
    a reduction; 'custom' means the user changed something afterwards or never picked one.
    Unknown names raise rather than silently doing nothing -- a preset that quietly failed
    to apply is the failure this module exists to prevent.
    """
    if name in (None, '', 'custom'):
        options['field_preset'] = 'custom'
        return options
    if name not in FIELD_PRESETS:
        raise ValueError(f'unknown field preset {name!r}; '
                         f'expected one of {sorted(FIELD_PRESETS)} or "custom"')
    options.update(FIELD_PRESETS[name]['options'])
    options['field_preset'] = name
    return options


def matching_preset(options):
    """The preset whose every option matches, or 'custom'.

    Used to label a reduction honestly when the user assembled the settings by hand: if
    they happen to be exactly a standard, say so; if one value differs, do not.
    """
    for name, preset in FIELD_PRESETS.items():
        if all(options.get(k) == v for k, v in preset['options'].items()):
            return name
    return 'custom'
