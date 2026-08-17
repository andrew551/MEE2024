"""
The calibration library: master darks keyed by gain and exposure, and a master flat.

Two things drive these tests. First, the matching rules are *refusals* as much as matches --
a tier that is not there must be reported as missing rather than approximated from its
neighbour, because a defect is `pedestal + rate x time` and a neighbouring tier is the wrong
answer rather than an approximate one. Second, none of this may be decided from
``IMAGETYP``: SharpCap's sequencer cannot set it, so every scripted dark and flat in the
campaign data says ``'Light'``.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from mee2024 import calibration
from mee2024.ui import batch


def _frames(folder, names, level=500.0, gain=0, exptime=1.0, obj='DARK_G0_1p0s',
            shape=(16, 16), seed=0, spikes=(), set_temp=10.0, ccd_temp=10.2,
            instrume='ZWO ASI2600MM Pro', telescop='FRA500 0.7x reducer', focuspos=17049,
            extra=None):
    """A folder of frames whose headers look like the ones SharpCap actually writes.

    ``FRAMETYP``/``IMAGETYP`` say 'Light' on purpose -- that is what the real darks say,
    and anything that keys on them must fail these tests.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    written = []
    for index, name in enumerate(names):
        data = rng.normal(level, 2.0, shape)
        for (row, col, value) in spikes:
            data[row, col] = value
        header = fits.Header({
            'OBJECT': obj, 'FRAMETYP': 'Light', 'IMAGETYP': 'Light',
            'EXPTIME': exptime, 'GAIN': gain, 'OFFSET': 50, 'BLKLEVEL': 50,
            'ADCBITS': 16, 'CCD-TEMP': ccd_temp, 'SET-TEMP': set_temp,
            'CAMID': '351918151B020900', 'INSTRUME': instrume, 'TELESCOP': telescop,
            'XBINNING': 1, 'YBINNING': 1, 'FOCUSPOS': focuspos, 'BIASADU': 500.9,
            'DATE-OBS': f'2026-08-11T22:34:{index:02d}',
        })
        for key, value in (extra or {}).items():
            header[key] = value
        path = folder / name
        fits.writeto(path, data.astype(np.float32), header=header, overwrite=True)
        written.append(str(path))
    return written


# --------------------------------------------------------------------------- keys

def test_the_dark_key_is_gain_and_exposure_not_temperature():
    """Temperature is recorded, not keyed: for an uncooled body it is measured rather than
    chosen, so keying on it would mean never matching."""
    a = {'GAIN': 101, 'EXPTIME': 4.0, 'XBINNING': 1, 'INSTRUME': 'ZWO ASI2600MM Pro',
         'SET-TEMP': 10.0}
    b = dict(a, **{'SET-TEMP': -10.0, 'CCD-TEMP': -9.5})
    assert calibration.dark_key(a) == calibration.dark_key(b)
    assert calibration.dark_key(a) == 'dark_ASI2600MM_g101_4p000s_bin1'


def test_a_dark_that_cannot_say_what_it_is_is_refused():
    with pytest.raises(calibration.CalibrationError, match='GAIN'):
        calibration.dark_key({'EXPTIME': 4.0})


def test_the_flat_key_omits_gain_deliberately():
    """One gain-0 flat set corrects the gain-101 night data, on the stated first-order
    assumption that PRNU and vignetting do not depend on gain."""
    a = {'GAIN': 0, 'XBINNING': 1, 'INSTRUME': 'ZWO ASI2600MM Pro',
         'TELESCOP': 'FRA500 0.7x reducer', 'FOCUSPOS': 17049}
    assert calibration.flat_key(dict(a, GAIN=101)) == calibration.flat_key(a)
    # focus is in the key, because dust-donut geometry moves with it
    assert calibration.flat_key(dict(a, FOCUSPOS=25000)) != calibration.flat_key(a)


# ---------------------------------------------------------------------- combining

def test_combine_gives_the_mean_and_the_per_pixel_spread(tmp_path):
    files = _frames(tmp_path / 'd', [f'd{i}.fits' for i in range(10)], level=500.0)
    mean, sigma, info = calibration.combine(files, reject=None)
    assert mean.shape == (16, 16)
    assert float(np.median(mean)) == pytest.approx(500.0, abs=1.5)
    # the spread is the frame-to-frame scatter, which is what finds telegraph pixels;
    # a mean alone finds only the hot ones
    assert float(np.median(sigma)) == pytest.approx(2.0, rel=0.4)
    assert info['n_frames'] == 10


def test_min_max_rejection_keeps_a_cosmic_ray_out_of_the_master(tmp_path):
    """A mean of fifty carries a fiftieth of one frame's hit into every light the master
    then calibrates, which is why any rejection exists at all."""
    files = _frames(tmp_path / 'd', [f'd{i}.fits' for i in range(10)], level=500.0)
    # one frame gets a huge spike at a single pixel
    with fits.open(files[3], mode='update') as hdul:
        hdul[0].data[5, 5] = 60000.0
    plain, _, _ = calibration.combine(files, reject=None)
    rejected, _, info = calibration.combine(files)
    assert plain[5, 5] > 5000                       # the hit reached the plain mean
    assert rejected[5, 5] == pytest.approx(500.0, abs=5.0)
    assert 'high and low frame' in info['combine']


def test_sigma_clipping_could_not_have_caught_that_spike(tmp_path):
    """Why min-max and not sigma-clipping. The only spread a streaming pass can measure is
    the per-pixel standard deviation, and the outlier inflates it: one frame at 60 000 among
    nine at 500 gives that pixel a sigma near 17 800, so even a 5-sigma cut keeps it."""
    files = _frames(tmp_path / 'd', [f'd{i}.fits' for i in range(10)], level=500.0)
    with fits.open(files[3], mode='update') as hdul:
        hdul[0].data[5, 5] = 60000.0
    mean, sigma, _ = calibration.combine(files, reject=None)
    assert abs(60000.0 - mean[5, 5]) < 5.0 * sigma[5, 5]


def test_a_hot_pixel_survives_rejection(tmp_path):
    """Rejection must discriminate on the right axis. A cosmic ray is extreme in one frame;
    a hot pixel is high in every frame, and losing it would defeat the whole purpose of a
    master dark."""
    files = _frames(tmp_path / 'd', [f'd{i}.fits' for i in range(10)], level=500.0,
                    spikes=[(7, 7, 9000.0)])
    mean, sigma, _ = calibration.combine(files)
    assert mean[7, 7] == pytest.approx(9000.0, rel=0.02)
    # ...and the untrimmed sigma map still shows the pixel's real scatter, which is what
    # finds telegraph/RTS pixels
    assert sigma.shape == mean.shape


def test_too_few_frames_to_reject_says_so_rather_than_trimming(tmp_path):
    """At four frames, trimming two discards half the data."""
    files = _frames(tmp_path / 'd', ['a.fits', 'b.fits', 'c.fits', 'd.fits'])
    _, _, info = calibration.combine(files)
    assert info['combine'] == 'mean'
    assert 'min-max rejection needs' in info['note']


def test_frames_of_different_sizes_are_refused_rather_than_broadcast(tmp_path):
    files = _frames(tmp_path / 'd', ['a.fits', 'b.fits'])
    files += _frames(tmp_path / 'd2', ['c.fits'], shape=(8, 8))
    with pytest.raises(calibration.CalibrationError, match='same size'):
        calibration.combine(files, reject=None)


# ------------------------------------------------------------------- the masters

def test_a_master_dark_carries_the_provenance_needed_to_reuse_it(tmp_path):
    """`NCOMBINE` alone says a master exists and nothing about whether it may be used, and
    reuse is the only reason to save one."""
    files = _frames(tmp_path / 'd', [f'd{i}.fits' for i in range(5)], gain=101, exptime=4.0)
    mean, sigma, meta = calibration.build_master_dark(files)
    directory = calibration.write_entry(tmp_path / 'lib', mean, sigma, meta)
    header = fits.getheader(directory / calibration.MASTER_FILE)
    assert header['CALKIND'] == 'dark'
    assert header['EXPTIME'] == 4.0 and header['GAIN'] == 101
    assert header['SET-TEMP'] == 10.0 and header['CCD-TEMP'] == 10.2
    assert header['CAMID'] == '351918151B020900'
    assert header['NCOMBINE'] == 5
    assert (directory / calibration.SIGMA_FILE).exists()


def test_a_flat_below_mid_range_says_so(tmp_path):
    """The mid-range fill is what makes it safe to take no flat-darks: the offset pedestal
    is left in, and its share of the signal grows as the fill falls."""
    good = _frames(tmp_path / 'ok', [f'f{i}.fits' for i in range(4)],
                   level=33000.0, obj='FLATS')
    _, _, meta = calibration.build_master_flat(good)
    assert meta['warnings'] == []
    assert meta['fill_fraction'] == pytest.approx(0.503, abs=0.02)

    dim = _frames(tmp_path / 'dim', [f'f{i}.fits' for i in range(4)],
                  level=3000.0, obj='FLATS')
    _, _, meta = calibration.build_master_flat(dim)
    assert any('full scale' in w for w in meta['warnings'])


def test_a_flat_level_is_not_mistaken_for_a_broken_bias_header(tmp_path):
    """A correctly exposed flat sits at ~33000 ADU against a BIASADU of ~500. That is not a
    broken header, and warning about it would train the reader to ignore the warning that
    matters."""
    files = _frames(tmp_path / 'f', [f'f{i}.fits' for i in range(4)],
                    level=33000.0, obj='FLATS')
    _, _, meta = calibration.build_master_flat(files)
    assert not any('BIASADU' in w for w in meta['warnings'])


def test_a_master_flat_is_normalised_to_about_one(tmp_path):
    files = _frames(tmp_path / 'f', [f'f{i}.fits' for i in range(4)],
                    level=33000.0, obj='FLATS')
    mean, _, meta = calibration.build_master_flat(files)
    assert float(np.median(mean)) == pytest.approx(1.0, abs=0.01)
    assert meta['normalised'] is True


def test_a_bias_header_that_is_wrong_by_a_factor_is_reported(tmp_path):
    """On the 533MM the header said 393.59 against a measured 94.66."""
    files = _frames(tmp_path / 'd', [f'd{i}.fits' for i in range(4)], level=95.0,
                    extra={'BIASADU': 393.59})
    _, _, meta = calibration.build_master_dark(files)
    assert any('BIASADU' in w for w in meta['warnings'])


# -------------------------------------------------------------------- the library

@pytest.fixture
def library(tmp_path):
    """Two dark tiers and one flat, in the layout the capture scripts produce."""
    _frames(tmp_path / 'DARKS' / 'DARK_G101_4s' / '22_39_39',
            [f'd{i}.fits' for i in range(4)], gain=101, exptime=4.0, obj='DARK_G101_4s')
    _frames(tmp_path / 'DARKS' / 'DARK_G101_6s' / '22_43_26',
            [f'd{i}.fits' for i in range(4)], gain=101, exptime=6.0, obj='DARK_G101_6s')
    _frames(tmp_path / 'FLATS' / '23_05_00', [f'f{i}.fits' for i in range(4)],
            level=33000.0, obj='FLATS')
    root = tmp_path / 'lib'
    calibration.build_library(root, darks_root=tmp_path / 'DARKS',
                              flats_root=tmp_path / 'FLATS',
                              on_note=lambda *a, **k: None)
    return root


def test_the_library_holds_one_entry_per_tier(library):
    entries = calibration.read_index(library)
    kinds = sorted((e['kind'], e['header'].get('EXPTIME')) for e in entries)
    assert kinds == [('dark', 4.0), ('dark', 6.0), ('flat', 1.0)]
    index = json.loads((Path(library) / calibration.INDEX_FILE).read_text(encoding='utf-8'))
    assert len(index['entries']) == 3


def test_rebuilding_a_tier_supersedes_it_rather_than_accumulating(tmp_path, library):
    """Two nights of the same tier should not leave two near-duplicates nobody can choose
    between."""
    before = len(calibration.read_index(library))
    _frames(tmp_path / 'MORE' / 'DARK_G101_4s' / '23_00_00',
            [f'd{i}.fits' for i in range(4)], gain=101, exptime=4.0, obj='DARK_G101_4s')
    calibration.build_library(library, darks_root=tmp_path / 'MORE',
                              on_note=lambda *a, **k: None)
    assert len(calibration.read_index(library)) == before


def test_a_matching_tier_is_found_by_gain_and_exposure(library):
    entries = calibration.read_index(library)
    entry, note = calibration.match_dark(
        entries, {'GAIN': 101, 'EXPTIME': 6.0, 'XBINNING': 1,
                  'INSTRUME': 'ZWO ASI2600MM Pro', 'SET-TEMP': 10.0})
    assert entry is not None
    assert entry['header']['EXPTIME'] == 6.0
    assert note is None


def test_a_missing_tier_is_reported_not_interpolated(library):
    """Defect amplitude is pedestal + rate x time -- the pedestal is 63% of a defect at
    0.1 s and 1.7% at 10 s -- so a neighbouring tier is wrong rather than approximate."""
    entries = calibration.read_index(library)
    entry, reason = calibration.match_dark(
        entries, {'GAIN': 101, 'EXPTIME': 5.0, 'XBINNING': 1,
                  'INSTRUME': 'ZWO ASI2600MM Pro'})
    assert entry is None
    assert 'not interpolated' in reason
    assert 'g101/4s' in reason and 'g101/6s' in reason


def test_a_different_gain_does_not_match_however_close_the_exposure(library):
    entries = calibration.read_index(library)
    entry, reason = calibration.match_dark(
        entries, {'GAIN': 0, 'EXPTIME': 4.0, 'XBINNING': 1,
                  'INSTRUME': 'ZWO ASI2600MM Pro'})
    assert entry is None
    assert 'gain 0' in reason


def test_a_temperature_mismatch_is_a_warning_not_a_refusal(library):
    entries = calibration.read_index(library)
    entry, note = calibration.match_dark(
        entries, {'GAIN': 101, 'EXPTIME': 4.0, 'XBINNING': 1,
                  'INSTRUME': 'ZWO ASI2600MM Pro', 'SET-TEMP': -10.0})
    assert entry is not None                     # still the right tier
    assert note and 'setpoint' in note


def test_a_flat_matches_across_gains_and_says_so(library):
    entries = calibration.read_index(library)
    entry, note = calibration.match_flat(
        entries, {'GAIN': 101, 'XBINNING': 1, 'INSTRUME': 'ZWO ASI2600MM Pro',
                  'TELESCOP': 'FRA500 0.7x reducer', 'FOCUSPOS': 17049})
    assert entry is not None
    assert note and 'gain' in note


def test_resolve_for_field_returns_paths_and_notes(tmp_path, library):
    frames = _frames(tmp_path / 'Z1' / '22_16_15', ['z1.fits', 'z2.fits'],
                     gain=101, exptime=4.0, obj='Z1_base')
    resolved = calibration.resolve_for_field(library, frames,
                                            on_note=lambda *a, **k: None)
    assert resolved['dark'] and Path(resolved['dark']).exists()
    assert resolved['flat'] and Path(resolved['flat']).exists()
    assert any('master dark' in note for note in resolved['notes'])


def test_a_field_with_no_matching_tier_runs_uncalibrated_and_says_why(tmp_path, library):
    frames = _frames(tmp_path / 'X' / '22_16_15', ['x1.fits', 'x2.fits'],
                     gain=101, exptime=99.0, obj='X_base')
    resolved = calibration.resolve_for_field(library, frames,
                                            on_note=lambda *a, **k: None)
    assert resolved['dark'] is None
    assert any('no dark' in note for note in resolved['notes'])


# ------------------------------------------------------- reading a master back in

def test_a_library_master_is_used_as_is_rather_than_recombined(tmp_path, library):
    """A normalised master flat must not be normalised twice: the second division is by a
    number near 1, so it does almost nothing and looks like it worked."""
    flat = next(e for e in calibration.read_index(library) if e['kind'] == 'flat')
    image, info = calibration.load_or_combine([flat['master']])
    assert info['source'] == 'library master'
    assert info['normalised'] is True
    assert float(np.median(image)) == pytest.approx(1.0, abs=0.01)


def test_an_ordinary_frame_is_not_mistaken_for_a_master(tmp_path):
    files = _frames(tmp_path / 'd', ['one.fits'])
    _, info = calibration.load_or_combine(files)
    assert info['source'] == 'single frame'
    assert info['normalised'] is False


# ------------------------------------------------- keeping calibration out of a batch

def test_calibration_folders_are_recognised_by_name(tmp_path):
    assert calibration.looks_like_calibration(tmp_path / 'DARKS' / 'DARK_G0_0p1s', tmp_path)
    assert calibration.looks_like_calibration(tmp_path / 'FLATS' / '23_05_00', tmp_path)
    assert not calibration.looks_like_calibration(tmp_path / 'Zenith' / 'Z1_base', tmp_path)
    # a word boundary, so this is not caught by 'dark'
    assert not calibration.looks_like_calibration(tmp_path / 'Darkfield_survey', tmp_path)


def test_calibration_folders_are_recognised_from_object_when_the_name_is_silent(tmp_path):
    """The capture scripts put the type in TARGETNAME, which lands in OBJECT. It is the only
    place a scripted capture can record it at all."""
    files = _frames(tmp_path / '22_34_05', ['a.fits', 'b.fits'], obj='DARK_G0_0p1s')
    assert calibration.classify_frames(files) == 'dark'
    files = _frames(tmp_path / '23_05_00', ['a.fits'], obj='FLATS')
    assert calibration.classify_frames(files) == 'flat'
    files = _frames(tmp_path / '22_16_15', ['a.fits'], obj='Z1_base')
    assert calibration.classify_frames(files) is None


def test_imagetyp_is_never_consulted(tmp_path):
    """Every scripted dark in the campaign data says IMAGETYP = 'Light', because the
    sequencer has no frame-type command. Anything keying on it treats fifty darks as a
    light field."""
    files = _frames(tmp_path / '22_34_05', ['a.fits'], obj='DARK_G0_0p1s')
    assert fits.getheader(files[0])['IMAGETYP'] == 'Light'
    assert calibration.classify_frames(files) == 'dark'


def test_a_batch_leaves_the_dark_tiers_out_and_says_how_many(tmp_path):
    _frames(tmp_path / 'DARKS' / 'DARK_G0_0p1s' / '22_34_05', ['a.fits', 'b.fits'],
            obj='DARK_G0_0p1s')
    _frames(tmp_path / 'DARKS' / 'DARK_G0_0p3s' / '22_34_24', ['a.fits', 'b.fits'],
            obj='DARK_G0_0p3s')
    _frames(tmp_path / 'FLATS' / '23_05_00', ['a.fits', 'b.fits'], obj='FLATS')
    _frames(tmp_path / 'Zenith' / 'Z1_base' / '22_16_15', ['a.fits', 'b.fits'],
            obj='Z1_base')
    fields, info = batch.find_fields(tmp_path)
    assert [f['name'] for f in fields] == ['22_16_15']
    assert len(info['calibration']) == 3
    assert '3 calibration folder(s) skipped' in batch.describe(fields, info)


def test_the_calibration_folders_are_findable_when_they_are_what_you_want(tmp_path):
    _frames(tmp_path / 'DARKS' / 'DARK_G0_0p1s' / '22_34_05', ['a.fits', 'b.fits'],
            obj='DARK_G0_0p1s')
    sets, _ = calibration.find_calibration_sets(tmp_path / 'DARKS')
    assert [s['kind'] for s in sets] == ['dark']
    # named for the tier, not for the capture timestamp that identifies nothing
    assert sets[0]['name'] == 'DARK_G0_0p1s'


def test_a_failed_cooler_is_caught_even_though_the_setpoint_matches(library):
    """The setpoint is a *request*: a cooler that failed to hold it still reports the value
    it was asked for, so checking the setpoint alone stays silent on the case worth
    catching."""
    entries = calibration.read_index(library)
    entry, note = calibration.match_dark(
        entries, {'GAIN': 101, 'EXPTIME': 4.0, 'XBINNING': 1,
                  'INSTRUME': 'ZWO ASI2600MM Pro',
                  'SET-TEMP': 10.0,        # the same request as the master
                  'CCD-TEMP': 31.0})       # ...and the cooler is not holding it
    assert entry is not None
    assert note and 'measured sensor temperature' in note


def test_normal_cooler_ripple_is_not_a_warning(library):
    """The Leon darks span 9.6-12.4 C on one +10 C setpoint. Warning about that would be
    warning about every cooled set there is."""
    entries = calibration.read_index(library)
    _, note = calibration.match_dark(
        entries, {'GAIN': 101, 'EXPTIME': 4.0, 'XBINNING': 1,
                  'INSTRUME': 'ZWO ASI2600MM Pro', 'SET-TEMP': 10.0, 'CCD-TEMP': 12.4})
    assert note is None
