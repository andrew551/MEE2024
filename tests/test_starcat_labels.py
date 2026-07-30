"""The bright-star label index, and the Hipparcos bright fill."""

import numpy as np
import pytest

from mee2024.starcat import ORIGIN_GAIA, ORIGIN_HIPPARCOS, ORIGIN_TYCHO
from mee2024.starcat import providers
from mee2024.starcat.labels import LabelIndex, raw_label

# HIP numbers of stars whose names everyone knows
SIRIUS, VEGA, ARCTURUS, CANOPUS = 32349, 91262, 69673, 30438

bundled_labels = pytest.mark.skipif(
    LabelIndex.try_bundled() is None,
    reason='the label index has not been built; run tools/build_hipparcos.py')
bundled_hipparcos = pytest.mark.skipif(
    providers.HipparcosProvider.try_bundled() is None,
    reason='the Hipparcos catalogue has not been built; run tools/build_hipparcos.py')


# ------------------------------------------------------------- raw fallbacks

def test_raw_label_formats_each_origin():
    assert raw_label(123456789012345678, ORIGIN_GAIA) == 'gaia:123456789012345678'
    assert raw_label(91262, ORIGIN_HIPPARCOS) == 'HIP 91262'
    packed = providers.encode_tycho_id(1234, 567, 1)
    assert raw_label(packed, ORIGIN_TYCHO) == 'TYC 1234-567-1'


# ------------------------------------------------------------- the real index

@bundled_labels
def test_index_reports_its_contents():
    index = LabelIndex.bundled()
    assert index.manifest['n_hip'] > 100000
    assert index.manifest['n_gaia_crossmatches'] > 90000
    assert index.manifest['n_named'] > 20
    assert 'HIP' in repr(index)


@bundled_labels
@pytest.mark.parametrize('hip,name', [
    (SIRIUS, 'Sirius'), (VEGA, 'Vega'), (ARCTURUS, 'Arcturus'), (CANOPUS, 'Canopus'),
])
def test_named_stars_resolve_from_a_hip_id(hip, name):
    index = LabelIndex.bundled()
    assert index.name_for([hip], ORIGIN_HIPPARCOS) == [name]
    assert index.label_for([hip], ORIGIN_HIPPARCOS) == [name]


@bundled_labels
def test_an_unnamed_hip_star_falls_back_to_its_hip_number():
    index = LabelIndex.bundled()
    # HIP 100 is a real star with no proper name
    assert index.name_for([100], ORIGIN_HIPPARCOS) == [None]
    assert index.label_for([100], ORIGIN_HIPPARCOS) == ['HIP 100']


@bundled_labels
def test_an_unknown_gaia_id_falls_back_to_the_raw_id():
    index = LabelIndex.bundled()
    assert index.label_for([1], ORIGIN_GAIA) == ['gaia:1']
    assert index.hip_for([1], ORIGIN_GAIA).tolist() == [0]


@bundled_labels
def test_a_gaia_id_resolves_through_the_crossmatch():
    """Take a real crossmatched pair out of the index and check it round trips."""
    index = LabelIndex.bundled()
    source_id = int(np.asarray(index.gaia_id)[1000])
    expected_hip = int(np.asarray(index.gaia_hip)[1000])
    assert index.hip_for([source_id], ORIGIN_GAIA).tolist() == [expected_hip]
    assert index.label_for([source_id], ORIGIN_GAIA)[0] == f'HIP {expected_hip}'


@bundled_labels
def test_lookups_are_vectorised_over_mixed_origins():
    index = LabelIndex.bundled()
    ids = np.array([VEGA, 1, SIRIUS], dtype=np.int64)
    origins = np.array([ORIGIN_HIPPARCOS, ORIGIN_GAIA, ORIGIN_HIPPARCOS], dtype=np.uint8)
    assert index.label_for(ids, origins) == ['Vega', 'gaia:1', 'Sirius']


@bundled_labels
def test_hip_numbers_stay_integral():
    """A HIP number must never be corrupted by a float round trip."""
    index = LabelIndex.bundled()
    assert index.hip_for([VEGA], ORIGIN_HIPPARCOS).dtype == np.int64


# ------------------------------------------------------- the Hipparcos filler

@bundled_hipparcos
def test_hipparcos_provider_reports_itself():
    provider = providers.HipparcosProvider()
    assert provider.is_offline
    assert provider.catalogue.n_stars > 100000
    assert 'hipparcos' in provider.describe()


@bundled_hipparcos
@pytest.mark.parametrize('name,ra,dec,v', [
    ('Sirius', 101.287, -16.716, -1.46), ('Vega', 279.234, 38.784, 0.03),
    ('Arcturus', 213.915, 19.182, -0.05), ('Canopus', 95.988, -52.696, -0.74),
])
def test_hipparcos_has_the_stars_gaia_lacks(name, ra, dec, v):
    """Gaia DR3 has no entry at all for these -- they saturate the instrument."""
    found = providers.HipparcosProvider().lookup(
        (ra - 0.05, ra + 0.05), (dec - 0.05, dec + 0.05), max_magnitude=12.0, epoch=2023.84)
    assert len(found) >= 1, f'{name} missing from Hipparcos'
    brightest = float(np.min(found.mag))
    # the estimated G should land within a magnitude of the catalogue V
    assert abs(brightest - v) < 1.0, f'{name}: estimated G={brightest:.2f} vs V={v}'


@bundled_hipparcos
def test_hipparcos_stars_are_precision_grade():
    """Unlike Tycho, Hipparcos astrometry is good enough for the distortion fit."""
    found = providers.HipparcosProvider().lookup((100.0, 103.0), (-18.0, -15.0), 12.0)
    assert len(found)
    assert found.is_precision_grade().all()
    assert np.all(found.origin == ORIGIN_HIPPARCOS)


@bundled_hipparcos
def test_hipparcos_carries_proper_motion_and_parallax():
    found = providers.HipparcosProvider().lookup((100.0, 103.0), (-18.0, -15.0), 10.0)
    assert found.has_proper_motion().mean() > 0.9
    assert np.isfinite(found.parallax).mean() > 0.9


# ------------------------------------------------------------ the merge chain

@bundled_hipparcos
def test_merge_prefers_hipparcos_over_tycho_for_the_bright_fill():
    """Hipparcos is precision-grade, so it must be tried first."""
    merged = providers.MergedProvider(primary=providers.TychoProvider())
    names = [provider.name for provider, _ in merged.fills]
    assert names.index('hipparcos') < names.index('tycho')


def test_merge_marks_a_mixed_magnitude_system():
    """Magnitudes from different catalogues must not be passed off as measured G."""
    from mee2024.starcat import StarTable

    def table(ra, mag, origin, band='G'):
        return StarTable(ra=np.radians(ra), dec=np.radians([0.0] * len(ra)), mag=mag,
                         ids=np.arange(1, len(ra) + 1, dtype=np.int64),
                         epoch=2024.0, origin=origin, band=band)

    class Fake(providers.CatalogueProvider):
        is_offline = True

        def __init__(self, t, name):
            self._t, self.name = t, name

        def lookup(self, ra_range, dec_range, max_magnitude=12.0, epoch=2024.0):
            out = self._t.select(self._t.mag < max_magnitude)
            out.epoch = epoch
            return out

    primary = Fake(table([10.0], [8.0], ORIGIN_GAIA), 'primary')
    filler = Fake(table([200.0], [3.0], ORIGIN_HIPPARCOS, 'G_est_from_Hp'), 'filler')
    merged = providers.MergedProvider(primary=primary, fills=[(filler, 9.0)])
    result = merged.lookup((0.0, 360.0), (-5.0, 5.0), 12.0)
    assert len(result) == 2
    assert result.band == 'G_mixed'


def test_distortion_fitter_excludes_non_precision_catalogues():
    """The guard that keeps Tycho positions out of the precision fit."""
    import inspect
    from mee2024 import distortion_fitter
    source = inspect.getsource(distortion_fitter.match_and_fit_distortion)
    assert 'is_precision_grade' in source
