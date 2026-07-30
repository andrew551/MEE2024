"""
Fetching a prebuilt catalogue on first use.

The intended user experience: pick an offline catalogue in the GUI or with
``--catalogue gaia_offline``, and if it is not present it downloads once, verifies itself
against the checksums in its own manifest, and is thereafter used with no network at all.

**No archive has been published yet**, so the registry below carries placeholder Zenodo
entries. Everything except the actual URL and checksum is in place: resolution, download
with progress, extraction, verification and the on-disk location. Filling in a real
record is a one-line edit per catalogue.
"""

import json
import shutil
import urllib.request
from pathlib import Path

from mee2024.MEE2024util import get_catalogue_root
from mee2024.starcat import store


class CatalogueRelease:
    """A downloadable catalogue archive.

    url/sha256 of None marks an unpublished placeholder: everything resolves and reports
    sensibly, but attempting a download raises a clear error rather than failing obscurely.
    """

    def __init__(self, name, description, url=None, sha256=None, size_bytes=None,
                 doi=None, magnitude_limit=None, n_stars=None):
        self.name = name
        self.description = description
        self.url = url
        self.sha256 = sha256
        self.size_bytes = size_bytes
        self.doi = doi
        self.magnitude_limit = magnitude_limit
        self.n_stars = n_stars

    @property
    def is_published(self):
        return self.url is not None

    def directory(self):
        return get_catalogue_root() / self.name

    def is_installed(self):
        try:
            store.read_manifest(self.directory())
            return True
        except (FileNotFoundError, ValueError):
            return False

    def human_size(self):
        if not self.size_bytes:
            return 'unknown size'
        return f'{self.size_bytes / 1e6:.0f} MB'

    def describe(self):
        state = 'installed' if self.is_installed() else (
            'available' if self.is_published else 'not yet published')
        return f'{self.name}: {self.description} ({self.human_size()}, {state})'


# ---------------------------------------------------------------------------
# The catalogue registry.
#
# Depth was chosen as G<12 (~2.9 M stars, ~139 MB): it exceeds the default
# max_star_mag_dist of 12, so it covers every ordinary run. The deep archive covering
# 12 < G < 13 is a separate optional download, because mag-13 stars are detectable in
# eclipse fields and worth having, but should not be forced on everyone. A provider can
# read both and concatenate.
# ---------------------------------------------------------------------------

RELEASES = {
    'gaia_dr3_g12': CatalogueRelease(
        name='gaia_dr3_g12',
        description='Gaia DR3, G < 12, with double-star neighbour flags',
        magnitude_limit=12.0,
        n_stars=2_900_000,      # estimated; corrected when the archive is built
        size_bytes=139_000_000,
        # --- placeholder: fill in once deposited on Zenodo ---
        doi=None,               # e.g. '10.5281/zenodo.XXXXXXX'
        url=None,               # e.g. 'https://zenodo.org/records/XXXXXXX/files/gaia_dr3_g12.zip'
        sha256=None,
    ),
    'gaia_dr3_g12_13': CatalogueRelease(
        name='gaia_dr3_g12_13',
        description='Gaia DR3, 12 < G < 13 (optional deep extension for eclipse fields)',
        magnitude_limit=13.0,
        n_stars=4_470_000,
        size_bytes=215_000_000,
        doi=None,
        url=None,
        sha256=None,
    ),
}

DEFAULT_RELEASE = 'gaia_dr3_g12'


def get_release(name=DEFAULT_RELEASE):
    if name not in RELEASES:
        raise KeyError(f'unknown catalogue release {name!r}; '
                       f'known: {sorted(RELEASES)}')
    return RELEASES[name]


def _download(url, destination, expected_sha256=None, progress=None):
    destination = Path(destination)
    partial = destination.with_suffix(destination.suffix + '.part')
    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get('Content-Length') or 0)
        if progress is not None:
            progress.start(total or 1, f'Downloading {destination.name}')
        done = 0
        with open(partial, 'wb') as out:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if progress is not None:
                    progress.update(min(done, total or done))
        if progress is not None:
            progress.finish()
    if expected_sha256 is not None:
        actual = store.sha256(partial)
        if actual != expected_sha256:
            partial.unlink(missing_ok=True)
            raise ValueError(f'download checksum mismatch for {url}: '
                             f'expected {expected_sha256}, got {actual}')
    partial.replace(destination)
    return destination


def ensure_available(name=DEFAULT_RELEASE, progress=None, allow_download=True):
    """Return the local directory for a catalogue, downloading it if necessary."""
    release = get_release(name)
    directory = release.directory()
    if release.is_installed():
        return directory

    if not release.is_published:
        raise RuntimeError(
            f'catalogue {name!r} has not been published yet, so it cannot be '
            f'downloaded automatically.\n'
            f'Build it locally with:\n'
            f'    python tools/build_gaia_offline.py --name {name}\n'
            f'and it will be written to {directory}')
    if not allow_download:
        raise RuntimeError(f'catalogue {name!r} is not installed at {directory} '
                           'and downloading is disabled')

    directory.parent.mkdir(parents=True, exist_ok=True)
    archive = directory.parent / f'{name}.zip'
    _download(release.url, archive, release.sha256, progress=progress)
    try:
        store.unpack(archive, directory, verify_checksums=True)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        archive.unlink(missing_ok=True)
    return directory


def installed_catalogues():
    """Every catalogue release currently present on disk."""
    return [release for release in RELEASES.values() if release.is_installed()]


def status():
    """A human-readable summary, for `mee2024 catalogue --list`."""
    lines = [f'catalogue directory: {get_catalogue_root()}']
    for release in RELEASES.values():
        lines.append('  ' + release.describe())
        if release.doi:
            lines.append(f'      doi: {release.doi}')
    return '\n'.join(lines)


def write_local_release_stub(directory, name, sha256_value=None):
    """Record a locally built catalogue so `status()` reports it.

    Used by the builder so a catalogue built from scratch looks the same to the rest of
    the code as one that was downloaded.
    """
    directory = Path(directory)
    manifest = store.read_manifest(directory)
    release = RELEASES.get(name)
    if release is not None:
        release.n_stars = manifest['n_stars']
        release.magnitude_limit = manifest['magnitude_limit']
    (directory / 'local_build.json').write_text(
        json.dumps({'name': name, 'sha256': sha256_value,
                    'n_stars': manifest['n_stars']}, indent=2), encoding='utf-8')
    return directory
