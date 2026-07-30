"""
Fetching a prebuilt catalogue on first use.

The intended user experience: pick an offline catalogue in the GUI or with
``--catalogue gaia_offline``, and if it is not present it downloads once, verifies itself
against the checksums in its own manifest, and is thereafter used with no network at all.

The registry points at GitHub release assets, with the SHA-256 of each archive baked in.
Until those assets are uploaded the URLs return 404, which is reported as an actionable
message telling the user to build locally instead. A Zenodo DOI replaces the GitHub URL
at publication; ``catalogue_sources`` in the config overrides either without a code change.
"""

import json
import shutil
import urllib.error
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

# The two archives are **disjoint magnitude slices**, not a base and a superset:
# gaia_dr3_g12 covers G < 12 and gaia_dr3_g12_13 covers 12 < G < 13. GaiaOfflineProvider
# concatenates whichever are installed, so "G < 13 coverage" means installing both. That
# split is the point: a calibration field needs only the 138 MB base, while a deep eclipse
# field adds the second archive.
GITHUB_REPO = 'andrew551/MEE2024'
#: the release tag holding the catalogue assets, kept separate from software releases
CATALOGUE_TAG = 'catalogues-v1'


def _github_asset(filename):
    return (f'https://github.com/{GITHUB_REPO}/releases/download/'
            f'{CATALOGUE_TAG}/{filename}')


RELEASES = {
    'gaia_dr3_g12': CatalogueRelease(
        name='gaia_dr3_g12',
        description='Gaia DR3, G < 12, with double-star neighbour flags',
        magnitude_limit=12.0,
        n_stars=3_087_821,
        size_bytes=137_952_319,
        doi=None,               # a Zenodo DOI replaces the GitHub URL at publication
        url=_github_asset('gaia_dr3_g12.zip'),
        sha256='f4a579e369c41b6d7099bac6b20d58c69f6b750092cd62f4085c77c670fbc5cb',
    ),
    'gaia_dr3_g12_13': CatalogueRelease(
        name='gaia_dr3_g12_13',
        description='Gaia DR3, 12 < G < 13 (optional deep extension for eclipse fields)',
        magnitude_limit=13.0,
        n_stars=4_281_806,
        size_bytes=188_640_212,
        doi=None,
        url=_github_asset('gaia_dr3_g12_13.zip'),
        sha256='28607c07a9f60c89f09ba653eac06f890ff89631d28252e9af7af63be1adb71b',
    ),
}

DEFAULT_RELEASE = 'gaia_dr3_g12'


def get_release(name=DEFAULT_RELEASE, options=None):
    """A release, with any URL and checksum from the user's config applied.

    Nothing is published yet, so the registry above carries no URLs. Rather than require a
    code edit to try a host, ``catalogue_sources`` in the config supplies them:

        {"gaia_dr3_g12": {"url": "https://...", "sha256": "..."}}

    Set it with:  mee2024 catalogue --set-source NAME --url URL [--sha256 HASH]

    That makes switching host -- an interim GitHub release now, Zenodo at publication --
    a config change rather than a release of the software.
    """
    if name not in RELEASES:
        raise KeyError(f'unknown catalogue release {name!r}; '
                       f'known: {sorted(RELEASES)}')
    release = RELEASES[name]
    source = (options or {}).get('catalogue_sources', {}).get(name) if options else None
    if source:
        release.url = source.get('url') or release.url
        release.sha256 = source.get('sha256') or release.sha256
        if source.get('size_bytes'):
            release.size_bytes = source['size_bytes']
    return release


def _google_drive_direct_url(url):
    """Rewrite a Google Drive share link into a direct-download URL, or return it as is.

    Drive is a poor host for this and is not recommended (see the note in this module),
    but if a link is given anyway, at least aim it at the download endpoint rather than
    the HTML preview page.
    """
    import re
    match = re.search(r'/file/d/([A-Za-z0-9_-]+)', url) or \
        re.search(r'[?&]id=([A-Za-z0-9_-]+)', url)
    if 'drive.google.com' in url and match:
        return f'https://drive.usercontent.google.com/download?id={match.group(1)}&export=download&confirm=t'
    return url


def _looks_like_html(head):
    start = head[:400].lstrip().lower()
    return start.startswith(b'<!doctype html') or start.startswith(b'<html')


def _download(url, destination, expected_sha256=None, progress=None):
    """Fetch a URL to a file, verifying it before it is put in place.

    Downloads to a .part file and only renames on success, so an interrupted or corrupt
    transfer never leaves something that looks like a usable archive.
    """
    destination = Path(destination)
    partial = destination.with_suffix(destination.suffix + '.part')
    request = urllib.request.Request(
        _google_drive_direct_url(url),
        headers={'User-Agent': 'mee2024', 'Accept': '*/*'})

    try:
        response_cm = urllib.request.urlopen(request)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                f'{url}\nreturned 404. The catalogue archive has not been uploaded to '
                f'that release yet.\nBuild it locally instead:\n'
                f'    python tools/build_gaia_offline.py --name '
                f'{destination.stem}\n'
                f'or point somewhere else with `mee2024 catalogue --set-source`.') from exc
        raise RuntimeError(f'{url}\nreturned HTTP {exc.code} {exc.reason}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'could not reach {url}: {exc.reason}') from exc

    with response_cm as response:
        content_type = (response.headers.get('Content-Type') or '').lower()
        total = int(response.headers.get('Content-Length') or 0)
        if progress is not None:
            progress.start(total or 1, f'Downloading {destination.name}')
        done = 0
        first_block = True
        with open(partial, 'wb') as out:
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                if first_block:
                    first_block = False
                    # A file host that returns an interstitial, a quota error or a login
                    # page answers 200 with HTML. Saving that as a .zip produces a
                    # baffling failure much later, so refuse it here.
                    if _looks_like_html(block) or 'text/html' in content_type:
                        out.close()
                        partial.unlink(missing_ok=True)
                        if progress is not None:
                            progress.finish()
                        raise ValueError(
                            f'{url} returned a web page, not a file. Hosts such as '
                            'Google Drive serve a confirmation or quota page for large '
                            'public files. Use a direct-download host (a GitHub release '
                            'asset or Zenodo), or build the catalogue locally.')
                out.write(block)
                done += len(block)
                if progress is not None:
                    progress.update(min(done, total or done))
        if progress is not None:
            progress.finish()

    if total and partial.stat().st_size != total:
        partial.unlink(missing_ok=True)
        raise ValueError(f'download truncated: expected {total} bytes, '
                         f'got {partial.stat().st_size}')
    if expected_sha256:
        actual = store.sha256(partial)
        if actual != expected_sha256:
            partial.unlink(missing_ok=True)
            raise ValueError(f'download checksum mismatch for {url}: '
                             f'expected {expected_sha256}, got {actual}')
    partial.replace(destination)
    return destination


def ensure_available(name=DEFAULT_RELEASE, progress=None, allow_download=True,
                     options=None):
    """Return the local directory for a catalogue, downloading it if necessary."""
    release = get_release(name, options=options)
    directory = release.directory()
    if release.is_installed():
        return directory

    if not release.is_published:
        raise RuntimeError(
            f'no download URL is configured for {name!r}.\n'
            f'Either point it at a host:\n'
            f'    mee2024 catalogue --set-source {name} --url URL --sha256 HASH\n'
            f'or build it locally:\n'
            f'    python tools/build_gaia_offline.py --name {name}\n'
            f'either way it ends up in {directory}')
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


def check_remote(name=None, options=None):
    """Is each catalogue actually downloadable from where the code expects?

    Sends a HEAD request and compares the reported size against the registry, without
    downloading 138 MB. Written for the moment just after uploading release assets, when
    the question is "did I get the tag and the filenames right?" -- a draft release, a
    typo in the tag, or a renamed asset all show up here as a plain 404.
    """
    names = [name] if name else list(RELEASES)
    results = []
    for entry in names:
        release = get_release(entry, options=options)
        result = {'name': entry, 'url': release.url, 'ok': False, 'detail': ''}
        if not release.is_published:
            result['detail'] = 'no URL configured'
            results.append(result)
            continue
        request = urllib.request.Request(release.url, method='HEAD',
                                         headers={'User-Agent': 'mee2024'})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                size = int(response.headers.get('Content-Length') or 0)
            result['size'] = size
            if release.size_bytes and size and size != release.size_bytes:
                result['detail'] = (f'size mismatch: expected {release.size_bytes} bytes, '
                                    f'server has {size}. Was a different file uploaded?')
            else:
                result['ok'] = True
                result['detail'] = f'reachable, {size / 1e6:.0f} MB'
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                result['detail'] = ('404 not found. Check the release tag is exactly '
                                    f'"{CATALOGUE_TAG}", the asset is named exactly '
                                    f'"{entry}.zip", and the release is published '
                                    'rather than left as a draft.')
            else:
                result['detail'] = f'HTTP {exc.code} {exc.reason}'
        except urllib.error.URLError as exc:
            result['detail'] = f'could not reach the host: {exc.reason}'
        results.append(result)
    return results


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
