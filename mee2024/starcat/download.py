"""
Fetching a prebuilt catalogue on first use.

The intended user experience: pick an offline catalogue in the GUI or with
``--catalogue gaia_offline``, and if it is not present it downloads once, verifies itself
against the checksums in its own manifest, and is thereafter used with no network at all.

The registry points at GitHub release assets, with the SHA-256 of each archive baked in.
If an asset is missing the URL returns 404, which is reported as an actionable message
telling the user to build locally instead. A Zenodo DOI replaces the GitHub URL at
publication; ``catalogue_sources`` in the config overrides either without a code change.
"""

import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path

from mee2024.MEE2024util import get_catalogue_root
from mee2024.starcat import store


#: A download at or above this size is not started without the user agreeing to it
#: first. The standard archive (320 MB) is well under; the deep one (1.57 GB) is not,
#: and it also wants several gigabytes of free disk while it unpacks.
LARGE_DOWNLOAD_BYTES = 1_000_000_000


class ConfirmationRequired(RuntimeError):
    """A download big enough that it has to be agreed to, not merely triggered.

    Raised by :func:`ensure_available` instead of quietly spending an hour of someone's
    connection and several GB of their disk. Every front end catches it and asks in
    whatever way suits it -- a dialog, a terminal prompt -- then calls again with
    ``confirm``. Carries the release so the caller can quote the exact size.
    """

    def __init__(self, release, message):
        super().__init__(message)
        self.release = release
        self.size_bytes = release.size_bytes


class CatalogueRelease:
    """A downloadable catalogue archive.

    url/sha256 of None marks an unpublished placeholder: everything resolves and reports
    sensibly, but attempting a download raises a clear error rather than failing obscurely.
    """

    def __init__(self, name, description, url=None, sha256=None, size_bytes=None,
                 doi=None, magnitude_limit=None, n_stars=None, role='base',
                 installed_bytes=None):
        self.name = name
        self.description = description
        self.url = url
        self.sha256 = sha256
        self.size_bytes = size_bytes
        #: measured size once unpacked, where it is known. The download and the unpacked
        #: copy exist at the same time, so the disk needed to install is the sum
        self.installed_bytes = installed_bytes
        self.doi = doi
        self.magnitude_limit = magnitude_limit
        self.n_stars = n_stars
        #: 'base' is usable on its own; 'extension' only adds depth on top of a base
        self.role = role

    @property
    def needs_confirmation(self):
        """Is this big enough that it should be agreed to before it starts?"""
        return bool(self.size_bytes and self.size_bytes >= LARGE_DOWNLOAD_BYTES)

    def disk_needed_bytes(self):
        """Peak free disk to install: the archive and its unpacked copy coexist."""
        if not self.size_bytes:
            return None
        # where the unpacked size has not been measured, the archives seen so far unpack
        # to about 1.2x their compressed size -- stated as approximate either way
        unpacked = self.installed_bytes or int(self.size_bytes * 1.2)
        return self.size_bytes + unpacked

    def size_warning(self):
        """What to tell someone before committing them to this download, or None.

        Exact bytes, not a rounded headline: "1.6 GB" reads as a detail, whereas the
        real number reads as a decision. Names the cheaper alternative too, because for
        almost everyone it is the right answer.
        """
        if not self.needs_confirmation:
            return None
        disk = self.disk_needed_bytes()
        default = RELEASES.get(DEFAULT_RELEASE)
        lines = [
            f'{self.name} is a large download: {self.size_bytes:,} bytes '
            f'({self.size_bytes / 1e9:.2f} GB)'
            + (f', {self.n_stars:,} stars.' if self.n_stars else '.'),
            f'It needs roughly {disk / 1e9:.1f} GB of free disk while it installs '
            f'(the archive and its unpacked copy exist at the same time), and on a '
            f'typical connection it takes tens of minutes.',
        ]
        if default is not None and default.name != self.name:
            lines.append(
                f'Most work does not need it: {default.name} '
                f'({default.human_size()}, G < {default.magnitude_limit:g}) is the '
                f'recommended archive and is enough for ordinary plate solving and '
                f'distortion fitting. Choose this one only if you specifically need '
                f'stars fainter than G = {default.magnitude_limit:g}.')
        return ' '.join(lines)

    @property
    def recommended(self):
        return self.name in RECOMMENDED_SETUP

    @property
    def superseded(self):
        """Replaced by a later archive, and no longer part of the user's choices."""
        return self.role in ('legacy', 'extension')

    @property
    def offered(self):
        """Should the app offer this as something to download?

        Superseded archives stay in ``RELEASES`` -- an existing install must keep
        verifying, and ``--merge`` needs to know about the pair it merges from -- but
        listing them beside the archive that replaced them only invites someone to
        download 327 MB of the same stars twice, or worse, to install the extension
        alone and end up with a catalogue containing nothing brighter than G=12.
        """
        return not self.superseded

    @property
    def shown_in_ui(self):
        """Should this appear in the interface at all -- even already installed?

        It should not. Showing every archive anyone ever installed turns a three-way
        choice into a six-way one, and the extra options are all worse than the ones
        beside them: g12 and the 12<G<13 extension are exactly the stars g13 already
        holds. They keep working if a config names one, and `--remove` still finds
        them; they are simply not offered as a choice any more.
        """
        return not self.superseded

    @property
    def is_published(self):
        return self.url is not None

    def directory(self):
        """Where an installed copy lives, and where a download writes."""
        return get_catalogue_root() / self.name

    def bundled_directory(self):
        """Where a copy shipped inside the program lives, if one was.

        The executable bundles the compact archive so it solves offline out of the box
        (MEE2024.spec). A downloaded archive in the user's data directory always wins,
        since it is the deeper one.
        """
        from mee2024.MEE2024util import resource_path
        return Path(resource_path(f'resources/catalogues/{self.name}'))

    def read_directory(self):
        """The copy to read: the installed one if present, else a bundled one."""
        for candidate in (self.directory(), self.bundled_directory()):
            try:
                store.read_manifest(candidate)
                return candidate
            except Exception:
                continue
        return self.directory()

    def is_bundled(self):
        try:
            store.read_manifest(self.bundled_directory())
            return True
        except Exception:
            return False

    def is_installed(self):
        """Is this archive readable -- from the user's data directory or bundled?"""
        for candidate in (self.directory(), self.bundled_directory()):
            try:
                store.read_manifest(candidate)
                return True
            except (FileNotFoundError, ValueError, OSError):
                continue
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

# One archive is the standard: **gaia_dr3_g13**, G < 13, everything an ordinary run
# needs (the default max_star_mag_dist is 12) and deep enough for eclipse fields. The
# original G<12 + 12<G<13 pair remains supported for anyone who already downloaded it,
# but splitting the standard depth across two disjoint slices cost more confusion than
# the 138 MB it saved -- and installing only the extension yields a catalogue containing
# nothing brighter than G=12, which is a genuine footgun. `mee2024 catalogue --merge`
# turns an installed pair into the single archive.
#
# The other two tiers exist for the ends of the range rather than to subdivide the
# middle: a compact archive small enough to sit beside the program, and a very deep one
# for work that genuinely needs it.
GITHUB_REPO = 'andrew551/MEE2024'
#: the release tag holding the catalogue assets, kept separate from software releases
CATALOGUE_TAG = 'catalogues-v1'


def _github_asset(filename):
    return (f'https://github.com/{GITHUB_REPO}/releases/download/'
            f'{CATALOGUE_TAG}/{filename}')


RELEASES = {
    'gaia_dr3_g13': CatalogueRelease(
        name='gaia_dr3_g13',
        description='Gaia DR3, G < 13 -- the standard archive',
        magnitude_limit=13.0,
        n_stars=7_369_627,
        size_bytes=319_719_061,
        doi=None,               # a Zenodo DOI replaces the GitHub URL at publication
        url=_github_asset('gaia_dr3_g13.zip'),
        sha256='897e6bc2ef32a4faf04c9294a48dde3318fd43edc6bd2041581a2cffc66453f0',
    ),
    'gaia_dr3_g10': CatalogueRelease(
        name='gaia_dr3_g10',
        description='Gaia DR3, G < 10 -- compact, for bright-star work',
        magnitude_limit=10.0,
        n_stars=None,
        size_bytes=None,
        doi=None,
        url=None,
        sha256=None,
        role='compact',
    ),
    'gaia_dr3_g15': CatalogueRelease(
        name='gaia_dr3_g15',
        description='Gaia DR3, G < 15 -- very deep, several GB, for special needs',
        magnitude_limit=15.0,
        n_stars=36_909_335,
        size_bytes=1_566_363_365,
        installed_bytes=1_919_290_665,     # measured, not estimated
        doi=None,               # a Zenodo DOI replaces the GitHub URL at publication
        url=_github_asset('gaia_dr3_g15.zip'),
        sha256='fb0711a6cf4e084129b401c493412fbef31ab1af74acca8c21dfc3307f35b29b',
        role='deep',
    ),
    # The original pair. Kept downloadable so an existing install keeps working and so
    # `--merge` has something to merge; superseded by gaia_dr3_g13.
    'gaia_dr3_g12': CatalogueRelease(
        name='gaia_dr3_g12',
        description='Gaia DR3, G < 12 (superseded by gaia_dr3_g13)',
        magnitude_limit=12.0,
        n_stars=3_087_821,
        size_bytes=137_952_319,
        doi=None,
        url=_github_asset('gaia_dr3_g12.zip'),
        sha256='f4a579e369c41b6d7099bac6b20d58c69f6b750092cd62f4085c77c670fbc5cb',
        role='legacy',
    ),
    'gaia_dr3_g12_13': CatalogueRelease(
        name='gaia_dr3_g12_13',
        description='Gaia DR3, 12 < G < 13 extension (superseded by gaia_dr3_g13)',
        magnitude_limit=13.0,
        n_stars=4_281_806,
        size_bytes=188_640_212,
        doi=None,
        url=_github_asset('gaia_dr3_g12_13.zip'),
        sha256='28607c07a9f60c89f09ba653eac06f890ff89631d28252e9af7af63be1adb71b',
        role='extension',
    ),
}

DEFAULT_RELEASE = 'gaia_dr3_g13'

#: what a fresh install should end up with
RECOMMENDED_SETUP = ('gaia_dr3_g13',)

#: Fetch order for a first-use download. Now that the standard archive is published this
#: is a list of one: falling back to the superseded G<12 base would quietly hand someone
#: a shallower catalogue than the one they were told they were getting, which is worse
#: than saying the download failed.
_FETCH_ORDER = ('gaia_dr3_g13',)

#: Catalogue names that *require* an offline archive: selecting one must trigger a
#: first-use download, and without an archive there is nothing to fall back on.
OFFLINE_CATALOGUES = ('gaia_offline', 'merged_offline')

#: Catalogue names that *prefer* an offline archive but can fall back to the online
#: archive. Declining the download is a warning, not an error -- the run still works,
#: just at minutes per field instead of milliseconds.
PREFER_OFFLINE_CATALOGUES = ('gaia',)


def preferred_release(options=None):
    """The archive a first-use download should fetch.

    The standard archive if it is installed or published, else the best published
    alternative, so this stays useful before gaia_dr3_g13 is uploaded.
    """
    for name in _FETCH_ORDER:
        release = get_release(name, options=options)
        if release.is_installed() or release.is_published:
            return name
    return DEFAULT_RELEASE


def releases_needed(catalogue, options=None):
    """Which archives must be downloaded before ``catalogue`` can be used.

    Returns [] when nothing is required -- an online catalogue, a bundled one, or an
    offline one whose archives are already present.
    """
    if catalogue in OFFLINE_CATALOGUES or catalogue in PREFER_OFFLINE_CATALOGUES:
        if installed_catalogues():
            return []
        return [preferred_release(options=options)]
    if catalogue in RELEASES:
        return [] if get_release(catalogue, options=options).is_installed() else [catalogue]
    return []


def prepare_catalogue(catalogue, options=None, progress_for=None, allow_download=True,
                      on_note=None):
    """Make ``catalogue`` usable before a long run starts, and report what is wrong.

    Two failures are worth catching up front. An offline catalogue that was never
    downloaded otherwise fails at stage 2, minutes in and with the stacking already
    done; and a magnitude limit deeper than the catalogue reaches silently returns fewer
    stars than asked for, which reads as a poor field rather than a catalogue that does
    not go that deep.

    ``progress_for(name)`` supplies a progress reporter per archive, and ``on_note(text)``
    receives running commentary -- both injected so this stays free of any opinion about
    how the caller displays things. Returns the list of warnings raised.
    """
    options = options or {}
    note = on_note or (lambda text: None)
    warnings = []
    needed = releases_needed(catalogue, options=options)
    # 'gaia' prefers an offline archive but can still run against the online one, so a
    # missing or declined download is a warning; a strictly offline catalogue is an error
    soft = catalogue in PREFER_OFFLINE_CATALOGUES
    if needed and not allow_download:
        message = (f'{catalogue} needs the {needed[0]} archive, which is not installed, '
                   f'and automatic downloading is switched off. Install it with '
                   f'`mee2024 catalogue --fetch {needed[0]}`.')
        if not soft:
            raise RuntimeError(message)
        warnings.append(message + ' Using the online Gaia archive instead, which takes '
                                  'minutes per field.')
        needed = []
    for name in needed:
        release = get_release(name, options=options)
        # a run must never start a multi-gigabyte download on the user's behalf: say what
        # it would cost and carry on without it, rather than asking mid-run or just going
        if release.needs_confirmation:
            message = (f'{catalogue} would need the {name} archive, and it is too large '
                       f'to fetch automatically. {release.size_warning()} '
                       f'Install it deliberately with `mee2024 catalogue --fetch {name}` '
                       f'if you do want it.')
            if not soft:
                raise RuntimeError(message)
            warnings.append(message)
            continue
        note(f'{catalogue} needs the {name} archive ({release.human_size()}); '
             f'downloading it now')
        try:
            ensure_available(name, options=options,
                             progress=progress_for(name) if progress_for else None)
        except Exception as exc:
            if not soft:
                raise
            warnings.append(
                f'could not install {name} ({exc}). Using the online Gaia archive '
                f'instead, which takes minutes per field.')
            break
        note(f'{name} installed')

    warning = magnitude_warning(catalogue, options.get('max_star_mag_dist'),
                                effective_magnitude_limit(catalogue, options=options))
    if warning:
        warnings.append(warning)
    return warnings


def magnitude_warning(catalogue, requested, limit):
    """The warning for a run asking for stars deeper than the catalogue reaches, or None.

    Asking for magnitude 14 from a G<13 archive is not an error -- the lookup simply
    truncates -- but silently returning fewer stars looks like a poor field rather than a
    catalogue that does not go that deep, so it is worth saying out loud.
    """
    if limit is None or requested is None or requested <= limit:
        return None
    if limit < 13:
        # Telling someone to install what they already have wastes their time and makes
        # the rest of the message look wrong too. If the deeper archive is on the machine,
        # the problem is the *choice* of catalogue, so say that instead.
        recommended = RELEASES.get(RECOMMENDED_SETUP[0])
        if recommended is not None and recommended.is_installed():
            advice = (f'{RECOMMENDED_SETUP[0]} is already installed -- choose it, or '
                      f'"gaia", as the star catalogue to reach G<13.')
        else:
            advice = f'Install {RECOMMENDED_SETUP[0]} to reach G<13.'
    else:
        advice = ('Deeper needs the gaia_dr3_g15 archive, or the online '
                  '"gaia_online" catalogue.')
    return (f'{catalogue} only contains stars to G<{limit:g}, but stars to magnitude '
            f'{requested:g} were requested: nothing fainter than G={limit:g} can be '
            f'matched. {advice}')


def effective_magnitude_limit(catalogue, options=None):
    """How deep ``catalogue`` actually reaches, or None if it is effectively unlimited.

    Used to warn before a run that asks for stars the catalogue cannot contain. Reports
    the depth of what is *installed*, not what could be installed, since that is what the
    run will really see.
    """
    if catalogue in OFFLINE_CATALOGUES or catalogue in PREFER_OFFLINE_CATALOGUES:
        limits = [r.magnitude_limit for r in installed_catalogues() if r.magnitude_limit]
        return max(limits) if limits else None
    if catalogue in RELEASES:
        return get_release(catalogue, options=options).magnitude_limit
    return None


def get_release(name=DEFAULT_RELEASE, options=None):
    """A release, with any URL and checksum from the user's config applied.

    The registry ships working URLs, but a user may want a different host -- a mirror, a
    local file server, or Zenodo ahead of a software release. Rather than require a code
    edit, ``catalogue_sources`` in the config overrides them:

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
                     options=None, confirm=False):
    """Return the local directory for a catalogue, downloading it if necessary.

    ``confirm`` gates downloads over :data:`LARGE_DOWNLOAD_BYTES`: pass True once the
    user has agreed, or a callable taking the release and returning a bool to be asked
    only if it turns out to be needed. Left False, a large download raises
    :class:`ConfirmationRequired` rather than starting -- so no code path, including an
    automatic one during a run, can commit someone to gigabytes without being asked.
    """
    release = get_release(name, options=options)
    directory = release.directory()
    if release.is_installed():
        return directory

    if release.needs_confirmation and release.is_published:
        agreed = confirm(release) if callable(confirm) else bool(confirm)
        if not agreed:
            raise ConfirmationRequired(release, release.size_warning())

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
            # a placeholder for a tier that is built locally is not a broken remote:
            # report it as skipped so a check over the whole registry still passes
            result['detail'] = 'no URL configured (built locally)'
            result['skipped'] = True
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


def remove(name, options=None):
    """Delete an installed catalogue and return the freed path and bytes.

    Releases the process's own cached, memory-mapped copy first: the app reads archives
    through a mmap, and Windows refuses to delete a mapped file, so without this the
    only way to reclaim the disk was to close the program. A bundled archive is not
    removable -- it lives inside the executable -- and says so rather than appearing to
    succeed while leaving the catalogue readable.
    """
    from mee2024 import database_cache

    release = get_release(name, options=options)
    directory = release.directory()
    if not directory.exists():
        if release.is_bundled():
            raise RuntimeError(
                f'{name} is bundled inside the program, not installed separately, '
                f'so there is nothing to remove and no disk to reclaim.')
        raise RuntimeError(f'{name} is not installed at {directory}')

    freed = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    database_cache.release_catalogues()
    try:
        shutil.rmtree(directory)
    except PermissionError as exc:
        raise RuntimeError(
            f'cannot remove {directory}: {exc.strerror or exc}. Something still has '
            f'the catalogue open -- close any other MEE window and try again.'
        ) from exc
    return {'name': name, 'path': str(directory), 'freed_bytes': freed}


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


def _recompute_neighbour_flags(table, radius_arcsec=10.0):
    """Nearest-neighbour separation and magnitude, over this table's own members.

    Worth redoing after a merge: flags computed inside a G<12 archive cannot know
    about a G=12.5 companion, so merging the pair and recomputing genuinely improves
    double-star flagging rather than only reshaping the files.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    vectors = table.get_vectors()
    distance, index = cKDTree(vectors).query(vectors, k=2, workers=-1)
    chord = distance[:, 1]
    table.nn_sep = np.degrees(2 * np.arcsin(np.clip(chord / 2, 0, 1))).astype(
        np.float32) * 3600
    table.nn_mag = table.mag[index[:, 1]].astype(np.float32)
    return table


def broken_catalogues():
    """Directories holding catalogue columns but no readable manifest.

    A half-finished download or unpack leaves exactly this, and every 'is it
    installed?' check answers no -- so the archive silently vanishes from the set a
    run or a merge sees. Worth reporting rather than ignoring.
    """
    root = get_catalogue_root()
    broken = []
    for directory in sorted(root.iterdir()) if root.is_dir() else []:
        if not directory.is_dir():
            continue
        try:
            store.read_manifest(directory)
        except Exception:
            if any(directory.glob('*.npy')):
                broken.append(directory.name)
    return broken


def build_compact_tier(name='gaia_dr3_g10', source=None, options=None, on_note=None):
    """Filter a deeper installed archive down to the compact tier.

    The compact archive is what the executable bundles (MEE2024.spec), so that a fresh
    install solves plates offline immediately instead of querying the online archive at
    minutes per field. Deriving it locally keeps 24 MB of generated data out of source
    control.
    """
    note = on_note or (lambda text: None)
    limit = RELEASES[name].magnitude_limit
    candidates = ([source] if source else
                  [r.name for r in installed_catalogues()
                   if r.name != name and (r.magnitude_limit or 0) > limit])
    for candidate in candidates:
        directory = get_release(candidate, options=options).read_directory()
        try:
            catalogue = store.OfflineCatalogue(directory)
        except Exception:
            continue
        note(f'filtering {candidate} to G<{limit}')
        table = catalogue.lookup((0.0, 360.0), (-90.0, 90.0),
                                 max_magnitude=limit, epoch=None)
        out = get_catalogue_root() / name
        manifest = store.write_catalogue(
            out, table, name=name, magnitude_limit=limit,
            provenance=f'filtered from {candidate} by build_compact_tier')
        write_local_release_stub(out, name)
        note(f'{name}: {manifest["n_stars"]} stars written to {out}')
        return out, manifest
    raise RuntimeError(
        f'no installed archive deeper than G<{limit} to filter. Install one first, '
        f'e.g. `mee2024 catalogue --fetch gaia_dr3_g12`.')


def repair_catalogue(name, options=None, on_note=None):
    """Rebuild a half-installed archive's manifest in place, or explain why not.

    Band, epoch and intended depth cannot be read off the data, so they come from the
    registry entry and from any sibling archive already installed -- Gaia DR3's
    reference epoch is the same for every slice of it.
    """
    note = on_note or (lambda text: None)
    release = get_release(name, options=options)
    directory = release.directory()
    if not directory.is_dir():
        raise RuntimeError(f'{name} is not installed at {directory}')
    try:
        store.read_manifest(directory)
    except Exception:
        pass
    else:
        note(f'{name} already has a valid manifest; nothing to repair')
        return directory, store.read_manifest(directory)

    epoch, band, source = None, 'G', None
    for other in installed_catalogues():
        if other.name == name:
            continue
        try:
            manifest = store.read_manifest(other.directory())
        except Exception:
            continue
        epoch, band, source = manifest['epoch'], manifest['band'], other.name
        break
    if epoch is None:
        epoch, band, source = 2016.0, 'G', 'the Gaia DR3 reference epoch'
    note(f'taking epoch {epoch} and band {band} from {source}')

    manifest = store.rebuild_manifest(
        directory, name=name, band=band, epoch=epoch,
        magnitude_limit=release.magnitude_limit,
        provenance=f'manifest rebuilt by `mee2024 catalogue --repair {name}` after an '
                   f'interrupted install; epoch and band from {source}')
    note(f'{name}: manifest rebuilt, {manifest["n_stars"]} stars to '
         f'G<{manifest["magnitude_limit"]}')
    return directory, manifest


def merge_installed(name=DEFAULT_RELEASE, sources=None, options=None, progress=None,
                    on_note=None, force=False):
    """Merge installed Gaia archives into one, and return (directory, manifest).

    This is how a machine holding the original G<12 + 12<G<13 pair reaches the single
    standard archive without downloading anything: the union is deduplicated by Gaia
    source id and its neighbour flags are recomputed over the whole set.
    """
    import numpy as np
    from mee2024.starcat import store
    from mee2024.starcat.table import concat

    note = on_note or (lambda text: None)
    names = list(sources) if sources else [r.name for r in installed_catalogues()
                                          if r.name != name]
    # A half-installed archive reports as absent, so merging around it would quietly
    # produce a catalogue missing everything that archive held. Refuse instead.
    damaged = [n for n in broken_catalogues() if n != name and n not in names]
    if damaged and not force:
        raise RuntimeError(
            f'{", ".join(damaged)} looks half-installed (data files but no '
            f'manifest.json), so it cannot be read and merging without it would '
            f'silently drop its stars. Reinstall it with '
            f'`mee2024 catalogue --fetch {damaged[0]}`, or pass --force to merge '
            f'only what is readable.')
    directories = [get_release(n, options=options).read_directory() for n in names]
    directories = [d for d in directories if d.is_dir()]
    if not directories:
        raise RuntimeError(
            'no installed catalogue archives to merge. Fetch one first, e.g. '
            '`mee2024 catalogue --fetch gaia_dr3_g12`.')

    tables, limits = [], []
    for directory in directories:
        catalogue = store.OfflineCatalogue(directory)
        note(f'reading {catalogue.manifest["name"]} ({len(catalogue)} stars)')
        tables.append(catalogue.lookup((0.0, 360.0), (-90.0, 90.0),
                                       max_magnitude=None, epoch=None))
        if catalogue.magnitude_limit:
            limits.append(catalogue.magnitude_limit)

    base_epoch = tables[0].epoch
    tables = [t if t.epoch == base_epoch else t.at_epoch(base_epoch) for t in tables]
    table = tables[0] if len(tables) == 1 else concat(tables)

    _, unique_index = np.unique(table.ids, return_index=True)
    if len(unique_index) < len(table):
        note(f'{len(table) - len(unique_index)} duplicate source ids dropped')
        table = table.select(np.sort(unique_index))

    # A standard archive must actually contain bright stars. Merging extensions alone
    # would otherwise write a catalogue whose recorded depth (G<13) is true of its
    # faint end and a lie about its bright end -- the exact footgun `role` exists for.
    brightest = float(np.min(table.mag)) if len(table) else float('inf')
    if brightest > 8.0 and not force:
        raise RuntimeError(
            f'the merged result contains nothing brighter than G={brightest:.1f}, so '
            f'it is an extension rather than a standard archive. Check that the base '
            f'archive is installed and readable (`mee2024 catalogue --status`), or '
            f'pass --force if a faint-only catalogue is really what you want.')

    note(f'recomputing neighbour flags over {len(table)} stars')
    table = _recompute_neighbour_flags(table)

    directory = get_catalogue_root() / name
    note(f'writing {name} to {directory}')
    manifest = store.write_catalogue(
        directory, table, name=name,
        magnitude_limit=max(limits) if limits else None,
        provenance=f'merged from {", ".join(names)} by mee2024 catalogue --merge; '
                   f'neighbour flags recomputed over the union')
    write_local_release_stub(directory, name)
    return directory, manifest


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
