"""
Plate solver v2: Gaia pattern database + (from S2) Kendall shape-space matching.

Selected with ``options['platesolver'] = 'v2'``; the production solver
(`mee2024.platesolve_triangle`) stays the default until v2 dominates the A/B bench
(``tools/solver_bench.py``). Design and stage plan: ``docs/PLATESOLVER_V2_DESIGN.md``.

``platesolve()`` honours the exact contract of ``platesolve_triangle.platesolve`` --
same signature, same return-dict keys, same ``SOLVE_CANDIDATE``/``SOLVE_RESULT``
events -- so the pipeline call sites never change. The one additive key is
``diagnostics`` (candidate counts, thresholds, timing) for the bench.
"""

import numpy as np

from mee2024 import events


def preflight(options=None):
    """Can a v2 solve actually run? Returns (ok, reason).

    The v1.1.0 default is 'v2' with automatic fallback: a fresh install has
    neither the pattern database nor the offline catalogue, and must keep solving
    (via the classic Tycho solver) rather than erroring several stages deep.
    """
    from mee2024.platesolve2 import pattern_db, verify
    try:
        dbs = pattern_db.resolve_layers(options or {})
        verify.open_verify_catalogue(dbs[0].manifest.get('verify') or {})
        return True, ''
    except Exception as exc:
        return False, str(exc)


#: How to build a pattern database when none is installed, by the depth of the star
#: catalogue available. The compact catalogue gets a smaller, faster database: fewer
#: legs per anchor means fewer triangles, which is both quicker to build (seconds) and
#: quicker to query, at some cost in coverage. A deeper catalogue earns the full one.
FIRST_BUILD = (
    (12.0, 'patdb_g13_t17k', {'theta_pat_deg': 1.7, 'e': 18, 'invariant': 'kendall'},
     'about three minutes'),
    (0.0, 'patdb_g10_t17k', {'theta_pat_deg': 1.7, 'e': 8, 'invariant': 'kendall'},
     'about twenty seconds'),
)


def ensure_pattern_db(options=None, progress=None, on_note=None):
    """Build a pattern database if none is installed. Returns True if v2 can solve.

    The database is derived from the star catalogue, so downloading it would mean
    shipping hundreds of megabytes of something the user's own machine can compute --
    and asking them to run a build command first is a worse first experience than
    simply doing it. Reports progress like any other pipeline stage, and never raises:
    if it cannot build, the caller falls back to the classic solver.
    """
    from mee2024.platesolve2 import build, pattern_db

    note = on_note or (lambda text: None)
    try:
        pattern_db.resolve_layers(options or {})
        return True
    except Exception:
        pass

    try:
        from mee2024.starcat import download
        installed = download.installed_catalogues()
        if not installed:
            note('no star catalogue is installed, so no pattern database can be built')
            return False
        deepest = max(installed, key=lambda r: r.magnitude_limit or 0)
        depth = deepest.magnitude_limit or 0
        for threshold, name, params, duration in FIRST_BUILD:
            if depth >= threshold:
                break
        note(f'building the plate-solving database from {deepest.name} '
             f'({duration}); this happens once')
        build.build_from_catalogue(name=name, catalogue_names=(deepest.name,),
                                  params=params, progress=progress)
        note(f'{name} built')
        return True
    except Exception as exc:
        note(f'could not build a pattern database: {exc}')
        return False


def platesolve(centroids, image_shape, options=None, output_dir=None,
               try_mirror_also=True, catalogue=None, db=None):
    """Lost-in-space solve of an (n, 2) brightest-first (y, x) centroid array.

    ``catalogue`` and ``db`` exist for tests and the bench: by default the pattern
    database is resolved from ``options['pattern_db']`` and the verification
    catalogue from that database's manifest.
    """
    from mee2024.platesolve2 import pattern_db, solve, verify

    options = options if options is not None else {}
    centroids = np.array(centroids)
    if not len(centroids.shape) == 2 or not centroids.shape[1] == 2:
        raise Exception("ERROR: expected an n by 2 array for centroids")

    if db is None:
        dbs = pattern_db.resolve_layers(options)
    else:
        dbs = db if isinstance(db, (list, tuple)) else [db]
    db = dbs[0]
    if catalogue is None:
        catalogue = verify.open_verify_catalogue(db.manifest.get('verify') or {})

    if db.invariant == pattern_db.INVARIANT_KENDALL:
        # mirror coverage is part of the single query pass (S2); additional FOV
        # layers extend blind coverage through the failure ladder (S6)
        result = solve.solve_kendall(dbs, catalogue, centroids, image_shape,
                                     options, output_dir=output_dir,
                                     try_mirror_also=try_mirror_also)
        _emit_solve_result(result)
        return result

    result = solve.solve_helper(db, catalogue, centroids, image_shape, options,
                                output_dir=output_dir)
    result['mirror'] = False
    if result['success'] or not try_mirror_also:
        _emit_solve_result(result)
        return result

    print('platesolve failed ... trying mirror image of field')
    mirrored = np.copy(centroids)
    mirrored[:, [0, 1]] = mirrored[:, [1, 0]]
    result = solve.solve_helper(db, catalogue, mirrored,
                                (image_shape[1], image_shape[0]), options,
                                output_dir=output_dir)
    result['mirror'] = False
    if result['success']:
        result['mirror'] = True
        result['matched_centroids'][:, [0, 1]] = result['matched_centroids'][:, [1, 0]]
    _emit_solve_result(result)
    return result


def _emit_solve_result(result):
    events.emit(events.SOLVE_RESULT, success=bool(result['success']),
                ra=result['ra'], dec=result['dec'], roll=result['roll'],
                platescale=result['platescale/arcsec'],
                mirror=bool(result.get('mirror')),
                n_matched=0 if result['matched_stars'] is None
                else int(len(result['matched_stars'])))
