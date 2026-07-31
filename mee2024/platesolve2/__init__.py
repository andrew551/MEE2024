"""
Plate solver v2: Gaia pattern database + Kendall shape-space triangle matching.

Selected with ``options['platesolver'] = 'v2'``; the production solver
(`mee2024.platesolve_triangle`) stays the default until v2 dominates the A/B bench
(``tools/solver_bench.py``). Design and stage plan: ``docs/PLATESOLVER_V2_DESIGN.md``.

``platesolve()`` here honours the exact contract of
``platesolve_triangle.platesolve`` -- same signature, same return-dict keys, same
``SOLVE_CANDIDATE``/``SOLVE_RESULT`` events -- so the pipeline call sites never change.
"""


def platesolve(centroids, image_shape, options=None, output_dir=None,
               try_mirror_also=True):
    raise NotImplementedError(
        "the v2 plate solver is not built yet (stage S1 of "
        "docs/PLATESOLVER_V2_DESIGN.md). Set platesolver='triangle' to use the "
        "production solver.")
