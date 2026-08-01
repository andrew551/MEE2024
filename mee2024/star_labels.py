"""
Labels for the identified stars the frontend draws over the stacked preview.

Both pipeline stages identify stars and both have something to say about them: the plate
solve knows which detections were matched at all, and the distortion fit -- working from a
deeper catalogue -- knows which of them it had to throw away as double stars. Each emits a
STARS event; the later, richer one supersedes the earlier in the frontend.

Positions travel as columns rather than as a drawn-on image so the frontend can decide how
many labels to show without another run: a wide field matches hundreds of stars, and which
of them are worth naming is a question about screen space, not about the data.
"""

import numpy as np

from mee2024 import events

#: label tiers the UI's slider steps through. A star qualifies for a tier if it qualifies
#: for every tier below it, so the slider only ever adds labels.
LABEL_TIERS = ('none', 'named', 'hip', 'bright', 'all')


def build_labels(magnitudes, ids=None, bright_limit=9.0):
    """A (label, tier) pair per star: a proper name if it has one, else HIP, else G mag."""
    labels = [''] * len(magnitudes)
    tiers = ['all'] * len(magnitudes)
    index = None
    if ids is not None:
        try:
            from mee2024.starcat.labels import LabelIndex
            index = LabelIndex.try_bundled()
        except Exception:
            index = None            # no bundled name list: magnitudes still label fine
    if index is not None:
        identifiers = np.asarray(ids, dtype=np.int64)
        names = index.name_for(identifiers)
        hips = index.hip_for(identifiers)
        for i, (name, hip) in enumerate(zip(names, hips)):
            if name:
                labels[i], tiers[i] = str(name), 'named'
            elif hip:
                labels[i], tiers[i] = f'HIP {int(hip)}', 'hip'
    for i, magnitude in enumerate(magnitudes):
        if labels[i]:
            continue
        labels[i] = f'G {magnitude:.1f}'
        tiers[i] = 'bright' if magnitude <= bright_limit else 'all'
    return labels, tiers


def emit(positions_yx, magnitudes, image_shape, ids=None, dropped=None,
         bright_limit=9.0, stage='stack'):
    """Emit a STARS event. ``dropped`` flags stars eliminated as double stars."""
    positions_yx = np.asarray(positions_yx, dtype=float)
    magnitudes = np.asarray(magnitudes, dtype=float)
    if not positions_yx.size:
        return None
    labels, tiers = build_labels(magnitudes, ids, bright_limit)
    if dropped is None:
        dropped = np.zeros(len(magnitudes), dtype=bool)
    return events.emit(events.STARS, stage=stage,
                       image_size=[int(image_shape[0]), int(image_shape[1])],
                       y=[float(v) for v in positions_yx[:, 0]],
                       x=[float(v) for v in positions_yx[:, 1]],
                       mag=[float(v) for v in magnitudes],
                       label=labels, tier=tiers,
                       dropped=[bool(v) for v in np.asarray(dropped)])


def emit_from_solution(solution, image_shape, bright_limit=9.0):
    """Emit the plate solve's matched stars (stage 1 knows of no eliminations yet)."""
    stars = solution.get('matched_stars')
    centroids = solution.get('matched_centroids')
    if stars is None or centroids is None or not len(stars):
        return None
    stars = np.asarray(stars)
    return emit(np.asarray(centroids), stars[:, 5], image_shape,
                ids=solution.get('matched_ids'), bright_limit=bright_limit)
