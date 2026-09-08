"""The admitted-star window for each matrix cell, with the document that fixes it.

A window is never chosen at the point of use. On 2026-09-08 Station 2's eclipse field was fitted
at 2-5.5 R_sun -- a bound that appears in no document and was not asked for; it silently dropped
three of the seventeen stars in each tier, all of them at 5.7-6.6 R_sun, and moved the answer.
The documented window is 2-10 R_sun at G <= 13 (`docs/MATRIX_2026.md`, cell 2 budget), and the
outer bound is itself a decision with a record: 9 -> 10 R_sun on Douglas' reading of the residual
pile-up (`docs/MATRIX_2026.md`, "the outer radius").

So the windows live here, each with its citation, and the tools import them. A new window is a
new entry with a source, which is a thing that cannot be invented by accident.
`tests/test_analysis_window.py` pins these values and refuses an uncited window literal in any
newly written tool.
"""
from typing import NamedTuple


class Window(NamedTuple):
    """mag: faintest G admitted. rmin, rmax: radial bounds in solar radii. source: where it says so."""
    mag: float
    rmin: float
    rmax: float
    source: str

    def admits(self, magV, rsun):
        """Boolean mask: the stars this window admits."""
        return (magV <= self.mag) & (rsun >= self.rmin) & (rsun <= self.rmax)


WINDOWS = {
    'mexico2024_station1': Window(
        13.0, 2.0, 10.0,
        "docs/MATRIX_2026.md, cell 2 budget: 'G <= 13, 2-10 R_sun, 639 observations of 192 "
        "stars'; the outer bound 9 -> 10 is recorded in the same file under 'the outer radius'"),
    'mexico2024_station2': Window(
        13.0, 2.0, 10.0,
        "the same window as Station 1: the same site, the same optic and the same convention, "
        "and Station 2's eclipse field is reduced by the Station 1 technique verbatim "
        "(docs/STEP3_2026.md, \"Station 2's eclipse field\")"),
}
