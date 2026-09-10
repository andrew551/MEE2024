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
    'husillos2026': Window(
        13.0, 2.0, 10.0,
        "inherited from cell 2 and cited as inheritance, which is what the cell-4 handoff asked "
        "for: 'the likely choice is cell 2's G <= 13, 2-10 R_sun by inheritance, cited as such'. "
        "The outer bound is NOT yet decided on this cell's own data -- cell 2 moved it 9 -> 10 on "
        "a residual pile-up (docs/MATRIX_2026.md, 'the outer radius') and cell 4 should decide it "
        "the same way, on a zenith-against-eclipse annulus comparison, once there is more than "
        "one zenith field. Until then this is a borrowed window and every number fitted through "
        "it is preliminary (docs/HUSILLOS2026_ECLIPSE.md). Two facts about this frame that the "
        "window has to live with: the corners reach 13.3 R_sun, so 10 does bite; and the field "
        "only detects to about G 10.3, so the magnitude bound does not bite at all."),
}
