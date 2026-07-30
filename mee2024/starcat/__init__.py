"""
Star catalogue access for MEE2024.

One columnar star-data type (`StarTable`) and one provider interface behind which the
online Gaia archive, a downloaded offline Gaia catalogue, the bundled Tycho catalogue and
any merge of them are interchangeable.

See docs/STARCAT_DESIGN.md for the design and the measurements behind it.
"""

from mee2024.starcat.table import (
    ORIGIN_GAIA,
    ORIGIN_TYCHO,
    ORIGIN_NAMES,
    StarTable,
    concat,
)

__all__ = [
    'ORIGIN_GAIA',
    'ORIGIN_TYCHO',
    'ORIGIN_NAMES',
    'StarTable',
    'concat',
]
