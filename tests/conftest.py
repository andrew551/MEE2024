"""Shared fixtures and helpers for the MEE2024 test suite."""

import matplotlib
matplotlib.use('Agg')  # no test may ever open a window

import numpy as np
import pytest

from mee2024.config import get_default_options


def pytest_addoption(parser):
    parser.addoption('--runslow', action='store_true', default=False,
                     help='run tests that need the triangle database or the network')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--runslow'):
        return
    skip = pytest.mark.skip(reason='needs --runslow')
    for item in items:
        if 'slow' in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def options():
    """Default options, with every display flag off."""
    opts = get_default_options()
    opts['flag_display'] = False
    opts['flag_display2'] = False
    opts['flag_display3'] = False
    opts['flag_debug'] = False
    return opts


def gaussian_star_field(shape=(256, 256), positions=(), fluxes=None, sigma=1.6,
                        background=100.0, noise=0.0, seed=0):
    """A synthetic image with Gaussian stars at exactly known sub-pixel positions.

    positions: sequence of (y, x) in pixel-centre coordinates, i.e. the centre of
    pixel [i, j] is (i, j).
    """
    rng = np.random.default_rng(seed)
    img = np.full(shape, background, dtype=np.float64)
    if fluxes is None:
        fluxes = [1000.0] * len(positions)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for (cy, cx), flux in zip(positions, fluxes):
        r2 = (yy - cy) ** 2 + (xx - cx) ** 2
        img += flux / (2 * np.pi * sigma ** 2) * np.exp(-r2 / (2 * sigma ** 2))
    if noise:
        img += rng.normal(0.0, noise, size=shape)
    return img


@pytest.fixture
def star_positions():
    """Well-separated stars at deliberately non-integer positions."""
    return [(40.3, 60.7), (100.5, 30.25), (150.75, 180.4), (200.1, 90.9), (60.6, 200.2)]
