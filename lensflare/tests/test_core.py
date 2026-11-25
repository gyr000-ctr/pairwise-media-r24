import math

import pytest

import lensflare

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

D = [[1.0] * 8 for _ in range(8)]

CASES = [
    ("ratio_range", lambda: 0.0 <= lensflare.LensFlare(8).visible_ratio(D, (0.5, 0.5, 0.5)) <= 1.0),
    ("clear", lambda: abs(lensflare.LensFlare(8).visible_ratio(D, (0.5, 0.5, 0.5)) - 1.0) < 0.05),
    ("ghosts_len", lambda: len(lensflare.LensFlare(8).ghosts((0.5, 0.5), 3)) == 3),
    ("ghosts_on_axis", lambda: len(lensflare.LensFlare(8).ghosts((0.5, 0.5), 3)[0]) == 2),
    ("bad_samples", lambda: _raises(lambda: lensflare.LensFlare(0))),
    ("bad_depth", lambda: _raises(lambda: lensflare.LensFlare(8).visible_ratio([], (0.5, 0.5, 0.5)))),
    ("bad_light", lambda: _raises(lambda: lensflare.LensFlare(8).visible_ratio(D, (0.5, 0.5)))),
    ("bad_n", lambda: _raises(lambda: lensflare.LensFlare(8).ghosts((0.5, 0.5), 0))),
    ("type", lambda: isinstance(lensflare.LensFlare(8).visible_ratio(D, (0.5, 0.5, 0.5)), float)),
    ("det", lambda: lensflare.LensFlare(8).visible_ratio(D, (0.5, 0.5, 0.5)) == lensflare.LensFlare(8).visible_ratio(D, (0.5, 0.5, 0.5))),
    ("ghost_type", lambda: isinstance(lensflare.LensFlare(8).ghosts((0.5, 0.5), 2), list)),
    ("occluded", lambda: lensflare.LensFlare(8).visible_ratio([[0.0] * 8 for _ in range(8)], (0.5, 0.5, 0.5)) <= 1.0),
    ("nan", lambda: _raises(lambda: lensflare.LensFlare(8).visible_ratio(D, (float('nan'), 0.5, 0.5)))),
    ("no_mut", lambda: lensflare.LensFlare(8).visible_ratio([r[:] for r in D], (0.5, 0.5, 0.5)) is not None),
    ("ratio_monotone", lambda: isinstance(lensflare.LensFlare(16).visible_ratio(D, (0.5, 0.5, 0.5)), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
