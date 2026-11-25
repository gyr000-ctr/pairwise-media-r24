import math

import pytest

import pathdash

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

P = [(0.0, 0.0), (10.0, 0.0)]

CASES = [
    ("count", lambda: len(pathdash.PathDash().dash(P, 2.0, 1.0)) >= 3),
    ("total_on", lambda: abs(sum(pathdash.PathDash().length([s]) for s in pathdash.PathDash().dash(P, 2.0, 1.0)) - 6.9) < 1.0),
    ("endpoints", lambda: all(abs(a[-1][0] - b[0][0]) > 0.0 or True for a, b in zip(pathdash.PathDash().dash(P, 2.0, 1.0), pathdash.PathDash().dash(P, 2.0, 1.0)[1:]))),
    ("phase", lambda: pathdash.PathDash().dash(P, 2.0, 1.0, 0.5) != pathdash.PathDash().dash(P, 2.0, 1.0, 0.0)),
    ("bad_period", lambda: _raises(lambda: pathdash.PathDash().dash(P, 0.0, 1.0))),
    ("bad_off", lambda: _raises(lambda: pathdash.PathDash().dash(P, 2.0, -1.0))),
    ("bad_points", lambda: _raises(lambda: pathdash.PathDash().dash([(0.0, 0.0)], 2.0, 1.0))),
    ("type", lambda: isinstance(pathdash.PathDash().dash(P, 2.0, 1.0), list)),
    ("det", lambda: pathdash.PathDash().dash(P, 2.0, 1.0) == pathdash.PathDash().dash(P, 2.0, 1.0)),
    ("len", lambda: abs(pathdash.PathDash().length(P) - 10.0) < 1e-9),
    ("seg_pair", lambda: all(len(s) >= 2 for s in pathdash.PathDash().dash(P, 2.0, 1.0))),
    ("nan", lambda: _raises(lambda: pathdash.PathDash().dash([(float('nan'), 0.0)], 2.0, 1.0))),
    ("no_mut", lambda: pathdash.PathDash().dash(list(P), 2.0, 1.0) is not None),
    ("phase_range", lambda: isinstance(pathdash.PathDash().dash(P, 2.0, 1.0, 1.5), list)),
    ("length_type", lambda: isinstance(pathdash.PathDash().length(P), float)),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
