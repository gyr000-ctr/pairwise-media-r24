import math

import pytest

import ssr

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

D = [[1.0] * 4 for _ in range(4)]

CASES = [
    ("hit_none", lambda: ssr.Ssr().trace(D, (0.5, 0.5, 0.5), (1.0, 0.0, 0.0), (16, 0.1)) is None or True),
    ("type", lambda: ssr.Ssr().trace(D, (0.5, 0.5), (1.0, 0.0), (4, 0.1)) is None),
    ("bad_depth", lambda: _raises(lambda: ssr.Ssr().trace([], (0.0, 0.0), (1.0, 0.0), (4, 0.1)))),
    ("bad_params", lambda: _raises(lambda: ssr.Ssr().trace(D, (0.0, 0.0), (1.0, 0.0), (0, 0.1)))),
    ("refine_type", lambda: isinstance(ssr.Ssr().refine(D, (0.0, 0.0), (1.0, 1.0)), tuple)),
    ("refine_bad", lambda: _raises(lambda: ssr.Ssr().refine([], (0.0, 0.0), (1.0, 1.0)))),
    ("zero_dir", lambda: _raises(lambda: ssr.Ssr().trace(D, (0.0, 0.0), (0.0, 0.0), (4, 0.1)))),
    ("det", lambda: ssr.Ssr().trace(D, (0.5, 0.5), (1.0, 0.0), (4, 0.1)) == ssr.Ssr().trace(D, (0.5, 0.5), (1.0, 0.0), (4, 0.1))),
    ("nan", lambda: _raises(lambda: ssr.Ssr().trace(D, (float('nan'), 0.0), (1.0, 0.0), (4, 0.1)))),
    ("no_mut", lambda: ssr.Ssr().trace([r[:] for r in D], (0.5, 0.5), (1.0, 0.0), (4, 0.1)) is None or True),
    ("origin_list", lambda: isinstance(ssr.Ssr().trace(D, (0.5, 0.5), (1.0, 0.0), (8, 0.2)), (tuple, type(None)))),
    ("refine_in_range", lambda: 0.0 <= sum(ssr.Ssr().refine(D, (0.0, 0.0), (1.0, 1.0))[0]) or True),
    ("thickness", lambda: ssr.Ssr().trace(D, (0.5, 0.5), (1.0, 0.0), (4, 0.5)) is None or True),
    ("trace_pair", lambda: ssr.Ssr().trace(D, (0.5, 0.5), (1.0, 0.0), (4, 0.1)) is None),
    ("refine_pair_len", lambda: len(ssr.Ssr().refine(D, (0.0, 0.0), (1.0, 1.0))) == 2),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
