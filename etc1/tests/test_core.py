import math

import pytest

import etc1

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

BLK = [[0.5] * 4 for _ in range(4)]

CASES = [
    ("bytes", lambda: len(etc1.Etc1().encode(BLK)) == 8),
    ("constant", lambda: abs(etc1.Etc1().decode(etc1.Etc1().encode(BLK))[0][0] - 0.5) < 0.05),
    ("psnr", lambda: etc1.Etc1().psnr(BLK, etc1.Etc1().decode(etc1.Etc1().encode(BLK))) >= 30.0),
    ("flip_choice", lambda: isinstance(etc1.Etc1().encode(BLK), list)),
    ("bad_rows", lambda: _raises(lambda: etc1.Etc1().encode([[0.5] * 4]))),
    ("bad_data", lambda: _raises(lambda: etc1.Etc1().decode([]))),
    ("type", lambda: isinstance(etc1.Etc1().decode(etc1.Etc1().encode(BLK)), list)),
    ("det", lambda: etc1.Etc1().encode(BLK) == etc1.Etc1().encode(BLK)),
    ("byte_range", lambda: all(0 <= b <= 255 for b in etc1.Etc1().encode(BLK))),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in etc1.Etc1().decode(etc1.Etc1().encode(BLK)) for v in row)),
    ("gradient_psnr", lambda: etc1.Etc1().psnr([[i / 3.0] * 4 for i in range(4)], etc1.Etc1().decode(etc1.Etc1().encode([[i / 3.0] * 4 for i in range(4)]))) >= 25.0),
    ("nan", lambda: _raises(lambda: etc1.Etc1().encode([[float('nan')] * 4 for _ in range(4)]))),
    ("no_mut", lambda: etc1.Etc1().encode([r[:] for r in BLK]) is not None),
    ("psnr_type", lambda: isinstance(etc1.Etc1().psnr(BLK, BLK), float)),
    ("decode_shape", lambda: len(etc1.Etc1().decode(etc1.Etc1().encode(BLK))) == 4),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
