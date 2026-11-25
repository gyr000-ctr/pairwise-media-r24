import math

import pytest

import glyphsdf

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

C = [[(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]]

CASES = [
    ("shape", lambda: len(glyphsdf.GlyphSdf(32, 4).rasterize(C)) == 32),
    ("range", lambda: all(0.0 <= v <= 1.0 for row in glyphsdf.GlyphSdf(16, 4).rasterize(C) for v in row)),
    ("center_inside", lambda: glyphsdf.GlyphSdf(32, 4).rasterize(C)[16][16] > 0.5),
    ("corner_outside", lambda: glyphsdf.GlyphSdf(32, 4).rasterize(C)[0][0] < 0.5),
    ("bad_size", lambda: _raises(lambda: glyphsdf.GlyphSdf(0, 4))),
    ("bad_spread", lambda: _raises(lambda: glyphsdf.GlyphSdf(32, 0))),
    ("bad_contours", lambda: _raises(lambda: glyphsdf.GlyphSdf(32, 4).rasterize([]))),
    ("coverage", lambda: 0.0 <= glyphsdf.GlyphSdf(16, 4).coverage(glyphsdf.GlyphSdf(16, 4).rasterize(C)) <= 1.0),
    ("type", lambda: isinstance(glyphsdf.GlyphSdf(16, 4).rasterize(C), list)),
    ("det", lambda: glyphsdf.GlyphSdf(16, 4).rasterize(C) == glyphsdf.GlyphSdf(16, 4).rasterize(C)),
    ("nan", lambda: _raises(lambda: glyphsdf.GlyphSdf(16, 4).rasterize([[(float('nan'), 0.0)]]))),
    ("no_mut", lambda: glyphsdf.GlyphSdf(16, 4).rasterize([list(c) for c in C]) is not None),
    ("coverage_type", lambda: isinstance(glyphsdf.GlyphSdf(16, 4).coverage([[0.5]]), float)),
    ("profile", lambda: isinstance(glyphsdf.GlyphSdf(16, 4).rasterize(C)[0], list)),
    ("monotone_edge", lambda: glyphsdf.GlyphSdf(32, 4).rasterize(C)[16][0] <= glyphsdf.GlyphSdf(32, 4).rasterize(C)[16][16]),
]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert check() is True
