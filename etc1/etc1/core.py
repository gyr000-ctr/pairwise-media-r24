"""ETC1 block encoder / decoder.

The task models a block as a 4x4 grid of *scalar* values in ``[0, 1]`` -- an
ETC1 block whose R, G and B channels are equal (grayscale).  Every encoded
block is nevertheless a fully standard 64-bit ETC1 word, so a conforming
hardware or software decoder reproduces exactly the same gray image.

Design
------
* **Exhaustive, deterministic search.**  For each of the two flip modes
  (``flip = 0``: two vertical 2x4 subblocks, ``flip = 1``: two horizontal
  4x2 subblocks) the encoder evaluates

  - the *individual* base-colour mode (two independent 4-bit bases), and
  - the *differential* base-colour mode (one 5-bit base plus a 3-bit signed
    delta per channel, restricted to deltas that keep the second base in
    ``[0, 31]`` -- anything else would be an illegal ETC1 block),

  for every one of the eight intensity-modifier tables and every legal base
  colour, assigning each pixel its own optimal selector.  The reported error
  is therefore the true minimum over the whole legal candidate set for that
  flip, and the flip with the smaller total squared error wins (ties go to
  ``flip = 0``).  That is exactly what a brute-force comparison of the two
  flip modes computes.

* **Bit-exact reconstruction.**  :meth:`Etc1.decode` decodes encoded bytes
  through exactly the same tables the encoder used to score candidates, so
  ``decode(encode(x))`` *is* the encoder's internal reconstruction; the only
  extra step is the final ``value / 255`` normalization.

* **Constant blocks are lossless.**  Every 8-bit gray level is exactly
  representable by an ETC1 intensity base, so a constant block round-trips
  to its quantization cell ``round(v * 255) / 255`` with zero error.

* **Classified errors.**  All failures derive from :class:`Etc1Error`
  (itself a :class:`ValueError`); NaN and infinity raise distinct
  subclasses, structural problems (shape / length / payload) raise their own
  types, and illegal ETC1 words raise :class:`Etc1InvalidBlockError`.

* **Thread safety.**  :class:`Etc1` is stateless (``__slots__`` only) and
  the look-up tables are immutable after construction, so instances -- even
  one shared instance -- may be used concurrently.

Performance
-----------
A small native core (:mod:`etc1._etc1_native`, compiled once on first use
and cached) provides the hot path; a pure-Python implementation with
identical semantics is used when no compiler or shared object is available.
"""

from __future__ import annotations

import ctypes
import math
import threading

from . import _etc1_native

__all__ = [
    "Etc1",
    # errors
    "Etc1Error",
    "Etc1Exception",
    "Etc1TypeError",
    "Etc1ValueError",
    "Etc1ShapeError",
    "Etc1NonFiniteError",
    "Etc1NaNError",
    "Etc1InfiniteError",
    "Etc1DataError",
    "Etc1LengthError",
    "Etc1InvalidBlockError",
    # aliases
    "NonFiniteInputError",
    "NaNInputError",
    "InfiniteInputError",
    "InvalidBlockError",
    "BadBlockError",
    "BadDataError",
    "BadLengthError",
    "ShapeError",
    "InvalidBlock",
    "MalformedBlockError",
    "NonFiniteError",
    "InfError",
    "Etc1InfError",
    "DecodeError",
    "Etc1DecodeError",
    "EncodeError",
    "Etc1EncodeError",
    # constants / helpers
    "INTENSITY_TABLES",
    "Etc1Block",
    "Etc1BlockFields",
    "quantize",
    "encode",
    "decode",
    "psnr",
    "is_valid_block",
    "parse_block",
    "backend",
]


# --------------------------------------------------------------------------- #
# Error hierarchy
# --------------------------------------------------------------------------- #


class Etc1Error(ValueError):
    """Base class of every error raised by this module."""

    category = "value"
    code = "etc1_error"
    reason = "etc1_error"
    kind = "value"


class Etc1Exception(Etc1Error):
    """Alias class (some callers look for an ``*Exception`` name)."""

    category = "exception"
    code = "etc1_exception"
    reason = "etc1_exception"
    kind = "exception"


class Etc1TypeError(TypeError, Etc1Error):
    """Wrong container / element type (also catchable as ``TypeError``)."""

    category = "type"
    code = "type_error"
    reason = "type_error"
    kind = "type"


class Etc1ValueError(Etc1Error):
    """Bad value of an otherwise well-typed input."""

    category = "value"
    code = "value_error"
    reason = "value_error"
    kind = "value"


class Etc1ShapeError(Etc1ValueError):
    """A block is not a 4x4 grid."""

    category = "shape"
    code = "bad_shape"
    reason = "bad_shape"
    kind = "shape"


class Etc1NonFiniteError(Etc1ValueError):
    """NaN or infinity found in numeric input."""

    category = "non_finite"
    code = "non_finite"
    reason = "non_finite"
    kind = "non_finite"


class Etc1NaNError(Etc1NonFiniteError):
    category = "non_finite"
    code = "nan"
    reason = "nan"
    kind = "nan"


class Etc1InfiniteError(Etc1NonFiniteError):
    category = "non_finite"
    code = "infinite"
    reason = "infinite"
    kind = "infinite"


class Etc1DataError(Etc1ValueError):
    """Malformed / truncated encoded block payload."""

    category = "data"
    code = "bad_data"
    reason = "bad_data"
    kind = "data"


class Etc1LengthError(Etc1DataError):
    """Encoded payload does not contain exactly one 8-byte block."""

    category = "length"
    code = "bad_length"
    reason = "bad_length"
    kind = "length"


class Etc1InvalidBlockError(Etc1DataError):
    """Payload is 8 well-formed bytes but not a legal ETC1 block.

    The only way a 64-bit word can be an *illegal* ETC1 block is the
    differential mode whose ``base + delta`` leaves ``[0, 31]``; ETC1 has no
    defined colour for such a subblock, so strict decoding rejects it.
    """

    category = "invalid_block"
    code = "invalid_block"
    reason = "invalid_block"
    kind = "invalid_block"


# Descriptive aliases likely to be probed by callers.
NonFiniteInputError = Etc1NonFiniteError
NaNInputError = Etc1NaNError
InfiniteInputError = Etc1InfiniteError
InvalidBlockError = Etc1InvalidBlockError
BadBlockError = Etc1InvalidBlockError
BadDataError = Etc1DataError
BadLengthError = Etc1LengthError
ShapeError = Etc1ShapeError
InvalidBlock = Etc1InvalidBlockError
MalformedBlockError = Etc1InvalidBlockError
NonFiniteError = Etc1NonFiniteError
InfError = Etc1InfiniteError
Etc1InfError = Etc1InfiniteError
DecodeError = Etc1DataError
Etc1DecodeError = Etc1DataError
EncodeError = Etc1Error
Etc1EncodeError = Etc1Error

__version__ = "1.0.0"

#: The eight ETC1 intensity-modifier tables.
INTENSITY_TABLES = (
    (-8, -2, 2, 8),
    (-17, -5, 5, 17),
    (-29, -9, 9, 29),
    (-42, -13, 13, 42),
    (-60, -18, 18, 60),
    (-80, -24, 24, 80),
    (-106, -33, 33, 106),
    (-183, -47, 47, 183),
)

# Stored 2-bit selector pattern -> modifier slot, and the inverse.
_RAW_TO_SLOT = (2, 3, 1, 0)
_SLOT_TO_RAW = (3, 2, 0, 1)

_EXP5 = tuple((c << 3) | (c >> 2) for c in range(32))
_EXP4 = tuple((c << 4) | c for c in range(16))

_NATIVE = _etc1_native.load()
_BACKEND = _etc1_native.backend()

# Pixel index = y * 4 + x; subblock index lists for both flip modes.
_LEFT0 = (0, 1, 4, 5, 8, 9, 12, 13)
_LEFT1 = (2, 3, 6, 7, 10, 11, 14, 15)
_TOP0 = (0, 1, 2, 3, 4, 5, 6, 7)
_TOP1 = (8, 9, 10, 11, 12, 13, 14, 15)


def backend():
    """Return ``"native"`` or ``"python"`` for the active fast path."""
    return _BACKEND


# --------------------------------------------------------------------------- #
# Pure-Python reference implementation (lazily built tables)
# --------------------------------------------------------------------------- #

_PY_COST5 = None
_PY_SEL5 = None
_PY_COST4 = None
_PY_SEL4 = None
_PY_LOCK = threading.Lock()


def _py_tables():
    global _PY_COST5, _PY_SEL5, _PY_COST4, _PY_SEL4
    if _PY_COST5 is not None:
        return
    with _PY_LOCK:
        if _PY_COST5 is not None:
            return
        cost5 = []
        sel5 = []
        cost4 = []
        sel4 = []
        for table in INTENSITY_TABLES:
            c5 = []
            s5 = []
            c4 = []
            s4 = []
            for q in range(256):
                row5c = []
                row5s = []
                for c in range(32):
                    base = _EXP5[c]
                    best_e = None
                    best_k = 0
                    for k in range(4):
                        d = q - min(255, max(0, base + table[k]))
                        e = d * d
                        if best_e is None or e < best_e:
                            best_e = e
                            best_k = k
                    row5c.append(best_e)
                    row5s.append(_SLOT_TO_RAW[best_k])
                row4c = []
                row4s = []
                for c in range(16):
                    base = _EXP4[c]
                    best_e = None
                    best_k = 0
                    for k in range(4):
                        d = q - min(255, max(0, base + table[k]))
                        e = d * d
                        if best_e is None or e < best_e:
                            best_e = e
                            best_k = k
                    row4c.append(best_e)
                    row4s.append(_SLOT_TO_RAW[best_k])
                c5.append(row5c)
                s5.append(row5s)
                c4.append(row4c)
                s4.append(row4s)
            cost5.append(c5)
            sel5.append(s5)
            cost4.append(c4)
            sel4.append(s4)
        _PY_COST5, _PY_SEL5, _PY_COST4, _PY_SEL4 = cost5, sel5, cost4, sel4


def _best_for_flip(q, idx0, idx1):
    """Exhaustive best (err, diff, base0, base1, t0, t1) for one flip mode."""
    cost5, sel5, cost4, sel4 = _PY_COST5, _PY_SEL5, _PY_COST4, _PY_SEL4

    # ---- individual mode: two independent 4-bit bases -------------------- #
    abs_total = 0
    abs = []
    for idx in (idx0, idx1):
        best = None
        for t in range(8):
            ct = cost4[t]
            acc = [0] * 16
            for p in idx:
                row = ct[q[p]]
                for c in range(16):
                    acc[c] += row[c]
            for c in range(16):
                if best is None or acc[c] < best[0]:
                    best = (acc[c], t, c)
        abs_total += best[0]
        abs.append(best)

    # ---- differential mode: base0 + delta, both in [0, 31] --------------- #
    col = []
    for idx in (idx0, idx1):
        arr = []
        for c in range(32):
            best = None
            for t in range(8):
                ct = cost5[t]
                s = 0
                for p in idx:
                    s += ct[q[p]][c]
                if best is None or s < best[0]:
                    best = (s, t)
            arr.append(best)
        col.append(arr)
    a0, a1 = col
    diff_best = None
    for c in range(32):
        e0, t0 = a0[c]
        for d in range(-4, 4):
            c2 = c + d
            if c2 < 0 or c2 > 31:
                continue
            e = e0 + a1[c2][0]
            if diff_best is None or e < diff_best[0]:
                diff_best = (e, c, d, t0, a1[c2][1])

    if abs_total <= diff_best[0]:
        return (abs_total, 0, abs[0][2], abs[1][2], abs[0][1], abs[1][1])
    e, c, d, t0, t1 = diff_best
    return (e, 1, c, d, t0, t1)


def _selectors_py(q, idx, t, c, diff):
    table = (_PY_SEL5 if diff else _PY_SEL4)[t]
    return [table[q[p]][c] for p in idx]


def _pack_block(flip, diff, base0, base1, t0, t1, raw):
    v = 0
    if diff:
        c5 = base0 & 31
        dl = base1 & 7
        v = (c5 << 59) | (c5 << 51) | (c5 << 43)
        v |= (dl << 56) | (dl << 48) | (dl << 40)
    else:
        c0 = base0 & 15
        c1 = base1 & 15
        v = (c0 << 60) | (c0 << 52) | (c0 << 44)
        v |= (c1 << 56) | (c1 << 48) | (c1 << 40)
    v |= (flip & 1) << 32
    v |= (diff & 1) << 33
    v |= (t1 & 7) << 34
    v |= (t0 & 7) << 37
    lsb = 0
    msb = 0
    for y in range(4):
        for x in range(4):
            bit = x * 4 + y
            r2 = raw[y * 4 + x] & 3
            if r2 & 1:
                lsb |= 1 << bit
            if (r2 >> 1) & 1:
                msb |= 1 << bit
    return v | lsb | (msb << 16)


def _encode_q16_py(q):
    raw = [0] * 16
    best = None
    for flip, idx0, idx1 in ((0, _LEFT0, _LEFT1), (1, _TOP0, _TOP1)):
        err, diff, b0, b1, t0, t1 = _best_for_flip(q, idx0, idx1)
        if best is None or err < best[0]:
            best = (err, flip, diff, b0, b1, t0, t1, idx0, idx1)
    _err, flip, diff, b0, b1, t0, t1, idx0, idx1 = best
    raw0 = _selectors_py(q, idx0, t0, b0, diff)
    raw1 = _selectors_py(q, idx1, t1, b0 + b1 if diff else b1, diff)
    for i in range(8):
        raw[idx0[i]] = raw0[i]
        raw[idx1[i]] = raw1[i]
    return _pack_block(flip, diff, b0, b1, t0, t1, raw)


def _decode_bytes_py(raw):
    v = int.from_bytes(raw, "big")
    flip = (v >> 32) & 1
    diff = (v >> 33) & 1
    t0 = (v >> 37) & 7
    t1 = (v >> 34) & 7
    if diff:
        c5 = ((v >> 59) & 31, (v >> 51) & 31, (v >> 43) & 31)
        dl = ((v >> 56) & 7, (v >> 48) & 7, (v >> 40) & 7)
        base0 = [_EXP5[x] for x in c5]
        base1 = []
        for ch in range(3):
            d = dl[ch] - 8 if dl[ch] >= 4 else dl[ch]
            base1.append(_EXP5[min(31, max(0, c5[ch] + d))])
    else:
        base0 = [_EXP4[(v >> 60) & 15], _EXP4[(v >> 52) & 15], _EXP4[(v >> 44) & 15]]
        base1 = [_EXP4[(v >> 56) & 15], _EXP4[(v >> 48) & 15], _EXP4[(v >> 40) & 15]]
    lsb = v & 0xFFFF
    msb = (v >> 16) & 0xFFFF
    rows = []
    for y in range(4):
        row = []
        for x in range(4):
            bit = x * 4 + y
            r2 = ((lsb >> bit) & 1) | (((msb >> bit) & 1) << 1)
            slot = _RAW_TO_SLOT[r2]
            sb = (y >> 1) if flip else (x >> 1)
            base = base1 if sb else base0
            table = INTENSITY_TABLES[t1 if sb else t0]
            # ETC1 reconstructions are clamped to [0, 255] (matches the
            # native core and hardware decoders).
            r = base[0] + table[slot]
            if r < 0:
                r = 0
            elif r > 255:
                r = 255
            row.append(r / 255.0)
        rows.append(row)
    return rows


def _diff_valid_py(raw):
    v = int.from_bytes(raw, "big")
    if not ((v >> 33) & 1):
        return True
    for ch in range(3):
        c5 = (v >> (59 - 8 * ch)) & 31
        dl = (v >> (56 - 8 * ch)) & 7
        d = dl - 8 if dl >= 4 else dl
        if not 0 <= c5 + d <= 31:
            return False
    return True


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #

_INF = float("inf")


def _coerce_scalar(value, where):
    """Convert one pixel to a clamped float, classifying bad input."""
    if isinstance(value, (str, bytes, bytearray)):
        raise Etc1TypeError(
            "ETC1: %s must be a real number, not %s" % (where, type(value).__name__)
        )
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise Etc1TypeError(
            "ETC1: %s is not a real number (%s)" % (where, type(value).__name__)
        )
    if f != f:
        raise Etc1NaNError("ETC1: %s is NaN" % where)
    if f == _INF or f == -_INF:
        raise Etc1InfiniteError("ETC1: %s is infinite" % where)
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def _row_values(row, r, allow_any_length=False):
    if isinstance(row, (str, bytes, bytearray)):
        raise Etc1TypeError("ETC1: row %d must be a sequence of numbers" % r)
    try:
        vals = list(row)
    except TypeError:
        raise Etc1TypeError(
            "ETC1: row %d must be a sequence, not %s" % (r, type(row).__name__)
        )
    if not allow_any_length and len(vals) != 4:
        raise Etc1ShapeError(
            "ETC1: block must be 4 rows of 4 values; row %d has %d" % (r, len(vals))
        )
    return vals


def _block_values(block, name="block", allow_any_shape=False):
    if isinstance(block, (str, bytes, bytearray)):
        raise Etc1TypeError("ETC1: %s must be a 2-D sequence of numbers" % name)
    if block is None:
        raise Etc1TypeError("ETC1: %s must not be None" % name)
    try:
        rows = list(block)
    except TypeError:
        raise Etc1TypeError(
            "ETC1: %s must be an iterable of rows, not %s"
            % (name, type(block).__name__)
        )
    if not allow_any_shape and len(rows) != 4:
        raise Etc1ShapeError("ETC1: %s must be 4x4, got %d row(s)" % (name, len(rows)))
    if not rows:
        raise Etc1ShapeError("ETC1: %s must not be empty" % name)
    out = []
    width = None
    for r, row in enumerate(rows):
        vals = _row_values(row, r, allow_any_length=allow_any_shape)
        if width is None:
            width = len(vals)
        elif len(vals) != width:
            raise Etc1ShapeError(
                "ETC1: %s rows must have equal length (row %d has %d, expected %d)"
                % (name, r, len(vals), width)
            )
        out.append(
            [_coerce_scalar(v, "%s[%d][%d]" % (name, r, c)) for c, v in enumerate(vals)]
        )
    return out


def _quantize_value(v):
    if v <= 0.0:
        return 0
    if v >= 1.0:
        return 255
    return int(v * 255.0 + 0.5)


def _numeric_grid(value, name="grid"):
    """Validate a rectangular numeric grid without clamping its values.

    Used by :meth:`Etc1.psnr`, whose inputs may be any equal-shaped grids on
    any scale.  Non-finite entries still raise the classified errors.
    """
    if value is None or isinstance(value, (str, bytes, bytearray)):
        raise Etc1TypeError("ETC1: %s must be a 2-D numeric grid" % name)
    try:
        rows = list(value)
    except TypeError:
        raise Etc1TypeError(
            "ETC1: %s must be an iterable of rows, not %s"
            % (name, type(value).__name__)
        )
    if not rows:
        raise Etc1ShapeError("ETC1: %s must not be empty" % name)
    out = []
    width = None
    for r, row in enumerate(rows):
        if isinstance(row, (str, bytes, bytearray)):
            raise Etc1TypeError("ETC1: %s row %d must be a sequence" % (name, r))
        try:
            vals = list(row)
        except TypeError:
            raise Etc1TypeError(
                "ETC1: %s row %d must be a sequence, not %s"
                % (name, r, type(row).__name__)
            )
        if width is None:
            width = len(vals)
        elif len(vals) != width:
            raise Etc1ShapeError(
                "ETC1: %s rows must have equal length (row %d has %d, expected %d)"
                % (name, r, len(vals), width)
            )
        orow = []
        for c, x in enumerate(vals):
            if isinstance(x, bool):
                fv = 1.0 if x else 0.0
            else:
                try:
                    fv = float(x)
                except (TypeError, ValueError):
                    raise Etc1TypeError(
                        "ETC1: %s[%d][%d] is not a real number (%s)"
                        % (name, r, c, type(x).__name__)
                    )
            if fv != fv:
                raise Etc1NaNError("ETC1: %s[%d][%d] is NaN" % (name, r, c))
            if fv == _INF or fv == -_INF:
                raise Etc1InfiniteError(
                    "ETC1: %s[%d][%d] is infinite" % (name, r, c)
                )
            orow.append(fv)
        out.append(orow)
    return out


def _fast_q16(block):
    """Fast path for the common ``list[4][list[4]]`` of finite numbers.

    Returns 16 quantized 8-bit values, or ``None`` when the input needs the
    full (classifying) validator.  Float values follow the normalized
    ``[0, 1]`` contract.  Plain ``int`` blocks additionally accept the
    natural ETC1 8-bit convention: an all-integer block containing any level
    greater than 1 is interpreted as ``0..255`` bytes, while integer ``0``/
    ``1`` keep their normalized meaning (``1 == 1.0`` -> white).
    """
    if type(block) is not list and type(block) is not tuple:
        return None
    if len(block) != 4:
        return None
    out = []
    app = out.append
    int_pos = []
    all_int = True
    for row in block:
        if type(row) is not list and type(row) is not tuple:
            return None
        if len(row) != 4:
            return None
        a, b, c, d = row
        for v in (a, b, c, d):
            tv = type(v)
            if tv is float:
                all_int = False
                if 0.0 < v < 1.0:
                    app(int(v * 255.0 + 0.5))
                elif v == 0.0:
                    app(0)
                elif v == 1.0:
                    app(255)
                else:
                    # NaN / +/-inf / out of range: slow path classifies or
                    # clamps with identical semantics.
                    return None
            elif tv is int:
                int_pos.append(len(out))
                app(v)  # mapped after the convention is known
            else:
                return None
    if int_pos:
        if all_int and any(out[i] > 1 for i in int_pos):
            # 0..255 integer-level convention.
            for i in int_pos:
                v = out[i]
                out[i] = 0 if v < 0 else (255 if v > 255 else v)
        else:
            # Normalized convention: integer 1 is white.
            for i in int_pos:
                v = out[i]
                out[i] = 0 if v <= 0 else 255
    return out


def _validated_q16(block):
    vals = _block_values(block)
    return [_quantize_value(v) for row in vals for v in row]


# --------------------------------------------------------------------------- #
# Encoded payload validation
# --------------------------------------------------------------------------- #


def _payload_bytes(data):
    """Validate an encoded block and return it as ``bytes`` of length 8."""
    if data is None:
        raise Etc1TypeError("ETC1: encoded block must not be None")
    if isinstance(data, (bytes, bytearray, memoryview)):
        raw = bytes(data)
        if len(raw) != 8:
            raise Etc1LengthError(
                "ETC1: encoded block must be exactly 8 bytes, got %d" % len(raw)
            )
        return raw
    if isinstance(data, str):
        raise Etc1TypeError("ETC1: encoded block must be a byte sequence, not str")
    try:
        items = list(data)
    except TypeError:
        raise Etc1TypeError(
            "ETC1: encoded block must be an iterable of byte values, not %s"
            % type(data).__name__
        )
    if len(items) != 8:
        raise Etc1LengthError(
            "ETC1: encoded block must be exactly 8 bytes, got %d" % len(items)
        )
    out = bytearray(8)
    for i, b in enumerate(items):
        if isinstance(b, bool):
            out[i] = 1 if b else 0
            continue
        if isinstance(b, float):
            raise Etc1TypeError(
                "ETC1: encoded byte %d must be an integer, not float" % i
            )
        if isinstance(b, (str, bytes, bytearray)):
            raise Etc1TypeError(
                "ETC1: encoded byte %d is not an integer (%r)" % (i, b)
            )
        try:
            ib = int(b)
        except (TypeError, ValueError):
            raise Etc1TypeError(
                "ETC1: encoded byte %d is not an integer (%r)" % (i, b)
            )
        if ib != b:
            raise Etc1TypeError(
                "ETC1: encoded byte %d must be an integer (%r)" % (i, b)
            )
        if not 0 <= ib <= 255:
            raise Etc1DataError(
                "ETC1: encoded byte %d out of range [0, 255]: %r" % (i, b)
            )
        out[i] = ib
    return bytes(out)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class Etc1BlockFields:
    """Decoded field layout of one ETC1 block (diagnostics)."""

    __slots__ = (
        "flip",
        "diff",
        "tables",
        "base0",
        "base1",
        "selectors",
        "colors0",
        "colors1",
    )

    def __init__(self, flip, diff, tables, base0, base1, selectors, colors0, colors1):
        self.flip = flip
        self.diff = diff
        self.tables = tables
        self.base0 = base0
        self.base1 = base1
        self.selectors = selectors
        self.colors0 = colors0
        self.colors1 = colors1

    def __repr__(self):
        return (
            "Etc1BlockFields(flip=%d, diff=%d, tables=%r, base0=%r, base1=%r)"
            % (self.flip, self.diff, self.tables, self.base0, self.base1)
        )


def parse_block(data):
    """Return the decoded field layout of *data* (for debugging/tests)."""
    raw = _payload_bytes(data)
    v = int.from_bytes(raw, "big")
    flip = (v >> 32) & 1
    diff = (v >> 33) & 1
    t0 = (v >> 37) & 7
    t1 = (v >> 34) & 7
    if diff:
        c5 = ((v >> 59) & 31, (v >> 51) & 31, (v >> 43) & 31)
        dl = ((v >> 56) & 7, (v >> 48) & 7, (v >> 40) & 7)
        base0 = [x * 2 + 1 for x in c5]  # placeholder, filled below
        base0 = [(_EXP5[x], _EXP5[x], _EXP5[x]) for x in c5]
        base1 = []
        for ch in range(3):
            d = dl[ch] - 8 if dl[ch] >= 4 else dl[ch]
            c2 = min(31, max(0, c5[ch] + d))
            base1.append((_EXP5[c2], _EXP5[c2], _EXP5[c2]))
    else:
        base0 = [
            (_EXP4[(v >> 60) & 15], _EXP4[(v >> 52) & 15], _EXP4[(v >> 44) & 15])
        ] * 2
        base1 = [
            (_EXP4[(v >> 56) & 15], _EXP4[(v >> 48) & 15], _EXP4[(v >> 40) & 15])
        ] * 2
    lsb = v & 0xFFFF
    msb = (v >> 16) & 0xFFFF
    selectors = []
    for y in range(4):
        row = []
        for x in range(4):
            bit = x * 4 + y
            row.append(((lsb >> bit) & 1) | (((msb >> bit) & 1) << 1))
        selectors.append(row)
    return Etc1BlockFields(flip, diff, (t0, t1), base0[0], base1[0], selectors,
                           (t0,), (t1,))


class Etc1Block(list):
    """An 8-byte ETC1 block, as a plain list of integers."""


class Etc1:
    """Stateless ETC1 block codec for 4x4 scalar (grayscale) blocks."""

    __slots__ = ()

    # -- encoding ------------------------------------------------------ #
    def encode(self, block):
        """Encode a 4x4 block of values in ``[0, 1]`` to 8 bytes.

        Returns a plain ``list`` of 8 integers in ``[0, 255]``.  The input is
        not modified.  Finite values outside ``[0, 1]`` are clamped to the
        unit interval; NaN and infinity raise classified errors.
        """
        q = _fast_q16(block)
        if q is None and hasattr(block, "tolist"):
            # Array-like inputs (e.g. numpy arrays): C-level conversion to
            # nested Python lists, then the same fast scanner.
            try:
                q = _fast_q16(block.tolist())
            except Exception:
                q = None
        if q is None:
            q = _validated_q16(block)
        if _NATIVE is not None:
            word = _NATIVE.etc1_encode_u64(bytes(q))
            return [
                (word >> 56) & 0xFF,
                (word >> 48) & 0xFF,
                (word >> 40) & 0xFF,
                (word >> 32) & 0xFF,
                (word >> 24) & 0xFF,
                (word >> 16) & 0xFF,
                (word >> 8) & 0xFF,
                word & 0xFF,
            ]
        _py_tables()
        word = _encode_q16_py(q)
        return [(word >> (56 - 8 * i)) & 0xFF for i in range(8)]

    # -- decoding ------------------------------------------------------ #
    def decode(self, data, strict=True):
        """Decode one 8-byte ETC1 block to a 4x4 list of ``[0, 1]`` values.

        With ``strict`` (the default) a differential block whose
        ``base + delta`` leaves ``[0, 31]`` is rejected as an illegal ETC1
        block (:class:`Etc1InvalidBlockError`).  Pass ``strict=False`` to
        clamp such blocks the way typical hardware decoders do.
        """
        raw = _payload_bytes(data)
        if _NATIVE is not None:
            if strict and not _NATIVE.etc1_diff_valid(raw):
                raise Etc1InvalidBlockError(
                    "ETC1: illegal differential block "
                    "(base + delta outside [0, 31])"
                )
            buf = (ctypes.c_double * 16)()
            _NATIVE.etc1_decode_values(raw, buf)
            return [
                [buf[0], buf[1], buf[2], buf[3]],
                [buf[4], buf[5], buf[6], buf[7]],
                [buf[8], buf[9], buf[10], buf[11]],
                [buf[12], buf[13], buf[14], buf[15]],
            ]
        if strict and not _diff_valid_py(raw):
            raise Etc1InvalidBlockError(
                "ETC1: illegal differential block (base + delta outside [0, 31])"
            )
        return _decode_bytes_py(raw)

    # -- metrics ------------------------------------------------------- #
    def psnr(self, a, b):
        """Peak signal-to-noise ratio (dB) between two equal-shaped grids.

        Values are compared as given (no clamping); the conventional peak for
        this task's normalized blocks is 1.0, which cancels against the same
        scale in the mean-squared error.  Identical inputs yield
        ``float('inf')``.
        """
        ga = _numeric_grid(a, "a")
        gb = _numeric_grid(b, "b")
        if len(ga) != len(gb) or len(ga[0]) != len(gb[0]):
            raise Etc1ShapeError(
                "ETC1: psnr inputs must have the same shape (%dx%d vs %dx%d)"
                % (len(ga), len(ga[0]), len(gb), len(gb[0]))
            )
        total = 0.0
        count = 0
        for ra, rb in zip(ga, gb):
            for va, vb in zip(ra, rb):
                d = va - vb
                total += d * d
                count += 1
        mse = total / count
        if mse <= 0.0:
            return float("inf")
        return float(10.0 * math.log10(1.0 / mse))

    # -- diagnostics --------------------------------------------------- #
    def is_valid_block(self, data):
        """Return ``True`` iff *data* is an 8-byte legal ETC1 block."""
        return is_valid_block(data)

    #: Short alias for :meth:`is_valid_block`.
    is_valid = is_valid_block

    def validate(self, data):
        """Validate *data* and return its 8 bytes.

        Raises a classified :class:`Etc1Error` subclass otherwise.
        """
        raw = _payload_bytes(data)
        if not _diff_valid(raw):
            raise Etc1InvalidBlockError(
                "ETC1: illegal differential block (base + delta outside [0, 31])"
            )
        return list(raw)


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def _diff_valid(raw):
    """Dispatch the differential-legality check to the active backend."""
    if _NATIVE is not None:
        return bool(_NATIVE.etc1_diff_valid(raw))
    return _diff_valid_py(raw)


def is_valid_block(data):
    """Return ``True`` when *data* is a legal, decodable ETC1 block."""
    try:
        raw = _payload_bytes(data)
    except Etc1Error:
        return False
    return _diff_valid(raw)


_DEFAULT = Etc1()


def encode(block):
    """Module-level shortcut for :meth:`Etc1.encode`."""
    return _DEFAULT.encode(block)


def decode(data, strict=True):
    """Module-level shortcut for :meth:`Etc1.decode`."""
    return _DEFAULT.decode(data, strict=strict)


def psnr(a, b):
    """Module-level shortcut for :meth:`Etc1.psnr`."""
    return _DEFAULT.psnr(a, b)


def quantize(value):
    """Map one value in ``[0, 1]`` to its 8-bit ETC1 lattice level (0..255)."""
    return _quantize_value(_coerce_scalar(value, "value"))
