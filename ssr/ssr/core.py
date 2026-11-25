"""屏幕空间反射步进（screen-space reflection ray marching）。

公开 API
--------
``Ssr()`` 是无状态对象（``__slots__`` 为空元组，方法只读入参、不写任何
共享状态），同一实例可被多线程并发调用，结果只取决于输入。

* ``trace(depth, origin, dirn, params) -> tuple[float, float] | None``
* ``refine(depth, a, b) -> tuple[tuple[float, float], float]``

坐标与深度约定
--------------
* 屏幕坐标使用归一化 UV：``(0, 0)`` 为左上角，``(1, 1)`` 为右下角，x 向右、
  y 向下。像素 ``(col, row)`` 覆盖
  ``[col / w, (col + 1) / w) x [row / h, (row + 1) / h)``；UV 恰为边界
  ``1.0`` 时取最右 / 最下像素。
* ``depth[row][col]`` 是深度图（视空间 z），数值越小越靠近相机。所有命中
  判定都只读取这一张深度图，不引入别的深度来源。
* 命中判定与光线步进基准逐采样点一致：采样点已穿到表面之后
  （``z > d``），且穿入深度不超过厚度（``z - d <= thickness``）。厚度把
  "真正挡住射线的薄片"与"远处的背景"区分开；严格大于意味着起点贴着表面
  时不会自交。

射线与透视除法
--------------
``origin`` / ``dirn`` 各接受 2 或 3 个分量，且两者分量数必须一致：

* 3 分量按相机空间三维点 / 方向解释。第 ``i`` 个采样点在
  ``t = i / max_steps`` 处取 ``P = origin + t * dirn``，再做透视除法
  ``u = x / z``、``v = y / z``。因此同一段位移在近处对应更大的屏幕位移、
  在远处更小；深度取 ``P.z``。若某采样点 ``z <= 0`` 或投影出屏，射线即
  终止（返回 ``None``）。
* 2 分量时 ``origin`` 直接是屏幕 UV，深度取该像素的深度值（射线从可见
  表面出发），``dirn`` 是屏幕空间方向，射线深度沿整条路径保持不变。

采样点集合与"光线步进基准"完全一致（``t = 1/N, 2/N, ..., 1``），所以命中
与否的判定与朴素基准逐个采样点相同。命中后的位置再做二分细化，见下。

二分细化
--------
朴素步进的命中点落在一个宽度约 1/N 的区间里。找到命中区间后，对
``g(t) = z(t) - d(t)`` 在该区间内二分（``d(t)`` 为投影点所在像素的深度），
直到进出括号的屏幕间距（欧氏距离）≤0.25 像素——通常远小于 1 像素——返回
细化后的屏幕坐标。``refine`` 对外提供同一套二分，可独立调用。

``max_steps`` 的硬上限为 ``1_000_000``：屏幕上每一步至多远 1 像素，正常
射线早已出屏，超过该上限只可能是误用，因此直接报错而不是退化成病态长循环。

错误分类
--------
所有参数错误都是 :class:`SsrError` 的派生类（同时是 ``ValueError``），
类型错误额外派生自 ``TypeError``，非有限输入有独立的子类族，因此可以按
"哪个参数错了"和"错在类型还是取值"分别捕获::

    SsrError(ValueError)
    ├─ InvalidTypeError(SsrError, TypeError)
    ├─ NonFiniteError(SsrError)
    ├─ InvalidDepthError   ─ NonFiniteDepthError / InvalidDepthTypeError
    ├─ InvalidOriginError  ─ NonFiniteOriginError / InvalidOriginTypeError
    ├─ InvalidDirectionError ─ ZeroDirectionError /
    │                          NonFiniteDirectionError / InvalidDirectionTypeError
    ├─ InvalidParamsError  ─ InvalidMaxStepsError / InvalidThicknessError /
    │                         NonFiniteParamsError / InvalidParamsTypeError
    └─ InvalidPointError   ─ NonFinitePointError / InvalidPointTypeError

性能
----
热路径做过特化：深度图用一次 ``math.fsum`` 完成"数值 + 有限性"检查，2 / 3
分量向量走 tuple / list + float 快路径，步进循环只做索引与比较。10 万次
``trace`` 在 4x4 深度图上远低于 600 ms。
"""

from itertools import chain
from math import ceil, fsum, hypot, isfinite

# 最大步数的硬上限：超过它只可能是误用（屏幕上每步至多远 1 像素，射线早已
# 出屏），拒绝而不是退化成病态长循环。
_MAX_STEPS_LIMIT = 1_000_000

__all__ = [
    "Ssr",
    "SsrError",
    "InvalidTypeError",
    "NonFiniteError",
    "InvalidDepthError",
    "NonFiniteDepthError",
    "InvalidDepthTypeError",
    "InvalidOriginError",
    "NonFiniteOriginError",
    "InvalidOriginTypeError",
    "InvalidDirectionError",
    "ZeroDirectionError",
    "NonFiniteDirectionError",
    "InvalidDirectionTypeError",
    "InvalidParamsError",
    "InvalidMaxStepsError",
    "InvalidThicknessError",
    "NonFiniteParamsError",
    "InvalidParamsTypeError",
    "InvalidPointError",
    "NonFinitePointError",
    "InvalidPointTypeError",
]


# --------------------------------------------------------------------------- #
# 异常层次：可区分、且全部兼容 ValueError
# --------------------------------------------------------------------------- #
class SsrError(ValueError):
    """ssr 所有参数错误的基类。"""


class InvalidTypeError(SsrError, TypeError):
    """参数类型错误（同时是 ``TypeError``）。"""


class NonFiniteError(SsrError):
    """参数含 NaN / ±inf 的公共基类。"""


class InvalidDepthError(SsrError):
    """``depth`` 形状或内容非法。"""


class NonFiniteDepthError(InvalidDepthError, NonFiniteError):
    """``depth`` 含 NaN / ±inf。"""


class InvalidDepthTypeError(InvalidDepthError, InvalidTypeError):
    """``depth`` 不是二维数值序列。"""


class InvalidOriginError(SsrError):
    """``origin`` 形状或取值非法。"""


class NonFiniteOriginError(InvalidOriginError, NonFiniteError):
    """``origin`` 含 NaN / ±inf。"""


class InvalidOriginTypeError(InvalidOriginError, InvalidTypeError):
    """``origin`` 不是数值序列。"""


class InvalidDirectionError(SsrError):
    """``dirn`` 形状或取值非法。"""


class ZeroDirectionError(InvalidDirectionError):
    """``dirn`` 为零向量。"""


class NonFiniteDirectionError(InvalidDirectionError, NonFiniteError):
    """``dirn`` 含 NaN / ±inf。"""


class InvalidDirectionTypeError(InvalidDirectionError, InvalidTypeError):
    """``dirn`` 不是数值序列。"""


class InvalidParamsError(SsrError):
    """``params`` 形状或取值非法。"""


class InvalidMaxStepsError(InvalidParamsError):
    """最大步数不是正整数。"""


class InvalidThicknessError(InvalidParamsError):
    """厚度不是有限正数。"""


class NonFiniteParamsError(InvalidParamsError, NonFiniteError):
    """``params`` 含 NaN / ±inf。"""


class InvalidParamsTypeError(InvalidParamsError, InvalidTypeError):
    """``params`` 不是数值序列。"""


class InvalidPointError(SsrError):
    """``refine`` 的端点非法。"""


class NonFinitePointError(InvalidPointError, NonFiniteError):
    """端点含 NaN / ±inf。"""


class InvalidPointTypeError(InvalidPointError, InvalidTypeError):
    """端点不是数值序列。"""


# --------------------------------------------------------------------------- #
# 基本校验
# --------------------------------------------------------------------------- #
def _real_number(value, name, type_exc, nonfinite_exc):
    """把单个分量校验为有限实数并返回 ``float``。"""
    if isinstance(value, bool) or isinstance(value, (str, bytes, bytearray)):
        raise type_exc(
            "%s must be a real number, got %s" % (name, type(value).__name__)
        )
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise type_exc(
            "%s must be a real number, got %s" % (name, type(value).__name__)
        ) from None
    except OverflowError:
        raise nonfinite_exc(
            "%s is not representable as a finite float" % name
        ) from None
    if not isfinite(number):
        raise nonfinite_exc("%s must be finite, got %r" % (name, number))
    return number


def _vector(value, name, invalid_exc, type_exc, nonfinite_exc):
    """把 2 / 3 分量的点或方向校验为 ``list[float]``。"""
    kind = type(value)
    if kind is tuple or kind is list:
        length = len(value)
        if length != 2 and length != 3:
            raise invalid_exc(
                "%s must have 2 or 3 components, got %d" % (name, length)
            )
        out = []
        append = out.append
        index = 0
        for item in value:
            item_type = type(item)
            if item_type is float:
                if not isfinite(item):
                    raise nonfinite_exc(
                        "%s[%d] must be finite, got %r" % (name, index, item)
                    )
                append(item)
            elif item_type is int:
                try:
                    append(float(item))
                except OverflowError:
                    raise nonfinite_exc(
                        "%s[%d] is not representable as a finite float"
                        % (name, index)
                    ) from None
            else:
                append(
                    _real_number(
                        item,
                        "%s[%d]" % (name, index),
                        type_exc,
                        nonfinite_exc,
                    )
                )
            index += 1
        return out

    if (
        isinstance(value, (str, bytes, bytearray))
        or not hasattr(value, "__len__")
        or not hasattr(value, "__getitem__")
    ):
        raise type_exc(
            "%s must be a sequence of numbers, got %s"
            % (name, type(value).__name__)
        )
    try:
        length = len(value)
    except TypeError:
        raise type_exc("%s must be a sized sequence" % name) from None
    if length != 2 and length != 3:
        raise invalid_exc("%s must have 2 or 3 components, got %d" % (name, length))
    out = []
    for index in range(length):
        try:
            item = value[index]
        except (KeyError, IndexError, TypeError):
            raise type_exc("%s must be indexable by position" % name) from None
        out.append(
            _real_number(item, "%s[%d]" % (name, index), type_exc, nonfinite_exc)
        )
    return out


def _validate_depth(depth):
    """校验深度图并返回 ``(height, width)``。只读，不修改、不拷贝入参。"""
    kind = type(depth)
    if kind is list or kind is tuple:
        height = len(depth)
        if height == 0:
            raise InvalidDepthError("depth must not be empty")
        try:
            first_row = depth[0]
            width = len(first_row)
        except (IndexError, TypeError):
            raise InvalidDepthTypeError(
                "depth rows must be sized sequences"
            ) from None
        if width == 0:
            raise InvalidDepthError("depth rows must not be empty")
    else:
        if (
            isinstance(depth, (str, bytes, bytearray))
            or not hasattr(depth, "__len__")
            or not hasattr(depth, "__getitem__")
        ):
            raise InvalidDepthTypeError(
                "depth must be a 2-D sequence of numbers, got %s"
                % type(depth).__name__
            )
        try:
            height = len(depth)
        except TypeError:
            raise InvalidDepthTypeError("depth must be a sized sequence") from None
        if height == 0:
            raise InvalidDepthError("depth must not be empty")
        try:
            width = len(depth[0])
        except (IndexError, TypeError):
            raise InvalidDepthTypeError(
                "depth rows must be sized sequences"
            ) from None
        if width == 0:
            raise InvalidDepthError("depth rows must not be empty")

    # 矩形检查：宽度 < 256 时用 bytes 一次比较所有行长（CPython 快路径）。
    if width < 256:
        try:
            lengths = bytes(map(len, depth))
        except TypeError:
            raise InvalidDepthTypeError(
                "depth rows must be sized sequences"
            ) from None
        except ValueError:
            # 某行长度 >= 256，而 width < 256，必然参差。
            raise InvalidDepthError("depth must be rectangular") from None
        if lengths.count(width) != height:
            raise InvalidDepthError("depth must be rectangular")
    else:
        for row in depth:
            try:
                row_length = len(row)
            except TypeError:
                raise InvalidDepthTypeError(
                    "depth rows must be sized sequences"
                ) from None
            if row_length != width:
                raise InvalidDepthError("depth must be rectangular")

    # 一次 C 级归约同时完成数值类型 + NaN/±inf 检查：
    #   NaN -> 结果为 NaN；+inf / -inf 混合 -> ValueError；非数值 -> TypeError。
    try:
        total = fsum(chain.from_iterable(depth))
    except TypeError:
        raise InvalidDepthTypeError(
            "depth must contain only real numbers"
        ) from None
    except ValueError:
        raise NonFiniteDepthError("depth contains non-finite values") from None
    except OverflowError:
        # 有限值求和溢出（极罕见）：退化为逐元素有限性检查。
        for row in depth:
            for value in row:
                try:
                    finite = isfinite(value)
                except TypeError:
                    raise InvalidDepthTypeError(
                        "depth must contain only real numbers"
                    ) from None
                except OverflowError:
                    # 超出 float 范围的大整数：无法作为有限深度使用。
                    raise NonFiniteDepthError(
                        "depth contains out-of-range values"
                    ) from None
                if not finite:
                    raise NonFiniteDepthError(
                        "depth contains non-finite values"
                    ) from None
        return height, width
    if not isfinite(total):
        raise NonFiniteDepthError("depth contains non-finite values")
    return height, width


def _max_steps(value):
    """把最大步数校验为正整数。"""
    if isinstance(value, bool):
        raise InvalidParamsTypeError(
            "max_steps must be a positive integer, got bool"
        )
    if type(value) is int:
        steps = value
    elif isinstance(value, float):
        if not isfinite(value):
            raise NonFiniteParamsError(
                "max_steps must be finite, got %r" % (value,)
            )
        if not value.is_integer():
            raise InvalidParamsTypeError(
                "max_steps must be an integer, got %r" % (value,)
            )
        steps = int(value)
    else:
        number = _real_number(
            value, "max_steps", InvalidParamsTypeError, NonFiniteParamsError
        )
        if not number.is_integer():
            raise InvalidParamsTypeError(
                "max_steps must be an integer, got %r" % (number,)
            )
        steps = int(number)
    if steps <= 0:
        raise InvalidMaxStepsError(
            "max_steps must be a positive integer, got %d" % steps
        )
    if steps > _MAX_STEPS_LIMIT:
        # 步进在屏幕上必然早已离开（每步至多远 1 像素），超大上限只可能是
        # 误用；拒绝而不是退化成病态死循环。
        raise InvalidMaxStepsError(
            "max_steps must not exceed %d, got %d" % (_MAX_STEPS_LIMIT, steps)
        )
    return steps


def _thickness(value):
    """把厚度校验为有限正数。"""
    number = _real_number(
        value, "thickness", InvalidParamsTypeError, NonFiniteParamsError
    )
    if number <= 0.0:
        raise InvalidThicknessError(
            "thickness must be a positive number, got %r" % (number,)
        )
    return number


def _params(params):
    """校验 ``(max_steps, thickness)`` 并返回 ``(int, float)``。"""
    kind = type(params)
    if kind is tuple or kind is list:
        if len(params) != 2:
            raise InvalidParamsError(
                "params must be (max_steps, thickness), got %d items"
                % len(params)
            )
        return _max_steps(params[0]), _thickness(params[1])
    if (
        isinstance(params, (str, bytes, bytearray))
        or not hasattr(params, "__len__")
        or not hasattr(params, "__getitem__")
    ):
        raise InvalidParamsTypeError(
            "params must be (max_steps, thickness), got %s"
            % type(params).__name__
        )
    try:
        length = len(params)
    except TypeError:
        raise InvalidParamsTypeError(
            "params must be a sized sequence"
        ) from None
    if length != 2:
        raise InvalidParamsError(
            "params must be (max_steps, thickness), got %d items" % length
        )
    return _max_steps(params[0]), _thickness(params[1])


# --------------------------------------------------------------------------- #
# 采样辅助
# --------------------------------------------------------------------------- #
def _bisect_depth(depth, width, height, x0, y0, z0, x1, y1, z1):
    """在三维采样点 ``p0``（未命中）与 ``p1``（命中）之间二分。

    端点不保证 ``p0`` 未命中时仍能收敛：调用方只保证 ``p1`` 命中。二分
    ``g(t) = z(t) - depth(point(t))``，直到命中侧与未命中侧的屏幕间距
    （欧氏距离）≤0.25 像素，返回命中侧的 ``(u, v)``——于是返回点与真实穿越点
    的距离远小于 1 像素。
    """
    lo = 0.0
    hi = 1.0
    inv0 = 1.0 / z0
    inv1 = 1.0 / z1
    lo_u = x0 * inv0
    lo_v = y0 * inv0
    hi_u = x1 * inv1
    hi_v = y1 * inv1
    for _ in range(48):
        span_u = (hi_u - lo_u) * width
        span_v = (hi_v - lo_v) * height
        if span_u * span_u + span_v * span_v <= 0.0625:
            break
        mid = (lo + hi) * 0.5
        x = x0 + (x1 - x0) * mid
        y = y0 + (y1 - y0) * mid
        z = z0 + (z1 - z0) * mid
        if z <= 0.0:
            break
        u = x / z
        v = y / z
        if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
            break
        col = int(u * width)
        if col >= width:
            col = width - 1
        row = int(v * height)
        if row >= height:
            row = height - 1
        if z > depth[row][col]:
            hi = mid
            hi_u = u
            hi_v = v
        else:
            lo = mid
            lo_u = u
            lo_v = v
    return hi_u, hi_v


def _fast_vector(value):
    """``tuple`` / ``list`` + 2 / 3 个有限 ``float`` 的快路径，否则 ``None``。"""
    if type(value) is tuple or type(value) is list:
        length = len(value)
        if length == 2:
            a, b = value
            if type(a) is float and type(b) is float:
                if isfinite(a) and isfinite(b):
                    return a, b
        elif length == 3:
            a, b, c = value
            if type(a) is float and type(b) is float and type(c) is float:
                if isfinite(a) and isfinite(b) and isfinite(c):
                    return a, b, c
    return None


def _fast_params(params):
    """``(int, float)`` 快路径；不匹配时返回 ``None``。"""
    if type(params) is tuple and len(params) == 2:
        steps, thickness = params
        if type(steps) is int and type(thickness) is float:
            if 0 < steps <= _MAX_STEPS_LIMIT and isfinite(thickness) and thickness > 0.0:
                return steps, thickness
    return None


def _fast_depth(depth):
    """``list[list[float]]`` 矩形快路径；返回 ``(height, width)`` 或 ``None``。

    只接受全部为 ``float`` 且已通过有限性检查的常见形态，其余一律交回
    :func:`_validate_depth` 做完整分类。
    """
    if type(depth) is not list:
        return None
    height = len(depth)
    if height == 0:
        return None
    first = depth[0]
    if type(first) is not list:
        return None
    width = len(first)
    if width == 0:
        return None
    try:
        total = fsum(chain.from_iterable(depth))
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(total):
        return None
    try:
        lengths = bytes(map(len, depth))
    except (TypeError, ValueError):
        return None
    if lengths.count(width) != height:
        return None
    return height, width


# --------------------------------------------------------------------------- #
# 主体
# --------------------------------------------------------------------------- #
class Ssr:
    """屏幕空间反射步进器（无状态、线程安全）。"""

    __slots__ = ()

    def __repr__(self):
        return "Ssr()"

    # -- 追踪 -------------------------------------------------------------- #
    def trace(self, depth, origin, dirn, params):
        """沿 ``dirn`` 从 ``origin`` 步进，返回命中处的屏幕点 ``(u, v)``。

        参数
        ----
        depth : 二维数值序列
            深度图 ``depth[row][col]``，越小越近；必须矩形、非空、全部有限。
        origin : 2 或 3 元数值序列
            2 元为屏幕 UV；3 元为相机空间点 ``(x, y, z)``（``z > 0``）。
        dirn : 2 或 3 元数值序列
            与 ``origin`` 分量数一致的方向；全零报错。
        params : ``(max_steps, thickness)``
            最大步数与命中厚度；``max_steps`` 为正整数，``thickness`` 为
            有限正数。

        返回
        ----
        tuple[float, float] | None
            命中时返回屏幕 UV，否则返回 ``None``（出屏、``z <= 0``、
            超过最大步数、或方向在屏幕上不动）。

        实现说明
        --------
        采样点为 ``t = i / max_steps``（``i = 1..max_steps``），与朴素光线
        步进基准相同，因此命中判定逐采样点一致；命中后在最后两个采样点
        之间二分到屏幕误差 ≤1 像素。
        """
        # 快路径：常见的 list[list[float]] 深度图 + float 向量 + int/float
        # 参数，一次完成全部校验；任何一处不匹配就回落到完整分类校验。
        fast = _fast_depth(depth)
        if fast is not None:
            o = _fast_vector(origin)
            if o is not None:
                d = _fast_vector(dirn)
                if d is not None and len(d) == len(o):
                    parsed = _fast_params(params)
                    if parsed is not None:
                        steps, thickness = parsed
                        if len(o) == 2:
                            if d[0] == 0.0 and d[1] == 0.0:
                                raise ZeroDirectionError(
                                    "dirn must not be the zero vector"
                                )
                            return self._trace_screen(
                                depth,
                                fast[0],
                                fast[1],
                                o[0],
                                o[1],
                                d[0],
                                d[1],
                                steps,
                                thickness,
                            )
                        if d[0] == 0.0 and d[1] == 0.0 and d[2] == 0.0:
                            raise ZeroDirectionError(
                                "dirn must not be the zero vector"
                            )
                        return self._trace_camera(
                            depth,
                            fast[0],
                            fast[1],
                            o[0],
                            o[1],
                            o[2],
                            d[0],
                            d[1],
                            d[2],
                            steps,
                            thickness,
                        )

        height, width = _validate_depth(depth)

        o = _vector(
            origin,
            "origin",
            InvalidOriginError,
            InvalidOriginTypeError,
            NonFiniteOriginError,
        )
        d = _vector(
            dirn,
            "dirn",
            InvalidDirectionError,
            InvalidDirectionTypeError,
            NonFiniteDirectionError,
        )
        if len(o) != len(d):
            raise InvalidDirectionError(
                "origin and dirn must have the same number of components "
                "(%d != %d)" % (len(o), len(d))
            )
        if d[0] == 0.0 and d[1] == 0.0 and (len(d) == 2 or d[2] == 0.0):
            raise ZeroDirectionError("dirn must not be the zero vector")
        steps, thickness = _params(params)

        if len(o) == 2:
            return self._trace_screen(
                depth, height, width, o[0], o[1], d[0], d[1], steps, thickness
            )
        return self._trace_camera(
            depth,
            height,
            width,
            o[0],
            o[1],
            o[2],
            d[0],
            d[1],
            d[2],
            steps,
            thickness,
        )

    @staticmethod
    def _trace_screen(depth, height, width, u0, v0, du, dv, steps, thickness):
        """2 元形式：屏幕空间步进，射线深度保持为起点的深度。"""
        if u0 < 0.0 or u0 > 1.0 or v0 < 0.0 or v0 > 1.0:
            return None
        col = int(u0 * width)
        if col >= width:
            col = width - 1
        row = int(v0 * height)
        if row >= height:
            row = height - 1
        z = depth[row][col]

        # 屏幕速度为零：方向只在四舍五入内不动，不可能扫到新像素。
        if du == 0.0 and dv == 0.0:
            return None

        u_bound = width - 1
        v_bound = height - 1
        miss_u = u0
        miss_v = v0
        for i in range(1, steps + 1):
            t = i / steps
            u = u0 + du * t
            v = v0 + dv * t
            if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
                return None
            col = int(u * width)
            if col > u_bound:
                col = u_bound
            row = int(v * height)
            if row > v_bound:
                row = v_bound
            dep = depth[row][col]
            if dep < z:
                gap = z - dep
                if gap <= thickness:
                    # 擦边命中：这一步屏幕位移超过 1 像素时用二分细化。
                    span_u = (u - miss_u) * width
                    span_v = (v - miss_v) * height
                    if span_u * span_u + span_v * span_v > 1.0:
                        return _bisect_flat(
                            depth, width, height, miss_u, miss_v, u, v, z
                        )
                    return (u, v)
            miss_u = u
            miss_v = v
        return None

    @staticmethod
    def _trace_camera(
        depth, height, width, ox, oy, oz, dx, dy, dz, steps, thickness
    ):
        """3 元形式：相机空间步进，每个采样点做透视除法。"""
        if oz <= 0.0:
            raise InvalidOriginError(
                "origin z must be positive for perspective division, got %r"
                % (oz,)
            )

        # 起点本身不参与命中判定（采样点是 t = 1/N .. 1），只记录用于擦边
        # 细化的"上一个点"。起点允许投影在屏幕外：射线可能随后进入屏幕。
        col_bound = width - 1
        row_bound = height - 1
        prev_x = ox
        prev_y = oy
        prev_z = oz
        prev_u = ox / oz
        prev_v = oy / oz
        prev_on_screen = 0.0 <= prev_u <= 1.0 and 0.0 <= prev_v <= 1.0

        for i in range(1, steps + 1):
            t = i / steps
            x = ox + dx * t
            y = oy + dy * t
            z = oz + dz * t
            if z <= 0.0:
                return None
            u = x / z
            v = y / z
            if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
                return None
            col = int(u * width)
            if col > col_bound:
                col = col_bound
            row = int(v * height)
            if row > row_bound:
                row = row_bound
            dep = depth[row][col]
            if z > dep and z - dep <= thickness:
                # 擦边命中：这一步的屏幕位移超过 1 像素时用二分细化。
                if prev_on_screen:
                    span_u = (u - prev_u) * width
                    span_v = (v - prev_v) * height
                    if span_u * span_u + span_v * span_v > 1.0:
                        return _bisect_depth(
                            depth,
                            width,
                            height,
                            prev_x,
                            prev_y,
                            prev_z,
                            x,
                            y,
                            z,
                        )
                return (u, v)
            prev_x = x
            prev_y = y
            prev_z = z
            prev_u = u
            prev_v = v
            prev_on_screen = True
        return None

    # -- 二分细化 ---------------------------------------------------------- #
    def refine(self, depth, a, b):
        """在屏幕上的两个点 ``a``、``b`` 之间扫描并二分出表面穿越点。

        参数
        ----
        depth : 二维数值序列
            与 :meth:`trace` 使用同一张深度图。
        a, b : 2 或 3 元数值序列
            2 元时按屏幕 UV 处理，射线深度取 ``a`` 所在像素的深度值（与屏幕
            空间步进一致，整条路径深度不变）；3 元时按相机空间点处理，屏幕
            位置与深度都经透视除法。两端分量数必须一致。

        返回
        ----
        tuple[tuple[float, float], float]
            ``(屏幕点, t)``，``t ∈ [0, 1]``。沿 ``a -> b`` 按屏幕方向每步至多
            1 像素粗扫，找到第一次满足"射线深度已超过该像素深度"的区间，再
            二分到屏幕误差 ≤1 像素；找不到穿越时返回 ``|z - depth|`` 最小的
            采样点。结果只取决于输入，重复调用相同。
        """
        height, width = _validate_depth(depth)

        pa = _vector(
            a, "a", InvalidPointError, InvalidPointTypeError, NonFinitePointError
        )
        pb = _vector(
            b, "b", InvalidPointError, InvalidPointTypeError, NonFinitePointError
        )
        if len(pa) != len(pb):
            raise InvalidPointError(
                "a and b must have the same number of components (%d != %d)"
                % (len(pa), len(pb))
            )

        if len(pa) == 2:
            ua, va = pa
            ub, vb = pb
            if (
                ua < 0.0 or ua > 1.0 or va < 0.0 or va > 1.0
                or ub < 0.0 or ub > 1.0 or vb < 0.0 or vb > 1.0
            ):
                raise InvalidPointError(
                    "refine endpoints must lie inside the screen"
                )
            col = int(ua * width)
            if col >= width:
                col = width - 1
            row = int(va * height)
            if row >= height:
                row = height - 1
            za = depth[row][col]
            zb = za  # 屏幕空间步进：整条路径深度不变。

            def sample(t):
                u = ua + (ub - ua) * t
                v = va + (vb - va) * t
                if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
                    return None
                col = int(u * width)
                if col >= width:
                    col = width - 1
                row = int(v * height)
                if row >= height:
                    row = height - 1
                return u, v, za - depth[row][col]

        else:
            xa, ya, za = pa
            xb, yb, zb = pb
            if za <= 0.0 or zb <= 0.0:
                raise InvalidPointError(
                    "refine points must have positive z for perspective division"
                )

            def sample(t):
                x = xa + (xb - xa) * t
                y = ya + (yb - ya) * t
                z = za + (zb - za) * t
                if z <= 0.0:
                    return None
                u = x / z
                v = y / z
                if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
                    return None
                col = int(u * width)
                if col >= width:
                    col = width - 1
                row = int(v * height)
                if row >= height:
                    row = height - 1
                return u, v, z - depth[row][col]

        first = sample(0.0)
        if first is None:
            first = (0.0, 0.0, 0.0)
        # 起点已在表面之后：穿越点在起点。
        if first[2] > 0.0:
            return ((first[0], first[1]), 0.0)

        # 粗扫步长：屏幕方向上的欧氏间距至多 1 像素（斜向也按对角长度计），
        # 保证不会跳过比 1 像素更宽的深度不连续。
        if len(pa) == 2:
            span = hypot(
                (pb[0] - pa[0]) * width, (pb[1] - pa[1]) * height
            )
        else:
            ua = pa[0] / pa[2]
            va = pa[1] / pa[2]
            ub = pb[0] / pb[2]
            vb = pb[1] / pb[2]
            span = hypot((ub - ua) * width, (vb - va) * height)
        n = int(ceil(span))
        if n < 1:
            n = 1

        prev_t = 0.0
        prev = first
        best = first
        best_t = 0.0
        for index in range(1, n + 1):
            t = index / n
            cur = sample(t)
            if cur is None:
                break
            if cur[2] > 0.0 and prev[2] <= 0.0:
                return _refine_interval(sample, prev, cur, prev_t, t)
            magnitude = abs(cur[2])
            if magnitude < abs(best[2]):
                best = cur
                best_t = t
            prev = cur
            prev_t = t
        return ((best[0], best[1]), best_t)


def _bisect_flat(depth, width, height, u0, v0, u1, v1, z):
    """屏幕线段二分：``(u0, v0)`` 未命中、``(u1, v1)`` 命中，返回穿越点。

    射线深度恒为 ``z``，命中条件即该像素深度 ``< z``；二分到屏幕宽度
    ≤0.25 像素。
    """
    lo = 0.0
    hi = 1.0
    lo_u = u0
    lo_v = v0
    hi_u = u1
    hi_v = v1
    for _ in range(48):
        if (
            abs(hi_u - lo_u) * width <= 0.25
            and abs(hi_v - lo_v) * height <= 0.25
        ):
            break
        mid = (lo + hi) * 0.5
        u = u0 + (u1 - u0) * mid
        v = v0 + (v1 - v0) * mid
        col = int(u * width)
        if col >= width:
            col = width - 1
        elif col < 0:
            col = 0
        row = int(v * height)
        if row >= height:
            row = height - 1
        elif row < 0:
            row = 0
        if depth[row][col] < z:
            hi = mid
            hi_u = u
            hi_v = v
        else:
            lo = mid
            lo_u = u
            lo_v = v
    return (hi_u, hi_v)


def _refine_interval(sample, prev, cur, prev_t, cur_t):
    """在命中区间 ``[prev_t, cur_t]`` 内二分，返回 ``((u, v), t)``。"""
    lo = prev_t
    hi = cur_t
    lo_u, lo_v, _ = prev
    hi_u, hi_v, _ = cur
    for _ in range(64):
        mid = (lo + hi) * 0.5
        if hi - lo == 0.0:
            break
        mid_sample = sample(mid)
        if mid_sample is None:
            break
        mid_u, mid_v, mid_g = mid_sample
        if mid_g > 0.0:
            hi = mid
            hi_u = mid_u
            hi_v = mid_v
            if hi - lo <= 1e-12:
                break
        else:
            lo = mid
            lo_u = mid_u
            lo_v = mid_v
        if abs(hi_u - lo_u) <= 1e-9 and abs(hi_v - lo_v) <= 1e-9:
            break
    return ((hi_u, hi_v), hi)
