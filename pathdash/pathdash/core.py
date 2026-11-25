"""pathdash 核心 API：按弧长对折线进行虚线化。

设计要点：
- 完全以弧长参数 s ∈ [0, L] 驱动状态机，虚线段/间隔的长度直接取
  名义值 ``on`` / ``off``（仅末端被路径终点裁剪），不累积浮点漂移；
- 切分点通过同一个 ``_point_at`` 反查得到，保证相邻段在同一弧长处
  取到逐位相同的坐标，端点天然重合；
- 类本身不持有任何可变状态，``dash`` / ``length`` 是纯函数，
  多线程并发调用互不干扰。
"""

import math
from bisect import bisect_left, bisect_right

__all__ = ["PathDash"]


class PathDash:
    # ------------------------------------------------------------------ #
    # 校验
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_points(points, min_points=2):
        """校验并返回点序列；非有限坐标/点数不足一律 ValueError。"""
        try:
            n = len(points)
        except TypeError as exc:
            raise ValueError("points 必须是点序列") from exc
        if n < min_points:
            raise ValueError("至少需要 %d 个点" % min_points)
        cleaned = []
        for p in points:
            try:
                x, y = p
            except (TypeError, ValueError) as exc:
                raise ValueError("每个点必须是 (x, y) 数值对") from exc
            # math.isfinite 对非数值抛 TypeError，对 nan/inf 返回 False
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                raise ValueError("点坐标必须是实数")
            xf = float(x)
            yf = float(y)
            if not math.isfinite(xf) or not math.isfinite(yf):
                raise ValueError("点坐标必须有限（不能是 nan/inf）")
            cleaned.append((xf, yf))
        return cleaned

    @staticmethod
    def _cumulative(points):
        """返回累积弧长数组 cum 与总弧长 L。"""
        cum = [0.0]
        total = 0.0
        for i in range(1, len(points)):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            total += math.hypot(x1 - x0, y1 - y0)
            cum.append(total)
        return cum, total

    # ------------------------------------------------------------------ #
    # 弧长 -> 坐标
    # ------------------------------------------------------------------ #
    @staticmethod
    def _point_at(points, cum, s):
        """按弧长 s 反查折线上的点（线性插值）。

        s 恰好落在顶点上时返回原始顶点对象，保证不同段在同一弧长处
        取到完全相同的坐标。
        """
        if s <= 0.0:
            return points[0]
        if s >= cum[-1]:
            return points[-1]
        k = bisect_left(cum, s)
        if cum[k] == s:
            return points[k]
        i = k - 1  # s 位于边 (i, i+1) 内
        d = cum[i + 1] - cum[i]
        if d == 0.0:  # 零长边退化保护
            return points[i]
        t = (s - cum[i]) / d
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)

    @staticmethod
    def _build_segment(points, cum, a, b):
        """构造弧长区间 [a, b] 对应的虚线段点序列。"""
        out = [PathDash._point_at(points, cum, a)]
        # 严格落在 (a, b) 内部的原始顶点
        k0 = bisect_right(cum, a)
        k1 = bisect_left(cum, b)
        for k in range(k0, k1):
            p = points[k]
            if p != out[-1]:  # 跳过零长边产生的重复点
                out.append(p)
        end = PathDash._point_at(points, cum, b)
        if end != out[-1]:
            out.append(end)
        return out

    # ------------------------------------------------------------------ #
    # 公开 API
    # ------------------------------------------------------------------ #
    def length(self, points):
        """总弧长，返回 float。

        支持两种输入：
        - 折线：``[(x, y), ...]``，至少 2 个点；
        - 段列表（``dash`` 的返回结构）：``[[(x, y), ...], ...]``，
          返回各段弧长之和。
        """
        try:
            first = points[0]
        except (TypeError, IndexError) as exc:
            raise ValueError("points 必须是非空点序列") from exc

        # 段列表：首元素本身是点序列（list/tuple），而非 (x, y) 坐标对
        if isinstance(first, (list, tuple)) and not (
            len(first) == 2
            and all(isinstance(c, (int, float)) for c in first)
        ):
            total = 0.0
            for seg in points:
                pts = self._validate_points(seg)
                for i in range(1, len(pts)):
                    x0, y0 = pts[i - 1]
                    x1, y1 = pts[i]
                    total += math.hypot(x1 - x0, y1 - y0)
            return float(total)

        pts = self._validate_points(points)
        total = 0.0
        for i in range(1, len(pts)):
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            total += math.hypot(x1 - x0, y1 - y0)
        return float(total)

    def dash(self, points, on, off, phase=0.0):
        """按弧长把折线切分为虚线。

        参数:
            points: ``[(x, y), ...]`` 折线顶点，至少 2 个点且坐标有限。
            on: 单段虚线实长，必须为正的有限实数（为 0 即零周期，报错）。
            off: 间隔长度，必须为非负有限实数。
            phase: 起始相位（弧长单位），可取任意有限实数，自动按周期
                取模到 [0, on+off)。

        返回:
            list[list[tuple[float, float]]]，每个元素是一条虚线段的
            点序列；路径总弧长为 0 时返回空列表。

        并发:
            不读写任何实例/全局可变状态，可安全并发调用。
        """
        pts = self._validate_points(points)

        if not isinstance(on, (int, float)) or not math.isfinite(on) or on <= 0:
            raise ValueError("on 必须为正的有限实数（周期不能为 0）")
        if not isinstance(off, (int, float)) or not math.isfinite(off) or off < 0:
            raise ValueError("off 必须为非负有限实数")
        if not isinstance(phase, (int, float)) or not math.isfinite(phase):
            raise ValueError("phase 必须为有限实数")
        on = float(on)
        off = float(off)
        phase = float(phase)

        cum, total = self._cumulative(pts)
        if total <= 0.0:
            return []

        period = on + off
        # 任意相位（含负数、超过周期）归一化到 [0, period)
        q = math.fmod(phase, period)
        if q < 0.0:
            q += period

        is_on = q < on
        remaining = (on - q) if is_on else (period - q)

        intervals = []
        s = 0.0
        start = 0.0 if is_on else None
        rest = total  # 剩余未消费弧长

        while rest > 0.0:
            if remaining >= rest:
                step = rest
                end = total  # 末端直接钉在总弧长上，避免 s+step 的舍入
            else:
                step = remaining
                end = s + step

            if is_on:
                if start is None:
                    start = s
            elif start is not None:
                intervals.append((start, s))
                start = None

            s = end
            rest -= step
            remaining -= step
            if remaining == 0.0:
                is_on = not is_on
                remaining = on if is_on else off

        if start is not None:
            intervals.append((start, s))

        return [
            self._build_segment(pts, cum, a, b)
            for a, b in intervals
            if b > a
        ]
