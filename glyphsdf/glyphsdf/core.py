"""glyphsdf 核心 API。

基于点到轮廓线段的精确欧氏距离生成有向距离场（SDF）：

* 距离符号由奇偶规则（even-odd）判定，内部为正、外部为负；逐行扫描线
  算法求内外，复杂度与每行活跃边数成正比；
* 归一化约定：轮廓处为 ``0.5``，沿法线每像素变化 ``1 / spread``，
  即值 ``v`` 对应有向距离（像素）``(v - 0.5) * spread``，结果裁剪到
  ``[0.0, 1.0]``；
* 距离计算采用与像素对齐的均匀网格分桶，查询半径按 spread 自适应，
  仅对可能落在未裁剪带（|d| < spread）内的线段求值；对无法从桶几何
  排除的像素走全量精确兜底，因此可重建带内的结果与逐线段暴力计算
  逐值一致；
* 覆盖率取双线性重建距离场在每个像素单元内的多点盒式积分，再经距离
  斜坡映射为覆盖率，对轮廓面积具有二阶精度。

全部计算基于分块向量化 numpy，对象不持有可变中间状态，多线程并发
调用安全。
"""

import math

import numpy as np

__all__ = ["GlyphSdf", "SDFError"]


class SDFError(ValueError):
    """非法尺寸、扩散半径或轮廓输入时抛出（``ValueError`` 子类）。"""


def _is_real_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


class GlyphSdf:
    """有向距离场光栅化器。

    Parameters
    ----------
    size:
        输出网格边长（正整数），输出为 ``size x size``。
    spread:
        距离场扩散半径（像素，正数）。值 ``0`` / ``1`` 对应距轮廓
        ``-spread`` / ``+spread`` 像素。
    """

    # 线段分块大小：控制单次向量化运算的内存占用。
    _SEG_BLOCK = 256
    # 空间桶边长（像素，整数对齐）。
    _BUCKET_PX = 32
    # 覆盖率像素内积分的每维采样数。
    _SUBDIV = 4

    def __init__(self, size, spread):
        if not _is_real_number(size):
            raise SDFError("size 必须为正整数")
        if isinstance(size, float) and not size.is_integer():
            raise SDFError("size 必须为正整数")
        size = int(size)
        if size <= 0:
            raise SDFError("size 必须为正整数")
        if not _is_real_number(spread) or not math.isfinite(float(spread)):
            raise SDFError("spread 必须为有限正数")
        spread = float(spread)
        if spread <= 0.0:
            raise SDFError("spread 必须为正数")

        self.size = size
        self.spread = spread

    # ------------------------------------------------------------------
    # 输入校验
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_contours(contours):
        if not isinstance(contours, (list, tuple)) or len(contours) == 0:
            raise SDFError("contours 必须为非空轮廓序列")

        total_points = 0
        for contour in contours:
            if not isinstance(contour, (list, tuple)) or len(contour) == 0:
                raise SDFError("每条轮廓必须为非空点序列")
            for point in contour:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise SDFError("轮廓点必须是长度为 2 的序列 (x, y)")
                x, y = point
                if not _is_real_number(x) or not _is_real_number(y):
                    raise SDFError("轮廓点坐标必须为数值")
                if not (math.isfinite(float(x)) and math.isfinite(float(y))):
                    raise SDFError("轮廓点坐标必须有限（不接受 NaN/Inf）")
            total_points += len(contour)

        if total_points < 2:
            raise SDFError("轮廓至少需要两个点才能构成几何形状")

    # ------------------------------------------------------------------
    # 光栅化
    # ------------------------------------------------------------------
    def rasterize(self, contours):
        """生成 ``size x size`` 的归一化 SDF（``list[list[float]]``）。

        轮廓坐标约定在单位正方形 ``[0, 1] x [0, 1]`` 内，单位坐标
        ``(x, y)`` 映射到像素中心 ``(x * size - 0.5, y * size - 0.5)``。
        不修改调用方传入的任何对象，重复调用结果确定。
        """
        self._validate_contours(contours)

        size = self.size
        spread = self.spread

        # 复制成独立数组，确保不持有、不修改调用方数据。
        point_chunks = []
        counts = []
        for contour in contours:
            pts = np.asarray(
                [(float(pt[0]), float(pt[1])) for pt in contour],
                dtype=np.float64,
            )
            point_chunks.append(pts)
            counts.append(len(pts))
        points = np.concatenate(point_chunks, axis=0) * size
        del point_chunks

        # 闭合轮廓：每条轮廓末点接回首点。
        start_idx = []
        end_idx = []
        offset = 0
        for n in counts:
            idx = np.arange(n) + offset
            start_idx.append(idx)
            end_idx.append(np.roll(idx, -1))
            offset += n
        s_idx = np.concatenate(start_idx)
        e_idx = np.concatenate(end_idx)
        p = points[s_idx]
        r = points[e_idx]
        del start_idx, end_idx, s_idx, e_idx, points

        seg_dir = r - p
        seg_len2 = np.einsum("ij,ij->i", seg_dir, seg_dir)
        degenerate = seg_len2 == 0.0

        ax_all = p[:, 0]
        ay_all = p[:, 1]
        bx_all = r[:, 0]
        by_all = r[:, 1]

        # 像素中心坐标（像素单位），axis[i] 即第 i 个像素中心。
        axis = np.arange(size, dtype=np.float64) + 0.5

        # ================================================================
        # 1) 内外判定：逐行扫描线 + 奇偶规则。
        #    每条非水平线段只在其 y 跨度内的行活跃；半开区间
        #    [y_lo, y_hi) 保证顶点只被计一次。
        # ================================================================
        horizontal = seg_dir[:, 1] == 0.0
        y_lo = np.minimum(ay_all, by_all)
        y_hi = np.maximum(ay_all, by_all)
        x_at_lo = np.where(ay_all < by_all, ax_all, bx_all)
        safe_dy = np.where(horizontal, 1.0, by_all - ay_all)
        slope_x = (bx_all - ax_all) / safe_dy
        active_idx = np.nonzero(~horizontal)[0]

        inside = np.zeros((size, size), dtype=bool)
        for row in range(size):
            y = axis[row]
            m = active_idx[(y_lo[active_idx] <= y) & (y < y_hi[active_idx])]
            if m.size == 0:
                continue
            xint = x_at_lo[m] + (y - y_lo[m]) * slope_x[m]
            xint.sort()
            # 严格位于像素中心右侧的交点数决定奇偶。
            greater = xint.size - np.searchsorted(xint, axis, side="right")
            inside[row, :] = (greater & 1) == 1

        # ================================================================
        # 2) 距离：与像素对齐的均匀网格分桶，仅保证 ±spread 带内精确。
        #    SDF 在 |d| >= spread 处裁剪，场外具体值不影响输出。桶边长
        #    B，查询半径 K = ceil(spread / B) + 1：查询区物理边界到桶内
        #    任一像素的距离至少为 spread，故区外线段距离必然 >= spread、
        #    必被裁剪，无需参与计算。仅当查询区在某方向贴图像边界、或
        #    存在 d_edge < spread 的歧义像素时才走全量精确兜底。
        # ================================================================
        bucket_px = self._BUCKET_PX
        n_buckets = max(1, (size + bucket_px - 1) // bucket_px)
        query_r = max(1, math.ceil(spread / bucket_px) + 1)

        def _bucket_of(coord):
            return np.clip(
                np.floor(coord).astype(np.int64) // bucket_px,
                0, n_buckets - 1,
            )

        seg_bx0 = np.minimum(_bucket_of(ax_all), _bucket_of(bx_all))
        seg_bx1 = np.maximum(_bucket_of(ax_all), _bucket_of(bx_all))
        seg_by0 = np.minimum(_bucket_of(ay_all), _bucket_of(by_all))
        seg_by1 = np.maximum(_bucket_of(ay_all), _bucket_of(by_all))

        bucket_segs = [[] for _ in range(n_buckets * n_buckets)]
        for s in range(len(p)):
            for bv in range(seg_by0[s], seg_by1[s] + 1):
                base = bv * n_buckets
                for bh in range(seg_bx0[s], seg_bx1[s] + 1):
                    bucket_segs[base + bh].append(s)

        best_d = np.full((size, size), np.inf, dtype=np.float64)
        sb = self._SEG_BLOCK

        def _dist_to_segs(qx_col, qy_row, ids):
            """对给定像素网格与线段 id 计算每像素最小欧氏距离。"""
            nrows = qy_row.size
            ncols = qx_col.size
            out = np.full((nrows, ncols), np.inf, dtype=np.float64)
            qx = qx_col[None, :, None]
            qy = qy_row[:, None, None]
            for j0 in range(0, ids.size, sb):
                jj = ids[j0:j0 + sb]
                axx = ax_all[jj][None, None, :]
                ayy = ay_all[jj][None, None, :]
                dxx = seg_dir[jj, 0][None, None, :]
                dyy = seg_dir[jj, 1][None, None, :]
                len2 = seg_len2[jj][None, None, :]
                deg = degenerate[jj][None, None, :]

                vx = qx - axx
                vy = qy - ayy
                safe_len2 = np.where(deg, 1.0, len2)
                t = (vx * dxx + vy * dyy) / safe_len2
                np.clip(t, 0.0, 1.0, out=t)
                cxp = vx - t * dxx
                cyp = vy - t * dyy
                dist2 = cxp * cxp + cyp * cyp
                point_dist2 = vx * vx + vy * vy
                dist2 = np.where(deg, point_dist2, dist2)
                np.minimum(out, np.sqrt(np.min(dist2, axis=2)), out=out)
            return out

        all_ids = np.arange(len(p), dtype=np.int64)

        for bv in range(n_buckets):
            for bh in range(n_buckets):
                r0 = bv * bucket_px
                r1 = min(size, r0 + bucket_px)
                c0 = bh * bucket_px
                c1 = min(size, c0 + bucket_px)

                # 收集查询半径内桶的线段。
                ids = []
                for ny in range(max(0, bv - query_r),
                                min(n_buckets, bv + query_r + 1)):
                    for nx in range(max(0, bh - query_r),
                                    min(n_buckets, bh + query_r + 1)):
                        ids.extend(bucket_segs[ny * n_buckets + nx])

                # 到查询区边界的距离：仅当该方向外仍有桶时才约束，
                # 贴图像边界的方向取 inf（区外不存在线段）。
                edge_left = ((bh - query_r) * bucket_px
                             if bh - query_r >= 0 else -np.inf)
                edge_right = ((bh + query_r + 1) * bucket_px
                              if bh + query_r + 1 <= n_buckets else np.inf)
                edge_top = ((bv - query_r) * bucket_px
                            if bv - query_r >= 0 else -np.inf)
                edge_bot = ((bv + query_r + 1) * bucket_px
                            if bv + query_r + 1 <= n_buckets else np.inf)
                qx_col = axis[c0:c1]
                qy_row = axis[r0:r1]
                d_edge = np.minimum(
                    np.minimum(qx_col[None, :] - edge_left,
                               edge_right - qx_col[None, :]),
                    np.minimum(qy_row[:, None] - edge_top,
                               edge_bot - qy_row[:, None]),
                )

                if not ids:
                    # 区外线段距离 >= d_edge；>=spread 者必被裁剪。
                    saturated = d_edge >= spread
                    result = np.where(saturated, spread + 1.0, np.inf)
                    best_d[r0:r1, c0:c1] = result
                    continue

                seg_ids = np.unique(np.asarray(ids, dtype=np.int64))
                dmin = _dist_to_segs(qx_col, qy_row, seg_ids)
                # dmin <= d_edge：最近线段确在查询区内，结果精确。
                exact = dmin <= d_edge
                far = d_edge >= spread
                result = np.where(
                    exact, dmin,
                    np.where(far, spread + 1.0, np.inf),
                )
                best_d[r0:r1, c0:c1] = result

        # 安全兜底：仅处理无法证明“区外线段必 >=spread”的像素。
        missing = np.argwhere(np.isinf(best_d))
        if missing.size:
            rows = np.unique(missing[:, 0])
            for row in rows:
                cols = missing[missing[:, 0] == row, 1]
                block = _dist_to_segs(axis[cols], axis[row:row + 1], all_ids)
                best_d[row, cols] = block[0]

        signed_d = best_d
        signed_d[~inside] = -signed_d[~inside]

        # 归一化：轮廓 0.5，每像素 1/spread，裁剪到 [0, 1]。
        field = 0.5 + signed_d / spread
        np.clip(field, 0.0, 1.0, out=field)

        return field.tolist()

    # ------------------------------------------------------------------
    # 覆盖率
    # ------------------------------------------------------------------
    def coverage(self, sdf):
        """将 SDF 转为抗锯齿覆盖率并返回全图平均覆盖率（``float``）。

        对每个像素，在其单元内对双线性重建的有向距离场做多点盒式
        积分，逐点经距离斜坡 ``clamp(0.5 + d / spread, 0, 1)`` 映射为
        覆盖率，再对全图取均值。
        """
        if not isinstance(sdf, (list, tuple)) or len(sdf) == 0:
            raise SDFError("sdf 必须为非空二维序列")
        try:
            arr = np.asarray(sdf, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise SDFError("sdf 必须为二维数值序列") from exc
        if arr.ndim != 2 or arr.size == 0:
            raise SDFError("sdf 必须为非空二维序列")
        if not np.isfinite(arr).all():
            raise SDFError("sdf 含非有限值（NaN/Inf）")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise SDFError("sdf 值必须位于 [0.0, 1.0]")

        # 像素中心有向距离（像素单位，轮廓为 0）。
        d = (arr - 0.5) * self.spread

        k = self._SUBDIV
        local = (np.arange(k, dtype=np.float64) + 0.5) / k - 0.5

        # 边缘外采用边界值复制，避免越界。
        D = np.pad(d, 1, mode="edge")
        H, W = d.shape

        accum = np.zeros((H, W), dtype=np.float64)
        for dy in local:
            base_y = int(math.floor(dy))
            wy = dy - base_y
            y0 = np.arange(H) + base_y + 1
            y1 = y0 + 1
            for dx in local:
                base_x = int(math.floor(dx))
                wx = dx - base_x
                x0 = np.arange(W) + base_x + 1
                x1 = x0 + 1

                f00 = D[np.ix_(y0, x0)]
                f01 = D[np.ix_(y0, x1)]
                f10 = D[np.ix_(y1, x0)]
                f11 = D[np.ix_(y1, x1)]
                top = f00 * (1.0 - wx) + f01 * wx
                bot = f10 * (1.0 - wx) + f11 * wx
                sample = top * (1.0 - wy) + bot * wy

                accum += np.clip(0.5 + sample / self.spread, 0.0, 1.0)

        accum /= k * k
        return float(accum.mean())
