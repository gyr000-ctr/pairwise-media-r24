"""glyphsdf 核心 API（初始未实现）。"""


class GlyphSdf:
    def __init__(self, size, spread):
        """构造"""
        raise NotImplementedError

    def rasterize(self, contours):
        """生成 SDF"""
        raise NotImplementedError

    def coverage(self, sdf):
        """覆盖率"""
        raise NotImplementedError
