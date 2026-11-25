"""lensflare 核心 API（初始未实现）。"""


class LensFlare:
    def __init__(self, samples):
        """构造"""
        raise NotImplementedError

    def visible_ratio(self, depth, light):
        """遮挡比例"""
        raise NotImplementedError

    def ghosts(self, light, n):
        """光晕位置"""
        raise NotImplementedError
