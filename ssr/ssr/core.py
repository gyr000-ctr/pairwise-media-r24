"""ssr 核心 API（初始未实现）。"""


class Ssr:
    def trace(self, depth, origin, dirn, params):
        """追踪"""
        raise NotImplementedError

    def refine(self, depth, a, b):
        """二分细化"""
        raise NotImplementedError
