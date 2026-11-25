"""etc1 核心 API（初始未实现）。"""


class Etc1:
    def encode(self, block):
        """编码"""
        raise NotImplementedError

    def decode(self, data):
        """解码"""
        raise NotImplementedError

    def psnr(self, a, b):
        """PSNR"""
        raise NotImplementedError
