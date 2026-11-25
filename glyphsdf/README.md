# glyphsdf

## 题面（可直接粘贴）

子目录：glyphsdf。SDF 字形光栅化。约束：距离场符号与内外一致；等值线在 0.5 处对应轮廓（≤1 像素）；双线性重建与解析距离差 ≤0.02；抗锯齿覆盖率与解析面积差 ≤5%；非法尺寸可区分报错；100 万像素不超过 600 ms；并发安全。验收：cd glyphsdf && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd glyphsdf && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
