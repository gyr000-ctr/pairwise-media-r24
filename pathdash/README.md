# pathdash

## 题面（可直接粘贴）

子目录：pathdash。路径虚线化。约束：按弧长切分虚线与间隔（长度误差 ≤1e-9）；首尾相位可偏移；相邻段端点重合（≤1e-9）；总虚线长度与 (总长/周期)*实长 一致（≤1e-6）；零周期报错；1 万段不超过 400 ms；并发安全。验收：cd pathdash && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd pathdash && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
