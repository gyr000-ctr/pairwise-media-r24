# lensflare

## 题面（可直接粘贴）

子目录：lensflare。屏幕空间镜头光晕。约束：光晕位置沿光源到屏幕中心的连线且按深度遮挡衰减（与深度测试一致）；遮挡比例用采样估计，误差 ≤5%；光晕强度随距离单调；无遮挡时不衰减；10 万次不超过 300 ms；非法参数可区分报错。验收：cd lensflare && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd lensflare && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
