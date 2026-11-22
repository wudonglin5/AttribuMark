# Mosaic 合谋攻击结果

## 攻击定义

- 同一张封面图分别嵌入 k 个用户的 30-bit 指纹。
- 将 k 张带指纹图切成网格，每个格子来自一个随机合谋者，拼成一张 mosaic 图。
- k=4 时为 2x2 四宫格；k=2 为左右两半；k=3 为三列；k=5 为 2x3 网格（6 个格子，其中一个合谋者重复占一格）。

## k=2..5，100 组/设置，min-11 码本

输出：

- `results/user_collusion_eval_30k_min11_mosaic_k2_100groups.json`
- `results/user_collusion_eval_30k_min11_mosaic_k3_100groups.json`
- `results/user_collusion_eval_30k_min11_mosaic_k4_100groups.json`
- `results/user_collusion_eval_30k_min11_mosaic_k5_100groups.json`
- 汇总：`results/mosaic_curve_summary.json`
- 图：`docs/mosaic_curve.png`

全局软 LLR 检测（top-1 不做拒识；Z=3.0 为保守工作点）：

| k | 条件 | top-1 | Z=3.0 检出 | Z=3.0 误指控 | Z=3.0 拒识 |
|---:|---|---:|---:|---:|---:|
| 2 | clean | 96% | 82% | 0% | 18% |
| 2 | VQGAN | 96% | 82% | 2% | 16% |
| 3 | clean | 83% | 47% | 5% | 48% |
| 3 | VQGAN | 80% | 44% | 5% | 51% |
| 4 | clean | 72% | 25% | 3% | 72% |
| 4 | VQGAN | 65% | 25% | 3% | 72% |
| 5 | clean | 65% | 27% | 3% | 70% |
| 5 | VQGAN | 66% | 27% | 3% | 70% |

结论：

- mosaic 是目前最强的合谋攻击；k=4 时硬判决 top-1 只有 61%/56%，软 LLR 提升到 72%/65%。
- k=2 时仍然很强：top-1 96%，Z=3.0 检出 82%；k>=3 后显著下降。
- k=4 时 Z=3.0 检出降到 25%，但误指控仍控制在 3%；大部分样本被保守拒识为“疑似合谋/无法溯源”。
- 按 64x64 格子单独解码不可行：裁剪破坏了解码器的空间同步，top-1 反而下降。该负结果不作为主方法。
- 这组结果适合作为论文的“最强攻击/失败安全”实验：检测器不强行指控，而是输出拒识；这比高误指控更符合取证要求。

## 下一步

- 主论文把 mosaic 放在攻击鲁棒性小节，不把它写成方法优势。
- 如需提升 mosaic 检出，优先做“同一用户多张图证据聚合”，而不是继续切图。
