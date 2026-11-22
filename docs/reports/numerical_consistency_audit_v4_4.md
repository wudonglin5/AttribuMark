# 全文数值一致性审查报告（v4.4）

- 审查日期：2026-11-03
- 审查对象：`docs/DiffMark_UserLLR_manuscript_zh_v4_4.docx`
- 对照底稿：`docs/DiffMark_UserLLR_manuscript_zh_v4_3.docx`
- 结论：表 1–8、表 10 共 523 个数据单元格与原始结果文件逐格一致，差异为 0；表 9 的三项消融与汇总结果一致。v4.4 没有修改任何实验数值，只修正了 2 处文字表述、1 处图 2 技术示意和 1 处图 1 图注措辞。

## 一、数据来源

- 表 1：`results/user_fingerprint_eval_30k_min11_probs.json`、`results/user_fingerprint_eval_30k_min11_simswap224_probs.json`、`results/user_fingerprint_eval_30k_min11_inswapper128_probs.json`
- 表 2：`results/multimage_llr_summary.json`、`results/simswap224_multimage_llr_summary.json`、`results/inswapper128_multimage_llr_summary.json`
- 表 3：三个 500 用户概率文件及 `results/codebook_scaling_500users*.json`
- 表 4：`results/vqgan_collusion_soft_detector_mean_100groups.json`、`results/simswap224_collusion_mix_summary.json`
- 表 5：`results/simswap224_collusion_mix_summary.json`
- 表 6：`results/inswapper128_collusion_by_k_{mean,median,mosaic}_100groups.json`
- 表 7：`results/simswap224_attack_strength.json`、`results/inswapper128_attack_strength.json`
- 表 8：`results/collusion_threshold_cv.json`
- 表 9：`results/multimage_llr_summary.json`、SimSwap-224 合谋记录、随机码本与最小距离码本汇总文件
- 表 10：`results/standard_distortions_100.json`

## 二、表级核对结果

| 表 | 规模 | 核对方式 | 结果 |
|---|---:|---|---|
| 表 1 | 4×7 | 原始逐图统计重算 | 28/28 一致 |
| 表 2 | 6×9 | 三类攻击的 LLR/概率平均汇总重算 | 54/54 一致 |
| 表 3 | 4×9 | 500 用户逐图统计与候选池重采样重算 | 36/36 一致 |
| 表 4 | 5×10 | VQGAN/SimSwap mean 合谋条件重算 | 50/50 一致 |
| 表 5 | 13×6 | SimSwap-224 三种混合逐 k 重算 | 78/78 一致 |
| 表 6 | 13×6 | InSwapper-128 三种混合逐 k 重算 | 78/78 一致 |
| 表 7 | 8×3 | 两张攻击强度 JSON 的均值重算 | 24/24 一致 |
| 表 8 | 17×7 | 五折阈值校准 JSON 重算 | 119/119 一致 |
| 表 9 | 5×3 | 三项消融逐项核对 | 全部一致 |
| 表 10 | 8×7 | 七种常规失真条件重算 | 56/56 一致 |

合计：表 1–8、表 10 共 **523 个数据单元格，0 处差异**。

## 三、发现并修正的问题

1. **5.4 节对 median 的表述过强。** 原文写“median 在 k=4/5 下明显优于 mean”，但表 5 中 k=4 仅相差 2 个百分点、k=5 仅相差 1 个百分点，不能写成“明显优于”。v4.4 改为“mean 与 median 的差异仅为 1–2 个百分点，处于同一量级”。
2. **图 2 的训练攻击链路容易误读。** 原图把 VQGAN、SimSwap 和 InSwapper 放在同一个“Attack channel”中，图注也写成三者在训练阶段联合优化；实际训练只使用冻结 VQGAN，SimSwap-224、InSwapper-128 与合谋是推理后的评估攻击。v4.4 重绘图 2 并改写图注，明确区分“训练期 VQGAN”与“评估期深伪/合谋”。
3. **图 1 图注中的 guidance 措辞过硬。** 主实验关闭 Deepfake-resistant guidance，因此将“带 guidance 的 DDIM 推理”改为“可选 guidance 的 DDIM 推理”，避免与方法设置冲突。

## 四、正文数值抽查

已逐句核对摘要、5.1–5.9、讨论和结论中的关键数字，包括：

- 干净水印 33.8561 dB、0.9258 SSIM、0.03921 LPIPS、BER 0.127%；
- 三种攻击单图 BER 均低于 0.13%，最大 Hamming 错误不超过 2 bit；
- 多图 LLR 的 z 从约 5.4 增长到约 12.1，概率平均停留在约 5.3；
- 500 用户一图一用户 top-1 均为 100%，Z=3 误指控均为 0%；
- 表 4–6 的 Z=3 误指控均不超过 4%；
- JPEG Q50 的 BER 3.73%、top-1 100%；90% 中心裁剪 BER 38.63%、top-1 12%、拒识 87%；
- InSwapper-128 的 mean/median top-1 为 97%–100%，mosaic 为 67%–93%。

以上均与对应原始结果文件一致。

## 五、结论

v4.4 可以作为当前数值版本的冻结稿。后续如果更换 checkpoint、改变码本、补充新攻击或修改阈值，必须重新生成本报告；不能只替换单张表或单个数字。