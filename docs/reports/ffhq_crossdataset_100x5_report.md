# FFHQ 跨数据集验证（100 用户 × 5 图）

## 协议

- 模型：`results/2026-10-30-20-26-32-353028/model030000.pt`（online 30k checkpoint，未重训）
- 码本：`results/codebook_100x30_min11.json`，100 个用户码，30 bit，最小 Hamming 距离 11，平均 15.13
- 数据：CelebA-HQ `test_128_500` 与 FFHQ `test_128_500`，各取 100 用户 × 5 张不同人脸，共 500 张
- 采样：DDIM 100 步，128×128，batch 4，无 guidance
- 归因：同一用户 5 张图的解码概率做 LLR 聚合，m=1..5
- 指控规则：真实用户 z ≥ 3 且为最近用户时检出；最近用户不是真实用户时误指控；否则拒识

## 结果

| 数据集 | 攻击 | 逐图 BER | 逐图全对 | 平均/最大 Hamming | m=5 真实用户 z | Top-1@100 | Z=3 检出/误指控/拒识 |
|---|---|---:|---:|---:|---:|---:|---:|
| CelebA-HQ | VQGAN | 0.08% | 97.6% | 0.024 / 1 | 12.073 | 100% | 100/0/0% |
| CelebA-HQ | SimSwap-224 | 0.12% | 97.2% | 0.036 / 2 | 12.052 | 100% | 100/0/0% |
| CelebA-HQ | InSwapper-128 | 0.07% | 98.0% | 0.022 / 2 | 12.098 | 100% | 100/0/0% |
| FFHQ | Clean | 0.01% | 99.8% | 0.002 / 1 | 12.172 | 100% | 100/0/0% |
| FFHQ | VQGAN | 0.51% | 86.0% | 0.152 / 3 | 11.927 | 100% | 100/0/0% |
| FFHQ | SimSwap-224 | 0.10% | 97.2% | 0.030 / 2 | 12.045 | 100% | 100/0/0% |
| FFHQ | InSwapper-128 | 0.12% | 96.4% | 0.036 / 1 | 12.056 | 100% | 100/0/0% |

## 结论

1. FFHQ 上四种条件（Clean、VQGAN、SimSwap-224、InSwapper-128）的多图 Top-1 均为 100%，m=5 真实用户 z 均在 11.9 以上，Z=3 下检出 100%、误指控 0%、拒识 0%。
2. 跨数据集后性能没有崩塌，说明固定 30-bit 用户码与多图 LLR 聚合不是只对 CelebA-HQ 过拟合。
3. FFHQ 的 VQGAN 逐图 BER 为 0.51%，明显高于 CelebA-HQ 的 0.08%，逐图全对从 97.6% 降到 86.0%；但 5 图聚合后仍恢复为 100% Top-1。这是本文最值得报告的跨域现象：单图退化可以由多图证据补偿。
4. InSwapper-128 在 FFHQ 上逐图 BER 0.12%、全对 96.4%，与 SimSwap-224 接近；在 CelebA-HQ 上为 0.07%、98.0%。两种换脸模型都没有突破多图归因。
5. 运行时间：Clean 383 s、VQGAN 396 s、SimSwap-224 393 s、InSwapper-128 812 s（RTX 3050 Laptop 4 GB）。InSwapper 较慢，原因是额外的检测/识别 ONNX worker，不是扩散采样变慢。

## 写作边界

- 这是同一 checkpoint 的跨数据集测试，不是跨数据集重训；不能写成“跨域训练泛化”。
- FFHQ VQGAN 的单图全对率只有 86.0%，应在正文中如实报告，不能只写多图 100%。
- 该实验补上了“单一数据集”的主要缺口，但仍没有外部水印基线；不能据此声称全面优于现有方法。
- 几何攻击的失败结论不变：90% 中心裁剪即使给定真实裁剪比例，BER 仍约 36%–39%，Top-1 仅 10%–19%。FFHQ 结果不改变这一限制。

## 原始文件

- `results/ffhq_crossdataset_100x5_clean_probs.json`
- `results/ffhq_crossdataset_100x5_vqgan_probs.json`
- `results/ffhq_crossdataset_100x5_simswap224_probs.json`
- `results/ffhq_crossdataset_100x5_inswapper128_probs.json`
- 对应 `*_multimage_llr.json` / `.txt`
- 日志：`logs/ffhq_crossdataset_100x5_clean`、`..._vqgan`、`..._simswap224`、`..._inswapper128`
- 汇总：`docs/ffhq_crossdataset_100x5_summary.json`
