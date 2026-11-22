# 深度伪造攻击扩展：VQGAN、SimSwap-224、InSwapper-128

本文档汇总 30k online checkpoint 下的三种攻击。所有正式结果位于 F 盘项目目录，不使用旧 128 自换脸数据。

## 1. 人脸示例的正确读法

![Three attacks on the same protected images](deepfake_attack_comparison.png)

每行从左到右为：原图、带水印图、身份源、SimSwap-224 输出、InSwapper-128 输出。

第三列 `Identity source` 是换脸时提供身份的另一张图，不是攻击输出。换脸只替换人脸区域，因此头发、衣服和背景通常仍来自原图。此前看起来像“上下错位”，主要是因为原图没有列标题。

## 2. 攻击强度与真实身份替换

| 指标 | SimSwap-224 | InSwapper-128 |
|---|---:|---:|
| 水印图到攻击图 PSNR | 21.2733 dB | 24.4123 dB |
| SSIM | 0.6026 | 0.7456 |
| LPIPS | 0.1940 | 0.1314 |
| 攻击图与身份源余弦 | 0.7691 | 0.7968 |
| 攻击图与水印身份余弦 | 0.0836 | 0.1279 |
| 身份 margin | 0.6856 | 0.6689 |
| 身份替换成功 | 100/100 | 100/100 |

![InSwapper attack strength](inswapper128_attack_strength.png)

解释边界：InSwapper-128 的像素改动更小、视觉上更温和，但身份源余弦更高，仍完成了真实身份替换。不能据此说 InSwapper 比 SimSwap 更弱，也不能说它对水印的威胁更小。

## 3. 单图水印解码

![Single-image attack comparison](single_image_attack_comparison.png)

### 100 用户 x 5 图（500 张）

| 攻击 | BER | 30-bit 全对 | 平均 Hamming | 最大 Hamming |
|---|---:|---:|---:|---:|
| VQGAN | 0.0800% | 97.6% | 0.024 | 1 |
| SimSwap-224 | 0.1200% | 97.2% | 0.036 | 2 |
| InSwapper-128 | 0.0733% | 98.0% | 0.022 | 2 |

### 500 用户 x 1 图（500 张）

| 攻击 | BER | 30-bit 全对 | 平均 Hamming | 最大 Hamming | 500 候选 top-1 |
|---|---:|---:|---:|---:|---:|
| VQGAN | 0.0533% | 98.4% | 0.016 | 1 | 100% |
| SimSwap-224 | 0.1733% | 95.0% | 0.052 | 2 | 100% |
| InSwapper-128 | 0.0600% | 98.4% | 0.018 | 2 | 100% |

500 用户实验是“一用户一图”，不是 500 用户 x 5 图的多图实验。

## 4. 多图 LLR 聚合

![Multi-image LLR under three attacks](multimage_llr_curve.png)

100 用户 x 5 图，枚举全部非空子集：

| m | VQGAN z | SimSwap-224 z | InSwapper-128 z | VQGAN margin | SimSwap-224 margin | InSwapper-128 margin |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.399 | 5.391 | 5.411 | 3.817 | 3.817 | 3.856 |
| 2 | 7.636 | 7.623 | 7.652 | 5.455 | 5.455 | 5.491 |
| 3 | 9.352 | 9.336 | 9.371 | 6.707 | 6.710 | 6.740 |
| 4 | 10.798 | 10.780 | 10.821 | 7.763 | 7.766 | 7.790 |
| 5 | 12.073 | 12.052 | 12.098 | 8.689 | 8.695 | 8.716 |

三种攻击下所有用户、所有非空子集的 top-1 都是 100%。曲线几乎重合并不能用绘图技巧强行拉开；可辩护的结论是“三种攻击下都能稳定追踪”，不是“某一种攻击显著更弱”。

![Evidence and threshold comparison](attack_evidence_comparison_three_models.png)

## 5. 500 候选用户规模

![Candidate-pool scaling](codebook_scaling_three_attacks.png)

| 候选用户 | VQGAN 平均 margin | SimSwap-224 平均 margin | InSwapper-128 平均 margin |
|---:|---:|---:|---:|
| 50 | 3.724 | 3.713 | 3.763 |
| 100 | 3.663 | 3.651 | 3.707 |
| 200 | 3.613 | 3.591 | 3.657 |
| 300 | 3.585 | 3.563 | 3.635 |
| 500 | 3.555 | 3.532 | 3.606 |

50 到 500 候选用户下，三种攻击的 top-1 都是 100%，Z=3.0 误指控均为 0%。这只是当前 500 用户样本池上的经验结果，不等同于大规模开放集保证。

## 6. 可写与不可写

可以写：

- DiffMark 水印在 VQGAN、SimSwap-224、InSwapper-128 三种攻击下单图 BER 都低于 0.18%。
- 500 用户一图一用户设置下，三种攻击的 top-1 均为 100%，InSwapper 候选池 margin 仍为正。
- SimSwap 和 InSwapper 都完成了真实身份替换，不是单纯颜色扰动。

不要写：

- “InSwapper 比 SimSwap 更强/更弱”：像素指标和身份指标方向不一致。
- “三种攻击差异显著”：当前差异很小，没有做显著性检验。
- “500 用户 x 5 图”：500 用户实验是一图一用户。
- “身份源图是换脸结果”：它只是输入身份。

## 7. 主要文件

- `results/user_fingerprint_eval_30k_min11_inswapper128_probs.json`
- `results/inswapper128_multimage_llr_summary.json`
- `results/inswapper128_multimage_llr_summary.txt`
- `results/inswapper128_attack_strength.json`
- `results/user_fingerprint_eval_30k_500users_bch_inswapper128_probs.json`
- `results/codebook_scaling_500users_inswapper128.json`
- `docs/inswapper128_attack_strength.md`
- `docs/codebook_scaling_500users_inswapper128.md`
- `docs/single_image_attack_comparison.md`
- `docs/multimage_llr_three_attacks.md`
- `docs/codebook_scaling_three_attacks.md`
