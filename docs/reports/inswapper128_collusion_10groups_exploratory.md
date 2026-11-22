# InSwapper-128 合谋探索：10 组

这是探索性小样本，不与正式 100 组合谋结果混用。

## 协议

- 100 个用户，30-bit min-Hamming-11 码本。
- 每个 k 只跑 10 组，k=2..5。
- 使用 `identity_offset=1`，换脸身份源和内容图不同。
- 三种混合：mean、median、mosaic。
- 混合后统一经过 InSwapper-128，再解码。

## 结果

![InSwapper collusion pilot](inswapper128_collusion_mix_10groups_exploratory.png)

| Mix | k | Clean top-1 | InSwapper top-1 | Z=3 detect | Z=3 false |
|---|---:|---:|---:|---:|---:|
| mean | 2 | 100% | 100% | 100% | 0% |
| mean | 3 | 100% | 100% | 100% | 0% |
| mean | 4 | 100% | 90% | 70% | 10% |
| mean | 5 | 100% | 100% | 70% | 0% |
| median | 2 | 100% | 100% | 100% | 0% |
| median | 3 | 100% | 100% | 100% | 0% |
| median | 4 | 100% | 100% | 90% | 0% |
| median | 5 | 100% | 100% | 70% | 0% |
| mosaic | 2 | 100% | 80% | 60% | 10% |
| mosaic | 3 | 80% | 80% | 20% | 0% |
| mosaic | 4 | 60% | 70% | 10% | 0% |
| mosaic | 5 | 80% | 80% | 20% | 10% |

## 方向判断

- mean 和 median：top-1 仍很强，主要问题是 k 增大后 Z=3 检出率下降。
- mosaic：明显破坏最强，top-1 和检出率都明显下降；但 clean 条件下也已经出现下降，说明这是 mosaic 本身加 InSwapper 的联合效果。
- 噪声很大：每组只有 10 个样本，Wilson 区间很宽，单点 10% 波动不能当作稳定结论。

## 是否值得跑 100 组

如果论文只需要说明“InSwapper 下合谋趋势与 SimSwap 一致”，这组 pilot 已足够支撑方向判断。

如果要把 InSwapper 合谋结果放进正式表格或画主图，建议再跑完整 100 组，尤其是 mean/median 的 k=4、5 和 mosaic 全段，避免 10 组样本造成误判。
