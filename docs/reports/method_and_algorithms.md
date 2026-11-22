# 方法与算法定稿（代码一致版）

本文档固定论文 Method 部分的符号、代码对应关系和算法伪代码。目标是让审稿人能够从方法描述复现实验，同时避免把 DiffMark 的原始嵌入器错误地包装成本文的新贡献。

## 1. 贡献边界

- 原始 DiffMark 提供扩散水印编码器、解码器和训练目标；本文不重新训练模型，也不改变 30-bit 消息接口。
- 本文新增的是用户级指纹层：一个用户的多张图共享同一个纠错码。
- 检测器把逐 bit 的 0/1 概率转成 log-likelihood ratio（LLR），并在多张图之间累加 log-odds。
- 最终决策允许拒识：证据不足时输出 `INCONCLUSIVE`，而不是强行指控最近邻用户。
- mean、median、mosaic 合谋属于威胁模型与实验协议，不作为新的训练算法。

## 2. 符号

| 符号 | 含义 |
|---|---|
| `U` | 候选用户数；主实验为 100，扩展实验为 500 |
| `L` | 指纹码长度；本实验 `L = 30` |
| `C` | 用户码本，`C = {c_1, ..., c_U}`，`c_u in {0,1}^L` |
| `c_{u,l}` | 用户 `u` 的第 `l` 个 bit |
| `d_min` | 码本最小 Hamming 距离；主实验为 11 |
| `x_0` | 128 x 128 的封面人脸图像 |
| `x_t` | 扩散时刻 `t` 的含噪图像 |
| `n_t` | 与 `x_t` 同步缩放的封面提示项 |
| `alpha_bar_t` | 时刻 `t` 的累计噪声系数 |
| `E_theta` | 带消息 cross-attention 的 DiffMark 编码 UNet |
| `D_phi` | 消息解码器，输出逐 bit 的 0/1 logits |
| `z` | 解码器 logits |
| `p_{j,l}` | 第 `j` 张图的第 `l` 个 bit 判为 1 的概率 |
| `r_{j,l}` | 该 bit 的 log-likelihood ratio |
| `m` | 同一用户被观察到的图片数 |
| `tau` | 指控阈值；低于阈值时拒识 |

## 3. 用户码本

部署前离线生成码本。主实验使用截短 BCH(31,11) 候选集合，从合法码字中选出 100 个用户码，使任意两个码字的最小 Hamming 距离为 11；500 用户扩展从 1024 个合法码字中选 500 个，最小距离仍为 11。

用户 `u` 的所有图片都嵌入同一个 `c_u`。这不是在每张图上独立生成随机消息；用户级归属依赖“同码多图”的证据积累。

## 4. 消息条件注入

实现与普通加性水印不同。消息不直接叠加到像素，而是转换为可学习 token：

```text
token_id(l, u) = l + c_{u,l} * L,    l = 0, ..., L - 1
e_{u,l} = Emb(token_id(l, u))
```

`Emb` 是大小为 `2L x 256` 的嵌入表。30 个 token 在 UNet 输出侧、分辨率为 16 和 8 的 cross-attention 层中注入特征图。编码器输入为 `[x_t; n_t]` 的通道拼接，时间步通过标准 timestep embedding 注入。

本次主实验使用 DDIM 100 步、无 classifier guidance。`use_guidance=False` 不是省略参数，而是冻结后的实验协议；所有报告结果都在该设置下产生。

## 5. 原论文的算法编号必须先弄清

用户指出的问题成立：DiffMark 原文的算法编号不是“嵌入算法 + 归因算法”，而是：

- `Algorithm 1`：DiffMark Training Framework，训练扩散编码器、解码器与冻结自编码器组成的端到端框架。
- `Algorithm 2`：DDIM Sampling with Deepfake-Resistant Guidance，带可选 Deepfake-resistant guidance 的 DDIM 采样。

因此，本文写作时不能把部署时的 DDIM 循环误称为 DiffMark 的 Algorithm 1。正确做法是：

1. 正文只放本文新增的 `Algorithm 1: Multi-image LLR attribution`。
2. 附录用 `Algorithm A1/A2` 复现原论文的训练与采样，供审稿人核对。
3. 正文引用 DiffMark 原文，不把原论文的方法重新包装成自己的贡献。

## 6. 原论文 Algorithm A1：DiffMark 训练框架

以下伪代码按原论文 Algorithm 1 复原。`AE(·)` 是冻结的 VQGAN 自编码器，`D_theta` 是水印解码器，`D_f` 是后面 guidance 使用的 Deepfake 模型。

```text
Algorithm A1: DiffMark Training Framework
Input : training distribution q(x_0), message length L,
        loss weights alpha and beta
Output: diffusion encoder E_theta and watermark decoder D_theta

1.  while not converged do
2.      x_0 ~ q(x_0)
3.      t ~ Uniform({1, ..., T})
4.      epsilon ~ N(0, I)
5.      x_t ~ N(sqrt(alpha_bar_t) * x_0,
                 (1 - alpha_bar_t) * I)
6.      x_c <- sqrt(1 - alpha_bar_t) * x_0
7.      w ~ {0, 1}^L
8.      xhat_0 <- E_theta(x_t, t, x_c, w)
9.      what <- D_theta(AE(xhat_0))
10.     对 theta 做梯度下降：
11.         grad( ||x_0 - xhat_0||_2^2
                    + alpha * L_lpips(xhat_0, x_0)
                    + beta  * L_ce(what, w) )
12. end while
```

与当前代码的对应关系：

- 代码里的 `n_t` 就是论文公式中的 `x_c = sqrt(1 - alpha_bar_t) x_0`。
- `E_theta` 的输入实际通过通道拼接 `[x_t; x_c]` 实现。
- `w` 的每个 bit 先通过位置-比特联合索引变成 token，再进入 cross-attention。
- `AE(·)` 在训练配置中是冻结的 VQGAN，代码中经 `Random_Noise` 包装后调用。
- 水印解码器输出 `L x 2L` 类 logits，CE 的标签是每个位置对应的嵌入索引。

## 7. 原论文 Algorithm A2：DDIM 采样与 Deepfake-resistant guidance

```text
Algorithm A2: DDIM Sampling with Deepfake-Resistant Guidance
Input : cover x_co, watermark w ~ {0,1}^L,
        boolean cond, gradient scale s
Output: watermarked image x_0 corresponding to x_co

1.  x_T ~ N(0, I)
2.  for t from T to 1 do
3.      x_c <- sqrt(1 - alpha_bar_{t-1}) * x_co
4.      xhat_0 <- E_theta(x_t, t, x_c, w)
5.      epshat <- (1 / sqrt(1 - alpha_bar_t)) * x_t
                  - sqrt(alpha_bar_t / (1 - alpha_bar_t)) * xhat_0
6.      if cond then
7.          epshat <- epshat - s * sqrt(1 - alpha_bar_t)
                      * grad_{x_t} log D_theta(w | D_f(xhat_0))
8.      end if
9.      x_{t-1} <- sqrt(alpha_bar_{t-1})
                    * ((x_t - sqrt(1 - alpha_bar_t) * epshat)
                       / sqrt(alpha_bar_t))
                    + sqrt(1 - alpha_bar_{t-1}) * epshat
10. end for
11. return x_0
```

当前复现实验的所有主结果使用 `cond = False`，即 `use_guidance=False`；因此实际采样是标准 DDIM 100 步。论文中若要报告 guidance 消融，必须另外开启 `cond=True`，不能把无 guidance 结果写成带 guidance 结果。

## 8. 当前复现与论文设置的差异

| 项目 | 原论文 | 当前复现代码/实验 | 写作要求 |
|---|---|---|---|
| 训练步数 | 151.2k 步，batch 16 | 30k 步，batch 8 | 不能声称训练已完全收敛 |
| 损失权重 | `beta=10`，128×128 在 5k 后降到 1 | `beta=1`，5k 后降到 0.1 | 必须如实说明；不能写成相同训练配方 |
| 模型规模 | 论文未在当前可核对段落给出完整配置 | 运行日志为 `num_channels=32, num_res_blocks=1, num_heads=4` | 投稿前核对官方配置，再决定是否补实验 |
| guidance | 论文同时报告 w/o 与 w/ | 主结果 `use_guidance=False` | 只能写无 guidance 的条件 |
| 训练攻击层 | 冻结 VQGAN | 冻结 VQGAN | 一致 |
| 测试换脸模型 | SimSwap 等 | SimSwap-224、InSwapper-128 | 属于测试攻击，不参与 30k 训练 |
| 分辨率 | 128×128 与 256×256 | 128×128 | 不能外推到 256×256 |

这些差异会影响“复现”措辞。更稳妥的说法是：

> 我们在 128×128 CelebA-HQ 上重新训练了 DiffMark 的公开实现，并固定 30k online checkpoint 作为后续用户级指纹实验的基线；训练步数、损失权重和模型宽度与原论文不完全相同，因此本文不声称逐参数复现。

## 9. 本文新增 Algorithm 1：多图 LLR 归因

这是论文正文唯一需要保留的算法。它对应 `scripts/analyze_multimage_llr.py` 和 `scripts/analyze_collusion_by_k.py`。

```text
Algorithm 1: Multi-image LLR attribution
Input : leaked images Y = {y_1, ..., y_m}
        user codebook C = {c_u}, c_u in {0,1}^L
        watermark decoder D_theta, accusation threshold tau
Output: accused user u*, score z*, or INCONCLUSIVE

1.  for j = 1 to m do
2.      Z_j <- D_theta(y_j)                 // shape: L x 2L
3.      for l = 1 to L do
4.          a_{j,l} <- Z_j[l, l]
5.          b_{j,l} <- Z_j[l, l + L]
6.          p_{j,l} <- sigmoid(b_{j,l} - a_{j,l})
7.          r_{j,l} <- log(p_{j,l} / (1 - p_{j,l}))
8.      end for
9.  end for

10. S_l <- sum_{j=1..m} r_{j,l}
11. N   <- sqrt(sum_{j=1..m} sum_{l=1..L} r_{j,l}^2)

12. for each candidate user u do
13.     score_u <- (1 / N) * sum_l (2*c_{u,l} - 1) * S_l
14. end for

15. u* <- argmax_u score_u
16. z* <- score_{u*}

17. if z* >= tau then
18.     return (u*, z*)
19. else
20.     return INCONCLUSIVE
21. end if
```

实现细节：

- 解码器每个位置有 `2L` 类输出，但只使用 `(l, l+L)` 这一对；因此 `p_{j,l}` 可直接写成两个 logits 之差的 sigmoid。
- 计算 `log` 前把概率裁剪到 `[1e-6, 1-1e-6]`。
- 归一化项 `N` 对同一批候选用户相同，不影响 top-1 排序，但使 `z*` 可跨图片数比较。
- 多图证据在 log-odds 域相加；直接平均概率会饱和。
- 阈值不是固定常数，正式部署应使用交叉验证或独立校准集选择 `tau`。

## 10. 合谋威胁模型

同一封面图分别嵌入 `k` 个合谋用户的码，得到 `x^(1), ..., x^(k)`。实验使用：

```text
mean:   y = (1/k) * sum_i x^(i)
median: y = median_i x^(i)
mosaic: y[M_r] = x^(r)[M_r],  r = 1, ..., k
```

mosaic 把图像划分成 `k` 个空间区域，每个区域取自不同合谋副本。主结果使用整图解码；按格子裁剪解码不稳定，只作为探索性分析。检测器不知道 `k`，也不知道使用了哪一种混合方式。

## 11. 论文中的呈现建议

- 主文只放本文 `Algorithm 1`，用一页以内说明 log-odds 聚合和拒识。
- 附录放原论文 `Algorithm A1/A2`，标明来自 DiffMark，用于复现，不当作新贡献。
- 主文同时给系统流程图：用户码 -> 同码多图嵌入 -> 泄露图/合谋 -> 多图 LLR -> 指控或拒识。
- 所有阈值结果同时报告 top-1、检出率、误指控率和拒识率。
- 10 组 InSwapper 合谋只能写成 exploratory pilot；正式表格需要完整组数或明确样本量。
- 初稿默认先写中文，英文版在中文逻辑和数字确认后再翻译。

## 12. 代码对应

| 内容 | 代码位置 |
|---|---|
| 消息 token 与 cross-attention | `guided_diffusion/unet.py` |
| 扩散采样与训练损失 | `guided_diffusion/gaussian_diffusion.py` |
| 训练循环与消息采样 | `guided_diffusion/train_util.py` |
| 用户级多图评估 | `scripts/user_fingerprint_eval.py` |
| 多图 LLR 与 all-subsets | `scripts/analyze_multimage_llr.py` |
| 合谋后的 soft 检测 | `scripts/analyze_collusion_by_k.py` |
| 合谋数据生成 | `scripts/user_collusion_eval.py` |

