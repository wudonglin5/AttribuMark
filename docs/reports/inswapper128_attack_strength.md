# InSwapper-128 attack strength

## Protocol

- Images: 100 watermarked CelebA-HQ 128x128 images.
- Watermark: 30-bit user code from the min-Hamming-11 codebook, DDIM 100.
- Identity source: the paired image in the same batch; no image is swapped with itself.
- Attack: watermarked content and a different identity image are aligned to 128x128 with five landmarks, processed by InSwapper-128, and pasted back to the 128x128 frame.
- Identity metrics use the official InsightFace w600k_r50 recognizer with 112x112 RGB [-1,1] normalization; generation additionally uses InSwapper's official emap projection.

## Pixel-level attack distortion

| Comparison | Mean +/- std | 95% CI | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| PSNR, watermarked to attacked (dB) | 24.4123 +/- 1.9998 | [24.0203, 24.8043] | 24.3556 | 17.3510 | 28.7321 |
| SSIM, watermarked to attacked | 0.7456 +/- 0.0559 | [0.7346, 0.7565] | 0.7525 | 0.5592 | 0.8567 |
| LPIPS, watermarked to attacked | 0.1314 +/- 0.0298 | [0.1256, 0.1372] | 0.1290 | 0.0677 | 0.2049 |

## Identity evidence

| Comparison | Mean +/- std | 95% CI | Median | Min | Max |
|---|---:|---:|---:|---:|---:|
| Cosine, watermarked vs. source | 0.0199 +/- 0.0609 | [0.0079, 0.0318] | 0.0259 | -0.1340 | 0.1454 |
| Cosine, attacked vs. source | 0.7968 +/- 0.0542 | [0.7861, 0.8074] | 0.8068 | 0.6417 | 0.8834 |
| Cosine, attacked vs. watermarked | 0.1279 +/- 0.0833 | [0.1116, 0.1442] | 0.1276 | -0.0636 | 0.3525 |
| Identity margin, attack-source minus attack-WM | 0.6689 +/- 0.1066 | [0.6480, 0.6898] | 0.6825 | 0.4277 | 0.9404 |

The identity margin is positive for 100.0% of images. This is the fraction for which the attacked face is closer to the supplied identity than to the watermarked content identity.

## Interpretation

- The pixel metrics quantify how much InSwapper changes the watermarked image.
- The ArcFace table separates true identity replacement from ordinary image distortion.
- These measurements describe attack strength only. Watermark BER and tracing results remain in the InSwapper single-image and collusion reports.

## Figures

![Attack strength](inswapper128_attack_strength.png)

![Attack examples](inswapper128_attack_strength_example.png)
