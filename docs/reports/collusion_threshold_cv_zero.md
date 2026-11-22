# Cross-validated collusion threshold

- Five-fold group-wise cross-validation; target calibration false-accusation rate <= 0.0%.
- Thresholds are selected on four folds and evaluated on the held-out fold.
- The table reports pooled out-of-fold decisions, so every group is evaluated exactly once.
- The calibration objective maximizes detection subject to the false-accusation budget.

| Dataset | k | Selected Z (fold range) | Detection | 95% CI | False accusation | 95% CI | Abstain |
|---|---:|---:|---:|---:|---:|---:|---:|
| VQGAN mean | 2 | 2.85 (2.85-2.95) | 98.0% | [93.0, 99.4] | 0.0% | [0.0, 3.7] | 2.0% |
| VQGAN mean | 3 | 3.30 (3.10-3.30) | 68.0% | [58.3, 76.3] | 1.0% | [0.2, 5.4] | 31.0% |
| VQGAN mean | 4 | 3.45 (3.10-3.45) | 26.0% | [18.4, 35.4] | 1.0% | [0.2, 5.4] | 73.0% |
| VQGAN mean | 5 | 3.20 (2.95-3.20) | 41.0% | [31.9, 50.8] | 1.0% | [0.2, 5.4] | 58.0% |
| SimSwap mean | 2 | 2.65 (2.65-2.95) | 99.0% | [94.6, 99.8] | 0.0% | [0.0, 3.7] | 1.0% |
| SimSwap mean | 3 | 3.10 (2.70-3.10) | 81.0% | [72.2, 87.5] | 1.0% | [0.2, 5.4] | 18.0% |
| SimSwap mean | 4 | 3.05 (2.75-3.05) | 56.0% | [46.2, 65.3] | 1.0% | [0.2, 5.4] | 43.0% |
| SimSwap mean | 5 | 3.00 (2.80-3.00) | 53.0% | [43.3, 62.5] | 1.0% | [0.2, 5.4] | 46.0% |
| SimSwap median | 2 | 2.60 (2.60-2.95) | 99.0% | [94.6, 99.8] | 0.0% | [0.0, 3.7] | 1.0% |
| SimSwap median | 3 | 3.40 (2.80-3.40) | 71.0% | [61.5, 79.0] | 1.0% | [0.2, 5.4] | 28.0% |
| SimSwap median | 4 | 2.70 (2.50-2.70) | 87.0% | [79.0, 92.2] | 1.0% | [0.2, 5.4] | 12.0% |
| SimSwap median | 5 | 3.15 (2.35-3.15) | 56.0% | [46.2, 65.3] | 2.0% | [0.6, 7.0] | 42.0% |
| SimSwap mosaic | 2 | 3.45 (2.95-3.45) | 53.0% | [43.3, 62.5] | 2.0% | [0.6, 7.0] | 45.0% |
| SimSwap mosaic | 3 | 3.35 (3.10-3.35) | 22.0% | [15.0, 31.1] | 1.0% | [0.2, 5.4] | 77.0% |
| SimSwap mosaic | 4 | 3.45 (3.45-3.45) | 8.0% | [4.1, 15.0] | 0.0% | [0.0, 3.7] | 92.0% |
| SimSwap mosaic | 5 | 3.70 (3.15-3.80) | 6.0% | [2.8, 12.5] | 0.0% | [0.0, 3.7] | 94.0% |

The selected thresholds vary by fold and collusion strength, which is expected with 100 groups. Report this variation instead of presenting a single threshold as universally optimal.

![Cross-validated threshold](collusion_threshold_cv_zero.png)
