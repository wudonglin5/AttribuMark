# Cross-validated collusion threshold

- Five-fold group-wise cross-validation; target calibration false-accusation rate <= 1.0%.
- Thresholds are selected on four folds and evaluated on the held-out fold.
- The table reports pooled out-of-fold decisions, so every group is evaluated exactly once.
- The calibration objective maximizes detection subject to the false-accusation budget.

| Dataset | k | Selected Z (fold range) | Detection | 95% CI | False accusation | 95% CI | Abstain |
|---|---:|---:|---:|---:|---:|---:|---:|
| VQGAN mean | 2 | 2.85 (2.85-2.95) | 98.0% | [93.0, 99.4] | 0.0% | [0.0, 3.7] | 2.0% |
| VQGAN mean | 3 | 3.05 (2.70-3.05) | 86.0% | [77.9, 91.5] | 2.0% | [0.6, 7.0] | 12.0% |
| VQGAN mean | 4 | 3.10 (2.75-3.10) | 61.0% | [51.2, 70.0] | 2.0% | [0.6, 7.0] | 37.0% |
| VQGAN mean | 5 | 2.95 (2.85-2.95) | 61.0% | [51.2, 70.0] | 2.0% | [0.6, 7.0] | 37.0% |
| SimSwap mean | 2 | 2.65 (2.65-2.95) | 99.0% | [94.6, 99.8] | 0.0% | [0.0, 3.7] | 1.0% |
| SimSwap mean | 3 | 2.70 (2.55-2.80) | 94.0% | [87.5, 97.2] | 2.0% | [0.6, 7.0] | 4.0% |
| SimSwap mean | 4 | 2.75 (2.75-2.75) | 75.0% | [65.7, 82.5] | 1.0% | [0.2, 5.4] | 24.0% |
| SimSwap mean | 5 | 2.80 (2.70-2.80) | 68.0% | [58.3, 76.3] | 2.0% | [0.6, 7.0] | 30.0% |
| SimSwap median | 2 | 2.60 (2.60-2.95) | 99.0% | [94.6, 99.8] | 0.0% | [0.0, 3.7] | 1.0% |
| SimSwap median | 3 | 2.80 (2.60-2.80) | 94.0% | [87.5, 97.2] | 2.0% | [0.6, 7.0] | 4.0% |
| SimSwap median | 4 | 2.50 (2.20-2.50) | 93.0% | [86.3, 96.6] | 3.0% | [1.0, 8.5] | 4.0% |
| SimSwap median | 5 | 3.00 (2.25-3.00) | 64.0% | [54.2, 72.7] | 2.0% | [0.6, 7.0] | 34.0% |
| SimSwap mosaic | 2 | 3.10 (2.95-3.10) | 72.0% | [62.5, 79.9] | 2.0% | [0.6, 7.0] | 26.0% |
| SimSwap mosaic | 3 | 3.10 (3.05-3.10) | 34.0% | [25.5, 43.7] | 2.0% | [0.6, 7.0] | 64.0% |
| SimSwap mosaic | 4 | 3.45 (3.05-3.45) | 11.0% | [6.3, 18.6] | 2.0% | [0.6, 7.0] | 87.0% |
| SimSwap mosaic | 5 | 3.15 (3.15-3.15) | 25.0% | [17.5, 34.3] | 0.0% | [0.0, 3.7] | 75.0% |

The selected thresholds vary by fold and collusion strength, which is expected with 100 groups. Report this variation instead of presenting a single threshold as universally optimal.

![Cross-validated threshold](collusion_threshold_cv.png)
