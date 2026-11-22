# Multi-image LLR evidence under three attacks

| m | VQGAN z | SimSwap-224 z | InSwapper-128 z | VQGAN margin | SimSwap-224 margin | InSwapper-128 margin |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.399 | 5.391 | 5.411 | 3.817 | 3.817 | 3.856 |
| 2 | 7.636 | 7.623 | 7.652 | 5.455 | 5.455 | 5.491 |
| 3 | 9.352 | 9.336 | 9.371 | 6.707 | 6.710 | 6.740 |
| 4 | 10.798 | 10.780 | 10.821 | 7.763 | 7.766 | 7.790 |
| 5 | 12.073 | 12.052 | 12.098 | 8.689 | 8.695 | 8.716 |

The largest absolute z difference from VQGAN is 0.025; this is not evidence that one attack is materially stronger.

![Multi-image LLR](multimage_llr_curve.png)
