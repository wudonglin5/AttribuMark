# VQGAN vs SimSwap-224 vs InSwapper-128

## Per-image decoding

| Attack | BER | Exact 30-bit | Mean Hamming |
|---|---:|---:|---:|
| VQGAN | 0.000800 | 0.9760 | 0.0240 |
| SimSwap-224 | 0.001200 | 0.9720 | 0.0360 |
| InSwapper-128 | 0.000733 | 0.9800 | 0.0220 |

## Multi-image LLR

| m | VQGAN z | SimSwap-224 z | InSwapper-128 z | VQGAN margin | SimSwap-224 margin | InSwapper-128 margin |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.399 | 5.391 | 5.411 | 3.817 | 3.817 | 3.856 |
| 2 | 7.636 | 7.623 | 7.652 | 5.455 | 5.455 | 5.491 |
| 3 | 9.352 | 9.336 | 9.371 | 6.707 | 6.710 | 6.740 |
| 4 | 10.798 | 10.780 | 10.821 | 7.763 | 7.766 | 7.790 |
| 5 | 12.073 | 12.052 | 12.098 | 8.689 | 8.695 | 8.716 |

Panel (a) shows the absolute evidence curves; panel (b) zooms the tiny differences from VQGAN instead of claiming that overlapping curves show a material attack ranking. Panels (c)-(e) are empirical threshold sweeps for each attack. Panel (f) reports the single-image bit-error rate.

The three attacks produce nearly identical evidence. The defensible claim is that the watermark remains traceable under all three, not that one attack is significantly stronger.
