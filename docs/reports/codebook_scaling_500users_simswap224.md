# User-count scaling under SimSwap-224

- 500 CelebA-HQ images and 500 distinct shortened BCH(31,11) codes, one image per user.
- 30-bit minimum Hamming distance 11; mean pairwise distance 14.66.
- SimSwap-224 attack, DDIM 100, online 30k checkpoint.
- Candidate subsets are sampled without replacement; each subset contains every true code used in that trial.

| Candidate users | Trials | Top-1 | 95% CI | Mean margin | Z=3 detect | Z=3 false |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1000 | 100.0% | [99.6, 100.0] | 3.713 | 100.0% | 0.0% |
| 100 | 2000 | 100.0% | [99.8, 100.0] | 3.651 | 100.0% | 0.0% |
| 200 | 4000 | 100.0% | [99.9, 100.0] | 3.591 | 100.0% | 0.0% |
| 300 | 6000 | 100.0% | [99.9, 100.0] | 3.563 | 100.0% | 0.0% |
| 500 | 500 | 100.0% | [99.2, 100.0] | 3.532 | 100.0% | 0.0% |

The figure shows (a) the separation between the true-user score and the best-impostor score at the largest candidate pool, and (b) the distribution of the best-versus-second margin as the pool grows. The flat top-1 and false-accusation values are reported as text in panel (b) instead of being drawn as uninformative flat lines.

![Codebook scaling](codebook_scaling_500users_simswap224.png)
