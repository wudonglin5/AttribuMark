# Candidate-pool scaling under three attacks

| Candidates | VQGAN mean margin | SimSwap mean margin | InSwapper mean margin | VQGAN min | SimSwap min | InSwapper min |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 3.724 | 3.713 | 3.763 | 3.038 | 2.287 | 3.013 |
| 100 | 3.663 | 3.651 | 3.707 | 3.059 | 2.731 | 3.013 |
| 200 | 3.613 | 3.591 | 3.657 | 3.038 | 2.460 | 2.868 |
| 300 | 3.585 | 3.563 | 3.635 | 3.038 | 2.287 | 2.868 |
| 500 | 3.555 | 3.532 | 3.606 | 3.038 | 2.287 | 2.868 |

All three attacks retain a positive worst-case margin, but the user identification claim is limited to the sampled 500-user candidate pool.

![Codebook scaling](codebook_scaling_three_attacks.png)
