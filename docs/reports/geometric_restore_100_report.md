# Geometric restore upper-bound experiment (2026-11-06)

## Protocol
- 100 users x 1 image, CelebA-HQ 128x128, online 30k checkpoint, DDIM 100.
- Attack: center crop to 90%, 80%, or 70%, then bilinear resize back to 128.
- Restoration: resize attacked image back to the crop size and pad the missing border with zero, edge replication, or reflection.
- This is an oracle upper bound: the crop fraction is known to the restorer.

## Result: 90% center crop
| Condition | BER | Top-1 | Detection | False accusation | Abstention | Mean true z |
|---|---:|---:|---:|---:|---:|---:|
| Attack only | 38.60% | 10.0% | 2.0% | 13.0% | 85.0% | 1.440 |
| Zero padding | 36.73% | 16.0% | 4.0% | 11.0% | 85.0% | 1.715 |
| Edge replication | 36.53% | 19.0% | 3.0% | 9.0% | 88.0% | 1.744 |
| Reflection | 36.40% | 15.0% | 4.0% | 11.0% | 85.0% | 1.713 |

## Additional crops
| Condition | BER | Top-1 | Detection | False accusation | Abstention |
|---|---:|---:|---:|---:|---:|
| 80% crop only | 43.10% | 4.0% | 0.0% | 11.0% | 89.0% |
| 80% crop + edge restore | 41.47% | 5.0% | 0.0% | 10.0% | 90.0% |
| 70% crop only | 50.13% | 1.0% | 0.0% | 9.0% | 91.0% |
| 70% crop + edge restore | 49.67% | 1.0% | 0.0% | 9.0% | 91.0% |

## Conclusion
- Even with the true crop fraction, post-hoc inverse scaling and border padding do not recover the user code.
- The failure is therefore not merely unknown crop geometry. The crop/upscale process removes or corrupts the watermark signal itself.
- Do not claim that geometric synchronization solves the crop limitation with this checkpoint.
- A genuine fix would require crop/scale augmentation during watermark training, a redundant multi-scale embedding, or a stronger registration/reconstruction model.
- For the current paper, keep crop as an explicit limitation and retain abstention as the safe failure mode.

## Artifacts
- Script: geometric_restore_100 JSON / script in the Codex work directory.
- Results: geometric_restore_100.json.