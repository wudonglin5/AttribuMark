# -*- coding: utf-8 -*-
"""Audit the certified 1% threshold on 500 users x 1 image.

For every image and every threshold tau, count
  - image-level event: any of the other 499 candidates reaches tau
  - candidate-level event: any innocent candidate reaches tau
and report true-user detection, top-1 and the innocent-score maximum.
"""
import argparse
import json
import math
import os
import statistics


def z_scores(probs, codes):
    llr = []
    for p in probs:
        p = min(max(float(p), 1e-6), 1.0 - 1e-6)
        llr.append(math.log(p / (1.0 - p)))
    denom = math.sqrt(sum(v * v for v in llr))
    return [
        sum((1.0 if bit else -1.0) * v for bit, v in zip(code, llr)) / denom
        for code in codes
    ]


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"F:\project\beifen\DiffMark-main\results")
    parser.add_argument("--out_json", default="")
    parser.add_argument("--out_md", default="")
    args = parser.parse_args()

    files = {
        "VQGAN": "user_fingerprint_eval_30k_500users_bch_vqgan_probs.json",
        "SimSwap-224": "user_fingerprint_eval_30k_500users_bch_simswap224_probs.json",
        "InSwapper-128": "user_fingerprint_eval_30k_500users_bch_inswapper128_probs.json",
    }
    thresholds = (4.023, 4.292)
    out = {}
    lines = []

    for name, filename in files.items():
        with open(os.path.join(args.root, filename), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        codes = [[int(bit) for bit in code] for code in data["codes"]]
        users = data["image_user_ids"]
        probs = data["image_probs"]
        assert len(users) == len(probs) == 500

        for tau in thresholds:
            top1 = 0
            true_detect = 0
            images_any_false = 0
            total_false = 0
            max_innocent = []
            margins = []
            true_z = []
            for user, prob in zip(users, probs):
                scores = z_scores(prob, codes)
                tz = scores[user]
                innocent = [v for j, v in enumerate(scores) if j != user]
                best_innocent = max(innocent)
                true_z.append(tz)
                max_innocent.append(best_innocent)
                margins.append(tz - best_innocent)
                top1 += int(max(range(len(scores)), key=lambda j: scores[j]) == user)
                total_false += sum(1 for v in innocent if v >= tau)
                images_any_false += int(any(v >= tau for v in innocent))
                true_detect += int(tz >= tau)

            lo, hi = wilson(images_any_false, 500)
            key = f"{name}|{tau}"
            out[key] = {
                "tau": tau,
                "top1": top1 / 500.0,
                "true_detect": true_detect / 500.0,
                "images_any_false": images_any_false,
                "image_false_rate": images_any_false / 500.0,
                "image_false_wilson95": [lo, hi],
                "innocent_trials": 500 * 499,
                "total_false_trials": total_false,
                "trial_false_rate": total_false / (500 * 499),
                "max_innocent_z": max(max_innocent),
                "mean_max_innocent_z": statistics.mean(max_innocent),
                "mean_margin": statistics.mean(margins),
                "min_margin": min(margins),
                "mean_true_z": statistics.mean(true_z),
            }
            lines.append(
                f"{name:15s} tau={tau:.3f} top1={top1/500:.4f} "
                f"detect={true_detect/500:.4f} "
                f"img_false={images_any_false}/500 (Wilson95<={hi*100:.3f}%) "
                f"candidate_false={total_false}/{500*499} "
                f"max_innocent_z={max(max_innocent):.4f} "
                f"mean_margin={statistics.mean(margins):.4f}"
            )

    text = "\n".join(lines)
    print(text)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(out, handle, ensure_ascii=False, indent=2)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as handle:
            handle.write(
                "# Certified 1% threshold audit: 500 users x 1 image\n\n"
                "Image-level false event = at least one of the 499 innocent "
                "candidates reaches tau. All three attacks use the shared "
                "500-user BCH codebook.\n\n"
                "```\n" + text + "\n```\n"
            )


if __name__ == "__main__":
    main()
