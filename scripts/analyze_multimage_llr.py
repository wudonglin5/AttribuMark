import argparse
import json
from itertools import combinations

import numpy as np


THRESHOLDS = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]


def load_codes(strings):
    return np.asarray(
        [[int(bit) for bit in code] for code in strings], dtype=np.float64
    )


def z_from_probs(probs, codes):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    llr = np.log(p / (1.0 - p))
    scores = ((2.0 * codes[None, :, :] - 1.0) * llr[:, None, :]).sum(axis=2)
    norm = np.sqrt((llr * llr).sum(axis=1, keepdims=True))
    return scores / norm


def z_from_llr_sum(probs_by_image, codes):
    p = np.clip(np.asarray(probs_by_image, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    llr = np.log(p / (1.0 - p))
    llr_sum = llr.sum(axis=1)
    scores = ((2.0 * codes[None, :, :] - 1.0) * llr_sum[:, None, :]).sum(axis=2)
    norm = np.sqrt((llr * llr).sum(axis=(1, 2)))[:, None]
    return scores / norm


def threshold_metrics(z_matrix, thresholds, true_indices=None):
    if true_indices is None:
        true_indices = np.arange(z_matrix.shape[0])
    rows_index = np.arange(z_matrix.shape[0])
    true_z = z_matrix[rows_index, true_indices]
    top1 = np.argmax(z_matrix, axis=1)
    top1_z = np.max(z_matrix, axis=1)
    correct = top1 == true_indices
    other = z_matrix.copy()
    other[rows_index, true_indices] = -np.inf
    margin = true_z - np.max(other, axis=1)
    rows = {}
    for threshold in thresholds:
        accused = top1_z >= threshold
        rows[str(threshold)] = {
            "detect": float(np.mean(accused & correct)),
            "false_accuse": float(np.mean(accused & ~correct)),
            "abstain": float(np.mean(~accused)),
            "accused": int(accused.sum()),
        }
    return {
        "top1": float(np.mean(correct)),
        "mean_true_z": float(true_z.mean()),
        "median_true_z": float(np.median(true_z)),
        "min_true_z": float(true_z.min()),
        "mean_margin": float(margin.mean()),
        "min_margin": float(margin.min()),
        "thresholds": rows,
    }


def hard_vote_metrics(mean_probs, codes):
    predictions = (mean_probs >= 0.5).astype(np.int64)
    distances = np.sum(predictions[:, None, :] != codes[None, :, :], axis=-1)
    true_distances = np.diag(distances)
    nearest = np.argmin(distances, axis=1)
    return {
        "top1": float(np.mean(nearest == np.arange(len(codes)))),
        "exact_match": float(np.mean(true_distances == 0)),
        "mean_true_hamming": float(true_distances.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_txt", default="")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    codes = load_codes(data["codes"])
    num_users = data["num_users"]
    images_per_user = data["images_per_user"]
    probs = np.asarray(data["image_probs"], dtype=np.float64)
    user_ids = np.asarray(data["image_user_ids"], dtype=np.int64)
    image_indices = np.asarray(data["processed_image_indices"], dtype=np.int64)

    probs_by_user = np.zeros(
        (num_users, images_per_user, data["message_length"]), dtype=np.float64
    )
    for position, user_id in enumerate(user_ids):
        slot = int(image_indices[position] - user_id * images_per_user)
        probs_by_user[user_id, slot] = probs[position]

    summary = {}
    for m in range(1, images_per_user + 1):
        llr_z = z_from_llr_sum(probs_by_user[:, :m, :], codes)
        mean_probs = probs_by_user[:, :m, :].mean(axis=1)
        mean_z = z_from_probs(mean_probs, codes)
        summary[str(m)] = {
            "llr": threshold_metrics(llr_z, THRESHOLDS),
            "mean_prob": threshold_metrics(mean_z, THRESHOLDS),
            "hard_vote": hard_vote_metrics(mean_probs, codes),
        }

    all_subsets = {}
    for m in range(1, images_per_user + 1):
        llr_rows = []
        mean_prob_rows = []
        for slots in combinations(range(images_per_user), m):
            llr_rows.append(z_from_llr_sum(probs_by_user[:, slots, :], codes))
            mean_probs = probs_by_user[:, slots, :].mean(axis=1)
            mean_prob_rows.append(z_from_probs(mean_probs, codes))
        true_indices = np.tile(np.arange(num_users), len(llr_rows))
        all_subsets[str(m)] = {
            "num_subsets": len(llr_rows),
            "llr": threshold_metrics(
                np.concatenate(llr_rows, axis=0), THRESHOLDS, true_indices
            ),
            "mean_prob": threshold_metrics(
                np.concatenate(mean_prob_rows, axis=0), THRESHOLDS, true_indices
            ),
        }

    output = {
        "input": args.input_json,
        "num_users": num_users,
        "images_per_user": images_per_user,
        "summary": summary,
        "all_subsets": all_subsets,
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    lines = [
        f"input: {args.input_json}",
        "m | method | top1 | exact | mean_z | min_z | mean_margin",
    ]
    for m in range(1, images_per_user + 1):
        row = summary[str(m)]
        lines.append(
            f"{m} | LLR | {row['llr']['top1']:.4f} | - | "
            f"{row['llr']['mean_true_z']:.3f} | {row['llr']['min_true_z']:.3f} | "
            f"{row['llr']['mean_margin']:.3f}"
        )
        lines.append(
            f"{m} | mean-prob | {row['mean_prob']['top1']:.4f} | "
            f"{row['hard_vote']['exact_match']:.4f} | "
            f"{row['mean_prob']['mean_true_z']:.3f} | "
            f"{row['mean_prob']['min_true_z']:.3f} | "
            f"{row['mean_prob']['mean_margin']:.3f}"
        )
        for method in ["llr", "mean_prob"]:
            for threshold in ["3.0", "6.0", "8.0", "10.0"]:
                item = summary[str(m)][method]["thresholds"][threshold]
                lines.append(
                    f"  m={m} {method} Z={threshold} "
                    f"detect={item['detect']:.4f} "
                    f"false={item['false_accuse']:.4f} "
                    f"abstain={item['abstain']:.4f}"
                )
    lines.append("")
    lines.append("all image subsets:")
    for m in range(1, images_per_user + 1):
        row = all_subsets[str(m)]
        lines.append(
            f"m={m} subsets={row['num_subsets']} "
            f"LLR top1={row['llr']['top1']:.4f} "
            f"mean_z={row['llr']['mean_true_z']:.3f} "
            f"min_z={row['llr']['min_true_z']:.3f} "
            f"mean_margin={row['llr']['mean_margin']:.3f}"
        )
        lines.append(
            f"  mean-prob top1={row['mean_prob']['top1']:.4f} "
            f"mean_z={row['mean_prob']['mean_true_z']:.3f} "
            f"mean_margin={row['mean_prob']['mean_margin']:.3f}"
        )
        for threshold in ["6.0", "8.0", "10.0"]:
            item = row["llr"]["thresholds"][threshold]
            lines.append(
                f"  LLR Z={threshold} detect={item['detect']:.4f} "
                f"false={item['false_accuse']:.4f} "
                f"abstain={item['abstain']:.4f}"
            )
    text = "\n".join(lines)
    print(text)
    if args.output_txt:
        with open(args.output_txt, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
