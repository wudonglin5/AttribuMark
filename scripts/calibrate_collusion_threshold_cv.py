import argparse
import json
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")


def load_codes(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return np.asarray(
        [[int(bit) for bit in code] for code in data["codes"]], dtype=np.float64
    )


def z_scores(probs, codes):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    llr = np.log(p / (1.0 - p))
    scores = ((2.0 * codes - 1.0) * llr[None, :]).sum(axis=1)
    return scores / np.sqrt((llr * llr).sum())


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def parse_input(spec):
    if "=" not in spec:
        raise ValueError("--input_json must be LABEL=PATH")
    label, path = spec.split("=", 1)
    return label.strip(), path.strip()


def record_outcome(record, condition, codes):
    scores = z_scores(record[condition + "_probs"], codes)
    top = int(np.argmax(scores))
    return float(scores[top]), top in set(record["colluders"])


def select_threshold(calibration, grid, target_false_accusation):
    best = None
    for threshold in grid:
        accused = calibration["z"] >= threshold
        n = len(accused)
        false_rate = float(np.sum(accused & ~calibration["hit"]) / n)
        if false_rate > target_false_accusation:
            continue
        detection = float(np.sum(accused & calibration["hit"]) / n)
        candidate = (detection, float(threshold), false_rate)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        threshold = float(grid[-1])
        accused = calibration["z"] >= threshold
        false_rate = float(np.sum(accused & ~calibration["hit"]) / len(accused))
        return threshold, false_rate
    return best[1], best[2]


def summarize_outcomes(outcomes):
    z = np.asarray([row[0] for row in outcomes], dtype=np.float64)
    hit = np.asarray([row[1] for row in outcomes], dtype=bool)
    return z, hit


def evaluate_threshold(outcomes, threshold):
    z, hit = summarize_outcomes(outcomes)
    accused = z >= threshold
    detected = accused & hit
    false = accused & ~hit
    n = len(outcomes)
    det_ci = wilson(int(detected.sum()), n)
    fa_ci = wilson(int(false.sum()), n)
    return {
        "n": n,
        "threshold": float(threshold),
        "detect": float(detected.sum() / n),
        "detect_ci95": [float(det_ci[0]), float(det_ci[1])],
        "false_accuse": float(false.sum() / n),
        "false_accuse_ci95": [float(fa_ci[0]), float(fa_ci[1])],
        "abstain": float((~accused).sum() / n),
        "accused": int(accused.sum()),
    }


def evaluate_fold_decisions(fold_decisions):
    accused = []
    hit = []
    for threshold, outcomes in fold_decisions:
        z, labels = summarize_outcomes(outcomes)
        accused.extend((z >= threshold).tolist())
        hit.extend(labels.tolist())
    accused = np.asarray(accused, dtype=bool)
    hit = np.asarray(hit, dtype=bool)
    detected = accused & hit
    false = accused & ~hit
    n = len(accused)
    det_ci = wilson(int(detected.sum()), n)
    fa_ci = wilson(int(false.sum()), n)
    return {
        "n": n,
        "detect": float(detected.sum() / n),
        "detect_ci95": [float(det_ci[0]), float(det_ci[1])],
        "false_accuse": float(false.sum() / n),
        "false_accuse_ci95": [float(fa_ci[0]), float(fa_ci[1])],
        "abstain": float((~accused).sum() / n),
        "accused": int(accused.sum()),
    }


def matrix(results, field, scale=100.0, threshold=False):
    labels = list(results.keys())
    ks = sorted(int(k) for k in results[labels[0]]["k"].keys())
    values = []
    for label in labels:
        row = []
        for k in ks:
            record = results[label]["k"][str(k)]
            if threshold:
                value = record["fold_threshold_median"]
            else:
                value = record["attack"]["cross_validated"][field] * scale
            row.append(float(value))
        values.append(row)
    return labels, ks, np.asarray(values, dtype=np.float64)


def heatmap(axis, values, labels, ks, title, cmap, vmin, vmax, fmt):
    image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_xticks(np.arange(len(ks)))
    axis.set_xticklabels([str(k) for k in ks])
    axis.set_yticks(np.arange(len(labels)))
    axis.set_yticklabels(labels, fontsize=9)
    axis.set_xlabel("Number of colluders k")
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            normalized = (values[row, column] - vmin) / max(vmax - vmin, 1e-9)
            color = "white" if normalized > 0.62 else "#1a202c"
            axis.text(
                column,
                row,
                fmt.format(values[row, column]),
                ha="center",
                va="center",
                fontsize=9,
                color=color,
            )
    return image


def save_plot(results, path, target):
    labels, ks, detect = matrix(results, "detect")
    _, _, false = matrix(results, "false_accuse")
    _, _, abstain = matrix(results, "abstain")
    _, _, thresholds = matrix(results, "threshold", threshold=True)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    image = heatmap(
        axes[0, 0],
        detect,
        labels,
        ks,
        "(a) Held-out detection rate (%)",
        "Blues",
        0.0,
        100.0,
        "{:.0f}",
    )
    fig.colorbar(image, ax=axes[0, 0], fraction=0.046, pad=0.03)
    false_max = max(3.0, float(false.max()))
    image = heatmap(
        axes[0, 1],
        false,
        labels,
        ks,
        "(b) Held-out false-accusation rate (%)",
        "Reds",
        0.0,
        false_max,
        "{:.1f}",
    )
    fig.colorbar(image, ax=axes[0, 1], fraction=0.046, pad=0.03)
    image = heatmap(
        axes[1, 0],
        abstain,
        labels,
        ks,
        "(c) Held-out abstention rate (%)",
        "Greys",
        0.0,
        100.0,
        "{:.0f}",
    )
    fig.colorbar(image, ax=axes[1, 0], fraction=0.046, pad=0.03)
    image = heatmap(
        axes[1, 1],
        thresholds,
        labels,
        ks,
        "(d) Median selected threshold Z",
        "viridis",
        float(thresholds.min()),
        float(thresholds.max()),
        "{:.2f}",
    )
    fig.colorbar(image, ax=axes[1, 1], fraction=0.046, pad=0.03)
    fig.suptitle(
        f"Group-wise cross-validated collusion tracing (calibration budget <= {target:.0%})",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_markdown(results, target, path, figure_name):
    lines = [
        "# Cross-validated collusion threshold",
        "",
        f"- Five-fold group-wise cross-validation; target calibration false-accusation rate <= {target:.1%}.",
        "- Thresholds are selected on four folds and evaluated on the held-out fold.",
        "- The table reports pooled out-of-fold decisions, so every group is evaluated exactly once.",
        "- The calibration objective maximizes detection subject to the false-accusation budget.",
        "",
        "| Dataset | k | Selected Z (fold range) | Detection | 95% CI | False accusation | 95% CI | Abstain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, dataset in results.items():
        for k in sorted(int(value) for value in dataset["k"].keys()):
            row = dataset["k"][str(k)]["attack"]["cross_validated"]
            thresholds = dataset["k"][str(k)]["fold_thresholds"]
            lines.append(
                f"| {label} | {k} | {np.median(thresholds):.2f} "
                f"({min(thresholds):.2f}-{max(thresholds):.2f}) | "
                f"{100.0 * row['detect']:.1f}% | "
                f"[{100.0 * row['detect_ci95'][0]:.1f}, {100.0 * row['detect_ci95'][1]:.1f}] | "
                f"{100.0 * row['false_accuse']:.1f}% | "
                f"[{100.0 * row['false_accuse_ci95'][0]:.1f}, {100.0 * row['false_accuse_ci95'][1]:.1f}] | "
                f"{100.0 * row['abstain']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "The selected thresholds vary by fold and collusion strength, which is expected with 100 groups. "
            "Report this variation instead of presenting a single threshold as universally optimal.",
            "",
            f"![Cross-validated threshold]({figure_name})",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", action="append", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--target_false_accusation", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    grid = np.arange(1.0, 5.01, 0.05)
    results = {}
    for spec in args.input_json:
        label, input_path = parse_input(spec)
        with open(input_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        codes = load_codes(data["codebook_json"])
        records = data["records"]
        dataset_result = {
            "input_json": input_path,
            "mix": data.get("mix", ""),
            "k": {},
        }
        for k in range(data["min_k"], data["max_k"] + 1):
            k_records = [record for record in records if record["k"] == k]
            fold_thresholds = []
            pooled = {"clean": [], "attack": []}
            fold_decisions = {"clean": [], "attack": []}
            for fold in range(args.folds):
                calibration = {"z": [], "hit": []}
                test = {"clean": [], "attack": []}
                for record in k_records:
                    row = record_outcome(record, "clean", codes)
                    target_condition = "attack" if record["group"] % args.folds == fold else "calibration"
                    if target_condition == "attack":
                        test["clean"].append(row)
                    else:
                        calibration["z"].append(row[0])
                        calibration["hit"].append(row[1])
                    attack_row = record_outcome(record, "vqgan", codes)
                    if target_condition == "attack":
                        test["attack"].append(attack_row)
                    else:
                        calibration["z"].append(attack_row[0])
                        calibration["hit"].append(attack_row[1])
                calibration["z"] = np.asarray(calibration["z"], dtype=np.float64)
                calibration["hit"] = np.asarray(calibration["hit"], dtype=bool)
                threshold, _ = select_threshold(
                    calibration, grid, args.target_false_accusation
                )
                fold_thresholds.append(threshold)
                pooled["clean"].extend(test["clean"])
                pooled["attack"].extend(test["attack"])
                fold_decisions["clean"].append((threshold, test["clean"]))
                fold_decisions["attack"].append((threshold, test["attack"]))
            dataset_result["k"][str(k)] = {
                "fold_thresholds": fold_thresholds,
                "fold_threshold_median": float(np.median(fold_thresholds)),
                "fold_threshold_range": [
                    float(min(fold_thresholds)),
                    float(max(fold_thresholds)),
                ],
                "clean": {
                    "cross_validated": evaluate_fold_decisions(fold_decisions["clean"])
                },
                "attack": {
                    "cross_validated": evaluate_fold_decisions(fold_decisions["attack"])
                },
                "fixed_z3": {
                    "clean": evaluate_threshold(pooled["clean"], 3.0),
                    "attack": evaluate_threshold(pooled["attack"], 3.0),
                },
            }
        results[label] = dataset_result

    output = {
        "target_false_accusation": args.target_false_accusation,
        "folds": args.folds,
        "threshold_grid": [float(grid[0]), float(grid[-1]), 0.05],
        "datasets": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    save_plot(results, args.output_png, args.target_false_accusation)
    write_markdown(
        results,
        args.target_false_accusation,
        args.output_md,
        os.path.basename(args.output_png),
    )
    print(f"saved: {args.output_json}")
    print(f"saved: {args.output_md}")
    print(f"saved: {args.output_png}")


if __name__ == "__main__":
    main()
