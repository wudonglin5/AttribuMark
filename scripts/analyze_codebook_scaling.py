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
    if llr.ndim == 2:
        scores = llr @ (2.0 * codes - 1.0).T
        norms = np.sqrt((llr * llr).sum(axis=1))
        return scores / norms[:, None]
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


def summarize_trials(trials, thresholds):
    hit = float(np.mean([trial["hit"] for trial in trials]))
    hit_ci = wilson(int(sum(trial["hit"] for trial in trials)), len(trials))
    best_z = np.asarray([trial["best_z"] for trial in trials])
    margin = np.asarray([trial["margin"] for trial in trials])
    result = {
        "trials": len(trials),
        "top1": hit,
        "top1_ci95": [float(hit_ci[0]), float(hit_ci[1])],
        "mean_best_z": float(best_z.mean()),
        "mean_margin": float(margin.mean()),
        "min_margin": float(margin.min()),
        "operating_points": {},
    }
    for threshold in thresholds:
        accused = np.asarray([trial["best_z"] >= threshold for trial in trials])
        hits = np.asarray([trial["hit"] for trial in trials])
        detected = accused & hits
        false = accused & ~hits
        det_ci = wilson(int(detected.sum()), len(trials))
        fa_ci = wilson(int(false.sum()), len(trials))
        result["operating_points"][f"{threshold:.1f}"] = {
            "detect": float(detected.mean()),
            "detect_ci95": [float(det_ci[0]), float(det_ci[1])],
            "false_accuse": float(false.mean()),
            "false_accuse_ci95": [float(fa_ci[0]), float(fa_ci[1])],
            "abstain": float((~accused).mean()),
        }
    return result


def describe_attack(attack):
    labels = {
        "VQGAN()": "VQGAN",
        "SimSwap(224)": "SimSwap-224",
        "InSwapper(128)": "InSwapper-128",
    }
    names = [labels.get(name, name) for name in attack] or ["unknown attack"]
    return " + ".join(names)


def save_plot(results, path, attack_label, trial_samples):
    sizes = sorted(results.keys(), key=int)
    final_size = sizes[-1]
    final_trials = trial_samples[final_size]
    true_z = np.asarray([trial["best_z"] for trial in final_trials])
    impostor_z = true_z - np.asarray([trial["margin"] for trial in final_trials])
    margins = [
        np.asarray([trial["margin"] for trial in trial_samples[size]])
        for size in sizes
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.suptitle(f"Candidate-pool scaling under {attack_label}", fontsize=12)

    bins = np.linspace(
        min(float(true_z.min()), float(impostor_z.min())) - 0.1,
        max(float(true_z.max()), float(impostor_z.max())) + 0.1,
        55,
    )
    axes[0].hist(
        true_z,
        bins=bins,
        density=True,
        alpha=0.58,
        color="#2b6cb0",
        label="True user",
    )
    axes[0].hist(
        impostor_z,
        bins=bins,
        density=True,
        alpha=0.58,
        color="#c53030",
        label="Best impostor",
    )
    axes[0].axvline(float(true_z.mean()), color="#2b6cb0", linestyle="--", linewidth=1.5)
    axes[0].axvline(float(impostor_z.mean()), color="#c53030", linestyle="--", linewidth=1.5)
    axes[0].set_title(f"(a) Score separation at {final_size} candidates")
    axes[0].set_xlabel("z score")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(alpha=0.22)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    boxplot = axes[1].boxplot(
        margins,
        tick_labels=sizes,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
    )
    for patch in boxplot["boxes"]:
        patch.set_facecolor("#c6f6d5")
        patch.set_edgecolor("#2f855a")
        patch.set_alpha(0.85)
    for element in ("whiskers", "caps", "medians"):
        for line in boxplot[element]:
            line.set_color("#2f855a")
    means = [float(values.mean()) for values in margins]
    axes[1].plot(
        np.arange(1, len(sizes) + 1),
        means,
        marker="o",
        color="#22543d",
        linewidth=2,
        label="Mean margin",
    )
    axes[1].set_title("(b) Best-vs-second margin distribution")
    axes[1].set_xlabel("Candidate users")
    axes[1].set_ylabel("z margin")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    final_top1 = 100.0 * results[final_size]["top1"]
    final_false = 100.0 * results[final_size]["operating_points"]["3.0"]["false_accuse"]
    axes[1].text(
        0.98,
        0.04,
        f"At {final_size}: top-1 {final_top1:.1f}%, Z=3 false {final_false:.1f}%",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4a5568",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2.0},
        zorder=10,
    )
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_markdown(summary, path, attack_label, figure_name):
    lines = [
        f"# User-count scaling under {attack_label}",
        "",
        "- 500 CelebA-HQ images and 500 distinct shortened BCH(31,11) codes, one image per user.",
        "- 30-bit minimum Hamming distance 11; mean pairwise distance 14.66.",
        f"- {attack_label} attack, DDIM 100, online 30k checkpoint.",
        "- Candidate subsets are sampled without replacement; each subset contains every true code used in that trial.",
        "",
        "| Candidate users | Trials | Top-1 | 95% CI | Mean margin | Z=3 detect | Z=3 false |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size in sorted(summary["candidate_sizes"].keys(), key=int):
        row = summary["candidate_sizes"][size]
        op = row["operating_points"]["3.0"]
        lines.append(
            f"| {size} | {row['trials']} | {100.0 * row['top1']:.1f}% | "
            f"[{100.0 * row['top1_ci95'][0]:.1f}, {100.0 * row['top1_ci95'][1]:.1f}] | "
            f"{row['mean_margin']:.3f} | {100.0 * op['detect']:.1f}% | "
            f"{100.0 * op['false_accuse']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "The figure shows (a) the separation between the true-user score and the best-impostor score at the largest candidate pool, and (b) the distribution of the best-versus-second margin as the pool grows. The flat top-1 and false-accusation values are reported as text in panel (b) instead of being drawn as uninformative flat lines.",
            "",
            f"![Codebook scaling]({figure_name})",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--candidate_sizes", default="50,100,200,300,500")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    parser.add_argument("--output_png", required=True)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    codes = load_codes(data["codebook_json"])
    probs = np.asarray(data["image_probs"], dtype=np.float64)
    user_ids = np.asarray(data["image_user_ids"], dtype=np.int64)
    if codes.shape[0] != data["num_users"] or probs.shape[0] != data["num_users"]:
        raise ValueError("codebook, probabilities, and user count do not match")
    if not np.array_equal(user_ids, np.arange(data["num_users"])):
        raise ValueError("this scaling analysis expects one image per user")

    rng = np.random.default_rng(args.seed)
    thresholds = [2.5, 3.0, 3.5]
    candidate_sizes = [int(value) for value in args.candidate_sizes.split(",")]
    results = {}
    trial_samples = {}
    for size in candidate_sizes:
        if size > data["num_users"]:
            raise ValueError(f"candidate size {size} exceeds 500 users")
        repeats = 1 if size == data["num_users"] else args.repeats
        trials = []
        for _ in range(repeats):
            selected = rng.choice(data["num_users"], size=size, replace=False)
            scores = z_scores(probs[selected], codes[selected])
            for position, user_id in enumerate(selected):
                row = scores[position]
                order = np.argsort(row)[::-1]
                top = int(order[0])
                second = int(order[1]) if size > 1 else top
                trials.append(
                    {
                        "hit": top == position,
                        "best_z": float(row[top]),
                        "margin": float(row[top] - row[second]),
                    }
                )
        results[str(size)] = summarize_trials(trials, thresholds)
        trial_samples[str(size)] = trials
        print(f"candidate users {size}: {results[str(size)]}")

    summary = {
        "input_json": args.input_json,
        "num_users": data["num_users"],
        "attack": data.get("noise_layer", []),
        "attack_label": describe_attack(data.get("noise_layer", [])),
        "seed": args.seed,
        "candidate_sizes": results,
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    attack_label = summary["attack_label"]
    figure_name = os.path.basename(args.output_png)
    save_plot(results, args.output_png, attack_label, trial_samples)
    write_markdown(summary, args.output_md, attack_label, figure_name)
    print(f"saved: {args.output_json}")
    print(f"saved: {args.output_md}")
    print(f"saved: {args.output_png}")


if __name__ == "__main__":
    main()
