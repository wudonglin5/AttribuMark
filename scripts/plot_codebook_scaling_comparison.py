#!/usr/bin/env python3
"""Compare candidate-pool scaling under VQGAN, SimSwap, and InSwapper."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ATTACKS = (
    ("VQGAN", "codebook_scaling_500users.json", "#4C78A8", "o", "--"),
    (
        "SimSwap-224",
        "codebook_scaling_500users_simswap224.json",
        "#F58518",
        "s",
        "-",
    ),
    (
        "InSwapper-128",
        "codebook_scaling_500users_inswapper128.json",
        "#2F855A",
        "^",
        "-",
    ),
)


def load_result(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_png",
        type=Path,
        default=ROOT / "docs" / "codebook_scaling_three_attacks.png",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "codebook_scaling_three_attacks.md",
    )
    args = parser.parse_args()

    curves = {}
    for label, filename, color, marker, linestyle in ATTACKS:
        data = load_result(ROOT / "results" / filename)
        sizes = sorted(data["candidate_sizes"], key=int)
        curves[label] = {
            "sizes": np.asarray(sizes, dtype=np.int64),
            "mean_margin": np.asarray(
                [data["candidate_sizes"][size]["mean_margin"] for size in sizes]
            ),
            "min_margin": np.asarray(
                [data["candidate_sizes"][size]["min_margin"] for size in sizes]
            ),
            "top1": np.asarray(
                [data["candidate_sizes"][size]["top1"] for size in sizes]
            ),
            "false_accuse": np.asarray(
                [
                    data["candidate_sizes"][size]["operating_points"]["3.0"][
                        "false_accuse"
                    ]
                    for size in sizes
                ]
            ),
            "color": color,
            "marker": marker,
            "linestyle": linestyle,
        }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    ax_mean, ax_min = axes

    for label, curve in curves.items():
        ax_mean.plot(
            curve["sizes"],
            curve["mean_margin"],
            color=curve["color"],
            linestyle=curve["linestyle"],
            marker=curve["marker"],
            linewidth=2.2,
            markersize=7,
            label=label,
        )
        ax_min.plot(
            curve["sizes"],
            curve["min_margin"],
            color=curve["color"],
            linestyle=curve["linestyle"],
            marker=curve["marker"],
            linewidth=2.2,
            markersize=7,
            label=label,
        )

    ax_mean.set_title("(a) Mean true-vs-second z margin")
    ax_mean.set_ylabel("Mean margin")
    ax_min.set_title("(b) Worst-case true-vs-second z margin")
    ax_min.set_ylabel("Minimum margin")

    for axis in axes:
        axis.set_xlabel("Candidate users")
        axis.set_xticks(curves["VQGAN"]["sizes"])
        axis.grid(alpha=0.25, axis="y")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    ax_mean.legend(frameon=False, fontsize=8.5)
    ax_min.legend(frameon=False, fontsize=8.5)

    final_row = curves["InSwapper-128"]
    fig.text(
        0.5,
        0.012,
        f"Top-1 at 500 users: VQGAN {curves['VQGAN']['top1'][-1]:.0%}, "
        f"SimSwap {curves['SimSwap-224']['top1'][-1]:.0%}, "
        f"InSwapper {final_row['top1'][-1]:.0%}. "
        "The margin curves show how evidence survives a larger candidate pool.",
        ha="center",
        fontsize=8.3,
        color="#475467",
    )
    fig.suptitle(
        "Candidate-pool scaling under three attacks",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Candidate-pool scaling under three attacks",
        "",
        "| Candidates | VQGAN mean margin | SimSwap mean margin | InSwapper mean margin | VQGAN min | SimSwap min | InSwapper min |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for index, size in enumerate(curves["VQGAN"]["sizes"]):
        lines.append(
            f"| {size} | {curves['VQGAN']['mean_margin'][index]:.3f} | "
            f"{curves['SimSwap-224']['mean_margin'][index]:.3f} | "
            f"{curves['InSwapper-128']['mean_margin'][index]:.3f} | "
            f"{curves['VQGAN']['min_margin'][index]:.3f} | "
            f"{curves['SimSwap-224']['min_margin'][index]:.3f} | "
            f"{curves['InSwapper-128']['min_margin'][index]:.3f} |"
        )
    lines.extend(
        [
            "",
            "All three attacks retain a positive worst-case margin, but the user identification claim is limited to the sampled 500-user candidate pool.",
            "",
            f"![Codebook scaling]({args.output_png.name})",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {args.output_png}")
    print(f"saved: {args.output_md}")


if __name__ == "__main__":
    main()
