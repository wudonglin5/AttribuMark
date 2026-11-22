import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ATTACKS = [
    (
        "VQGAN",
        ROOT / "results" / "multimage_llr_summary.json",
        "#4C78A8",
        "o",
        "-",
    ),
    (
        "SimSwap-224",
        ROOT / "results" / "simswap224_multimage_llr_summary.json",
        "#F58518",
        "s",
        "-",
    ),
    (
        "InSwapper-128",
        ROOT / "results" / "inswapper128_multimage_llr_summary.json",
        "#2F855A",
        "^",
        "-",
    ),
]


def load_curves(path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    rows = data["all_subsets"]
    ms = np.asarray(sorted(rows, key=int), dtype=np.int64)
    return {
        "ms": ms,
        "z": np.asarray([rows[str(m)]["llr"]["mean_true_z"] for m in ms]),
        "margin": np.asarray([rows[str(m)]["llr"]["mean_margin"] for m in ms]),
        "detect_z6": np.asarray(
            [rows[str(m)]["llr"]["thresholds"]["6.0"]["detect"] for m in ms]
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_png", type=Path, default=ROOT / "docs" / "multimage_llr_curve.png")
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "multimage_llr_three_attacks.md",
    )
    args = parser.parse_args()

    curves = {}
    for label, path, color, marker, linestyle in ATTACKS:
        curves[label] = load_curves(path)
        curves[label]["color"] = color
        curves[label]["marker"] = marker
        curves[label]["linestyle"] = linestyle

    reference = curves["VQGAN"]
    ms = reference["ms"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0))
    ax_z, ax_margin = axes[0]
    ax_delta, ax_gain = axes[1]

    for label, curve in curves.items():
        ax_z.plot(
            curve["ms"],
            curve["z"],
            color=curve["color"],
            linestyle=curve["linestyle"],
            marker=curve["marker"],
            linewidth=2.0,
            markersize=6,
            label=label,
        )
        ax_margin.plot(
            curve["ms"],
            curve["margin"],
            color=curve["color"],
            linestyle=curve["linestyle"],
            marker=curve["marker"],
            linewidth=2.0,
            markersize=6,
            label=label,
        )

    max_delta = max(
        float(np.max(np.abs(curve["z"] - reference["z"])))
        for label, curve in curves.items()
        if label != "VQGAN"
    )
    ax_z.set_title("(a) Mean true-user evidence")
    ax_z.set_ylabel("Mean z")
    ax_z.text(
        0.04,
        0.96,
        f"All three attacks overlap\nmax |delta z| = {max_delta:.3f}",
        transform=ax_z.transAxes,
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )

    ax_margin.set_title("(b) True-vs-second score margin")
    ax_margin.set_ylabel("Mean z margin")

    width = 0.34
    offset = np.asarray([-width / 2.0, width / 2.0])
    delta_series = [
        (label, curve["z"] - reference["z"])
        for label, curve in curves.items()
        if label != "VQGAN"
    ]
    for index, (label, delta) in enumerate(delta_series):
        curve = curves[label]
        ax_delta.bar(
            ms + offset[index],
            delta,
            width=width,
            color=curve["color"],
            alpha=0.86,
            label=f"{label} - VQGAN",
        )
    ax_delta.axhline(0.0, color="#333333", linewidth=0.8)
    ax_delta.set_title("(c) Difference from VQGAN (zoomed)")
    ax_delta.set_ylabel("Delta mean z")
    ax_delta.legend(frameon=False, fontsize=8)

    for label, curve in curves.items():
        gain = np.diff(curve["z"])
        ax_gain.plot(
            ms[1:],
            gain,
            color=curve["color"],
            linestyle=curve["linestyle"],
            marker=curve["marker"],
            linewidth=2.0,
            markersize=5,
            label=label,
        )
    ax_gain.set_title("(d) Additional evidence from each leaked image")
    ax_gain.set_ylabel("Delta z from m-1 to m")

    for axis in axes.flat:
        axis.set_xlabel("Number of leaked images m")
        axis.set_xticks(ms)
        axis.grid(alpha=0.25, axis="y")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    ax_z.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax_margin.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax_gain.legend(frameon=False, fontsize=8.5, loc="upper right")

    fig.suptitle(
        "Multi-image LLR evidence under three attacks",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "100 users x 5 images; all non-empty image subsets. Curves overlap because attack choice changes z by at most 0.025.",
        ha="center",
        fontsize=8.5,
        color="#475467",
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Multi-image LLR evidence under three attacks",
        "",
        "| m | VQGAN z | SimSwap-224 z | InSwapper-128 z | VQGAN margin | SimSwap-224 margin | InSwapper-128 margin |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for position, m in enumerate(ms):
        values = [curves[label] for label, *_ in ATTACKS]
        lines.append(
            f"| {m} | {values[0]['z'][position]:.3f} | {values[1]['z'][position]:.3f} | "
            f"{values[2]['z'][position]:.3f} | {values[0]['margin'][position]:.3f} | "
            f"{values[1]['margin'][position]:.3f} | {values[2]['margin'][position]:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The largest absolute z difference from VQGAN is {max_delta:.3f}; this is not evidence that one attack is materially stronger.",
            "",
            f"![Multi-image LLR]({args.output_png.name})",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {args.output_png}")
    print(f"saved: {args.output_md}")


if __name__ == "__main__":
    main()
