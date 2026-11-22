#!/usr/bin/env python3
"""Compare per-image decoding under three attacks and two user protocols."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

ATTACKS = (
    ("VQGAN", "#4C78A8"),
    ("SimSwap-224", "#F58518"),
    ("InSwapper-128", "#2F855A"),
)

PROTOCOLS = (
    (
        "100 users x 5 images",
        (
            "user_fingerprint_eval_30k_min11_probs.json",
            "user_fingerprint_eval_30k_min11_simswap224_probs.json",
            "user_fingerprint_eval_30k_min11_inswapper128_probs.json",
        ),
    ),
    (
        "500 users x 1 image",
        (
            "user_fingerprint_eval_30k_500users_bch_vqgan_probs.json",
            "user_fingerprint_eval_30k_500users_bch_simswap224_probs.json",
            "user_fingerprint_eval_30k_500users_bch_inswapper128_probs.json",
        ),
    ),
)


def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_png",
        type=Path,
        default=ROOT / "docs" / "single_image_attack_comparison.png",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=ROOT / "docs" / "single_image_attack_comparison.md",
    )
    args = parser.parse_args()

    rows = []
    for protocol, filenames in PROTOCOLS:
        for (attack, _), filename in zip(ATTACKS, filenames):
            data = load(ROOT / "results" / filename)
            rows.append(
                {
                    "protocol": protocol,
                    "attack": attack,
                    "ber": 100.0 * data["per_image"]["ber"],
                    "exact": 100.0 * data["per_image"]["exact_match"],
                    "hamming": data["per_image"]["mean_hamming"],
                    "max_hamming": data["per_image"]["max_hamming"],
                    "top1": 100.0 * data["final"]["user_identification_accuracy"],
                    "users": data["num_users"],
                    "images_per_user": data["images_per_user"],
                }
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    protocols = [label for label, _ in PROTOCOLS]
    x = np.arange(len(protocols))
    width = 0.24

    for index, (attack, color) in enumerate(ATTACKS):
        values_ber = [
            next(
                row["ber"]
                for row in rows
                if row["protocol"] == protocol and row["attack"] == attack
            )
            for protocol in protocols
        ]
        values_exact = [
            next(
                row["exact"]
                for row in rows
                if row["protocol"] == protocol and row["attack"] == attack
            )
            for protocol in protocols
        ]
        offset = (index - 1) * width
        bars_ber = axes[0].bar(
            x + offset,
            values_ber,
            width=width,
            color=color,
            label=attack,
        )
        bars_exact = axes[1].bar(
            x + offset,
            values_exact,
            width=width,
            color=color,
            label=attack,
        )
        for bar in bars_ber:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.002,
                f"{bar.get_height():.4f}%",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )
        for bar in bars_exact:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.05,
                f"{bar.get_height():.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )

    axes[0].set_title("(a) Bit-error rate")
    axes[0].set_ylabel("BER (percent)")
    max_ber = max(row["ber"] for row in rows)
    axes[0].set_ylim(0.0, max_ber * 1.35)
    axes[1].set_title("(b) Fully correct 30-bit messages")
    axes[1].set_ylabel("Exact match (percent)")
    axes[1].set_ylim(90.0, 100.4)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(protocols)
        axis.grid(alpha=0.25, axis="y")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(
            frameon=True,
            framealpha=0.94,
            edgecolor="#d0d5dd",
            fontsize=8.5,
            loc="lower right",
        )

    fig.suptitle(
        "Per-image watermark decoding under three attacks",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.012,
        "The two groups are separate protocols: 500 images from 100 users, and 500 images from 500 users.",
        ha="center",
        fontsize=8.4,
        color="#475467",
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Per-image decoding under three attacks",
        "",
        "| Protocol | Attack | BER | Exact 30-bit | Mean Hamming | Max Hamming | User top-1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['protocol']} | {row['attack']} | {row['ber']:.4f}% | "
            f"{row['exact']:.1f}% | {row['hamming']:.3f} | {row['max_hamming']} | "
            f"{row['top1']:.1f}% |"
        )
    lines.extend(
        [
            "",
            "The 500-user protocol uses one image per user and is not a 500-user multi-image experiment.",
            "All three attacks remain far below one bit error per image on average; small differences should not be presented as a significant attack ranking.",
            "",
            f"![Single-image comparison]({args.output_png.name})",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {args.output_png}")
    print(f"saved: {args.output_md}")


if __name__ == "__main__":
    main()
