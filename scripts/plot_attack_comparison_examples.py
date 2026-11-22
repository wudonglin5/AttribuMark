#!/usr/bin/env python3
"""Build clearly labelled deepfake comparison figures from saved attack outputs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def load_image(path):
    image = mpimg.imread(path)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def draw_rows(rows, columns, path, title, note=None, row_count=4):
    width = 2.05 * len(columns)
    height = 2.22 * row_count + 0.8
    fig, axes = plt.subplots(
        row_count,
        len(columns),
        figsize=(width, height),
        squeeze=False,
    )

    for row in range(row_count):
        for column, (label, directory, suffix) in enumerate(columns):
            image_path = directory / f"{row:04d}_{suffix}.png"
            axis = axes[row, column]
            axis.imshow(load_image(image_path))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color("#d0d5dd")
                spine.set_linewidth(0.8)
            if row == 0:
                axis.set_title(label, fontsize=10.5, fontweight="bold", pad=7)
            if column == 0:
                axis.set_ylabel(
                    f"Sample {row + 1}",
                    fontsize=9,
                    rotation=0,
                    labelpad=35,
                    va="center",
                )

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.985)
    if note:
        fig.text(0.5, 0.012, note, ha="center", fontsize=8.5, color="#475467")
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.93,
        bottom=0.07 if note else 0.035,
        wspace=0.04,
        hspace=0.04,
    )
    fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--simswap_dir", type=Path)
    parser.add_argument("--inswapper_dir", type=Path)
    parser.add_argument("--output_combined", type=Path)
    parser.add_argument("--output_simswap", type=Path)
    parser.add_argument("--output_inswapper", type=Path)
    parser.add_argument("--output_smoke", type=Path)
    parser.add_argument("--output_simswap_example", type=Path)
    args = parser.parse_args()

    simswap_dir = args.simswap_dir or ROOT / "results" / "simswap224_attack_strength_images"
    inswapper_dir = args.inswapper_dir or ROOT / "results" / "inswapper128_attack_strength_images"
    output_combined = args.output_combined or ROOT / "docs" / "deepfake_attack_comparison.png"
    output_simswap = args.output_simswap or ROOT / "docs" / "simswap224_attack_strength_example.png"
    output_inswapper = args.output_inswapper or ROOT / "docs" / "inswapper128_attack_strength_example.png"
    output_smoke = args.output_smoke or ROOT / "docs" / "inswapper128_smoke.png"
    output_simswap_example = (
        args.output_simswap_example or ROOT / "docs" / "simswap224_example.png"
    )

    for directory in (simswap_dir, inswapper_dir):
        if not directory.is_dir():
            raise FileNotFoundError(directory)

    common = [
        ("Original", inswapper_dir, "original"),
        ("Watermarked", inswapper_dir, "watermarked"),
        ("Identity source\n(not an attack output)", inswapper_dir, "identity_source"),
    ]
    draw_rows(
        None,
        common + [("SimSwap-224 output", simswap_dir, "attacked")],
        output_simswap,
        "SimSwap-224 face-swap attack",
        "Columns 1-2 show the protected image; column 3 is a different identity used as the swap source. Column 4 is SimSwap output.",
    )
    draw_rows(
        None,
        common + [("InSwapper-128 output", inswapper_dir, "attacked")],
        output_inswapper,
        "InSwapper-128 face-swap attack",
        "Columns 1-2 show the protected image; column 3 is a different identity used as the swap source. Column 4 is InSwapper output.",
    )
    draw_rows(
        None,
        common
        + [
            ("SimSwap-224 output", simswap_dir, "attacked"),
            ("InSwapper-128 output", inswapper_dir, "attacked"),
        ],
        output_combined,
        "Deepfake attacks on the same protected images",
        "The identity source is deliberately different from the protected face; it is the input identity, not an attack result.",
    )
    draw_rows(
        None,
        common
        + [
            ("SimSwap-224 output", simswap_dir, "attacked"),
            ("InSwapper-128 output", inswapper_dir, "attacked"),
        ],
        output_simswap_example,
        "Face-swap comparison: SimSwap-224 and InSwapper-128",
        "Column 3 is the supplied identity, not an attack output.",
    )
    draw_rows(
        None,
        common + [("InSwapper-128 output", inswapper_dir, "attacked")],
        output_smoke,
        "InSwapper-128 identity replacement",
        "Identity source is deliberately different from the protected face.",
        row_count=5,
    )


if __name__ == "__main__":
    main()
