#!/usr/bin/env python3
"""Create a figure that explains what the user fingerprint actually is."""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
DATA_JSON = RESULTS / "user_fingerprint_eval_30k_min11_probs.json"
VIS_DIR = ROOT / "visualization" / "test_128_500"
OUTPUT = DOCS / "fingerprint_explained.png"
USER_ID = 2


def load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def aggregate_z(per_image_llr, codes):
    """Sum per-image candidate scores, normalized by the per-image norms."""
    signs = 2.0 * codes - 1.0
    scores = per_image_llr @ signs.T
    norms = np.sqrt((per_image_llr * per_image_llr).sum(axis=1, keepdims=True))
    return scores.sum(axis=0) / np.sqrt((norms * norms).sum())


def main():
    data = load_json(DATA_JSON)
    codebook = load_json(Path(data["codebook_json"]))
    codes = np.asarray(
        [[int(bit) for bit in code] for code in codebook["codes"]], dtype=np.float64
    )
    codes_hex = codebook["codes_hex"]
    probs = np.asarray(data["image_probs"], dtype=np.float64)
    user_ids = np.asarray(data["image_user_ids"], dtype=np.int64)

    positions = np.flatnonzero(user_ids == USER_ID)
    if positions.size != data["images_per_user"]:
        raise ValueError("unexpected number of images for the selected user")
    p = np.clip(probs[positions], 1e-6, 1.0 - 1e-6)
    llr = np.log(p / (1.0 - p))

    all_z = aggregate_z(llr, codes)
    order = np.argsort(all_z)[::-1]
    top = order[:10]
    true_rank = int(np.flatnonzero(order == USER_ID)[0]) + 1

    watermarked = sorted(VIS_DIR.glob("watermarked_*.png"))
    if len(watermarked) < int(positions.max()) + 1:
        raise ValueError("not enough watermarked images for the selected user")

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.22)

    code_grid = grid[0, 0].subgridspec(1, 4, wspace=0.22)
    displayed_users = [2, 3, 4, 5]
    for index, user_id in enumerate(displayed_users):
        axis = fig.add_subplot(code_grid[0, index])
        bits = codes[user_id].reshape(6, 5)
        axis.imshow(bits, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#2d3748")
            spine.set_linewidth(1.2)
        axis.set_title(f"User {user_id}\n0x{codes_hex[user_id]}", fontsize=10)
    fig.text(
        0.035,
        0.935,
        "(a) User fingerprint = one 30-bit codeword, visualized as a 5 x 6 card",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.035,
        0.505,
        "Black/white is only a display convention; the watermark embeds the 30 bits invisibly.",
        fontsize=9,
        color="#4a5568",
    )

    image_grid = grid[0, 1].subgridspec(1, 5, wspace=0.08)
    for index, global_index in enumerate(positions):
        axis = fig.add_subplot(image_grid[0, index])
        axis.imshow(plt.imread(watermarked[int(global_index)]))
        axis.axis("off")
        axis.set_title(f"Image {index + 1}", fontsize=9)
    fig.text(
        0.545,
        0.935,
        f"(b) Five different images released by user {USER_ID} carry the same invisible code",
        fontsize=12,
        fontweight="bold",
    )

    axis = fig.add_subplot(grid[1, 0])
    top_scores = all_z[top]
    colors = ["#c53030" if int(user_id) == USER_ID else "#a0aec0" for user_id in top]
    axis.bar(np.arange(len(top)), top_scores, color=colors, width=0.72)
    axis.axhline(6.0, color="#2b6cb0", linestyle="--", linewidth=1.4, label="accusation threshold Z=6")
    axis.set_xticks(np.arange(len(top)))
    axis.set_xticklabels([str(int(user_id)) for user_id in top], fontsize=9)
    axis.set_xlabel("Candidate user ID")
    axis.set_ylabel("Aggregated evidence z")
    axis.set_title(f"(c) Candidate lookup from five images: user {USER_ID} ranks #{true_rank}")
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, fontsize=9)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis = fig.add_subplot(grid[1, 1])
    image_counts = np.arange(1, positions.size + 1)
    true_z = []
    for count in image_counts:
        scores = aggregate_z(llr[:count], codes)
        true_z.append(float(scores[USER_ID]))
    axis.plot(image_counts, true_z, marker="o", linewidth=2.4, color="#2f855a")
    for count, value in zip(image_counts, true_z):
        axis.annotate(f"{value:.2f}", (count, value), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    axis.axhline(6.0, color="#2b6cb0", linestyle="--", linewidth=1.4, label="Z=6")
    axis.axhline(10.0, color="#805ad5", linestyle=":", linewidth=1.6, label="Z=10")
    axis.set_xlabel("Number of images from the same user")
    axis.set_ylabel(f"True-user evidence z for user {USER_ID}")
    axis.set_title("(d) More images produce stronger evidence")
    axis.set_xticks(image_counts)
    axis.set_ylim(0, max(true_z) * 1.18)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=9, loc="lower right")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    fig.suptitle(
        "Where is the fingerprint? It is a user codeword, not a visible QR sticker on the face",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUTPUT}")
    print(f"user {USER_ID}: rank={true_rank}, z={all_z[USER_ID]:.4f}")
    print("top10:", [(int(user_id), float(all_z[user_id])) for user_id in top])


if __name__ == "__main__":
    main()
