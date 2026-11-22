import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def series(summary, condition, field, threshold=None):
    data = summary["conditions"][condition]
    ks = sorted(int(k) for k in data)
    if threshold is None:
        values = [data[str(k)][field] for k in ks]
    else:
        values = [data[str(k)]["operating_points"][threshold][field] for k in ks]
    return ks, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--output_png", required=True)
    parser.add_argument("--attack_name", default="SimSwap")
    parser.add_argument("--vqgan_summary_json", default="")
    args = parser.parse_args()

    with open(args.summary_json, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    vqgan = None
    if args.vqgan_summary_json:
        with open(args.vqgan_summary_json, "r", encoding="utf-8") as handle:
            vqgan = json.load(handle)

    ks, clean_top1 = series(summary, "clean", "top1")
    _, attack_top1 = series(summary, args.attack_name, "top1")
    _, clean_det = series(summary, "clean", "detect", "3.0")
    _, attack_det = series(summary, args.attack_name, "detect", "3.0")
    _, clean_fa = series(summary, "clean", "false_accuse", "3.0")
    _, attack_fa = series(summary, args.attack_name, "false_accuse", "3.0")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(ks, clean_top1, "o-", label="clean")
    if vqgan is not None:
        _, vqgan_top1 = series(vqgan, "VQGAN", "top1")
        axes[0].plot(ks, vqgan_top1, "^-", label="VQGAN after collusion")
    axes[0].plot(ks, attack_top1, "s-", label=f"{args.attack_name} after collusion")
    axes[0].set_title("Soft top-1 attribution")
    axes[0].set_ylabel("Accuracy")

    axes[1].plot(ks, clean_det, "o-", label="clean, Z=3.0")
    if vqgan is not None:
        _, vqgan_det = series(vqgan, "VQGAN", "detect", "3.0")
        axes[1].plot(ks, vqgan_det, "^-", label="VQGAN, Z=3.0")
    axes[1].plot(ks, attack_det, "s-", label=f"{args.attack_name}, Z=3.0")
    axes[1].plot(ks, clean_fa, "o--", alpha=0.5, label="clean false accusation")
    if vqgan is not None:
        _, vqgan_fa = series(vqgan, "VQGAN", "false_accuse", "3.0")
        axes[1].plot(ks, vqgan_fa, "^--", alpha=0.5, label="VQGAN false accusation")
    axes[1].plot(ks, attack_fa, "s--", alpha=0.5, label=f"{args.attack_name} false accusation")
    axes[1].set_title("Detection at conservative threshold")
    axes[1].set_ylabel("Rate")

    for axis in axes:
        axis.set_xlabel("Number of colluders k")
        axis.set_xticks(ks)
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_png, dpi=200)
    print("saved", args.output_png)


if __name__ == "__main__":
    main()
