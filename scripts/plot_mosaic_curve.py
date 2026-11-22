import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PATHS = {
    2: "results/user_collusion_eval_30k_min11_mosaic_k2_100groups.json",
    3: "results/user_collusion_eval_30k_min11_mosaic_k3_100groups.json",
    4: "results/user_collusion_eval_30k_min11_mosaic_k4_100groups.json",
    5: "results/user_collusion_eval_30k_min11_mosaic_k5_100groups.json",
}


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


def metrics(records, codes, condition, threshold=3.0):
    hits = []
    top_z = []
    for record in records:
        z = z_scores(record[condition + "_probs"], codes)
        top = int(np.argmax(z))
        hits.append(top in set(record["colluders"]))
        top_z.append(float(z[top]))
    hits = np.asarray(hits, dtype=bool)
    top_z = np.asarray(top_z, dtype=np.float64)
    accused = top_z >= threshold
    return {
        "top1": float(hits.mean()),
        "detect": float((accused & hits).mean()),
        "false_accuse": float((accused & ~hits).mean()),
        "abstain": float((~accused).mean()),
    }


def main():
    codes = load_codes("results/codebook_100x30_min11.json")
    summary = {}
    for k, path in PATHS.items():
        with open(path, "r", encoding="utf-8") as handle:
            records = json.load(handle)["records"]
        summary[k] = {
            condition: metrics(records, codes, condition)
            for condition in ["clean", "vqgan"]
        }

    ks = sorted(summary)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for condition, label, color in [
        ("clean", "Clean mosaic", "#1f77b4"),
        ("vqgan", "VQGAN mosaic", "#d62728"),
    ]:
        axes[0].plot(
            ks, [summary[k][condition]["top1"] for k in ks], "o-", label=label, color=color
        )
        axes[1].plot(
            ks,
            [summary[k][condition]["detect"] for k in ks],
            "o-",
            label=label,
            color=color,
        )
    axes[0].set_title("Top-1 suspect accuracy")
    axes[1].set_title("Detection at Z=3.0")
    for axis in axes:
        axis.set_xlabel("Number of colluders k")
        axis.set_ylabel("Rate")
        axis.set_ylim(0.0, 1.05)
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig("docs/mosaic_curve.png", dpi=200)
    with open("results/mosaic_curve_summary.json", "w", encoding="utf-8") as handle:
        json.dump({str(k): value for k, value in summary.items()}, handle, indent=2)
    for k in ks:
        print(k, summary[k])


if __name__ == "__main__":
    main()
