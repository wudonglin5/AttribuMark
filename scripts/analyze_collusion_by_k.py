import argparse
import json
import math

import numpy as np


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
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--codebook_json", required=True)
    parser.add_argument("--output_txt", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--clean_label", default="clean")
    parser.add_argument("--attack_label", default="vqgan")
    parser.add_argument("--attack_name", default="VQGAN")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = data["records"]
    codes = load_codes(args.codebook_json)
    if codes.shape[0] != data["num_users"]:
        raise ValueError("codebook user count does not match experiment")
    if codes.shape[1] != data["message_length"]:
        raise ValueError("codebook message length does not match experiment")

    thresholds = [2.5, 3.0, 3.5, 4.0]
    lines = [
        f"input: {args.input_json}",
        f"num_groups: {data['num_groups']} k: {data['min_k']}..{data['max_k']} "
        f"mix: {data.get('mix', '')} attack: {args.attack_name}",
        f"identity_offset: {data.get('identity_offset', 0)}",
    ]
    summary = {
        "input_json": args.input_json,
        "mix": data.get("mix", ""),
        "attack": args.attack_name,
        "identity_offset": data.get("identity_offset", 0),
        "num_groups": data["num_groups"],
        "thresholds": thresholds,
        "conditions": {},
    }

    for condition, label in [
        (args.clean_label, "clean"),
        (args.attack_label, args.attack_name),
    ]:
        condition_summary = {}
        lines.append(f"\n[{label}]")
        for k in range(data.get("min_k", min(r["k"] for r in records)), data.get("max_k", max(r["k"] for r in records)) + 1):
            selected = [record for record in records if record["k"] == k]
            if not selected:
                continue
            z = []
            hits = []
            for record in selected:
                scores = z_scores(record[condition + "_probs"], codes)
                top = int(np.argmax(scores))
                z.append(float(scores[top]))
                hits.append(top in set(record["colluders"]))
            z = np.asarray(z, dtype=np.float64)
            hits = np.asarray(hits, dtype=bool)
            n = len(selected)
            ci_low, ci_high = wilson(int(hits.sum()), n)
            lines.append(
                f"k={k} n={n} top1={hits.mean():.4f} "
                f"top1_ci95=[{ci_low:.4f},{ci_high:.4f}] "
                f"mean_top1_z={z.mean():.4f} median_top1_z={np.median(z):.4f}"
            )
            k_summary = {
                "n": n,
                "top1": float(hits.mean()),
                "top1_ci95": [float(ci_low), float(ci_high)],
                "mean_top1_z": float(z.mean()),
                "median_top1_z": float(np.median(z)),
                "operating_points": {},
            }
            for threshold in thresholds:
                accused = z >= threshold
                detected = accused & hits
                false = accused & ~hits
                det_rate = float(detected.sum() / n)
                fa_rate = float(false.sum() / n)
                abstain_rate = float((~accused).sum() / n)
                lines.append(
                    f"  Z={threshold:.1f} detect={det_rate:.4f} false_accuse={fa_rate:.4f} "
                    f"abstain={abstain_rate:.4f} accused={int(accused.sum())}"
                )
                k_summary["operating_points"][str(threshold)] = {
                    "detect": det_rate,
                    "false_accuse": fa_rate,
                    "abstain": abstain_rate,
                    "accused": int(accused.sum()),
                }
            condition_summary[str(k)] = k_summary
        summary["conditions"][label] = condition_summary

    text = "\n".join(lines)
    print(text)
    with open(args.output_txt, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
