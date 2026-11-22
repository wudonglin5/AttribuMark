import argparse
import json

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


def summarize(records, codes, condition):
    top1_hits = []
    top1_z = []
    for record in records:
        z = z_scores(record[condition + "_probs"], codes)
        top = int(np.argmax(z))
        top1_hits.append(top in set(record["colluders"]))
        top1_z.append(float(z[top]))
    top1_hits = np.asarray(top1_hits, dtype=bool)
    top1_z = np.asarray(top1_z, dtype=np.float64)
    n = len(records)
    lines = [
        f"{condition}: n={n} top1={top1_hits.mean():.4f} "
        f"mean_top1_z={top1_z.mean():.4f} median_top1_z={np.median(top1_z):.4f}"
    ]
    for threshold in [2.0, 2.5, 3.0, 3.5, 4.0]:
        accused = top1_z >= threshold
        detected = accused & top1_hits
        false = accused & ~top1_hits
        lines.append(
            f"  Z={threshold:.1f} detect={detected.sum()/n:.4f} "
            f"false_accuse={false.sum()/n:.4f} "
            f"abstain={(~accused).sum()/n:.4f} accused={int(accused.sum())}"
        )
    return lines


def summarize_tiles(records, codes, condition):
    top1_hits = []
    top1_z = []
    coverage = []
    all_colluders = []
    any_colluder = []
    for record in records:
        per_tile = np.stack(
            [
                z_scores(probs, codes)
                for probs in record[condition + "_tile_probs"]
            ]
        )
        zmax = per_tile.max(axis=0)
        top = int(np.argmax(zmax))
        top1_hits.append(top in set(record["colluders"]))
        top1_z.append(float(zmax[top]))
        colluder_z = per_tile[:, np.asarray(record["colluders"])].max(axis=0)
        coverage.append(float(np.mean(colluder_z >= 3.0)))
        all_colluders.append(bool(np.all(colluder_z >= 3.0)))
        any_colluder.append(bool(np.any(colluder_z >= 3.0)))
    top1_hits = np.asarray(top1_hits, dtype=bool)
    top1_z = np.asarray(top1_z, dtype=np.float64)
    n = len(records)
    lines = [
        f"{condition} tiles: n={n} top1={top1_hits.mean():.4f} "
        f"mean_top1_z={top1_z.mean():.4f} median_top1_z={np.median(top1_z):.4f}"
    ]
    for threshold in [2.0, 2.5, 3.0, 3.5, 4.0]:
        accused = top1_z >= threshold
        detected = accused & top1_hits
        false = accused & ~top1_hits
        lines.append(
            f"  Z={threshold:.1f} detect={detected.sum()/n:.4f} "
            f"false_accuse={false.sum()/n:.4f} "
            f"abstain={(~accused).sum()/n:.4f} accused={int(accused.sum())}"
        )
    lines.append(
        f"  Z=3.0 any_colluder={np.mean(any_colluder):.4f} "
        f"all_colluders={np.mean(all_colluders):.4f} "
        f"mean_coverage={np.mean(coverage):.4f}"
    )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True)
    parser.add_argument("--codebook_json", required=True)
    parser.add_argument("--output_txt", default="")
    parser.add_argument("--tile", action="store_true")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    records = data["records"]
    codes = load_codes(args.codebook_json)
    if codes.shape[0] != data["num_users"]:
        raise ValueError("codebook user count does not match experiment")
    if codes.shape[1] != data["message_length"]:
        raise ValueError("codebook message length does not match experiment")

    lines = [
        f"input: {args.input_json}",
        f"num_groups: {data['num_groups']} k: {data['min_k']}..{data['max_k']} "
        f"mix: {data['mix']}",
    ]
    for condition in ["clean", "vqgan"]:
        if args.tile:
            lines.extend(summarize_tiles(records, codes, condition))
        else:
            lines.extend(summarize(records, codes, condition))
    text = "\n".join(lines)
    print(text)
    if args.output_txt:
        with open(args.output_txt, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
