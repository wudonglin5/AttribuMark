# -*- coding: utf-8 -*-
"""Recompute collusion attribution under three explicit semantics.

Metrics use the same soft LLR score as the manuscript:
    z_u = <(2*c_u - 1), log(p/(1-p))> / ||log(p/(1-p))||_2

any-colluder : argmax z is any member of the colluding set
innocent     : argmax z is not a colluder (false accusation)
rank/margin  : rank of the best colluder and its z margin over the strongest innocent
first        : sensitivity only; symmetric mixtures have no privileged main colluder
"""
import argparse
import json
import math
import os
import statistics


def load_codes(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [[int(bit) for bit in code] for code in data["codes"]]


def llr_scores(probs, codes):
    llr = []
    for p in probs:
        p = min(max(float(p), 1e-6), 1.0 - 1e-6)
        llr.append(math.log(p / (1.0 - p)))
    denom = math.sqrt(sum(v * v for v in llr))
    if denom == 0.0:
        return [0.0 for _ in codes]
    return [
        sum((1.0 if bit else -1.0) * v for bit, v in zip(code, llr)) / denom
        for code in codes
    ]


def analyze_record(record, prob_key, hard_key):
    probs = record.get(prob_key)
    if not probs:
        return None
    codes = record["_codes"]
    scores = llr_scores(probs, codes)
    colluders = list(record["colluders"])
    colluder_set = set(colluders)
    best_colluder_score = max(scores[u] for u in colluders)
    innocent = [u for u in range(len(codes)) if u not in colluder_set]
    best_innocent_score = max(scores[u] for u in innocent)
    top = max(range(len(scores)), key=lambda u: scores[u])
    rank = 1 + sum(1 for u in innocent if scores[u] > best_colluder_score)
    hard = record.get(hard_key) or {}
    return {
        "top": top,
        "any_colluder": top in colluder_set,
        "first_colluder": top == colluders[0],
        "innocent": top not in colluder_set,
        "rank": rank,
        "margin": best_colluder_score - best_innocent_score,
        "unique_separation": best_colluder_score > best_innocent_score,
        "tie_with_innocent": best_colluder_score == best_innocent_score,
        "exact_code_any": bool(hard.get("exact_match_any_colluder", False)),
        "top_z": scores[top],
        "k": len(colluders),
    }


def summarize(rows):
    n = len(rows)

    def rate(key):
        return 100.0 * sum(1 for row in rows if row[key]) / n

    ranks = [row["rank"] for row in rows]
    margins = [row["margin"] for row in rows]
    return {
        "n": n,
        "any_colluder_rate": rate("any_colluder"),
        "first_colluder_rate": rate("first_colluder"),
        "innocent_rate": rate("innocent"),
        "unique_separation_rate": rate("unique_separation"),
        "tie_with_innocent_rate": rate("tie_with_innocent"),
        "exact_code_any_rate": rate("exact_code_any"),
        "mean_best_colluder_rank": statistics.mean(ranks),
        "median_best_colluder_rank": statistics.median(ranks),
        "max_best_colluder_rank": max(ranks),
        "mean_margin": statistics.mean(margins),
        "median_margin": statistics.median(margins),
        "min_margin": min(margins),
        "mean_top_z": statistics.mean([row["top_z"] for row in rows]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=r"F:\project\beifen\DiffMark-main")
    parser.add_argument("--out_json", default="")
    parser.add_argument("--out_md", default="")
    args = parser.parse_args()

    root = args.root
    codes = load_codes(os.path.join(root, "results", "codebook_100x30_min11.json"))
    seed_root = os.path.join(root, "results", "seed_stability", "seed{seed}")

    sources = []
    for condition, filename in [
        ("VQGAN-mean", "collusion_vqgan_mean.json"),
        ("SimSwap-mean", "collusion_simswap224_mean.json"),
        ("SimSwap-median", "collusion_simswap224_median.json"),
        ("SimSwap-mosaic", "collusion_simswap224_mosaic.json"),
    ]:
        for seed in (2026, 2027, 2028):
            sources.append((condition, seed, os.path.join(seed_root.format(seed=seed), filename)))
    for condition, filename in [
        ("InSwapper-mean", "user_collusion_eval_30k_min11_inswapper128_mean_100groups.json"),
        ("InSwapper-median", "user_collusion_eval_30k_min11_inswapper128_median_100groups.json"),
        ("InSwapper-mosaic", "user_collusion_eval_30k_min11_inswapper128_mosaic_100groups.json"),
    ]:
        sources.append((condition, 0, os.path.join(root, "results", filename)))

    records = []
    for condition, seed, path in sources:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        for record in data["records"]:
            record["_codes"] = codes
            for stage, prob_key, hard_key in [
                ("clean", "clean_probs", "clean"),
                ("attack", "vqgan_probs", "vqgan"),
            ]:
                row = analyze_record(record, prob_key, hard_key)
                if row is None:
                    continue
                row.update(condition=condition, seed=seed, k=record["k"],
                           group=record["group"], stage=stage)
                records.append(row)

    summary = {}
    conditions = ["VQGAN-mean", "SimSwap-mean", "SimSwap-median", "SimSwap-mosaic",
                  "InSwapper-mean", "InSwapper-median", "InSwapper-mosaic"]
    for condition in conditions:
        for stage in ("clean", "attack"):
            for k in range(2, 6):
                rows = [row for row in records if row["condition"] == condition
                        and row["stage"] == stage and row["k"] == k]
                if rows:
                    summary[f"{condition}|{stage}|{k}"] = summarize(rows)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)

    lines = []
    for condition in conditions:
        for stage in ("attack", "clean"):
            for k in range(2, 6):
                key = f"{condition}|{stage}|{k}"
                if key not in summary:
                    continue
                row = summary[key]
                lines.append(
                    f"{condition:18s} {stage:6s} k={k} n={row['n']:4d} "
                    f"any={row['any_colluder_rate']:6.2f} "
                    f"innocent={row['innocent_rate']:6.2f} "
                    f"unique={row['unique_separation_rate']:6.2f} "
                    f"rank={row['mean_best_colluder_rank']:4.2f} "
                    f"margin={row['mean_margin']:5.2f} "
                    f"first={row['first_colluder_rate']:5.2f} "
                    f"exact={row['exact_code_any_rate']:5.2f}"
                )
    text = "\n".join(lines)
    print(text)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as handle:
            handle.write(
                "# Collusion attribution: any-colluder / innocent / rank-margin\n\n"
                "Main semantics follow the manuscript: correct if argmax LLR is a "
                "member of the colluding set; false accusation if it is an innocent "
                "candidate. `first` is a sensitivity check only and is not a claim.\n\n"
                "```\n" + text + "\n```\n"
            )


if __name__ == "__main__":
    main()
