import argparse
import csv
import json
import math
import os

import numpy as np


THRESHOLDS = np.arange(1.0, 5.01, 0.05)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_codes_from_data(data):
    if "codes" in data:
        codes = data["codes"]
    else:
        path = data.get("codebook_json", "")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"codebook not found: {path!r}")
        with open(path, "r", encoding="utf-8") as handle:
            codes = json.load(handle)["codes"]
    return np.asarray(
        [[int(bit) for bit in code] for code in codes], dtype=np.float64
    )


def z_scores(probs, codes):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    llr = np.log(p / (1.0 - p))
    scores = ((2.0 * codes - 1.0) * llr[None, :]).sum(axis=1)
    return scores / np.sqrt((llr * llr).sum())


def record_outcome(record, condition, codes):
    scores = z_scores(record[condition + "_probs"], codes)
    top = int(np.argmax(scores))
    return float(scores[top]), top in set(record["colluders"])


def bootstrap_mean(values, n_boot, rng):
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n == 0:
        return np.full(n_boot, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    return values[idx].mean(axis=1)


def percentile_ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarize_seed_metric(seed_arrays, n_boot, rng):
    arrays = [np.asarray(a, dtype=np.float64) for a in seed_arrays]
    point = [float(a.mean()) for a in arrays]
    boot = np.stack([bootstrap_mean(a, n_boot, rng) for a in arrays], axis=0)
    per_seed_ci = [percentile_ci(boot[i]) for i in range(boot.shape[0])]
    # Two-stage bootstrap: resample seeds with replacement, then draw an
    # independent unit-bootstrap replicate within each selected seed.
    seed_idx = rng.integers(0, len(arrays), size=(n_boot, len(arrays)))
    hier = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        chosen = seed_idx[b]
        replicas = [boot[j, int(rng.integers(0, n_boot))] for j in chosen]
        hier[b] = np.mean(replicas)
    return {
        "seed_values": point,
        "mean": float(np.mean(point)),
        "min": float(np.min(point)),
        "max": float(np.max(point)),
        "sd": float(np.std(point, ddof=1)) if len(point) > 1 else 0.0,
        "ci95_hierarchical": percentile_ci(hier),
        "ci95_per_seed": per_seed_ci,
    }


def fingerprint_metrics(path):
    data = load_json(path)
    codes = load_codes_from_data(data)
    hamming = np.asarray(data["image_hamming"], dtype=np.float64)
    n_bits = float(data["message_length"])
    ber = hamming / n_bits
    exact = (hamming == 0).astype(np.float64)
    top1 = np.asarray([1.0 if row["correct"] else 0.0 for row in data["per_user"]], dtype=np.float64)
    probs = np.asarray(data["image_probs"], dtype=np.float64)
    user_ids = np.asarray(data["image_user_ids"], dtype=np.int64)
    image_indices = np.asarray(data["processed_image_indices"], dtype=np.int64)

    predictions = (probs >= 0.5).astype(np.int64)
    distances = np.sum(predictions[:, None, :] != codes[None, :, :], axis=-1)
    true_distances = distances[np.arange(len(distances)), user_ids]
    second_best = np.sort(distances, axis=1)[:, 1]
    margin = (second_best - true_distances).astype(np.float64)

    num_users = int(data["num_users"])
    images_per_user = int(data["images_per_user"])
    probs_by_user = np.zeros((num_users, images_per_user, data["message_length"]), dtype=np.float64)
    for position, user_id in enumerate(user_ids):
        slot = int(image_indices[position] - user_id * images_per_user)
        probs_by_user[user_id, slot] = probs[position]
    llr = np.log(np.clip(probs_by_user, 1e-6, 1.0 - 1e-6) / (1.0 - np.clip(probs_by_user, 1e-6, 1.0 - 1e-6)))
    llr_sum = llr.sum(axis=1)
    z_matrix = ((2.0 * codes[None, :, :] - 1.0) * llr_sum[:, None, :]).sum(axis=2)
    z_matrix = z_matrix / np.sqrt((llr * llr).sum(axis=(1, 2)))[:, None]
    true_z = z_matrix[np.arange(num_users), np.arange(num_users)]

    return {
        "ber_image": ber,
        "exact_image": exact,
        "top1_user": top1,
        "hamming_image": hamming,
        "margin_image": margin,
        "max_hamming_seed": np.asarray([hamming.max()], dtype=np.float64),
        "true_z_user": true_z,
    }


def summarize_scalar(values):
    values = [float(v) for v in values]
    return {
        "seed_values": values,
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def collusion_metrics(path):
    data = load_json(path)
    codes = load_codes_from_data(data)
    records = data["records"]
    result = {}
    for k in range(int(data["min_k"]), int(data["max_k"]) + 1):
        k_records = [record for record in records if int(record["k"]) == k]
        for condition in ["clean", "vqgan"]:
            top1 = []
            z = []
            hit = []
            for record in k_records:
                score, is_hit = record_outcome(record, condition, codes)
                top1.append(1.0 if is_hit else 0.0)
                z.append(score)
                hit.append(1.0 if is_hit else 0.0)
            z = np.asarray(z, dtype=np.float64)
            hit = np.asarray(hit, dtype=bool)
            accused = z >= 3.0
            result[(k, condition)] = {
                "top1": np.asarray(top1, dtype=np.float64),
                "detect_z3": (accused & hit).astype(np.float64),
                "false_z3": (accused & ~hit).astype(np.float64),
                "abstain_z3": (~accused).astype(np.float64),
                "z": z,
            }
    return result


def select_threshold(calibration_z, calibration_hit, target_false_accusation):
    best = None
    for threshold in THRESHOLDS:
        accused = calibration_z >= threshold
        n = len(accused)
        false_rate = float(np.sum(accused & ~calibration_hit) / n)
        if false_rate > target_false_accusation:
            continue
        detection = float(np.sum(accused & calibration_hit) / n)
        candidate = (detection, float(threshold), false_rate)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        threshold = float(THRESHOLDS[-1])
        accused = calibration_z >= threshold
        false_rate = float(np.sum(accused & ~calibration_hit) / len(accused))
        return threshold, false_rate
    return best[1], best[2]


def threshold_cv_metrics(path, folds=5, target_false_accusation=0.01):
    data = load_json(path)
    codes = load_codes_from_data(data)
    records = data["records"]
    result = {}
    for k in range(int(data["min_k"]), int(data["max_k"]) + 1):
        k_records = [record for record in records if int(record["k"]) == k]
        outcomes = {}
        for condition in ["clean", "vqgan"]:
            rows = []
            for record in k_records:
                score, is_hit = record_outcome(record, condition, codes)
                rows.append((score, is_hit, int(record["group"])))
            outcomes[condition] = rows

        fold_thresholds = []
        decisions = {"clean": [], "vqgan": []}
        for fold in range(folds):
            cal_z = []
            cal_hit = []
            test = {"clean": [], "vqgan": []}
            for condition in ["clean", "vqgan"]:
                for score, is_hit, group in outcomes[condition]:
                    if group % folds == fold:
                        test[condition].append((score, is_hit, group))
                    else:
                        cal_z.append(score)
                        cal_hit.append(is_hit)
            cal_z = np.asarray(cal_z, dtype=np.float64)
            cal_hit = np.asarray(cal_hit, dtype=bool)
            threshold, _ = select_threshold(cal_z, cal_hit, target_false_accusation)
            fold_thresholds.append(threshold)
            for condition in ["clean", "vqgan"]:
                for score, is_hit, group in test[condition]:
                    decisions[condition].append((threshold, score, is_hit, group))

        for condition in ["clean", "vqgan"]:
            rows = decisions[condition]
            z = np.asarray([row[1] for row in rows], dtype=np.float64)
            hit = np.asarray([row[2] for row in rows], dtype=bool)
            thr = np.asarray([row[0] for row in rows], dtype=np.float64)
            accused = z >= thr
            result[(k, condition)] = {
                "detect": (accused & hit).astype(np.float64),
                "false": (accused & ~hit).astype(np.float64),
                "abstain": (~accused).astype(np.float64),
                "fold_thresholds": np.asarray(fold_thresholds, dtype=np.float64),
            }
    return result


def add_metric(store, key, arrays, n_boot, rng):
    store[key] = summarize_seed_metric(arrays, n_boot, rng)


def fmt_metric(metric, scale=100.0):
    lo, hi = metric["ci95_hierarchical"]
    return f"{scale * metric['mean']:.2f} [{scale * metric['min']:.2f}, {scale * metric['max']:.2f}] ({scale * lo:.2f}-{scale * hi:.2f})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=12345)
    args = parser.parse_args()

    rng = np.random.default_rng(args.bootstrap_seed)
    fingerprints = {}
    for name in ["vqgan", "simswap224", "inswapper128"]:
        per_seed = []
        for seed in args.seeds:
            path = os.path.join(args.seed_root, f"seed{seed}", f"fingerprint_{name}.json")
            per_seed.append(fingerprint_metrics(path))
        fingerprints[name] = {}
        for metric in ["ber_image", "exact_image", "top1_user", "hamming_image", "margin_image", "true_z_user"]:
            add_metric(
                fingerprints[name],
                metric,
                [item[metric] for item in per_seed],
                args.n_boot,
                rng,
            )
        fingerprints[name]["max_hamming"] = summarize_scalar(
            [item["max_hamming_seed"][0] for item in per_seed]
        )

    collusion_specs = {
        "VQGAN-mean": "collusion_vqgan_mean.json",
        "SimSwap-mean": "collusion_simswap224_mean.json",
        "SimSwap-median": "collusion_simswap224_median.json",
        "SimSwap-mosaic": "collusion_simswap224_mosaic.json",
    }
    collusion = {}
    for label, filename in collusion_specs.items():
        per_seed = []
        for seed in args.seeds:
            path = os.path.join(args.seed_root, f"seed{seed}", filename)
            per_seed.append(collusion_metrics(path))
        keys = sorted(per_seed[0].keys())
        collusion[label] = {}
        for (k, condition) in keys:
            key = f"k{k}_{condition}"
            collusion[label][key] = {}
            for metric in ["top1", "detect_z3", "false_z3", "abstain_z3", "z"]:
                add_metric(
                    collusion[label][key],
                    metric,
                    [item[(k, condition)][metric] for item in per_seed],
                    args.n_boot,
                    rng,
                )

    threshold_cv = {}
    for label, filename in collusion_specs.items():
        per_seed = []
        for seed in args.seeds:
            path = os.path.join(args.seed_root, f"seed{seed}", filename)
            per_seed.append(threshold_cv_metrics(path))
        keys = sorted(per_seed[0].keys())
        threshold_cv[label] = {}
        for (k, condition) in keys:
            key = f"k{k}_{condition}"
            threshold_cv[label][key] = {}
            for metric in ["detect", "false", "abstain"]:
                add_metric(
                    threshold_cv[label][key],
                    metric,
                    [item[(k, condition)][metric] for item in per_seed],
                    args.n_boot,
                    rng,
                )
            thresholds = [item[(k, condition)]["fold_thresholds"] for item in per_seed]
            threshold_cv[label][key]["fold_thresholds"] = {
                "seed_values": [float(np.median(t)) for t in thresholds],
                "mean": float(np.mean([np.median(t) for t in thresholds])),
                "min": float(np.min([np.min(t) for t in thresholds])),
                "max": float(np.max([np.max(t) for t in thresholds])),
            }

    output = {
        "seeds": args.seeds,
        "n_boot": args.n_boot,
        "bootstrap_seed": args.bootstrap_seed,
        "fingerprint": fingerprints,
        "collusion": collusion,
        "threshold_cv": threshold_cv,
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)

    rows = []
    for name, metrics in fingerprints.items():
        for metric in ["ber_image", "exact_image", "top1_user", "hamming_image", "margin_image", "true_z_user"]:
            scale = 1.0 if metric in ("hamming_image", "margin_image", "true_z_user") else 100.0
            rows.append(["fingerprint", name, metric, fmt_metric(metrics[metric], scale)])
        rows.append(["fingerprint", name, "max_hamming", f"{metrics['max_hamming']['mean']:.3f} [{metrics['max_hamming']['min']:.0f}, {metrics['max_hamming']['max']:.0f}]"])
    for label, ks in collusion.items():
        for key, metrics in ks.items():
            for metric in ["top1", "detect_z3", "false_z3", "abstain_z3", "z"]:
                scale = 1.0 if metric == "z" else 100.0
                rows.append(["collusion", label, key + "/" + metric, fmt_metric(metrics[metric], scale)])
    for label, ks in threshold_cv.items():
        for key, metrics in ks.items():
            for metric in ["detect", "false", "abstain"]:
                rows.append(["threshold_cv", label, key + "/" + metric, fmt_metric(metrics[metric], 100.0)])
            rows.append(["threshold_cv", label, key + "/threshold_median", f"{metrics['fold_thresholds']['mean']:.3f}"])
    with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["section", "condition", "metric", "mean [seed_min, seed_max] (CI95)"])
        writer.writerows(rows)
    print(f"saved: {args.output_json}")
    print(f"saved: {args.output_csv}")


if __name__ == "__main__":
    main()
