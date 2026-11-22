import argparse
import json
import os
import random
import time


def hamming(a, b):
    return (a ^ b).bit_count()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--message_length", type=int, default=30)
    parser.add_argument("--num_candidates", type=int, default=300000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target_min_distance", type=int, default=11)
    parser.add_argument(
        "--output_json", type=str, default="results/codebook_100x30_min11.json"
    )
    args = parser.parse_args()
    if args.message_length > 31:
        raise ValueError("this greedy implementation supports at most 31 bits")

    rng = random.Random(args.seed)
    seen = set()
    candidates = []
    while len(candidates) < args.num_candidates:
        value = rng.getrandbits(args.message_length)
        if value not in seen:
            seen.add(value)
            candidates.append(value)

    selected = [0, (1 << args.message_length) - 1]
    remaining = [x for x in candidates if x not in (0, (1 << args.message_length) - 1)]
    started = time.time()
    while len(selected) < args.num_users:
        best = None
        best_distance = -1
        for candidate in remaining:
            distance = min(hamming(candidate, chosen) for chosen in selected)
            if distance > best_distance:
                best_distance = distance
                best = candidate
        if best is None:
            raise RuntimeError("candidate pool exhausted")
        selected.append(best)
        remaining.remove(best)

    distances = []
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            distances.append(hamming(selected[i], selected[j]))
    min_distance = min(distances)
    if min_distance < args.target_min_distance:
        raise RuntimeError(
            f"achieved min distance {min_distance}, target {args.target_min_distance}"
        )

    output = {
        "method": "greedy_max_min_hamming",
        "num_users": args.num_users,
        "message_length": args.message_length,
        "num_candidates": args.num_candidates,
        "seed": args.seed,
        "min_distance": min_distance,
        "mean_distance": sum(distances) / len(distances),
        "max_distance": max(distances),
        "elapsed_seconds": time.time() - started,
        "codes": [
            format(value, f"0{args.message_length}b") for value in selected
        ],
        "codes_hex": [format(value, "08x") for value in selected],
    }
    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps({key: output[key] for key in [
        "num_users", "message_length", "min_distance", "mean_distance",
        "max_distance", "elapsed_seconds"
    ]}, indent=2))
    print(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
