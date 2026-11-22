import argparse
import json
import os
import time

import numpy as np
import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.noiser import Random_Noise
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_endecoder_and_diffusion,
    endecoder_and_diffusion_defaults,
    str2bool,
)


def build_codebook(num_users, message_length, seed):
    rng = np.random.RandomState(seed)
    codes = []
    seen = set()
    while len(codes) < num_users:
        code = rng.randint(0, 2, size=message_length).astype(np.int64)
        key = "".join(str(int(x)) for x in code)
        if key in seen:
            continue
        seen.add(key)
        codes.append(code)
    return np.stack(codes, axis=0)


def load_codebook(path, num_users, message_length):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    codes = np.asarray(
        [[int(bit) for bit in code] for code in data["codes"]], dtype=np.int64
    )
    if codes.shape[0] < num_users or codes.shape[1] != message_length:
        raise ValueError(
            f"codebook shape {codes.shape} does not match "
            f"num_users={num_users}, message_length={message_length}"
        )
    return codes[:num_users]


def decode_binary_probabilities(logits):
    if logits.ndim != 3:
        raise ValueError(f"unexpected decoder output shape: {tuple(logits.shape)}")
    _, message_length, classes = logits.shape
    if classes != 2 * message_length:
        raise ValueError(
            "decoder output must have 2 * message_length classes, "
            f"got {tuple(logits.shape)}"
        )
    positions = th.arange(message_length, device=logits.device)
    pair_logits = th.stack(
        [logits[:, positions, positions], logits[:, positions, positions + message_length]],
        dim=-1,
    )
    return th.softmax(pair_logits, dim=-1)[..., 1]


def score_probs(probs, codes, colluders, invalid_threshold):
    message_length = codes.shape[1]
    predictions = (probs >= 0.5).astype(np.int64)
    distances = np.sum(predictions[None, :] != codes, axis=1)
    nearest = int(np.argmin(distances))
    nearest_hamming = int(distances[nearest])
    colluder_distances = distances[np.asarray(colluders, dtype=np.int64)]
    best_colluder_hamming = int(colluder_distances.min())
    sorted_distances = np.sort(distances)
    second_best = int(sorted_distances[1]) if len(sorted_distances) > 1 else nearest_hamming
    confidence = float(np.maximum(probs, 1.0 - probs).mean())
    nearest_is_colluder = nearest in colluders
    return {
        "nearest_user": nearest,
        "nearest_hamming": nearest_hamming,
        "best_colluder_hamming": best_colluder_hamming,
        "ber_best_colluder": float(best_colluder_hamming / message_length),
        "exact_match_any_colluder": bool(best_colluder_hamming == 0),
        "nearest_is_colluder": bool(nearest_is_colluder),
        "nearest_is_innocent": bool(not nearest_is_colluder),
        "false_accusation": bool(not nearest_is_colluder),
        "invalid_fingerprint": bool(best_colluder_hamming > invalid_threshold),
        "second_best_margin": int(second_best - nearest_hamming),
        "mean_confidence": confidence,
    }


def aggregate(records):
    keys_bool = [
        "exact_match_any_colluder",
        "nearest_is_colluder",
        "nearest_is_innocent",
        "false_accusation",
        "invalid_fingerprint",
    ]
    keys_float = [
        "nearest_hamming",
        "best_colluder_hamming",
        "ber_best_colluder",
        "second_best_margin",
        "mean_confidence",
    ]
    result = {"num_groups": len(records)}
    for key in keys_bool:
        result[key + "_rate"] = float(np.mean([float(item[key]) for item in records]))
    for key in keys_float:
        result["mean_" + key] = float(np.mean([item[key] for item in records]))
    result["max_best_colluder_hamming"] = int(
        max(item["best_colluder_hamming"] for item in records)
    )
    return result


def mosaic_boxes(k, height, width):
    rows = int(np.floor(np.sqrt(k)))
    cols = int(np.ceil(k / rows))
    row_edges = np.linspace(0, height, rows + 1).round().astype(np.int64)
    col_edges = np.linspace(0, width, cols + 1).round().astype(np.int64)
    boxes = []
    for row in range(rows):
        for col in range(cols):
            top, bottom = row_edges[row], row_edges[row + 1]
            left, right = col_edges[col], col_edges[col + 1]
            boxes.append((int(top), int(bottom), int(left), int(right)))
    return boxes


def mosaic_mix(samples, rng):
    k, _, height, width = samples.shape
    boxes = mosaic_boxes(k, height, width)
    order = rng.permutation(k)
    mixed = samples[order[0] : order[0] + 1].clone()
    for cell, (top, bottom, left, right) in enumerate(boxes):
        source = int(order[cell % k])
        mixed[:, :, top:bottom, left:right] = samples[
            source : source + 1, :, top:bottom, left:right
        ]
    return mixed


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    device = dist_util.dev()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    if args.codebook_json:
        codes = load_codebook(args.codebook_json, args.num_users, args.message_length)
    else:
        codes = build_codebook(args.num_users, args.message_length, args.code_seed)
    required_covers = args.num_groups
    data_loader = load_data(
        data_dir=args.data_dir,
        batch_size=min(args.batch_size, required_covers),
        image_size=args.image_size,
        deterministic=True,
    )
    covers = []
    for batch in data_loader:
        for image in batch:
            covers.append(image)
            if len(covers) >= required_covers:
                break
        if len(covers) >= required_covers:
            break
    covers = th.stack(covers, dim=0)
    if covers.shape[0] < required_covers:
        raise ValueError(
            f"need {required_covers} covers, but only found {covers.shape[0]}"
        )

    logger.log(f"checkpoint: {args.model_path}")
    logger.log(f"users: {args.num_users}, groups: {args.num_groups}")
    logger.log(f"collusion sizes: {args.min_k}..{args.max_k}")
    logger.log(f"mix: {args.mix}")

    model, diffusion = create_endecoder_and_diffusion(
        **args_to_dict(args, endecoder_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    noise_layers = [item.strip() for item in args.noise_layer.split(";") if item.strip()]
    noise_layer = Random_Noise(noise_layers)
    collusion_rng = np.random.RandomState(args.collusion_seed)
    mosaic_rng = np.random.RandomState(args.collusion_seed + 100003)
    invalid_threshold = int(np.floor(args.invalid_ber * args.message_length))

    records = []
    started = time.time()
    with th.no_grad():
        for k in range(args.min_k, args.max_k + 1):
            for group_index in range(args.num_groups):
                colluders = collusion_rng.choice(
                    args.num_users, size=k, replace=False
                ).astype(np.int64)
                cover = covers[group_index : group_index + 1].to(device)
                if args.identity_offset:
                    source_index = (group_index + args.identity_offset) % args.num_groups
                    attack_identity = covers[source_index : source_index + 1].to(device)
                else:
                    attack_identity = cover
                cover_batch = cover.repeat(k, 1, 1, 1)
                message = th.from_numpy(codes[colluders]).float().to(device)
                samples = diffusion.ddim_sample_loop(
                    model,
                    cover_batch,
                    message,
                    clip_denoised=args.clip_denoised,
                    model_kwargs={},
                )
                if args.mix == "mean":
                    mixed = samples.mean(dim=0, keepdim=True)
                elif args.mix == "median":
                    sorted_samples = samples.sort(dim=0).values
                    if k % 2 == 1:
                        mixed = sorted_samples[k // 2 : k // 2 + 1]
                    else:
                        mixed = 0.5 * (
                            sorted_samples[k // 2 - 1 : k // 2]
                            + sorted_samples[k // 2 : k // 2 + 1]
                        )
                elif args.mix == "mosaic":
                    mixed = mosaic_mix(samples, mosaic_rng)
                else:
                    raise ValueError(f"unsupported mix: {args.mix}")
                probs_clean = decode_binary_probabilities(model.decoder(mixed))[0]
                distorted = noise_layer([mixed, attack_identity])
                probs_distorted = decode_binary_probabilities(
                    model.decoder(distorted)
                )[0]
                clean_tile_probs = []
                vqgan_tile_probs = []
                if args.mix == "mosaic" and args.save_tiles:
                    boxes = mosaic_boxes(k, mixed.shape[2], mixed.shape[3])
                    clean_tiles = th.cat(
                        [
                            mixed[:, :, top:bottom, left:right]
                            for top, bottom, left, right in boxes
                        ],
                        dim=0,
                    )
                    vqgan_tiles = th.cat(
                        [
                            distorted[:, :, top:bottom, left:right]
                            for top, bottom, left, right in boxes
                        ],
                        dim=0,
                    )
                    if args.tile_resize > 0:
                        size = (args.tile_resize, args.tile_resize)
                        clean_tiles = th.nn.functional.interpolate(
                            clean_tiles,
                            size=size,
                            mode="bilinear",
                            align_corners=False,
                        )
                        vqgan_tiles = th.nn.functional.interpolate(
                            vqgan_tiles,
                            size=size,
                            mode="bilinear",
                            align_corners=False,
                        )
                    clean_tile_probs = (
                        decode_binary_probabilities(model.decoder(clean_tiles))
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                        .tolist()
                    )
                    vqgan_tile_probs = (
                        decode_binary_probabilities(model.decoder(vqgan_tiles))
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                        .tolist()
                    )
                clean_score = score_probs(
                    probs_clean.detach().cpu().numpy().astype(np.float64),
                    codes,
                    colluders,
                    invalid_threshold,
                )
                distorted_score = score_probs(
                    probs_distorted.detach().cpu().numpy().astype(np.float64),
                    codes,
                    colluders,
                    invalid_threshold,
                )
                records.append(
                    {
                        "k": k,
                        "group": group_index,
                        "cover_index": group_index,
                        "colluders": colluders.tolist(),
                        "clean": clean_score,
                        "vqgan": distorted_score,
                        "clean_probs": probs_clean.detach().cpu().numpy().astype(np.float64).tolist(),
                        "vqgan_probs": probs_distorted.detach().cpu().numpy().astype(np.float64).tolist(),
                        "clean_tile_probs": clean_tile_probs,
                        "vqgan_tile_probs": vqgan_tile_probs,
                    }
                )
                logger.log(
                    f"k={k} group={group_index} colluders={colluders.tolist()} "
                    f"clean_nearest={clean_score['nearest_user']} "
                    f"clean_hamming={clean_score['best_colluder_hamming']} "
                    f"vqgan_nearest={distorted_score['nearest_user']} "
                    f"vqgan_hamming={distorted_score['best_colluder_hamming']}"
                )

    summary = {}
    for k in range(args.min_k, args.max_k + 1):
        for condition in ["clean", "vqgan"]:
            selected = [
                item[condition] for item in records if item["k"] == k
            ]
            summary[f"k{k}_{condition}"] = aggregate(selected)

    output = {
        "checkpoint": os.path.abspath(args.model_path),
        "data_dir": os.path.abspath(args.data_dir),
        "num_users": args.num_users,
        "num_groups": args.num_groups,
        "min_k": args.min_k,
        "max_k": args.max_k,
        "mix": args.mix,
        "message_length": args.message_length,
        "codebook_json": os.path.abspath(args.codebook_json) if args.codebook_json else "",
        "image_size": args.image_size,
        "timestep_respacing": args.timestep_respacing,
        "noise_layer": noise_layers,
        "identity_offset": args.identity_offset,
        "invalid_ber_threshold": args.invalid_ber,
        "elapsed_seconds": time.time() - started,
        "summary": summary,
        "records": records,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    for key, value in summary.items():
        logger.log(f"{key}: {value}")
    logger.log(f"saved: {args.output_json}")
    logger.log("collusion evaluation complete")


def create_argparser():
    defaults = endecoder_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="CelebA-HQ/test_128_500")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_denoised", type=str2bool, default=False)
    parser.add_argument("--noise_layer", type=str, default="VQGAN()")
    parser.add_argument("--identity_offset", type=int, default=0)
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--num_groups", type=int, default=20)
    parser.add_argument("--min_k", type=int, default=2)
    parser.add_argument("--max_k", type=int, default=5)
    parser.add_argument(
        "--mix", type=str, choices=["mean", "median", "mosaic"], default="mean"
    )
    parser.add_argument("--invalid_ber", type=float, default=0.2)
    parser.add_argument("--save_tiles", action="store_true")
    parser.add_argument("--tile_resize", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--code_seed", type=int, default=2026)
    parser.add_argument("--codebook_json", type=str, default="")
    parser.add_argument("--collusion_seed", type=int, default=2026)
    parser.add_argument(
        "--output_json", type=str, default="results/user_collusion_eval_30k.json"
    )
    return parser


if __name__ == "__main__":
    main()
