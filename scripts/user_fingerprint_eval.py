import argparse
import json
import os
import time

import numpy as np
import torch as th

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import _list_image_files_recursively, load_data
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


def pairwise_hamming(codes):
    num_codes = codes.shape[0]
    distances = []
    for i in range(num_codes):
        diff = codes[i + 1 :] != codes[i]
        distances.extend(diff.sum(axis=1).tolist())
    distances = np.asarray(distances, dtype=np.int64)
    return {
        "min": int(distances.min()) if distances.size else 0,
        "mean": float(distances.mean()) if distances.size else 0.0,
        "max": int(distances.max()) if distances.size else 0,
    }


def decode_binary_probabilities(logits):
    if logits.ndim != 3:
        raise ValueError(f"unexpected decoder output shape: {tuple(logits.shape)}")
    batch_size, message_length, classes = logits.shape
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


def summarize_votes(votes, codes):
    predictions = (votes >= 0.5).astype(np.int64)
    distances = np.sum(predictions[:, None, :] != codes[None, :, :], axis=-1)
    true_distances = np.diag(distances)
    nearest = np.argmin(distances, axis=1)
    true_indices = np.arange(len(distances))
    tied = np.sum(distances == true_distances[:, None], axis=1)
    sorted_distances = np.sort(distances, axis=1)
    second_best = sorted_distances[:, 1] if distances.shape[1] > 1 else true_distances
    confidence = np.maximum(votes, 1.0 - votes)
    return {
        "ber": float(np.mean(predictions != codes)),
        "exact_match": float(np.mean(np.all(predictions == codes, axis=1))),
        "user_identification_accuracy": float(np.mean(nearest == true_indices)),
        "mean_true_hamming": float(true_distances.mean()),
        "min_true_hamming": int(true_distances.min()),
        "max_true_hamming": int(true_distances.max()),
        "mean_second_best_margin": float(np.mean(second_best - true_distances)),
        "mean_confidence": float(confidence.mean()),
        "ties_with_true_code": int(np.sum(tied > 1)),
        "predictions": predictions,
        "per_user_hamming": true_distances,
        "per_user_nearest": nearest,
    }


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
    code_stats = pairwise_hamming(codes)
    needed = args.num_users * args.images_per_user
    all_files = _list_image_files_recursively(args.data_dir)
    if len(all_files) < needed:
        raise ValueError(
            f"need {needed} images, but only found {len(all_files)} in {args.data_dir}"
        )

    logger.log(f"checkpoint: {args.model_path}")
    logger.log(f"users: {args.num_users}, images per user: {args.images_per_user}")
    logger.log(f"usable images: {needed} / {len(all_files)}")
    logger.log(f"codebook hamming distance: {code_stats}")

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
    data_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
    )

    all_probs = []
    all_user_ids = []
    all_image_indices = []
    seen = 0
    started = time.time()
    with th.no_grad():
        for batch_index, images in enumerate(data_loader):
            if seen >= needed:
                break
            take = min(images.shape[0], needed - seen)
            images = images[:take].to(device)
            image_indices = th.arange(seen, seen + take, device=device)
            user_ids = th.div(image_indices, args.images_per_user, rounding_mode="floor")
            message = th.from_numpy(codes[user_ids.detach().cpu().numpy()]).float().to(device)
            sample = diffusion.ddim_sample_loop(
                model,
                images,
                message,
                clip_denoised=args.clip_denoised,
                model_kwargs={},
            )
            distorted = noise_layer([sample, images])
            logits = model.decoder(distorted)
            probs = decode_binary_probabilities(logits)
            all_probs.append(probs.detach().cpu().numpy().astype(np.float64))
            all_user_ids.append(user_ids.detach().cpu().numpy().astype(np.int64))
            all_image_indices.append(image_indices.detach().cpu().numpy().astype(np.int64))
            seen += take
            if batch_index % 10 == 0 or seen == needed:
                logger.log(f"processed {seen}/{needed} images")

    probs = np.concatenate(all_probs, axis=0)
    user_ids = np.concatenate(all_user_ids, axis=0)
    image_indices = np.concatenate(all_image_indices, axis=0)
    if probs.shape[0] != needed:
        raise RuntimeError(f"expected {needed} images, got {probs.shape[0]}")

    image_predictions = (probs >= 0.5).astype(np.int64)
    image_codes = codes[user_ids]
    image_hamming = np.sum(image_predictions != image_codes, axis=1)
    image_ber = image_hamming / args.message_length

    per_image = {
        "ber": float(image_ber.mean()),
        "ber_std": float(image_ber.std()),
        "exact_match": float(np.mean(image_hamming == 0)),
        "mean_hamming": float(image_hamming.mean()),
        "min_hamming": int(image_hamming.min()),
        "max_hamming": int(image_hamming.max()),
    }

    votes_by_user = np.zeros(
        (args.num_users, args.images_per_user, args.message_length), dtype=np.float64
    )
    for position, user_id in enumerate(user_ids):
        image_slot = int(image_indices[position] - user_id * args.images_per_user)
        votes_by_user[user_id, image_slot] = probs[position]

    curve = []
    for k in range(1, args.images_per_user + 1):
        votes = votes_by_user[:, :k, :].mean(axis=1)
        result = summarize_votes(votes, codes)
        curve.append(
            {
                "k": k,
                "ber": result["ber"],
                "exact_match": result["exact_match"],
                "user_identification_accuracy": result["user_identification_accuracy"],
                "mean_true_hamming": result["mean_true_hamming"],
                "min_true_hamming": result["min_true_hamming"],
                "max_true_hamming": result["max_true_hamming"],
                "mean_second_best_margin": result["mean_second_best_margin"],
                "mean_confidence": result["mean_confidence"],
                "ties_with_true_code": result["ties_with_true_code"],
            }
        )
        if k == args.images_per_user:
            final_per_user_hamming = result["per_user_hamming"]
            final_per_user_nearest = result["per_user_nearest"]

    per_user = []
    for user_id in range(args.num_users):
        per_user.append(
            {
                "user_id": int(user_id),
                "hamming": int(final_per_user_hamming[user_id]),
                "ber": float(final_per_user_hamming[user_id] / args.message_length),
                "nearest_user": int(final_per_user_nearest[user_id]),
                "correct": bool(final_per_user_nearest[user_id] == user_id),
            }
        )

    elapsed = time.time() - started
    output = {
        "checkpoint": os.path.abspath(args.model_path),
        "data_dir": os.path.abspath(args.data_dir),
        "num_users": args.num_users,
        "images_per_user": args.images_per_user,
        "message_length": args.message_length,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "timestep_respacing": args.timestep_respacing,
        "noise_layer": noise_layers,
        "seed": args.seed,
        "code_seed": args.code_seed,
        "codebook_json": os.path.abspath(args.codebook_json) if args.codebook_json else "",
        "elapsed_seconds": elapsed,
        "codebook_hamming": code_stats,
        "per_image": per_image,
        "soft_voting_curve": curve,
        "final": curve[-1],
        "per_user": per_user,
        "codes": ["".join(str(int(bit)) for bit in code) for code in codes],
        "processed_image_indices": image_indices.tolist(),
        "image_user_ids": user_ids.tolist(),
        "image_hamming": image_hamming.tolist(),
        "image_probs": probs.tolist(),
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    logger.log(f"per-image: {per_image}")
    for row in curve:
        logger.log(f"vote k={row['k']}: {row}")
    logger.log(f"saved: {args.output_json}")
    logger.log("user fingerprint evaluation complete")


def create_argparser():
    defaults = endecoder_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="CelebA-HQ/test_128_500")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_denoised", type=str2bool, default=False)
    parser.add_argument("--noise_layer", type=str, default="VQGAN()")
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--images_per_user", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--code_seed", type=int, default=2026)
    parser.add_argument("--codebook_json", type=str, default="")
    parser.add_argument(
        "--output_json", type=str, default="results/user_fingerprint_eval_30k.json"
    )
    return parser


if __name__ == "__main__":
    main()
