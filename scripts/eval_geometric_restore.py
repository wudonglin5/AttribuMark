import argparse
import json
import os
import time

import numpy as np
import torch as th
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import _list_image_files_recursively, load_data
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_endecoder_and_diffusion,
    endecoder_and_diffusion_defaults,
    str2bool,
)


def load_codes(path, num_users, message_length):
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
    batch_size, message_length, classes = logits.shape
    if classes != 2 * message_length:
        raise ValueError(f"unexpected decoder output shape: {tuple(logits.shape)}")
    positions = th.arange(message_length, device=logits.device)
    pair_logits = th.stack(
        [logits[:, positions, positions], logits[:, positions, positions + message_length]],
        dim=-1,
    )
    return th.softmax(pair_logits, dim=-1)[..., 1]


def center_crop_resize(images, fraction):
    height, width = images.shape[-2:]
    crop_h = int(round(height * fraction))
    crop_w = int(round(width * fraction))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    cropped = images[:, :, top : top + crop_h, left : left + crop_w]
    return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)


def restore_center_crop(images, fraction, padding):
    height, width = images.shape[-2:]
    crop_h = int(round(height * fraction))
    crop_w = int(round(width * fraction))
    resized = F.interpolate(
        images, size=(crop_h, crop_w), mode="bilinear", align_corners=False
    )
    top = (height - crop_h) // 2
    bottom = height - crop_h - top
    left = (width - crop_w) // 2
    right = width - crop_w - left
    if padding == "zero":
        return F.pad(resized, (left, right, top, bottom), mode="constant", value=-1.0)
    if padding == "edge":
        return F.pad(resized, (left, right, top, bottom), mode="replicate")
    if padding == "reflect":
        return F.pad(resized, (left, right, top, bottom), mode="reflect")
    raise ValueError(f"unknown padding mode: {padding}")


def make_condition(images, condition):
    if condition == "clean":
        return images
    if condition == "crop090":
        return center_crop_resize(images, 0.90)
    if condition == "crop090_inv_zero":
        return restore_center_crop(images, 0.90, "zero")
    if condition == "crop090_inv_edge":
        return restore_center_crop(images, 0.90, "edge")
    if condition == "crop090_inv_reflect":
        return restore_center_crop(images, 0.90, "reflect")
    if condition == "crop080":
        return center_crop_resize(images, 0.80)
    if condition == "crop080_inv_edge":
        return restore_center_crop(images, 0.80, "edge")
    if condition == "crop070":
        return center_crop_resize(images, 0.70)
    if condition == "crop070_inv_edge":
        return restore_center_crop(images, 0.70, "edge")
    raise ValueError(f"unknown condition: {condition}")


def summarize(probs, codes, user_ids, threshold):
    probabilities = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    llr = np.log(probabilities / (1.0 - probabilities))
    positive_codes = 2.0 * codes.astype(np.float64) - 1.0
    scores = llr @ positive_codes.T
    z = scores / np.sqrt((llr * llr).sum(axis=1, keepdims=True))

    predictions = (probabilities >= 0.5).astype(np.int64)
    true_codes = codes[user_ids]
    hamming = np.sum(predictions != true_codes, axis=1)
    top1 = np.argmax(z, axis=1)
    top1_hit = top1 == user_ids
    true_z = z[np.arange(len(user_ids)), user_ids]
    z_masked = z.copy()
    z_masked[np.arange(len(user_ids)), user_ids] = -np.inf
    best_other_z = np.max(z_masked, axis=1)
    accused = np.max(z, axis=1) >= threshold
    detected = accused & top1_hit
    false_accused = accused & ~top1_hit
    abstained = ~accused

    return {
        "num_images": int(len(user_ids)),
        "ber": float(np.mean(hamming / codes.shape[1])),
        "exact_match": float(np.mean(hamming == 0)),
        "mean_hamming": float(np.mean(hamming)),
        "max_hamming": int(np.max(hamming)),
        "top1_accuracy": float(np.mean(top1_hit)),
        "mean_true_z": float(np.mean(true_z)),
        "mean_top1_z": float(np.mean(np.max(z, axis=1))),
        "mean_margin_z": float(np.mean(true_z - best_other_z)),
        "detection_rate": float(np.mean(detected)),
        "false_accusation_rate": float(np.mean(false_accused)),
        "abstention_rate": float(np.mean(abstained)),
        "threshold": float(threshold),
    }


def create_argparser():
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, endecoder_and_diffusion_defaults())
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="CelebA-HQ/test_128_500")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--clip_denoised", type=str2bool, default=False)
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--codebook_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    return parser


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    device = dist_util.dev()

    codes = load_codes(args.codebook_json, args.num_users, args.message_length)
    all_files = _list_image_files_recursively(args.data_dir)
    if len(all_files) < args.num_users:
        raise ValueError(f"need {args.num_users} images, but only found {len(all_files)}")

    model, diffusion = create_endecoder_and_diffusion(
        **args_to_dict(args, endecoder_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    data_loader = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
    )

    conditions = [
        "clean",
        "crop090",
        "crop090_inv_zero",
        "crop090_inv_edge",
        "crop090_inv_reflect",
        "crop080",
        "crop080_inv_edge",
        "crop070",
        "crop070_inv_edge",
    ]
    probabilities = {condition: [] for condition in conditions}
    seen = 0
    started = time.time()
    with th.no_grad():
        for batch_index, images in enumerate(data_loader):
            if seen >= args.num_users:
                break
            take = min(images.shape[0], args.num_users - seen)
            images = images[:take].to(device)
            image_indices = th.arange(seen, seen + take, device=device)
            user_ids = image_indices
            message = th.from_numpy(codes[user_ids.detach().cpu().numpy()]).float().to(device)
            watermarked = diffusion.ddim_sample_loop(
                model,
                images,
                message,
                clip_denoised=args.clip_denoised,
                model_kwargs={},
            )
            for condition in conditions:
                attacked = make_condition(watermarked, condition)
                logits = model.decoder(attacked)
                probs = decode_binary_probabilities(logits).detach().cpu().numpy()
                probabilities[condition].append(probs)
            seen += take
            if batch_index % 5 == 0 or seen == args.num_users:
                logger.log(f"processed {seen}/{args.num_users} watermarked images")

    user_ids_array = np.arange(args.num_users, dtype=np.int64)
    results = {}
    for condition in conditions:
        combined = np.concatenate(probabilities[condition], axis=0)
        results[condition] = summarize(combined, codes, user_ids_array, args.threshold)

    output = {
        "checkpoint": os.path.abspath(args.model_path),
        "data_dir": os.path.abspath(args.data_dir),
        "codebook_json": os.path.abspath(args.codebook_json),
        "num_users": args.num_users,
        "message_length": args.message_length,
        "image_size": args.image_size,
        "timestep_respacing": args.timestep_respacing,
        "elapsed_seconds": time.time() - started,
        "conditions": conditions,
        "results": results,
        "codes": ["".join(str(int(bit)) for bit in code) for code in codes],
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    logger.log("geometric restore results")
    for condition in conditions:
        logger.log(f"{condition}: {results[condition]}")
    logger.log(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()