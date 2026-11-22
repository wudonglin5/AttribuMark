import argparse
import io
import json
import math
import os
import time

import numpy as np
import torch as th
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

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


def jpeg_roundtrip(images, quality):
    output = []
    for image in images.detach().cpu():
        array = (
            ((image.clamp(-1.0, 1.0) + 1.0) * 127.5)
            .round()
            .to(th.uint8)
            .permute(1, 2, 0)
            .numpy()
        )
        buffer = io.BytesIO()
        Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        decoded = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float32)
        output.append(th.from_numpy(decoded / 127.5 - 1.0).permute(2, 0, 1))
    return th.stack(output, dim=0).to(images.device)


def center_crop_resize(images, fraction):
    height, width = images.shape[-2:]
    crop_h = int(round(height * fraction))
    crop_w = int(round(width * fraction))
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    cropped = images[:, :, top : top + crop_h, left : left + crop_w]
    return F.interpolate(cropped, size=(height, width), mode="bilinear", align_corners=False)


def resize_roundtrip(images, scale):
    height, width = images.shape[-2:]
    down = F.interpolate(
        images,
        size=(max(1, int(round(height * scale))), max(1, int(round(width * scale)))),
        mode="bilinear",
        align_corners=False,
    )
    return F.interpolate(down, size=(height, width), mode="bilinear", align_corners=False)


def apply_condition(images, condition, seed, cover=None, vqgan=None):
    if condition == "clean":
        return images
    if condition == "vqgan":
        if cover is None or vqgan is None:
            raise ValueError("vqgan condition requires cover images and a VQGAN module")
        return vqgan([images, cover])
    if condition == "jpeg70":
        return jpeg_roundtrip(images, 70)
    if condition == "jpeg50":
        return jpeg_roundtrip(images, 50)
    if condition == "jpeg30":
        return jpeg_roundtrip(images, 30)
    if condition == "blur_sigma1":
        return TF.gaussian_blur(images, kernel_size=5, sigma=(1.0, 1.0))
    if condition == "noise_sigma002":
        generator = th.Generator(device=images.device)
        generator.manual_seed(seed)
        return images + 0.02 * th.randn(images.shape, generator=generator, device=images.device)
    if condition == "resize075":
        return resize_roundtrip(images, 0.75)
    if condition == "crop090":
        return center_crop_resize(images, 0.90)
    if condition == "crop080":
        return center_crop_resize(images, 0.80)
    if condition == "crop070":
        return center_crop_resize(images, 0.70)
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--clip_denoised", type=str2bool, default=False)
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--code_start", type=int, default=0)
    parser.add_argument("--include_vqgan", type=str2bool, default=False)
    parser.add_argument("--codebook_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    device = dist_util.dev()
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    codes = load_codes(args.codebook_json, args.num_users + args.code_start, args.message_length)
    needed = args.num_users
    all_files = _list_image_files_recursively(args.data_dir)
    if len(all_files) < needed:
        raise ValueError(f"need {needed} images, but only found {len(all_files)}")

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
    vqgan = Random_Noise(["VQGAN()"]) if args.include_vqgan else None

    conditions = ["clean"]
    if args.include_vqgan:
        conditions.append("vqgan")
    conditions.extend(["jpeg70", "jpeg50", "jpeg30", "blur_sigma1", "noise_sigma002", "resize075", "crop090", "crop080", "crop070"])
    probabilities = {condition: [] for condition in conditions}
    records = []
    seen = 0
    started = time.time()
    with th.no_grad():
        for batch_index, images in enumerate(data_loader):
            if seen >= needed:
                break
            take = min(images.shape[0], needed - seen)
            images = images[:take].to(device)
            image_indices = th.arange(seen, seen + take, device=device)
            user_ids = image_indices + args.code_start
            message = th.from_numpy(codes[user_ids.detach().cpu().numpy()]).float().to(device)
            watermarked = diffusion.ddim_sample_loop(
                model,
                images,
                message,
                clip_denoised=args.clip_denoised,
                model_kwargs={},
            )
            for condition in conditions:
                attacked = apply_condition(watermarked, condition, args.seed + seen, cover=images, vqgan=vqgan)
                logits = model.decoder(attacked)
                probs = decode_binary_probabilities(logits).detach().cpu().numpy()
                probabilities[condition].append(probs)
                for local_index, user_id in enumerate(user_ids.detach().cpu().numpy()):
                    records.append(
                        {
                            "image_index": int(seen + local_index),
                            "user_id": int(user_id),
                            "condition": condition,
                            "probs": probs[local_index].astype(float).tolist(),
                        }
                    )
            seen += take
            if batch_index % 5 == 0 or seen == needed:
                logger.log(f"processed {seen}/{needed} watermarked images")

    user_ids_array = np.arange(args.code_start, args.code_start + needed, dtype=np.int64)
    results = {}
    for condition in conditions:
        combined = np.concatenate(probabilities[condition], axis=0)
        results[condition] = summarize(combined, codes, user_ids_array, args.threshold)

    output = {
        "checkpoint": os.path.abspath(args.model_path),
        "data_dir": os.path.abspath(args.data_dir),
        "codebook_json": os.path.abspath(args.codebook_json),
        "num_users": args.num_users,
        "code_start": args.code_start,
        "message_length": args.message_length,
        "image_size": args.image_size,
        "timestep_respacing": args.timestep_respacing,
        "elapsed_seconds": time.time() - started,
        "conditions": conditions,
        "results": results,
        "records": records,
        "codes": ["".join(str(int(bit)) for bit in code) for code in codes],
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_json))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    logger.log("standard distortion results")
    for condition in conditions:
        logger.log(f"{condition}: {results[condition]}")
    logger.log(f"saved: {args.output_json}")


if __name__ == "__main__":
    main()
