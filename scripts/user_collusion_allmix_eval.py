# -*- coding: utf-8 -*-
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
from user_collusion_eval import (
    aggregate,
    decode_binary_probabilities,
    load_codebook,
    mosaic_mix,
    score_probs,
)


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    device = dist_util.dev()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    codes = load_codebook(args.codebook_json, args.num_users, args.message_length)
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
    logger.log(f"mix: all (mean/median/mosaic)")

    model, diffusion = create_endecoder_and_diffusion(
        **args_to_dict(args, endecoder_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    noise_layer = Random_Noise([args.noise_layer])
    collusion_rng = np.random.RandomState(args.collusion_seed)
    mosaic_rng = np.random.RandomState(args.collusion_seed + 100003)
    invalid_threshold = int(np.floor(args.invalid_ber * args.message_length))
    mixes = ["mean", "median", "mosaic"]
    records = {mix: [] for mix in mixes}

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
                for mix in mixes:
                    if mix == "mean":
                        mixed = samples.mean(dim=0, keepdim=True)
                    elif mix == "median":
                        sorted_samples = samples.sort(dim=0).values
                        if k % 2 == 1:
                            mixed = sorted_samples[k // 2 : k // 2 + 1]
                        else:
                            mixed = 0.5 * (
                                sorted_samples[k // 2 - 1 : k // 2]
                                + sorted_samples[k // 2 : k // 2 + 1]
                            )
                    else:
                        mixed = mosaic_mix(samples, mosaic_rng)
                    probs_clean = decode_binary_probabilities(model.decoder(mixed))[0]
                    distorted = noise_layer([mixed, attack_identity])
                    probs_distorted = decode_binary_probabilities(
                        model.decoder(distorted)
                    )[0]
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
                    records[mix].append(
                        {
                            "k": k,
                            "group": group_index,
                            "cover_index": group_index,
                            "colluders": colluders.tolist(),
                            "clean": clean_score,
                            "vqgan": distorted_score,
                            "clean_probs": probs_clean.detach().cpu().numpy().astype(np.float64).tolist(),
                            "vqgan_probs": probs_distorted.detach().cpu().numpy().astype(np.float64).tolist(),
                            "clean_tile_probs": [],
                            "vqgan_tile_probs": [],
                        }
                    )
                logger.log(
                    f"k={k} group={group_index} colluders={colluders.tolist()}"
                )

    elapsed = time.time() - started
    for mix in mixes:
        summary = {}
        for k in range(args.min_k, args.max_k + 1):
            for condition in ["clean", "vqgan"]:
                selected = [
                    item[condition] for item in records[mix] if item["k"] == k
                ]
                summary[f"k{k}_{condition}"] = aggregate(selected)
        output = {
            "checkpoint": os.path.abspath(args.model_path),
            "data_dir": os.path.abspath(args.data_dir),
            "num_users": args.num_users,
            "num_groups": args.num_groups,
            "min_k": args.min_k,
            "max_k": args.max_k,
            "mix": mix,
            "message_length": args.message_length,
            "codebook_json": os.path.abspath(args.codebook_json),
            "image_size": args.image_size,
            "timestep_respacing": args.timestep_respacing,
            "noise_layer": [args.noise_layer],
            "identity_offset": args.identity_offset,
            "invalid_ber_threshold": args.invalid_ber,
            "elapsed_seconds": elapsed,
            "summary": summary,
            "records": records[mix],
        }
        path = args.output_prefix + "_" + mix + ".json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        logger.log(f"saved: {path}")


def create_argparser():
    defaults = endecoder_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="CelebA-HQ/test_128_500")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--clip_denoised", type=str2bool, default=False)
    parser.add_argument("--noise_layer", type=str, default="SimSwap(224)")
    parser.add_argument("--identity_offset", type=int, default=1)
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--num_groups", type=int, default=100)
    parser.add_argument("--min_k", type=int, default=2)
    parser.add_argument("--max_k", type=int, default=5)
    parser.add_argument("--invalid_ber", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--code_seed", type=int, default=2026)
    parser.add_argument("--collusion_seed", type=int, default=2026)
    parser.add_argument("--codebook_json", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, required=True)
    return parser


if __name__ == "__main__":
    main()