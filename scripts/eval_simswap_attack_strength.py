import argparse
import json
import math
import os
import time

import kornia.losses
import lpips as clpips
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    create_endecoder_and_diffusion,
    endecoder_and_diffusion_defaults,
    str2bool,
)
from noise_layers.simswap.test_one_image import SimSwap
from noise_layers.inswapper.inswapper import InSwapper


matplotlib.use("Agg")


def load_codes(path, num_users, message_length):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    codes = np.asarray(
        [[int(bit) for bit in code] for code in data["codes"]], dtype=np.int64
    )
    if codes.shape[0] < num_users or codes.shape[1] != message_length:
        raise ValueError(f"unexpected codebook shape: {codes.shape}")
    return codes[:num_users]


def arcface_input(image):
    image = (image + 1.0) / 2.0
    image = F.interpolate(image, size=(112, 112), mode="bilinear", align_corners=False)
    mean = th.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = th.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
    return (image - mean) / std


def arcface_embeddings(model, image):
    return F.normalize(model(arcface_input(image)), dim=1)


def cosine(left, right):
    return th.sum(F.normalize(left, dim=1) * F.normalize(right, dim=1), dim=1)


def psnr_per_image(prediction, target):
    mse = th.mean((prediction - target) ** 2, dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * th.log10(4.0 / mse)


def ssim_per_image(prediction, target):
    values = []
    for index in range(prediction.shape[0]):
        loss = kornia.losses.ssim_loss(
            prediction[index : index + 1],
            target[index : index + 1],
            window_size=11,
            reduction="mean",
        )
        values.append(1.0 - 2.0 * loss)
    return th.stack(values)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    count = values.size
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(count)) if count > 1 else 0.0
    return {
        "count": int(count),
        "mean": mean,
        "std": std,
        "ci95_low": mean - ci95,
        "ci95_high": mean + ci95,
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def format_metric(row):
    return (
        f"{row['mean']:.4f} +/- {row['std']:.4f} | "
        f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | "
        f"{row['median']:.4f} | {row['min']:.4f} | {row['max']:.4f}"
    )


def save_triplet_grid(originals, watermarked, identities, attacked, path, label):
    columns = [
        ("Original", 0),
        ("Watermarked", 1),
        ("Identity source\n(not an attack output)", 2),
        (f"{label} output", 3),
    ]
    count = min(4, originals.shape[0])
    fig, axes = plt.subplots(count, 4, figsize=(8.6, 2.12 * count + 0.8))
    tensors = (originals, watermarked, identities, attacked)
    for row in range(count):
        for column, (title, tensor_index) in enumerate(columns):
            image = ((tensors[tensor_index][row] + 1.0) / 2.0).clamp(0.0, 1.0)
            image = image.permute(1, 2, 0).cpu().numpy()
            axis = axes[row, column]
            axis.imshow(image)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
            if column == 0:
                axis.set_ylabel(
                    f"Sample {row + 1}",
                    rotation=0,
                    fontsize=8.5,
                    labelpad=30,
                    va="center",
                )
    fig.suptitle(f"{label} face-swap attack", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.012,
        "The identity source is a deliberately different face used as the swap input.",
        ha="center",
        fontsize=8.2,
        color="#475467",
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.95])
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_plot(records, path, label="SimSwap-224"):
    psnr = np.asarray([row["psnr_watermarked_to_attacked"] for row in records])
    lpips = np.asarray([row["lpips_watermarked_to_attacked"] for row in records])
    sim_wm_source = np.asarray([row["arcface_watermarked_source"] for row in records])
    sim_att_source = np.asarray([row["arcface_attacked_source"] for row in records])
    sim_att_wm = np.asarray([row["arcface_attacked_watermarked"] for row in records])
    margin = np.asarray([row["identity_margin"] for row in records])

    colors = ["#2b6cb0", "#dd6b20", "#2f855a", "#805ad5"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    fig.suptitle(f"{label} attack strength", fontsize=12)

    axes[0, 0].hist(psnr, bins=18, color=colors[0], alpha=0.85, edgecolor="white")
    axes[0, 0].axvline(psnr.mean(), color="#1a202c", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("(a) Watermarked vs. attacked: PSNR")
    axes[0, 0].set_xlabel("PSNR (dB)")
    axes[0, 0].set_ylabel("Images")

    axes[0, 1].hist(lpips, bins=18, color=colors[1], alpha=0.85, edgecolor="white")
    axes[0, 1].axvline(lpips.mean(), color="#1a202c", linestyle="--", linewidth=1.5)
    axes[0, 1].set_title("(b) Watermarked vs. attacked: LPIPS")
    axes[0, 1].set_xlabel("LPIPS")
    axes[0, 1].set_ylabel("Images")

    axes[1, 0].boxplot(
        [sim_wm_source, sim_att_source, sim_att_wm],
        tick_labels=["WM-source", "attack-source", "attack-WM"],
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "#cbd5e0"},
        medianprops={"color": "#c53030", "linewidth": 1.8},
        meanprops={
            "marker": "o",
            "markerfacecolor": "#2d3748",
            "markeredgecolor": "#2d3748",
        },
    )
    axes[1, 0].set_title("(c) ArcFace identity cosine")
    axes[1, 0].set_ylabel("Cosine similarity")
    axes[1, 0].grid(axis="y", alpha=0.25)

    axes[1, 1].hist(margin, bins=18, color=colors[2], alpha=0.85, edgecolor="white")
    axes[1, 1].axvline(0.0, color="#c53030", linestyle="--", linewidth=1.5)
    share = float(np.mean(margin > 0.0))
    axes[1, 1].set_title(f"(d) Identity margin, positive for {share:.1%} of images")
    axes[1, 1].set_xlabel("ArcFace cosine margin")
    axes[1, 1].set_ylabel("Images")

    for axis in axes.ravel():
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_markdown(summary, path):
    psnr = summary["metrics"]["psnr_watermarked_to_attacked"]
    ssim = summary["metrics"]["ssim_watermarked_to_attacked"]
    lpips = summary["metrics"]["lpips_watermarked_to_attacked"]
    att_source = summary["metrics"]["arcface_attacked_source"]
    wm_source = summary["metrics"]["arcface_watermarked_source"]
    att_wm = summary["metrics"]["arcface_attacked_watermarked"]
    margin = summary["metrics"]["identity_margin"]
    identity_switch = 100.0 * summary["fractions"]["identity_margin_positive"]
    attack = summary["attack"]
    label = summary["attack_label"]
    if attack == "simswap":
        attack_protocol = (
            "- Attack: watermarked content and a different identity image are "
            "upsized to 224x224, processed by SimSwap-224, then resized to 128x128."
        )
        identity_protocol = (
            "- Identity metrics use the SimSwap ArcFace checkpoint: [0,1] RGB, "
            "112x112, ImageNet mean/std normalization."
        )
        attack_interpretation = (
            "- The pixel metrics quantify how much SimSwap changes the watermarked image."
        )
        tracing_note = (
            "- These measurements describe attack strength only. Watermark BER and "
            "tracing results remain in the SimSwap single-image and collusion reports."
        )
    else:
        attack_protocol = (
            "- Attack: watermarked content and a different identity image are aligned "
            "to 128x128 with five landmarks, processed by InSwapper-128, and pasted "
            "back to the 128x128 frame."
        )
        identity_protocol = (
            "- Identity metrics use the official InsightFace w600k_r50 recognizer "
            "with 112x112 RGB [-1,1] normalization; generation additionally uses "
            "InSwapper's official emap projection."
        )
        attack_interpretation = (
            "- The pixel metrics quantify how much InSwapper changes the watermarked image."
        )
        tracing_note = (
            "- These measurements describe attack strength only. Watermark BER and "
            "tracing results remain in the InSwapper single-image and collusion reports."
        )
    lines = [
        f"# {label} attack strength",
        "",
        "## Protocol",
        "",
        f"- Images: {summary['num_images']} watermarked CelebA-HQ 128x128 images.",
        "- Watermark: 30-bit user code from the min-Hamming-11 codebook, DDIM 100.",
        "- Identity source: the paired image in the same batch; no image is swapped with itself.",
        attack_protocol,
        identity_protocol,
        "",
        "## Pixel-level attack distortion",
        "",
        "| Comparison | Mean +/- std | 95% CI | Median | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
        f"| PSNR, watermarked to attacked (dB) | {format_metric(psnr)} |",
        f"| SSIM, watermarked to attacked | {format_metric(ssim)} |",
        f"| LPIPS, watermarked to attacked | {format_metric(lpips)} |",
        "",
        "## Identity evidence",
        "",
        "| Comparison | Mean +/- std | 95% CI | Median | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Cosine, watermarked vs. source | {format_metric(wm_source)} |",
        f"| Cosine, attacked vs. source | {format_metric(att_source)} |",
        f"| Cosine, attacked vs. watermarked | {format_metric(att_wm)} |",
        f"| Identity margin, attack-source minus attack-WM | {format_metric(margin)} |",
        "",
        f"The identity margin is positive for {identity_switch:.1f}% of images. "
        "This is the fraction for which the attacked face is closer to the supplied identity than to the watermarked content identity.",
        "",
        "## Interpretation",
        "",
        attack_interpretation,
        "- The ArcFace table separates true identity replacement from ordinary image distortion.",
        tracing_note,
        "",
        "## Figures",
        "",
        f"![Attack strength]({summary['figure_png']})",
        "",
        f"![Attack examples]({summary['figure_example_png']})",
        "",
    ]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()
    device = dist_util.dev()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    if args.num_images % 2:
        raise ValueError(
            "--num_images must be even because the attack uses paired identity images"
        )
    num_users = math.ceil(args.num_images / args.images_per_user)
    codes = load_codes(args.codebook_json, num_users, args.message_length)
    codes_tensor = th.from_numpy(codes).float().to(device)

    model, diffusion = create_endecoder_and_diffusion(
        **args_to_dict(args, endecoder_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    label = args.attack_label or (
        "SimSwap-224" if args.attack == "simswap" else "InSwapper-128"
    )
    if args.attack == "simswap":
        attack = SimSwap(resolution=args.simswap_resolution)
        arcface = (
            th.load(args.arcface_checkpoint, map_location="cpu", weights_only=False)
            .to(device)
            .eval()
        )
    else:
        attack = InSwapper(128)
        arcface = None
    lpips_model = clpips.LPIPS(net="vgg").to(device).eval()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_md)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_png)), exist_ok=True)

    loader = load_data(
        data_dir=args.data_dir,
        batch_size=2,
        image_size=args.image_size,
        deterministic=True,
    )
    records = []
    start = time.time()
    examples = None
    with th.no_grad():
        for images in loader:
            if len(records) >= args.num_images:
                break
            take = min(images.shape[0], args.num_images - len(records))
            if take != 2:
                raise RuntimeError("the final batch must contain two images")
            images = images.to(device)
            current = len(records)
            user_ids = (
                th.arange(current, current + take, device=device) // args.images_per_user
            )
            message = codes_tensor[user_ids]
            watermarked = diffusion.ddim_sample_loop(
                model,
                images,
                message,
                clip_denoised=args.clip_denoised,
                model_kwargs={},
            )
            attacked = attack([watermarked, images])

            if args.attack == "inswapper":
                emb_wm = attack.embed(watermarked)
                emb_att = attack.embed(attacked)
                emb_source = attack.embed(images.flip(0))
                emb_original = attack.embed(images)
            else:
                emb_wm = arcface_embeddings(arcface, watermarked)
                emb_att = arcface_embeddings(arcface, attacked)
                emb_source = arcface_embeddings(arcface, images.flip(0))
                emb_original = arcface_embeddings(arcface, images)

            psnr_values = psnr_per_image(attacked, watermarked).cpu().numpy()
            ssim_values = ssim_per_image(attacked, watermarked).cpu().numpy()
            lpips_values = (
                lpips_model(attacked, watermarked).reshape(-1).cpu().numpy()
            )
            sim_att_source = cosine(emb_att, emb_source).cpu().numpy()
            sim_wm_source = cosine(emb_wm, emb_source).cpu().numpy()
            sim_att_wm = cosine(emb_att, emb_wm).cpu().numpy()
            sim_att_original = cosine(emb_att, emb_original).cpu().numpy()
            sim_wm_original = cosine(emb_wm, emb_original).cpu().numpy()

            for offset in range(take):
                global_index = current + offset
                source_index = global_index + 1 if offset == 0 else global_index - 1
                records.append(
                    {
                        "image_index": int(global_index),
                        "user_id": int(user_ids[offset].item()),
                        "identity_source_index": int(source_index),
                        "psnr_watermarked_to_attacked": float(psnr_values[offset]),
                        "ssim_watermarked_to_attacked": float(ssim_values[offset]),
                        "lpips_watermarked_to_attacked": float(lpips_values[offset]),
                        "arcface_watermarked_source": float(sim_wm_source[offset]),
                        "arcface_attacked_source": float(sim_att_source[offset]),
                        "arcface_attacked_watermarked": float(sim_att_wm[offset]),
                        "arcface_attacked_original": float(sim_att_original[offset]),
                        "arcface_watermarked_original": float(sim_wm_original[offset]),
                        "identity_margin": float(
                            sim_att_source[offset] - sim_att_wm[offset]
                        ),
                    }
                )
                save_image(
                    ((images[offset] + 1.0) / 2.0).clamp(0.0, 1.0).cpu(),
                    os.path.join(args.save_dir, f"{global_index:04d}_original.png"),
                )
                save_image(
                    ((watermarked[offset] + 1.0) / 2.0).clamp(0.0, 1.0).cpu(),
                    os.path.join(args.save_dir, f"{global_index:04d}_watermarked.png"),
                )
                save_image(
                    ((images[1 - offset] + 1.0) / 2.0).clamp(0.0, 1.0).cpu(),
                    os.path.join(
                        args.save_dir, f"{global_index:04d}_identity_source.png"
                    ),
                )
                save_image(
                    ((attacked[offset] + 1.0) / 2.0).clamp(0.0, 1.0).cpu(),
                    os.path.join(args.save_dir, f"{global_index:04d}_attacked.png"),
                )

            if examples is None:
                examples = (
                    images.detach().cpu(),
                    watermarked.detach().cpu(),
                    images.flip(0).detach().cpu(),
                    attacked.detach().cpu(),
                )
            logger.log(f"processed {len(records)}/{args.num_images} images")

    metric_names = [
        "psnr_watermarked_to_attacked",
        "ssim_watermarked_to_attacked",
        "lpips_watermarked_to_attacked",
        "arcface_watermarked_source",
        "arcface_attacked_source",
        "arcface_attacked_watermarked",
        "arcface_attacked_original",
        "arcface_watermarked_original",
        "identity_margin",
    ]
    metrics = {
        name: summarize([row[name] for row in records]) for name in metric_names
    }
    fractions = {
        "identity_margin_positive": float(
            np.mean([row["identity_margin"] > 0.0 for row in records])
        ),
        "attacked_closer_to_source_than_original": float(
            np.mean(
                [
                    row["arcface_attacked_source"]
                    > row["arcface_attacked_original"]
                    for row in records
                ]
            )
        ),
    }
    summary = {
        "checkpoint": os.path.abspath(args.model_path),
        "data_dir": os.path.abspath(args.data_dir),
        "codebook_json": os.path.abspath(args.codebook_json),
        "attack": args.attack,
        "attack_label": label,
        "arcface_checkpoint": (
            os.path.abspath(args.arcface_checkpoint)
            if args.attack == "simswap"
            else ""
        ),
        "inswapper_onnx": (
            str(attack.onnx_path) if args.attack == "inswapper" else ""
        ),
        "inswapper_recognizer": (
            str(attack.recognizer_path) if args.attack == "inswapper" else ""
        ),
        "simswap_resolution": args.simswap_resolution,
        "num_images": len(records),
        "image_size": args.image_size,
        "images_per_user": args.images_per_user,
        "seed": args.seed,
        "elapsed_seconds": time.time() - start,
        "metrics": metrics,
        "fractions": fractions,
        "figure_png": os.path.basename(args.output_png),
        "figure_example_png": os.path.basename(args.output_example_png),
        "records": records,
    }

    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    save_triplet_grid(*examples, path=args.output_example_png, label=label)
    save_plot(records, args.output_png, label=label)
    write_markdown(summary, args.output_md)

    if args.attack == "inswapper":
        attack.close()

    for name in metric_names:
        logger.log(f"{name}: {metrics[name]}")
    logger.log(f"fractions: {fractions}")
    logger.log(f"saved: {args.output_json}")
    logger.log(f"saved: {args.output_md}")
    logger.log(f"saved: {args.output_png}")
    logger.log("attack strength evaluation complete")


def create_argparser():
    defaults = endecoder_and_diffusion_defaults()
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/2026-10-30-20-26-32-353028/model030000.pt",
    )
    parser.add_argument("--data_dir", type=str, default="CelebA-HQ/test_128_500")
    parser.add_argument("--num_images", type=int, default=100)
    parser.add_argument("--images_per_user", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--clip_denoised", type=str2bool, default=False)
    parser.add_argument(
        "--codebook_json", type=str, default="results/codebook_100x30_min11.json"
    )
    parser.add_argument(
        "--arcface_checkpoint",
        type=str,
        default="noise_layers/simswap/arcface_checkpoint.tar",
    )
    parser.add_argument("--simswap_resolution", type=int, default=224)
    parser.add_argument(
        "--attack",
        type=str,
        choices=["simswap", "inswapper"],
        default="simswap",
    )
    parser.add_argument("--attack_label", type=str, default="")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/simswap224_attack_strength_images",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/simswap224_attack_strength.json",
    )
    parser.add_argument(
        "--output_md",
        type=str,
        default="docs/simswap224_attack_strength.md",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="docs/simswap224_attack_strength.png",
    )
    parser.add_argument(
        "--output_example_png",
        type=str,
        default="docs/simswap224_attack_strength_example.png",
    )
    return parser


if __name__ == "__main__":
    main()
