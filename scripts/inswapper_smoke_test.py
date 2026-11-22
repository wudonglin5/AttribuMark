import json
import os
import time

import kornia.losses
import lpips as clpips
import numpy as np
import torch as th
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    create_endecoder_and_diffusion,
    endecoder_and_diffusion_defaults,
)
from noise_layers.inswapper.inswapper import InSwapper


ROOT = r"F:\project\beifen\DiffMark-main"


def unit(image):
    return ((image + 1.0) / 2.0).clamp(0.0, 1.0)


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


def main():
    dist_util.setup_dist()
    logger.configure()
    device = dist_util.dev()
    th.manual_seed(2026)
    np.random.seed(2026)

    with open(
        os.path.join(ROOT, "results", "codebook_100x30_min11.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        codes = json.load(handle)["codes"]
    codes = th.tensor(
        [[int(bit) for bit in code] for code in codes],
        dtype=th.float32,
    ).to(device)

    defaults = endecoder_and_diffusion_defaults()
    defaults.update(
        image_size=128,
        message_length=30,
        embedding_dim=256,
        num_channels=32,
        num_heads=4,
        num_res_blocks=1,
        attention_resolutions="16,8",
        diffusion_steps=100,
        timestep_respacing="ddim100",
    )
    model, diffusion = create_endecoder_and_diffusion(**defaults)
    model.load_state_dict(
        dist_util.load_state_dict(
            os.path.join(
                ROOT,
                "results",
                "2026-10-30-20-26-32-353028",
                "model030000.pt",
            ),
            map_location="cpu",
        )
    )
    model.to(device)
    model.eval()

    inswapper = InSwapper(128)
    lpips_model = clpips.LPIPS(net="vgg").to(device).eval()
    loader = load_data(
        data_dir=os.path.join(ROOT, "CelebA-HQ", "test_128_500"),
        batch_size=2,
        image_size=128,
        deterministic=True,
    )

    records = []
    grid_rows = []
    started = time.time()
    with th.no_grad():
        for images in loader:
            if len(records) >= 10:
                break
            images = images.to(device)
            user_ids = th.arange(
                len(records), len(records) + images.shape[0], device=device
            ) // 2
            message = codes[user_ids]
            watermarked = diffusion.ddim_sample_loop(
                model,
                images,
                message,
                clip_denoised=False,
                model_kwargs={},
            )
            attacked = inswapper([watermarked, images])
            emb_wm = inswapper.embed(watermarked)
            emb_att = inswapper.embed(attacked)
            emb_source = inswapper.embed(images.flip(0))
            emb_original = inswapper.embed(images)

            psnr = psnr_per_image(attacked, watermarked).cpu().numpy()
            ssim = ssim_per_image(attacked, watermarked).cpu().numpy()
            lpips = lpips_model(attacked, watermarked).reshape(-1).cpu().numpy()
            att_source = cosine(emb_att, emb_source).cpu().numpy()
            att_wm = cosine(emb_att, emb_wm).cpu().numpy()
            att_original = cosine(emb_att, emb_original).cpu().numpy()

            for offset in range(images.shape[0]):
                records.append(
                    {
                        "image_index": len(records),
                        "user_id": int(user_ids[offset].item()),
                        "psnr": float(psnr[offset]),
                        "ssim": float(ssim[offset]),
                        "lpips": float(lpips[offset]),
                        "arcface_attacked_source": float(att_source[offset]),
                        "arcface_attacked_watermarked": float(att_wm[offset]),
                        "arcface_attacked_original": float(att_original[offset]),
                        "identity_margin": float(att_source[offset] - att_wm[offset]),
                    }
                )
                grid_rows.extend(
                    [
                        unit(images[offset]).cpu(),
                        unit(watermarked[offset]).cpu(),
                        unit(images[1 - offset]).cpu(),
                        unit(attacked[offset]).cpu(),
                    ]
                )

    save_dir = os.path.join(ROOT, "results", "inswapper128_smoke")
    os.makedirs(save_dir, exist_ok=True)
    save_image(
        make_grid(grid_rows, nrow=4, padding=2),
        os.path.join(ROOT, "docs", "inswapper128_smoke.png"),
    )
    for row in records:
        index = row["image_index"]
        row["elapsed_seconds"] = time.time() - started
        with open(
            os.path.join(save_dir, f"{index:04d}.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(row, handle, indent=2)

    metric_names = [
        "psnr",
        "ssim",
        "lpips",
        "arcface_attacked_source",
        "arcface_attacked_watermarked",
        "arcface_attacked_original",
        "identity_margin",
    ]
    summary = {
        "model": os.path.join(
            ROOT,
            "results",
            "2026-10-30-20-26-32-353028",
            "model030000.pt",
        ),
        "onnx": str(inswapper.onnx_path),
        "recognizer": str(inswapper.recognizer_path),
        "num_images": len(records),
        "elapsed_seconds": time.time() - started,
        "metrics": {
            name: {
                "mean": float(np.mean([row[name] for row in records])),
                "std": float(np.std([row[name] for row in records], ddof=1)),
                "min": float(np.min([row[name] for row in records])),
                "max": float(np.max([row[name] for row in records])),
            }
            for name in metric_names
        },
        "identity_margin_positive": float(
            np.mean([row["identity_margin"] > 0 for row in records])
        ),
        "records": records,
    }
    output_json = os.path.join(ROOT, "results", "inswapper128_smoke.json")
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    inswapper.close()
    for name in metric_names:
        logger.log(f"{name}: {summary['metrics'][name]}")
    logger.log(f"identity_margin_positive: {summary['identity_margin_positive']}")
    logger.log(f"saved: {output_json}")
    logger.log(f"saved: {os.path.join(ROOT, 'docs', 'inswapper128_smoke.png')}")
    logger.log("InSwapper smoke test complete")


if __name__ == "__main__":
    main()
