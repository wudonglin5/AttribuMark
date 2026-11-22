import json
import os

import torch as th
from torchvision.utils import make_grid, save_image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.noiser import Random_Noise
from guided_diffusion.script_util import (
    create_endecoder_and_diffusion,
    endecoder_and_diffusion_defaults,
)


ROOT = r"F:\project\beifen\DiffMark-main"


def main():
    dist_util.setup_dist()
    logger.configure()
    device = dist_util.dev()

    with open(
        os.path.join(ROOT, "results", "codebook_100x30_min11.json"),
        "r",
        encoding="utf-8",
    ) as handle:
        codes = json.load(handle)["codes"]
    codes = th.tensor(
        [[int(bit) for bit in code] for code in codes[:4]], dtype=th.float32
    ).to(device)

    loader = load_data(
        data_dir=os.path.join(ROOT, "CelebA-HQ", "test_128_500"),
        batch_size=4,
        image_size=128,
        deterministic=True,
    )
    covers = next(iter(loader)).to(device)

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

    with th.no_grad():
        watermarked = diffusion.ddim_sample_loop(
            model, covers, codes, clip_denoised=False, model_kwargs={}
        )
        noise_layer = Random_Noise(["SimSwap(224)"])
        swapped = noise_layer([watermarked, covers])

    def to_unit(image):
        return ((image + 1.0) / 2.0).clamp(0.0, 1.0)

    rows = []
    for index in range(4):
        rows.extend(
            [to_unit(covers[index]), to_unit(watermarked[index]), to_unit(swapped[index])]
        )
    grid = make_grid(rows, nrow=3, padding=2)
    output = os.path.join(ROOT, "docs", "simswap224_example.png")
    save_image(grid, output)
    print("saved", output)


if __name__ == "__main__":
    main()
