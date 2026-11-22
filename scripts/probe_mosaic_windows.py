import argparse
import sys

import numpy as np
import torch as th

sys.path.insert(0, "scripts")
from user_collusion_eval import (  # noqa: E402
    decode_binary_probabilities,
    load_codebook,
    mosaic_mix,
)

from guided_diffusion import dist_util  # noqa: E402
from guided_diffusion.image_datasets import load_data  # noqa: E402
from guided_diffusion.script_util import (  # noqa: E402
    create_endecoder_and_diffusion,
    endecoder_and_diffusion_defaults,
)


def z_scores(probs, codes):
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    llr = np.log(p / (1.0 - p))
    scores = ((2.0 * codes - 1.0) * llr[None, :]).sum(axis=1)
    return scores / np.sqrt((llr * llr).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", default="CelebA-HQ/test_128_500")
    parser.add_argument("--codebook_json", required=True)
    args = parser.parse_args()

    dist_util.setup_dist()
    device = dist_util.dev()
    codes = load_codebook(args.codebook_json, 100, 30)
    loader = load_data(
        data_dir=args.data_dir, batch_size=1, image_size=128, deterministic=True
    )
    cover = next(iter(loader))[0:1].to(device)

    defaults = endecoder_and_diffusion_defaults()
    model_args = {
        "image_size": 128,
        "message_length": 30,
        "embedding_dim": 256,
        "num_channels": 32,
        "num_heads": 4,
        "num_res_blocks": 1,
        "attention_resolutions": "16,8",
        "diffusion_steps": 100,
        "timestep_respacing": "ddim100",
    }
    for key, value in defaults.items():
        model_args.setdefault(key, value)
    model, diffusion = create_endecoder_and_diffusion(**model_args)
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(device)
    model.eval()

    colluders = np.asarray([98, 10, 4, 18], dtype=np.int64)
    message = th.from_numpy(codes[colluders]).float().to(device)
    with th.no_grad():
        samples = diffusion.ddim_sample_loop(
            model,
            cover.repeat(4, 1, 1, 1),
            message,
            clip_denoised=False,
            model_kwargs={},
        )
        mixed = mosaic_mix(samples, np.random.RandomState(1))

        for size in [64, 80, 96, 112, 128]:
            stride = 16 if size < 128 else 128
            windows = []
            for top in range(0, 128 - size + 1, stride):
                for left in range(0, 128 - size + 1, stride):
                    windows.append(mixed[:, :, top : top + size, left : left + size])
            batch = th.cat(windows, dim=0)
            if size != 128:
                batch = th.nn.functional.interpolate(
                    batch, size=(128, 128), mode="bilinear", align_corners=False
                )
            probs = decode_binary_probabilities(model.decoder(batch)).cpu().numpy()
            z = np.stack([z_scores(p, codes) for p in probs])
            zmax = z.max(axis=0)
            top = int(np.argmax(zmax))
            colluder_z = z[:, colluders].max(axis=0)
            print(
                f"size={size} windows={len(windows)} top1={top} "
                f"top1_is_colluder={top in set(colluders.tolist())} "
                f"top_z={zmax[top]:.3f} colluder_z={np.round(colluder_z, 3).tolist()}"
            )


if __name__ == "__main__":
    main()
