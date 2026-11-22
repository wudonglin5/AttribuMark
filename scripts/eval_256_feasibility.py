import argparse
import json
from pathlib import Path

import kornia.losses
import numpy as np
import torch as th

import lpips as clpips
from guided_diffusion import dist_util
from guided_diffusion.image_datasets import load_data
from guided_diffusion.noiser import Random_Noise
from guided_diffusion.script_util import create_endecoder_and_diffusion


def load_codes(path, num_users, message_length):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    codes = np.asarray([[int(bit) for bit in code] for code in data["codes"]], dtype=np.int64)
    if codes.shape[0] < num_users or codes.shape[1] != message_length:
        raise ValueError(f"bad codebook shape: {codes.shape}")
    return codes[:num_users]


def decode_probs(logits, message_length):
    positions = th.arange(message_length, device=logits.device)
    pair = th.stack(
        [logits[:, positions, positions], logits[:, positions, positions + message_length]],
        dim=-1,
    )
    return th.softmax(pair, dim=-1)[..., 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--codebook_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--code_start", type=int, default=2)
    parser.add_argument("--message_length", type=int, default=30)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--num_channels", type=int, default=32)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_res_blocks", type=int, default=1)
    parser.add_argument("--attention_resolutions", default="32,16")
    parser.add_argument("--channel_mult", default="0.5,1,2,4,8")
    parser.add_argument("--timestep_respacing", default="ddim10")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    dist_util.setup_dist()
    device = dist_util.dev()
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    codes = load_codes(args.codebook_json, 100, args.message_length)
    model, diffusion = create_endecoder_and_diffusion(
        image_size=256,
        message_length=args.message_length,
        embedding_dim=args.embedding_dim,
        class_cond=False,
        num_channels=args.num_channels,
        num_res_blocks=args.num_res_blocks,
        channel_mult=args.channel_mult,
        conv_resample=True,
        num_heads=args.num_heads,
        num_head_channels=-1,
        num_heads_upsample=-1,
        attention_resolutions=args.attention_resolutions,
        dropout=0.1,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=False,
        pool="linear",
        learn_sigma=False,
        diffusion_steps=100,
        noise_schedule="linear",
        timestep_respacing=args.timestep_respacing,
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(device).eval()
    vqgan = Random_Noise(["VQGAN()"])
    lpips_model = clpips.LPIPS(net="vgg").to(device).eval()
    loader = load_data(
        data_dir=args.data_dir,
        batch_size=1,
        image_size=256,
        deterministic=True,
    )

    records = []
    with th.no_grad():
        for index, images in enumerate(loader):
            if index >= args.num_images:
                break
            images = images.to(device)
            code_index = args.code_start + index
            message_np = codes[code_index:code_index + 1]
            message = th.from_numpy(message_np).float().to(device)
            sample = diffusion.ddim_sample_loop(
                model,
                images,
                message,
                clip_denoised=False,
                model_kwargs={},
            )
            quality = {
                "psnr": float((-kornia.losses.psnr_loss(sample, images, 2)).item()),
                "ssim": float((1 - 2 * kornia.losses.ssim_loss(sample, images, window_size=11, reduction="mean")).item()),
                "lpips": float(th.mean(lpips_model(sample, images)).item()),
            }
            clean_logits = model.decoder(sample)
            attacked = vqgan([sample, images])
            vqgan_logits = model.decoder(attacked)
            clean_probs = decode_probs(clean_logits, args.message_length)[0].cpu().numpy()
            vqgan_probs = decode_probs(vqgan_logits, args.message_length)[0].cpu().numpy()
            record = {"index": index, "true_user": code_index, **quality}
            for name, probs in (("clean", clean_probs), ("vqgan", vqgan_probs)):
                pred = (probs >= 0.5).astype(np.int64)
                hamming = np.sum(pred[None, :] != codes, axis=1)
                record[f"{name}_ber"] = float(np.mean(pred != codes[code_index]))
                record[f"{name}_hamming"] = int(hamming[code_index])
                record[f"{name}_exact"] = bool(hamming[code_index] == 0)
                record[f"{name}_top1"] = bool(int(np.argmin(hamming)) == code_index)
                record[f"{name}_predicted_user"] = int(np.argmin(hamming))
            records.append(record)
            print(f"{index + 1}/{args.num_images} clean_ber={record['clean_ber']:.4f} vqgan_ber={record['vqgan_ber']:.4f} top1={record['vqgan_top1']}", flush=True)

    summary = {}
    for key in ("psnr", "ssim", "lpips", "clean_ber", "clean_hamming", "vqgan_ber", "vqgan_hamming"):
        summary[key] = float(np.mean([record[key] for record in records]))
    summary["clean_exact_rate"] = float(np.mean([record["clean_exact"] for record in records]))
    summary["vqgan_exact_rate"] = float(np.mean([record["vqgan_exact"] for record in records]))
    summary["clean_top1"] = float(np.mean([record["clean_top1"] for record in records]))
    summary["vqgan_top1"] = float(np.mean([record["vqgan_top1"] for record in records]))
    output = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "num_images": len(records),
        "num_candidate_users": int(codes.shape[0]),
        "timestep_respacing": args.timestep_respacing,
        "summary": summary,
        "records": records,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
