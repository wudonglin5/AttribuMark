import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from guided_diffusion.script_util import create_endecoder


def build(image_size, attention_resolutions):
    return create_endecoder(
        image_size=image_size,
        message_length=30,
        embedding_dim=256,
        num_channels=32,
        num_res_blocks=1,
        channel_mult="0.5,1,2,4,8",
        conv_resample=True,
        learn_sigma=False,
        class_cond=False,
        use_checkpoint=False,
        attention_resolutions=attention_resolutions,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        dropout=0.1,
        resblock_updown=True,
        use_fp16=False,
        pool="linear",
    )


def resize_parameter(value, target_shape):
    if value.shape == target_shape:
        return value
    src = value.float()
    if src.ndim == 1:
        out = F.interpolate(src.view(1, 1, 1, -1), size=(1, target_shape[0]), mode="bilinear", align_corners=False)
        return out.view(target_shape)
    if src.ndim == 2:
        out = F.interpolate(src.view(1, 1, src.shape[0], src.shape[1]), size=tuple(target_shape), mode="bilinear", align_corners=False)
        return out.view(target_shape)
    raise ValueError(f"unsupported parameter shape: {tuple(value.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_state = torch.load(args.source, map_location="cpu")
    target = build(256, "32,16")
    target_state = target.state_dict()
    migrated = {}
    resized = []
    untouched = []
    with torch.no_grad():
        for name, target_value in target_state.items():
            if name not in source_state:
                migrated[name] = target_value.clone()
                untouched.append(name)
                continue
            source_value = source_state[name]
            if source_value.shape == target_value.shape:
                migrated[name] = source_value.clone()
            else:
                migrated[name] = resize_parameter(source_value, tuple(target_value.shape))
                resized.append((name, tuple(source_value.shape), tuple(target_value.shape)))
        target.load_state_dict(migrated, strict=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(target.state_dict(), output)
    print(f"saved: {output}")
    print(f"parameters: {sum(p.numel() for p in target.parameters())}")
    print(f"direct/unchanged: {len(target_state) - len(resized) - len(untouched)}")
    print(f"resized: {len(resized)}")
    for item in resized:
        print("  RESIZE", item)
    print(f"newly initialized: {len(untouched)}")


if __name__ == "__main__":
    main()
