"""
Extract individual component weights from Hallo's combined net.pth.

Hallo's train_stage2.py expects separate .pth files in stage1_ckpt_dir:
  reference_unet.pth, denoising_unet.pth, face_locator.pth, imageproj.pth

The HuggingFace release only ships net.pth (combined state dict). This
script splits it so per-actor fine-tuning can proceed.

Run once before running finetune_hallo.py:
  python src/track3/extract_hallo_components.py \
    --net_pth tools/Hallo/pretrained_models/hallo/net.pth \
    --out_dir tools/Hallo/pretrained_models/hallo
"""

import argparse
from collections import defaultdict
from pathlib import Path

import torch


COMPONENT_PREFIXES = {
    "reference_unet":  "reference_unet.",
    "denoising_unet":  "denoising_unet.",
    "face_locator":    "face_locator.",
    "imageproj":       "imageproj.",
    "audioproj":       "audioproj.",
}


def main():
    parser = argparse.ArgumentParser(
        description="Split Hallo net.pth into per-component .pth files."
    )
    parser.add_argument("--net_pth", required=True,
                        help="Path to pretrained_models/hallo/net.pth")
    parser.add_argument("--out_dir", required=True,
                        help="Directory to write component .pth files")
    args = parser.parse_args()

    net_pth = Path(args.net_pth)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {net_pth} ...")
    state = torch.load(str(net_pth), map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        # net.pth may contain {"state_dict": {...}}
        state = state.get("state_dict", state)

    # Partition keys by component prefix
    components: dict[str, dict] = defaultdict(dict)
    for key, tensor in state.items():
        for comp_name, prefix in COMPONENT_PREFIXES.items():
            if key.startswith(prefix):
                # Strip the prefix so weights load cleanly into the sub-module
                components[comp_name][key[len(prefix):]] = tensor
                break

    for comp_name, weights in components.items():
        out_path = out_dir / f"{comp_name}.pth"
        torch.save(weights, str(out_path))
        print(f"  {comp_name}.pth - {len(weights)} tensors -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
