"""Encode images into clean latents, the only input the fitting code needs.

    python -m lism.extract IMAGE_FOLDER latents.npy [--pooled]

Writes latents [N, C, H, W] (float16) in sorted file order, and the file list next to it.
--pooled writes spatial means [N, C] (float32), which is all a class or attribute probe and a
pooled covariance read: about 1/250 of the size.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def to_tensor(image, size=256):
    """Center-crop to a square, resize, RGB in [0, 1]."""
    image = image.convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side)).resize((size, size), Image.BICUBIC)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255


def encode(model, images, batch=32, size=256, pooled=False):
    """images: PIL images or paths. Returns latents [N, C, H, W] (float16) or their spatial means [N, C] on the CPU."""
    images, out = list(images), []
    for start in range(0, len(images), batch):
        chunk = [image if isinstance(image, Image.Image) else Image.open(image) for image in images[start:start + batch]]
        z = model.encode(torch.stack([to_tensor(image, size) for image in chunk]))
        out.append(z.mean((2, 3)).float().cpu() if pooled else z.half().cpu())
    return torch.cat(out)


def main():
    from .backbones import load_rae
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--pooled", action="store_true")
    args = parser.parse_args()
    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in SUFFIXES)
    model = load_rae()
    model.codec.encoder.to(model.device)
    np.save(args.output, encode(model, files, pooled=args.pooled).numpy())
    args.output.with_suffix(".txt").write_text("\n".join(str(p.relative_to(args.folder)) for p in files))


if __name__ == "__main__":
    main()
