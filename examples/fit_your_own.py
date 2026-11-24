"""Fit your own class probe from a folder of images and guide with it.

    python examples/fit_your_own.py IMAGES my_task [target]

IMAGES has one subfolder per class (IMAGES/<class name>/*.jpg). The script encodes the images,
fits an affine probe and a covariance on the pooled latents, writes the task folder `my_task`,
and samples the class `target` (default: the first one) with Fit x Path under the null condition.

The other probe kinds work the same way with their own labels (see lism/fit.py):
attributes [N, A], mask [N, H, W] on the latent grid, keypoints [N, J, 3].
"""
import sys
from pathlib import Path
import numpy as np
from lism import guide, load_task, sample, save_task, to_pil
from lism.backbones import load_rae
from lism.extract import SUFFIXES, encode
from lism.fit import fit_covariance, fit_probe


def main(images, folder, target=None, alpha=.125):
    images, folder = Path(images), Path(folder)
    names = sorted(p.name for p in images.iterdir() if p.is_dir())
    files = [(p, k) for k, name in enumerate(names) for p in sorted((images / name).rglob("*")) if p.suffix.lower() in SUFFIXES]
    model = load_rae()
    model.codec.encoder.to(model.device)

    # 1. Clean latents of the generator's encoder. A class probe only reads their spatial mean.
    pooled = encode(model, [p for p, _ in files], pooled=True).numpy()
    labels = np.array([k for _, k in files])

    # 2. The probe (W, b) and the covariance of the pooled latents.
    probe = fit_probe(pooled, labels, "class", device=model.device)
    shape = model.latent_shape
    cov = fit_covariance(pooled, "pooled", patches=shape[1] * shape[2])
    save_task(folder, "class", probe, cov, latent_shape=list(shape), labels=names)

    # 3. Guide. A new probe has no tuned strength: alpha is yours to choose (try a few values).
    task = load_task(folder.name, folder.parent)
    target = names.index(target) if target else 0
    image = sample(model, task.request(target), guide("fit", "path", task.cov, alpha=alpha))
    to_pil(image).save(folder / f"{names[target]}.png")
    print(f"Wrote {folder}/{names[target]}.png")


if __name__ == "__main__":
    main(*sys.argv[1:])
