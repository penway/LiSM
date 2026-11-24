"""Fit the two things LiSM needs from data: an affine probe (W, b) and a Gaussian covariance.

Both are fitted on clean latents z = E(x) of the generator's own encoder (extract.py); labels
come from any dataset. Nothing here touches the generator. The command line writes a task
folder that lism.load_task reads:

    python -m lism.fit probe KIND latents.npy labels.npy TASK_FOLDER       KIND: class | attributes | mask | keypoints
    python -m lism.fit cov pooled latents.npy TASK_FOLDER                  class, attributes
    python -m lism.fit cov latent latents.npy TASK_FOLDER [--calibrate]    mask, keypoints
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .covariance import Latent, Pooled


def fit_probe(latents, labels, kind, epochs=100, lr=1e-2, decay=1e-4, batch=256, image_size=256, device="cuda", seed=0):
    """Affine probe on clean latents [N, C, H, W]. Returns dict(w, b).

    class and attributes only read the spatial mean and also accept pooled latents [N, C].

    kind        labels                               probe
    class       [N] class ids                        w [K, C] on the spatial mean, softmax
    attributes  [N, A] in {0, 1}, NaN = unlabeled    w [A, C] on the spatial mean, logistic
    mask        [N, H, W] in [0, 1]                  w [C] on every patch, logistic
    keypoints   [N, J, 3] pixel x, y, visible (2)    w [J, C] heatmap, softmax over patches
    """
    if kind not in ("class", "attributes", "mask", "keypoints"):
        raise ValueError(kind)
    labels = torch.as_tensor(np.asarray(labels))
    fetch = lambda ids: torch.as_tensor(np.asarray(latents[np.sort(np.asarray(ids))])).float()   # arrays may be memory-mapped
    pooled = kind in ("class", "attributes")
    # Standardize what the probe reads, for conditioning; the returned (w, b) act on raw latents.
    torch.manual_seed(seed)
    pool = lambda x: x if x.ndim == 2 else x.mean((2, 3))
    some = fetch(torch.randperm(len(latents))[:2048])
    if pooled:
        mean, std = pool(some).mean(0).to(device), pool(some).std(0).clamp_min(1e-6).to(device)
    else:
        mean, std = some.mean((0, 2, 3)).to(device), some.std((0, 2, 3)).clamp_min(1e-6).to(device)
    outputs = {"class": lambda: int(labels.max()) + 1, "attributes": lambda: labels.shape[1],
               "mask": lambda: 1, "keypoints": lambda: labels.shape[1]}[kind]()
    w = torch.zeros((outputs, latents.shape[1]), device=device, requires_grad=True)
    b = torch.zeros(outputs, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([w, b], lr=lr, weight_decay=decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * -(-len(latents) // batch))
    for _ in range(epochs):
        for ids in torch.randperm(len(latents)).split(batch):
            ids = ids.sort().values
            x, y = fetch(ids).to(device), labels[ids].to(device)
            if pooled:
                logits = (pool(x) - mean) / std @ w.T + b
            else:
                x = (x - mean[:, None, None]) / std[:, None, None]
                logits = torch.einsum("bcp,jc->bjp", x.flatten(2), w) + b[:, None]       # [B, J, P]
            if kind == "class":
                loss = F.cross_entropy(logits, y.long())
            elif kind == "attributes":
                known = ~torch.isnan(y.float())
                loss = F.binary_cross_entropy_with_logits(logits[known], y.float()[known])
            elif kind == "mask":
                loss = F.binary_cross_entropy_with_logits(logits[:, 0], y.float().flatten(1))
            else:
                height, width = latents.shape[2:]
                visible = y[:, :, 2] == 2
                grid = torch.round(y[:, :, :2].float() * width / image_size - .5).long()     # nearest patch centre
                cell = grid[:, :, 1].clamp(0, height - 1) * width + grid[:, :, 0].clamp(0, width - 1)
                loss = F.cross_entropy(logits[visible], cell[visible])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
    w, b = w.detach() / std, b.detach()
    b = b - w @ mean
    if kind == "mask":
        w = w[0]
    return dict(w=w.double().cpu(), b=b.double().cpu())


def fit_covariance(latents, kind, rank=16, shrinkage=.05, patches=None, probe=None, seed=0):
    """Gaussian fit of clean latents [N, C, H, W]: 'pooled' (global requests) or 'latent' (local requests).

    Fit it on images of the generator's condition, e.g. one class, when the request keeps that condition.
    'pooled' also accepts spatial means [N, C] with the number of `patches` they average.
    'latent' with `probe` (channel weights [C] of a per-patch probe, e.g. the mask head) fits the
    variance along the probe direction at every patch, so the probe's own score variance is exact.
    """
    x = torch.from_numpy(np.array(latents))           # a copy: the file may be memory-mapped read-only
    if kind == "pooled":
        x = x.double()
        if x.ndim == 4:
            patches = x.shape[2] * x.shape[3]
            x = x.mean((2, 3))
        cov = torch.cov(x.T)
        cov = (1 - shrinkage) * cov + shrinkage * cov.trace() / len(cov) * torch.eye(len(cov), dtype=cov.dtype)
        values, vectors = torch.linalg.eigh(cov)
        if values.min() <= 0:
            raise ValueError("Pooled covariance must be positive definite: use more images or more shrinkage")
        return Pooled(values, vectors, patches)
    if kind != "latent":
        raise ValueError(kind)
    x = x.float().flatten(1)
    x = x - x.mean(0)
    torch.manual_seed(seed)
    _, s, v = torch.pca_lowrank(x, q=min(2 * rank, len(x) - 1), center=False, niter=4)
    values = s[:rank].double().square() / (len(x) - 1)
    total = float(x.double().square().sum() / (len(x) - 1))
    residual = (total - float(values.sum())) / (x.shape[1] - rank)
    if residual <= 0 or not bool((values > residual).all()):
        raise ValueError("Rank is too large for this many images")
    cov = Latent(values, v[:, :rank].contiguous(), residual)
    if probe is None:
        return cov
    direction = torch.as_tensor(np.asarray(probe)).double().flatten()
    cov.direction = direction / direction.norm()
    channels = len(cov.direction)
    scores = torch.einsum("ncp,c->np", x.reshape(len(x), channels, -1), cov.direction.float()).double()
    cov.along = scores.square().sum(0) / (len(x) - 1) - cov.projected().square() @ (values - residual)
    if not bool((cov.along > 0).all()):
        raise ValueError("The low-rank part already exceeds the probe's score variance: lower the rank")
    return cov


def main():
    from .hub import save_task
    from .io import load_config, load_probe, save_covariance
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=("probe", "cov"))
    parser.add_argument("kind")
    parser.add_argument("latents")
    parser.add_argument("rest", nargs="+", help="probe: labels.npy TASK_FOLDER; cov: TASK_FOLDER")
    parser.add_argument("--grid", type=int, nargs=2, default=(16, 16), help="latent grid H W, for pooled latents [N, C]")
    parser.add_argument("--labels", nargs="+", help="names of the attributes or joints, stored in config.json")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--calibrate", action="store_true", help="cov latent: fit the variance along the folder's probe")
    args = parser.parse_args()
    latents = np.load(args.latents, mmap_mode="r")
    shape = latents.shape[1:] if latents.ndim == 4 else (latents.shape[1], *args.grid)
    if args.what == "probe":
        labels, folder = args.rest
        probe = fit_probe(latents, np.load(labels), args.kind)
        extra = dict(labels=args.labels) if args.labels else {}
        save_task(folder, args.kind, probe, latent_shape=list(shape), **extra)
    else:
        folder, = map(Path, args.rest)
        probe = load_probe(folder / "probe.safetensors")["w"] if args.calibrate else None
        if probe is not None and load_config(folder / "config.json")["kind"] != "mask":
            raise ValueError("--calibrate needs a probe with one channel row (a mask probe)")
        cov = fit_covariance(latents, args.kind, rank=args.rank, patches=shape[1] * shape[2], probe=probe)
        save_covariance(cov, folder / "covariance.safetensors")


if __name__ == "__main__":
    main()
