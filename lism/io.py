"""Files: probes and covariances as safetensors, a task's settings as config.json, images as PIL."""
import json
from pathlib import Path
import torch
from PIL import Image
from safetensors.torch import load_file, safe_open, save_file
from .covariance import Latent, Pooled


def save_probe(probe, path):
    """probe: dict(w, b), as returned by fit_probe."""
    save_file({k: torch.as_tensor(v).contiguous() for k, v in probe.items()}, str(path))


def load_probe(path):
    return load_file(str(path))


def save_covariance(cov, path):
    """Eigenvectors are stored in float32, everything else as fitted."""
    kind = "pooled" if isinstance(cov, Pooled) else "latent"
    fields = {k: torch.as_tensor(v) for k, v in vars(cov).items() if v is not None and not k.startswith("_")}
    fields["vectors"] = fields["vectors"].float()
    save_file({k: v.contiguous() for k, v in fields.items()}, str(path), metadata={"kind": kind})


def load_covariance(path, device="cuda"):
    with safe_open(str(path), "pt") as file:
        kind = file.metadata()["kind"]
    data = load_file(str(path))
    if kind == "pooled":
        return Pooled(data["values"].double().to(device), data["vectors"].double().to(device), int(data["patches"]))
    extra = [data[k].double().to(device) for k in ("direction", "along") if k in data]
    # The latent eigenvectors stay in float32: they are large.
    return Latent(data["values"].double().to(device), data["vectors"].to(device), float(data["residual"]), *extra)


def save_config(config, path):
    Path(path).write_text(json.dumps(config, indent=2) + "\n")


def load_config(path):
    return json.loads(Path(path).read_text())


def to_pil(image):
    """One CHW or single-image BCHW tensor in [0, 1] -> RGB image."""
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Expected one RGB image tensor")
    pixels = image.detach().clamp(0, 1).mul(255).round().byte()
    return Image.fromarray(pixels.permute(1, 2, 0).cpu().numpy())
