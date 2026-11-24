"""A task is a folder: probe.safetensors, covariance.safetensors and config.json.

load_task reads one from a local folder or a Hugging Face model repository, so a probe fitted
by anyone loads the same way as the prepared ones.

config.json
    kind          class | attributes | mask | keypoints   (which request the probe answers)
    latent_shape  [C, H, W] of the generator's latent
    backbone      the generator the probe was fitted for
    labels        names of the attributes or joints (class: optional)
    presets       {"fit_path": {"alpha": ..., "window": [...], "cap": ...}, ...}  tuned strengths
    threshold     mask: probability threshold           margin, image_size   keypoints
    conditions    optional: generator conditions with their own covariance_<condition>.safetensors
"""
import os
from dataclasses import dataclass
from pathlib import Path
from . import request as requests
from .io import load_config, load_covariance, load_probe, save_config, save_covariance, save_probe

REPO = "penway47/LiSM"        # Hugging Face repository of the prepared tasks; $LISM_TASKS overrides it
KINDS = ("class", "attributes", "mask", "keypoints")


@dataclass
class Task:
    """A fitted probe, its covariance and its settings. `request(...)` turns what you ask for into (A, c)."""
    kind: str
    probe: dict
    cov: object
    config: dict
    device: str = "cuda"

    def request(self, wanted, **options):
        """class: target class id.  attributes: {name: +1 | -1}.  mask: binary mask on the latent grid.
        keypoints: [joints, 3] pixel x, y, visibility (2 = requested), or {joint name: (x, y)}."""
        w, b, shape, config = self.probe["w"], self.probe["b"], tuple(self.config["latent_shape"]), self.config
        if self.kind == "class":
            return requests.class_request(w, b, int(wanted), shape, self.device)
        if self.kind == "attributes":
            signs = {config["labels"].index(name): sign for name, sign in wanted.items()}
            return requests.attribute_request(w, b, signs, shape, self.device)
        if self.kind == "mask":
            options.setdefault("threshold", config.get("threshold", .2))
            return requests.mask_request(w, b, wanted, shape, device=self.device, **options)
        if isinstance(wanted, dict):
            wanted = requests.points_from_names(wanted, config["labels"])
        options.setdefault("margin", config.get("margin", 1.))
        options.setdefault("image_size", config.get("image_size", 256))
        return requests.keypoint_request(w, wanted, shape, device=self.device, **options)


def _folder(name, source):
    source = source or os.environ.get("LISM_TASKS") or REPO
    if source is None:
        raise FileNotFoundError("No task source: pass source=<folder or Hugging Face repository> or set LISM_TASKS")
    if Path(source).is_dir():
        return Path(source) / name
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(source, allow_patterns=[f"{name}/*"])) / name


def load_task(name, source=None, condition=None, device="cuda"):
    """Load task `name` from a folder or a Hugging Face repository that holds one subfolder per task.

    condition: for tasks with one covariance per generator condition (the attributes: an ImageNet class id).
    """
    folder = _folder(name, source)
    config = load_config(folder / "config.json")
    if config["kind"] not in KINDS:
        raise ValueError(f"Unknown task kind {config['kind']!r}")
    if "conditions" in config:
        if condition not in config["conditions"]:
            raise KeyError(f"Task {name!r} has one covariance per condition: pass condition, one of {config['conditions']}")
        file = folder / f"covariance_{condition}.safetensors"
    else:
        file = folder / "covariance.safetensors"
    cov = load_covariance(file, device) if file.exists() else None      # Point weighting needs none
    return Task(config["kind"], load_probe(folder / "probe.safetensors"), cov, config, device)


def save_task(folder, kind, probe, cov=None, **config):
    """Write a task folder that load_task reads. config: latent_shape, backbone, labels, presets, ..."""
    if kind not in KINDS:
        raise ValueError(f"Kind is one of {KINDS}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    save_probe(probe, folder / "probe.safetensors")
    if cov is not None:
        save_covariance(cov, folder / "covariance.safetensors")
    save_config(dict(kind=kind, **config), folder / "config.json")
