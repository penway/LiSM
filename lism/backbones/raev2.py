"""RAEv2 (DINOv3-L latents, 1024 x 16 x 16): a thin adapter to the upstream code and checkpoints.

Code: the RAEv2 repository, by default the submodule third_party/RAEv2 of this repository
(git clone --recursive), else `rae_root` or $LISM_RAE_ROOT.
Checkpoints: `weights` or $LISM_RAE_WEIGHTS, else <RAEv2>/pretrained_models if it exists, else
they are downloaded once from the Hugging Face repository nyu-visionx/RAEv2-models (about 12.6 GB).
The upstream DINOv3 loader fetches the DINOv3 code from GitHub unless $DINOV3_REPO_DIR names a
local checkout.
"""
from pathlib import Path
import os
import sys
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

SUBMODULE = Path(__file__).resolve().parents[2] / "third_party/RAEv2"
WEIGHTS_REPO = "nyu-visionx/RAEv2-models"
WEIGHTS = ("stage1/imagenet/dinov3l-k7/*", "stage2/imagenet/dinov3l-k7/*",
           "encoders/dinov3/dinov3_vitl16_pretrain_lvd1689m-*.pth")


class RAEModel:
    def __init__(self, predictor, codec, latent_shape, device, class_id=1000):
        self.predictor = predictor.eval().requires_grad_(False)
        self.codec = codec.eval().requires_grad_(False)
        self.codec.encoder.cpu()                       # keep GPU memory for prediction and decoding
        self.latent_shape = tuple(latent_shape)
        self.device = torch.device(device)
        self.dtype = next(predictor.parameters()).dtype
        self.class_id = class_id                       # native class condition; 1000 is the null class

    def predict_clean(self, z, t):
        time = torch.full((len(z),), t, device=z.device, dtype=z.dtype)
        context = torch.full((len(z),), self.class_id, device=z.device, dtype=torch.long)
        with sdpa_kernel(SDPBackend.MATH):             # MATH attention supports forward-mode differentiation
            result = self.predictor(z, time, context=context, attn_mask=None, omega=torch.ones_like(time))
        return result[0] if isinstance(result, tuple) else result

    def encode(self, image):
        """RGB in [0, 1], [B, 3, H, W] -> normalized latent on the model device."""
        encoder_device = next(self.codec.encoder.parameters()).device
        with torch.no_grad():
            return self.codec.encode(image.to(encoder_device)).to(self.device)

    def decode(self, latent):
        return self.codec.decode(latent)


def _weights(root, weights):
    weights = weights or os.environ.get("LISM_RAE_WEIGHTS")
    if weights:
        return Path(weights).resolve()
    if (root / "pretrained_models").is_dir():
        return root / "pretrained_models"
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(WEIGHTS_REPO, allow_patterns=list(WEIGHTS)))


def load_rae(device="cuda", class_id=1000, rae_root=None, weights=None, config=None):
    # Full FP32: cuDNN convolutions otherwise default to TF32 and change trajectories.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(rae_root or os.environ.get("LISM_RAE_ROOT", SUBMODULE)).resolve()
    if not (root / "src").is_dir():
        raise FileNotFoundError(f"No RAEv2 code at {root}: clone this repository with --recursive, "
                                "run git submodule update --init, or set LISM_RAE_ROOT")
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    weights = _weights(root, weights)
    os.environ.setdefault("DINOV3_CKPT_DIR", str(weights / "encoders/dinov3"))
    from omegaconf import OmegaConf
    from configs import Stage2Config
    from utils.model_utils import get_obj_from_str, instantiate_from_config

    raw = OmegaConf.load(config or root / "configs/stage2/sampling/imagenet-dinov3l-k7.yaml")
    placed = lambda path: str(weights / Path(path).relative_to("pretrained_models"))      # configs name them under the checkout
    raw.stage_1.params.decoder_config_path = str(root / raw.stage_1.params.decoder_config_path)
    for key in ("pretrained_decoder_path", "normalization_stat_path"):
        raw.stage_1.params[key] = placed(raw.stage_1.params[key])
    raw.stage_2.ckpt = placed(raw.stage_2.ckpt)
    raw.guidance.cfg.scale = 1.
    raw.guidance.ig.scale = 1.
    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Stage2Config), raw))
    cfg.post_process()
    cfg.prepare_model_params()
    if cfg.transport.prediction != "x":
        raise ValueError("This adapter requires a clean-prediction RAE checkpoint")
    codec = get_obj_from_str(raw.stage_1.target)(**OmegaConf.to_container(raw.stage_1.params))
    predictor = instantiate_from_config(cfg.stage_2)
    return RAEModel(predictor.to(device), codec.to(device), cfg.misc.latent_size, device, class_id)
