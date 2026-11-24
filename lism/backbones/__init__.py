"""Generators with a clean-prediction denoiser f(z, t) and a decoder D. Register more in BACKBONES."""
from .raev2 import load_rae

BACKBONES = {"raev2": load_rae}
