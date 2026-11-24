"""Algorithm 1: one guided velocity evaluation. Everything else is a choice of (A, c), L, G, R."""
from collections import namedtuple
from dataclasses import dataclass
import torch

# What the rules G and R may read at one evaluation: state, time, clean prediction, denoiser, request.
Step = namedtuple("Step", "z t z_hat f A")


def velocity(z, t, f, A, L, G, R, alpha=1., window=(.01, .995), cap=None, step=None):
    if not 0 < t <= 1:
        raise ValueError("Velocity requires 0 < t <= 1")
    active = alpha != 0 and window[0] <= t <= window[1]
    graph = active and (getattr(G, "needs_grad", False) or getattr(R, "needs_grad", False))
    z = z.detach().requires_grad_(graph)          # VJP-type rules reuse the graph of line (1)
    with torch.set_grad_enabled(graph):
        z_hat = f(z, t)                           # (1) clean prediction
        if not active:
            return (z - z_hat).detach() / t
        s = Step(z, t, z_hat, f, A)
        mu = A.scores(z_hat)                      # (1) mean scores           mu = A z_hat + c
        lam = G(L, mu, s)                         # (2) request weights
        u = A.adjoint(lam).detach()               # (2) probe direction       u = A^T lam
        delta = alpha * R(u, s)                   # (3) representation correction
        if cap is not None:                       # optional: bound the guided part of one Euler step
            delta = delta * _cap(delta, z, t, step, cap)
        v = (z - (z_hat + delta.to(z_hat))) / t   # (4) guided velocity
    if not torch.isfinite(v).all():
        raise FloatingPointError("Nonfinite LiSM velocity")
    return v.detach()


def _cap(delta, z, t, step, cap):
    """Scale so that the state change delta * step / t stays within `cap` of the state's RMS."""
    rms = lambda x: x.detach().double().square().flatten(1).mean(1).sqrt()
    fraction = rms(delta) * step / t / rms(z).clamp_min(1e-8)
    return (cap / fraction.clamp_min(cap)).reshape(-1, *[1] * (delta.ndim - 1))


@dataclass
class Guide:
    """The method-specific inputs of Algorithm 1: weight function, two rules and the strength."""
    L: object
    G: object
    R: object
    alpha: float = 1.
    window: tuple = (.01, .995)
    cap: float = None               # largest guided change of the state per Euler step, relative to its RMS
