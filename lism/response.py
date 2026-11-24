"""Response rules R:  delta = R(u), the change of the clean prediction for a probe direction u.

c_t = t^2 / (1 - t) converts a shift of z_t into the tilt of the clean prediction.
Id and Cov need no extra network call; JVP, FD and Path query the denoiser forward; VJP
goes backward through the graph of the clean prediction.
"""
import torch


def _c(t):
    return t * t / (1 - t)


def normalized(u):
    """Per image: unit direction (largest patch norm 1) and that largest patch norm."""
    u = u.detach().double()
    if not torch.isfinite(u).all():
        raise FloatingPointError("Nonfinite probe direction")
    shape = (-1,) + (1,) * (u.ndim - 1)
    peak = u.flatten(1).abs().amax(1).reshape(shape)                 # rescale first: tiny directions stay finite
    scaled = u / torch.where(peak > 0, peak, 1.)
    length = scaled.square().sum(1, keepdim=True).sqrt().flatten(1).amax(1).reshape(shape)
    return scaled / torch.where(length > 0, length, 1.), peak * length


def identity():
    return lambda u, s: u


def covariance(cov):
    """Fitted posterior covariance times u: no network call."""
    return lambda u, s: cov.at(s.t).apply(u)


def jvp():
    def R(u, s):
        unit, size = normalized(u)
        with torch.no_grad():
            _, tangent = torch.func.jvp(lambda x: s.f(x, s.t), (s.z.detach(),), (unit.to(s.z),))
        return _c(s.t) * size * tangent.double()
    return R


def vjp():
    def R(u, s):
        if not torch.any(u != 0):
            return torch.zeros_like(u)
        gradient, = torch.autograd.grad(s.z_hat, s.z, grad_outputs=u.to(s.z_hat), retain_graph=True)
        return _c(s.t) * gradient.double()
    R.needs_grad = True
    return R


def path(radius=2., fixed=False):
    """One forward query along u at the requested shift c_t |u|, at most `radius` (largest patch norm).

    The response is scaled back to the full requested shift. fixed: always query at exactly `radius`.
    """
    @torch.no_grad()
    def R(u, s):
        unit, size = normalized(u)
        if not torch.any(size > 0):
            return torch.zeros_like(u)
        requested = _c(s.t) * size
        r = torch.full_like(requested, radius) if fixed else requested.clamp(max=radius)
        r = torch.where(size > 0, r, torch.ones_like(r))
        response = s.f(s.z.detach() + (r * unit).to(s.z), s.t).double() - s.z_hat.detach().double()
        return requested / r * response
    return R


def fd(radius=.2):
    """Finite difference: Path with a small fixed query radius, extrapolated to the requested shift."""
    return path(radius, fixed=True)


RESPONSES = {"id": lambda cov: identity(), "cov": covariance, "jvp": lambda cov: jvp(),
             "vjp": lambda cov: vjp(), "fd": lambda cov: fd(), "path": lambda cov: path()}
