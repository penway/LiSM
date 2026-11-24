"""Sampling: noise, time grid, ODE solve of the guided velocity, decode."""
import torch
from .lism import velocity


def time_grid(steps=100, shift=8.):
    if steps < 1 or shift <= 0:
        raise ValueError("Steps and time shift must be positive")
    times = torch.linspace(1., 0., steps + 1)
    return [float(t) for t in shift * times / (1 + (shift - 1) * times)]


def euler(velocity, z, times, callback=None):
    """dz/dt = v(z, t) from t = 1 to 0 on a decreasing time grid."""
    times = [float(t) for t in times]
    if len(times) < 2 or times[0] > 1 or times[-1] < 0 or not all(a > b for a, b in zip(times, times[1:])):
        raise ValueError("Expected a strictly decreasing time grid inside [0, 1]")
    z = z.detach().clone()
    for step, (t, t_next) in enumerate(zip(times, times[1:])):
        v = velocity(z, t)
        if callback is not None:
            callback(step, t, z, v)                 # may raise to stop sampling
        with torch.no_grad():
            z = z + (t_next - t) * v
    return z


# An ODE solver is ode_solver(velocity, z, times, callback=None) -> z. Register more here.
ODE_SOLVERS = {"euler": euler}


def sample(model, request, guide, seed=42, steps=100, shift=8., ode_solver="euler", callback=None, decode=True):
    """One image. `guide` holds (L, G, R, alpha); `request` is (A, c); the model supplies f and D."""
    generator = torch.Generator(device="cpu").manual_seed(seed)       # CPU noise: independent of the GPU RNG
    z = torch.randn((1, *model.latent_shape), generator=generator).to(device=model.device, dtype=model.dtype)
    times = time_grid(steps, shift)
    size = dict(zip(times, (a - b for a, b in zip(times, times[1:]))))      # step taken from each grid time (for the cap)
    v = lambda z, t: velocity(z, t, model.predict_clean, request, guide.L, guide.G, guide.R, guide.alpha, guide.window,
                              guide.cap, size.get(t))
    solve = ODE_SOLVERS[ode_solver] if isinstance(ode_solver, str) else ode_solver
    z = solve(v, z, times, callback=callback)
    if not decode:
        return z
    with torch.no_grad():
        return model.decode(z)
