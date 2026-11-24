"""Weight functions L and weighting rules G:  lam = G(L, mu).

L says what a satisfied request is; G says how uncertainty about the clean scores enters.
Point reads L at the mean scores. Iso and Fit average L over a Gaussian around them.
"""
import math
import numpy as np
import torch


# ---- weight functions L: derivative of the log-likelihood of the request ----

class Event:
    """Hard event: every score is positive. Only defined under uncertainty."""
    def point(self, mu):
        raise ValueError("A hard event has no pointwise gradient: Point weighting uses Logistic")

    def smoothed(self, mu, variance):
        if not torch.isfinite(variance).all() or (variance <= 0).any():
            raise ValueError("Score variances must be finite and positive")
        sd = variance.sqrt()
        eta = mu / sd                             # d/dmu log Phi(mu/sd)
        return torch.exp(-.5 * eta.square() - .5 * math.log(2 * math.pi) - torch.special.log_ndtr(eta)) / sd


    def shared(self, mu, wanted, rival, cross, slots, filled, tolerance=1e-3):
        """Groups of scores "wanted minus rival" that share the wanted score: one event per group.

        mu [M]; wanted [J] variance of the wanted score; rival, cross [J, R] variance of each rival
        score and its covariance with the wanted one; slots, filled [J, R] the rows of each group.
        Given the wanted score, the rivals are taken as independent, which leaves a one-dimensional
        integral over the standardized wanted score x per group, with log-density
            f(x) = -x^2/2 + sum_r log Phi(offset_r + slope_r x),      strictly concave.
        Returns d log P / d mu [M]. Covariances are held fixed.
        """
        mask = filled.to(mu.dtype)
        sd = wanted.sqrt()[:, None]
        conditional = rival - cross.square() / wanted[:, None]
        if (wanted <= 0).any() or ((conditional <= 0) & filled).any():
            raise ValueError("Shared-target conditional variances must be positive")
        tau = torch.where(filled, conditional, torch.ones_like(conditional)).sqrt()
        offset = torch.where(filled, mu[slots] / tau, torch.zeros_like(tau))
        slope = torch.where(filled, (sd - cross / sd) / tau, torch.zeros_like(tau))
        log_pdf = lambda a: -.5 * a * a - .5 * math.log(2 * math.pi)

        def density(x):                             # x [J, N] -> f [J, N]
            return -.5 * x * x + (torch.special.log_ndtr(offset[:, None] + x[:, :, None] * slope[:, None]) * mask[:, None]).sum(-1)

        # Mode: f' is decreasing, so bracket it by its sign on doubling points, then refine a grid argmax.
        powers = 2. ** torch.arange(11).to(mu)
        x = torch.cat([-powers.flip(0), torch.zeros(1).to(mu), powers])
        a = offset[:, None] + x[None, :, None] * slope[:, None]
        rising = (-x[None] + (slope[:, None] * torch.exp(log_pdf(a) - torch.special.log_ndtr(a)) * mask[:, None]).sum(-1)) > 0
        count = rising.sum(1)
        left = torch.where(count == 0, 2 * x[0], x[(count - 1).clamp(0)])
        right = torch.where(count == len(x), 2 * x[-1], x[count.clamp(max=len(x) - 1)])
        for _ in range(4):
            grid = left[:, None] + (right - left)[:, None] * torch.linspace(0, 1, 33).to(mu)[None]
            best = density(grid).argmax(1, keepdim=True)
            left, right = grid.gather(1, (best - 1).clamp(0)).squeeze(1), grid.gather(1, (best + 1).clamp(max=32)).squeeze(1)
        mode = (left + right) / 2
        # Integration interval: where f is within 55 of its peak, inside mode +/- 12, rounded outward.
        grid = mode[:, None] + torch.linspace(-12, 12, 193).to(mu)[None]
        keep = density(grid) >= density(mode[:, None]) - 55.
        index = torch.arange(193, device=mu.device)
        lo = grid.gather(1, (torch.where(keep, index, 193).min(1).values - 1).clamp(0)[:, None]).squeeze(1)
        hi = grid.gather(1, (torch.where(keep, index, -1).max(1).values + 1).clamp(max=192)[:, None]).squeeze(1)

        # Gauss-Legendre with doubling node counts; each group takes the first count that agrees with the previous one.
        result, done, previous = torch.zeros_like(tau), torch.zeros(len(tau), dtype=torch.bool, device=mu.device), None
        for nodes in (32, 64, 128, 256, 512, 1024):
            points, weights = (torch.as_tensor(v).to(mu) for v in np.polynomial.legendre.leggauss(nodes))
            x = (hi + lo)[:, None] / 2 + (hi - lo)[:, None] / 2 * points[None]
            a = offset[:, None] + x[:, :, None] * slope[:, None]
            log_cdf = torch.special.log_ndtr(a)
            mass = -.5 * x * x + (log_cdf * mask[:, None]).sum(-1) + weights.log()[None]
            total = torch.logsumexp(mass, 1)
            gradient = torch.einsum("jn,jnr->jr", torch.exp(mass - total[:, None]), torch.exp(log_pdf(a) - log_cdf) * mask[:, None]) / tau
            log_probability = total + torch.log((hi - lo) / 2) - .5 * math.log(2 * math.pi)
            if previous is not None:
                change = (gradient - previous[0]).norm(dim=1) / gradient.norm(dim=1).clamp_min(1e-30)
                agree = (change <= tolerance) & ((log_probability - previous[1]).abs() <= 1e-5) & ~done
                result[agree], done = gradient[agree], done | agree
                if done.all():
                    break
            previous = gradient, log_probability
        if not done.all():
            raise ArithmeticError("Shared-target quadrature unresolved")
        if not torch.isfinite(result).all() or (result < 0).any():
            raise ArithmeticError("Invalid shared-target weights")
        lam = torch.zeros_like(mu)
        lam[slots[filled]] = result[filled]
        return lam


class Logistic:
    """Soft event log sigmoid(score): the weight function of Point weighting."""
    def point(self, mu):
        return torch.sigmoid(-mu)

    def smoothed(self, mu, variance):
        raise ValueError("Logistic has no closed-form Gaussian average: Iso and Fit use Event")


# ---- weighting rules G(L, mu, s) ----

def point():
    return lambda L, mu, s: L.point(mu.detach())


def fit(cov):
    """Gaussian score uncertainty from a fitted covariance, conditioned on z_t.

    Scores are independent events, except the groups of a request that share a wanted score.
    """
    def G(L, mu, s):
        at, mu = cov.at(s.t), mu.detach()
        if getattr(s.A, "group", None) is None:
            return L.smoothed(mu, s.A.variance(at))
        moments = s.A.shared(at)
        return torch.stack([L.shared(row, *moments) for row in mu])
    return G


def iso(cov):
    """The same with an isotropic covariance of equal total variance."""
    return fit(cov.iso())


WEIGHTINGS = {"point": lambda cov: point(), "iso": iso, "fit": fit}
DEFAULT_L = {"point": Logistic, "iso": Event, "fit": Event}
