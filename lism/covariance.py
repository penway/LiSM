"""Fitted Gaussian uncertainty of the clean representation.

Pooled: covariance of the spatial mean, in channel coordinates (global requests).
Latent: low rank plus a diagonal residual over the whole latent (local requests).
Both give the posterior at time t, where z_t = (1-t) z_0 + t eps.
"""
from dataclasses import dataclass
import torch


@dataclass
class Pooled:
    values: torch.Tensor            # [C] eigenvalues of the pooled-mean covariance
    vectors: torch.Tensor           # [C, C] orthonormal eigenvectors
    patches: int                    # P: averaging P noise patches divides the noise variance by P

    def at(self, t):
        c = ((1 - t) / t) ** 2
        return Pooled(self.values / (1 + self.patches * c * self.values), self.vectors, self.patches)

    def iso(self):
        return Pooled(self.values.mean().expand_as(self.values), self.vectors, self.patches)

    def apply(self, u):
        """Covariance times a latent-shaped direction; only its spatial mean is affected."""
        mean = u.double().flatten(2).mean(2)
        out = self.patches * ((mean @ self.vectors) * self.values) @ self.vectors.T
        return out[:, :, None, None].expand_as(u)


@dataclass
class Latent:
    """Low rank plus a diagonal part: `residual` in every direction, except that one channel
    `direction` may carry its own variance `along` at each patch (a probe's own score variance)."""
    values: torch.Tensor            # [K] leading eigenvalues (total, not excess over the residual)
    vectors: torch.Tensor           # [D, K] orthonormal eigenvectors, D = C*H*W
    residual: float                 # isotropic variance of the remaining directions
    direction: torch.Tensor = None  # [C] optional unit channel direction
    along: torch.Tensor = None      # [P] variance along it at each patch, in place of the residual

    def projected(self):
        """[P, K]: the eigenvectors read along `direction` at every patch. Computed once."""
        if getattr(self, "_projected", None) is None:
            basis = self.vectors.double().reshape(len(self.direction), -1, self.vectors.shape[1])
            self._projected = torch.einsum("c,cpk->pk", self.direction, basis)
        return self._projected

    def at(self, t):
        """Posterior (S^-1 + c I)^-1 by the matrix inversion lemma: a K x K solve."""
        c = ((1 - t) / t) ** 2
        keep = 1 / (1 + c * self.residual)
        middle = torch.diag(1 / (self.values - self.residual)) + c * keep * torch.eye(len(self.values)).to(self.values)
        if self.direction is None:
            return LatentAt(self.vectors, torch.linalg.inv(middle), keep * self.residual, keep)
        keep_along, A = 1 / (1 + c * self.along), self.projected()
        middle = middle + c * A.T @ ((keep_along - keep)[:, None] * A)
        return LatentAt(self.vectors, torch.linalg.inv(middle), keep * self.residual, keep,
                        self.direction, keep_along * self.along, keep_along, A)

    def iso(self):
        """Isotropic covariance of equal total variance."""
        D, K = self.vectors.shape
        total = self.residual * D + float((self.values - self.residual).sum())
        if self.direction is not None:
            total += float((self.along - self.residual).sum())
        return Latent(self.values[:0], self.vectors[:, :0], total / D)

    def apply(self, u):
        return self.at(1.).apply(u)


@dataclass
class LatentAt:
    """A Latent covariance at one time: S = D + (B U) core (B U)^T.

    D and B are diagonal: `variance` and `keep` in every direction, `along` and `keep_along`
    along `direction` at each patch.
    """
    vectors: torch.Tensor           # [D, K] U
    core: torch.Tensor              # [K, K]
    variance: float
    keep: float
    direction: torch.Tensor = None  # [C]
    along: torch.Tensor = None      # [P]
    keep_along: torch.Tensor = None # [P]
    projected: torch.Tensor = None  # [P, K] U read along the direction

    def _diagonal(self, flat, everywhere, along):
        """D or B times latents [B, C, P]."""
        if self.direction is None:
            return everywhere * flat
        component = torch.einsum("c,bcp->bp", self.direction, flat)
        return everywhere * flat + ((along - everywhere) * component)[:, None, :] * self.direction[None, :, None]

    def apply(self, u):
        flat, vectors = u.double().flatten(2), self.vectors.double()
        low = self._diagonal(flat, self.keep, self.keep_along).flatten(1) @ vectors @ self.core @ vectors.T
        out = self._diagonal(flat, self.variance, self.along) + self._diagonal(low.reshape(flat.shape), self.keep, self.keep_along)
        return out.reshape(u.shape)
