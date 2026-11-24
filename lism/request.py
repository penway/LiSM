"""Request operators (A, c), built from a fitted probe (W, b).

Every request is the event "all scores A z + c are positive". Two operator types share
one interface: scores, linear (A z), adjoint (A^T lam), variance (diag A S A^T). A local
request may group rows that share a wanted patch (one joint against its rival patches).
"""
from dataclasses import dataclass
from math import log, prod
import numpy as np
import torch
from .covariance import LatentAt, Pooled


@dataclass
class GlobalRequest:
    """Affine scores of the spatially averaged latent."""
    rows: torch.Tensor              # [M, C]
    bias: torch.Tensor              # [M]
    shape: tuple                    # (C, H, W)

    @property
    def patches(self):
        return prod(self.shape[1:])

    def linear(self, z):
        return z.double().flatten(2).mean(2) @ self.rows.T

    def scores(self, z):
        return self.linear(z) + self.bias

    def adjoint(self, lam):
        return (lam @ self.rows / self.patches)[:, :, None, None].expand(-1, *self.shape)

    def variance(self, cov):
        if not isinstance(cov, Pooled):
            raise TypeError("A global request needs a Pooled covariance")
        return (self.rows @ cov.vectors).square() @ cov.values


@dataclass
class LocalRequest:
    """Affine scores at single patches, optionally minus the same row at a rival patch."""
    rows: torch.Tensor              # [M, C]
    bias: torch.Tensor              # [M]
    patch: torch.Tensor             # [M] flattened patch index
    shape: tuple                    # (C, H, W)
    rival: torch.Tensor = None      # [M] optional: score = row.(z[patch] - z[rival]) + bias
    group: torch.Tensor = None      # [M] optional: rows of one group share their row and wanted patch

    def __post_init__(self):
        as_ids = lambda ids: torch.as_tensor(ids, device=self.rows.device, dtype=torch.long)
        self.patch = as_ids(self.patch)
        self.rival = None if self.rival is None else as_ids(self.rival)
        ids = self.patch if self.rival is None else torch.cat([self.patch, self.rival])
        if self.rows.shape != (len(self.patch), self.shape[0]) or self.bias.shape != (len(self.patch),):
            raise ValueError("Each local score needs one channel row, one bias and one patch")
        if (ids < 0).any() or (ids >= prod(self.shape[1:])).any():
            raise ValueError("Patch index outside the latent grid")
        if self.group is not None:                                    # slots [J, R]: the rows of each group, padded
            self.group = as_ids(self.group)
            if self.rival is None or (self.patch == self.rival).any() or (self.group[1:] < self.group[:-1]).any():
                raise ValueError("Grouped rows need distinct rival patches and must be ordered by group")
            sizes = torch.bincount(self.group)
            first = torch.cumsum(sizes, 0) - sizes
            column = torch.arange(len(self.group), device=self.rows.device) - first[self.group]
            self._slots = first[:, None].repeat(1, int(sizes.max()))
            self._slots[self.group, column] = torch.arange(len(self.group), device=self.rows.device)
            self._filled = torch.arange(int(sizes.max()), device=self.rows.device)[None] < sizes[:, None]
            head = self._slots[:, :1]
            if not (torch.equal(self.rows[self._slots][self._filled], self.rows[head.expand_as(self._slots)][self._filled])
                    and torch.equal(self.patch[self._slots][self._filled], self.patch[head.expand_as(self._slots)][self._filled])):
                raise ValueError("Rows of one group must share their channel row and wanted patch")

    def linear(self, z):
        z = z.double().flatten(2)
        picked = z[:, :, self.patch]
        if self.rival is not None:
            picked = picked - z[:, :, self.rival]
        return torch.einsum("bcm,mc->bm", picked, self.rows)

    def scores(self, z):
        return self.linear(z) + self.bias

    def adjoint(self, lam):
        parts = lam[:, None, :] * self.rows.T[None]                   # [B, C, M]
        u = lam.new_zeros((len(lam), self.shape[0], prod(self.shape[1:])))
        u.index_add_(2, self.patch, parts)                            # rows sharing a patch add up
        if self.rival is not None:
            u.index_add_(2, self.rival, -parts)
        return u.reshape(-1, *self.shape)

    def _sides(self, cov):
        """Per row and side (wanted patch, rival patch): the diagonal variance term and the low-rank coordinates."""
        if not isinstance(cov, LatentAt):
            raise TypeError("A local request needs a Latent covariance at a time: cov.at(t)")
        sides = (self.patch,) if self.rival is None else (self.patch, self.rival)
        if getattr(self, "_basis", None) is not cov.vectors:          # time-independent part, computed once
            basis = cov.vectors.T.reshape(cov.vectors.shape[1], self.shape[0], prod(self.shape[1:]))    # [K, C, P]
            self._projected = [torch.einsum("kcm,mc->mk", basis[:, :, ids].to(self.rows), self.rows) for ids in sides]
            self._basis, self._norm2 = cov.vectors, self.rows.square().sum(1)
        diagonal, low = [cov.variance * self._norm2 for _ in sides], [cov.keep * p for p in self._projected]
        if cov.direction is not None:                                 # rows read along the calibrated direction
            component = self.rows @ cov.direction
            shrink = (cov.keep_along - cov.keep)[:, None] * cov.projected
            diagonal = [d + component.square() * (cov.along - cov.variance)[ids] for d, ids in zip(diagonal, sides)]
            low = [l + component[:, None] * shrink[ids] for l, ids in zip(low, sides)]
        return diagonal, low

    def variance(self, cov):
        diagonal, low = self._sides(cov)
        if self.rival is None:
            return diagonal[0] + ((low[0] @ cov.core) * low[0]).sum(1)
        difference = low[0] - low[1]
        return (diagonal[0] + diagonal[1]) * (self.patch != self.rival) + ((difference @ cov.core) * difference).sum(1)

    def shared(self, cov):
        """Moments of the grouped scores: per group the wanted score's variance [J]; per slot [J, R] the
        rival score's variance and its covariance with the wanted score; and the slots' rows and mask."""
        diagonal, low = self._sides(cov)
        wanted = diagonal[0] + ((low[0] @ cov.core) * low[0]).sum(1)
        rival = diagonal[1] + ((low[1] @ cov.core) * low[1]).sum(1)
        cross = ((low[0] @ cov.core) * low[1]).sum(1)                 # different patches share only the low rank
        return wanted[self._slots[:, 0]], rival[self._slots], cross[self._slots], self._slots, self._filled


# ---- builders: fitted probe (W, b) + what the user asks for -> (A, c) ----

def _double(x, device):
    return torch.as_tensor(np.asarray(x), device=device, dtype=torch.float64)


def class_request(w, b, target, shape, device="cuda"):
    """The target logit exceeds every rival logit. w: [classes, C], b: [classes]."""
    w, b = _double(w, device), _double(b, device)
    if not 0 <= target < len(b):
        raise ValueError("Target is outside the probe's class range")
    rivals = torch.arange(len(b), device=device) != target
    return GlobalRequest(w[target] - w[rivals], b[target] - b[rivals], tuple(shape))


def attribute_request(w, b, signs, shape, device="cuda"):
    """signs maps attribute index to +1 (present) or -1 (absent). w: [attributes, C]."""
    if not signs or any(s not in (-1, 1) for s in signs.values()):
        raise ValueError("Request at least one attribute with sign +1 or -1")
    ids = list(signs)
    sign = _double(list(signs.values()), device)
    return GlobalRequest(sign[:, None] * _double(w, device)[ids], sign * _double(b, device)[ids], tuple(shape))


def mask_request(w, b, mask, shape, threshold=.2, device="cuda"):
    """sigmoid(logit) > threshold inside the mask and < threshold outside. w: [C], rows in logit units."""
    mask = torch.as_tensor(np.asarray(mask), device=device)
    if mask.shape != tuple(shape[1:]) or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Pass a binary mask on the latent grid")
    if not 0 < threshold < 1:
        raise ValueError("Probability threshold must be inside (0, 1)")
    sign = torch.where(mask.flatten() > 0, 1., -1.).double()
    rows = sign[:, None] * _double(w, device).flatten()
    bias = sign * (float(np.asarray(b).reshape(-1)[0]) - log(threshold / (1 - threshold)))
    return LocalRequest(rows, bias, torch.arange(mask.numel(), device=device), tuple(shape))


def keypoint_request(w, points, shape, image_size=256, margin=1., shared=True, device="cuda"):
    """Each visible joint's heatmap at its target patch beats every patch outside the 3x3 neighbourhood.

    w: [joints, C]. points: [joints, 3] with x, y in image pixels and visibility (2 = visible).
    shared: Iso and Fit weighting treat a joint's rows as one event with a shared target score;
    otherwise every row is its own event.
    """
    points = np.asarray(points)
    if points.shape != (len(w), 3) or not np.isfinite(points).all() or not np.any(points[:, 2] == 2):
        raise ValueError("Provide one finite (x, y, visibility) row per joint, at least one visible")
    height, width = shape[1:]
    cell = image_size / width
    cells = np.rint(points[:, :2] / cell - .5).astype(int)
    cells = np.stack([cells[:, 0].clip(0, width - 1), cells[:, 1].clip(0, height - 1)], 1)
    joints, wanted, rivals = [], [], []
    for joint in np.flatnonzero(points[:, 2] == 2):
        x, y = cells[joint]
        for rival in range(height * width):
            if abs(rival % width - x) > 1 or abs(rival // width - y) > 1:
                joints.append(joint); wanted.append(y * width + x); rivals.append(rival)
    if not joints:
        raise ValueError("The latent grid has no patch outside a joint's 3x3 neighbourhood")
    rows = _double(np.asarray(w)[joints], device)
    bias = torch.full((len(joints),), -float(margin), device=device, dtype=torch.float64)
    group = np.unique(joints, return_inverse=True)[1] if shared else None
    return LocalRequest(rows, bias, wanted, tuple(shape), rivals, group)


# ---- requests written as text ----

def mask_from_text(text, inside="#"):
    """A mask drawn as rows of characters, one per patch: `inside` marks the object, anything else the background."""
    rows = text.split()
    if not rows or len(set(map(len, rows))) != 1:
        raise ValueError("Every row of the mask needs the same number of characters")
    return np.array([[c == inside for c in row] for row in rows])


def pose_from_text(text):
    """Lines `joint x y` in image pixels -> {joint: (x, y)}; joints left out are not requested."""
    return {name: (float(x), float(y)) for name, x, y in map(str.split, text.strip().splitlines())}


def points_from_names(pose, labels):
    """{joint name: (x, y)} in image pixels -> [joints, 3] in the probe's joint order; other joints are not requested."""
    unknown = set(pose) - set(labels)
    if unknown:
        raise KeyError(f"Unknown joints {sorted(unknown)}; the probe has {list(labels)}")
    return np.array([[*pose[name], 2.] if name in pose else [0., 0., 0.] for name in labels])
