"""Information-theoretic comparisons over spatial probability maps.

These operate on plain probability arrays (e.g. from
``RoomEnv.label_probability_maps()``) and have no dependency on the room
module, so they can be reused for any ground-truth or retrieved spatial
distribution.
"""

from typing import Dict

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(P || Q) between two probability arrays of the same shape.

    Both distributions are smoothed by ``eps`` before comparing, so that
    cells where one map is exactly zero don't produce -inf/undefined terms;
    this trades a small amount of bias for an always-finite, comparable
    KL value.
    """
    p = p.astype(float).ravel()
    q = q.astype(float).ravel()
    p = (p + eps) / (p + eps).sum()
    q = (q + eps) / (q + eps).sum()
    return float(np.sum(p * np.log(p / q)))


def kl_from_uniform(label_maps: Dict[str, np.ndarray], eps: float = 1e-12) -> Dict[str, float]:
    """KL(P_label || Uniform over the whole spatial region) for each label.

    Quantifies how concentrated each label's spatial footprint is relative to
    assuming it could be anywhere with equal probability: a label confined to
    a small area (e.g. "donut") has much higher KL than one spread across
    most of the room (e.g. "room").
    """
    shape = next(iter(label_maps.values())).shape
    uniform = np.full(shape, 1.0 / np.prod(shape))
    return {label: kl_divergence(p, uniform, eps) for label, p in label_maps.items()}


def mix_uniform_noise(p: np.ndarray, noise_level: float) -> np.ndarray:
    """Blend a probability map with a uniform "could be anywhere" component.

    ``noise_level`` in [0, 1] is the fraction of probability mass reassigned
    uniformly over the whole map - a stand-in for random false-positive mass
    (e.g. as if some fraction of retrieved samples were pure noise). At
    ``noise_level=0`` this returns ``p`` unchanged; at ``1`` it returns the
    uniform distribution.
    """
    p = p.astype(float)
    p = p / p.sum()
    uniform = np.full_like(p, 1.0 / p.size)
    return (1 - noise_level) * p + noise_level * uniform


def blur_map(p: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-blur a 2D probability map with a separable kernel, renormalized to sum to 1.

    A stand-in for the spatial generalization a coarser SSP length scale
    would introduce: larger ``sigma`` (in grid cells) spreads mass away from
    the true footprint, similar to querying memory at a longer length scale.
    """
    if sigma <= 0:
        return p / p.sum()
    radius = max(1, int(3 * sigma))
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=p)
    blurred = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=blurred)
    return blurred / blurred.sum()
