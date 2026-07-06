"""Baseline VSA spatial memory: bind position and class, superpose into one memory vector.

Every labeled point (x, y, class) is encoded as
``bind(ssp_space.encode([x, y]), sp_space.encode(class))`` and all such
records are bundled (summed) into a single memory vector. A query point
unbinds its own position back out of that memory, leaving a noisy class
vector that is scored against every class vector to get a probability
distribution over classes at that point.

This uses one fixed length-scale for every point/class - it is deliberately
the simplest version of this idea, meant as a point of comparison for
fancier models that vary encoding resolution per label/region (see
``multiscalessps.envs.room.RoomEnv.label_to_scale``).
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np

from ..metrics import kl_divergence
from ..ssps.spspace import SPSpace
from ..ssps.sspspace import RandomSSPSpace, SSPSpace, _get_rng


class VSASpatialMemory:
    """Stores (position, class) pairs in a single VSA memory vector.

    Parameters
    ----------
    ssp_space : SSPSpace
        Encodes continuous (x, y) positions into SSP vectors. A single,
        fixed length-scale is used for every point regardless of class.
    sp_space : SPSpace
        Encodes discrete class labels into SP vectors. ``sp_space.names``
        gives the label vocabulary.
    """

    def __init__(self, ssp_space: SSPSpace, sp_space: SPSpace):
        assert ssp_space.ssp_dim == sp_space.dim, (
            f"ssp_space.ssp_dim ({ssp_space.ssp_dim}) must match sp_space.dim ({sp_space.dim})"
        )
        self.ssp_space = ssp_space
        self.sp_space = sp_space
        self.memory = np.zeros((1, ssp_space.ssp_dim))

    def fit(
        self,
        points: np.ndarray,
        labels: Sequence[str],
        normalize_by_class: bool = False,
    ) -> "VSASpatialMemory":
        """Encode and store labeled points, replacing any existing memory.

        Parameters
        ----------
        points : np.ndarray
            (N, 2) array of (x, y) positions.
        labels : sequence of str
            Length-N sequence of class names, each present in ``sp_space.names``.
        normalize_by_class : bool
            If False (default), every point's record contributes equal energy
            to the memory, so a class with more points (e.g. "room" covering
            most of the grid) dominates the bundle. If True, records are
            summed per class first and each class's sum is normalized to unit
            norm before being added to the memory, so every class contributes
            equally regardless of how many points it had.
        """
        points = np.atleast_2d(points)
        class_idx = np.array([self.sp_space.name_to_idx[label] for label in labels])

        pos_ssps = self.ssp_space.encode(points)
        class_sps = self.sp_space.encode(class_idx)
        records = self.ssp_space.bind(pos_ssps, class_sps)

        if normalize_by_class:
            memory = np.zeros((1, self.ssp_space.ssp_dim))
            for idx in np.unique(class_idx):
                class_sum = records[class_idx == idx].sum(axis=0, keepdims=True)
                memory += self.ssp_space.normalize(class_sum)
            self.memory = memory
        else:
            self.memory = records.sum(axis=0, keepdims=True)
        return self

    def query(self, points: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Return a (N, n_classes) array of class probabilities at each point.

        Unbinds position from the stored memory and scores the residual
        against every class vector with a softmax, so each row sums to 1.
        Column order follows ``sp_space.names``.
        """
        points = np.atleast_2d(points)
        pos_ssps = self.ssp_space.encode(points)
        inv_pos = self.ssp_space.invert(pos_ssps)
        residual = self.ssp_space.bind(self.memory, inv_pos)

        sims = (residual @ self.sp_space.vectors.T) / temperature
        sims -= sims.max(axis=1, keepdims=True)
        exp_sims = np.exp(sims)
        return exp_sims / exp_sims.sum(axis=1, keepdims=True)

    def predict(self, points: np.ndarray, temperature: float = 1.0):
        """Return (predicted_label, class_probabilities) for each point."""
        probs = self.query(points, temperature=temperature)
        pred_idx = np.argmax(probs, axis=1)
        labels = np.array(self.sp_space.names)[pred_idx]
        return labels, probs

    def class_probability_maps(
        self, room, temperature: float = 1.0, eps: float = 1e-12
    ) -> Dict[str, np.ndarray]:
        """Estimated per-label spatial probability map over ``room``'s base grid.

        Mirrors ``RoomEnv.label_probability_maps()``: for each label this is a
        (H, W) array normalized to sum to 1, suitable for KL-divergence
        comparison against ground truth. Smoothed by ``eps`` before
        normalizing, since a class that a saturated softmax never favors
        anywhere would otherwise be exactly zero everywhere and undefined
        to normalize.
        """
        H, W = room.grid_size
        coords = np.array([room.cell_to_coord(i, j) for i in range(H) for j in range(W)])
        probs = self.query(coords, temperature=temperature)

        maps = {}
        for name, idx in self.sp_space.name_to_idx.items():
            m = probs[:, idx].reshape(H, W) + eps
            maps[name] = m / m.sum()
        return maps

    def evaluate_kl(self, room, temperature: float = 1.0) -> Dict[str, float]:
        """KL(estimated label map || ground-truth label map) for each shared label."""
        est_maps = self.class_probability_maps(room, temperature=temperature)
        gt_maps = room.label_probability_maps()
        return {
            label: kl_divergence(est_maps[label], gt_maps[label])
            for label in gt_maps
            if label in est_maps
        }

    @classmethod
    def from_room(
        cls,
        room,
        ssp_dim: int = 257,
        length_scale: float = 0.3,
        normalize_by_class: bool = False,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> "VSASpatialMemory":
        """Build and fit a baseline VSA memory directly from a RoomEnv's dense positions."""
        rng = _get_rng(rng)
        positions = room.dense_positions()
        xy = np.array([p for p, _ in positions])
        labels = [label for _, label in positions]

        ssp_space = RandomSSPSpace(
            domain_dim=2,
            ssp_dim=ssp_dim,
            domain_bounds=np.array(room.bounds),
            length_scale=length_scale,
            rng=rng,
        )
        label_names = sorted(set(labels))
        sp_space = SPSpace(
            domain_size=len(label_names),
            dim=ssp_space.ssp_dim,
            names=label_names,
            rng=rng,
        )

        return cls(ssp_space, sp_space).fit(xy, labels, normalize_by_class=normalize_by_class)
