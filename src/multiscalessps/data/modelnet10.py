"""PyTorch dataset for ModelNet10.

ModelNet10 is a collection of CAD models stored as ASCII ``.off`` meshes,
organized on disk as::

    ModelNet10/<category>/<split>/<category>_<id>.off

where ``<split>`` is either ``train`` or ``test`` and there are 10 object
categories (bathtub, bed, chair, ...).

The dataset parses each mesh and, by default, samples a fixed-size point cloud
from the mesh surface so that every sample has the same shape and can be
collated into a batch. Set ``num_points=None`` to instead return the raw
(variable-length) vertices and faces.

Download: http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_off(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an ASCII OFF mesh file.

    Returns
    -------
    vertices : (V, 3) float32 array
    faces : (F, 3) int64 array of vertex indices (triangulated)
    """
    path = Path(path)
    with open(path, "r") as f:
        first = f.readline().strip()
        # Some ModelNet files glue the header word and the counts together,
        # e.g. "OFF6 8 12" instead of a clean "OFF\n6 8 12".
        if first == "OFF":
            counts_line = f.readline().strip()
        elif first.startswith("OFF"):
            counts_line = first[len("OFF"):].strip()
        else:
            raise ValueError(f"Not a valid OFF file (missing 'OFF' header): {path}")

        n_verts, n_faces, _n_edges = (int(x) for x in counts_line.split())

        vertices = np.empty((n_verts, 3), dtype=np.float32)
        for i in range(n_verts):
            vertices[i] = [float(x) for x in f.readline().split()[:3]]

        faces = []
        for _ in range(n_faces):
            parts = [int(x) for x in f.readline().split()]
            k, idx = parts[0], parts[1:]
            # Fan-triangulate any polygon into triangles.
            for j in range(1, k - 1):
                faces.append((idx[0], idx[j], idx[j + 1]))

    return vertices, np.asarray(faces, dtype=np.int64)


def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample ``num_points`` points from the surface of a triangle mesh.

    Faces are chosen with probability proportional to their area, then a point
    is drawn uniformly within each chosen triangle via barycentric coordinates.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total = areas.sum()
    if total <= 0:
        # Degenerate mesh: fall back to sampling vertices directly.
        choice = rng.integers(0, len(vertices), size=num_points)
        return vertices[choice]

    probs = areas / total
    face_idx = rng.choice(len(faces), size=num_points, p=probs)

    u = rng.random((num_points, 1))
    w = rng.random((num_points, 1))
    over = (u + w) > 1.0
    u[over] = 1.0 - u[over]
    w[over] = 1.0 - w[over]

    a = v0[face_idx]
    b = v1[face_idx]
    c = v2[face_idx]
    return a + u * (b - a) + w * (c - a)


def _unit_sphere(points: np.ndarray) -> np.ndarray:
    """Center a point set at the origin and scale it to fit the unit sphere."""
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points


class ModelNet10Dataset(Dataset):
    """ModelNet10 as a PyTorch ``Dataset``.

    Parameters
    ----------
    root : str or Path
        Path to the extracted ``ModelNet10`` directory (the folder that
        directly contains the per-category subfolders).
    split : {"train", "test"}
        Which split to load.
    num_points : int or None
        If an int, sample this many points from each mesh surface so every
        item is a fixed ``(num_points, 3)`` tensor. If ``None``, return the raw
        vertices and faces (variable length; not directly batchable).
    normalize : bool
        If True, center each point cloud at the origin and scale it to fit in
        the unit sphere. Only applied when ``num_points`` is not None.
    transform : callable, optional
        Optional callable applied to the points tensor before returning.
    seed : int
        Base seed for the per-sample surface-sampling RNG (kept deterministic
        so a given index always yields the same point cloud).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        num_points: int | None = 1024,
        normalize: bool = True,
        transform=None,
        seed: int = 0,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"ModelNet10 root not found: {self.root}. Download and unzip "
                "ModelNet10.zip into the data folder first."
            )

        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.transform = transform
        self.seed = seed

        # Categories are the sorted subdirectories that contain a split folder.
        self.classes = sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and (p / split).is_dir()
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            for off in sorted((self.root / cls / split).glob("*.off")):
                self.samples.append((off, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(
                f"No .off files found under {self.root} for split '{split}'."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        vertices, faces = parse_off(path)

        if self.num_points is None:
            return {
                "vertices": torch.from_numpy(vertices),
                "faces": torch.from_numpy(faces),
                "label": label,
                "class_name": self.classes[label],
                "path": str(path),
            }

        rng = np.random.default_rng(self.seed + idx)
        points = sample_surface(vertices, faces, self.num_points, rng)

        if self.normalize:
            points = _unit_sphere(points)

        points = torch.from_numpy(points.astype(np.float32))
        if self.transform is not None:
            points = self.transform(points)

        return points, label


class ModelNet10PairDataset(Dataset):
    """ModelNet10 for encoding-vs-geometry comparison.

    Each item pairs a **down-sampled sparse point cloud** (the encoder input)
    with the **full dense point cloud** (the reference "true" geometry), so you
    can score an encoding built from the sparse points against the full cloud
    (e.g. Chamfer distance, or an SSP density-field comparison).

    The sparse input and the full cloud are **independently sampled** from the
    mesh surface, so the input is not a memorizable subset of the target -- both
    are fresh draws of the same underlying geometry. They still share one
    coordinate frame: normalization parameters are computed from the full cloud
    and applied to both.

    Parameters
    ----------
    root, split
        As in :class:`ModelNet10Dataset`.
    categories : list[str] or None
        Restrict to these category names (e.g. ``["chair"]``). ``None`` = all.
    num_full_points : int
        Size of the dense reference cloud (target geometry).
    num_input_points : int
        Size of the down-sampled input cloud.
    normalize : bool
        Normalize into the unit sphere (frame set by the full cloud).
    seed : int or None
        If ``None`` (default), each access draws fresh random points, so the
        clouds vary across epochs. If an int, sampling is deterministic per
        index (reproducible) using ``seed + idx``.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        categories: list[str] | None = None,
        num_full_points: int = 2048,
        num_input_points: int = 256,
        normalize: bool = True,
        seed: int | None = None,
    ):
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"ModelNet10 root not found: {self.root}. Download and unzip "
                "ModelNet10.zip into the data folder first."
            )

        self.split = split
        self.num_full_points = num_full_points
        self.num_input_points = num_input_points
        self.normalize = normalize
        self.seed = seed

        available = sorted(
            p.name
            for p in self.root.iterdir()
            if p.is_dir() and (p / split).is_dir()
        )
        if categories is None:
            self.classes = available
        else:
            missing = [c for c in categories if c not in available]
            if missing:
                raise ValueError(
                    f"Unknown categories {missing}; available: {available}"
                )
            self.classes = sorted(categories)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples: list[tuple[Path, int]] = []
        for cls in self.classes:
            for off in sorted((self.root / cls / split).glob("*.off")):
                self.samples.append((off, self.class_to_idx[cls]))

        if not self.samples:
            raise RuntimeError(
                f"No .off files found under {self.root} for split '{split}' "
                f"and categories {self.classes}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        vertices, faces = parse_off(path)

        seed = None if self.seed is None else self.seed + idx
        rng = np.random.default_rng(seed)

        # Independent surface draws for the target and the down-sampled input.
        full = sample_surface(vertices, faces, self.num_full_points, rng)
        sparse = sample_surface(vertices, faces, self.num_input_points, rng)

        if self.normalize:
            # Shared frame: normalization params come from the full cloud.
            center = full.mean(axis=0, keepdims=True)
            full = full - center
            scale = np.linalg.norm(full, axis=1).max()
            if scale > 0:
                full = full / scale
                sparse = (sparse - center) / scale
            else:
                sparse = sparse - center

        return {
            "points": torch.from_numpy(sparse.astype(np.float32)),
            "full_points": torch.from_numpy(full.astype(np.float32)),
            "label": label,
            "class_name": self.classes[label],
            "path": str(path),
        }
