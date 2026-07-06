"""Ground-truth room environment with region-dependent spatial resolution.

This module only produces ground truth: a rasterized "pixelated" grid of
semantic labels (e.g. "room", "donut"), and objects/regions that carry those
labels. It does not perform any SSP/VSA encoding — that is the job of a
downstream encoder module that consumes ``RoomEnv``.

Spatial *scale* (coarse/medium/fine) is never stored directly on a region or
object. It is derived from a semantic label via ``RoomEnv.label_to_scale``,
so the same room layout can be re-purposed under a different resolution
policy (e.g. "donut is coarse this time") without changing the ground truth.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

Bounds = Tuple[Tuple[float, float], Tuple[float, float]]

# Reference length scales associated with each scale name. Downstream SSP
# encoders may use these as defaults; RoomEnv itself only uses the names.
DEFAULT_LENGTH_SCALES: Dict[str, float] = {
    "coarse": 0.4,
    "medium": 0.15,
    "fine": 0.05,
}

# Default policy mapping a semantic label to the scale it should be encoded
# at. This is deliberately separate from the ground-truth labels themselves.
DEFAULT_LABEL_SCALES: Dict[str, str] = {
    "room": "coarse",
    "donut": "fine",
    "floor": "coarse",
    "wall": "fine",
}


@dataclass
class Region:
    """A labeled area of the room.

    shape: "rect" or "circle".
    params: for "rect" -> {"xmin", "xmax", "ymin", "ymax"};
            for "circle" -> {"cx", "cy", "radius"}.
    label: semantic identity of the region, e.g. "room". Scale is derived
           from this via ``RoomEnv.label_to_scale``, not stored here.
    priority: when regions overlap, the highest-priority region wins.
    """

    shape: str
    params: Dict[str, float]
    label: str
    priority: int = 0

    def contains(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if self.shape == "rect":
            p = self.params
            return (
                (x >= p["xmin"]) & (x <= p["xmax"])
                & (y >= p["ymin"]) & (y <= p["ymax"])
            )
        if self.shape == "circle":
            p = self.params
            return (x - p["cx"]) ** 2 + (y - p["cy"]) ** 2 <= p["radius"] ** 2
        raise ValueError(f"Unknown region shape: {self.shape!r}")


@dataclass
class RoomObject:
    """A point object (e.g. a donut) that locally overrides the region label.

    ``name`` is both the object's identity and its semantic label (e.g.
    "donut"); scale is derived from it via ``RoomEnv.label_to_scale``, not
    stored here.

    ``density``, if set, overrides how densely ``RoomEnv.dense_positions()``
    samples this object's patch: a ``density x density`` grid over its
    bounding box (clipped to the circular patch), independent of the room's
    ``grid_size``. Leave as ``None`` to just take whatever base grid cells
    happen to fall inside the patch (the previous, coarser behavior).
    """

    name: str
    position: Tuple[float, float]
    patch_radius: float = 0.15
    density: Optional[int] = None


class RoomEnv:
    """A room with region- and object-dependent spatial resolution.

    Coordinates are continuous over ``bounds`` (default the standard SSP
    domain [-1, 1] x [-1, 1]). ``grid_size`` controls only the resolution of
    the rasterized "pixelated" view used for visualization / ground-truth
    lookup tables; label/scale queries at arbitrary continuous points do not
    depend on it.

    ``label_to_scale`` is the policy that derives a coarse/medium/fine scale
    from a region's/object's semantic label. It is stored on the env (not on
    the regions/objects) so the same layout can be re-scaled by swapping the
    policy.
    """

    def __init__(
        self,
        bounds: Bounds = ((-1.0, 1.0), (-1.0, 1.0)),
        grid_size: Tuple[int, int] = (64, 64),
        regions: Optional[List[Region]] = None,
        objects: Optional[List[RoomObject]] = None,
        default_label: str = "room",
        label_to_scale: Optional[Dict[str, str]] = None,
    ):
        self.bounds = bounds
        self.grid_size = grid_size
        self.regions = list(regions) if regions else []
        self.objects = list(objects) if objects else []
        self.default_label = default_label
        self.label_to_scale = dict(label_to_scale) if label_to_scale else dict(DEFAULT_LABEL_SCALES)
        self._grid = None

    @property
    def grid(self) -> np.ndarray:
        """Rasterized grid of semantic labels (ground truth, not scale)."""
        if self._grid is None:
            self._grid = self._rasterize()
        return self._grid

    def scale_of(self, label: str) -> str:
        """Derive the scale for a semantic label via ``label_to_scale``."""
        return self.label_to_scale[label]

    def cell_to_coord(self, i: int, j: int) -> Tuple[float, float]:
        (xmin, xmax), (ymin, ymax) = self.bounds
        H, W = self.grid_size
        x = xmin + (j + 0.5) / W * (xmax - xmin)
        y = ymax - (i + 0.5) / H * (ymax - ymin)
        return x, y

    def coord_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        (xmin, xmax), (ymin, ymax) = self.bounds
        H, W = self.grid_size
        j = int(np.clip((x - xmin) / (xmax - xmin) * W, 0, W - 1))
        i = int(np.clip((ymax - y) / (ymax - ymin) * H, 0, H - 1))
        return i, j

    def label_at(self, x: float, y: float) -> str:
        """Return the ground-truth semantic label at a point (objects win over regions)."""
        for obj in self.objects:
            ox, oy = obj.position
            if (x - ox) ** 2 + (y - oy) ** 2 <= obj.patch_radius ** 2:
                return obj.name
        for region in sorted(self.regions, key=lambda r: r.priority, reverse=True):
            if region.contains(x, y):
                return region.label
        return self.default_label

    def scale_at(self, x: float, y: float) -> str:
        """Return the scale at a point, derived from its semantic label."""
        return self.scale_of(self.label_at(x, y))

    def sample_position(
        self,
        scale: Optional[str] = None,
        label: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
        max_tries: int = 10000,
    ) -> Tuple[float, float]:
        """Sample a uniformly random point, optionally restricted to a given scale and/or label."""
        rng = rng or np.random.default_rng()
        (xmin, xmax), (ymin, ymax) = self.bounds
        for _ in range(max_tries):
            x = rng.uniform(xmin, xmax)
            y = rng.uniform(ymin, ymax)
            if label is not None and self.label_at(x, y) != label:
                continue
            if scale is not None and self.scale_at(x, y) != scale:
                continue
            return x, y
        raise RuntimeError(
            f"Could not sample a position with scale={scale!r}, label={label!r} after {max_tries} tries"
        )

    def dense_positions(self) -> List[Tuple[Tuple[float, float], str]]:
        """Every grid cell center and its semantic label, at full grid density.

        This is uniform coverage of the whole room at ``grid_size``
        resolution, labeled with ground truth (e.g. "room", "donut") rather
        than a scale. Downstream consumers derive scale via ``scale_of()``
        and filter/subsample per scale to get whatever density they want.

        Objects with an explicit ``density`` override are excluded from the
        base grid enumeration and resampled on their own ``density x density``
        grid instead (see ``RoomObject.density``).
        """
        H, W = self.grid_size
        grid = self.grid
        overridden = {obj.name: obj for obj in self.objects if obj.density is not None}

        positions = [
            (self.cell_to_coord(i, j), grid[i, j])
            for i in range(H)
            for j in range(W)
            if grid[i, j] not in overridden
        ]
        for obj in overridden.values():
            positions.extend(self._dense_object_positions(obj))
        return positions

    def _dense_object_positions(self, obj: RoomObject) -> List[Tuple[Tuple[float, float], str]]:
        ox, oy = obj.position
        r = obj.patch_radius
        n = obj.density
        xs = ox - r + (np.arange(n) + 0.5) / n * (2 * r)
        ys = oy - r + (np.arange(n) + 0.5) / n * (2 * r)
        xx, yy = np.meshgrid(xs, ys)
        mask = (xx - ox) ** 2 + (yy - oy) ** 2 <= r ** 2
        return [((float(x), float(y)), obj.name) for x, y in zip(xx[mask], yy[mask])]

    def _rasterize(self) -> np.ndarray:
        (xmin, xmax), (ymin, ymax) = self.bounds
        H, W = self.grid_size
        xs = xmin + (np.arange(W) + 0.5) / W * (xmax - xmin)
        ys = ymax - (np.arange(H) + 0.5) / H * (ymax - ymin)
        xx, yy = np.meshgrid(xs, ys)

        grid = np.full((H, W), self.default_label, dtype=object)
        for region in sorted(self.regions, key=lambda r: r.priority):
            grid[region.contains(xx, yy)] = region.label
        for obj in self.objects:
            ox, oy = obj.position
            mask = (xx - ox) ** 2 + (yy - oy) ** 2 <= obj.patch_radius ** 2
            grid[mask] = obj.name
        return grid

    def render(self, ax=None, show_objects: bool = True):
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        if ax is None:
            _, ax = plt.subplots()

        scale_order = list(DEFAULT_LENGTH_SCALES.keys())
        scale_to_idx = {s: i for i, s in enumerate(scale_order)}
        label_to_idx = {
            label: scale_to_idx[self.scale_of(label)] for label in np.unique(self.grid)
        }
        idx_grid = np.vectorize(label_to_idx.get)(self.grid).astype(float)

        (xmin, xmax), (ymin, ymax) = self.bounds
        cmap = ListedColormap(["#d9d9d9", "#6baed6", "#08306b"])
        ax.imshow(
            idx_grid,
            extent=(xmin, xmax, ymin, ymax),
            origin="upper",
            cmap=cmap,
            vmin=0,
            vmax=len(scale_order) - 1,
        )

        if show_objects:
            for obj in self.objects:
                ox, oy = obj.position
                ax.plot(ox, oy, marker="o", markersize=10, color="orange", markeredgecolor="black")
                ax.annotate(obj.name, (ox, oy), textcoords="offset points", xytext=(6, 6))

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title("Room: spatial resolution regions")
        return ax


def _rotate_cells_cw(cells: List[Tuple[int, int]], n: int) -> List[Tuple[int, int]]:
    """Rotate (row, col) grid cells 90 deg clockwise within an n x n bounding box."""
    return [(c, n - 1 - r) for r, c in cells]


def _l_shaped_alcove_regions(
    origin: Tuple[float, float] = (-0.95, 0.95),
    voxel_size: float = 0.35,
    wall_thickness: float = 0.06,
    rotation_deg: int = 0,
) -> List[Region]:
    """Floor + wall regions for a 3-voxel L-shaped alcove, top-left corner at ``origin``.

    Voxel occupancy (row 0 = top, col 0 = left) before rotation:
        X .
        X X
    ``rotation_deg`` rotates this footprint clockwise within its 2x2
    bounding box (multiples of 90). Walls are only placed on edges exterior
    to the L (shared edges between adjacent voxels are left open so the
    three voxels read as one room).
    """
    cells = [(0, 0), (1, 0), (1, 1)]
    for _ in range((rotation_deg // 90) % 4):
        cells = _rotate_cells_cw(cells, n=2)
    occupied = set(cells)
    x0, y0 = origin
    s, w = voxel_size, wall_thickness

    def voxel_bounds(r, c):
        xmin = x0 + c * s
        xmax = xmin + s
        ymax = y0 - r * s
        ymin = ymax - s
        return xmin, xmax, ymin, ymax

    regions = []
    for r, c in cells:
        xmin, xmax, ymin, ymax = voxel_bounds(r, c)
        regions.append(
            Region(
                shape="rect",
                params={"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
                label="floor",
                priority=1,
            )
        )

        # (edge name, neighbor cell, wall strip bounds)
        edges = {
            "top": ((r - 1, c), {"xmin": xmin, "xmax": xmax, "ymin": ymax - w, "ymax": ymax}),
            "bottom": ((r + 1, c), {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymin + w}),
            "left": ((r, c - 1), {"xmin": xmin, "xmax": xmin + w, "ymin": ymin, "ymax": ymax}),
            "right": ((r, c + 1), {"xmin": xmax - w, "xmax": xmax, "ymin": ymin, "ymax": ymax}),
        }
        for _, (neighbor, wall_params) in edges.items():
            if neighbor in occupied:
                continue  # shared edge between two voxels: leave it open
            regions.append(
                Region(shape="rect", params=wall_params, label="wall", priority=2)
            )

    return regions


def make_default_room() -> RoomEnv:
    """The Demo-1 setup: an open room with a donut, plus an L-shaped alcove."""
    regions = [
        Region(
            shape="rect",
            params={"xmin": -1.0, "xmax": 1.0, "ymin": -1.0, "ymax": 1.0},
            label="room",
            priority=0,
        ),
        *_l_shaped_alcove_regions(rotation_deg=90),
    ]
    objects = [
        RoomObject(name="donut", position=(0.3, -0.2), patch_radius=0.08, density=10),
    ]
    return RoomEnv(regions=regions, objects=objects)
