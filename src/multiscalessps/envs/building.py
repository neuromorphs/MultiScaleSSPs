"""Ground-truth hierarchical building environment: rooms containing furniture.

Unlike ``RoomEnv`` (``envs/room.py``), where every point carries exactly one
flat label (its scale derived separately via ``label_to_scale``), points
here carry a *nested* label: a point on a piece of furniture is
simultaneously inside a room (coarser context) and on that furniture (finer
detail) -- the room and furniture labels aren't mutually-exclusive
alternatives, they co-occur at different resolutions for the same point.
This is the property a single shared VSA memory can actually exploit (see
the "Part 4" mixture-learning results in ``baseline_comparison.ipynb``:
multi-scale mixing only did real work when different scales had to compete
for the same memory, not when scales lined up with separate, non-competing
classes).

This module only produces ground truth: rasterized label grids and a
``label_at`` query. It does not perform any SSP/VSA encoding.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

Bounds = Tuple[Tuple[float, float], Tuple[float, float]]

# Reference length scales for each hierarchy level. Downstream SSP encoders
# may use these as defaults; BuildingEnv itself only uses the level names.
DEFAULT_LEVEL_LENGTH_SCALES: Dict[str, float] = {
    "room": 0.4,
    "furniture": 0.1,
}


@dataclass
class Room:
    """A named rectangular room. ``bounds`` is the room's floor area,
    already excluding the surrounding wall margin."""

    name: str
    bounds: Dict[str, float]  # {"xmin", "xmax", "ymin", "ymax"}

    def contains(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        b = self.bounds
        return (x >= b["xmin"]) & (x <= b["xmax"]) & (y >= b["ymin"]) & (y <= b["ymax"])


@dataclass
class Furniture:
    """A named rectangular piece of furniture, nested inside a room."""

    name: str
    room: str
    bounds: Dict[str, float]

    def contains(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        b = self.bounds
        return (x >= b["xmin"]) & (x <= b["xmax"]) & (y >= b["ymin"]) & (y <= b["ymax"])


class BuildingEnv:
    """A building subdivided into rooms, each optionally containing furniture.

    Every point has a *nested* ground-truth label: which room it's in (or
    "wall" if it's in an inter-room/exterior wall), and, if applicable,
    which piece of furniture it's on. A point on furniture is always also
    inside that furniture's room -- querying furniture never means "instead
    of" a room, only "in addition to, at finer resolution."

    Coordinates are continuous over ``bounds``. ``grid_size`` controls the
    resolution of the rasterized grids used for visualization / ground-truth
    lookup tables.
    """

    def __init__(
        self,
        bounds: Bounds = ((-1.0, 1.0), (-1.0, 1.0)),
        grid_size: Tuple[int, int] = (128, 128),
        rooms: Optional[List[Room]] = None,
        furniture: Optional[List[Furniture]] = None,
    ):
        self.bounds = bounds
        self.grid_size = grid_size
        self.rooms = list(rooms) if rooms else []
        self.furniture = list(furniture) if furniture else []
        self.room_names = [r.name for r in self.rooms]
        self.furniture_names = [f.name for f in self.furniture]
        self._room_grid = None
        self._furniture_grid = None

    @property
    def room_grid(self) -> np.ndarray:
        """Rasterized grid of room names (or "wall")."""
        if self._room_grid is None:
            self._rasterize()
        return self._room_grid

    @property
    def furniture_grid(self) -> np.ndarray:
        """Rasterized grid of furniture names (or "" where there's no furniture)."""
        if self._furniture_grid is None:
            self._rasterize()
        return self._furniture_grid

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

    def room_at(self, x: float, y: float) -> str:
        """Return the room name at (x, y), or "wall" if in an inter-room/exterior wall."""
        for room in self.rooms:
            if room.bounds["xmin"] <= x <= room.bounds["xmax"] and room.bounds["ymin"] <= y <= room.bounds["ymax"]:
                return room.name
        return "wall"

    def furniture_at(self, x: float, y: float) -> Optional[str]:
        """Return the furniture name at (x, y), or None if there's no furniture there."""
        for item in self.furniture:
            if item.bounds["xmin"] <= x <= item.bounds["xmax"] and item.bounds["ymin"] <= y <= item.bounds["ymax"]:
                return item.name
        return None

    def label_at(self, x: float, y: float) -> Dict[str, Optional[str]]:
        """Full nested ground-truth label at a point: {"room": ..., "furniture": ...}.

        "room" is "wall" if the point is in a wall (furniture is always None
        there). Otherwise "room" is the room name, and "furniture" is the
        furniture name if the point falls on a piece of furniture, else None
        (open floor).
        """
        room = self.room_at(x, y)
        furniture = self.furniture_at(x, y) if room != "wall" else None
        return {"room": room, "furniture": furniture}

    def dense_positions(self) -> List[Tuple[Tuple[float, float], Dict[str, Optional[str]]]]:
        """Every grid cell center and its full nested label, at full grid density."""
        H, W = self.grid_size
        room_grid = self.room_grid
        furniture_grid = self.furniture_grid
        positions = []
        for i in range(H):
            for j in range(W):
                furniture = furniture_grid[i, j] or None
                positions.append((self.cell_to_coord(i, j), {"room": room_grid[i, j], "furniture": furniture}))
        return positions

    def room_probability_maps(self) -> Dict[str, np.ndarray]:
        """Ground-truth spatial probability map per room name (including "wall"),
        over the base grid. Each map sums to 1, suitable for KL-divergence comparisons."""
        grid = self.room_grid
        maps = {}
        for name in np.unique(grid):
            mask = grid == name
            maps[str(name)] = mask.astype(float) / mask.sum()
        return maps

    def furniture_probability_maps(self) -> Dict[str, np.ndarray]:
        """Ground-truth spatial probability map per furniture name, over the base grid.
        Cells with no furniture are excluded (there's no single "no furniture" footprint
        that's meaningful to compare via KL the way a specific piece of furniture's is)."""
        grid = self.furniture_grid
        maps = {}
        for name in np.unique(grid):
            if name == "":
                continue
            mask = grid == name
            maps[str(name)] = mask.astype(float) / mask.sum()
        return maps

    def _rasterize(self):
        H, W = self.grid_size
        (xmin, xmax), (ymin, ymax) = self.bounds
        xs = xmin + (np.arange(W) + 0.5) / W * (xmax - xmin)
        ys = ymax - (np.arange(H) + 0.5) / H * (ymax - ymin)
        xx, yy = np.meshgrid(xs, ys)

        room_grid = np.full((H, W), "wall", dtype=object)
        for room in self.rooms:
            room_grid[room.contains(xx, yy)] = room.name

        furniture_grid = np.full((H, W), "", dtype=object)
        for item in self.furniture:
            furniture_grid[item.contains(xx, yy)] = item.name

        self._room_grid = room_grid
        self._furniture_grid = furniture_grid

    def render(self, ax=None, show_furniture: bool = True, show_labels: bool = True):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))

        room_palette = ["#c6dbef", "#fdd0a2", "#c7e9c0", "#dadaeb", "#fbb4b9", "#ffffb3"]
        room_colors = {name: room_palette[i % len(room_palette)] for i, name in enumerate(self.room_names)}

        (xmin, xmax), (ymin, ymax) = self.bounds
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor="#8a8a8a", zorder=0))

        for room in self.rooms:
            b = room.bounds
            ax.add_patch(Rectangle(
                (b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                facecolor=room_colors.get(room.name, "#eeeeee"), edgecolor="none", zorder=1,
            ))
            if show_labels:
                cx = (b["xmin"] + b["xmax"]) / 2
                cy = b["ymax"] - 0.025
                ax.text(cx, cy, room.name, ha="center", va="top", fontsize=9, style="italic", color="#444444", zorder=4)

        if show_furniture:
            for item in self.furniture:
                b = item.bounds
                ax.add_patch(Rectangle(
                    (b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                    facecolor="#6b3e26", edgecolor="black", linewidth=0.8, zorder=2,
                ))
                if show_labels:
                    cx = (b["xmin"] + b["xmax"]) / 2
                    cy = (b["ymin"] + b["ymax"]) / 2
                    ax.text(cx, cy, item.name, ha="center", va="center", fontsize=7, color="white", zorder=3)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title("Building: rooms and furniture")
        return ax


def make_default_building() -> BuildingEnv:
    """A 2x2-room building (living_room, bedroom, kitchen, bathroom), each with
    a couple of furniture pieces nested inside it, separated by 0.06-wide walls."""
    rooms = [
        Room("living_room", {"xmin": -0.94, "xmax": -0.03, "ymin": 0.03, "ymax": 0.94}),
        Room("bedroom", {"xmin": 0.03, "xmax": 0.94, "ymin": 0.03, "ymax": 0.94}),
        Room("kitchen", {"xmin": -0.94, "xmax": -0.03, "ymin": -0.94, "ymax": -0.03}),
        Room("bathroom", {"xmin": 0.03, "xmax": 0.94, "ymin": -0.94, "ymax": -0.03}),
    ]
    furniture = [
        Furniture("sofa", "living_room", {"xmin": -0.9, "xmax": -0.5, "ymin": 0.1, "ymax": 0.35}),
        Furniture("tv_stand", "living_room", {"xmin": -0.5, "xmax": -0.2, "ymin": 0.68, "ymax": 0.82}),
        Furniture("bed", "bedroom", {"xmin": 0.15, "xmax": 0.65, "ymin": 0.5, "ymax": 0.82}),
        Furniture("wardrobe", "bedroom", {"xmin": 0.75, "xmax": 0.93, "ymin": 0.1, "ymax": 0.5}),
        Furniture("table", "kitchen", {"xmin": -0.7, "xmax": -0.35, "ymin": -0.65, "ymax": -0.35}),
        Furniture("stove", "kitchen", {"xmin": -0.93, "xmax": -0.75, "ymin": -0.93, "ymax": -0.75}),
        Furniture("sink", "bathroom", {"xmin": 0.1, "xmax": 0.3, "ymin": -0.3, "ymax": -0.1}),
        Furniture("tub", "bathroom", {"xmin": 0.5, "xmax": 0.9, "ymin": -0.9, "ymax": -0.55}),
    ]
    return BuildingEnv(rooms=rooms, furniture=furniture)
