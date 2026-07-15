#!/usr/bin/env python
"""A 2x3 six-room indoor environment with ModelNet10 furniture landmarks.

Rooms are laid out on a grid and connected by doorways through every shared
wall (7 doors), so the whole map is traversable. Each room has a THEME that
controls the scale, density, and spatial distribution of its features --
the six-room analog of the quadrant blob world of
scripts/quadrant_scale_modulation.py:

    bedroom   -- few large items, wall-anchored (bed + nightstands + dresser)
    living    -- sparse medium/large, wall + open floor (sofa, table, TV)
    dining    -- one central table with a ring of chairs (tight cluster)
    office    -- desks in a regular row, each with a chair (structured)
    bathroom  -- few small/medium fixtures on walls (tub, toilet, cabinet)
    storage   -- many small items scattered densely (clutter)

Feature shapes are top-down silhouettes of real ModelNet10 meshes (z-up,
verified visually): surface points are projected to the ground plane,
rasterized, closed/filled, and traced into an outline polygon. Footprints are
cached in data/mn10_footprints.npz so the meshes are only processed once.

The module is import-friendly for the SSP encoding/decoding experiments:

    env = make_env(seed)
    env.items                # PlacedItem: room, category, center, polygon, ...
    env.rooms                # Room: name, theme, bounds, door centers
    env.inside(pts)          # (n,) bool: point is inside any feature
    env.item_ids_at(pts)     # (n,) int: item index at point (-1 = free)

Run as a script to generate + render the environment:

    python scripts/indoor_env.py --seed 7 --out results/indoor_env/indoor_env.png
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path as FSPath

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.path import Path as MplPath
from scipy import ndimage

# ---------------------------------------------------------------------------
# Geometry constants (meters)
# ---------------------------------------------------------------------------

ROOM_W, ROOM_H = 5.0, 5.0
N_COLS, N_ROWS = 3, 2
WORLD_W, WORLD_H = N_COLS * ROOM_W, N_ROWS * ROOM_H
WALL_T = 0.12          # wall half-thickness is WALL_T / 2 on each side
DOOR_W = 1.0
DOOR_CLEAR = 0.65      # keep-free radius in front of each door
GRID_RES = 0.025       # occupancy grid resolution
MARGIN = 0.08          # min gap between furniture items

MN10_ROOT = FSPath("data/ModelNet10")
FOOTPRINT_CACHE = FSPath("data/mn10_footprints.npz")

# real-world footprint sizes: max horizontal extent in meters (mean, jitter)
CATEGORY_SIZE = {
    "bed": (2.1, 0.1), "sofa": (2.05, 0.1), "bathtub": (1.7, 0.1),
    "desk": (1.5, 0.1), "table": (1.5, 0.15), "dresser": (1.3, 0.1),
    "toilet": (0.75, 0.04), "chair": (0.58, 0.05), "night_stand": (0.5, 0.05),
    "monitor": (0.55, 0.05),
}
N_INSTANCES = 4  # cached mesh instances per category


# ---------------------------------------------------------------------------
# Footprint extraction: ModelNet10 mesh -> top-down outline polygon
# ---------------------------------------------------------------------------


def _mask_to_polygon(mask, extent, n_max=80):
    """Trace the 0.5 contour of a binary mask, return the longest closed path
    decimated to <= n_max vertices, in the coordinates given by extent."""
    m = np.zeros((mask.shape[0] + 4, mask.shape[1] + 4))
    m[2:-2, 2:-2] = ndimage.gaussian_filter(mask.astype(float), 1.2)
    x0, x1, y0, y1 = extent
    dx = (x1 - x0) / (mask.shape[1] - 1)
    dy = (y1 - y0) / (mask.shape[0] - 1)
    xs = x0 + (np.arange(m.shape[1]) - 2) * dx
    ys = y0 + (np.arange(m.shape[0]) - 2) * dy
    fig = plt.figure()
    try:
        cs = plt.contour(xs, ys, m, levels=[0.5])
        segs = [s for level in cs.allsegs for s in level]
    finally:
        plt.close(fig)
    poly = max(segs, key=len)
    step = max(1, len(poly) // n_max)
    poly = poly[::step]
    return poly


def extract_footprint(off_path, n_pts=20000, grid_n=160):
    """Top-down (z-up) silhouette polygon of one mesh, normalized so the long
    axis lies along x with max extent 1, centered on the bbox center."""
    import trimesh

    mesh = trimesh.load(str(off_path), force="mesh")
    pts, _ = trimesh.sample.sample_surface(mesh, n_pts)
    xy = pts[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    pad = 0.03 * (hi - lo).max()
    lo, hi = lo - pad, hi + pad
    ij = np.clip(((xy - lo) / (hi - lo) * (grid_n - 1)).astype(int), 0, grid_n - 1)
    mask = np.zeros((grid_n, grid_n), bool)
    mask[ij[:, 1], ij[:, 0]] = True
    mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
    mask = ndimage.binary_fill_holes(mask)
    lab, n = ndimage.label(mask)
    if n > 1:  # keep the largest connected component
        mask = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    poly = _mask_to_polygon(mask, (lo[0], hi[0], lo[1], hi[1]))

    # normalize: center bbox, long axis along x, max extent 1
    lo2, hi2 = poly.min(0), poly.max(0)
    poly = poly - (lo2 + hi2) / 2
    ext = hi2 - lo2
    if ext[1] > ext[0]:
        poly = poly[:, ::-1] * [1, -1]  # rotate 90 deg
        ext = ext[::-1]
    return poly / ext.max()


def load_footprints(refresh=False):
    """{category: [poly, ...]} with polys normalized; cached across runs."""
    if FOOTPRINT_CACHE.exists() and not refresh:
        z = np.load(FOOTPRINT_CACHE)
        out = {}
        for key in z.files:
            cat, i = key.rsplit("__", 1)
            out.setdefault(cat, {})[int(i)] = z[key]
        return {c: [v[i] for i in sorted(v)] for c, v in out.items()}
    print("extracting ModelNet10 footprints (one-time, cached) ...")
    out = {}
    for cat in CATEGORY_SIZE:
        polys = []
        idx = 1
        while len(polys) < N_INSTANCES and idx < 40:
            p = MN10_ROOT / cat / "train" / f"{cat}_{idx:04d}.off"
            idx += 1
            if not p.exists():
                continue
            try:
                polys.append(extract_footprint(p))
                print(f"  {p.name}: {len(polys[-1])} vertices")
            except Exception as e:  # noqa: BLE001 -- skip unreadable meshes
                print(f"  {p.name}: skipped ({e})")
        out[cat] = polys
    np.savez(FOOTPRINT_CACHE, **{f"{c}__{i}": p for c, v in out.items()
                                 for i, p in enumerate(v)})
    print(f"cached footprints to {FOOTPRINT_CACHE}")
    return out


# ---------------------------------------------------------------------------
# Environment data structures
# ---------------------------------------------------------------------------


@dataclass
class PlacedItem:
    category: str
    instance: int
    room: str
    center: np.ndarray      # (2,)
    angle: float
    scale: float
    polygon: np.ndarray     # (n, 2) world coords


@dataclass
class Room:
    name: str
    theme: str
    row: int
    col: int
    bounds: tuple           # x0, x1, y0, y1 (room cell incl. wall centerline)
    doors: list = field(default_factory=list)   # (cx, cy) of each doorway

    @property
    def interior(self):
        x0, x1, y0, y1 = self.bounds
        t = WALL_T / 2
        return (x0 + t, x1 - t, y0 + t, y1 - t)

    @property
    def center(self):
        x0, x1, y0, y1 = self.bounds
        return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


class IndoorEnv:
    def __init__(self, rooms, doors, wall_rects, items, item_grid, block_grid):
        self.rooms = rooms
        self.doors = doors            # (room_a, room_b, cx, cy, horizontal)
        self.wall_rects = wall_rects  # (x, y, w, h) solid wall segments
        self.items = items
        self.item_grid = item_grid    # int grid, -1 free, else item index
        self.block_grid = block_grid

    def _cell(self, pts):
        ij = np.floor(np.atleast_2d(pts) / GRID_RES).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, self.item_grid.shape[1] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, self.item_grid.shape[0] - 1)
        return ij

    def item_ids_at(self, pts):
        ij = self._cell(pts)
        return self.item_grid[ij[:, 1], ij[:, 0]]

    def inside(self, pts):
        return self.item_ids_at(pts) >= 0


# ---------------------------------------------------------------------------
# Placement machinery (occupancy-grid rejection sampling)
# ---------------------------------------------------------------------------


class Placer:
    def __init__(self, rng):
        self.rng = rng
        nx = int(round(WORLD_W / GRID_RES))
        ny = int(round(WORLD_H / GRID_RES))
        self.block = np.zeros((ny, nx), bool)   # walls + clearance + items
        self.item_grid = np.full((ny, nx), -1, np.int32)
        self.items = []

    def mark_rect(self, x, y, w, h, grid=None):
        g = self.block if grid is None else grid
        i0, i1 = int(x / GRID_RES), int(np.ceil((x + w) / GRID_RES))
        j0, j1 = int(y / GRID_RES), int(np.ceil((y + h) / GRID_RES))
        g[max(j0, 0):j1, max(i0, 0):i1] = True

    def mark_disk(self, cx, cy, r):
        ny, nx = self.block.shape
        i0, i1 = max(int((cx - r) / GRID_RES), 0), min(int(np.ceil((cx + r) / GRID_RES)), nx)
        j0, j1 = max(int((cy - r) / GRID_RES), 0), min(int(np.ceil((cy + r) / GRID_RES)), ny)
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1))
        d2 = (ii * GRID_RES + GRID_RES / 2 - cx) ** 2 + (jj * GRID_RES + GRID_RES / 2 - cy) ** 2
        sel = d2 <= r * r
        self.block[jj[sel], ii[sel]] = True

    def _poly_cells(self, poly):
        lo, hi = poly.min(0), poly.max(0)
        i0, i1 = int(lo[0] / GRID_RES), int(np.ceil(hi[0] / GRID_RES)) + 1
        j0, j1 = int(lo[1] / GRID_RES), int(np.ceil(hi[1] / GRID_RES)) + 1
        if i0 < 0 or j0 < 0 or i1 > self.block.shape[1] or j1 > self.block.shape[0]:
            return None
        ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1))
        cent = np.column_stack([(ii.ravel() + 0.5) * GRID_RES,
                                (jj.ravel() + 0.5) * GRID_RES])
        ins = MplPath(poly).contains_points(cent).reshape(ii.shape)
        return ii, jj, ins

    def try_place(self, footprints, category, instance, room, center, angle, scale):
        """Transform footprint -> world polygon; commit if free of collisions
        and inside the room interior. Returns PlacedItem or None."""
        base = footprints[category][instance]
        c, s = np.cos(angle), np.sin(angle)
        poly = base * scale @ np.array([[c, s], [-s, c]]) + center
        x0, x1, y0, y1 = room.interior
        lo, hi = poly.min(0), poly.max(0)
        if lo[0] < x0 or hi[0] > x1 or lo[1] < y0 or hi[1] > y1:
            return None
        cells = self._poly_cells(poly)
        if cells is None:
            return None
        ii, jj, ins = cells
        grown = ndimage.binary_dilation(ins, iterations=int(MARGIN / GRID_RES))
        if self.block[jj, ii][grown].any():
            return None
        item = PlacedItem(category, instance, room.name, np.asarray(center, float),
                          float(angle), float(scale), poly)
        idx = len(self.items)
        self.items.append(item)
        self.block[jj[grown], ii[grown]] = True
        self.item_grid[jj[ins], ii[ins]] = idx
        return item

    def size_of(self, category):
        mu, jit = CATEGORY_SIZE[category]
        return mu * (1 + self.rng.uniform(-jit, jit))

    def pick(self, footprints, category):
        return int(self.rng.integers(len(footprints[category])))

    # -- placement strategies ------------------------------------------------

    def on_wall(self, fp, category, room, scale=None, walls="SNWE", face_out=False,
                tries=60, u_range=(0.12, 0.88)):
        """Back of the item against a random wall of the room. face_out=False
        aligns the footprint's long (x) axis with the wall; face_out=True
        points the long axis into the room (e.g. a bed's headboard on the
        wall)."""
        scale = scale or self.size_of(category)
        inst = self.pick(fp, category)
        base = fp[category][inst] * scale
        ext = base.max(0) - base.min(0)
        x0, x1, y0, y1 = room.interior
        for _ in range(tries):
            wall = self.rng.choice(list(walls))
            u = self.rng.uniform(*u_range)
            ang = {"S": 0.0, "N": np.pi, "W": -np.pi / 2, "E": np.pi / 2}[wall]
            if face_out:
                ang += np.pi / 2
            depth = ext[0 if face_out else 1]
            along = ext[1 if face_out else 0]
            gap = MARGIN + 0.03 + depth / 2
            if wall == "S":
                pos = (x0 + along / 2 + u * (x1 - x0 - along), y0 + gap)
            elif wall == "N":
                pos = (x0 + along / 2 + u * (x1 - x0 - along), y1 - gap)
            elif wall == "W":
                pos = (x0 + gap, y0 + along / 2 + u * (y1 - y0 - along))
            else:
                pos = (x1 - gap, y0 + along / 2 + u * (y1 - y0 - along))
            item = self.try_place(fp, category, inst, room, pos, ang, scale)
            if item:
                return item
        return None

    def scatter(self, fp, category, room, scale=None, tries=80):
        scale = scale or self.size_of(category)
        inst = self.pick(fp, category)
        x0, x1, y0, y1 = room.interior
        for _ in range(tries):
            pos = self.rng.uniform([x0 + scale / 2, y0 + scale / 2],
                                   [x1 - scale / 2, y1 - scale / 2])
            ang = self.rng.uniform(0, 2 * np.pi)
            item = self.try_place(fp, category, inst, room, pos, ang, scale)
            if item:
                return item
        return None

    def ext_of(self, fp, category, inst, scale):
        b = fp[category][inst] * scale
        return b.max(0) - b.min(0)

    def at(self, fp, category, room, pos, ang, scale=None, jitter=0.0, tries=25,
           inst=None):
        scale = scale or self.size_of(category)
        inst = self.pick(fp, category) if inst is None else inst
        for _ in range(tries):
            p = np.asarray(pos) + self.rng.uniform(-jitter, jitter, 2)
            item = self.try_place(fp, category, inst, room, p, ang, scale)
            if item:
                return item
        return None


# ---------------------------------------------------------------------------
# Room themes
# ---------------------------------------------------------------------------


def furnish_bedroom(pl, fp, room):
    # headboard on a door-free wall so both nightstands have space
    doors = wall_doors(room)
    free = "".join(w for w in "SNWE" if not doors[w]) or "SNWE"
    bed = pl.on_wall(fp, "bed", room, walls=free, face_out=True, u_range=(0.25, 0.75))
    if bed is not None:
        # nightstands flank the headboard, against the same wall
        d = np.array([np.cos(bed.angle), np.sin(bed.angle)])   # long-axis dir
        n = np.array([-d[1], d[0]])                            # across the bed
        ext = bed.polygon.max(0) - bed.polygon.min(0)
        head = bed.center - d * 0.5 * max(ext) * 0.78
        w = CATEGORY_SIZE["night_stand"][0]
        across = float(np.abs(ext) @ np.abs(n))   # bed width across n (axis-aligned)
        for side in (-1, 1):
            pl.at(fp, "night_stand", room, head + n * side * (across / 2 + w / 1.6),
                  bed.angle, jitter=0.04)
    pl.on_wall(fp, "dresser", room)


WALL_OF_ANGLE = {0.0: "S", np.pi: "N", -np.pi / 2: "W", np.pi / 2: "E"}
OPPOSITE_WALL = {"S": "N", "N": "S", "W": "E", "E": "W"}


def wall_doors(room):
    """Which walls of the room contain a doorway."""
    x0, x1, y0, y1 = room.bounds
    has = {w: False for w in "SNWE"}
    for cx, cy in room.doors:
        if abs(cy - y0) < 1e-6:
            has["S"] = True
        elif abs(cy - y1) < 1e-6:
            has["N"] = True
        elif abs(cx - x0) < 1e-6:
            has["W"] = True
        elif abs(cx - x1) < 1e-6:
            has["E"] = True
    return has


def furnish_living(pl, fp, room):
    sofa = pl.on_wall(fp, "sofa", room, u_range=(0.3, 0.7))
    tv_walls = "SNWE"
    if sofa is not None:
        # low table in front of the sofa, TV (monitor) on the opposite wall
        inward = room.center - sofa.center
        inward = inward / np.linalg.norm(inward)
        pl.at(fp, "table", room, sofa.center + inward * 1.35, sofa.angle,
              scale=1.05, jitter=0.1)
        tv_walls = OPPOSITE_WALL.get(WALL_OF_ANGLE.get(sofa.angle), "SNWE")
    pl.on_wall(fp, "dresser", room, scale=1.5)
    pl.on_wall(fp, "monitor", room, walls=tv_walls)
    pl.scatter(fp, "chair", room)


def furnish_dining(pl, fp, room):
    c = room.center + pl.rng.uniform(-0.3, 0.3, 2)
    table = pl.at(fp, "table", room, c, pl.rng.uniform(0, np.pi), jitter=0.05)
    if table is None:
        return
    r_ring = table.scale / 2 + CATEGORY_SIZE["chair"][0] * 0.75
    th0 = pl.rng.uniform(0, 2 * np.pi)
    for k in range(6):
        th = th0 + k * np.pi / 3
        pos = table.center + r_ring * np.array([np.cos(th), np.sin(th)])
        pl.at(fp, "chair", room, pos, th + np.pi / 2, jitter=0.06)


def furnish_office(pl, fp, room):
    # desk row along a door-free wall (door clearance would break the row)
    doors = wall_doors(room)
    free = [w for w in "SNWE" if not doors[w]] or ["S"]
    wall = pl.rng.choice(free)
    x0, x1, y0, y1 = room.interior
    lo, hi = (x0, x1) if wall in "SN" else (y0, y1)
    for k in range(3):
        u = lo + (k + 0.5) * (hi - lo) / 3
        scale = pl.size_of("desk")
        inst = pl.pick(fp, "desk")
        depth = pl.ext_of(fp, "desk", inst, scale)[1]
        d_off = MARGIN + 0.04 + depth / 2
        c_off = MARGIN + depth + 0.55
        pos, ang, cpos, cang = {
            "S": ((u, y0 + d_off), 0.0, (u, y0 + c_off), np.pi),
            "N": ((u, y1 - d_off), np.pi, (u, y1 - c_off), 0.0),
            "W": ((x0 + d_off, u), -np.pi / 2, (x0 + c_off, u), np.pi / 2),
            "E": ((x1 - d_off, u), np.pi / 2, (x1 - c_off, u), -np.pi / 2),
        }[wall]
        desk = pl.at(fp, "desk", room, pos, ang, scale=scale, inst=inst, jitter=0.02)
        if desk is not None:
            pl.at(fp, "chair", room, cpos, cang, jitter=0.05)
    pl.on_wall(fp, "dresser", room, walls=OPPOSITE_WALL[wall], scale=1.1)


def furnish_bathroom(pl, fp, room):
    pl.on_wall(fp, "bathtub", room, u_range=(0.12, 0.4))
    pl.on_wall(fp, "toilet", room, face_out=True)
    pl.on_wall(fp, "night_stand", room)   # cabinet


def furnish_storage(pl, fp, room):
    cats = ["chair", "night_stand", "monitor", "toilet", "chair", "night_stand"]
    for k in range(15):
        cat = cats[k % len(cats)]
        pl.scatter(fp, cat, room, scale=pl.size_of(cat) * pl.rng.uniform(0.8, 1.1),
                   tries=40)


THEMES = {
    "bedroom": furnish_bedroom, "living": furnish_living, "dining": furnish_dining,
    "office": furnish_office, "bathroom": furnish_bathroom, "storage": furnish_storage,
}
# (row, col) -> theme; row 0 = bottom
ROOM_LAYOUT = {
    (1, 0): "bedroom", (1, 1): "living", (1, 2): "bathroom",
    (0, 0): "office", (0, 1): "dining", (0, 2): "storage",
}


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------


def make_env(seed=7, footprints=None, refresh_footprints=False):
    rng = np.random.default_rng(seed)
    fp = footprints or load_footprints(refresh=refresh_footprints)

    rooms = []
    for (r, c), theme in sorted(ROOM_LAYOUT.items()):
        bounds = (c * ROOM_W, (c + 1) * ROOM_W, r * ROOM_H, (r + 1) * ROOM_H)
        rooms.append(Room(name=f"{theme}", theme=theme, row=r, col=c, bounds=bounds))
    by_rc = {(rm.row, rm.col): rm for rm in rooms}

    # doors on every shared wall of the grid
    doors = []
    for (r, c), rm in by_rc.items():
        if (r, c + 1) in by_rc:   # vertical wall to the right neighbor
            u = rng.uniform(0.3, 0.7)
            cy = r * ROOM_H + u * ROOM_H
            doors.append((rm.name, by_rc[(r, c + 1)].name, (c + 1) * ROOM_W, cy, False))
        if (r + 1, c) in by_rc:   # horizontal wall to the upper neighbor
            u = rng.uniform(0.3, 0.7)
            cx = c * ROOM_W + u * ROOM_W
            doors.append((rm.name, by_rc[(r + 1, c)].name, cx, (r + 1) * ROOM_H, True))

    # wall segments = full grid walls minus door gaps
    pl = Placer(rng)
    wall_rects = []

    def add_wall_span(fixed, lo, hi, horizontal, gaps):
        """One wall line from lo..hi at coordinate `fixed`, minus door gaps."""
        segs, start = [], lo
        for g0, g1 in sorted(gaps):
            segs.append((start, g0))
            start = g1
        segs.append((start, hi))
        for s0, s1 in segs:
            if s1 - s0 < 1e-6:
                continue
            if horizontal:
                wall_rects.append((s0, fixed - WALL_T / 2, s1 - s0, WALL_T))
            else:
                wall_rects.append((fixed - WALL_T / 2, s0, WALL_T, s1 - s0))

    for c in range(N_COLS + 1):   # vertical wall lines
        x = c * ROOM_W
        gaps = [(cy - DOOR_W / 2, cy + DOOR_W / 2)
                for _, _, dx, cy, hor in doors if not hor and abs(dx - x) < 1e-6]
        add_wall_span(x, 0.0, WORLD_H, horizontal=False, gaps=gaps)
    for r in range(N_ROWS + 1):   # horizontal wall lines
        y = r * ROOM_H
        gaps = [(cx - DOOR_W / 2, cx + DOOR_W / 2)
                for _, _, cx, dy, hor in doors if hor and abs(dy - y) < 1e-6]
        add_wall_span(y, 0.0, WORLD_W, horizontal=True, gaps=gaps)

    for x, y, w, h in wall_rects:
        pl.mark_rect(x, y, w, h)
    for _, _, cx, cy, _ in doors:
        pl.mark_disk(cx, cy, DOOR_CLEAR)

    for _, _, cx, cy, _ in doors:   # record doors on both rooms
        for rm in rooms:
            x0, x1, y0, y1 = rm.bounds
            if x0 - 1e-6 <= cx <= x1 + 1e-6 and y0 - 1e-6 <= cy <= y1 + 1e-6:
                rm.doors.append((cx, cy))

    for rm in rooms:
        THEMES[rm.theme](pl, fp, rm)

    return IndoorEnv(rooms, doors, wall_rects, pl.items, pl.item_grid, pl.block)


# ---------------------------------------------------------------------------
# Rendering + export
# ---------------------------------------------------------------------------

CATEGORY_COLOR = {
    "bed": "#8aa9c9", "sofa": "#9db98f", "bathtub": "#a3c4c9", "desk": "#c9b18a",
    "table": "#c9a98f", "dresser": "#b3a1c7", "toilet": "#c9c3a3",
    "chair": "#d0907f", "night_stand": "#c7a1b4", "monitor": "#8f8f9e",
}
THEME_TINT = {
    "bedroom": "#f6f1ea", "living": "#eef3ec", "dining": "#f5eeee",
    "office": "#edf1f5", "bathroom": "#ecf4f5", "storage": "#f3f0ea",
}


def render_env(env, path, dpi=170):
    fig, ax = plt.subplots(figsize=(13.2, 9.0), facecolor="white")
    for rm in env.rooms:
        x0, x1, y0, y1 = rm.bounds
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor=THEME_TINT[rm.theme], edgecolor="none", zorder=0))
        n_items = sum(1 for it in env.items if it.room == rm.name)
        ax.text(x0 + 0.18, y1 - 0.18, rm.theme.upper(), fontsize=11, color="0.45",
                ha="left", va="top", fontweight="bold", zorder=5)
        ax.text(x0 + 0.18, y1 - 0.52, f"{n_items} items", fontsize=8.5,
                color="0.55", ha="left", va="top", zorder=5)
    for x, y, w, h in env.wall_rects:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="0.15", edgecolor="none", zorder=4))
    for _, _, cx, cy, hor in env.doors:   # door leaf: thin open line
        if hor:
            ax.plot([cx - DOOR_W / 2, cx + DOOR_W / 2], [cy, cy], color="0.75",
                    lw=1.0, ls=(0, (4, 3)), zorder=4)
        else:
            ax.plot([cx, cx], [cy - DOOR_W / 2, cy + DOOR_W / 2], color="0.75",
                    lw=1.0, ls=(0, (4, 3)), zorder=4)
    for it in env.items:
        ax.add_patch(MplPolygon(it.polygon, closed=True,
                                facecolor=CATEGORY_COLOR[it.category],
                                edgecolor="0.25", lw=0.7, zorder=2))
    handles = [plt.Line2D([], [], marker="s", ls="none", ms=9,
                          markerfacecolor=c, markeredgecolor="0.25", label=cat)
               for cat, c in CATEGORY_COLOR.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=10, fontsize=9, frameon=False)
    ax.set_xlim(-0.25, WORLD_W + 0.25)
    ax.set_ylim(-0.25, WORLD_H + 0.25)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("six-room indoor environment (2x3, ModelNet10 furniture footprints)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def export_env(env, path):
    data = {
        "world": [WORLD_W, WORLD_H], "grid_res": GRID_RES,
        "rooms": [{"name": rm.name, "theme": rm.theme, "row": rm.row, "col": rm.col,
                   "bounds": list(rm.bounds), "doors": [list(d) for d in rm.doors]}
                  for rm in env.rooms],
        "doors": [{"rooms": [a, b], "center": [cx, cy], "horizontal": hor,
                   "width": DOOR_W} for a, b, cx, cy, hor in env.doors],
        "walls": [list(w) for w in env.wall_rects],
        "items": [{"category": it.category, "instance": it.instance, "room": it.room,
                   "center": it.center.tolist(), "angle": it.angle, "scale": it.scale,
                   "polygon": it.polygon.tolist()} for it in env.items],
    }
    with open(path, "w") as f:
        json.dump(data, f)
    np.savez_compressed(FSPath(path).with_suffix(".grids.npz"),
                        item_grid=env.item_grid, block_grid=env.block_grid)
    print(f"wrote {path} (+ .grids.npz)")


def print_stats(env):
    print(f"\n{'room':<10} {'items':>5} {'mean size':>10} {'size range':>13}  distribution")
    style = {"bedroom": "wall-anchored", "living": "wall + open floor",
             "dining": "central cluster", "office": "regular rows",
             "bathroom": "wall fixtures", "storage": "dense scatter"}
    for rm in env.rooms:
        its = [it for it in env.items if it.room == rm.name]
        if not its:
            print(f"{rm.theme:<10} {0:>5}  (placement failed!)")
            continue
        sizes = [it.scale for it in its]
        print(f"{rm.theme:<10} {len(its):>5} {np.mean(sizes):>9.2f}m "
              f"{min(sizes):>5.2f}-{max(sizes):.2f}m  {style[rm.theme]}")
    print(f"total items: {len(env.items)}, doors: {len(env.doors)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--refresh-footprints", action="store_true")
    ap.add_argument("--out", type=str, default="results/indoor_env/indoor_env.png")
    args = ap.parse_args()

    out = FSPath(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = make_env(args.seed, refresh_footprints=args.refresh_footprints)
    print_stats(env)
    render_env(env, out)
    export_env(env, out.with_suffix(".json"))


if __name__ == "__main__":
    main()
