#!/usr/bin/env python
"""Two static top-down occupancy scenes for the SSP encoded-map demos.

indoor  : fine-line room geometry in a 10x10 m square -- thin walls with
          door gaps forming five rooms (no furniture)
outdoor : open 10x10 m environment with a few large tree/vegetation blobs

Run as a script to render the black-and-white two-panel figure and save the
boolean occupancy grids:

    python scripts/make_scene_maps.py

Import build_scenes() to get polygons + grids (e.g. for the gain widget).
"""

import argparse
from pathlib import Path as FSPath

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.path import Path as MplPath

WORLD = 10.0
WALL_T = 0.12


# ---------------------------------------------------------------------------
# Indoor: five rooms drawn only as thin wall lines with door gaps
# ---------------------------------------------------------------------------

# walls as (x, y, w, h) rectangles: outer ring + interior walls w/ door gaps
WALLS = [
    (0, 0, WORLD, WALL_T), (0, WORLD - WALL_T, WORLD, WALL_T),
    (0, 0, WALL_T, WORLD), (WORLD - WALL_T, 0, WALL_T, WORLD),
    # vertical wall at x=4.5, door gaps at y 2.0-3.0 and 6.9-7.9
    (4.5 - WALL_T / 2, 0.0, WALL_T, 2.0),
    (4.5 - WALL_T / 2, 3.0, WALL_T, 3.9),
    (4.5 - WALL_T / 2, 7.9, WALL_T, 2.1),
    # horizontal wall at y=5 (right half), door gap at x 6.4-7.4
    (4.5, 5.0 - WALL_T / 2, 1.9, WALL_T),
    (7.4, 5.0 - WALL_T / 2, 2.6, WALL_T),
    # horizontal wall at y=3.4 (left half), door gap at x 1.6-2.6
    (0.0, 3.4 - WALL_T / 2, 1.6, WALL_T),
    (2.6, 3.4 - WALL_T / 2, 1.9, WALL_T),
    # vertical wall at x=7.3 (top right), door gap at y 8.3-9.3
    (7.3 - WALL_T / 2, 5.0, WALL_T, 3.3),
    (7.3 - WALL_T / 2, 9.3, WALL_T, 0.7),
]


def indoor_polygons():
    return []


# ---------------------------------------------------------------------------
# Outdoor: a few large vegetation blobs
# ---------------------------------------------------------------------------

# (center, mean radius, xy-stretch, angle_deg) -- tree canopies + a hedge
VEG = [
    ((2.7, 7.2), 1.65, (1.0, 1.0), 0),      # large canopy
    ((7.4, 6.1), 1.20, (1.0, 1.0), 0),      # medium canopy
    ((8.3, 2.2), 1.00, (1.0, 1.0), 0),      # medium canopy
    ((1.7, 2.0), 0.85, (1.0, 1.0), 0),      # small tree
    ((4.9, 3.4), 0.60, (1.0, 1.0), 0),      # bush
    ((6.3, 9.0), 0.75, (2.1, 0.55), -12),   # hedge row
]


def blob(rng, center, r, stretch=(1, 1), angle_deg=0.0, n=72, wobble=0.22):
    """Organic closed outline: radius perturbed by a few low harmonics."""
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rad = np.ones(n)
    for h in (2, 3, 5, 8):
        rad += wobble / h * rng.uniform(0.5, 1.5) * np.sin(
            h * th + rng.uniform(0, 2 * np.pi))
    pts = np.column_stack([np.cos(th), np.sin(th)]) * (r * rad)[:, None]
    pts *= stretch
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    return pts @ np.array([[c, s], [-s, c]]) + center


def outdoor_polygons(seed=3):
    rng = np.random.default_rng(seed)
    return [blob(rng, c, r, st, ang) for c, r, st, ang in VEG]


# ---------------------------------------------------------------------------
# Rasterization + rendering
# ---------------------------------------------------------------------------


def rasterize(polys, rects, gn):
    """Boolean occupancy grid, row 0 = bottom (y up)."""
    cc = (np.arange(gn) + 0.5) / gn * WORLD
    X, Y = np.meshgrid(cc, cc)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    occ = np.zeros(gn * gn, bool)
    for p in polys:
        occ |= MplPath(p).contains_points(pts)
    for x, y, w, h in rects:
        occ |= ((pts[:, 0] >= x) & (pts[:, 0] <= x + w) &
                (pts[:, 1] >= y) & (pts[:, 1] <= y + h))
    return occ.reshape(gn, gn)


def build_scenes(gn=220, seed=3):
    """Both scenes: polygons/rects for drawing, occupancy grids for encoding."""
    indoor_polys = indoor_polygons()
    veg_polys = outdoor_polygons(seed)
    return {
        "world": WORLD,
        "indoor": {"polys": indoor_polys, "rects": WALLS,
                   "occ": rasterize(indoor_polys, WALLS, gn)},
        "outdoor": {"polys": veg_polys, "rects": [],
                    "occ": rasterize(veg_polys, [], gn)},
    }


def render(scenes, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.8), facecolor="white")
    for ax, key, title in [(axes[0], "indoor", "indoor (fine geometry)"),
                           (axes[1], "outdoor", "outdoor (large vegetation)")]:
        sc = scenes[key]
        for x, y, w, h in sc["rects"]:
            ax.add_patch(Rectangle((x, y), w, h, facecolor="black",
                                   edgecolor="none"))
        for p in sc["polys"]:
            ax.add_patch(MplPolygon(p, closed=True, facecolor="black",
                                    edgecolor="none"))
        ax.set_xlim(0, WORLD); ax.set_ylim(0, WORLD)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#c8cdd3")
        ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid-n", type=int, default=220)
    ap.add_argument("--seed", type=int, default=3, help="vegetation blob shapes")
    ap.add_argument("--out", type=str, default="results/widgets")
    args = ap.parse_args()

    out = FSPath(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = build_scenes(args.grid_n, args.seed)
    render(scenes, out / "scene_maps.png")
    np.savez_compressed(out / "scene_maps.npz", world=scenes["world"],
                        indoor=scenes["indoor"]["occ"],
                        outdoor=scenes["outdoor"]["occ"])
    for key in ("indoor", "outdoor"):
        occ = scenes[key]["occ"]
        print(f"{key}: {occ.mean() * 100:.1f}% occupied, "
              f"{len(scenes[key]['polys'])} objects")


if __name__ == "__main__":
    main()
