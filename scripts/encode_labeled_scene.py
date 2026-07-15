#!/usr/bin/env python
"""SSP-encode a labeled RGB scene (SceneNN / Semantic3D) and render the
similarity field color-coded by semantic label, one file per viewing angle.

All (voxel-subsampled) scene points are encoded and bundled into one mean SSP
with the DISTMOD kernel modulation (as in flythrough_ssp_similarity.py): a
Gaussian band over log scale whose center slides from the finest SSP scale at
the viewpoint to the coarsest at the far side of the scene, so each rendered
angle has fine detail nearby and coarse structure in the distance. The same
per-point gains modulate the query-grid encodings.

Each output file (one per azimuth) shows, from that angle:
    left  : the input cloud colored by ground-truth label
    right : the similarity cloud -- grid points above threshold, colored by
            the label of the nearest scene point, alpha ~ similarity, with
            the modulation viewpoint marked (white star)

Usage:
    python scripts/encode_labeled_scene.py --dataset scenenn --scene 005
    python scripts/encode_labeled_scene.py --dataset semantic3d \
        --scene bildstein_station3 --crop-radius 20 --grid-res 0.4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree

from flythrough_ssp_similarity import distance_band_gains, encode_chunked
from multiscalessps.data import (LabeledScene, SceneNNScene,
                                 Semantic3DScene)
from visualize_ssp_lidar import (expand_gains, make_grid, make_ssp_space,
                                 modulate)

SEM3D_PALETTE = {          # semantic-8 class id -> display color
    0: "#777777", 1: "#8c6d31", 2: "#5aae61", 3: "#1b7837", 4: "#a6dba0",
    5: "#d6604d", 6: "#9970ab", 7: "#fddbc7", 8: "#2166ac",
}


def load_scene(dataset, scene, data_dir):
    if dataset == "scenenn":
        return SceneNNScene(data_dir / "scenenn" / scene)
    return Semantic3DScene(data_dir / "semantic3d" / scene)


def label_colors(labels, label_names, dataset, max_classes=12):
    """Per-point RGB by label plus legend handles (top classes by count)."""
    ids, counts = np.unique(labels, return_counts=True)
    order = ids[np.argsort(-counts)]
    if dataset == "semantic3d":
        lut = {i: matplotlib.colors.to_rgb(SEM3D_PALETTE[int(i)])
               for i in ids}
        legend_ids = [i for i in order if i != 0][:max_classes]
    else:
        cmap = plt.get_cmap("tab20")
        legend_ids = [i for i in order
                      if not label_names.get(int(i), "").startswith("instance")
                      ][:max_classes]
        lut = {int(i): (0.45, 0.45, 0.45) for i in ids}       # default grey
        for k, i in enumerate(legend_ids):
            lut[int(i)] = cmap(k % 20)[:3]
    colors = np.array([lut[int(i)] for i in labels], dtype=np.float32)
    handles = [Line2D([], [], marker="o", ls="", color=lut[int(i)],
                      label=label_names.get(int(i), f"label {i}"))
               for i in legend_ids]
    return colors, handles


def setup_axes(ax, lo, hi, elev, azim):
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo)
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--dataset", choices=["scenenn", "semantic3d"],
                        default="scenenn")
    parser.add_argument("--scene", default="005")
    parser.add_argument("--voxel", type=float, default=None,
                        help="Subsample voxel (m); default 0.05 indoor, "
                             "0.15 outdoor.")
    parser.add_argument("--num-points", type=int, default=40000,
                        help="Cap on encoded points after voxel subsample.")
    parser.add_argument("--crop-radius", type=float, default=None,
                        help="Keep only points within this radius (m) of the "
                             "scene median (recommended ~20 for semantic3d).")
    parser.add_argument("--region-frac", type=float, default=None,
                        help="Encode/plot only a ball of radius "
                             "region_frac * scene extent (e.g. 0.1) around "
                             "--region-center. Voxel, grid-res and base-ls "
                             "rescale to the region unless given explicitly.")
    parser.add_argument("--region-center", type=float, nargs=3, default=None,
                        help="Region center (m, scene coords); default: the "
                             "scene point nearest the median.")
    parser.add_argument("--ssp-dim", type=int, default=1500)
    parser.add_argument("--base-ls", type=float, default=None,
                        help="Base SSP length scale (m); default 0.25 indoor, "
                             "0.5 outdoor.")
    parser.add_argument("--sigma-log", type=float, default=0.6)
    parser.add_argument("--grid-res", type=float, default=None,
                        help="Query grid spacing (m); default 0.12 indoor, "
                             "0.4 outdoor.")
    parser.add_argument("--angles", type=float, nargs="+",
                        default=[30, 120, 210, 300])
    parser.add_argument("--elev", type=float, default=25.0)
    parser.add_argument("--thr", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--out-prefix", default=None,
                        help="Default: ssp_label_<dataset><scene>")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    outdoor = args.dataset == "semantic3d"
    prefix = args.out_prefix or f"ssp_label_{args.dataset}{args.scene}"

    scene = load_scene(args.dataset, args.scene, args.data_dir)

    if args.region_frac:
        full = scene.points
        r = args.region_frac * float(np.ptp(full, axis=0).max())
        if args.region_center is not None:
            rc = np.asarray(args.region_center, dtype=np.float32)
        else:
            med = np.median(full, axis=0)
            rc = full[np.argmin(np.linalg.norm(full - med, axis=1))]
        mask = np.linalg.norm(full - rc, axis=1) < r
        scene = LabeledScene(full[mask], scene.colors[mask],
                             scene.labels[mask], scene.label_names)
        # Rescale resolution and kernel defaults to the region size.
        voxel = args.voxel or max(r / 60, 0.01)
        base_ls = args.base_ls or r / 8
        grid_res = args.grid_res or r / 22
        prefix += f"_r{args.region_frac:g}"
        print(f"region: radius {r:.2f} m around {rc.round(2)} "
              f"({mask.sum():,} raw points)")
    else:
        voxel = args.voxel or (0.15 if outdoor else 0.05)
        base_ls = args.base_ls or (0.5 if outdoor else 0.25)
        grid_res = args.grid_res or (0.4 if outdoor else 0.12)

    pts, cols, labs = scene.subsample(voxel)
    if args.crop_radius:
        center = np.median(pts, axis=0)
        keep = np.linalg.norm(pts - center, axis=1) < args.crop_radius
        pts, cols, labs = pts[keep], cols[keep], labs[keep]
    rng = np.random.default_rng(args.seed)
    if len(pts) > args.num_points:
        sel = rng.choice(len(pts), args.num_points, replace=False)
        pts, cols, labs = pts[sel], cols[sel], labs[sel]
    center = pts.mean(axis=0)
    pts = pts - center
    print(f"{args.dataset}/{args.scene}: encoding {len(pts)} points, "
          f"extent {np.ptp(pts, axis=0).round(1)} m")

    pad = base_ls
    lo, hi = pts.min(axis=0) - pad, pts.max(axis=0) + pad
    n_longest = int(np.ceil((hi - lo).max() / grid_res))
    grid, counts = make_grid(lo, hi, n_longest)
    print(f"grid {counts[0]}x{counts[1]}x{counts[2]} = {len(grid)}")

    sp, scale_idx, scales = make_ssp_space(base_ls, args.ssp_dim,
                                           seed=args.seed)
    print("encoding cloud + grid...")
    enc_cloud = encode_chunked(sp, pts)
    enc_grid = encode_chunked(sp, grid)

    # Label of each grid point = label of nearest scene point.
    grid_labs = labs[cKDTree(pts).query(grid, workers=-1)[1]]
    cloud_rgb, handles = label_colors(labs, scene.label_names, args.dataset)
    grid_rgb, _ = label_colors(grid_labs, scene.label_names, args.dataset)

    # Viewpoints orbit the scene at each azimuth (matching the render view);
    # distmod distances are measured from that viewpoint.
    extent = hi - lo
    orbit_r = 0.75 * float(extent[:2].max())
    cam_z = lo[2] + 0.75 * extent[2]

    # Pass 1: similarities for every angle (shared normalization).
    all_sims, viewpoints = [], []
    for az in args.angles:
        a = np.radians(az)
        vp = np.array([orbit_r * np.cos(a), orbit_r * np.sin(a), cam_z])
        d_cloud = np.linalg.norm(pts - vp, axis=1)
        d_grid = np.linalg.norm(grid - vp, axis=1)
        # Sweep the band over the actual distance range so the fine->coarse
        # gradient spans the plotted volume even when it is a small region.
        d0, d1 = float(d_cloud.min()), float(d_cloud.max())
        g_c = expand_gains(distance_band_gains(d_cloud - d0, d1 - d0, scales,
                                               args.sigma_log), scale_idx)
        g_g = expand_gains(distance_band_gains(d_grid - d0, d1 - d0, scales,
                                               args.sigma_log), scale_idx)
        mod_mean = modulate(jnp.asarray(enc_cloud), g_c).mean(axis=0)
        sims = np.asarray(modulate(jnp.asarray(enc_grid), g_g) @ mod_mean)
        print(f"  az={az:g}: sim range [{sims.min():.4f}, {sims.max():.4f}]")
        all_sims.append(sims.astype(np.float32))
        viewpoints.append(vp)
    smax = max(s.max() for s in all_sims) + 1e-12

    # Pass 2: one file per angle (dark theme -- label colors and faint
    # similarities read much better than on white).
    BG = "#0b0e14"
    FG = "#c8d4e3"
    for az, sims, vp in zip(args.angles, all_sims, viewpoints):
        s = np.clip(sims / smax, 0, 1)
        keep = s > args.thr
        alpha = np.clip((s[keep] - args.thr) / (1 - args.thr), 0, 1) \
            ** args.gamma
        rgba = np.concatenate([grid_rgb[keep], alpha[:, None]], axis=1)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5.8),
                                subplot_kw={"projection": "3d"},
                                facecolor=BG)
        axs[0].scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cloud_rgb,
                       s=1.5, linewidths=0)
        axs[0].set_title("input cloud, ground-truth labels", fontsize=10,
                         color=FG, y=0.98)
        axs[1].scatter(grid[keep, 0], grid[keep, 1], grid[keep, 2],
                       c=rgba, s=7, linewidths=0)
        axs[1].scatter(*vp, marker="*", s=180, c="white",
                       edgecolors="black", linewidths=0.6, zorder=10)
        axs[1].set_title("SSP similarity (distmod), label-colored",
                         fontsize=10, color=FG, y=0.98)
        for ax in axs:
            ax.set_facecolor(BG)
            setup_axes(ax, lo, hi, args.elev, az)
        leg = fig.legend(handles=handles, loc="lower center",
                         ncol=min(6, max(1, len(handles))), fontsize=8,
                         frameon=False)
        for t in leg.get_texts():
            t.set_color(FG)
        fig.suptitle(
            f"{args.dataset}/{args.scene} · {len(pts)} pts encoded · "
            f"dim={args.ssp_dim} · base ls={base_ls:g} m · azim={az:g}°",
            fontsize=12, color=FG, y=0.99)
        fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.1,
                            wspace=-0.1)
        out = f"{prefix}_az{az:03.0f}.png"
        fig.savefig(out, dpi=args.dpi, facecolor=BG)
        plt.close(fig)
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
