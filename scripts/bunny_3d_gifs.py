#!/usr/bin/env python
"""3D renders of the bunny bundle similarity volume (bunny_kernel_fit.py).

Modes (default: all):
    cloud   -- static 3D scatter of the encoded 2048-point cloud
    volume  -- rotating camera around the similarity volume, drawn as an
               alpha-blended 3D scatter (alpha ~ sim^2): initial | learned
    scan3d  -- learned volume (faint) with a translucent slice plane
               (alpha ~ sim) sweeping through while the camera rotates

Run scripts/bunny_kernel_fit.py first (needs its params.msgpack).

    python scripts/bunny_3d_gifs.py [cloud volume scan3d]

Plot axes are (x, z, y) so the bunny stands upright; the scan plane sweeps
the mesh's z (front-back) axis, matching bunny_scan_gif.py.
"""

import argparse
from pathlib import Path as FSPath

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import jax
import jax.numpy as jnp
from flax import serialization

from bunny_kernel_fit import (PhaseNet, load_cloud, make_pool, make_sim_fn,
                              eval_chunked)
from ssp_encoding_animations import COLORS, frame_of, write_outputs
from vsagym.spaces import HexagonalSSPSpace

LIM = 0.9   # plot/volume extent


def style3d(ax):
    ax.set_facecolor(COLORS["background"])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_color(COLORS["frame_edge"])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_box_aspect((1, 1, 1))
    ax.grid(False)


def setup(args):
    cloud, dense = load_cloud(args.mesh, args.n_cloud, args.n_dense, args.seed)
    ssp_space = HexagonalSSPSpace(domain_dim=3, n_scales=args.n_scales,
                                  n_rotates=args.n_rotates,
                                  length_scale=args.length_scale, rng=args.seed)
    d = ssp_space.ssp_dim
    nf = (d - 1) // 2
    a_base = jnp.asarray(ssp_space.phase_matrix[1:nf + 1]
                         / ssp_space.length_scale.flatten(), jnp.float32)
    model = PhaseNet(hidden=tuple(args.hidden), n_free=nf, dim=3, w0=args.w0)
    net0 = model.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, 3)))
    params0 = {"net": net0, "alpha": jnp.array(1.0), "beta": jnp.array(0.0)}
    params = serialization.from_bytes(params0, FSPath(args.params).read_bytes())
    sim_fn = make_sim_fn(model, a_base, d, args.dphase_scale,
                         jnp.asarray(cloud))
    # baseline affine readout, fit once on held-out points
    tree = cKDTree(dense)
    eval_pts, eval_t = make_pool(tree, dense, args.sigma, 100000,
                                 np.random.default_rng(args.seed + 99))
    raw = eval_chunked(sim_fn, params0, eval_pts)
    (al, be), *_ = np.linalg.lstsq(
        np.column_stack([raw, np.ones_like(raw)]), eval_t, rcond=None)
    return cloud, sim_fn, params0, params, (al, be)


# ---------------------------------------------------------------------------


def render_cloud(cloud, args, out_dir):
    fig = plt.figure(figsize=(10.4, 5.4), facecolor=COLORS["background"])
    for i, azim in enumerate((-55, 25)):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        style3d(ax)
        ax.scatter(cloud[:, 0], cloud[:, 2], cloud[:, 1], s=3,
                   color=COLORS["arrow"], alpha=0.75, edgecolors="none")
        ax.view_init(elev=14, azim=azim)
    fig.suptitle(f"encoded point cloud ({len(cloud)} points)", fontsize=12)
    fig.tight_layout()
    path = out_dir / "bunny_cloud.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def volume_values(sim_fn, params, res, affine=None, chunk_print=""):
    cc = np.linspace(-LIM, LIM, res).astype(np.float32)
    GX, GY, GZ = np.meshgrid(cc, cc, cc, indexing="ij")
    pts = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])
    v = eval_chunked(sim_fn, params, pts)
    if affine is not None:
        v = affine[0] * v + affine[1]
    print(f"  volume{chunk_print}: {res}^3 evaluated")
    return pts, np.clip(v, 0, 1)


def volume_scatter(ax, pts, v, thr, max_pts, rng, jitter=0.0):
    """Alpha-blended scatter standing in for a 3D pcolormesh. Jitter breaks
    up the regular voxel grid so it does not render as stripes."""
    sel = np.flatnonzero(v > thr)
    if len(sel) > max_pts:
        sel = rng.choice(sel, max_pts, replace=False)
    vv = v[sel]
    p = pts[sel] + rng.uniform(-jitter, jitter, (len(sel), 3))
    colors = plt.get_cmap(COLORS["sim_cmap"])(vv)
    colors[:, 3] = 0.02 + 0.6 * vv ** 2
    ax.scatter(p[:, 0], p[:, 2], p[:, 1], c=colors, s=3,
               marker=".", edgecolors="none", depthshade=False)
    return len(sel)


def render_volume(cloud, sim_fn, params0, params, affine, args, out_dir):
    rng = np.random.default_rng(args.seed)
    pts, v0 = volume_values(sim_fn, params0, args.vol_res, affine, " initial")
    _, v1 = volume_values(sim_fn, params, args.vol_res, None, " learned")

    fig = plt.figure(figsize=(10.8, 5.6), dpi=120,
                     facecolor=COLORS["background"])
    axes = []
    for i, (v, title) in enumerate([(v0, "initial: HexSSP bundle"),
                                    (v1, "learned $A(x)$ bundle")]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        style3d(ax)
        n = volume_scatter(ax, pts, v, args.thr, args.max_pts, rng,
                           jitter=LIM / args.vol_res)
        ax.set_title(title, fontsize=12)
        print(f"  {title}: {n} voxels drawn")
        axes.append(ax)
    fig.tight_layout(pad=0.2)

    def frames():
        for f in range(args.frames):
            for ax in axes:
                ax.view_init(elev=16, azim=360 * f / args.frames)
            yield frame_of(fig)

    write_outputs(frames(), args.frames, out_dir / "bunny_volume", args)
    plt.close(fig)


def render_scan3d(cloud, sim_fn, params, args, out_dir):
    rng = np.random.default_rng(args.seed)
    pts, v1 = volume_values(sim_fn, params, args.vol_res, None, " learned")

    # slice planes: sim on a vertical plane at bunny-z, alpha ~ sim
    res = args.plane_res
    cc = np.linspace(-LIM, LIM, res).astype(np.float32)
    PX, PY = np.meshgrid(cc, cc)                       # bunny x, y
    zs = np.linspace(cloud[:, 2].min() - 0.06, cloud[:, 2].max() + 0.06,
                     args.n_slices)
    cmap = plt.get_cmap(COLORS["sim_cmap"])
    planes = []
    for z in zs:
        p = np.column_stack([PX.ravel(), PY.ravel(),
                             np.full(PX.size, z, np.float32)])
        sv = np.clip(eval_chunked(sim_fn, params, p), 0, 1).reshape(res, res)
        rgba = cmap(sv)
        rgba[..., 3] = 0.15 + 0.8 * sv                 # low sim -> see-through
        planes.append(rgba)
    print(f"  {len(planes)} slice planes evaluated")

    fig = plt.figure(figsize=(6.2, 6.2), dpi=120,
                     facecolor=COLORS["background"])
    ax = fig.add_subplot(projection="3d")
    style3d(ax)
    volume_scatter(ax, pts, 0.55 * v1, args.thr, args.max_pts, rng,
                   jitter=LIM / args.vol_res)          # faint context volume
    surf = None
    ztxt = ax.text2D(0.04, 0.94, "", transform=ax.transAxes, fontsize=11,
                     color="0.35")

    order = list(range(len(zs))) + list(range(len(zs) - 2, 0, -1))

    def frames():
        nonlocal surf
        for f, i in enumerate(order):
            if surf is not None:
                surf.remove()
            surf = ax.plot_surface(PX, np.full_like(PX, zs[i]), PY,
                                   facecolors=planes[i], rstride=1, cstride=1,
                                   shade=False, linewidth=0)
            ztxt.set_text(f"z = {zs[i]:+.2f}")
            ax.view_init(elev=16, azim=360 * f / len(order))
            yield frame_of(fig)

    write_outputs(frames(), len(order), out_dir / "bunny_scan3d", args)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modes", nargs="*",
                    choices=["cloud", "volume", "scan3d"])
    ap.add_argument("--mesh", type=str, default="data/meshes/bunny.ply")
    ap.add_argument("--n-cloud", type=int, default=2048)
    ap.add_argument("--n-dense", type=int, default=200000)
    ap.add_argument("--sigma", type=float, default=0.04)
    ap.add_argument("--n-scales", type=int, default=6)
    ap.add_argument("--n-rotates", type=int, default=8)
    ap.add_argument("--length-scale", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--w0", type=float, default=15.0)
    ap.add_argument("--dphase-scale", type=float, default=30.0)
    ap.add_argument("--params", type=str,
                    default="results/bunny_kernel_fit/params.msgpack")
    ap.add_argument("--vol-res", type=int, default=64)
    ap.add_argument("--plane-res", type=int, default=110)
    ap.add_argument("--n-slices", type=int, default=60)
    ap.add_argument("--thr", type=float, default=0.15,
                    help="volume scatter: drop voxels below this sim")
    ap.add_argument("--max-pts", type=int, default=45000,
                    help="volume scatter: max voxels drawn per panel")
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif-stride", type=int, default=2)
    ap.add_argument("--gif-scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/animations")
    args = ap.parse_args()
    args.modes = args.modes or ["cloud", "volume", "scan3d"]

    out = FSPath(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cloud, sim_fn, params0, params, affine = setup(args)

    if "cloud" in args.modes:
        render_cloud(cloud, args, FSPath("results/bunny_kernel_fit"))
    if "volume" in args.modes:
        render_volume(cloud, sim_fn, params0, params, affine, args, out)
    if "scan3d" in args.modes:
        render_scan3d(cloud, sim_fn, params, args, out)


if __name__ == "__main__":
    main()
