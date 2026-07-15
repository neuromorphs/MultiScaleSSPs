#!/usr/bin/env python
"""Encode an ETH ASL lidar scan as a mean SSP and visualize the recovered
scene as a 3D similarity field at several length scales.

Figure layout (columns = viewing angles):
    row 0   : the subsampled input point cloud
    row 1.. : similarity of the mean SSP with a 3D grid, one length scale
              per row (largest to smallest, in meters)
    then    : similarity with a trained per-voxel length scale (learned
              phase gain, as in visualize_ssp_object.py), and the learned
              length-scale field itself colored by ell(x)

In the similarity rows each grid point's alpha is tied to its similarity, so
low-similarity space fades out and only the scene structure remains visible.

Expects data downloaded by scripts/download_eth_asl.py; scans in
csv_global/ are already registered into a common world frame, so multiple
scans can be merged by passing several indices to --scans.

Usage:
    python scripts/visualize_ssp_lidar.py --sequence apartment --scans 0
    python scripts/visualize_ssp_lidar.py --sequence gazebo_summer \
        --scans 0 1 2 --length-scales 2.0 1.0 0.5 0.25 --out ssp_gazebo.png
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.colors import Normalize

from vsagym.spaces import HexagonalSSPSpace


def load_scans(seq_dir, scan_indices):
    """Load and merge csv_global scans (world frame) as an (N, 3) array."""
    clouds = []
    for i in scan_indices:
        path = seq_dir / "csv_global" / f"PointCloud{i}.csv"
        if not path.exists():
            raise SystemExit(f"{path} not found -- run scripts/download_eth_asl.py")
        # columns: Time_in_sec,x,y,z,Intensities,2DscanId,Points
        clouds.append(np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1, 2, 3)))
    return np.concatenate(clouds, axis=0)


def voxel_downsample(cloud, voxel):
    """Keep one point per occupied voxel to flatten the strong near-sensor
    density bias of lidar scans (the mean SSP otherwise just encodes density)."""
    idx = np.floor((cloud - cloud.min(axis=0)) / voxel).astype(np.int64)
    _, first = np.unique(idx, axis=0, return_index=True)
    return cloud[first]


def sim_field(points, grid_pts, length_scale, ssp_dim, seed=0, chunk=20000):
    """Mean-SSP-vs-grid similarity for a given length scale (fresh SSP space)."""
    sp = HexagonalSSPSpace(domain_dim=3, ssp_dim=ssp_dim,
                           length_scale=length_scale, rng=seed)
    mean_ssp = sp.encode(points).mean(axis=0, keepdims=True).astype(np.float32)
    sims = np.empty(len(grid_pts), dtype=np.float32)
    for s in range(0, len(grid_pts), chunk):
        grid_ssps = sp.encode(grid_pts[s:s + chunk]).astype(np.float32)
        sims[s:s + chunk] = (grid_ssps @ mean_ssp.T).ravel()
    return sims


# ---------------------------------------------------------------------------
# Trained per-voxel length-scale field (kernel-style phase weighting), as in
# visualize_ssp_object.py but over the scene's non-cubic bounding box.
# ---------------------------------------------------------------------------

def make_ssp_space(base_ls, ssp_dim, seed=0):
    """SSP space plus the rfft scale-block index used to expand per-scale gains
    to the half spectrum."""
    sp = HexagonalSSPSpace(domain_dim=3, ssp_dim=ssp_dim,
                           length_scale=base_ls, rng=seed)
    blk = sp.domain_dim + 1
    scale_idx = np.tile(np.repeat(np.arange(sp.n_scales), blk), sp.n_rotates)
    scales = np.asarray(sp.scales, dtype=np.float32)          # (n_scales,)
    return sp, jnp.asarray(scale_idx), jnp.asarray(scales)


def expand_gains(g_scales, scale_idx):
    """Per-scale gains (..., n_scales) -> rfft half-spectrum (..., d//2+1),
    block-constant across scales with the DC gain fixed at 1."""
    dc = jnp.ones((*g_scales.shape[:-1], 1))
    return jnp.concatenate([dc, g_scales[..., scale_idx]], axis=-1)


def modulate(ssps, g):
    """Apply real rfft gains g to encoded SSPs, then renormalize to unit norm so
    the gain reshapes phase/direction (the kernel), never magnitude."""
    mod = jnp.fft.irfft(g * jnp.fft.rfft(ssps, axis=-1), n=ssps.shape[-1], axis=-1)
    return mod / jnp.linalg.norm(mod, axis=-1, keepdims=True)


def ls_to_gains(ell, scales):
    """Length scale -> per-scale Gaussian low-pass gain g_k = exp(-1/2 ell^2 s_k^2).
    ell: (n_vox,); scales: (n_scales,) -> (n_vox, n_scales)."""
    return jnp.exp(-0.5 * (ell[:, None] ** 2) * (scales[None, :] ** 2))


def nearest_voxel(pts, lo, hi, counts):
    """Flat index (row-major ix,iy,iz) of the voxel nearest each point, over a
    counts[0] x counts[1] x counts[2] grid spanning the bounding box."""
    frac = (np.asarray(pts) - lo) / (hi - lo)
    idx = np.rint(frac * (counts - 1)).astype(int)
    idx = np.clip(idx, 0, counts - 1)
    return jnp.asarray((idx[:, 0] * counts[1] + idx[:, 1]) * counts[2] + idx[:, 2])


def train_voxel_lengthscale(sparse_pts, full_pts, sp, scale_idx, scales,
                            lo, hi, counts, n_neg=5000, n_steps=300,
                            learning_rate=1e-2, init_ls=1.0, seed=0):
    """Learn one length scale per voxel so the modulated similarity is 1 at the
    full-cloud points and 0 at random points. The same per-voxel gain modulates
    both the mean (from the sparse points) and every query. Returns the learned
    ell field (n_vox,) and the loss history."""
    sparse_enc = jnp.asarray(sp.encode(sparse_pts), dtype=jnp.float32)

    rng = np.random.default_rng(seed)
    neg_pts = rng.uniform(lo, hi, size=(n_neg, 3))
    query_pts = np.concatenate([full_pts, neg_pts], axis=0)
    query_enc = jnp.asarray(sp.encode(query_pts), dtype=jnp.float32)
    target = jnp.concatenate([jnp.ones(len(full_pts)), jnp.zeros(n_neg)])

    sparse_vox = nearest_voxel(sparse_pts, lo, hi, counts)
    query_vox = nearest_voxel(query_pts, lo, hi, counts)

    def gains(raw):
        ell = jax.nn.softplus(raw) + 1e-3
        return expand_gains(ls_to_gains(ell, scales), scale_idx)   # (n_vox, half)

    def loss_fn(params):
        g = gains(params["raw"])
        mod_mean = modulate(sparse_enc, g[sparse_vox]).mean(axis=0, keepdims=True)
        sims = (modulate(query_enc, g[query_vox]) @ mod_mean.T).ravel()
        return jnp.mean((sims - target) ** 2)

    raw0 = jnp.full((int(np.prod(counts)),), float(np.log(np.expm1(init_ls))))
    params = {"raw": raw0}
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = np.empty(n_steps)
    for i in range(n_steps):
        params, opt_state, loss = step(params, opt_state)
        losses[i] = loss
    ell = np.asarray(jax.nn.softplus(params["raw"]) + 1e-3)
    print(f"[voxel-ls] loss {losses[0]:.4f} -> {losses[-1]:.5f}  "
          f"ell range [{ell.min():.3f}, {ell.max():.3f}] m")
    return ell, losses


def voxel_sim_field(grid_pts, sparse_pts, sp, scale_idx, scales, ell,
                    lo, hi, counts, chunk=20000):
    """Render the trained per-voxel-length-scale similarity over grid_pts."""
    g = expand_gains(ls_to_gains(jnp.asarray(ell, jnp.float32), scales), scale_idx)
    sparse_enc = jnp.asarray(sp.encode(sparse_pts), dtype=jnp.float32)
    sparse_vox = nearest_voxel(sparse_pts, lo, hi, counts)
    mod_mean = modulate(sparse_enc, g[sparse_vox]).mean(axis=0, keepdims=True)

    grid_vox = nearest_voxel(grid_pts, lo, hi, counts)
    out = np.empty(len(grid_pts), dtype=np.float32)
    for s in range(0, len(grid_pts), chunk):
        enc = jnp.asarray(sp.encode(grid_pts[s:s + chunk]), dtype=jnp.float32)
        mod = modulate(enc, g[grid_vox[s:s + chunk]])
        out[s:s + chunk] = np.asarray(mod @ mod_mean.T).ravel()
    return out


def make_grid(lo, hi, n_longest):
    """Grid over the bounding box with ~cubic voxels: n_longest points along
    the longest axis, proportionally fewer along the others."""
    extent = hi - lo
    counts = np.maximum(4, np.rint(n_longest * extent / extent.max())).astype(int)
    axes = [np.linspace(lo[d], hi[d], counts[d]) for d in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]), counts


def sim_to_rgba(sims, cmap, thr=0.25, gamma=2.0):
    """Map similarities to RGBA with alpha rising from `thr` to 1, and a mask
    that drops sub-threshold points so only the scene shows. Normalizing by
    the positive max (not min-max) keeps the noise floor at ~0 alpha even
    when the whole similarity range is small."""
    s = np.clip(sims / (sims.max() + 1e-12), 0.0, 1.0)
    alpha = np.clip(((s - thr) / (1.0 - thr)), 0.0, 1.0) ** gamma
    rgba = cmap(s)
    rgba[:, 3] = alpha
    return rgba, s > thr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/eth_asl"))
    parser.add_argument("--sequence", default="apartment")
    parser.add_argument("--scans", type=int, nargs="+", default=[0],
                        help="csv_global PointCloud indices to merge.")
    parser.add_argument("--num-points", type=int, default=10000,
                        help="Subsample size of the cloud that gets encoded.")
    parser.add_argument("--voxel-size", type=float, default=0.1,
                        help="Voxel size (m) for density-equalizing downsample "
                             "before the random subsample; 0 disables it.")
    parser.add_argument("--ssp-dim", type=int, default=1500)
    parser.add_argument("--length-scales", type=float, nargs="+",
                        default=[2.0, 1.0, 0.5, 0.25],
                        help="One similarity row per length scale (meters).")
    parser.add_argument("--grid", type=int, default=64,
                        help="Grid resolution along the longest scene axis.")
    parser.add_argument("--angles", type=float, nargs="+", default=[30, 120, 210],
                        help="Azimuth angles (degrees), one per column.")
    parser.add_argument("--elev", type=float, default=30.0)
    parser.add_argument("--thr", type=float, default=0.25,
                        help="Normalized-similarity threshold below which grid "
                             "points are hidden.")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="Exponent shaping alpha above the threshold.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="ssp_lidar.png")
    parser.add_argument("--dpi", type=int, default=250)
    # Trained per-voxel length-scale field
    parser.add_argument("--vox-grid", type=int, default=24,
                        help="Voxels along the longest axis for the learned "
                             "length-scale field.")
    parser.add_argument("--base-ls", type=float, default=None,
                        help="Base SSP length scale (m) for modulation "
                             "(default: smallest of --length-scales).")
    parser.add_argument("--max-pos", type=int, default=20000,
                        help="Max positive (target-1) points for training.")
    parser.add_argument("--n-neg", type=int, default=5000,
                        help="Random negative (target-0) points for training.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--init-ls", type=float, default=1.0,
                        help="Initial length scale (m) for training.")
    args = parser.parse_args()

    seq_dir = args.data_dir / args.sequence
    cloud = load_scans(seq_dir, args.scans)
    cloud -= cloud.mean(axis=0)                     # center; keep meters
    print(f"{args.sequence} scans {args.scans}: {len(cloud)} points, "
          f"extent {np.ptp(cloud, axis=0).round(1)} m")

    if args.voxel_size > 0:
        cloud = voxel_downsample(cloud, args.voxel_size)
        print(f"voxel downsample ({args.voxel_size} m): {len(cloud)} points")
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(cloud), size=min(args.num_points, len(cloud)),
                     replace=False)
    points = cloud[idx]

    pad = max(args.length_scales) / 2
    lo, hi = points.min(axis=0) - pad, points.max(axis=0) + pad
    grid_pts, counts = make_grid(lo, hi, args.grid)
    print(f"grid {counts[0]}x{counts[1]}x{counts[2]} = {len(grid_pts)} points")

    cmap = plt.get_cmap("viridis")
    sim_rows = []
    for ls in args.length_scales:
        sims = sim_field(points, grid_pts, ls, args.ssp_dim, seed=args.seed)
        print(f"  ls={ls}: sim range [{sims.min():.3f}, {sims.max():.3f}]")
        rgba, mask = sim_to_rgba(sims, cmap, thr=args.thr, gamma=args.gamma)
        sim_rows.append((ls, rgba, mask))

    # Trained per-voxel length-scale field (learned phase gain). Positives are
    # the (density-equalized) cloud beyond the encoded subsample; negatives are
    # uniform in the bounding box.
    base_ls = min(args.length_scales) if args.base_ls is None else args.base_ls
    extent = hi - lo
    vox_counts = np.maximum(2, np.rint(args.vox_grid * extent / extent.max())
                            ).astype(int)
    full_pts = cloud[rng.choice(len(cloud), size=min(args.max_pos, len(cloud)),
                                replace=False)]
    print(f"training per-voxel length scale "
          f"(vox {vox_counts[0]}x{vox_counts[1]}x{vox_counts[2]}, "
          f"base_ls={base_ls}, {len(full_pts)} positives)...")
    sp_mod, scale_idx, scales = make_ssp_space(base_ls, args.ssp_dim,
                                               seed=args.seed)
    ell, _ = train_voxel_lengthscale(
        points, full_pts, sp_mod, scale_idx, scales, lo, hi, vox_counts,
        n_neg=args.n_neg, n_steps=args.steps, learning_rate=args.lr,
        init_ls=args.init_ls, seed=args.seed,
    )
    sims_voxel = voxel_sim_field(grid_pts, points, sp_mod, scale_idx, scales,
                                 ell, lo, hi, vox_counts)
    print(f"  trained voxel-ls: sim range "
          f"[{sims_voxel.min():.3f}, {sims_voxel.max():.3f}]")
    rgba_v, mask_v = sim_to_rgba(sims_voxel, cmap, thr=args.thr,
                                 gamma=args.gamma)

    # Learned length-scale field: same recovered points as the trained row,
    # colored by each voxel's ell instead of by similarity.
    grid_vox = np.asarray(nearest_voxel(grid_pts, lo, hi, vox_counts))
    ell_grid = ell[grid_vox]
    ls_cmap = plt.get_cmap("plasma")
    ls_norm = Normalize(vmin=float(ell_grid[mask_v].min()),
                        vmax=float(ell_grid[mask_v].max()))
    rgba_ell = ls_cmap(ls_norm(ell_grid))
    rgba_ell[:, 3] = rgba_v[:, 3]        # reuse trained-sim alpha

    rows = [("input cloud", None, None)]
    rows += [(f"ls={ls:g} m", rgba, mask) for ls, rgba, mask in sim_rows]
    rows += [("trained vox ls", rgba_v, mask_v),
             (r"learned $\ell(x)$", rgba_ell, mask_v)]

    nrows, ncols = len(rows), len(args.angles)
    # scale panel height to the scene's aspect so flat scenes don't waste space
    row_h = 4.2 * max(0.45, float(extent[2] / extent.max()))
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, row_h * nrows),
                            subplot_kw={"projection": "3d"})
    axs = np.atleast_2d(axs)

    row_titles = [title for title, _, _ in rows]
    for r, (_, rgba, mask) in enumerate(rows):
        for c, azim in enumerate(args.angles):
            ax = axs[r, c]
            if rgba is None:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           s=1.5, c="tab:blue")
            else:
                ax.scatter(grid_pts[mask, 0], grid_pts[mask, 1],
                           grid_pts[mask, 2], c=rgba[mask], s=6,
                           edgecolors="none")
            ax.view_init(elev=args.elev, azim=azim)
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_zlim(lo[2], hi[2])
            ax.set_box_aspect(extent)   # true meter proportions
            ax.set_axis_off()
            if r == 0:
                ax.set_title(f"azim={azim:g}", fontsize=10)

    fig.suptitle(
        f"SSP encoding of {args.sequence} scan(s) {args.scans} "
        f"(dim={args.ssp_dim}, {len(points)} points)",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.02,
                        wspace=-0.2, hspace=-0.35)
    # Row labels at each row's true center (axes overlap with negative hspace,
    # so per-axes text2D would collide).
    for r, title in enumerate(row_titles):
        pos = axs[r, 0].get_position()
        fig.text(0.02, (pos.y0 + pos.y1) / 2, title, rotation=90,
                 va="center", ha="center", fontsize=11)

    # Colorbar for the learned length-scale row.
    sm = plt.cm.ScalarMappable(norm=ls_norm, cmap=ls_cmap)
    fig.colorbar(sm, ax=axs[-1, -1], fraction=0.03, shrink=0.6,
                 label=r"length scale $\ell$ (m)")
    fig.savefig(args.out, dpi=args.dpi)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
