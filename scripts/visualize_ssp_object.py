#!/usr/bin/env python
"""Encode one ModelNet10 object's (sparse) point cloud as a mean SSP and
visualize the recovered object as a 3D similarity field.

Figure layout (columns = viewing angles):
    row 0 : the sparse input point cloud
    row 1 : similarity of the mean SSP with a 3D grid, LARGE length scale
    row 2 : same, SMALL length scale

In the similarity rows each grid point's alpha is tied to its similarity, so
low-similarity space fades out and only the object cloud remains visible. A
large length scale gives a broad, blobby kernel (coarse object); a small one
gives a tight kernel that recovers finer structure.

Usage:
    python scripts/visualize_ssp_object.py --category chair --index 0
    python scripts/visualize_ssp_object.py --category chair --index 0 \
        --large-ls 0.4 --small-ls 0.08 --grid 32 --out ssp_object.png
"""

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib.colors import Normalize

from multiscalessps.data import ModelNet10PairDataset
from vsagym.spaces import HexagonalSSPSpace


def sim_field(points, grid_pts, length_scale, ssp_dim, seed=0):
    """Mean-SSP-vs-grid similarity for a given length scale (fresh SSP space)."""
    sp = HexagonalSSPSpace(domain_dim=3, ssp_dim=ssp_dim,
                           length_scale=length_scale, rng=seed)
    mean_ssp = sp.encode(points).mean(axis=0, keepdims=True)
    grid_ssps = sp.encode(grid_pts).astype(np.float32)
    return (grid_ssps @ mean_ssp.astype(np.float32).T).ravel()


# ---------------------------------------------------------------------------
# Trained per-voxel length-scale field (kernel-style phase weighting)
# ---------------------------------------------------------------------------

def make_ssp_space(base_ls, ssp_dim, seed=0):
    """SSP space plus the rfft scale-block index used to expand per-scale gains
    to the half spectrum (validated against the phase-matrix row norms)."""
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


def nearest_voxel(pts, vox_n, lim):
    """Flat index (row-major ix,iy,iz) of the vox_n^3 voxel nearest each point."""
    idx = np.rint((np.asarray(pts) + lim) / (2 * lim) * (vox_n - 1)).astype(int)
    idx = np.clip(idx, 0, vox_n - 1)
    return jnp.asarray((idx[:, 0] * vox_n + idx[:, 1]) * vox_n + idx[:, 2])


def train_voxel_lengthscale(sparse_pts, full_pts, sp, scale_idx, scales,
                            vox_n, lim, n_neg=3000, n_steps=300,
                            learning_rate=1e-2, init_ls=0.3, seed=0):
    """Learn one length scale per voxel so the modulated similarity is 1 at the
    full-cloud points and 0 at random points. The same per-voxel gain modulates
    both the mean (from the sparse points) and every query. Returns the learned
    ell field (n_vox,) and the loss history."""
    sparse_enc = jnp.asarray(sp.encode(sparse_pts), dtype=jnp.float32)

    rng = np.random.default_rng(seed)
    neg_pts = rng.uniform(-lim, lim, size=(n_neg, 3))
    query_pts = np.concatenate([full_pts, neg_pts], axis=0)
    query_enc = jnp.asarray(sp.encode(query_pts), dtype=jnp.float32)
    target = jnp.concatenate([jnp.ones(len(full_pts)), jnp.zeros(n_neg)])

    sparse_vox = nearest_voxel(sparse_pts, vox_n, lim)
    query_vox = nearest_voxel(query_pts, vox_n, lim)

    def gains(raw):
        ell = jax.nn.softplus(raw) + 1e-3
        return expand_gains(ls_to_gains(ell, scales), scale_idx)   # (n_vox, half)

    def loss_fn(params):
        g = gains(params["raw"])
        mod_mean = modulate(sparse_enc, g[sparse_vox]).mean(axis=0, keepdims=True)
        sims = (modulate(query_enc, g[query_vox]) @ mod_mean.T).ravel()
        return jnp.mean((sims - target) ** 2)

    raw0 = jnp.full((vox_n ** 3,), float(np.log(np.expm1(init_ls))))
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
          f"ell range [{ell.min():.3f}, {ell.max():.3f}]")
    return ell, losses


def voxel_sim_field(grid_pts, sparse_pts, sp, scale_idx, scales, ell, vox_n, lim,
                    chunk=20000):
    """Render the trained per-voxel-length-scale similarity over grid_pts."""
    g = expand_gains(ls_to_gains(jnp.asarray(ell, jnp.float32), scales), scale_idx)
    sparse_enc = jnp.asarray(sp.encode(sparse_pts), dtype=jnp.float32)
    sparse_vox = nearest_voxel(sparse_pts, vox_n, lim)
    mod_mean = modulate(sparse_enc, g[sparse_vox]).mean(axis=0, keepdims=True)

    grid_vox = nearest_voxel(grid_pts, vox_n, lim)
    out = np.empty(len(grid_pts), dtype=np.float32)
    for s in range(0, len(grid_pts), chunk):
        enc = jnp.asarray(sp.encode(grid_pts[s:s + chunk]), dtype=jnp.float32)
        mod = modulate(enc, g[grid_vox[s:s + chunk]])
        out[s:s + chunk] = np.asarray(mod @ mod_mean.T).ravel()
    return out


def sim_to_rgba(sims, cmap, thr=0.45, gamma=3.0):
    """Map similarities to RGBA with alpha rising from `thr` to 1, and a mask
    that drops sub-threshold points so only the object cloud shows."""
    s = (sims - sims.min()) / (np.ptp(sims) + 1e-12)
    alpha = np.clip(((s - thr) / (1.0 - thr)), 0.0, 1.0) ** gamma
    rgba = cmap(s)
    rgba[:, 3] = alpha
    return rgba, s > thr


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/ModelNet10")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--category", default="chair")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-points", type=int, default=1024,
                        help="Size of the sparse input cloud to encode.")
    parser.add_argument("--ssp-dim", type=int, default=1500)
    parser.add_argument("--large-ls", type=float, default=0.4)
    parser.add_argument("--small-ls", type=float, default=0.08)
    parser.add_argument("--grid", type=int, default=32,
                        help="Resolution per axis of the 3D similarity grid.")
    parser.add_argument("--angles", type=float, nargs="+", default=[30, 120, 210],
                        help="Azimuth angles (degrees), one per column.")
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--out", default="ssp_object.png")
    # Trained per-voxel length-scale field
    parser.add_argument("--vox-n", type=int, default=20,
                        help="Voxels per axis for the learned length-scale field.")
    parser.add_argument("--base-ls", type=float, default=None,
                        help="Base SSP length scale for modulation (default: small-ls).")
    parser.add_argument("--n-neg", type=int, default=3000,
                        help="Random negative (target-0) points for training.")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--init-ls", type=float, default=0.3)
    args = parser.parse_args()

    # Deterministic seed so the sparse cloud is fixed for the figure.
    ds = ModelNet10PairDataset(
        root=args.root, split=args.split, categories=[args.category],
        num_input_points=args.num_points, seed=0,
    )
    sample = ds[args.index]
    points = sample["points"].numpy()
    full_points = sample["full_points"].numpy()
    print(f"{sample['class_name']} (index {args.index}): {points.shape[0]} sparse points, "
          f"{full_points.shape[0]} full-cloud points")

    # 3D grid over the (unit-sphere-normalized) domain.
    lim = 1.1
    axis = np.linspace(-lim, lim, args.grid)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="ij")
    grid_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    print("computing similarity fields...")
    sims_large = sim_field(points, grid_pts, args.large_ls, args.ssp_dim)
    sims_small = sim_field(points, grid_pts, args.small_ls, args.ssp_dim)
    print(f"  large ls={args.large_ls}: sim range [{sims_large.min():.3f}, {sims_large.max():.3f}]")
    print(f"  small ls={args.small_ls}: sim range [{sims_small.min():.3f}, {sims_small.max():.3f}]")

    # Trained per-voxel length-scale field (kernel-style phase weighting).
    base_ls = args.small_ls if args.base_ls is None else args.base_ls
    print(f"training per-voxel length scale (vox_n={args.vox_n}, base_ls={base_ls})...")
    sp_mod, scale_idx, scales = make_ssp_space(base_ls, args.ssp_dim)
    ell, _ = train_voxel_lengthscale(
        points, full_points, sp_mod, scale_idx, scales,
        vox_n=args.vox_n, lim=lim, n_neg=args.n_neg, n_steps=args.steps,
        learning_rate=args.lr, init_ls=args.init_ls,
    )
    sims_voxel = voxel_sim_field(grid_pts, points, sp_mod, scale_idx, scales,
                                 ell, args.vox_n, lim)
    print(f"  trained voxel-ls: sim range [{sims_voxel.min():.3f}, {sims_voxel.max():.3f}]")

    cmap = plt.get_cmap("viridis")
    rgba_l, mask_l = sim_to_rgba(sims_large, cmap)
    rgba_s, mask_s = sim_to_rgba(sims_small, cmap)
    rgba_v, mask_v = sim_to_rgba(sims_voxel, cmap)

    # Learned length-scale field: same recovered points as the trained-sim row,
    # colored by each voxel's ell instead of by similarity.
    grid_vox = np.asarray(nearest_voxel(grid_pts, args.vox_n, lim))
    ell_grid = ell[grid_vox]
    ls_cmap = plt.get_cmap("plasma")
    ls_norm = Normalize(vmin=float(ell_grid[mask_v].min()),
                        vmax=float(ell_grid[mask_v].max()))
    rgba_ell = ls_cmap(ls_norm(ell_grid))
    rgba_ell[:, 3] = rgba_v[:, 3]        # reuse trained-sim alpha (only object shows)

    ncols = len(args.angles)
    fig, axs = plt.subplots(5, ncols, figsize=(4.2 * ncols, 20.5),
                            subplot_kw={"projection": "3d"})
    axs = np.atleast_2d(axs)

    rows = [
        ("point cloud", None, None, None),
        (f"similarity, large ls={args.large_ls}", grid_pts, rgba_l, mask_l),
        (f"similarity, small ls={args.small_ls}", grid_pts, rgba_s, mask_s),
        (f"trained per-voxel ls ({args.vox_n}³)", grid_pts, rgba_v, mask_v),
        (r"learned length scale $\ell(x)$", grid_pts, rgba_ell, mask_v),
    ]

    for r, (title, gpts, rgba, mask) in enumerate(rows):
        for c, azim in enumerate(args.angles):
            ax = axs[r, c]
            if gpts is None:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                           s=6, c="tab:blue")
            else:
                ax.scatter(gpts[mask, 0], gpts[mask, 1], gpts[mask, 2],
                           c=rgba[mask], s=8, edgecolors="none")
            ax.view_init(elev=args.elev, azim=azim)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_box_aspect([1, 1, 1])
            ax.set_axis_off()   # no grey panes, grid, or ticks -- clean
            if r == 0:
                ax.set_title(f"azim={azim:g}", fontsize=10)
            if c == 0:
                ax.text2D(-0.04, 0.5, title, transform=ax.transAxes,
                          rotation=90, va="center", ha="center", fontsize=11)

    # Colorbar for the learned length-scale row (values otherwise unreadable).
    sm = plt.cm.ScalarMappable(norm=ls_norm, cmap=ls_cmap)
    fig.colorbar(sm, ax=axs[4, -1], fraction=0.03, shrink=0.6,
                 label=r"length scale $\ell$")

    fig.suptitle(
        f"SSP encoding of {sample['class_name']} "
        f"(index {args.index}, {points.shape[0]} points)",
        fontsize=13,
    )
    # Halved column spacing (was tight_layout's default gap).
    fig.subplots_adjust(left=0.05, right=0.92, top=0.96, bottom=0.02,
                        wspace=-0.35, hspace=0.05)
    fig.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
