#!/usr/bin/env python
"""Static check for location-dependent SSP scale modulation, per quadrant.

Uses the blob world of scripts/ssp_modulation_animation.py (UL many large
blobs, UR many small dense, LL few large, LR few small). Each blob contributes
a fixed number of interior feature points regardless of its size, so large
blobs are sparsely covered and small ones densely -- that sparsity is what
should make different scale profiles optimal in different quadrants.

Each quadrant's blobs are encoded as the mean of the (modulated) SSP
embeddings of their feature points; the similarity of that one vector with a
grid of location encodings should recover the quadrant's blob indicator.
Three modulations are compared:

  none      -- unmodulated encodings (g = 1)
  global    -- ONE learned non-negative n_scales gain vector shared by all
               quadrants (can reshape the kernel, but not per location)
  quadrant  -- a separate learned gain vector per quadrant (the static stand-in
               for a location-conditioned modulation map)

Gains are optimized directly (no MLP) so the static setup isolates the
question: does per-region scale selection help, and do the learned profiles
differ across quadrants? Feature points are resampled every training step and
held-out points are used for evaluation. Outputs a figure (similarity maps per
quadrant x modulation + learned gain profiles) and printed metrics.

Example:
    python scripts/quadrant_scale_modulation.py --out results/scale_modulation/quadrant_static.png
"""

import argparse
import sys
import time
from pathlib import Path as FSPath

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import jax
import jax.numpy as jnp
import optax
from vsagym.spaces import HexagonalSSPSpace

sys.path.insert(0, str(FSPath(__file__).parent))
from shape_scale_modulation import (  # noqa: E402
    SOFTPLUS_INV_1, expand_gains, make_scale_index, metrics_from_simmap,
    modulate_spectral,
)
from ssp_modulation_animation import (  # noqa: E402
    WORLD, inside_blobs, make_blobs, sample_blob_points,
)

QUADS = {  # name: (x0, x1, y0, y1)
    "UL: many large": (0.0, 5.0, 5.0, 10.0),
    "UR: many small": (5.0, 10.0, 5.0, 10.0),
    "LL: few large": (0.0, 5.0, 0.0, 5.0),
    "LR: few small": (5.0, 10.0, 0.0, 5.0),
}


def quad_blobs(blobs, bounds):
    x0, x1, y0, y1 = bounds
    keep = (blobs[:, 0] >= x0) & (blobs[:, 0] < x1) & (blobs[:, 1] >= y0) & (blobs[:, 1] < y1)
    return blobs[keep]


def quad_grid(bounds, n):
    x0, x1, y0, y1 = bounds
    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    return X, Y, np.column_stack([X.ravel(), Y.ravel()])


def gains_from_raw(raw):
    """Non-negative gains, identity (1) at raw = 0."""
    return jax.nn.softplus(raw + SOFTPLUS_INV_1)


class QuadrantData:
    """Per-quadrant constants: grid encodings, normalized target, blobs."""

    def __init__(self, name, bounds, blobs, ssp_space, grid_n):
        self.name, self.bounds = name, bounds
        self.blobs = quad_blobs(blobs, bounds)
        self.X, self.Y, grid_pts = quad_grid(bounds, grid_n)
        self.grid_ssps = jnp.asarray(ssp_space.encode(grid_pts), dtype=jnp.float32)
        t = inside_blobs(grid_pts, self.blobs).astype(np.float32)
        self.target = jnp.asarray(t / np.linalg.norm(t))
        self.inside = t.astype(bool)


def sims_of(g, grid_ssps, pt_ssps, scale_idx_j):
    """Similarity map of the modulated mean encoding of pt_ssps (one shared
    gain vector g) against the modulated grid encodings."""
    spec = expand_gains(g, scale_idx_j)
    mean = modulate_spectral(pt_ssps, spec).mean(axis=0)
    return modulate_spectral(grid_ssps, spec) @ mean


def train_gains(quads, ssp_space, scale_idx_j, args, seed, shared: bool):
    """Optimize raw gain params. shared=True -> one gain vector for all
    quadrants; shared=False -> one per quadrant. Returns (n_quads, n_scales)
    gains (rows identical when shared)."""
    rng = np.random.default_rng(seed)
    n_sc = ssp_space.n_scales
    n_g = 1 if shared else len(quads)
    params = jnp.zeros((n_g, n_sc))
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    # coarse bias weights: penalize fine scales more (as in the notebook)
    w_coarse = jnp.asarray(np.asarray(ssp_space.scales) / np.max(ssp_space.scales))

    def loss_fn(params, pt_ssps_all):
        loss = 0.0
        for qi, q in enumerate(quads):
            g = gains_from_raw(params[0 if shared else qi])
            s = sims_of(g, q.grid_ssps, pt_ssps_all[qi], scale_idx_j)
            if args.loss == "cosine":
                # scale-invariant: match the sim map's shape, not its amplitude
                fit = 1.0 - jnp.dot(s / (jnp.linalg.norm(s) + 1e-8), q.target)
            else:
                fit = jnp.sum((s - q.target) ** 2)
            reg = (args.reg_w * jnp.sum(jnp.abs(g))            # sparsity: few scales
                   + args.coarse_w * jnp.sum(w_coarse * g ** 2))  # slight coarse bias
            loss += fit + reg
        return loss / len(quads)

    @jax.jit
    def step(params, opt_state, *pt_ssps_all):
        loss, grads = jax.value_and_grad(loss_fn)(params, pt_ssps_all)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = np.empty(args.steps)
    for i in range(args.steps):
        if args.encode == "centers":
            # one encoding per blob at its center: the learned kernel width
            # alone must account for each blob's spatial extent
            pts_all = [q.blobs[:, :2] for q in quads]
        else:
            pts_all = [sample_blob_points(q.blobs, rng, args.pts_per_blob) for q in quads]
        pt_ssps = [jnp.asarray(ssp_space.encode(p), dtype=jnp.float32) for p in pts_all]
        params, opt_state, loss = step(params, opt_state, *pt_ssps)
        losses[i] = float(loss)
        if i % 200 == 0 or i == args.steps - 1:
            print(f"  [{'global' if shared else 'quadrant'}] step {i:>4}: loss {losses[i]:.4f}")
    g = np.asarray(gains_from_raw(params))
    return (np.repeat(g, len(quads), axis=0) if shared else g), losses


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scales", type=int, default=10)
    ap.add_argument("--n-rotates", type=int, default=33,
                    help="more rotations -> higher ssp_dim (= 6*n_scales*n_rotates + 1) -> "
                         "less bundling crosstalk when many points share one vector")
    ap.add_argument("--scale-min", type=float, default=1.0)
    ap.add_argument("--scale-max", type=float, default=10.0)
    ap.add_argument("--scale-sampling", type=str, default="log", choices=["lin", "log", "rand"])
    ap.add_argument("--ls", type=float, default=0.9,
                    help="base length scale; kernel widths span ls/scale_max .. ls/scale_min, "
                         "which should bracket the blob radii")
    ap.add_argument("--encode", choices=["samples", "centers"], default="centers",
                    help="'samples': mean over interior feature points; 'centers': one "
                         "encoding per blob at its center, so the learned kernel width "
                         "alone must express each blob's size")
    ap.add_argument("--pts-per-blob", type=int, default=30)
    ap.add_argument("--steps", type=int, default=1500)  # centers+cosine trains in ~20 s
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--loss", choices=["sse", "cosine"], default="cosine",
                    help="'sse': notebook-style squared error on the sim map; 'cosine': "
                         "scale-invariant shape match (aligned with the eval metric)")
    ap.add_argument("--reg-w", type=float, default=0.003, help="L1 on the gains (sparsity)")
    ap.add_argument("--coarse-w", type=float, default=0.01,
                    help="scale-weighted quadratic penalty biasing toward coarse scales")
    ap.add_argument("--grid-n", type=int, default=50, help="per-quadrant grid resolution")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=str, default="results/scale_modulation/quadrant_static.png")
    args = ap.parse_args()

    out_path = FSPath(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blobs = make_blobs(np.random.default_rng(args.seed))
    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=args.n_scales, n_rotates=args.n_rotates,
                                  scale_min=args.scale_min, scale_max=args.scale_max,
                                  scale_sampling=args.scale_sampling,
                                  length_scale=args.ls, rng=0)
    scale_idx_j = jnp.asarray(make_scale_index(ssp_space))
    n_sc = ssp_space.n_scales
    print(f"ssp_dim={ssp_space.ssp_dim} (n_scales={n_sc}, n_rotates={ssp_space.n_rotates}), "
          f"ls={args.ls}, kernel widths {args.ls/args.scale_max:.2f}..{args.ls/args.scale_min:.2f}, "
          f"{len(blobs)} blobs, {args.pts_per_blob} pts/blob")

    quads = [QuadrantData(name, b, blobs, ssp_space, args.grid_n) for name, b in QUADS.items()]

    t0 = time.time()
    g_global, losses_global = train_gains(quads, ssp_space, scale_idx_j, args, args.seed, shared=True)
    g_quad, losses_quad = train_gains(quads, ssp_space, scale_idx_j, args, args.seed, shared=False)
    print(f"trained in {time.time() - t0:.0f}s")

    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor="white")
    ax.plot(losses_global, label="global", color="0.6")
    ax.plot(losses_quad, label="quadrant", color="tab:blue")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (mean over quadrants)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    loss_path = out_path.with_name(out_path.stem + "_losses.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"wrote {loss_path}")

    # held-out evaluation points
    eval_rng = np.random.default_rng(2024)
    variants = [("none", np.ones((len(quads), n_sc))), ("global", g_global), ("quadrant", g_quad)]
    all_metrics, sim_maps, eval_pts = {}, {}, {}
    for qi, q in enumerate(quads):
        pts = (q.blobs[:, :2] if args.encode == "centers"
               else sample_blob_points(q.blobs, eval_rng, args.pts_per_blob))
        eval_pts[q.name] = pts
        pt_ssps = jnp.asarray(ssp_space.encode(pts), dtype=jnp.float32)
        for vname, g in variants:
            s = np.asarray(sims_of(jnp.asarray(g[qi]), q.grid_ssps, pt_ssps, scale_idx_j))
            m = metrics_from_simmap(s, q.inside)
            all_metrics[(q.name, vname)] = m
            sim_maps[(q.name, vname)] = s
            print(f"{q.name:<16} {vname:<9} cos={m['cosine']:.3f} auc={m['auc']:.3f} iou={m['best_iou']:.3f}")

    # figure: rows = none / global / quadrant sims + gain profiles, cols = quadrants
    fig, axs = plt.subplots(4, len(quads), figsize=(4.1 * len(quads), 14.6), facecolor="white")
    for qi, q in enumerate(quads):
        for row, (vname, _) in enumerate(variants):
            ax = axs[row, qi]
            m = all_metrics[(q.name, vname)]
            im = ax.pcolormesh(q.X, q.Y, sim_maps[(q.name, vname)].reshape(q.X.shape),
                               cmap="viridis", shading="auto")
            for cx, cy, r in q.blobs:
                ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor="white", lw=1.0))
            ax.scatter(eval_pts[q.name][:, 0], eval_pts[q.name][:, 1], s=3,
                       color="white", alpha=0.7, linewidths=0)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{q.name}  |  {vname}\ncos={m['cosine']:.3f}  IoU={m['best_iou']:.3f}",
                         fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046)
        ax = axs[3, qi]
        ax.bar(np.arange(n_sc) - 0.2, g_global[qi], width=0.4, label="global", color="0.6")
        ax.bar(np.arange(n_sc) + 0.2, g_quad[qi], width=0.4, label="quadrant", color="tab:blue")
        ax.set_title(f"{q.name}: learned gains", fontsize=9)
        ax.set_xlabel("scale (coarse -> fine)")
        if qi == 0:
            ax.set_ylabel("gain")
            ax.legend(fontsize=8)
    fig.suptitle("Per-quadrant shape encoding: unmodulated vs global vs per-quadrant learned scale gains",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
