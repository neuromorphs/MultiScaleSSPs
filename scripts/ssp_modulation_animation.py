#!/usr/bin/env python
"""Animated demo of location-conditioned SSP scale modulation.

A 10x10 world contains circular blobs whose size/density varies by quadrant:
upper-right = many densely packed small blobs, lower-right = few small blobs,
lower-left = few large blobs, upper-left = many large blobs. An agent follows
a smooth random trajectory through the world. At each timestep the blobs whose
CENTERS lie within a radius R of the agent are encoded as the mean of their
center SSP embeddings -- one encoding per object, so the learned kernel width
alone must express each blob's size -- with the Fourier scale blocks modulated
by a small MLP mapping the *agent's current location* (not pixel locations) to
a non-negative n_scales gain vector. Training minimizes a cosine
(shape-matching) loss between the modulated similarity map and the local blob
indicator, with L1 sparsity and a coarse-bias regularizer (scale-weighted
gain magnitude), the config validated in scripts/quadrant_scale_modulation.py.

The animation shows:
  left  -- blob outlines (black), the trajectory, an 'x' at the current
           location, and the local similarity map of the current encoding
           overlaid with alpha=0 where similarity is low, so a colored patch
           follows the path and aligns with the blob outlines.
  right -- a column of fixed hexagonal grid-module interference patterns,
           finest scale at the top, coarsest at the bottom; each panel
           cross-fades from greyscale (gain ~ 0) to full color (high gain)
           according to the learned modulation at the current location.

Also saves a diagnostic figure of the learned gain profile at the four
quadrant centres.

Example:
    python scripts/ssp_modulation_animation.py --out results/scale_modulation/ssp_modulation_demo.mp4
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
from scipy.interpolate import CubicSpline
import imageio.v2 as imageio

import jax
import jax.numpy as jnp
import optax
from vsagym.spaces import HexagonalSSPSpace

sys.path.insert(0, str(FSPath(__file__).parent))
from shape_scale_modulation import (  # noqa: E402
    SOFTPLUS_INV_1, expand_gains, init_mlp, make_scale_index, mlp_forward,
    modulate_spectral, n_params,
)

WORLD = 10.0  # world is [0, WORLD]^2


# ---------------------------------------------------------------------------
# Blob world
# ---------------------------------------------------------------------------


def make_blobs(rng) -> np.ndarray:
    """(n_blobs, 3) array of (cx, cy, r). Quadrant layout: UL many large,
    UR many small (dense), LL few large, LR few small."""
    lo, hi, mid = 0.4, WORLD - 0.4, WORLD / 2
    quads = [  # (x0, x1, y0, y1, n, rmin, rmax, margin)
        (lo, mid - 0.3, mid + 0.3, hi, 8, 0.45, 0.70, 0.12),   # UL: many large
        (mid + 0.3, hi, mid + 0.3, hi, 45, 0.13, 0.22, 0.08),  # UR: many small, dense
        (lo, mid - 0.3, lo, mid - 0.3, 3, 0.60, 0.90, 0.30),   # LL: few large
        (mid + 0.3, hi, lo, mid - 0.3, 4, 0.13, 0.22, 0.60),   # LR: few small
    ]
    blobs = []
    for x0, x1, y0, y1, n, rmin, rmax, margin in quads:
        placed, tries, shrink = 0, 0, 1.0
        while placed < n:
            tries += 1
            if tries % 2000 == 0:
                shrink *= 0.93  # quadrant too packed for this draw: relax sizes
            r = rng.uniform(rmin, rmax) * shrink
            c = rng.uniform([x0 + r, y0 + r], [x1 - r, y1 - r])
            if all(np.hypot(*(c - b[:2])) > r + b[2] + margin * shrink for b in blobs):
                blobs.append(np.array([c[0], c[1], r]))
                placed += 1
    return np.array(blobs)


def inside_blobs(pts: np.ndarray, blobs: np.ndarray) -> np.ndarray:
    """(n,) bool: point lies inside any blob."""
    d2 = ((pts[:, None, :] - blobs[None, :, :2]) ** 2).sum(-1)
    return (d2 < blobs[None, :, 2] ** 2).any(axis=1)


def sample_blob_points(blobs, rng, k: int) -> np.ndarray:
    """k feature points uniform inside EACH blob, (n_blobs * k, 2). Kept for
    the 'samples' mode of scripts/quadrant_scale_modulation.py."""
    n = len(blobs)
    th = rng.uniform(0, 2 * np.pi, (n, k))
    r = blobs[:, 2:3] * np.sqrt(rng.uniform(0, 1, (n, k)))
    return (blobs[:, None, :2]
            + np.stack([r * np.cos(th), r * np.sin(th)], axis=-1)).reshape(-1, 2)


def make_trajectory(rng, n_frames: int) -> np.ndarray:
    """Smooth random tour through all four quadrants, (n_frames, 2)."""
    centers = [(2.5, 2.5), (2.5, 7.5), (7.5, 7.5), (7.5, 2.5)]  # LL UL UR LR
    order = [0, 1, 2, 3, 0]
    way = []
    for q in order:
        cx, cy = centers[q]
        for _ in range(2):
            way.append([cx + rng.uniform(-1.6, 1.6), cy + rng.uniform(-1.6, 1.6)])
    way = np.clip(np.array(way), 0.9, WORLD - 0.9)
    t = np.linspace(0, 1, len(way))
    spline = CubicSpline(t, way, axis=0)
    return np.clip(spline(np.linspace(0, 1, n_frames)), 0.7, WORLD - 0.7)


def normalize_xy(pts: np.ndarray) -> np.ndarray:
    return 2.0 * np.asarray(pts) / WORLD - 1.0


# ---------------------------------------------------------------------------
# Local (centers-only) encoding + modulation
# ---------------------------------------------------------------------------


def window_grid(radius: float, n: int):
    """Square window grid of offsets covering the encoding disk."""
    w = radius + 0.4
    xs = np.linspace(-w, w, n)
    X, Y = np.meshgrid(xs, xs)
    return w, np.column_stack([X.ravel(), Y.ravel()])


def local_features(p, centers, radius: float, n_max: int):
    """Blob centers within `radius` of p, padded to n_max.
    Returns (pts (n_max, 2), mask (n_max,)); padding sits at p with mask 0."""
    sel = centers[np.linalg.norm(centers - p, axis=1) <= radius][:n_max]
    pts = np.tile(p, (n_max, 1))
    mask = np.zeros(n_max, dtype=np.float32)
    pts[: len(sel)] = sel
    mask[: len(sel)] = 1.0
    return pts, mask


def gains_of(params, xy_norm):
    """Non-negative per-scale gains at (normalized) locations; init ~1."""
    return jax.nn.softplus(mlp_forward(params, xy_norm) + SOFTPLUS_INV_1)


def local_sims(g_spec, grid_ssps, samp_ssps, samp_mask):
    """Similarity of the modulated local mean encoding with the modulated
    window grid encodings; one gain spectrum g_spec for the whole timestep."""
    mod_grid = modulate_spectral(grid_ssps, g_spec)
    mod_samp = modulate_spectral(samp_ssps, g_spec)
    m = samp_mask[:, None]
    mean = (mod_samp * m).sum(0) / jnp.maximum(m.sum(), 1.0)
    return mod_grid @ mean


# ---------------------------------------------------------------------------
# Training: MLP location -> gains, cosine loss + sparsity/coarse regs
# ---------------------------------------------------------------------------


def train(ssp_space, scale_idx, blobs, args, seed):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    scale_idx_j = jnp.asarray(scale_idx)
    w_coarse = jnp.asarray(np.asarray(ssp_space.scales) / np.max(ssp_space.scales))

    _, offsets = window_grid(args.radius, args.grid_n)
    centers = blobs[:, :2]

    params = init_mlp(key, args.hidden, ssp_space.n_scales)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    def one_loss(params, loc_norm, grid_ssps, samp_ssps, samp_mask, target):
        g = gains_of(params, loc_norm)
        sims = local_sims(expand_gains(g, scale_idx_j), grid_ssps, samp_ssps, samp_mask)
        # scale-invariant shape match (target is unit-norm or all-zero).
        # smoothed norm: differentiable even when the local region is empty
        # and sims is identically zero (plain norm has a NaN gradient at 0)
        fit = 1.0 - jnp.dot(sims / jnp.sqrt(jnp.sum(sims ** 2) + 1e-12), target)
        reg = (args.reg_w * jnp.sum(jnp.abs(g))
               + args.coarse_w * jnp.sum(w_coarse * g ** 2))
        return fit + reg

    def loss_fn(params, locs_norm, grid_ssps, samp_ssps, samp_masks, targets):
        return jax.vmap(one_loss, in_axes=(None, 0, 0, 0, 0, 0))(
            params, locs_norm, grid_ssps, samp_ssps, samp_masks, targets).mean()

    @jax.jit
    def step(params, opt_state, *batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, *batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = np.empty(args.steps)
    for i in range(args.steps):
        locs = rng.uniform(0.7, WORLD - 0.7, size=(args.loc_batch, 2))
        grid_s, samp_s, masks, targets = [], [], [], []
        for p in locs:
            gp = p + offsets
            sp, mask = local_features(p, centers, args.radius, args.max_local)
            grid_s.append(ssp_space.encode(gp))
            samp_s.append(ssp_space.encode(sp))
            masks.append(mask)
            t = (inside_blobs(gp, blobs)
                 & (np.linalg.norm(offsets, axis=1) <= args.radius)).astype(np.float32)
            n = np.linalg.norm(t)
            targets.append(t / n if n > 0 else t)
        batch = (jnp.asarray(normalize_xy(locs), dtype=jnp.float32),
                 jnp.asarray(np.stack(grid_s), dtype=jnp.float32),
                 jnp.asarray(np.stack(samp_s), dtype=jnp.float32),
                 jnp.asarray(np.stack(masks), dtype=jnp.float32),
                 jnp.asarray(np.stack(targets)))
        params, opt_state, loss = step(params, opt_state, *batch)
        losses[i] = float(loss)
        if i % 200 == 0 or i == args.steps - 1:
            print(f"  step {i:>4}: loss {losses[i]:.4f}")
    return params, losses


# ---------------------------------------------------------------------------
# Hex grid-module panel images
# ---------------------------------------------------------------------------


def module_images(ssp_space, scale_indices, panel_n=160, panel_span=5.0):
    """Fixed interference pattern of one hexagonal module per chosen scale:
    sum of cosines over the (domain_dim + 1) simplex phase vectors of rotation
    0 at that scale, using the effective phases A/length_scale that encode()
    applies. Returns {scale_index: (panel_n, panel_n) image in [0, 1]}."""
    blk = ssp_space.domain_dim + 1
    ls = float(np.asarray(ssp_space.length_scale).ravel()[0])
    xs = np.linspace(0, panel_span, panel_n)
    X, Y = np.meshgrid(xs, xs)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    out = {}
    for k in scale_indices:
        A = ssp_space.phase_matrix[1 + k * blk: 1 + (k + 1) * blk] / ls
        img = np.cos(pts @ A.T).sum(axis=1).reshape(panel_n, panel_n) / blk
        out[k] = (img + 1.0) / 2.0
    return out


def panel_rgb(img01, cmap_name="viridis"):
    """(color_rgb, grey_rgb) renderings of a [0,1] module image. The grey
    version is the colored image's luminance, so cross-fading between them
    reads as a pure saturation change (grey = gain 0, colorful = high gain)."""
    color = plt.get_cmap(cmap_name)(img01)[..., :3]
    lum = color @ np.array([0.299, 0.587, 0.114])
    grey = np.repeat(lum[..., None], 3, axis=-1)
    return color, grey


# ---------------------------------------------------------------------------
# Diagnostics + animation
# ---------------------------------------------------------------------------


def fig_quadrant_gains(params, scales, path):
    centers = {"upper left (many large)": (2.5, 7.5), "upper right (many small)": (7.5, 7.5),
               "lower left (few large)": (2.5, 2.5), "lower right (few small)": (7.5, 2.5)}
    fig, axs = plt.subplots(2, 2, figsize=(9, 6.5), sharey=True)
    order = np.argsort(scales)
    for ax, (label, c) in zip(axs.ravel(), centers.items()):
        g = np.asarray(gains_of(params, jnp.asarray(normalize_xy(np.array([c])), dtype=jnp.float32)))[0]
        ax.bar(np.arange(len(scales)), g[order], color="tab:blue")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("scale (coarse -> fine)")
        ax.set_ylabel("gain")
    fig.suptitle("Learned gain profile at each quadrant centre")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_animation(params, ssp_space, scale_idx, blobs, traj, args, out_path):
    scale_idx_j = jnp.asarray(scale_idx)
    win, offsets = window_grid(args.radius, args.grid_n)
    centers = blobs[:, :2]
    offset_r = np.linalg.norm(offsets, axis=1)

    @jax.jit
    def frame_sims(g, grid_ssps, samp_ssps, samp_mask):
        return local_sims(expand_gains(g, scale_idx_j), grid_ssps, samp_ssps, samp_mask)

    # per-frame gains (for panel colors), plus a global max for a stable scale
    traj_gains = np.asarray(gains_of(params, jnp.asarray(normalize_xy(traj), dtype=jnp.float32)))
    g_max = traj_gains.max()

    # panel scales: finest at top -> coarsest at bottom
    n_sc = ssp_space.n_scales
    panel_scales = list(np.linspace(n_sc - 1, 0, args.panels).round().astype(int))
    mods = module_images(ssp_space, panel_scales)
    panel_imgs = {k: panel_rgb(mods[k]) for k in panel_scales}  # (color, grey)
    scales = np.asarray(ssp_space.scales)

    fig = plt.figure(figsize=(11.5, 8.4), facecolor="white")
    gs = fig.add_gridspec(args.panels, 2, width_ratios=[4.4, 1.0], wspace=0.08, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor("white")
    for cx, cy, r in blobs:
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor="black", lw=1.2))
    ax.plot(traj[:, 0], traj[:, 1], color="0.82", lw=1.2, zorder=1)
    trail, = ax.plot([], [], color="0.45", lw=1.6, zorder=2)
    cross, = ax.plot([], [], "x", color="crimson", ms=13, mew=3, zorder=5)
    ring = Circle((0, 0), args.radius, facecolor="none", edgecolor="0.75",
                  ls="--", lw=1.0, zorder=2)
    ax.add_patch(ring)
    overlay = ax.imshow(np.zeros((args.grid_n, args.grid_n, 4)), origin="lower",
                        extent=[0, 1, 0, 1], zorder=3, interpolation="bilinear")
    ax.set_xlim(0, WORLD)
    ax.set_ylim(0, WORLD)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("local shape encoding similarity (modulated by g(current location))")

    panel_ims = []
    for row, k in enumerate(panel_scales):
        pax = fig.add_subplot(gs[row, 1])
        im = pax.imshow(panel_imgs[k][1], origin="lower")  # start grey
        pax.set_xticks([])
        pax.set_yticks([])
        pax.set_title(f"scale {scales[k]:.2f}" + ("  (fine)" if row == 0 else
                      "  (coarse)" if row == args.panels - 1 else ""), fontsize=9)
        panel_ims.append(im)

    cmap = plt.get_cmap("viridis")
    writer = imageio.get_writer(out_path, fps=args.fps, macro_block_size=1)
    gif_frames = []
    t0 = time.time()
    for f, p in enumerate(traj):
        grid_pts = p + offsets
        samp_pts, mask = local_features(p, centers, args.radius, args.max_local)
        sims = np.asarray(frame_sims(
            jnp.asarray(traj_gains[f]),
            jnp.asarray(ssp_space.encode(grid_pts), dtype=jnp.float32),
            jnp.asarray(ssp_space.encode(samp_pts), dtype=jnp.float32),
            jnp.asarray(mask, dtype=jnp.float32)))
        # RGBA overlay: colored by similarity, transparent where it is low
        # (normalize by a high quantile so one sharp peak can't wash out the rest)
        smax = np.quantile(sims, 0.98)
        simsn = np.clip(sims / smax, 0, 1) if smax > 1e-6 else np.zeros_like(sims)
        rgba = cmap(simsn.reshape(args.grid_n, args.grid_n))
        alpha = np.clip((simsn - args.alpha_thr) / (1 - args.alpha_thr), 0, 1)
        alpha *= (offset_r <= args.radius)  # clip overlay to the encoding disk
        rgba[..., 3] = 0.85 * alpha.reshape(args.grid_n, args.grid_n)
        overlay.set_data(rgba)
        overlay.set_extent([p[0] - win, p[0] + win, p[1] - win, p[1] + win])

        trail.set_data(traj[: f + 1, 0], traj[: f + 1, 1])
        cross.set_data([p[0]], [p[1]])
        ring.set_center(p)
        # panels: cross-fade grey (gain 0) -> full color (max gain along the
        # trajectory), plus a subtle alpha sweep in [0.8, 1] with the gain
        for im, k in zip(panel_ims, panel_scales):
            w = float(np.clip(traj_gains[f, k] / g_max, 0.0, 1.0))
            color, grey = panel_imgs[k]
            im.set_data((1.0 - w) * grey + w * color)
            im.set_alpha(0.8 + 0.2 * w)

        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        writer.append_data(frame)
        if f % args.gif_stride == 0:  # gif: temporally + spatially downsampled
            gif_frames.append(frame[:: args.gif_scale, :: args.gif_scale])
        if f % 50 == 0:
            print(f"  frame {f}/{len(traj)} ({time.time() - t0:.0f}s)")
    writer.close()
    plt.close(fig)
    gif_path = out_path.with_suffix(".gif")
    imageio.mimsave(gif_path, gif_frames, fps=max(1, round(args.fps / args.gif_stride)), loop=0)
    print(f"wrote {gif_path} ({len(gif_frames)} frames)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scales", type=int, default=10)
    ap.add_argument("--n-rotates", type=int, default=33, help="d = 6*n_scales*n_rotates + 1")
    ap.add_argument("--scale-min", type=float, default=1.0)
    ap.add_argument("--scale-max", type=float, default=10.0)
    ap.add_argument("--scale-sampling", type=str, default="log", choices=["lin", "log", "rand"])
    ap.add_argument("--ls", type=float, default=0.9,
                    help="base length scale; kernel widths span ls/scale_max .. ls/scale_min")
    ap.add_argument("--radius", type=float, default=2.0, help="encoding radius around the agent")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loc-batch", type=int, default=6, help="training locations per step")
    ap.add_argument("--max-local", type=int, default=48, help="local blob-center cap (pad size)")
    ap.add_argument("--grid-n", type=int, default=45, help="local window grid resolution")
    ap.add_argument("--reg-w", type=float, default=0.003, help="L1 on the gains (sparsity)")
    ap.add_argument("--coarse-w", type=float, default=0.01,
                    help="scale-weighted quadratic penalty biasing toward coarse scales")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--gif-stride", type=int, default=2, help="keep every Nth frame in the gif")
    ap.add_argument("--gif-scale", type=int, default=2, help="spatial downsample factor for the gif")
    ap.add_argument("--load-params", type=str, default=None,
                    help="path to a saved *_params.npz to skip training (render-only tweaks)")
    ap.add_argument("--panels", type=int, default=5, help="grid-module panels (scales shown)")
    ap.add_argument("--alpha-thr", type=float, default=0.25,
                    help="similarity fraction below which the overlay is transparent")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=str, default="results/scale_modulation/ssp_modulation_demo.mp4")
    args = ap.parse_args()

    out_path = FSPath(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    blobs = make_blobs(rng)
    traj = make_trajectory(rng, args.frames)
    print(f"world: {len(blobs)} blobs; trajectory {args.frames} frames")

    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=args.n_scales, n_rotates=args.n_rotates,
                                  scale_min=args.scale_min, scale_max=args.scale_max,
                                  scale_sampling=args.scale_sampling,
                                  length_scale=args.ls, rng=0)
    scale_idx = make_scale_index(ssp_space)
    print(f"ssp_dim={ssp_space.ssp_dim}, n_scales={ssp_space.n_scales}, "
          f"kernel widths {args.ls/args.scale_max:.2f}..{args.ls/args.scale_min:.2f}; "
          f"MLP params: {n_params(init_mlp(jax.random.PRNGKey(0), args.hidden, ssp_space.n_scales))}")

    params_path = out_path.with_name(out_path.stem + "_params.npz")
    if args.load_params:
        loaded = np.load(args.load_params)
        params = {k: jnp.asarray(loaded[k]) for k in loaded.files}
        print(f"loaded MLP params from {args.load_params} (skipping training)")
    else:
        print("training location -> gains MLP ...")
        params, losses = train(ssp_space, scale_idx, blobs, args, args.seed)
        np.save(out_path.with_suffix(".losses.npy"), losses)
        np.savez(params_path, **{k: np.asarray(v) for k, v in params.items()})
        print(f"saved MLP params to {params_path}")

        fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor="white")
        ax.plot(losses, color="tab:blue")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out_path.with_name(out_path.stem + "_losses.png"), dpi=150)
        plt.close(fig)

    fig_quadrant_gains(params, np.asarray(ssp_space.scales),
                       out_path.with_name(out_path.stem + "_quadrant_gains.png"))

    print("rendering animation ...")
    render_animation(params, ssp_space, scale_idx, blobs, traj, args, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
