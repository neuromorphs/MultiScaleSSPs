#!/usr/bin/env python
"""Evaluate SSP shape encodings: fixed length scales vs learned scale modulation.

A shape is encoded as the mean of the SSP embeddings of points sampled inside
it (as in examples/star_ssp_mean.ipynb). The similarity map of that mean SSP
against a grid of location encodings should recover the shape's indicator
function. This script compares, on a dataset of multi-scale 2D shapes:

  fixed      -- plain HexagonalSSPSpace encodings at a range of length scales
                (no training; one method per length scale)
  mlp-scales -- learned modulation g: location -> R^{n_scales}, a free gain per
                scale block of the Fourier spectrum (renormalized, so the gain
                reshapes each encoding's direction, never its magnitude)
  mlp-kernel -- learned modulation g: location -> R^2 = (mu, sigma) of a
                discretized Gaussian over the scale blocks (a smooth
                scale-band selector; same renormalized modulation)
  mlp-scalar -- learned modulation g: location -> R, a single positive gain
                multiplying the whole encoding (no renormalization -- a uniform
                spectral gain would otherwise cancel; it acts as an importance
                weight on each point's contribution to the mean)

Each learned map comes in two parameterizations (--models): 'mlp', a
lightweight 2-layer MLP (coords -> tanh hidden -> head; default hidden=16:
269 / 82 / 65 params) whose field is a smooth function of continuous
location; and 'lut', the notebook's per-pixel lookup table on a lut_n x lut_n
grid (default 50: 32500 / 5000 / 2500 params) as an upper-bound check on how
much the MLP's small capacity costs.

All methods are scored on held-out data (fresh sample points for the mean, a
fine render grid never used in training) with scale-invariant metrics: cosine
similarity to the shape indicator, ROC AUC (inside vs outside), and best-
threshold IoU. Outputs figures + metrics.csv/json under --outdir.

Example:
    python scripts/shape_scale_modulation.py --outdir results/scale_modulation
    python scripts/shape_scale_modulation.py --shapes star cross --steps 200
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path as FSPath
from typing import Callable

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from scipy.stats import rankdata

import jax
import jax.numpy as jnp
import optax
from vsagym.spaces import HexagonalSSPSpace


# ---------------------------------------------------------------------------
# Shape dataset: each shape is a vectorized indicator over a common domain.
# ---------------------------------------------------------------------------

DOMAIN = (-1.2, -1.2, 1.2, 1.2)  # x0, y0, x1, y1 (shared by grid + sampling)


@dataclass
class Shape:
    name: str
    contains: Callable[[np.ndarray], np.ndarray]  # (n, 2) -> (n,) bool


def _circ(pts, cx, cy, r):
    return (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2 < r * r


def make_shapes() -> dict[str, Shape]:
    star_path = MplPath.unit_regular_star(5, innerCircle=0.382)
    spiky_path = MplPath.unit_regular_star(8, innerCircle=0.15)

    def cross(p):
        return ((np.abs(p[:, 0]) < 0.12) & (np.abs(p[:, 1]) < 0.9)) | (
            (np.abs(p[:, 1]) < 0.12) & (np.abs(p[:, 0]) < 0.9)
        )

    def blobs(p):
        # one coarse blob + three fine satellites: needs different scales at once
        return (
            _circ(p, -0.35, -0.35, 0.55)
            | _circ(p, 0.6, 0.6, 0.13)
            | _circ(p, 0.65, -0.35, 0.13)
            | _circ(p, -0.35, 0.65, 0.13)
        )

    def comb(p):
        spine = (np.abs(p[:, 0]) < 0.9) & (p[:, 1] > -0.85) & (p[:, 1] < -0.55)
        teeth = np.zeros(len(p), dtype=bool)
        for cx in np.linspace(-0.8, 0.8, 5):
            teeth |= (np.abs(p[:, 0] - cx) < 0.06) & (p[:, 1] >= -0.55) & (p[:, 1] < 0.75)
        return spine | teeth

    shapes = [
        Shape("star", star_path.contains_points),
        Shape("spiky_star", spiky_path.contains_points),
        Shape("annulus", lambda p: _circ(p, 0, 0, 0.85) & ~_circ(p, 0, 0, 0.6)),
        Shape("cross", cross),
        Shape("blobs", blobs),
        Shape("crescent", lambda p: _circ(p, 0, 0, 0.8) & ~_circ(p, 0.35, 0, 0.62)),
        Shape("comb", comb),
    ]
    return {s.name: s for s in shapes}


def sample_points(shape: Shape, n: int, rng: np.random.Generator) -> np.ndarray:
    """Rejection-sample n points uniformly inside the shape."""
    x0, y0, x1, y1 = DOMAIN
    pts = np.empty((0, 2))
    while len(pts) < n:
        cand = rng.uniform([x0, y0], [x1, y1], size=(8 * n, 2))
        pts = np.vstack([pts, cand[shape.contains(cand)]])
    return pts[:n]


def get_grid(grid_n: int):
    x0, y0, x1, y1 = DOMAIN
    xs = np.linspace(x0, x1, grid_n)
    ys = np.linspace(y0, y1, grid_n)
    X, Y = np.meshgrid(xs, ys)
    return X, Y, np.column_stack([X.ravel(), Y.ravel()])


def normalize_xy(pts: np.ndarray) -> np.ndarray:
    """Map domain coords to [-1, 1]^2 for the MLP input."""
    x0, y0, x1, y1 = DOMAIN
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    h = np.array([(x1 - x0) / 2, (y1 - y0) / 2])
    return (pts - c) / h


# ---------------------------------------------------------------------------
# SSP spectral machinery (same layout as the notebook)
# ---------------------------------------------------------------------------


def make_scale_index(ssp_space) -> np.ndarray:
    """Scale-block index of each non-DC rfft half-spectrum coefficient.

    The phase matrix rows are [DC; K; -flip(K)] with K rotation-major and,
    within a rotation, the scales contiguous in groups of (domain_dim + 1)
    simplex phases.
    """
    d = ssp_space.ssp_dim
    assert d % 2 == 1, "expects odd ssp_dim (no Nyquist bin)"
    blk = ssp_space.domain_dim + 1
    scale_idx = np.tile(
        np.repeat(np.arange(ssp_space.n_scales), blk), ssp_space.n_rotates
    )
    assert len(scale_idx) == (d - 1) // 2
    norms = np.linalg.norm(ssp_space.phase_matrix[1 : 1 + (d - 1) // 2], axis=1)
    assert np.allclose(norms, np.asarray(ssp_space.scales)[scale_idx])
    return scale_idx


def expand_gains(g_scales, scale_idx):
    """Per-scale gains (..., n_scales) -> rfft half-spectrum gains (..., d//2+1),
    block-constant across scales, DC gain fixed at 1. Real gains on the half
    spectrum keep the modulated SSPs real."""
    dc = jnp.ones((*g_scales.shape[:-1], 1))
    return jnp.concatenate([dc, g_scales[..., scale_idx]], axis=-1)


def modulate_spectral(ssps, g):
    """Apply per-frequency gains g (real, rfft half-spectrum), then renormalize
    to unit norm so the gain reshapes only the kernel's phase/direction."""
    mod = jnp.fft.irfft(g * jnp.fft.rfft(ssps, axis=-1), n=ssps.shape[-1], axis=-1)
    return mod / jnp.linalg.norm(mod, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# Lightweight 2-layer MLP: coords in [-1,1]^2 -> head-specific modulation
# ---------------------------------------------------------------------------

SOFTPLUS_INV_1 = float(np.log(np.e - 1.0))  # softplus(0.5413) = 1


def init_mlp(key, hidden: int, out_dim: int):
    k1, k2 = jax.random.split(key)
    return {
        "W1": jax.random.normal(k1, (2, hidden)) / np.sqrt(2.0),
        "b1": jnp.zeros(hidden),
        # near-zero output weights: every variant starts at (almost) identity
        "W2": jax.random.normal(k2, (hidden, out_dim)) * 1e-2,
        "b2": jnp.zeros(out_dim),
    }


def mlp_forward(params, xy):
    h = jnp.tanh(xy @ params["W1"] + params["b1"])
    return h @ params["W2"] + params["b2"]


def n_params(params) -> int:
    return int(sum(p.size for p in jax.tree_util.tree_leaves(params)))


def lut_index(xy, lut_n):
    """Nearest cell of a lut_n x lut_n grid over [-1,1]^2 (row-major flat index)."""
    ij = jnp.clip(jnp.round((xy + 1.0) / 2.0 * (lut_n - 1)), 0, lut_n - 1).astype(jnp.int32)
    return ij[..., 1] * lut_n + ij[..., 0]


class Variant:
    """A learned modulation: a map from location to raw head inputs -- either a
    2-layer MLP of continuous coords ('mlp') or a per-pixel lookup table on a
    lut_n x lut_n grid ('lut', the notebook's parameterization) -- plus a head
    turning that output into gains, and how the gains modulate encoded SSPs.
    All heads are identity modulation at raw output 0 (the LUT init)."""

    def __init__(self, kind: str, head: str, out_dim: int, n_scales: int, scale_idx,
                 hidden: int, lut_n: int):
        self.kind, self.head = kind, head
        self.name = f"{kind}-{head}"
        self.out_dim = out_dim
        self.n_scales = n_scales
        self.scale_idx = jnp.asarray(scale_idx)
        self.hidden, self.lut_n = hidden, lut_n

    def init_params(self, key):
        if self.kind == "mlp":
            return init_mlp(key, self.hidden, self.out_dim)
        return {"table": jnp.zeros((self.lut_n ** 2, self.out_dim))}

    def raw_out(self, params, xy):
        if self.kind == "mlp":
            return mlp_forward(params, xy)
        return params["table"][lut_index(xy, self.lut_n)]

    def gains(self, params, xy):
        """Head output at (normalized) locations xy. Shape (..., head_dim)."""
        out = self.raw_out(params, xy)
        if self.head == "scales":
            return 1.0 + out  # free per-scale gains, init ~1
        if self.head == "kernel":
            mu = (self.n_scales - 1) / 2 + out[..., 0:1]
            sigma = jax.nn.softplus(out[..., 1:2] + 5.0) + 1e-3  # init: wide
            k = jnp.arange(self.n_scales)
            return jnp.exp(-0.5 * ((k - mu) / sigma) ** 2)
        if self.head == "scalar":
            return jax.nn.softplus(out[..., 0:1] + SOFTPLUS_INV_1)  # init ~1
        raise ValueError(self.head)

    def apply(self, params, xy, ssps):
        """Modulated encodings of points at (normalized) locations xy."""
        g = self.gains(params, xy)
        if self.head == "scalar":
            return g * ssps  # importance weight; no renorm (see module docstring)
        return modulate_spectral(ssps, expand_gains(g, self.scale_idx))


def make_variants(ssp_space, scale_idx, kinds, hidden, lut_n) -> list[Variant]:
    n_sc = ssp_space.n_scales
    heads = [("scales", n_sc), ("kernel", 2), ("scalar", 1)]
    return [Variant(k, h, od, n_sc, scale_idx, hidden, lut_n)
            for k in kinds for h, od in heads]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_variant(variant, shape, ssp_space, args, seed):
    """Fit one MLP modulation for one shape. Returns (params, losses)."""
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)

    # loss grid: target indicator (unit norm) + its encodings, fixed for the fit
    _, _, grid_pts = get_grid(args.train_grid_n)
    grid_ssps = jnp.asarray(ssp_space.encode(grid_pts), dtype=jnp.float32)
    grid_xy = jnp.asarray(normalize_xy(grid_pts), dtype=jnp.float32)
    target = shape.contains(grid_pts).astype(np.float32)
    target_j = jnp.asarray(target / np.linalg.norm(target))

    # pool of interior points, encoded once; each step draws a fresh minibatch
    pool_pts = sample_points(shape, args.pool, rng)
    pool_ssps = jnp.asarray(ssp_space.encode(pool_pts), dtype=jnp.float32)
    pool_xy = jnp.asarray(normalize_xy(pool_pts), dtype=jnp.float32)

    params = variant.init_params(key)
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, batch_ssps, batch_xy):
        mod_grid = variant.apply(params, grid_xy, grid_ssps)
        mod_mean = variant.apply(params, batch_xy, batch_ssps).mean(0, keepdims=True)
        sims = (mod_grid @ mod_mean.T).ravel()
        loss = jnp.sum((sims - target_j) ** 2)
        if args.reg_w > 0 and variant.head == "scales":
            # L1 toward 0: "use as few scales as possible"
            loss += args.reg_w * jnp.mean(jnp.sum(jnp.abs(variant.gains(params, grid_xy)), axis=-1))
        return loss

    @jax.jit
    def step(params, opt_state, batch_ssps, batch_xy):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch_ssps, batch_xy)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = np.empty(args.steps)
    for i in range(args.steps):
        idx = rng.choice(args.pool, size=args.batch, replace=False)
        params, opt_state, loss = step(params, opt_state, pool_ssps[idx], pool_xy[idx])
        losses[i] = float(loss)
    return params, losses


# ---------------------------------------------------------------------------
# Evaluation (held-out): scale-invariant metrics on the render grid
# ---------------------------------------------------------------------------


def metrics_from_simmap(sims: np.ndarray, inside: np.ndarray) -> dict:
    """sims, inside: flat arrays over the render grid. All metrics are
    invariant to affine rescaling of sims, so untrained baselines compete
    fairly with trained modulations."""
    t = inside.astype(np.float64)
    cos = float(sims @ t / (np.linalg.norm(sims) * np.linalg.norm(t) + 1e-12))
    # ROC AUC via rank statistic (Mann-Whitney U)
    ranks = rankdata(sims)
    n_pos, n_neg = int(t.sum()), int((1 - t).sum())
    auc = float((ranks[inside].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    # best IoU over a threshold sweep
    best_iou = 0.0
    for thr in np.quantile(sims, np.linspace(0.01, 0.99, 99)):
        pred = sims > thr
        inter = np.count_nonzero(pred & inside)
        union = np.count_nonzero(pred | inside)
        if union:
            best_iou = max(best_iou, inter / union)
    return {"cosine": cos, "auc": auc, "best_iou": float(best_iou)}


def eval_fixed(shape, test_pts, render_pts, inside_render, ssp_dim, ls) -> tuple[dict, np.ndarray]:
    space = HexagonalSSPSpace(domain_dim=2, ssp_dim=ssp_dim, length_scale=ls, rng=0)
    mean_ssp = space.encode(test_pts).mean(axis=0)
    sims = space.encode(render_pts) @ mean_ssp
    return metrics_from_simmap(sims, inside_render), sims


def eval_variant(variant, params, ssp_space, test_pts, render_pts, inside_render):
    test_ssps = jnp.asarray(ssp_space.encode(test_pts), dtype=jnp.float32)
    test_xy = jnp.asarray(normalize_xy(test_pts), dtype=jnp.float32)
    render_ssps = jnp.asarray(ssp_space.encode(render_pts), dtype=jnp.float32)
    render_xy = jnp.asarray(normalize_xy(render_pts), dtype=jnp.float32)
    mod_mean = variant.apply(params, test_xy, test_ssps).mean(0, keepdims=True)
    sims = np.asarray(variant.apply(params, render_xy, render_ssps) @ mod_mean.T).ravel()
    return metrics_from_simmap(sims, inside_render), sims


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _panel(ax, X, Y, data, title, inside_map, cmap="viridis"):
    im = ax.pcolormesh(X, Y, data.reshape(X.shape), cmap=cmap, shading="auto")
    ax.contour(X, Y, inside_map.reshape(X.shape), levels=[0.5], colors="white", linewidths=1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046)


def fig_similarity(shape_name, X, Y, inside, panels, path):
    """panels: list of (label, sims, metrics-or-None)."""
    fig, axs = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 3.4))
    for ax, (label, sims, m) in zip(np.atleast_1d(axs), panels):
        title = label if m is None else f"{label}\ncos={m['cosine']:.3f}  AUC={m['auc']:.3f}  IoU={m['best_iou']:.3f}"
        _panel(ax, X, Y, sims, title, inside)
    fig.suptitle(shape_name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_fields(shape_name, X, Y, inside, variants, trained, render_xy_j, path):
    """Visualize the learned modulation fields on the render grid."""
    cols = []
    for v in variants:
        params = trained[v.name][0]
        g = np.asarray(v.gains(params, render_xy_j))
        if v.head == "scales":
            cols.append((rf"{v.name}: $\|g(x)\|$", np.linalg.norm(g, axis=-1), "magma"))
            cols.append((rf"{v.name}: argmax scale", np.argmax(np.abs(g), axis=-1).astype(float), "plasma"))
        elif v.head == "kernel":
            out = np.asarray(v.raw_out(params, render_xy_j))
            n_sc = v.n_scales
            cols.append((rf"{v.name}: $\mu(x)$", (n_sc - 1) / 2 + out[:, 0], "plasma"))
            cols.append((rf"{v.name}: $\sigma(x)$", np.logaddexp(0, out[:, 1] + 5.0) + 1e-3, "cividis"))
        elif v.head == "scalar":
            cols.append((rf"{v.name}: $w(x)$", g[:, 0], "magma"))
    fig, axs = plt.subplots(1, len(cols), figsize=(3.4 * len(cols), 3.4))
    for ax, (title, data, cmap) in zip(np.atleast_1d(axs), cols):
        _panel(ax, X, Y, data, title, inside, cmap=cmap)
    fig.suptitle(f"{shape_name}: learned modulation fields")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_summary(shape_panels, X, Y, path):
    """One demo grid: a row per shape, a column per method.

    shape_panels: {shape_name: (inside, [(label, sims, metrics-or-None), ...])}
    """
    n_rows = len(shape_panels)
    n_cols = max(len(p) for _, p in shape_panels.values())
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows))
    axs = np.atleast_2d(axs)
    for i, (name, (inside, panels)) in enumerate(shape_panels.items()):
        for j, (label, sims, m) in enumerate(panels):
            title = label if m is None else f"{label}\ncos={m['cosine']:.3f}  IoU={m['best_iou']:.3f}"
            _panel(axs[i, j], X, Y, sims, title, inside)
        axs[i, 0].set_ylabel(name, fontsize=11)
        for j in range(len(panels), n_cols):
            axs[i, j].axis("off")
    fig.suptitle("Shape similarity maps: fixed length scale vs learned modulation", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_losses(shape_name, trained, path):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for name, (_, losses) in trained.items():
        ax.plot(losses, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(shape_name)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shapes", nargs="+", default=None, help="subset of shapes (default: all)")
    ap.add_argument("--ssp-dim", type=int, default=1015)
    ap.add_argument("--base-ls", type=float, default=0.1, help="length scale of the modulated space")
    ap.add_argument("--fixed-ls", type=float, nargs="+",
                    default=[0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
                    help="fixed length-scale sweep (untrained baselines)")
    ap.add_argument("--models", nargs="+", choices=["mlp", "lut"], default=["mlp", "lut"],
                    help="modulation parameterizations to train")
    ap.add_argument("--hidden", type=int, default=16, help="MLP hidden units (keep small)")
    ap.add_argument("--lut-n", type=int, default=50, help="lookup-table grid resolution")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--pool", type=int, default=16384, help="pre-encoded interior point pool")
    ap.add_argument("--train-grid-n", type=int, default=48, help="loss/target grid resolution")
    ap.add_argument("--render-n", type=int, default=100, help="held-out eval grid resolution")
    ap.add_argument("--test-n", type=int, default=2000, help="held-out points for the eval mean")
    ap.add_argument("--reg-w", type=float, default=0.0, help="L1 gain reg for mlp-scales")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--outdir", type=str, default="results/scale_modulation")
    args = ap.parse_args()

    outdir = FSPath(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_shapes = make_shapes()
    names = args.shapes or list(all_shapes)
    unknown = [n for n in names if n not in all_shapes]
    if unknown:
        ap.error(f"unknown shapes {unknown}; available: {list(all_shapes)}")

    ssp_space = HexagonalSSPSpace(domain_dim=2, ssp_dim=args.ssp_dim, length_scale=args.base_ls, rng=0)
    print(f"base space: ssp_dim={ssp_space.ssp_dim}, n_scales={ssp_space.n_scales}, "
          f"n_rotates={ssp_space.n_rotates}, base length_scale={args.base_ls}")
    scale_idx = make_scale_index(ssp_space)
    variants = make_variants(ssp_space, scale_idx, args.models, args.hidden, args.lut_n)
    for v in variants:
        print(f"  {v.name}: head dim {v.out_dim}, "
              f"{n_params(v.init_params(jax.random.PRNGKey(0)))} params")

    Xr, Yr, render_pts = get_grid(args.render_n)
    render_xy_j = jnp.asarray(normalize_xy(render_pts), dtype=jnp.float32)

    rows = []
    shape_panels = {}
    for name in names:
        shape = all_shapes[name]
        t0 = time.time()
        inside_render = shape.contains(render_pts)
        test_pts = sample_points(shape, args.test_n, np.random.default_rng(2024))

        # fixed length-scale sweep (untrained)
        fixed_results = {}
        for ls in args.fixed_ls:
            m, sims = eval_fixed(shape, test_pts, render_pts, inside_render, args.ssp_dim, ls)
            fixed_results[ls] = (m, sims)
            rows.append({"shape": name, "method": f"fixed-ls={ls}", "params": 0, **m})
        best_ls = max(fixed_results, key=lambda k: fixed_results[k][0]["cosine"])

        # learned modulations
        trained, learned_panels = {}, []
        for v in variants:
            params, losses = train_variant(v, shape, ssp_space, args, args.seed)
            trained[v.name] = (params, losses)
            m, sims = eval_variant(v, params, ssp_space, test_pts, render_pts, inside_render)
            rows.append({"shape": name, "method": v.name, "params": n_params(params), **m})
            learned_panels.append((v.name, sims, m))
            print(f"[{name}] {v.name}: loss {losses[0]:.4f} -> {losses[-1]:.5f}, "
                  f"cos={m['cosine']:.3f} auc={m['auc']:.3f} iou={m['best_iou']:.3f}")

        # figures
        panels = [("target", inside_render.astype(float), None),
                  (f"best fixed (ls={best_ls})", fixed_results[best_ls][1],
                   fixed_results[best_ls][0])] + learned_panels
        shape_panels[name] = (inside_render, panels)
        np.savez(outdir / f"sims_{name}.npz",
                 render_shape=(args.render_n, args.render_n),
                 inside=inside_render,
                 **{label.split("\n")[0].split(" (")[0]: sims for label, sims, _ in panels},
                 **{f"fixed-ls={ls}": s for ls, (_, s) in fixed_results.items()})
        fig_similarity(name, Xr, Yr, inside_render, panels, outdir / f"sims_{name}.png")
        fig_similarity(f"{name}: fixed length-scale sweep", Xr, Yr, inside_render,
                       [(f"ls={ls}", s, m) for ls, (m, s) in fixed_results.items()],
                       outdir / f"fixed_sweep_{name}.png")
        fig_fields(name, Xr, Yr, inside_render, variants, trained, render_xy_j,
                   outdir / f"fields_{name}.png")
        fig_losses(name, trained, outdir / f"losses_{name}.png")
        print(f"[{name}] done in {time.time() - t0:.1f}s")

    # summary
    fig_summary(shape_panels, Xr, Yr, outdir / "summary_all_shapes.png")
    with open(outdir / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["shape", "method", "params", "cosine", "auc", "best_iou"])
        w.writeheader()
        w.writerows(rows)
    with open(outdir / "metrics.json", "w") as f:
        json.dump({"args": vars(args), "rows": rows}, f, indent=2)

    print(f"\n{'shape':<12} {'method':<16} {'params':>6} {'cosine':>7} {'auc':>7} {'iou':>7}")
    for r in rows:
        print(f"{r['shape']:<12} {r['method']:<16} {r['params']:>6} "
              f"{r['cosine']:>7.3f} {r['auc']:>7.3f} {r['best_iou']:>7.3f}")
    print(f"\nwrote {outdir}/metrics.csv and figures")


if __name__ == "__main__":
    main()
