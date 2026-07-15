#!/usr/bin/env python
"""Learn a 3D encoding matrix A(x) so a bundled point cloud decodes sharply.

The Stanford bunny surface is sampled into a point cloud which is bundled
into a single SSP, M = sum_i phi(x_i), with a position-dependent phase
matrix A(x) = A_hex + s * MLP(x) (SIREN, zero-init head). The similarity
field of the bundle,

    sim(x') = phi(x') . M
            = (N + 2 sum_k Re[e^{i a_k(x') . x'} conj(b_k)]) / d,
    b_k     = sum_i e^{i a_k(x_i) . x_i},

is trained (through both the query and the cloud encodings, plus an affine
readout) to match a high-resolution surface-density target
t(x) = exp(-dist(x, surface)^2 / 2 sigma^2). At initialization the head is
zero, so the baseline is the standard HexSSP bundle of the same cloud.

Training points are drawn half uniformly over the domain and half near the
surface. Outputs (to --out): bunny_fit.png (z-slices: target | initial |
learned), params.msgpack, and printed metrics.

    python scripts/bunny_kernel_fit.py
"""

import argparse
from pathlib import Path as FSPath

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

import jax

jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import optax
import flax.linen as nn
from flax import serialization

from vsagym.spaces import HexagonalSSPSpace


# ---------------------------------------------------------------------------
# Geometry: cloud + high-res surface-density target
# ---------------------------------------------------------------------------


def load_cloud(path, n_cloud, n_dense, seed):
    """Surface point cloud + dense sample for the density target, both in a
    centered domain with max extent 1.6 (inside [-1, 1]^3)."""
    import trimesh

    mesh = trimesh.load(str(path), force="mesh")
    lo, hi = mesh.bounds
    mesh.apply_translation(-(lo + hi) / 2)
    mesh.apply_scale(1.6 / (hi - lo).max())
    cloud, _ = trimesh.sample.sample_surface(mesh, n_cloud, seed=seed)
    dense, _ = trimesh.sample.sample_surface(mesh, n_dense, seed=seed + 1)
    return np.asarray(cloud, np.float32), np.asarray(dense, np.float32)


def make_pool(dense_tree, dense, sigma, n_each, rng):
    """Training points: half uniform over the domain, half hugging the
    surface; targets t = exp(-d^2 / 2 sigma^2) from the dense-sample tree."""
    uni = rng.uniform(-1, 1, (n_each, 3)).astype(np.float32)
    near = (dense[rng.integers(0, len(dense), n_each)]
            + rng.normal(0, 2.5 * sigma, (n_each, 3))).astype(np.float32)
    pts = np.vstack([uni, near])
    d, _ = dense_tree.query(pts, workers=-1)
    t = np.exp(-d ** 2 / (2 * sigma ** 2)).astype(np.float32)
    return pts, t


# ---------------------------------------------------------------------------
# Model: A(x) = a_base + dphase_scale * PhaseNet(x)
# ---------------------------------------------------------------------------


def _siren_first_init(key, shape, dtype=jnp.float32):
    return jax.random.uniform(key, shape, dtype, -1.0, 1.0) / shape[0]


_siren_hidden_init = nn.initializers.variance_scaling(2.0, "fan_in", "uniform")


class PhaseNet(nn.Module):
    """x in [-1,1]^dim -> dA(x), the residual half-spectrum phase matrix.
    SIREN with zero-init head: A(x) = a_base at initialization."""

    hidden: tuple
    n_free: int
    dim: int = 3
    w0: float = 15.0

    @nn.compact
    def __call__(self, x):
        h = x
        for i, w in enumerate(self.hidden):
            init = _siren_first_init if i == 0 else _siren_hidden_init
            h = nn.Dense(w, kernel_init=init)(h)
            h = jnp.sin(self.w0 * h if i == 0 else h)
        dA = nn.Dense(self.dim * self.n_free, kernel_init=nn.initializers.zeros,
                      bias_init=nn.initializers.zeros)(h)
        return dA.reshape(x.shape[:-1] + (self.n_free, self.dim))


def make_sim_fn(model, a_base, d, dphase_scale, cloud_j):
    """sim(params, xq) -> (affine-read similarity of the bundle, reg term).
    Both the cloud and the query are encoded with the learned A(x)."""
    n_cloud = cloud_j.shape[0]

    def sim(params, xq):
        dAc = dphase_scale * model.apply(params["net"], cloud_j)
        thc = jnp.einsum("nkj,nj->nk", a_base + dAc, cloud_j)   # (N, nf)
        bre, bim = jnp.cos(thc).sum(0), jnp.sin(thc).sum(0)     # bundle b_k
        dAq = dphase_scale * model.apply(params["net"], xq)
        thq = jnp.einsum("...kj,...j->...k", a_base + dAq, xq)
        raw = (n_cloud + 2 * (jnp.cos(thq) * bre + jnp.sin(thq) * bim).sum(-1)) \
            / (d * n_cloud)
        reg = jnp.mean(dAq ** 2) + jnp.mean(dAc ** 2)
        return params["alpha"] * raw + params["beta"], reg

    return sim


def eval_chunked(sim_fn, params, pts, chunk=65536):
    sim_j = jax.jit(lambda p, x: sim_fn(p, x)[0])
    return np.concatenate([np.asarray(sim_j(params, jnp.asarray(pts[i:i + chunk])))
                           for i in range(0, len(pts), chunk)])


def affine_refit(pred, t):
    """Closed-form (alpha, beta) refit, used to give the baseline its best
    possible affine readout."""
    A = np.column_stack([pred, np.ones_like(pred)])
    (al, be), *_ = np.linalg.lstsq(A, t, rcond=None)
    return al * pred + be


def metrics(pred, t):
    cos = pred @ t / (np.linalg.norm(pred) * np.linalg.norm(t) + 1e-12)
    r2 = 1 - np.sum((pred - t) ** 2) / np.sum((t - t.mean()) ** 2)
    return cos, r2


# ---------------------------------------------------------------------------


def train(model, sim_fn, params, pool_pts, pool_t, args):
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)
    pts_j, t_j = jnp.asarray(pool_pts), jnp.asarray(pool_t)

    def loss_fn(params, xq, t):
        pred, reg = sim_fn(params, xq)
        return jnp.mean((pred - t) ** 2) + args.reg_da * reg

    @jax.jit
    def step(params, opt_state, key):
        idx = jax.random.randint(key, (args.batch,), 0, pts_j.shape[0])
        loss, grads = jax.value_and_grad(loss_fn)(params, pts_j[idx], t_j[idx])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    key = jax.random.PRNGKey(args.seed)
    for i in range(args.steps):
        key, sub = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, sub)
        if (i + 1) % args.eval_every == 0 or i == 0:
            print(f"step {i + 1:5d}  batch loss {float(loss):.5f}  "
                  f"alpha {float(params['alpha']):.2f}")
    return params


def fig_slices(z_slices, res, tree, sigma, sim0, sim1, out_path):
    """Rows: target | initial bundle | learned. Columns: z slices."""
    cc = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(cc, cc)
    fig, axes = plt.subplots(3, len(z_slices),
                             figsize=(3.1 * len(z_slices), 9.4),
                             facecolor="white")
    for c, z in enumerate(z_slices):
        pts = np.column_stack([X.ravel(), Y.ravel(),
                               np.full(X.size, z)]).astype(np.float32)
        d, _ = tree.query(pts, workers=-1)
        panels = [np.exp(-d ** 2 / (2 * sigma ** 2)), sim0[c], sim1[c]]
        for r, (img, name) in enumerate(zip(panels,
                                            ["target", "initial", "learned"])):
            ax = axes[r, c]
            ax.imshow(img.reshape(res, res), origin="lower", cmap="Blues",
                      vmin=0, vmax=1, extent=(-1, 1, -1, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"z = {z:+.2f}", fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", type=str, default="data/meshes/bunny.ply")
    ap.add_argument("--n-cloud", type=int, default=2048)
    ap.add_argument("--n-dense", type=int, default=200000)
    ap.add_argument("--sigma", type=float, default=0.04,
                    help="target surface-shell width")
    ap.add_argument("--n-scales", type=int, default=6)
    ap.add_argument("--n-rotates", type=int, default=8)
    ap.add_argument("--length-scale", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--w0", type=float, default=15.0)
    ap.add_argument("--dphase-scale", type=float, default=30.0)
    ap.add_argument("--pool", type=int, default=1000000,
                    help="training points per half (uniform / near-surface)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--reg-da", type=float, default=1e-6)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--slice-res", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/bunny_kernel_fit")
    args = ap.parse_args()

    out = FSPath(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    cloud, dense = load_cloud(args.mesh, args.n_cloud, args.n_dense, args.seed)
    tree = cKDTree(dense)
    print(f"cloud {cloud.shape}, dense target sample {dense.shape}, "
          f"extents {np.round(cloud.max(0) - cloud.min(0), 2)}")

    ssp_space = HexagonalSSPSpace(domain_dim=3, n_scales=args.n_scales,
                                  n_rotates=args.n_rotates,
                                  length_scale=args.length_scale, rng=args.seed)
    d = ssp_space.ssp_dim
    nf = (d - 1) // 2
    a_base = jnp.asarray(ssp_space.phase_matrix[1:nf + 1]
                         / ssp_space.length_scale.flatten(), jnp.float32)

    model = PhaseNet(hidden=tuple(args.hidden), n_free=nf, dim=3, w0=args.w0)
    net = model.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, 3)))
    params = {"net": net, "alpha": jnp.array(1.0), "beta": jnp.array(0.0)}
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(net))
    print(f"ssp_dim={d} (n_free={nf}), mlp params={n_params}")

    sim_fn = make_sim_fn(model, a_base, d, args.dphase_scale,
                         jnp.asarray(cloud))

    pool_pts, pool_t = make_pool(tree, dense, args.sigma, args.pool, rng)
    print(f"pool {pool_pts.shape}, target mean {pool_t.mean():.3f}")

    # held-out eval set + baseline (zero-init net = standard HexSSP bundle)
    eval_pts, eval_t = make_pool(tree, dense, args.sigma, 100000,
                                 np.random.default_rng(args.seed + 99))
    base_raw = eval_chunked(sim_fn, params, eval_pts)
    base = affine_refit(base_raw, eval_t)
    cos0, r20 = metrics(base, eval_t)

    params = train(model, sim_fn, params, pool_pts, pool_t, args)

    pred = eval_chunked(sim_fn, params, eval_pts)
    cos1, r21 = metrics(pred, eval_t)
    print(f"baseline (hex bundle): cos={cos0:.3f} R2={r20:.3f}\n"
          f"learned  A(x) bundle : cos={cos1:.3f} R2={r21:.3f}")

    # slice figures through the bunny
    z_slices = np.quantile(cloud[:, 2], [0.2, 0.45, 0.7, 0.9])
    res = args.slice_res
    cc = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(cc, cc)
    sim0_sl, sim1_sl = [], []
    net0 = model.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, 3)))
    params0 = {"net": net0, "alpha": params["alpha"], "beta": params["beta"]}
    for z in z_slices:
        pts = np.column_stack([X.ravel(), Y.ravel(),
                               np.full(X.size, z)]).astype(np.float32)
        raw0 = eval_chunked(sim_fn, params0, pts)
        d_sl, _ = tree.query(pts, workers=-1)
        t_sl = np.exp(-d_sl ** 2 / (2 * args.sigma ** 2))
        sim0_sl.append(affine_refit(raw0, t_sl))
        sim1_sl.append(eval_chunked(sim_fn, params, pts))
    fig_slices(z_slices, res, tree, args.sigma, sim0_sl, sim1_sl,
               out / "bunny_fit.png")

    (out / "params.msgpack").write_bytes(serialization.to_bytes(params))
    print(f"wrote {out / 'params.msgpack'}")


if __name__ == "__main__":
    main()
