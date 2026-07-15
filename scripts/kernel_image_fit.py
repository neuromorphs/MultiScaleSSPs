#!/usr/bin/env python
"""Fit the SSP self-similarity kernel to a target image by learning A(x).

A standard SSP encodes a point as phi(x) = ifft(exp(i A x / ls)) with a fixed
phase matrix A, so the similarity map s(x) = phi(0) . phi(x) is a stationary
sinc-like kernel centered on the origin:

    s(x) = (1 + 2 sum_k cos(a_k . x)) / d,   a_k = A[k] / ls  (half spectrum)

Here we make the phase matrix a function of position, A(x) = A_hex + dA(x),
where dA is produced by a small flax MLP taking the (normalized) pixel
location as input. The final layer is zero-initialized, so at step 0 the
similarity map is exactly the standard HexSSP kernel; training minimizes the
MSE between s(x) and a grayscale target image over pixel locations.

Outputs (to --out):
    kernel_fit.png    target | initial kernel | learned similarity | error
    loss_curve.png    full-grid MSE during training
    freq_field.png    mean row norm of A(x): the local frequency magnitude
    params.msgpack    trained flax parameters
    sim_maps.npz      target, initial, and learned similarity maps

Example:
    python scripts/kernel_image_fit.py --image fractal_target.jpg
"""

import argparse
from pathlib import Path as FSPath

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import jax

# TF32 matmuls + an ill-conditioned net make train/eval arithmetic diverge;
# huge learned phases mean rounding noise in theta shifts cos() visibly.
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import optax
import flax.linen as nn
from flax import serialization

from vsagym.spaces import HexagonalSSPSpace


# ---------------------------------------------------------------------------
# Target image + coordinate grid
# ---------------------------------------------------------------------------


def load_target(path, res):
    """Grayscale target in [0, 1], flipped so row 0 is the bottom (y up)."""
    img = Image.open(path).convert("L")
    if res and res != img.size[0]:
        img = img.resize((res, res), Image.LANCZOS)
    t = np.asarray(img, np.float32) / 255.0
    return t[::-1].copy()


def pixel_grid(res):
    """Pixel-center coordinates in [-1, 1]^2, shape (res*res, 2), row-major
    from the bottom to match load_target."""
    c = (np.arange(res) + 0.5) / res * 2 - 1
    X, Y = np.meshgrid(c, c)
    return np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)


# ---------------------------------------------------------------------------
# Similarity: s(x) = phi(0) . phi(x) = (1 + 2 sum_k cos(a_k(x) . x)) / d
# ---------------------------------------------------------------------------


def _siren_first_init(key, shape, dtype=jnp.float32):
    # SIREN first layer: U(-1/fan_in, 1/fan_in), pre-activation scaled by w0
    return jax.random.uniform(key, shape, dtype, -1.0, 1.0) / shape[0]


# SIREN hidden layers: U(-sqrt(6/fan_in), sqrt(6/fan_in))
_siren_hidden_init = nn.initializers.variance_scaling(2.0, "fan_in", "uniform")


class PhaseNet(nn.Module):
    """xy in [-1,1]^2 -> dA(x), the residual half-spectrum phase matrix.

    act='sin' is a SIREN (Sitzmann 2020): sin(w0 W x + b) on the first layer,
    sin on hidden layers, with the matching uniform inits. Zero-init head
    either way: A(x) = A_hex at initialization, so training starts from the
    standard SSP kernel."""

    hidden: tuple
    n_free: int
    act: str = "sin"
    w0: float = 30.0

    @nn.compact
    def __call__(self, xy):
        h = xy
        for i, w in enumerate(self.hidden):
            if self.act == "sin":
                init = _siren_first_init if i == 0 else _siren_hidden_init
                h = nn.Dense(w, kernel_init=init)(h)
                h = jnp.sin(self.w0 * h if i == 0 else h)
            else:
                h = nn.tanh(nn.Dense(w)(h))
        dA = nn.Dense(2 * self.n_free, kernel_init=nn.initializers.zeros,
                      bias_init=nn.initializers.zeros)(h)
        return dA.reshape(xy.shape[:-1] + (self.n_free, 2))


def make_sim_fn(model, a_base, ssp_dim, dphase_scale):
    """sim(params, xy) -> (similarity with phi(0), mean dA^2) at each xy."""

    def sim(params, xy):
        dA = dphase_scale * model.apply(params, xy)           # (..., n_free, 2)
        theta = jnp.einsum("...kj,...j->...k", a_base + dA, xy)
        return (1.0 + 2.0 * jnp.cos(theta).sum(-1)) / ssp_dim, jnp.mean(dA ** 2)

    return sim


def eval_full(sim_fn, params, xy, chunk=60000):
    sim_jit = jax.jit(lambda p, x: sim_fn(p, x)[0])
    out = [jax.device_get(sim_jit(params, xy[i:i + chunk]))
           for i in range(0, xy.shape[0], chunk)]
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(model, sim_fn, params, xy, target, args):
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)
    xy_j, t_j = jnp.asarray(xy), jnp.asarray(target.ravel())

    def loss_fn(params, bxy, bt):
        sim, da2 = sim_fn(params, bxy)
        return jnp.mean((sim - bt) ** 2) + args.reg_da * da2

    @jax.jit
    def step(params, opt_state, key):
        idx = jax.random.randint(key, (args.batch,), 0, xy_j.shape[0])
        loss, grads = jax.value_and_grad(loss_fn)(params, xy_j[idx], t_j[idx])
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    key = jax.random.PRNGKey(args.seed)
    curve = []
    for i in range(args.steps):
        key, sub = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, sub)
        if (i + 1) % args.eval_every == 0 or i == 0:
            full = eval_full(sim_fn, params, xy_j)
            mse = float(np.mean((full - target.ravel()) ** 2))
            curve.append((i + 1, mse))
            print(f"step {i + 1:5d}  batch loss {float(loss):.5f}  full MSE {mse:.5f}")
    return params, curve


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

EXTENT = (-1, 1, -1, 1)


def _panel(ax, data, title, cmap="gray", vmin=None, vmax=None):
    im = ax.imshow(data, origin="lower", extent=EXTENT, cmap=cmap,
                   vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def fig_maps(target, sim0, sim1, path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    _panel(axes[0], target, "target image", vmin=0, vmax=1)
    _panel(axes[1], sim0, r"initial: $\phi(0)\cdot\phi(x,y)$, HexSSP")
    _panel(axes[2], sim1, r"learned: $A(x)$ from MLP", vmin=0, vmax=1)
    im = _panel(axes[3], np.abs(sim1 - target), "|error|", cmap="magma")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_loss(curve, path):
    steps, mses = zip(*curve)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.semilogy(steps, mses)
    ax.set_xlabel("step"); ax.set_ylabel("full-grid MSE")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def fig_freq_field(model, params, a_base, dphase_scale, xy, res, path):
    def row_norm(bxy):
        A = a_base + dphase_scale * model.apply(params, bxy)
        return jnp.linalg.norm(A, axis=-1).mean(-1)

    vals = np.concatenate([jax.device_get(row_norm(jnp.asarray(xy[i:i + 65536])))
                           for i in range(0, xy.shape[0], 65536)])
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = _panel(ax, vals.reshape(res, res), r"mean $\|a_k(x)\|$ (local frequency)",
                cmap="viridis")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=str, default="fractal_target.jpg")
    ap.add_argument("--res", type=int, default=600, help="train/render resolution")
    ap.add_argument("--n-scales", type=int, default=8)
    ap.add_argument("--n-rotates", type=int, default=8)
    ap.add_argument("--length-scale", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    ap.add_argument("--act", choices=["sin", "tanh"], default="sin")
    ap.add_argument("--w0", type=float, default=30.0,
                    help="SIREN first-layer frequency scale")
    ap.add_argument("--dphase-scale", type=float, default=30.0,
                    help="multiplier on the MLP output (units: rad per coord unit)")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--reg-da", type=float, default=1e-6,
                    help="L2 penalty on the learned phase residual dA")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/kernel_image_fit")
    args = ap.parse_args()

    out = FSPath(args.out)
    out.mkdir(parents=True, exist_ok=True)

    target = load_target(args.image, args.res)
    xy = pixel_grid(args.res)

    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=args.n_scales,
                                  n_rotates=args.n_rotates,
                                  length_scale=args.length_scale, rng=args.seed)
    d = ssp_space.ssp_dim
    n_free = (d - 1) // 2
    a_base = jnp.asarray(ssp_space.phase_matrix[1:n_free + 1]
                         / ssp_space.length_scale.flatten(), jnp.float32)

    model = PhaseNet(hidden=tuple(args.hidden), n_free=n_free, act=args.act,
                     w0=args.w0)
    params = model.init(jax.random.PRNGKey(args.seed), jnp.zeros((1, 2)))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"ssp_dim={d} (n_free={n_free}), mlp params={n_params}, "
          f"pixels={xy.shape[0]}")

    sim_fn = make_sim_fn(model, a_base, d, args.dphase_scale)

    # sanity check: zero-init MLP must reproduce the vsagym kernel exactly
    sim0 = eval_full(sim_fn, params, jnp.asarray(xy)).reshape(args.res, args.res)
    probe = xy[:: xy.shape[0] // 7]
    ref = (ssp_space.encode(probe) @ ssp_space.encode(np.zeros((1, 2))).T).ravel()
    got = np.asarray(sim_fn(params, jnp.asarray(probe))[0])
    assert np.abs(ref - got).max() < 1e-4, "init kernel != vsagym encode"
    print(f"init kernel matches vsagym encode (max dev {np.abs(ref - got).max():.2e})")

    params, curve = train(model, sim_fn, params, xy, target, args)

    sim1 = eval_full(sim_fn, params, jnp.asarray(xy)).reshape(args.res, args.res)
    mse = float(np.mean((sim1 - target) ** 2))
    print(f"final MSE {mse:.5f}  (PSNR {-10 * np.log10(mse):.1f} dB)  "
          f"cos sim {np.dot(sim1.ravel(), target.ravel()) / (np.linalg.norm(sim1) * np.linalg.norm(target)):.4f}")

    fig_maps(target, sim0, sim1, out / "kernel_fit.png")
    fig_loss(curve, out / "loss_curve.png")
    fig_freq_field(model, params, a_base, args.dphase_scale, xy, args.res,
                   out / "freq_field.png")
    (out / "params.msgpack").write_bytes(serialization.to_bytes(params))
    np.savez_compressed(out / "sim_maps.npz", target=target, initial=sim0,
                        learned=sim1)
    print(f"wrote {out / 'params.msgpack'} and {out / 'sim_maps.npz'}")


if __name__ == "__main__":
    main()
