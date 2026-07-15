#!/usr/bin/env python
"""Animate a z-slice scan through the bunny bundle similarity volume.

Left: the standard HexSSP bundle of the 2048-point cloud (with its best
affine readout). Right: the learned A(x) bundle of bunny_kernel_fit.py.
The slice sweeps bottom -> top -> bottom in a ping-pong loop.

Run scripts/bunny_kernel_fit.py first (needs its params.msgpack).

    python scripts/bunny_scan_gif.py
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # geometry/space/model args must match the bunny_kernel_fit run
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
    ap.add_argument("--slice-res", type=int, default=220)
    ap.add_argument("--n-slices", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif-stride", type=int, default=2)
    ap.add_argument("--gif-scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/animations")
    args = ap.parse_args()

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
    params = serialization.from_bytes(
        params0, FSPath(args.params).read_bytes())
    sim_fn = make_sim_fn(model, a_base, d, args.dphase_scale,
                         jnp.asarray(cloud))

    # global affine readout for the baseline, fit once on held-out points
    tree = cKDTree(dense)
    eval_pts, eval_t = make_pool(tree, dense, args.sigma, 100000,
                                 np.random.default_rng(args.seed + 99))
    raw = eval_chunked(sim_fn, params0, eval_pts)
    (al, be), *_ = np.linalg.lstsq(
        np.column_stack([raw, np.ones_like(raw)]), eval_t, rcond=None)
    print(f"baseline affine: alpha={al:.1f} beta={be:.2f}")

    # precompute both slice stacks
    res = args.slice_res
    cc = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(cc, cc)
    zs = np.linspace(cloud[:, 2].min() - 0.06, cloud[:, 2].max() + 0.06,
                     args.n_slices)
    maps0, maps1 = [], []
    for f, z in enumerate(zs):
        pts = np.column_stack([X.ravel(), Y.ravel(),
                               np.full(X.size, z)]).astype(np.float32)
        maps0.append(np.clip(al * eval_chunked(sim_fn, params0, pts) + be,
                             0, 1).reshape(res, res))
        maps1.append(np.clip(eval_chunked(sim_fn, params, pts),
                             0, 1).reshape(res, res))
        if f % 30 == 0:
            print(f"  slice {f}/{len(zs)}")

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11.2, 5.6), dpi=120,
                                   facecolor=COLORS["background"])
    ims = []
    for ax, title in ((axl, "initial: HexSSP bundle"),
                      (axr, "learned $A(x)$ bundle")):
        ims.append(ax.imshow(np.zeros((res, res)), origin="lower",
                             cmap=COLORS["sim_cmap"], vmin=0, vmax=1,
                             extent=(-1, 1, -1, 1)))
        ax.set_title(title, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(COLORS["frame_edge"])
    ztxt = axl.text(-0.94, 0.88, "", fontsize=11, color="0.35")
    fig.tight_layout(pad=0.6)

    order = list(range(len(zs))) + list(range(len(zs) - 2, 0, -1))

    def frames():
        for i in order:
            ims[0].set_data(maps0[i])
            ims[1].set_data(maps1[i])
            ztxt.set_text(f"z = {zs[i]:+.2f}")
            yield frame_of(fig)

    out = FSPath(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_outputs(frames(), len(order), out / "bunny_scan", args)
    plt.close(fig)


if __name__ == "__main__":
    main()
