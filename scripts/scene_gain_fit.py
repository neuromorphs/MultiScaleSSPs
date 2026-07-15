#!/usr/bin/env python
"""Encode scene occupancy as a bundled SSP and fit the best scale gains.

Every occupied cell x_i of each scene (see make_scene_maps.py) is encoded and
bundled: M = sum_i phi(x_i). In the Fourier domain that is one complex
coefficient b_k = sum_i e^{i a_k . x_i} per frequency row, so the decoded
similarity with per-scale gains g,

    sim_g(x') = sum_s g_s B_s(x'),   B_s(x') = sum_{k in scale s} Re[e^{i a_k . x'} conj(b_k)]

is linear in g. The best non-negative gains (plus a DC/bias term, which is
the gain on the constant Fourier row) are then the exact solution of a tiny
NNLS problem against the binary occupancy map -- no training loop.

The SSP space matches the gain widget: HexSSP, 7 log scales pi/27..pi,
15 rotations, length_scale 1/3 (d = 631; effective band pi/9..3*pi rad/m).

Outputs: results/widgets/scene_gain_fit.png (occupancy | uniform gains |
learned gains, per scene), and scene_gains.npz with the learned gains.
"""

import argparse
from pathlib import Path as FSPath

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls

from vsagym.spaces import HexagonalSSPSpace
from make_scene_maps import build_scenes


def bundle_coeffs(a_half, pts):
    """b_k = sum_i e^{i a_k . x_i} for every half-spectrum row."""
    return np.exp(1j * pts @ a_half.T).sum(axis=0)


def wall_segments(rects):
    """Centerline segment ((x0,y0), (x1,y1)) of each wall rectangle."""
    segs = []
    for x, y, w, h in rects:
        if w >= h:
            segs.append(((x, y + h / 2), (x + w, y + h / 2)))
        else:
            segs.append(((x + w / 2, y), (x + w / 2, y + h)))
    return segs


def segment_bundle_coeffs(a_half, segments):
    """Exact line-integral encoding of segments: since int e^{iax} dx has a
    closed form, int_seg e^{i a . x} dl = L e^{i a . mid} sinc(a . u L / 2)
    (u = unit direction; np.sinc includes the pi factor)."""
    b = np.zeros(len(a_half), complex)
    for p0, p1 in segments:
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = np.linalg.norm(p1 - p0)
        u = (p1 - p0) / L
        mid = (p0 + p1) / 2
        b += L * np.exp(1j * a_half @ mid) * np.sinc(a_half @ u * L / (2 * np.pi))
    return b


def scale_basis_maps(a_half, row_s, ns, b, grid):
    """B_s(x') = sum_{k in s} Re[e^{i a_k x'} conj(b_k)], stacked (ns, n_grid)."""
    contrib = np.exp(1j * grid @ a_half.T) * np.conj(b)   # (n_grid, n_rows)
    return np.stack([contrib[:, row_s == s].real.sum(axis=1)
                     for s in range(ns)])


def fit_gains(B, occ):
    """Non-negative gains + DC bias minimizing ||g.B + b - occ||^2."""
    A = np.column_stack([B.T, np.ones(B.shape[1])])
    sol, _ = nnls(A, occ.astype(float))
    return sol[:-1], sol[-1]


def uniform_fit(B, occ):
    """Best (scale alpha, bias) for the g = ones map, for a fair baseline."""
    s1 = B.sum(axis=0)
    A = np.column_stack([s1, np.ones_like(s1)])
    (alpha, b), *_ = np.linalg.lstsq(A, occ.astype(float), rcond=None)
    return alpha * s1 + b


def metrics(pred, occ):
    o = occ.astype(float)
    cos = pred @ o / (np.linalg.norm(pred) * np.linalg.norm(o))
    r2 = 1 - np.sum((pred - o) ** 2) / np.sum((o - o.mean()) ** 2)
    return cos, r2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scales", type=int, default=7)
    ap.add_argument("--n-rotates", type=int, default=15)
    ap.add_argument("--scale-min", type=float, default=np.pi / 27)
    ap.add_argument("--length-scale", type=float, default=1 / 3,
                    help="nominal scales stay <= pi; ls=1/3 shifts the "
                         "effective band to pi/9..3*pi rad/m")
    ap.add_argument("--grid-n", type=int, default=220)
    ap.add_argument("--sim-floor", type=float, default=0.25,
                    help="display floor for sim maps, fraction of the max")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="results/widgets")
    args = ap.parse_args()

    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=args.n_scales,
                                  n_rotates=args.n_rotates,
                                  scale_min=args.scale_min,
                                  scale_sampling="log",
                                  length_scale=args.length_scale, rng=args.seed)
    d = ssp_space.ssp_dim
    n_free = (d - 1) // 2
    a_half = ssp_space.phase_matrix[1:n_free + 1] / ssp_space.length_scale.flatten()
    nv, ns = ssp_space.grid_basis_dim, ssp_space.n_scales
    row_s = (np.arange(len(a_half)) // nv) % ns
    scales = np.asarray(ssp_space.scales)
    print(f"ssp_dim={d}, scales={np.round(scales, 3)}")

    scenes = build_scenes(args.grid_n)
    world, gn = scenes["world"], args.grid_n
    cc = (np.arange(gn) + 0.5) / gn * world
    X, Y = np.meshgrid(cc, cc)
    grid = np.column_stack([X.ravel(), Y.ravel()])

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.8), facecolor="white")
    results = {}
    for row, key in enumerate(("indoor", "outdoor")):
        occ = scenes[key]["occ"].ravel()
        pts = grid[occ]
        if key == "indoor":
            # walls: exact line-integral encoding of the centerlines
            segs = wall_segments(scenes[key]["rects"])
            b = segment_bundle_coeffs(a_half, segs)
            src = f"{len(segs)} wall line integrals"
        else:
            b = bundle_coeffs(a_half, pts)
            src = f"{len(pts)} bundled points"
        B = scale_basis_maps(a_half, row_s, ns, b, grid)

        g, bias = fit_gains(B, occ)
        pred = g @ B + bias
        base = uniform_fit(B, occ)
        cos_g, r2_g = metrics(pred, occ)
        cos_1, r2_1 = metrics(base, occ)
        results[key] = g
        gtxt = np.round(g / g.max(), 2)
        print(f"{key:8s} [{src}]  gains(coarse->fine)={gtxt} "
              f"(bias {bias:.3f})\n"
              f"         uniform: cos={cos_1:.3f} R2={r2_1:.3f}   "
              f"learned: cos={cos_g:.3f} R2={r2_g:.3f}")

        panels = [(occ, "occupancy (target)", "gray_r", (0, 1)),
                  (base, f"uniform gains  cos={cos_1:.3f}", "Blues", None),
                  (pred, f"learned gains  cos={cos_g:.3f}", "Blues", None)]
        for col, (img, title, cmap, vr) in enumerate(panels):
            ax = axes[row, col]
            if vr is None:   # floor the display so low-sim noise reads white
                vmax = np.quantile(img, 0.995)
                vr = (args.sim_floor * vmax, vmax)
            ax.imshow(img.reshape(gn, gn), origin="lower", cmap=cmap,
                      vmin=vr[0], vmax=vr[1], extent=(0, world, 0, world))
            ax.set_title(f"{key}: {title}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 2].text(0.02, 0.02, "g = " + str(gtxt.tolist()),
                          transform=axes[row, 2].transAxes, fontsize=8,
                          color="#444", va="bottom")

    fig.tight_layout()
    out = FSPath(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "scene_gain_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    np.savez(out / "scene_gains.npz", scales=scales,
             indoor=results["indoor"], outdoor=results["outdoor"])
    print(f"wrote {out / 'scene_gain_fit.png'} and {out / 'scene_gains.npz'}")


if __name__ == "__main__":
    main()
