#!/usr/bin/env python
"""ID-tagged blob maps per quadrant: adaptive-resolution direct-optim decoding.

Extends scripts/quadrant_scale_modulation.py (same blob world and learned
per-quadrant scale gains). Instead of representing each quadrant as a mean of
location encodings, each blob gets a UNIQUE random unitary semantic pointer
(an ID, not spatial) bound (circular convolution) to the modulated SSP of its
center; the quadrant map is the bundle (sum):

    M_q = sum_i  ID_i (*) modulate_g_q( SSP(c_i) )

Each blob's location is recovered by unbinding, M_q (*) ID_i^{-1}, then
decoding with the library's 'direct-optim' method (vsagym sspspace.decode):
a coarse grid argmax followed by L-BFGS-B refinement. The grid density uses
the library's length-scale rule (sspspace.py get_sample_points,
'length-scale' branch):

    num_pts_per_dim = 2 * ceil(domain_range / length_scale)

but with a PER-QUADRANT effective length scale derived from the learned
gains. Definitions and compute-accounting assumptions:

  1. Effective length scale l_eff := FWHM of the modulated similarity kernel
     K_g(delta) = <enc_g(x), enc_g(x + delta)> (unit-norm encodings,
     direction-averaged, measured numerically). With the library rule the
     grid spacing is ~l_eff/2 = the kernel's half-max radius, so some coarse
     grid point always lands inside the global peak's central lobe -- the
     basin the L-BFGS-B refinement needs to start in. No gain thresholds.
  2. Baseline "single fine resolution": the FINEST learned quadrant profile
     applied to every quadrant -- without location-dependent modulation, one
     fixed representation must resolve the smallest blobs anywhere. Two grid
     sizings of it are reported: (a) 'measured kernel' -- the same FWHM rule
     (generous: assumes the kernel width was measured); (b) 'spectral content'
     -- spacing that resolves the finest Fourier component, l = ls/scale_max
     (the safe worst-case sizing when the kernel width is unknown, matching
     the library rule's semantics of "2 points per length scale"). Identical
     code path everywhere; only the gains and grids differ.
  3. Modulated encoding costs the same as plain encoding: the gain spectrum
     folds into the Fourier encoding, enc_g(x) = ifft(g_full * exp(iAx/ls)),
     one length-d inverse FFT + O(d) multiplies either way (verified equal to
     modulate-after-encode to machine precision).
  4. Compute counted per quadrant: one-time decode-grid encoding (amortized
     over that quadrant's queries), per-query similarity matvec (2*N*d flops),
     and L-BFGS-B refinement (actual scipy nfev, finite-difference gradients;
     each eval = one encode + one dot). FLOP model: encode ~ 5*d*log2(d) (FFT)
     + 6*d (phase/gain multiplies). Wall-clock is also measured directly.
  5. Both arms share the same IDs and per-quadrant bundles (n_q bound pairs),
     so binding crosstalk noise -- which scales like sqrt(n_q/d) -- is matched.

Outputs: decode scatter figure ('.' true vs 'x' decoded), per-quadrant
accuracy + compute tables (stdout, CSV, JSON).

Example:
    python scripts/quadrant_id_map_decode.py --out results/scale_modulation/id_map/quadrant_id_map.png
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path as FSPath

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.optimize import minimize

import jax.numpy as jnp
from vsagym.spaces import HexagonalSSPSpace

sys.path.insert(0, str(FSPath(__file__).parent))
from shape_scale_modulation import make_scale_index  # noqa: E402
from ssp_modulation_animation import make_blobs  # noqa: E402
from quadrant_scale_modulation import QUADS, QuadrantData, train_gains  # noqa: E402


# ---------------------------------------------------------------------------
# Modulated encoding with the gains folded into the Fourier transform
# (single ifft; equals modulate_spectral(encode(x)) to machine precision)
# ---------------------------------------------------------------------------


def full_gain_spectrum(g, scale_idx):
    """Per-scale gains (n_scales,) -> full-spectrum gains (d,), DC = 1,
    conjugate-symmetric (odd d)."""
    gk = np.asarray(g, dtype=np.float64)[scale_idx]
    return np.concatenate([[1.0], gk, gk[::-1]])


def encode_mod(ssp_space, g_full, pts, chunk=4096):
    """Unit-norm modulated encodings of pts, (n, d). One FFT per point."""
    pts = np.atleast_2d(pts)
    scaled = (pts / ssp_space.length_scale.flatten()).T
    out = np.empty((len(pts), ssp_space.ssp_dim))
    for i in range(0, len(pts), chunk):
        ph = np.exp(1j * ssp_space.phase_matrix @ scaled[:, i : i + chunk])
        out[i : i + chunk] = np.fft.ifft(g_full[:, None] * ph, axis=0).real.T
    return out / np.maximum(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12)


def kernel_fwhm(ssp_space, g_full, r_max=2.5, n_r=1000, n_dirs=8):
    """FWHM of the direction-averaged modulated similarity kernel."""
    rs = np.linspace(0.0, r_max, n_r)
    thetas = np.linspace(0.0, np.pi, n_dirs, endpoint=False)
    v0 = encode_mod(ssp_space, g_full, np.zeros((1, 2)))[0]
    K = np.zeros(n_r)
    for th in thetas:
        pts = np.outer(rs, [np.cos(th), np.sin(th)])
        K += encode_mod(ssp_space, g_full, pts) @ v0
    K /= n_dirs
    below = np.nonzero(K < 0.5 * K[0])[0]
    assert len(below), "kernel never drops below half max within r_max"
    return 2.0 * rs[below[0]]


def pts_per_dim(domain_range, l_eff):
    """Library grid rule (sspspace.py get_sample_points, 'length-scale')."""
    return 2 * int(np.ceil(domain_range / l_eff))


# ---------------------------------------------------------------------------
# direct-optim decode of one unbound query (library method, custom encoding)
# ---------------------------------------------------------------------------


def direct_optim_decode(query, ssp_space, g_full, bounds, grid_pts, mod_grid_ssps):
    """Coarse grid argmax + L-BFGS-B refinement of -<enc_g(x), query>.
    Returns (x_hat, nfev)."""
    q = query / max(np.linalg.norm(query), 1e-6)
    x0 = grid_pts[int(np.argmax(mod_grid_ssps @ q))]

    def neg_sim(x):
        return -float(encode_mod(ssp_space, g_full, x).ravel() @ q)

    soln = minimize(neg_sim, x0, method="L-BFGS-B", bounds=bounds)
    return soln.x, int(soln.nfev)


# ---------------------------------------------------------------------------
# Per-quadrant experiment
# ---------------------------------------------------------------------------


def decode_quadrant(ssp_space, g, scale_idx, q, ids, l_eff):
    """Build the ID-bound map of one quadrant with gains g, decode every blob
    with a grid sized by l_eff. Returns per-blob results + compute tally."""
    d = ssp_space.ssp_dim
    centers = q.blobs[:, :2]
    n_q = len(centers)
    g_full = full_gain_spectrum(g, scale_idx)
    x0, x1, y0, y1 = q.bounds
    n_pts = pts_per_dim(x1 - x0, l_eff)

    t0 = time.perf_counter()
    # map: bundle of ID (*) modulated location SSP
    V = encode_mod(ssp_space, g_full, centers)
    M = ssp_space.bind(ids, V).sum(axis=0)

    # decode grid (one-time per quadrant, amortized over its queries)
    X, Y = np.meshgrid(np.linspace(x0, x1, n_pts), np.linspace(y0, y1, n_pts))
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])
    mod_grid = encode_mod(ssp_space, g_full, grid_pts).astype(np.float32)

    decoded = np.empty_like(centers)
    nfev_tot = 0
    queries = ssp_space.bind(M[None], ssp_space.invert(ids))  # unbind all IDs
    for i in range(n_q):
        decoded[i], nfev = direct_optim_decode(
            queries[i], ssp_space, g_full, [(x0, x1), (y0, y1)], grid_pts, mod_grid)
        nfev_tot += nfev
    wall = time.perf_counter() - t0

    errs = np.linalg.norm(decoded - centers, axis=1)
    n_grid = n_pts ** 2
    enc_flops = 5 * d * np.log2(d) + 6 * d
    flops = dict(
        grid_encode=n_grid * enc_flops,
        sims=n_q * n_grid * 2 * d,
        refine=nfev_tot * (enc_flops + 2 * d),
    )
    return dict(
        name=q.name, bounds=q.bounds, n_blobs=n_q, l_eff=l_eff,
        n_pts_per_dim=n_pts, n_grid=n_grid, nfev=nfev_tot, wall=wall,
        centers=centers, decoded=decoded, errs=errs, flops=flops,
        flops_total=float(sum(flops.values())),
    )


def summarize(results, label):
    errs = np.concatenate([r["errs"] for r in results])
    print(f"\n=== {label} ===")
    print(f"{'quadrant':<16} {'l_eff':>6} {'pts/dim':>8} {'grid':>7} {'nfev':>6} "
          f"{'med err':>8} {'max err':>8} {'<0.25':>6} {'GFLOP':>7} {'wall s':>7}")
    for r in results:
        e = r["errs"]
        print(f"{r['name']:<16} {r['l_eff']:>6.3f} {r['n_pts_per_dim']:>8d} "
              f"{r['n_grid']:>7d} {r['nfev']:>6d} {np.median(e):>8.4f} "
              f"{e.max():>8.4f} {np.mean(e < 0.25) * 100:>5.0f}% "
              f"{r['flops_total'] / 1e9:>7.2f} {r['wall']:>7.2f}")
    tot_flops = sum(r["flops_total"] for r in results)
    tot_wall = sum(r["wall"] for r in results)
    tot_grid = sum(r["n_grid"] for r in results)
    print(f"{'TOTAL':<16} {'':>6} {'':>8} {tot_grid:>7d} "
          f"{sum(r['nfev'] for r in results):>6d} {np.median(errs):>8.4f} "
          f"{errs.max():>8.4f} {np.mean(errs < 0.25) * 100:>5.0f}% "
          f"{tot_flops / 1e9:>7.2f} {tot_wall:>7.2f}")
    print(f"overall: RMSE={np.sqrt(np.mean(errs ** 2)):.4f}  "
          f"mean={errs.mean():.4f}  acc(<0.25)={np.mean(errs < 0.25) * 100:.1f}%")
    return dict(rmse=float(np.sqrt(np.mean(errs ** 2))), mean=float(errs.mean()),
                median=float(np.median(errs)), max=float(errs.max()),
                acc_025=float(np.mean(errs < 0.25)),
                flops=tot_flops, wall=tot_wall, grid_pts=tot_grid)


def fig_decode(results_by_label, blobs, path):
    labels = list(results_by_label)
    fig, axs = plt.subplots(1, len(labels) + 1, figsize=(6.4 * len(labels) + 5.4, 6.2),
                            facecolor="white")
    for ax, label in zip(axs, labels):
        results = results_by_label[label]
        for cx, cy, r in blobs:
            ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor="0.8", lw=1.0))
        ax.axhline(5.0, color="0.85", lw=0.8, zorder=0)
        ax.axvline(5.0, color="0.85", lw=0.8, zorder=0)
        errs = np.concatenate([r["errs"] for r in results])
        for ri, r in enumerate(results):
            for c, dxy in zip(r["centers"], r["decoded"]):
                ax.plot([c[0], dxy[0]], [c[1], dxy[1]], color="crimson", lw=0.8,
                        alpha=0.6, zorder=2)
            ax.plot(r["centers"][:, 0], r["centers"][:, 1], ".", color="black",
                    ms=7, ls="none", zorder=3, label="true" if ri == 0 else None)
            ax.plot(r["decoded"][:, 0], r["decoded"][:, 1], "x", color="crimson",
                    ms=7, mew=1.6, ls="none", zorder=4, label="decoded" if ri == 0 else None)
            x0, x1, y0, y1 = r["bounds"]
            ax.text((x0 + x1) / 2, y1 - 0.25,
                    f"$\\ell_{{eff}}$={r['l_eff']:.2f}, {r['n_pts_per_dim']}$^2$ pts",
                    ha="center", va="top", fontsize=9, color="0.35")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\nmedian err={np.median(errs):.3f}, "
                     f"acc(<0.25)={np.mean(errs < 0.25) * 100:.0f}%")
        ax.legend(loc="lower left", fontsize=9)

    # compute panel: decode FLOPs per quadrant
    ax = axs[-1]
    names = [r["name"].split(":")[0] for r in results_by_label[labels[0]]]
    xpos = np.arange(len(names))
    width = 0.8 / len(labels)
    colors = ["tab:blue", "0.6", "tab:red"]
    for li, label in enumerate(labels):
        vals = [r["flops_total"] / 1e9 for r in results_by_label[label]]
        ax.bar(xpos + (li - (len(labels) - 1) / 2) * width, vals, width,
               label=label, color=colors[li % len(colors)])
    ax.set_xticks(xpos)
    ax.set_xticklabels(names)
    ax.set_yscale("log")
    ax.set_ylabel("decode GFLOPs (grid encode + sims + refine)")
    ax.set_title("decode compute per quadrant")
    ax.legend(fontsize=9)
    fig.suptitle("ID (*) location maps: direct-optim decoding at learned vs single-fine resolution",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # winning config of quadrant_scale_modulation.py (centers + cosine, 10 log scales)
    ap.add_argument("--n-scales", type=int, default=10)
    ap.add_argument("--n-rotates", type=int, default=33)
    ap.add_argument("--scale-min", type=float, default=1.0)
    ap.add_argument("--scale-max", type=float, default=10.0)
    ap.add_argument("--scale-sampling", type=str, default="log", choices=["lin", "log", "rand"])
    ap.add_argument("--ls", type=float, default=0.9)
    ap.add_argument("--encode", choices=["samples", "centers"], default="centers")
    ap.add_argument("--pts-per-blob", type=int, default=30)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--loss", choices=["sse", "cosine"], default="cosine")
    ap.add_argument("--reg-w", type=float, default=0.003)
    ap.add_argument("--coarse-w", type=float, default=0.01)
    ap.add_argument("--grid-n", type=int, default=50, help="training target grid resolution")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--load-gains", type=str, default=None,
                    help="path to a saved *_gains.npz to skip training")
    ap.add_argument("--out", type=str,
                    default="results/scale_modulation/id_map/quadrant_id_map.png")
    args = ap.parse_args()

    out_path = FSPath(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    blobs = make_blobs(np.random.default_rng(args.seed))
    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=args.n_scales, n_rotates=args.n_rotates,
                                  scale_min=args.scale_min, scale_max=args.scale_max,
                                  scale_sampling=args.scale_sampling,
                                  length_scale=args.ls, rng=0)
    d = ssp_space.ssp_dim
    scale_idx = make_scale_index(ssp_space)
    print(f"ssp_dim={d} (n_scales={ssp_space.n_scales}, n_rotates={ssp_space.n_rotates}), "
          f"base ls={args.ls}, {len(blobs)} blobs")

    quads = [QuadrantData(name, b, blobs, ssp_space, args.grid_n) for name, b in QUADS.items()]

    gains_path = out_path.with_name(out_path.stem + "_gains.npz")
    if args.load_gains:
        g_quad = np.load(args.load_gains)["gains"]
        print(f"loaded per-quadrant gains from {args.load_gains}")
    else:
        print("training per-quadrant scale gains ...")
        t0 = time.time()
        g_quad, _ = train_gains(quads, ssp_space, jnp.asarray(scale_idx), args,
                                args.seed, shared=False)
        print(f"trained in {time.time() - t0:.0f}s")
        np.savez(gains_path, gains=g_quad, scales=np.asarray(ssp_space.scales))
        print(f"saved gains to {gains_path}")

    # effective length scale = measured FWHM of each modulated kernel
    l_effs = np.array([kernel_fwhm(ssp_space, full_gain_spectrum(g, scale_idx))
                       for g in g_quad])
    fine_qi = int(np.argmin(l_effs))
    g_fine, l_fine = g_quad[fine_qi], float(l_effs[fine_qi])
    for q, l in zip(quads, l_effs):
        print(f"  {q.name:<16} kernel FWHM = {l:.3f} -> {pts_per_dim(5.0, l)} pts/dim")
    print(f"single-fine baseline: {quads[fine_qi].name} profile everywhere "
          f"(FWHM {l_fine:.3f}, {pts_per_dim(5.0, l_fine)} pts/dim)")

    # one unique unitary ID per blob, partitioned by quadrant
    id_rng = np.random.default_rng(args.seed + 1000)
    all_ids = {q.name: ssp_space.make_unitary(id_rng.normal(size=(len(q.blobs), d)))
               for q in quads}

    l_spectral = args.ls / args.scale_max
    runs = {
        "adaptive (learned $\\ell_{eff}$)": [
            decode_quadrant(ssp_space, g_quad[qi], scale_idx, q, all_ids[q.name],
                            float(l_effs[qi])) for qi, q in enumerate(quads)],
        "single fine (measured kernel)": [
            decode_quadrant(ssp_space, g_fine, scale_idx, q, all_ids[q.name],
                            l_fine) for q in quads],
        "single fine (spectral content)": [
            decode_quadrant(ssp_space, g_fine, scale_idx, q, all_ids[q.name],
                            l_spectral) for q in quads],
    }

    summaries = {label: summarize(results, label) for label, results in runs.items()}
    ad = summaries["adaptive (learned $\\ell_{eff}$)"]
    print("\n=== compute comparison (assumptions in module docstring) ===")
    for label, s in summaries.items():
        if s is ad:
            continue
        print(f"adaptive vs {label}:")
        print(f"  grid points encoded: {ad['grid_pts']} vs {s['grid_pts']} "
              f"({s['grid_pts'] / ad['grid_pts']:.1f}x)")
        print(f"  estimated decode FLOPs: {ad['flops'] / 1e9:.2f} vs {s['flops'] / 1e9:.2f} GFLOP "
              f"({s['flops'] / ad['flops']:.1f}x)")
        print(f"  measured wall-clock: {ad['wall']:.2f}s vs {s['wall']:.2f}s "
              f"({s['wall'] / ad['wall']:.1f}x)")

    fig_decode(runs, blobs, out_path)

    rows = []
    for label, results in runs.items():
        for r in results:
            rows.append({"method": label, "quadrant": r["name"], "n_blobs": r["n_blobs"],
                         "l_eff": r["l_eff"], "pts_per_dim": r["n_pts_per_dim"],
                         "grid_pts": r["n_grid"], "nfev": r["nfev"],
                         "median_err": float(np.median(r["errs"])),
                         "max_err": float(r["errs"].max()),
                         "acc_0.25": float(np.mean(r["errs"] < 0.25)),
                         "gflops": r["flops_total"] / 1e9, "wall_s": r["wall"]})
    with open(out_path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump({"args": vars(args), "summaries": summaries, "rows": rows}, f, indent=2)
    print(f"wrote {out_path.with_suffix('.csv')} and .json")


if __name__ == "__main__":
    main()
