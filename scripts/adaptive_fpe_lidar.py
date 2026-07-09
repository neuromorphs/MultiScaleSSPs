#!/usr/bin/env python3
"""
Adaptive variable-scale positional encoding of a lidar point map into a
single superposed vector memory (fractional power encoding / RFF).

Pipeline
  1. Synthesize a dense 2D lidar-style point map.
  2. Adaptive tokenization: quadtree-subdivide until each cell's points are
     well approximated by ONE Gaussian token (mu, Sigma, weight=n_points).
     Straight walls collapse into a few elongated tokens; corners and
     clutter stay fine-grained. Split criterion is error-driven: compare
     the exact kernel field of the cell's points against the token's
     analytic field at probe locations.
  3. Encode each token as an attenuated phasor
         v_k = n_k * exp(-1/2 w_j^T Sigma_k w_j) * exp(i w_j . mu_k)
     (the VSA analogue of mip-NeRF integrated positional encoding)
     and bundle by summation into ONE D-dim complex memory vector.
  4. Evaluate: query memory at grid points, compare to the exact kernel
     density of the raw points. Baselines: naive all-points memory, and a
     fixed-scale voxel downsample given the SAME token budget.

Kernel math (Gaussian phases w ~ N(0, ell^-2 I)):
  point-point readout  ~ exp(-|dx|^2 / (2 ell^2))
  point-token readout  ~ n * ell^d/sqrt(det(ell^2 I + Sigma))
                           * exp(-1/2 dx^T (ell^2 I + Sigma)^-1 dx)
3D: change dims=3, quadtree -> octree. Everything else is unchanged.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import time

rng = np.random.default_rng(0)

# ----------------------------------------------------------------------
# 1. Synthetic lidar map
# ----------------------------------------------------------------------
def sample_segment(a, b, spacing, noise):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / spacing))
    t = np.linspace(0, 1, n)
    return a + t[:, None] * (b - a) + rng.normal(0, noise, (n, 2))

def sample_arc(c, r, th0, th1, spacing, noise):
    n = max(2, int(abs(th1 - th0) * r / spacing))
    th = np.linspace(th0, th1, n)
    pts = np.c_[c[0] + r * np.cos(th), c[1] + r * np.sin(th)]
    return pts + rng.normal(0, noise, pts.shape)

def build_map(spacing=0.008, noise=0.01):
    segs = [((0, 0), (10, 0)), ((10, 0), (10, 8)), ((10, 8), (0, 8)),
            ((0, 8), (0, 0)),                       # outer walls
            ((4, 0), (4, 5)), ((4, 5), (7, 5)),     # interior L wall
            ((2, 6.5), (3.5, 6.5))]                 # shelf
    P = [sample_segment(a, b, spacing, noise) for a, b in segs]
    P.append(sample_arc((8, 2), 0.8, 0, 2 * np.pi, spacing, noise))    # pillar
    P.append(sample_arc((1.5, 2.5), 0.4, 0, 2 * np.pi, spacing, noise))
    for (x, y, s) in [(6, 6.8, 0.3), (7.2, 1.0, 0.25)]:                # clutter boxes
        for a, b in [((x, y), (x + s, y)), ((x + s, y), (x + s, y + s)),
                     ((x + s, y + s), (x, y + s)), ((x, y + s), (x, y))]:
            P.append(sample_segment(a, b, spacing, noise))
    return np.vstack(P)

# ----------------------------------------------------------------------
# 2. Exact (RFF-noise-free) fields, used for split criterion + evaluation
# ----------------------------------------------------------------------
def exact_point_field(query, pts, ell, chunk=2000):
    """f(q) = sum_i exp(-|q - x_i|^2 / (2 ell^2))"""
    out = np.zeros(len(query))
    for i in range(0, len(query), chunk):
        d2 = ((query[i:i + chunk, None, :] - pts[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.exp(-d2 / (2 * ell ** 2)).sum(1)
    return out

def exact_token_field(query, tokens, ell):
    """Analytic readout of Gaussian tokens (mu, Sigma, n)."""
    d = query.shape[1]
    out = np.zeros(len(query))
    for mu, Sig, n in tokens:
        C = ell ** 2 * np.eye(d) + Sig
        Ci = np.linalg.inv(C)
        amp = n * ell ** d / np.sqrt(np.linalg.det(C))
        dx = query - mu
        out += amp * np.exp(-0.5 * np.einsum('ni,ij,nj->n', dx, Ci, dx))
    return out

# ----------------------------------------------------------------------
# 3. Adaptive tokenizer (quadtree, error-driven splits)
# ----------------------------------------------------------------------
def fit_token(pts):
    mu = pts.mean(0)
    d = pts - mu
    return mu, d.T @ d / len(pts), len(pts)

def leaf_error(pts, ell, max_probe=48):
    """Relative RMS mismatch between the cell's exact field and its
    single-token approximation, probed at (jittered) point locations."""
    tok = fit_token(pts)
    idx = rng.choice(len(pts), min(len(pts), max_probe), replace=False)
    probes = pts[idx] + rng.normal(0, ell, (len(idx), 2))
    fe = exact_point_field(probes, pts, ell)
    ft = exact_token_field(probes, [tok], ell)
    return np.sqrt(np.mean((ft - fe) ** 2)) / (np.sqrt(np.mean(fe ** 2)) + 1e-12)

def adaptive_tokens(pts, ell, tol=0.12, min_cell=0.05):
    tokens = []
    def recurse(P, x0, y0, size):
        if len(P) == 0:
            return
        if len(P) <= 2 or size <= min_cell or leaf_error(P, ell) <= tol:
            tokens.append(fit_token(P))
            return
        h = size / 2
        for dx in (0, 1):
            for dy in (0, 1):
                m = ((P[:, 0] >= x0 + dx * h) & (P[:, 0] < x0 + (dx + 1) * h) &
                     (P[:, 1] >= y0 + dy * h) & (P[:, 1] < y0 + (dy + 1) * h))
                recurse(P[m], x0 + dx * h, y0 + dy * h, h)
    mn = pts.min(0) - 0.1
    size = (pts.max(0) - pts.min(0)).max() + 0.3
    recurse(pts, mn[0], mn[1], size)
    return tokens

# ----------------------------------------------------------------------
# 4. Fixed-scale baseline: voxel downsample to a target token count
# ----------------------------------------------------------------------
def voxel_tokens(pts, K_target):
    def voxelize(s):
        keys, inv, cnt = np.unique(np.floor(pts / s).astype(np.int64),
                                   axis=0, return_inverse=True, return_counts=True)
        cx = np.bincount(inv, weights=pts[:, 0]) / cnt
        cy = np.bincount(inv, weights=pts[:, 1]) / cnt
        return [(np.array([cx[i], cy[i]]), np.zeros((2, 2)), cnt[i])
                for i in range(len(cnt))]
    lo, hi = 0.005, 3.0
    for _ in range(40):                     # binary search voxel size
        mid = (lo + hi) / 2
        K = len(voxelize(mid))
        if K > K_target:
            lo = mid
        else:
            hi = mid
    return voxelize(hi)

# ----------------------------------------------------------------------
# 5. FPE / RFF encoder and superposed memory
# ----------------------------------------------------------------------
class FPE:
    def __init__(self, D, ell, dims=2, seed=1):
        self.D = D
        self.W = np.random.default_rng(seed).normal(0, 1 / ell, (D, dims))

    def phasor(self, X):                     # (N, dims) -> (N, D)
        return np.exp(1j * (X @ self.W.T))

    def token_vec(self, mu, Sig, n):
        att = np.exp(-0.5 * np.einsum('di,ij,dj->d', self.W, Sig, self.W))
        return n * att * np.exp(1j * (self.W @ mu))

    def bundle_points(self, X, chunk=2000):
        M = np.zeros(self.D, complex)
        for i in range(0, len(X), chunk):
            M += self.phasor(X[i:i + chunk]).sum(0)
        return M

    def bundle_tokens(self, tokens):
        M = np.zeros(self.D, complex)
        for mu, Sig, n in tokens:
            M += self.token_vec(mu, Sig, n)
        return M

    def readout_multi(self, Ms, X, chunk=1000):
        """Query several memories at points X. Ms: (m, D) -> (N, m)."""
        Ms = np.atleast_2d(Ms)
        out = np.zeros((len(X), len(Ms)))
        for i in range(0, len(X), chunk):
            Z = self.phasor(X[i:i + chunk]).conj()
            out[i:i + chunk] = (Z @ Ms.T).real / self.D
        return out

# ----------------------------------------------------------------------
# 6. Demo
# ----------------------------------------------------------------------
def main():
    t0 = time.time()
    ELL, D, TOL = 0.10, 4096, 0.12

    pts = build_map()
    print(f"map points: {len(pts)}")

    tokens = adaptive_tokens(pts, ELL, tol=TOL)
    K = len(tokens)
    print(f"adaptive tokens: {K}  (compression {len(pts)/K:.1f}x)  "
          f"[{time.time()-t0:.1f}s]")

    vox = voxel_tokens(pts, K)
    print(f"voxel baseline tokens: {len(vox)}")

    enc = FPE(D, ELL)
    M_naive = enc.bundle_points(pts)
    M_vox   = enc.bundle_tokens(vox)
    M_adap  = enc.bundle_tokens(tokens)

    # evaluation grid
    gx, gy = np.meshgrid(np.linspace(-0.5, 10.5, 220), np.linspace(-0.5, 8.5, 180))
    grid = np.c_[gx.ravel(), gy.ravel()]

    f_true = exact_point_field(grid, pts, ELL)            # ground truth
    f_tok  = exact_token_field(grid, tokens, ELL)         # tokenization only
    R = enc.readout_multi(np.stack([M_naive, M_vox, M_adap]), grid)
    print(f"fields evaluated [{time.time()-t0:.1f}s]")

    def rel(a):
        return np.sqrt(np.mean((a - f_true) ** 2)) / np.sqrt(np.mean(f_true ** 2))

    print("\n--- relative field RMSE vs exact KDE of all points ---")
    print(f"adaptive tokens, exact readout (tokenization error only): {rel(f_tok):.3f}")
    print(f"naive RFF memory      K={len(pts):5d}: {rel(R[:,0]):.3f}")
    print(f"voxel RFF memory      K={len(vox):5d}: {rel(R[:,1]):.3f}")
    print(f"adaptive RFF memory   K={K:5d}: {rel(R[:,2]):.3f}")

    # ------------------------------------------------------------------
    # figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    vmax = np.percentile(f_true, 99.8)

    ax = axes[0, 0]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c='k', alpha=0.4)
    for mu, Sig, n in tokens:
        ev, evec = np.linalg.eigh(Sig)
        ang = np.degrees(np.arctan2(evec[1, -1], evec[0, -1]))
        e = Ellipse(mu, 2 * 2 * np.sqrt(max(ev[-1], 1e-6)),
                    2 * 2 * np.sqrt(max(ev[0], 1e-6)),
                    angle=ang, fill=False, color='crimson', lw=0.8, alpha=0.8)
        ax.add_patch(e)
    ax.set_title(f"adaptive tokens: {K} ellipses (2$\\sigma$) for {len(pts)} points")
    ax.set_aspect('equal')

    for ax, f, title in [(axes[0, 1], f_true, "exact field (all points, ground truth)"),
                         (axes[1, 0], R[:, 2], f"adaptive RFF memory readout (K={K}, D={D})"),
                         (axes[1, 1], R[:, 0], f"naive RFF memory readout (K={len(pts)}, D={D})")]:
        im = ax.imshow(f.reshape(gx.shape), origin='lower',
                       extent=[-0.5, 10.5, -0.5, 8.5], cmap='inferno',
                       vmin=0, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.03)

    plt.tight_layout()
    plt.savefig("adaptive_fpe_lidar.png", dpi=130)
    print(f"\nsaved adaptive_fpe_lidar.png [{time.time()-t0:.1f}s]")

if __name__ == "__main__":
    main()
