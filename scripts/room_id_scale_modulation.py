#!/usr/bin/env python
"""Per-room SSP scale modulation for ID-tagged object maps in the indoor env.

Each object in a room of scripts/indoor_env.py is encoded from K points
sampled uniformly INSIDE its footprint polygon (fixed K regardless of object
size, so large furniture is sparsely covered and small items densely -- the
same sparsity that made per-region scales optimal in the blob world). The
object encoding is the mean of the (per-room scale-modulated) point SSPs,
bound with a unique random unitary semantic pointer, and the room map is the
bundle:

    M_r = sum_j  ID_j (*) mean_p mod_{g_r}( SSP(p_jp) )

Per-room gains g_r (one non-negative weight per hexagonal scale block) are
learned by decoding THROUGH the map: for every object, the similarity map of
the unbound query M_r (*) ID_j^{-1} against the modulated grid encodings
should match that object's footprint indicator (cosine/shape loss + L1
sparsity + coarse-bias regularizers, the config validated in
scripts/quadrant_scale_modulation.py). Binding crosstalk between the room's
objects is therefore part of the training signal.

Training trick (exact, verified against the explicit path): modulation and
binding are both elementwise in the Fourier domain and every encoding has a
unit-magnitude spectrum, so each object's sim map is LINEAR in the per-scale
weights w = [1, 2*g_1^2, ..., 2*g_S^2]:

    sims_j(x) = sum_k w_k * D[j, x, k],
    D[j, x, s] = 2 * Re sum_{bins b in scale s} G_hat[x, b] * conj(U_hat[j, b])

with G_hat the grid encoding spectra and U_hat the unbound-query spectra of
the UNmodulated map. D is precomputed once per point draw (a few MB per
room), so the training loop needs no FFTs at all. Cosine normalization
constants are gain-dependent but object-global, hence cancel in the loss.

Outputs: an overlay figure of all unbound similarity maps (one color per
object, alpha proportional to similarity, so overlapping maps stay visible),
the same figure for the unmodulated map, learned gain profiles, loss curve,
and per-room cosine metrics (stdout + json + npz).

Example:
    python scripts/room_id_scale_modulation.py --out results/indoor_env/room_id_modulation.png
"""

import argparse
import json
import sys
import time
from pathlib import Path as FSPath

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Rectangle
from matplotlib.path import Path as MplPath

import jax
import jax.numpy as jnp
import optax
from vsagym.spaces import HexagonalSSPSpace

sys.path.insert(0, str(FSPath(__file__).parent))
from shape_scale_modulation import SOFTPLUS_INV_1, make_scale_index  # noqa: E402
from quadrant_id_map_decode import full_gain_spectrum, encode_mod  # noqa: E402
from indoor_env import DOOR_W, make_env  # noqa: E402


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def polygon_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def sample_in_polygon(poly, k, rng):
    """k points uniform inside a polygon (rejection sampling)."""
    path = MplPath(poly)
    lo, hi = poly.min(0), poly.max(0)
    out = []
    while len(out) < k:
        cand = rng.uniform(lo, hi, (max(4 * k, 64), 2))
        out.extend(cand[path.contains_points(cand)])
    return np.asarray(out[:k])


def room_grid(room, n):
    x0, x1, y0, y1 = room.interior
    X, Y = np.meshgrid(np.linspace(x0, x1, n), np.linspace(y0, y1, n))
    return X, Y, np.column_stack([X.ravel(), Y.ravel()])


# ---------------------------------------------------------------------------
# Fourier half-spectrum machinery (d odd: bins 0..(d-1)/2)
# ---------------------------------------------------------------------------


def half_spectrum_of_points(ssp_space, pts):
    """rfft(SSP(x)) computed directly: exp(i * A_half x / ls), (n, n_bins)."""
    n_bins = (ssp_space.ssp_dim + 1) // 2
    scaled = (np.atleast_2d(pts) / ssp_space.length_scale.flatten()).T
    return np.exp(1j * ssp_space.phase_matrix[:n_bins] @ scaled).T


def scale_collapse_matrix(ssp_space, scale_idx):
    """(n_bins, n_scales + 1) map from per-bin products to per-scale sums;
    column 0 = DC (weight 1), non-DC bins contribute x2 (conjugate pair)."""
    n_bins = (ssp_space.ssp_dim + 1) // 2
    W = np.zeros((n_bins, ssp_space.n_scales + 1), np.float64)
    W[0, 0] = 1.0
    W[np.arange(1, n_bins), 1 + np.asarray(scale_idx)] = 2.0
    return W


def compute_D(G_hat, V_hat, ids_hat, W):
    """Per-scale collapsed sim coefficients for one point draw and ID set:
    D[j, x, s] such that sims_j(x) = D[j, x] @ [1, g_1^2, ..., g_S^2]."""
    M_hat = (ids_hat * V_hat).sum(axis=0)
    U_hat = M_hat[None] * ids_hat.conj()
    D = np.empty((len(V_hat), len(G_hat), W.shape[1]), np.float32)
    for s in range(W.shape[1]):
        bins = np.nonzero(W[:, s])[0]
        D[:, :, s] = W[bins[0], s] * np.real(U_hat[:, bins].conj() @ G_hat[:, bins].T)
    return D


class RoomData:
    """Per-room constants + per-draw collapsed sim coefficients D."""

    def __init__(self, room, items, ssp_space, W, ids_hat, grid_n, k_fn, draws_rng, n_draws):
        self.room, self.items = room, items
        self.k_per_item = [k_fn(it) for it in items]
        self.X, self.Y, grid_pts = room_grid(room, grid_n)
        self.grid_pts = grid_pts
        self.G_hat = half_spectrum_of_points(ssp_space, grid_pts)     # (n_grid, n_bins)
        self.ids_hat = ids_hat                                        # (n_obj, n_bins)

        # unit-norm footprint indicator per object (guard tiny/thin items)
        T = []
        for it in items:
            t = MplPath(it.polygon).contains_points(grid_pts).astype(np.float32)
            if t.sum() == 0:
                t[np.argmin(np.linalg.norm(grid_pts - it.center, axis=1))] = 1.0
            T.append(t / np.linalg.norm(t))
        self.targets = np.stack(T)                                    # (n_obj, n_grid)

        # V spectra + D for each point draw: 0..n_draws-1 train, n_draws = eval
        self.point_draws, self.V_hats, self.D = [], [], []
        for _ in range(n_draws + 1):
            pts = [sample_in_polygon(it.polygon, k, draws_rng)
                   for it, k in zip(items, self.k_per_item)]
            self.point_draws.append(pts)
            V_hat = np.stack([half_spectrum_of_points(ssp_space, p).mean(axis=0)
                              for p in pts])                          # (n_obj, n_bins)
            self.V_hats.append(V_hat)
            self.D.append(compute_D(self.G_hat, V_hat, ids_hat, W))
        self.D_train = jnp.asarray(np.stack(self.D[:-1]))             # (n_draws, n_obj, n_grid, S+1)
        self.D_eval = self.D[-1]


# ---------------------------------------------------------------------------
# Training: per-room gains through the unbind-decode loss
# ---------------------------------------------------------------------------


def gains_from_raw(raw):
    return jax.nn.softplus(raw + SOFTPLUS_INV_1)


def cos_maps(D, w, targets):
    """Per-object cosine between sim maps D @ w and unit-norm targets."""
    sims = jnp.einsum("ogk,k->og", D, w)
    sims = sims / (jnp.linalg.norm(sims, axis=1, keepdims=True) + 1e-8)
    return jnp.sum(sims * targets, axis=1)


def train(rooms_data, ssp_space, args):
    n_rooms, n_sc = len(rooms_data), ssp_space.n_scales
    params = jnp.zeros((n_rooms, n_sc))
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    w_coarse = jnp.asarray(np.asarray(ssp_space.scales) / np.max(ssp_space.scales))
    D_all = [rd.D_train for rd in rooms_data]
    T_all = [jnp.asarray(rd.targets) for rd in rooms_data]

    def loss_fn(params, draw):
        loss = 0.0
        for r in range(n_rooms):
            g = gains_from_raw(params[r])
            w = jnp.concatenate([jnp.ones(1), g ** 2])
            fit = 1.0 - cos_maps(D_all[r][draw], w, T_all[r]).mean()
            reg = (args.reg_w * jnp.sum(jnp.abs(g))
                   + args.coarse_w * jnp.sum(w_coarse * g ** 2))
            loss += fit + reg
        return loss / n_rooms

    @jax.jit
    def step(params, opt_state, draw):
        loss, grads = jax.value_and_grad(loss_fn)(params, draw)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = np.empty(args.steps)
    for i in range(args.steps):
        params, opt_state, loss = step(params, opt_state, i % args.n_draws)
        losses[i] = float(loss)
        if i % 200 == 0 or i == args.steps - 1:
            print(f"  step {i:>4}: loss {losses[i]:.4f}")
    return np.asarray(gains_from_raw(params)), losses


def train_ids(rooms_data, ssp_space, scale_idx, args):
    """Jointly train per-room gains AND per-object unitary IDs. IDs are
    parameterized by their Fourier phases, ID_hat = exp(i*phi) with the DC
    term fixed at 1, so every spectral coefficient has magnitude exactly 1
    (the unitary constraint holds by construction and unbinding stays exact).
    The collapsed-D shortcut does not apply (D assumes fixed IDs), so sim
    maps are computed from the precomputed grid/object spectra directly."""
    n_rooms, n_sc = len(rooms_data), ssp_space.n_scales
    scale_idx_j = jnp.asarray(scale_idx)
    w_coarse = jnp.asarray(np.asarray(ssp_space.scales) / np.max(ssp_space.scales))
    G = [jnp.asarray(rd.G_hat.astype(np.complex64)) for rd in rooms_data]
    V = [jnp.asarray(np.stack(rd.V_hats[:-1]).astype(np.complex64)) for rd in rooms_data]
    T = [jnp.asarray(rd.targets) for rd in rooms_data]

    params = {"g": jnp.zeros((n_rooms, n_sc))}
    for r, rd in enumerate(rooms_data):   # init at the random IDs' phases
        params[f"ph{r}"] = jnp.asarray(np.angle(rd.ids_hat[:, 1:]), dtype=jnp.float32)

    def room_fit(g, ph, Gr, Vr, Tr):
        w_bin = jnp.concatenate([jnp.ones(1), 2.0 * (g ** 2)[scale_idx_j]])
        idh = jnp.exp(1j * jnp.concatenate(
            [jnp.zeros((ph.shape[0], 1)), ph], axis=1))               # unit-magnitude
        M_hat = (idh * Vr).sum(axis=0)
        U_hat = M_hat[None] * idh.conj()
        sims = jnp.real((U_hat.conj() * w_bin) @ Gr.T)
        sims = sims / (jnp.linalg.norm(sims, axis=1, keepdims=True) + 1e-8)
        return 1.0 - jnp.sum(sims * Tr, axis=1).mean()

    def loss_fn(params, draw):
        loss = 0.0
        for r in range(n_rooms):
            g = gains_from_raw(params["g"][r])
            fit = room_fit(g, params[f"ph{r}"], G[r], V[r][draw], T[r])
            reg = (args.reg_w * jnp.sum(jnp.abs(g))
                   + args.coarse_w * jnp.sum(w_coarse * g ** 2))
            loss += fit + reg
        return loss / n_rooms

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, draw):
        loss, grads = jax.value_and_grad(loss_fn)(params, draw)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss

    losses = np.empty(args.id_steps)
    t0 = time.time()
    for i in range(args.id_steps):
        params, opt_state, loss = step(params, opt_state, i % args.n_draws)
        losses[i] = float(loss)
        if i % 100 == 0 or i == args.id_steps - 1:
            print(f"  step {i:>4}: loss {losses[i]:.4f} ({time.time() - t0:.0f}s)")

    gains = np.asarray(gains_from_raw(params["g"]))
    ids_by_room = {}
    for r, rd in enumerate(rooms_data):
        ph = np.asarray(params[f"ph{r}"])
        idh = np.exp(1j * np.concatenate([np.zeros((len(ph), 1)), ph], axis=1))
        ids_by_room[rd.room.name] = np.fft.irfft(idh, n=ssp_space.ssp_dim, axis=-1)
    return gains, ids_by_room, losses


# ---------------------------------------------------------------------------
# Explicit-path evaluation (verifies the collapsed-D factorization)
# ---------------------------------------------------------------------------


def explicit_sim_maps(ssp_space, g, scale_idx, rd, ids):
    """Sim maps via explicit modulated encodings on the eval draw."""
    g_full = full_gain_spectrum(g, scale_idx)
    V = np.stack([encode_mod(ssp_space, g_full, p).mean(axis=0)
                  for p in rd.point_draws[-1]])
    M = ssp_space.bind(ids, V).sum(axis=0)
    U = ssp_space.bind(M[None], ssp_space.invert(ids))
    mod_grid = encode_mod(ssp_space, g_full, rd.grid_pts)
    return U @ mod_grid.T                                             # (n_obj, n_grid)


def eval_room(ssp_space, g, scale_idx, W, rd, ids):
    sims = explicit_sim_maps(ssp_space, g, scale_idx, rd, ids)
    simn = sims / (np.linalg.norm(sims, axis=1, keepdims=True) + 1e-8)
    cos_explicit = (simn * rd.targets).sum(axis=1)
    ids_hat = np.fft.rfft(ids, axis=-1)[:, :W.shape[0]]
    D_eval = compute_D(rd.G_hat, rd.V_hats[-1], ids_hat, W)
    w = np.concatenate([[1.0], np.asarray(g) ** 2])
    cos_D = np.asarray(cos_maps(jnp.asarray(D_eval), jnp.asarray(w),
                                jnp.asarray(rd.targets)))
    assert np.abs(cos_explicit - cos_D).max() < 1e-3, "factorization mismatch"
    return sims, cos_explicit


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def overlay_figure(env, rooms_data, sims_by_room, cos_by_room, title, path,
                   grid_n, alpha_thr=0.2):
    fig, ax = plt.subplots(figsize=(13.2, 9.2), facecolor="white")
    cmap20 = plt.get_cmap("tab20")
    for rd, sims in zip(rooms_data, sims_by_room):
        x0, x1, y0, y1 = rd.room.interior
        for j, it in enumerate(rd.items):
            color = np.array(cmap20(j % 20)[:3])
            s = np.clip(sims[j], 0, None)
            q = np.quantile(s, 0.995)
            simn = s / q if q > 1e-9 else s
            a = np.clip((np.clip(simn, 0, 1) - alpha_thr) / (1 - alpha_thr), 0, 1) ** 1.2
            rgba = np.empty((grid_n, grid_n, 4))
            rgba[..., :3] = color
            rgba[..., 3] = (0.9 * a).reshape(grid_n, grid_n)
            ax.imshow(rgba, origin="lower", extent=[x0, x1, y0, y1],
                      interpolation="bilinear", zorder=2)
            ax.add_patch(MplPolygon(it.polygon, closed=True, facecolor="none",
                                    edgecolor="0.15", lw=0.6, zorder=3))
        cos = cos_by_room[rd.room.name]
        ax.text(x0 + 0.1, y1 - 0.12,
                f"{rd.room.theme.upper()}  cos={np.mean(cos):.3f}",
                fontsize=10, color="0.25", ha="left", va="top",
                fontweight="bold", zorder=5)
    for x, y, w, h in env.wall_rects:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="0.15", edgecolor="none", zorder=4))
    for _, _, cx, cy, hor in env.doors:
        dx, dy = (DOOR_W / 2, 0) if hor else (0, DOOR_W / 2)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color="0.75", lw=1.0,
                ls=(0, (4, 3)), zorder=4)
    ax.set_xlim(-0.25, 15.25)
    ax.set_ylim(-0.25, 10.25)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def gains_figure(rooms_data, gains, scales, path):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True, facecolor="white")
    # arrange panels to match the world layout (row 1 on top)
    by_rc = {(rd.room.row, rd.room.col): (rd, g) for rd, g in zip(rooms_data, gains)}
    for (r, c), (rd, g) in by_rc.items():
        ax = axs[1 - r, c]
        ax.bar(np.arange(len(scales)), g, color="tab:blue")
        ax.set_title(f"{rd.room.theme} ({len(rd.items)} items)", fontsize=10)
        ax.set_xlabel("scale (coarse -> fine)")
        if c == 0:
            ax.set_ylabel("gain")
    fig.suptitle("learned per-room scale gains")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scales", type=int, default=10)
    ap.add_argument("--n-rotates", type=int, default=33)
    ap.add_argument("--scale-min", type=float, default=1.0)
    ap.add_argument("--scale-max", type=float, default=10.0)
    ap.add_argument("--scale-sampling", type=str, default="log", choices=["lin", "log", "rand"])
    ap.add_argument("--ls", type=float, default=0.9)
    ap.add_argument("--k-mode", choices=["fixed", "area"], default="area",
                    help="'fixed': --k-pts samples per object regardless of size; "
                         "'area': constant density --pts-per-m2 (clamped)")
    ap.add_argument("--k-pts", type=int, default=25, help="samples per object (fixed mode)")
    ap.add_argument("--pts-per-m2", type=float, default=40.0,
                    help="sample density (area mode)")
    ap.add_argument("--k-min", type=int, default=8)
    ap.add_argument("--k-max", type=int, default=120)
    ap.add_argument("--n-draws", type=int, default=8, help="training point draws (one more for eval)")
    ap.add_argument("--grid-n", type=int, default=50, help="per-room grid resolution")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--train-ids", action="store_true",
                    help="jointly train unitary object IDs (Fourier phases) with the gains")
    ap.add_argument("--id-steps", type=int, default=1000, help="joint training steps")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--reg-w", type=float, default=0.003)
    ap.add_argument("--coarse-w", type=float, default=0.01)
    ap.add_argument("--env-seed", type=int, default=7)
    ap.add_argument("--seed", type=int, default=3, help="IDs + point sampling")
    ap.add_argument("--out", type=str, default="results/indoor_env/room_id_modulation.png")
    args = ap.parse_args()

    out_path = FSPath(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_seed)
    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=args.n_scales, n_rotates=args.n_rotates,
                                  scale_min=args.scale_min, scale_max=args.scale_max,
                                  scale_sampling=args.scale_sampling,
                                  length_scale=args.ls, rng=0)
    d = ssp_space.ssp_dim
    scale_idx = make_scale_index(ssp_space)
    W = scale_collapse_matrix(ssp_space, scale_idx)

    if args.k_mode == "area":
        def k_fn(it):
            return int(np.clip(round(args.pts_per_m2 * polygon_area(it.polygon)),
                               args.k_min, args.k_max))
    else:
        def k_fn(it):
            return args.k_pts
    print(f"ssp_dim={d}, {len(env.items)} objects in {len(env.rooms)} rooms, "
          f"k_mode={args.k_mode}, {args.n_draws}+1 point draws")

    rng = np.random.default_rng(args.seed)
    rooms_data, ids_by_room = [], {}
    t0 = time.time()
    for room in env.rooms:
        items = [it for it in env.items if it.room == room.name]
        ids = ssp_space.make_unitary(rng.normal(size=(len(items), d)))
        ids_by_room[room.name] = ids
        n_bins = (d + 1) // 2
        ids_hat = np.fft.rfft(ids, axis=-1)[:, :n_bins]
        rd = RoomData(room, items, ssp_space, W, ids_hat,
                      args.grid_n, k_fn, rng, args.n_draws)
        rooms_data.append(rd)
        print(f"  {room.theme:<10} K per object: {min(rd.k_per_item)}-{max(rd.k_per_item)} "
              f"(total {sum(rd.k_per_item)})")
    print(f"precomputed D coefficients in {time.time() - t0:.0f}s")

    print("training per-room gains through the unbind-decode loss ...")
    t0 = time.time()
    gains, losses = train(rooms_data, ssp_space, args)
    print(f"trained in {time.time() - t0:.0f}s")

    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor="white")
    ax.plot(losses, color="tab:blue")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path.with_name(out_path.stem + "_losses.png"), dpi=150)
    plt.close(fig)

    gains_j, ids_trained = None, None
    if args.train_ids:
        print("jointly training gains + unitary IDs (Fourier phases) ...")
        gains_j, ids_trained, losses_id = train_ids(rooms_data, ssp_space, scale_idx, args)
        fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor="white")
        ax.plot(losses_id, color="tab:red")
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(out_path.with_name(out_path.stem + "_id_losses.png"), dpi=150)
        plt.close(fig)

    # held-out evaluation, explicit path (also verifies the factorization)
    g_ones = np.ones(ssp_space.n_scales)
    header = f"\n{'room':<10} {'n_obj':>5} {'cos none':>9} {'cos gains':>10}"
    if args.train_ids:
        header += f" {'cos +IDs':>9} {'min +IDs':>9}"
    print(header)
    sims_n, sims_l, sims_t = [], [], []
    cos_n, cos_l, cos_t = {}, {}, {}
    for ri, (rd, g) in enumerate(zip(rooms_data, gains)):
        ids = ids_by_room[rd.room.name]
        s_n, c_n = eval_room(ssp_space, g_ones, scale_idx, W, rd, ids)
        s_l, c_l = eval_room(ssp_space, g, scale_idx, W, rd, ids)
        sims_n.append(s_n); sims_l.append(s_l)
        cos_n[rd.room.name], cos_l[rd.room.name] = c_n, c_l
        line = (f"{rd.room.theme:<10} {len(rd.items):>5} {np.mean(c_n):>9.3f} "
                f"{np.mean(c_l):>10.3f}")
        if args.train_ids:
            s_t, c_t = eval_room(ssp_space, gains_j[ri], scale_idx, W, rd,
                                 ids_trained[rd.room.name])
            sims_t.append(s_t)
            cos_t[rd.room.name] = c_t
            line += f" {np.mean(c_t):>9.3f} {np.min(c_t):>9.3f}"
        print(line)
    all_n = np.concatenate(list(cos_n.values()))
    all_l = np.concatenate(list(cos_l.values()))
    line = f"{'ALL':<10} {len(all_l):>5} {all_n.mean():>9.3f} {all_l.mean():>10.3f}"
    if args.train_ids:
        all_t = np.concatenate(list(cos_t.values()))
        line += f" {all_t.mean():>9.3f} {all_t.min():>9.3f}"
    print(line)

    if args.train_ids:
        overlay_figure(env, rooms_data, sims_t, cos_t,
                       "unbound object similarity maps, trained unitary IDs + "
                       "learned gains (one color per object, alpha ~ similarity)",
                       out_path, args.grid_n)
        overlay_figure(env, rooms_data, sims_l, cos_l,
                       "unbound object similarity maps, random IDs + learned gains",
                       out_path.with_name(out_path.stem + "_randomIDs.png"), args.grid_n)
    else:
        overlay_figure(env, rooms_data, sims_l, cos_l,
                       "unbound object similarity maps, learned per-room scale gains "
                       "(one color per object, alpha ~ similarity)",
                       out_path, args.grid_n)
    overlay_figure(env, rooms_data, sims_n, cos_n,
                   "unbound object similarity maps, unmodulated (gains = 1)",
                   out_path.with_name(out_path.stem + "_unmodulated.png"), args.grid_n)
    gains_figure(rooms_data, gains_j if args.train_ids else gains,
                 np.asarray(ssp_space.scales),
                 out_path.with_name(out_path.stem + "_gains.png"))

    save = dict(gains=gains, scales=np.asarray(ssp_space.scales),
                rooms=[rd.room.theme for rd in rooms_data])
    if args.train_ids:
        save["gains_joint"] = gains_j
        for rd in rooms_data:
            save[f"ids_{rd.room.theme}"] = ids_trained[rd.room.name]
    np.savez(out_path.with_name(out_path.stem + "_gains.npz"), **save)
    metrics = {rd.room.theme: {"n_obj": len(rd.items),
                               "k_per_item": rd.k_per_item,
                               "cos_none": cos_n[rd.room.name].tolist(),
                               "cos_learned": cos_l[rd.room.name].tolist(),
                               **({"cos_trained_ids": cos_t[rd.room.name].tolist()}
                                  if args.train_ids else {})}
               for rd in rooms_data}
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)
    print(f"wrote {out_path.with_suffix('.json')} and _gains.npz")


if __name__ == "__main__":
    main()
