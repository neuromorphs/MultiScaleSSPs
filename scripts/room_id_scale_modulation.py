#!/usr/bin/env python
"""Per-room SSP scale modulation for ID-tagged object maps in the indoor env.

Each object in a room of scripts/indoor_env.py is encoded from its footprint
polygon, in one of two ways (--encode-mode):

    integral (default) -- the exact area integral of the SSP over the
        footprint, computed on a dense fixed grid (--integral-res). The
        encoding is deterministic and its spectrum is the footprint's Fourier
        transform sampled at the phase-matrix frequencies, so LARGE objects
        genuinely carry no fine-scale signal -- fine gains only add crosstalk
        for them. This is what lets rooms with different object sizes learn
        genuinely different gain profiles.
    points -- points sampled uniformly inside the footprint at constant
        density (area-proportional K). The constant inter-sample spacing sets
        a preferred 'gap-filling' scale that is the SAME for every room, which
        homogenizes the learned profiles (kept for comparison).

The object encoding is the mean of the (per-room scale-modulated) point SSPs,
bound with a fixed semantic pointer from an orthogonal unitary codebook
(vsagym SPSpace), and the room map is the bundle:

    M_r = sum_j  ID_j (*) mean_p mod_{g_r}( SSP(p_jp) )

Per-room gains g_r (one non-negative weight per hexagonal scale block) are
learned by decoding THROUGH the map: for every object, the similarity map of
the unbound query M_r (*) ID_j^{-1} against the modulated grid encodings
should match that object's footprint indicator. The fit loss is regularized
so that sparse and coarse (small-scale) gain profiles are favored when they
decode equally well: an L1 penalty on the gains (strong enough to zero out
scales a room does not need, so each room 'picks out' its scales) plus a
scale-weighted L2 penalty that charges fine scales more. Binding crosstalk
between the room's objects is part of the training signal.

As a baseline, a single SHARED gain profile is trained the same way for all
rooms at once, to show what per-room adaptation buys.

On top of the fixed codebook, object IDs can also be LEARNED jointly with the
gains (--train-ids, on by default): IDs are parameterized by their Fourier
phases, ID_hat = exp(i*phi) with DC fixed at 1, so every spectral coefficient
has magnitude exactly 1 (unitary by construction, unbinding stays exact).
Phases are initialized at the orthogonal codebook. Only the crosstalk term
depends on the IDs, so this is codebook optimization: interference is shaped
to cancel on the footprint targets.

Finally, object POSITIONS are decoded from the maps with the adaptive
direct-optim method of scripts/quadrant_id_map_decode.py: the effective
length scale l_eff = FWHM of the modulated similarity kernel sizes the
initial sampling grid (library rule), argmax over that grid seeds an
L-BFGS-B refinement of the unbound query similarity. Coarse rooms get coarse
(cheap) grids, cluttered rooms fine ones.

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
the same figure for the shared-gain map and for the trained-ID map, learned
per-room gain profiles (with the shared profile for reference), a decoded-
positions figure (colored 'x' per object with each room's initial sampling
grid drawn), and per-room cosine/decode metrics (stdout + json + npz).

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
from vsagym.spaces import HexagonalSSPSpace, SPSpace

sys.path.insert(0, str(FSPath(__file__).parent))
from shape_scale_modulation import SOFTPLUS_INV_1, make_scale_index  # noqa: E402
from quadrant_id_map_decode import (full_gain_spectrum, encode_mod,  # noqa: E402
                                    pts_per_dim, direct_optim_decode)
from indoor_env import DOOR_W, make_env  # noqa: E402
from vsa_bin_pruning import (ranking_curve, random_mask,  # noqa: E402
                             retained_energy, progressive_gate_masks)


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


def polygon_grid_points(poly, res):
    """Dense fixed grid of points inside a polygon (numerical area integral).
    Refines the resolution for footprints too thin to catch >= 4 cells."""
    path = MplPath(poly)
    lo, hi = poly.min(0), poly.max(0)
    for r in (res, res / 2, res / 4):
        xs = np.arange(lo[0] + r / 2, hi[0], r)
        ys = np.arange(lo[1] + r / 2, hi[1], r)
        X, Y = np.meshgrid(xs, ys)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        pts = pts[path.contains_points(pts)]
        if len(pts) >= 4:
            return pts
    return np.mean(poly, axis=0, keepdims=True)   # degenerate: centroid


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


# ---------------------------------------------------------------------------
# Post-hoc bin-level pruning machinery (see scripts/vsa_bin_pruning.py for the
# generic mask/gate arithmetic; everything here is the SSP-specific glue that
# feeds it a per-bin score/loss). All of this generalizes scale_collapse_matrix
# / compute_D above from scale-block resolution to individual-bin resolution:
# a diagonal W (one nonzero bin per column) makes compute_D's per-column loop
# degenerate to the exact per-bin coefficient, so masking bin b of the final
# room map is provably identical to masking bin b of every object's V_hat
# before bundling -- "whole-map" vs "per-object" pruning differ only in which
# signal chooses the mask, never in how a chosen mask is applied.
# ---------------------------------------------------------------------------


def gain_half_spectrum(g, scale_idx, n_bins):
    """(n_bins,) real gain applied to each half-spectrum bin (DC fixed at 1),
    a single power -- the per-side factor encode_mod/RoomData implicitly
    apply once, before compute_D's bilinear sim squares it."""
    gh = np.empty(n_bins)
    gh[0] = 1.0
    gh[1:] = np.asarray(g)[scale_idx]
    return gh


def bin_priority(ssp_space, scale_idx):
    """Fixed, memory-blind per-bin priority score: 0 at DC, else the bin's own
    scale value (coarse = small). The FHRR-frequency-norm "lowpass" priority
    of vsa_pruning.md, specialized to this codebase -- scale_idx already IS
    each non-DC bin's frequency-magnitude bucket (make_scale_index asserts
    phase_matrix row norms equal ssp_space.scales[scale_idx])."""
    return np.concatenate([[0.0], np.asarray(ssp_space.scales)[scale_idx]])


def bins_kept_by_scale(masks, scale_idx, n_scales):
    """(n_dprimes, n_scales) fraction of each scale's non-DC bins kept by
    every row of `masks` (n_dprimes, n_bins) -- lets you see WHICH scales a
    strategy prunes first vs. keeps longest, independent of which criterion
    chose the mask (e.g. `priority`'s score IS the scale value, so this
    breakdown is exactly "how many bins were pruned at each priority
    level"; for magnitude/gate it shows whether the map's own learned
    energy/loss concentrates in a similar coarse-first pattern)."""
    scale_idx = np.asarray(scale_idx)
    kept_non_dc = masks[:, 1:]                      # drop the always-on DC bin
    counts = np.array([(scale_idx == s).sum() for s in range(n_scales)])
    kept = np.stack([kept_non_dc[:, scale_idx == s].sum(axis=1) for s in range(n_scales)], axis=1)
    return kept / counts[None, :]


def bin_diag_weight(g, scale_idx, n_bins):
    """Diagonal (n_bins,) weight for compute_D_bin: DC fixed at 1 (never
    gained/learned anywhere in this script); each non-DC bin gets its scale's
    learned gain squared times the conjugate-pair factor of 2 -- the bilinear
    (memory-side * query-side) coefficient the trained map actually deploys,
    not the raw {1,2} convention scale_collapse_matrix uses for an ungained
    ranking."""
    gh = gain_half_spectrum(g, scale_idx, n_bins)
    w = 2.0 * gh ** 2
    w[0] = 1.0
    return w


def compute_D_bin(G_hat, V_hat, ids_hat, w_bin):
    """Whole-map (crosstalk-inclusive) per-bin sim coefficients: D_bin[j,x,b]
    such that sims_j(x) = D_bin[j,x,:] @ mu is linear in a per-bin mask mu.
    Vectorized generalization of compute_D for a diagonal W (every column has
    exactly one nonzero bin) -- (n_obj, n_grid, n_bins)."""
    M_hat = (ids_hat * V_hat).sum(axis=0)
    U_hat = M_hat[None] * ids_hat.conj()
    return w_bin[None, None, :] * np.real(U_hat.conj()[:, None, :] * G_hat[None, :, :])


def compute_D_bin_selfonly(G_hat, V_hat, w_bin):
    """Per-object (crosstalk-free) per-bin sim coefficients, as if each object
    were alone in the room. Needs no ids_hat: self-unbind of a unit-magnitude
    ID cancels exactly (|ids_hat|^2 == 1), so U_hat_alone[j] == V_hat[j]."""
    return w_bin[None, None, :] * np.real(V_hat.conj()[:, None, :] * G_hat[None, :, :])


def magnitude_score_whole(ids_hat, V_hat, g_half):
    """Per-bin |M_hat| magnitude of the trained, gained, bundled room memory
    -- a single power of gain (ranks the memory's own per-bin coefficient,
    not the bilinear sim coefficient compute_D_bin later scores masks with)."""
    M_hat = (ids_hat * V_hat).sum(axis=0)
    return np.abs(g_half * M_hat)


def magnitude_score_object(V_hat, g_half):
    """Total per-bin energy any object in the room demands, pre-bundle:
    sum_j |g_half * V_hat[j]|^2."""
    return np.sum(np.abs(g_half[None, :] * V_hat) ** 2, axis=0)


class RoomData:
    """Per-room constants + per-draw collapsed sim coefficients D.

    integral_res set: one deterministic 'draw' of dense in-footprint grid
    points (numerical area integral). Otherwise n_draws random point draws
    for training plus one more for held-out evaluation."""

    def __init__(self, room, items, ssp_space, W, ids_hat, grid_n, k_fn, draws_rng,
                 n_draws, integral_res=None):
        self.room, self.items = room, items
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

        # V spectra + D per draw (integral: single deterministic draw)
        if integral_res is not None:
            draws = [[polygon_grid_points(it.polygon, integral_res) for it in items]]
        else:
            draws = [[sample_in_polygon(it.polygon, k_fn(it), draws_rng) for it in items]
                     for _ in range(n_draws + 1)]
        self.k_per_item = [len(p) for p in draws[0]]
        self.point_draws, self.V_hats, self.D = [], [], []
        for pts in draws:
            self.point_draws.append(pts)
            V_hat = np.stack([half_spectrum_of_points(ssp_space, p).mean(axis=0)
                              for p in pts])                          # (n_obj, n_bins)
            self.V_hats.append(V_hat)
            self.D.append(compute_D(self.G_hat, V_hat, ids_hat, W))
        # train draws: all but the last (points) / the single draw (integral)
        self.D_train = jnp.asarray(np.stack(self.D[:-1] or self.D))
        self.D_eval = self.D[-1]


# ---------------------------------------------------------------------------
# Training: per-room gains through the unbind-decode loss
# ---------------------------------------------------------------------------


def gains_from_raw(raw):
    return jax.nn.softplus(raw + SOFTPLUS_INV_1)


def gain_param(args, n_sc):
    """(gain_fn, init_row) for the chosen gain parameterization.

    'free': one non-negative gain per scale (softplus, init 1).
    'gaussian': the star_ssp_mean notebook's kernel -- gains are a discretized
    1-D Gaussian over the scale-block index, only center mu and width sigma
    are learned (peak amplitude fixed at 1); init centered and wide
    (near-flat), sigma = softplus(raw) + 1e-3.
    'window': a sliding boxcar of --window-width consecutive 1s (0 outside);
    only its center is learned. Trained as a soft window (product of two
    sigmoids, temperature 0.5, so the center gets gradients), snapped to the
    hard binary window for evaluation and decoding.
    """
    if args.gain_param == "free":
        return gains_from_raw, jnp.zeros(n_sc)
    k = jnp.arange(n_sc)

    if args.gain_param == "window":
        half = args.window_width / 2.0

        def window(p):
            return (jax.nn.sigmoid((k - (p[0] - half)) / 0.5)
                    * jax.nn.sigmoid(((p[0] + half) - k) / 0.5))

        return window, jnp.array([(n_sc - 1) / 2.0])

    def gauss(p):
        sigma = jax.nn.softplus(p[1]) + 1e-3
        return jnp.exp(-0.5 * ((k - p[0]) / sigma) ** 2)

    return gauss, jnp.array([(n_sc - 1) / 2.0, 5.0])


def snap_window(raw, n_sc, width):
    """Hard binary windows (width consecutive 1s) nearest each soft center."""
    starts = np.clip(np.round(np.asarray(raw)[:, 0] - (width - 1) / 2.0),
                     0, n_sc - width).astype(int)
    g = np.zeros((len(starts), n_sc))
    for i, s in enumerate(starts):
        g[i, s:s + width] = 1.0
    return g


def softplus_np(x):
    return np.logaddexp(0.0, x)


def cos_maps(D, w, targets):
    """Per-object cosine between sim maps D @ w and unit-norm targets."""
    sims = jnp.einsum("ogk,k->og", D, w)
    sims = sims / (jnp.linalg.norm(sims, axis=1, keepdims=True) + 1e-8)
    return jnp.sum(sims * targets, axis=1)


def train(rooms_data, ssp_space, args, shared=False):
    """Learn non-negative per-scale gains through the unbind-decode loss.

    shared=True fits ONE gain profile for all rooms jointly (the baseline);
    otherwise each room gets its own row. The fit term (1 - cosine to the
    footprint targets) is regularized by an L1 penalty (sparsity: unused
    scales are pushed to zero) and a scale-weighted L2 penalty (coarse bias:
    fine scales cost proportionally more), both mild relative to the fit.
    """
    n_rooms, n_sc = len(rooms_data), ssp_space.n_scales
    gain_fn, init = gain_param(args, n_sc)
    params = jnp.tile(init[None], (1 if shared else n_rooms, 1))
    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)
    w_coarse = jnp.asarray(np.asarray(ssp_space.scales) / np.max(ssp_space.scales))
    D_all = [rd.D_train for rd in rooms_data]
    T_all = [jnp.asarray(rd.targets) for rd in rooms_data]

    def loss_fn(params, draw):
        loss = 0.0
        for r in range(n_rooms):
            g = gain_fn(params[0 if shared else r])
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
    gains = np.stack([np.asarray(gain_fn(p)) for p in params])
    return gains, losses, np.asarray(params)


def train_ids(rooms_data, ssp_space, scale_idx, args, shared=False):
    """Jointly train gains AND per-object unitary IDs. IDs are parameterized
    by their Fourier phases, ID_hat = exp(i*phi) with the DC term fixed at 1,
    so every spectral coefficient has magnitude exactly 1 (the unitary
    constraint holds by construction and unbinding stays exact). Phases are
    initialized at the orthogonal codebook. shared=True fits ONE gain profile
    for all rooms (IDs stay per-object) to isolate what per-room gains add on
    top of ID learning. The collapsed-D shortcut does not apply (D assumes
    fixed IDs), so sim maps are computed from the precomputed grid/object
    spectra directly."""
    n_rooms, n_sc = len(rooms_data), ssp_space.n_scales
    scale_idx_j = jnp.asarray(scale_idx)
    w_coarse = jnp.asarray(np.asarray(ssp_space.scales) / np.max(ssp_space.scales))
    G = [jnp.asarray(rd.G_hat.astype(np.complex64)) for rd in rooms_data]
    V = [jnp.asarray(np.stack(rd.V_hats[:-1] or rd.V_hats).astype(np.complex64))
         for rd in rooms_data]
    T = [jnp.asarray(rd.targets) for rd in rooms_data]

    gain_fn, init = gain_param(args, n_sc)
    params = {"g": jnp.tile(init[None], (1 if shared else n_rooms, 1))}
    for r, rd in enumerate(rooms_data):   # init at the codebook IDs' phases
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
            g = gain_fn(params["g"][0 if shared else r])
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
    for i in range(args.id_steps):
        params, opt_state, loss = step(params, opt_state, i % args.n_draws)
        losses[i] = float(loss)
        if i % 200 == 0 or i == args.id_steps - 1:
            print(f"  step {i:>4}: loss {losses[i]:.4f}")

    gains = np.stack([np.asarray(gain_fn(p)) for p in params["g"]])
    ids_by_room = {}
    for r, rd in enumerate(rooms_data):
        ph = np.asarray(params[f"ph{r}"])
        idh = np.exp(1j * np.concatenate([np.zeros((len(ph), 1)), ph], axis=1))
        ids_by_room[rd.room.name] = np.fft.irfft(idh, n=ssp_space.ssp_dim, axis=-1)
    return gains, ids_by_room, losses, np.asarray(params["g"])


# ---------------------------------------------------------------------------
# Post-hoc bin-level pruning: one room's strategy x criterion sweep
# ---------------------------------------------------------------------------


def room_prune_curves(rd, ssp_space, W, ids_hat, g, scale_idx, dprimes, strategies, criteria, args):
    """Post-hoc bin-level pruning sweep for one room. For every requested
    (strategy, criterion) pair, build a nested sequence of per-bin masks
    across `dprimes` and score EVERY one against the room's crosstalk-
    inclusive D_whole, so strategies are compared on the same footing
    regardless of which criterion chose their mask (magnitude/gate are the
    only strategies with a real whole-map/per-object split; priority is
    memory-blind by construction and random is a null baseline, so both
    report once under a bare key).

    D_whole/D_self are local temporaries -- never stored on `rd` -- built
    from the single held-out eval draw (moot for --encode-mode integral,
    which only ever has one draw) unless args.prune_gate_all_draws asks the
    gate strategy to fit against every training draw instead (points mode
    only; multiplies gate memory/compute by ~n_draws).
    """
    n_bins = rd.G_hat.shape[1]
    targets = jnp.asarray(rd.targets)
    g_half = gain_half_spectrum(g, scale_idx, n_bins)
    w_bin = bin_diag_weight(g, scale_idx, n_bins)
    V_eval = rd.V_hats[-1]

    D_whole = jnp.asarray(compute_D_bin(rd.G_hat, V_eval, ids_hat, w_bin))
    mu_full = jnp.ones(n_bins)
    baseline_cos = float(cos_maps(D_whole, mu_full, targets).mean())

    # self-consistency check against the already-trusted scale-grouped shortcut
    D_scale = jnp.asarray(compute_D(rd.G_hat, V_eval, ids_hat, W))
    w_scale = jnp.asarray(np.concatenate([[1.0], np.asarray(g) ** 2]))
    ref_cos = float(cos_maps(D_scale, w_scale, targets).mean())
    assert abs(baseline_cos - ref_cos) < 1e-3, (
        f"bin-level D_whole ({baseline_cos:.5f}) disagrees with the scale-grouped "
        f"shortcut ({ref_cos:.5f}) at D'=n_bins -- bin_diag_weight/compute_D_bin bug")

    M_hat_raw = (ids_hat * V_eval).sum(axis=0)
    component = g_half * M_hat_raw
    total_energy = retained_energy(component, np.ones(n_bins, dtype=bool))

    dprimes_desc = sorted(set(dprimes), reverse=True)
    curves, gate_histories = {}, {}

    def record(key, masks):
        cos_mean, cos_per_obj, energy_frac, mask_rows = [], [], [], []
        for dp in dprimes_desc:
            mu = masks[dp]
            cos = np.asarray(cos_maps(D_whole, jnp.asarray(mu, dtype=jnp.float32), targets))
            cos_per_obj.append(cos)
            cos_mean.append(float(cos.mean()))
            energy_frac.append(retained_energy(component, mu) / total_energy)
            mask_rows.append(mu)
        # store ascending in D' -- more readable plots/tables/threshold lookups
        curves[key] = dict(cos_mean=np.array(cos_mean[::-1]),
                            cos_per_obj=np.array(cos_per_obj[::-1]),
                            energy_frac=np.array(energy_frac[::-1]),
                            masks=np.array(mask_rows[::-1]))

    need_self = "gate" in strategies and "per-object" in criteria
    if args.prune_gate_all_draws and len(rd.V_hats) > 1:
        train_draws = rd.V_hats[:-1]
    else:
        train_draws = [V_eval]
    D_whole_train = [jnp.asarray(compute_D_bin(rd.G_hat, v, ids_hat, w_bin)) for v in train_draws]
    D_self_train = ([jnp.asarray(compute_D_bin_selfonly(rd.G_hat, v, w_bin)) for v in train_draws]
                    if need_self else None)

    for strategy in strategies:
        if strategy == "priority":
            priority = bin_priority(ssp_space, scale_idx)
            record("priority", ranking_curve(priority, dprimes_desc, descending=False))
            continue
        if strategy == "random":
            rng = np.random.default_rng(args.prune_seed)
            record("random", {dp: random_mask(n_bins, dp, rng=rng) for dp in dprimes_desc})
            continue
        for criterion in criteria:
            key = f"{strategy}_{criterion}"
            if strategy == "magnitude":
                score = (magnitude_score_whole(ids_hat, V_eval, g_half) if criterion == "whole-map"
                         else magnitude_score_object(V_eval, g_half))
                record(key, ranking_curve(score, dprimes_desc, descending=True))
            elif strategy == "gate":
                D_list = D_whole_train if criterion == "whole-map" else D_self_train

                def loss_fn(mu, D_list=D_list):
                    return jnp.mean(jnp.stack([1.0 - cos_maps(D, mu, targets).mean() for D in D_list]))

                masks, gate_results = progressive_gate_masks(
                    n_bins, dprimes_desc, loss_fn, steps=args.prune_gate_steps,
                    lr=args.prune_gate_lr, lam=args.prune_gate_lam)
                gate_histories[key] = {int(dp): gate_results[dp].loss_history.tolist()
                                       for dp in dprimes_desc}
                record(key, masks)
            else:
                raise ValueError(f"unknown prune strategy {strategy!r}")

    return dict(n_bins=n_bins, dprimes=np.array(dprimes_desc[::-1]), baseline_cos=baseline_cos,
                curves=curves, gate_loss_histories=gate_histories)


def dprime_index_at_threshold(cv, base, threshold):
    """Index into a strategy's dprimes-ascending arrays of the smallest D'
    retaining >= threshold * base cosine, or None if the swept range never
    reaches it."""
    hits = np.nonzero(cv["cos_mean"] >= threshold * base)[0]
    return int(hits[0]) if len(hits) else None


def mask_at_threshold(curves, key, threshold):
    """(D', mask, cos_per_obj) for the smallest swept D' retaining
    >= threshold of a room's baseline cosine under `key`'s strategy;
    falls back to the largest swept D' if the threshold was never reached."""
    cv = curves["curves"][key]
    idx = dprime_index_at_threshold(cv, curves["baseline_cos"], threshold)
    if idx is None:
        idx = len(curves["dprimes"]) - 1
    return int(curves["dprimes"][idx]), cv["masks"][idx], cv["cos_per_obj"][idx]


def pruned_room_sims(rd, ids_hat, g, scale_idx, mask):
    """Actual per-object similarity maps for a room's map after applying a
    bin-level prune mask -- reuses the same D_bin construction
    room_prune_curves scores masks with, to visualize (not just summarize)
    what a specific pruned map looks like spatially. Shape/convention matches
    explicit_sim_maps's output, so it drops straight into overlay_figure."""
    n_bins = rd.G_hat.shape[1]
    w_bin = bin_diag_weight(g, scale_idx, n_bins)
    D_whole = compute_D_bin(rd.G_hat, rd.V_hats[-1], ids_hat, w_bin)
    return np.einsum("ogb,b->og", D_whole, mask.astype(np.float64))


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
# Adaptive direct-optim position decoding (grid sized by the kernel FWHM)
# ---------------------------------------------------------------------------


def kernel_bump_fwhm(ssp_space, g_full, r_max=5.0, n_r=1000, n_dirs=8):
    """FWHM of the modulated kernel's central bump with the DC pedestal
    subtracted. The pedestal (DC gain fixed at 1) carries no spatial
    information, and for weakly-modulated gains it keeps the raw kernel
    above half max everywhere, which breaks kernel_fwhm's rule."""
    rs = np.linspace(0.0, r_max, n_r)
    v0 = encode_mod(ssp_space, g_full, np.zeros((1, 2)))[0]
    K = np.zeros(n_r)
    for th in np.linspace(0.0, np.pi, n_dirs, endpoint=False):
        pts = np.outer(rs, [np.cos(th), np.sin(th)])
        K += encode_mod(ssp_space, g_full, pts) @ v0
    K /= n_dirs
    ped = K[int(0.8 * n_r):].mean()
    bump = (K - ped) / max(K[0] - ped, 1e-9)
    below = np.nonzero(bump < 0.5)[0]
    return 2.0 * rs[below[0]] if len(below) else 2.0 * r_max


def decode_room(ssp_space, g, scale_idx, rd, ids):
    """Decode every object's position from the room map on the eval draw:
    initial grid sized by the modulated kernel's bump FWHM (pedestal-
    corrected library rule), coarse argmax + L-BFGS-B refinement."""
    g_full = full_gain_spectrum(g, scale_idx)
    x0, x1, y0, y1 = rd.room.interior
    l_eff = min(kernel_bump_fwhm(ssp_space, g_full, r_max=5.0), x1 - x0)
    n_pts = pts_per_dim(x1 - x0, l_eff)
    gx, gy = np.linspace(x0, x1, n_pts), np.linspace(y0, y1, n_pts)
    X, Y = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([X.ravel(), Y.ravel()])
    mod_grid = encode_mod(ssp_space, g_full, grid_pts).astype(np.float32)

    V = np.stack([encode_mod(ssp_space, g_full, p).mean(axis=0)
                  for p in rd.point_draws[-1]])
    M = ssp_space.bind(ids, V).sum(axis=0)
    queries = ssp_space.bind(M[None], ssp_space.invert(ids))
    decoded = np.empty((len(rd.items), 2))
    nfev = 0
    for i in range(len(rd.items)):
        decoded[i], nf = direct_optim_decode(
            queries[i], ssp_space, g_full, [(x0, x1), (y0, y1)], grid_pts, mod_grid)
        nfev += nf
    hits = np.array([MplPath(it.polygon).contains_point(p, radius=1e-9)
                     for it, p in zip(rd.items, decoded)])
    errs = np.array([np.linalg.norm(p - it.center)
                     for it, p in zip(rd.items, decoded)])
    return dict(l_eff=l_eff, n_pts=n_pts, grid_x=gx, grid_y=gy,
                decoded=decoded, hits=hits, errs=errs, nfev=nfev)


def print_decode(label, rooms_data, dec_by_room):
    print(f"\ndirect-optim decode ({label}), init grid from kernel FWHM")
    print(f"{'room':<10} {'l_eff':>6} {'pts/dim':>8} {'hits':>7} {'med err':>8} {'nfev':>6}")
    for rd in rooms_data:
        dec = dec_by_room[rd.room.name]
        print(f"{rd.room.theme:<10} {dec['l_eff']:>6.3f} {dec['n_pts']:>8} "
              f"{int(dec['hits'].sum()):>4}/{len(rd.items):<2} "
              f"{np.median(dec['errs']):>8.3f} {dec['nfev']:>6}")
    hits = np.concatenate([dec_by_room[rd.room.name]["hits"] for rd in rooms_data])
    errs = np.concatenate([dec_by_room[rd.room.name]["errs"] for rd in rooms_data])
    print(f"{'ALL':<10} {'':>6} {'':>8} {int(hits.sum()):>4}/{len(hits):<2} "
          f"{np.median(errs):>8.3f}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def overlay_figure(env, rooms_data, sims_by_room, cos_by_room, title, path,
                   grid_n, alpha_thr=0.2, extra_by_room=None, caption=None):
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
        label = f"{rd.room.theme.upper()}  cos={np.mean(cos):.3f}"
        if extra_by_room is not None:
            label += f"  {extra_by_room[rd.room.name]}"
        ax.text(x0 + 0.1, y1 - 0.12, label,
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
    if caption is not None:
        fig.text(0.5, 0.002, caption, ha="center", va="bottom", fontsize=8.5, color="0.4")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def gains_figure(rooms_data, gains, g_shared, scales, path):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True, facecolor="white")
    # arrange panels to match the world layout (row 1 on top)
    by_rc = {(rd.room.row, rd.room.col): (rd, g) for rd, g in zip(rooms_data, gains)}
    x = np.arange(len(scales))
    for (r, c), (rd, g) in by_rc.items():
        ax = axs[1 - r, c]
        ax.bar(x, g, color="tab:blue", label="per-room")
        ax.step(x, g_shared, where="mid", color="0.3", lw=1.4, label="shared")
        ax.set_title(f"{rd.room.theme} ({len(rd.items)} items)", fontsize=10)
        ax.set_xlabel("scale (coarse -> fine)")
        if c == 0:
            ax.set_ylabel("gain")
    axs[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("learned per-room scale gains (shared single profile for reference)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def decode_figure(env, rooms_data, dec_by_room, title, path):
    """Decoded object positions (colored 'x', colors match the overlay
    figures) over each room's initial sampling grid (thin black lines)."""
    fig, ax = plt.subplots(figsize=(13.2, 9.2), facecolor="white")
    cmap20 = plt.get_cmap("tab20")
    for rd in rooms_data:
        dec = dec_by_room[rd.room.name]
        x0, x1, y0, y1 = rd.room.interior
        for gx in dec["grid_x"]:
            ax.plot([gx, gx], [y0, y1], color="black", lw=0.6, alpha=0.5, zorder=1)
        for gy in dec["grid_y"]:
            ax.plot([x0, x1], [gy, gy], color="black", lw=0.6, alpha=0.5, zorder=1)
        for j, it in enumerate(rd.items):
            color = cmap20(j % 20)
            ax.add_patch(MplPolygon(it.polygon, closed=True, facecolor="none",
                                    edgecolor=color, lw=1.1, zorder=3))
            ax.plot(*dec["decoded"][j], marker="x", color=color, ms=9, mew=2.4,
                    ls="none", zorder=6)
        ax.text(x0 + 0.1, y1 - 0.12,
                f"{rd.room.theme.upper()}  {dec['n_pts']}x{dec['n_pts']} grid  "
                f"hits {int(dec['hits'].sum())}/{len(rd.items)}",
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


def prune_figure(rooms_data, curves_by_room, path):
    """Per-room retained-cosine vs D' compression curves: one line per
    strategy/criterion key (solid=whole-map or criterion-agnostic,
    dashed=per-object), dotted horizontal reference at baseline cosine."""
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), sharey=True, facecolor="white")
    by_rc = {(rd.room.row, rd.room.col): rd for rd in rooms_data}
    cmap = plt.get_cmap("tab10")
    for (r, c), rd in by_rc.items():
        ax = axs[1 - r, c]
        curves = curves_by_room[rd.room.name]
        dprimes = curves["dprimes"]
        for i, (key, cv) in enumerate(curves["curves"].items()):
            ls = "--" if key.endswith("per-object") else "-"
            ax.plot(dprimes, cv["cos_mean"], ls, marker=".", ms=4, color=cmap(i % 10), label=key)
        ax.axhline(curves["baseline_cos"], color="0.3", lw=1.0, ls=":")
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_title(f"{rd.room.theme} ({len(rd.items)} items)", fontsize=10)
        ax.set_xlabel("D' (bins kept)")
        if c == 0:
            ax.set_ylabel("retained cosine")
    axs[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("post-hoc bin-level pruning: retained cosine vs. bins kept")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def prune_scale_breakdown_figure(rooms_data, curves_by_room, scale_idx, n_scales, key, path):
    """Per room: fraction of each scale's bins kept across the D' sweep, for
    one strategy/criterion key -- shows WHERE a strategy's compression comes
    from (e.g. priority prunes fine scales first by construction; whether
    magnitude/gate do too is an empirical question this answers)."""
    fig, axs = plt.subplots(2, 3, figsize=(14, 7), facecolor="white", sharey=True)
    by_rc = {(rd.room.row, rd.room.col): rd for rd in rooms_data}
    im = None
    for (r, c), rd in by_rc.items():
        ax = axs[1 - r, c]
        curves = curves_by_room[rd.room.name]
        masks = curves["curves"][key]["masks"]                 # (n_dprimes, n_bins) bool
        frac = bins_kept_by_scale(masks, scale_idx, n_scales)   # (n_dprimes, n_scales)
        im = ax.imshow(frac.T, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(curves["dprimes"])))
        ax.set_xticklabels(curves["dprimes"], rotation=45, fontsize=6)
        ax.set_title(f"{rd.room.theme} ({len(rd.items)} items)", fontsize=10)
        ax.set_xlabel("D' (bins kept)")
        if c == 0:
            ax.set_yticks(range(n_scales))
            ax.set_ylabel("scale (coarse -> fine)")
    fig.suptitle(f"bins kept per scale across the D' sweep -- {key}")
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    fig.colorbar(im, ax=axs, shrink=0.7, label="fraction of that scale's bins kept")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def prune_summary_table(rooms_data, curves_by_room):
    """Smallest D' (of the swept values) retaining >= 95%/90%/80% of the
    room's baseline cosine, per room and per strategy/criterion."""
    thresholds = (0.95, 0.90, 0.80)
    print("\npost-hoc bin-level pruning: smallest swept D' retaining >= X% of baseline cosine")
    print(f"{'room':<10} {'strategy':<24}" + "".join(f"{f'>={int(t * 100)}%':>8}" for t in thresholds))
    for rd in rooms_data:
        curves = curves_by_room[rd.room.name]
        base = curves["baseline_cos"]
        for key, cv in curves["curves"].items():
            row = f"{rd.room.theme:<10} {key:<24}"
            for t in thresholds:
                idx = dprime_index_at_threshold(cv, base, t)
                dp = int(curves["dprimes"][idx]) if idx is not None else -1
                row += f"{dp:>8}"
            print(row)


def save_prune_npz(rooms_data, curves_by_room, path):
    save = {}
    for rd in rooms_data:
        curves = curves_by_room[rd.room.name]
        theme = rd.room.theme
        save[f"prune_{theme}_dprimes"] = curves["dprimes"]
        for key, cv in curves["curves"].items():
            save[f"prune_{theme}_{key}_cos_mean"] = cv["cos_mean"]
            save[f"prune_{theme}_{key}_cos_per_obj"] = cv["cos_per_obj"]
            save[f"prune_{theme}_{key}_energy"] = cv["energy_frac"]
            save[f"prune_{theme}_{key}_mask"] = cv["masks"]
    np.savez(path, **save)
    print(f"wrote {path}")


def prune_json_payload(args, rooms_data, curves_by_room):
    rooms = {}
    for rd in rooms_data:
        curves = curves_by_room[rd.room.name]
        rooms[rd.room.theme] = dict(
            n_obj=len(rd.items), baseline_cos=curves["baseline_cos"],
            dprimes=curves["dprimes"].tolist(),
            curves={key: dict(cos_mean=cv["cos_mean"].tolist(), energy_frac=cv["energy_frac"].tolist())
                    for key, cv in curves["curves"].items()},
            gate_loss_histories=curves["gate_loss_histories"])
    prune_args = {k: v for k, v in vars(args).items() if k.startswith("prune")}
    return dict(args=prune_args, rooms=rooms)


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
    ap.add_argument("--encode-mode", choices=["integral", "points"], default="integral",
                    help="'integral': exact footprint area integral on a dense grid; "
                         "'points': random interior point samples (--k-mode)")
    ap.add_argument("--integral-res", type=float, default=0.04,
                    help="grid resolution (m) for the area integral")
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
    ap.add_argument("--gain-param", choices=["free", "gaussian", "window"], default="free",
                    help="'free': one gain per scale; 'gaussian': discretized "
                         "Gaussian over scale blocks, learn only (mu, sigma); "
                         "'window': boxcar of --window-width 1s, learn only its center")
    ap.add_argument("--window-width", type=int, default=4,
                    help="number of consecutive scales in the sliding window")
    ap.add_argument("--train-ids", action=argparse.BooleanOptionalAction, default=True,
                    help="jointly train unitary object IDs (Fourier phases) with the gains")
    ap.add_argument("--id-steps", type=int, default=1000, help="joint training steps")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--reg-w", type=float, default=None,
                    help="L1 sparsity penalty on the gains "
                         "(default: 0.08 free / 0 gaussian)")
    ap.add_argument("--coarse-w", type=float, default=None,
                    help="scale-weighted penalty biasing toward coarse scales "
                         "(default: 0.06 free / 0.03 gaussian)")
    ap.add_argument("--env-seed", type=int, default=7)
    ap.add_argument("--seed", type=int, default=3, help="IDs + point sampling")
    ap.add_argument("--out", type=str, default="results/indoor_env/room_id_modulation.png")
    ap.add_argument("--prune", action=argparse.BooleanOptionalAction, default=False,
                    help="run a post-hoc bin-level pruning sweep after gain/ID training "
                         "(see scripts/vsa_bin_pruning.py)")
    ap.add_argument("--prune-strategies", nargs="+", default=["magnitude", "priority", "gate"],
                    choices=["magnitude", "priority", "gate", "random"])
    ap.add_argument("--prune-criteria", nargs="+", default=["whole-map", "per-object"],
                    choices=["whole-map", "per-object"],
                    help="only magnitude/gate have a real whole-map/per-object split; "
                         "priority (memory-blind) and random report once regardless")
    ap.add_argument("--prune-gains", choices=["learned", "shared", "raw"], default="learned",
                    help="which already-trained per-room gain profile to prune on top of")
    ap.add_argument("--prune-ids", choices=["auto", "codebook", "trained"], default="auto",
                    help="'auto': trained IDs if --train-ids else codebook IDs")
    ap.add_argument("--prune-dprimes", type=int, nargs="+", default=None,
                    help="explicit D' sweep (bins kept, DC always included); "
                         "default: log-spaced sweep from n_bins to --prune-dprimes-floor")
    ap.add_argument("--prune-dprimes-n", type=int, default=12)
    ap.add_argument("--prune-dprimes-floor", type=int, default=8)
    ap.add_argument("--prune-gate-steps", type=int, default=150)
    ap.add_argument("--prune-gate-lr", type=float, default=0.05)
    ap.add_argument("--prune-gate-lam", type=float, default=1e-3)
    ap.add_argument("--prune-gate-all-draws", action=argparse.BooleanOptionalAction, default=False,
                    help="fit the gate strategy against every training point-draw instead of "
                         "just the held-out eval draw (--encode-mode points only; ~n_draws more "
                         "memory/compute)")
    ap.add_argument("--prune-seed", type=int, default=None, help="default: --seed")
    ap.add_argument("--prune-map", action=argparse.BooleanOptionalAction, default=False,
                    help="also render pruned-map similarity overlays (like overlay_figure) at "
                         "--prune-map-thresholds cosine-retention points (requires --prune)")
    ap.add_argument("--prune-map-strategy", type=str, nargs="+", default=["magnitude_whole-map"],
                    help="which --prune curves key(s) to visualize, e.g. magnitude_whole-map, "
                         "magnitude_per-object, priority, gate_whole-map, gate_per-object, random -- "
                         "pass 'all' to render every key actually computed by --prune-strategies/"
                         "--prune-criteria (one figure per key per threshold, reusing the same sweep)")
    ap.add_argument("--prune-map-thresholds", type=float, nargs="+", default=[0.9, 0.8, 0.6],
                    help="fraction of baseline cosine to retain at each rendered pruned map")
    ap.add_argument("--prune-scale-breakdown", action=argparse.BooleanOptionalAction, default=False,
                    help="render a bins-kept-per-scale heatmap across the D' sweep, per strategy "
                         "in --prune-scale-breakdown-strategy (requires --prune)")
    ap.add_argument("--prune-scale-breakdown-strategy", type=str, nargs="+", default=["priority"],
                    help="which --prune curves key(s) to break down by scale; 'all' for every "
                         "computed key (same key names as --prune-map-strategy)")
    args = ap.parse_args()
    if args.prune_seed is None:
        args.prune_seed = args.seed
    if args.prune_ids == "trained" and not args.train_ids:
        ap.error("--prune-ids trained requires --train-ids")
    if args.prune_scale_breakdown and not args.prune:
        ap.error("--prune-scale-breakdown requires --prune")
    if args.prune_map and not args.prune:
        ap.error("--prune-map requires --prune")

    # per-parameterization reg defaults: the fixed-amplitude Gaussian needs no
    # L1 (2 params regularize themselves) and a milder coarse bias
    if args.reg_w is None:
        args.reg_w = 0.08 if args.gain_param == "free" else 0.0
    if args.coarse_w is None:
        args.coarse_w = 0.06 if args.gain_param == "free" else 0.03

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
    integral_res = args.integral_res if args.encode_mode == "integral" else None
    if args.encode_mode == "integral":
        args.n_draws = 1   # the area integral is deterministic
        print(f"ssp_dim={d}, {len(env.items)} objects in {len(env.rooms)} rooms, "
              f"encode=integral (res {args.integral_res} m)")
    else:
        print(f"ssp_dim={d}, {len(env.items)} objects in {len(env.rooms)} rooms, "
              f"encode=points (k_mode={args.k_mode}), {args.n_draws}+1 point draws")

    rng = np.random.default_rng(args.seed)
    rooms_data, ids_by_room = [], {}
    t0 = time.time()
    for room in env.rooms:
        items = [it for it in env.items if it.room == room.name]
        # fixed orthogonal unitary codebook (not trained), one SP per object
        ids = SPSpace(len(items), d, rng=rng).vectors
        ids_by_room[room.name] = ids
        n_bins = (d + 1) // 2
        ids_hat = np.fft.rfft(ids, axis=-1)[:, :n_bins]
        rd = RoomData(room, items, ssp_space, W, ids_hat,
                      args.grid_n, k_fn, rng, args.n_draws, integral_res=integral_res)
        rooms_data.append(rd)
        print(f"  {room.theme:<10} pts per object: {min(rd.k_per_item)}-{max(rd.k_per_item)} "
              f"(total {sum(rd.k_per_item)})")
    print(f"precomputed D coefficients in {time.time() - t0:.0f}s")

    def report_params(raw, rds=None):
        names = [rd.room.theme for rd in rds] if rds else ["shared"]
        if args.gain_param == "gaussian":
            print("  " + ", ".join(f"{n}: mu={p[0]:.2f} sig={softplus_np(p[1]) + 1e-3:.2f}"
                                   for n, p in zip(names, raw)))
        elif args.gain_param == "window":
            print("  " + ", ".join(f"{n}: c={p[0]:.2f}" for n, p in zip(names, raw)))

    print("training per-room gains through the unbind-decode loss ...")
    t0 = time.time()
    gains, _, raw_l = train(rooms_data, ssp_space, args)
    report_params(raw_l, rooms_data)
    print("training one shared gain profile for all rooms ...")
    g_shared_all, _, raw_s = train(rooms_data, ssp_space, args, shared=True)
    g_shared = g_shared_all[0]
    report_params(raw_s)
    gains_j, ids_trained = None, None
    if args.train_ids:
        print("jointly training gains + unitary IDs (Fourier phases) ...")
        gains_j, ids_trained, _, raw_j = train_ids(rooms_data, ssp_space, scale_idx, args)
        report_params(raw_j, rooms_data)
        print("jointly training ONE shared gain profile + unitary IDs ...")
        gains_js, ids_trained_sh, _, raw_js = train_ids(rooms_data, ssp_space, scale_idx,
                                                        args, shared=True)
        g_shared_j = gains_js[0]
        report_params(raw_js)
    print(f"trained in {time.time() - t0:.0f}s")

    if args.gain_param == "window":   # snap soft training windows to hard 0/1
        n_sc = ssp_space.n_scales
        gains = snap_window(raw_l, n_sc, args.window_width)
        g_shared = snap_window(raw_s, n_sc, args.window_width)[0]
        if args.train_ids:
            gains_j = snap_window(raw_j, n_sc, args.window_width)
            g_shared_j = snap_window(raw_js, n_sc, args.window_width)[0]

    # held-out evaluation, explicit path (also verifies the factorization)
    g_ones = np.ones(ssp_space.n_scales)
    header = f"\n{'room':<10} {'n_obj':>5} {'cos none':>9} {'cos shared':>11} {'cos per-room':>13}"
    if args.train_ids:
        header += f" {'cos sh+IDs':>11} {'cos +IDs':>9}"
    print(header)
    sims_l, sims_s, sims_t = [], [], []
    cos_n, cos_s, cos_l, cos_ts, cos_t = {}, {}, {}, {}, {}
    for ri, (rd, g) in enumerate(zip(rooms_data, gains)):
        ids = ids_by_room[rd.room.name]
        _, c_n = eval_room(ssp_space, g_ones, scale_idx, W, rd, ids)
        s_s, c_s = eval_room(ssp_space, g_shared, scale_idx, W, rd, ids)
        s_l, c_l = eval_room(ssp_space, g, scale_idx, W, rd, ids)
        sims_s.append(s_s); sims_l.append(s_l)
        cos_n[rd.room.name], cos_s[rd.room.name], cos_l[rd.room.name] = c_n, c_s, c_l
        line = (f"{rd.room.theme:<10} {len(rd.items):>5} {np.mean(c_n):>9.3f} "
                f"{np.mean(c_s):>11.3f} {np.mean(c_l):>13.3f}")
        if args.train_ids:
            _, c_ts = eval_room(ssp_space, g_shared_j, scale_idx, W, rd,
                                ids_trained_sh[rd.room.name])
            s_t, c_t = eval_room(ssp_space, gains_j[ri], scale_idx, W, rd,
                                 ids_trained[rd.room.name])
            sims_t.append(s_t)
            cos_ts[rd.room.name], cos_t[rd.room.name] = c_ts, c_t
            line += f" {np.mean(c_ts):>11.3f} {np.mean(c_t):>9.3f}"
        print(line)
    all_n = np.concatenate(list(cos_n.values()))
    all_s = np.concatenate(list(cos_s.values()))
    all_l = np.concatenate(list(cos_l.values()))
    line = (f"{'ALL':<10} {len(all_l):>5} {all_n.mean():>9.3f} "
            f"{all_s.mean():>11.3f} {all_l.mean():>13.3f}")
    if args.train_ids:
        line += (f" {np.concatenate(list(cos_ts.values())).mean():>11.3f}"
                 f" {np.concatenate(list(cos_t.values())).mean():>9.3f}")
    print(line)

    if args.prune:
        use_trained_ids = args.prune_ids == "trained" or (args.prune_ids == "auto" and args.train_ids)
        n_bins = (d + 1) // 2
        if args.prune_dprimes:
            dprimes = args.prune_dprimes
        else:
            sweep = np.geomspace(n_bins, args.prune_dprimes_floor, args.prune_dprimes_n)
            dprimes = sorted(set(np.round(sweep).astype(int).tolist()))
        # gains must be paired with the SAME training run that produced the
        # chosen IDs (gains_j/g_shared_j were jointly trained alongside
        # ids_trained -- pairing trained IDs with the plain, non-joint gains
        # array evaluates a map nobody actually trained)
        if use_trained_ids:
            gains_by_mode = {"learned": gains_j, "shared": np.tile(g_shared_j, (len(rooms_data), 1)),
                             "raw": np.tile(g_ones, (len(rooms_data), 1))}
        else:
            gains_by_mode = {"learned": gains, "shared": np.tile(g_shared, (len(rooms_data), 1)),
                             "raw": np.tile(g_ones, (len(rooms_data), 1))}
        gains_src = gains_by_mode[args.prune_gains]

        print(f"\nrunning post-hoc bin-level pruning sweep (n_bins={n_bins}, D'={dprimes}, "
              f"ids={'trained' if use_trained_ids else 'codebook'}, gains={args.prune_gains}) ...")
        curves_by_room, ids_hat_by_room_prune = {}, {}
        for ri, rd in enumerate(rooms_data):
            ids_hat_room = (np.fft.rfft(ids_trained[rd.room.name], axis=-1)[:, :n_bins]
                            if use_trained_ids else rd.ids_hat)
            ids_hat_by_room_prune[rd.room.name] = ids_hat_room
            curves_by_room[rd.room.name] = room_prune_curves(
                rd, ssp_space, W, ids_hat_room, gains_src[ri], scale_idx, dprimes,
                args.prune_strategies, args.prune_criteria, args)
            print(f"  {rd.room.theme:<10} baseline cos {curves_by_room[rd.room.name]['baseline_cos']:.3f}")
        prune_summary_table(rooms_data, curves_by_room)
        prune_figure(rooms_data, curves_by_room, out_path.with_name(out_path.stem + "_prune.png"))
        save_prune_npz(rooms_data, curves_by_room, out_path.with_name(out_path.stem + "_prune.npz"))
        with open(out_path.with_name(out_path.stem + "_prune.json"), "w") as f:
            json.dump(prune_json_payload(args, rooms_data, curves_by_room), f, indent=2)

        if args.prune_map:
            sample_curves = next(iter(curves_by_room.values()))
            map_strategies = (list(sample_curves["curves"]) if args.prune_map_strategy == ["all"]
                              else args.prune_map_strategy)
            unknown = [s for s in map_strategies if s not in sample_curves["curves"]]
            if unknown:
                ap.error(f"--prune-map-strategy {unknown} was not computed "
                         f"(available: {list(sample_curves['curves'])}); check "
                         f"--prune-strategies/--prune-criteria")
            for map_strategy in map_strategies:
                for threshold in args.prune_map_thresholds:
                    sims_by_room, cos_by_room_map, dprimes_used = [], {}, {}
                    for ri, rd in enumerate(rooms_data):
                        curves = curves_by_room[rd.room.name]
                        dp, mask, cos_po = mask_at_threshold(curves, map_strategy, threshold)
                        sims_by_room.append(pruned_room_sims(
                            rd, ids_hat_by_room_prune[rd.room.name], gains_src[ri], scale_idx, mask))
                        cos_by_room_map[rd.room.name] = cos_po
                        reduction_pct = 100.0 * (1.0 - dp / n_bins)
                        dprimes_used[rd.room.name] = f"D'={dp} (-{reduction_pct:.0f}%)"
                    pct = int(round(threshold * 100))
                    suffix = f"_prune_map_{map_strategy}_p{pct}"
                    title = f"pruned map: {map_strategy}, >= {pct}% baseline cosine"
                    caption = (f"D' = bins kept out of n_bins={n_bins} total rfft half-spectrum "
                               f"bins (rest zeroed); (-X%) = percent reduction, 100*(1 - D'/n_bins)")
                    overlay_figure(env, rooms_data, sims_by_room, cos_by_room_map, title,
                                  out_path.with_name(out_path.stem + suffix + ".png"), args.grid_n,
                                  extra_by_room=dprimes_used, caption=caption)

        if args.prune_scale_breakdown:
            sample_curves = next(iter(curves_by_room.values()))
            breakdown_strategies = (list(sample_curves["curves"])
                                    if args.prune_scale_breakdown_strategy == ["all"]
                                    else args.prune_scale_breakdown_strategy)
            unknown = [s for s in breakdown_strategies if s not in sample_curves["curves"]]
            if unknown:
                ap.error(f"--prune-scale-breakdown-strategy {unknown} was not computed "
                         f"(available: {list(sample_curves['curves'])})")
            for key in breakdown_strategies:
                prune_scale_breakdown_figure(
                    rooms_data, curves_by_room, scale_idx, ssp_space.n_scales, key,
                    out_path.with_name(out_path.stem + f"_prune_scale_{key}.png"))

    # adaptive direct-optim position decoding, per-room grids from l_eff
    dec_l = {rd.room.name: decode_room(ssp_space, g, scale_idx, rd,
                                       ids_by_room[rd.room.name])
             for rd, g in zip(rooms_data, gains)}
    print_decode("codebook IDs + per-room gains", rooms_data, dec_l)
    dec_t = None
    if args.train_ids:
        dec_t = {rd.room.name: decode_room(ssp_space, gains_j[ri], scale_idx, rd,
                                           ids_trained[rd.room.name])
                 for ri, rd in enumerate(rooms_data)}
        print_decode("trained IDs + joint gains", rooms_data, dec_t)

    overlay_figure(env, rooms_data, sims_l, cos_l,
                   "unbound object similarity maps, learned per-room scale gains "
                   "(one color per object, alpha ~ similarity)",
                   out_path, args.grid_n)
    overlay_figure(env, rooms_data, sims_s, cos_s,
                   "unbound object similarity maps, one shared gain profile "
                   "for all rooms (one color per object, alpha ~ similarity)",
                   out_path.with_name(out_path.stem + "_sharedgain.png"), args.grid_n)
    if args.train_ids:
        overlay_figure(env, rooms_data, sims_t, cos_t,
                       "unbound object similarity maps, trained unitary IDs + "
                       "joint gains (one color per object, alpha ~ similarity)",
                       out_path.with_name(out_path.stem + "_trainedIDs.png"), args.grid_n)
    gains_figure(rooms_data, gains, g_shared, np.asarray(ssp_space.scales),
                 out_path.with_name(out_path.stem + "_gains.png"))
    dec_fig = dec_t if args.train_ids else dec_l
    decode_figure(env, rooms_data, dec_fig,
                  "direct-optim decoded object positions (x), initial sampling "
                  "grid sized by each room's modulated kernel FWHM",
                  out_path.with_name(out_path.stem + "_decode.png"))

    save = dict(gains=gains, gains_shared=g_shared,
                scales=np.asarray(ssp_space.scales),
                rooms=[rd.room.theme for rd in rooms_data])
    if args.gain_param != "free":
        save["param_raw"] = raw_l
        save["param_raw_shared"] = raw_s
        if args.train_ids:
            save["param_raw_joint"] = raw_j
            save["param_raw_shared_joint"] = raw_js
    if args.train_ids:
        save["gains_joint"] = gains_j
        save["gains_shared_joint"] = g_shared_j
        for rd in rooms_data:
            save[f"ids_{rd.room.theme}"] = ids_trained[rd.room.name]
            save[f"ids_shared_{rd.room.theme}"] = ids_trained_sh[rd.room.name]
    np.savez(out_path.with_name(out_path.stem + "_gains.npz"), **save)

    def dec_metrics(dec):
        return {"l_eff": float(dec["l_eff"]), "n_pts_per_dim": int(dec["n_pts"]),
                "hits": dec["hits"].tolist(), "errs": dec["errs"].tolist(),
                "nfev": int(dec["nfev"])}

    metrics = {rd.room.theme: {"n_obj": len(rd.items),
                               "k_per_item": rd.k_per_item,
                               "cos_none": cos_n[rd.room.name].tolist(),
                               "cos_shared": cos_s[rd.room.name].tolist(),
                               "cos_learned": cos_l[rd.room.name].tolist(),
                               "decode_learned": dec_metrics(dec_l[rd.room.name]),
                               **({"cos_shared_trained_ids": cos_ts[rd.room.name].tolist(),
                                   "cos_trained_ids": cos_t[rd.room.name].tolist(),
                                   "decode_trained_ids": dec_metrics(dec_t[rd.room.name])}
                                  if args.train_ids else {})}
               for rd in rooms_data}
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)
    print(f"wrote {out_path.with_suffix('.json')} and _gains.npz")


if __name__ == "__main__":
    main()
