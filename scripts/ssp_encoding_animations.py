#!/usr/bin/env python
"""Animations demonstrating SSP encodings and the power of learning them.

Each animation is registered in ANIMATIONS and selected by name:

    python scripts/ssp_encoding_animations.py walk
    python scripts/ssp_encoding_animations.py modulate

Layout (both): left panel is the square environment with a Blues similarity
map (low values fade to white), the grey walk path, and a black dot at the
current location x; right panel shows rows a_k of the encoding phase matrix A
as arrows out of a central hub.

walk : the idea of SSPs, built up in phases (PHASE_TIME seconds each)
       while the agent walks one continuous random path from the center (the
       pattern tracks x(t), the grey trail never resets):
       1-3. the vectors of one hex group are introduced one at a time, each
            with its plane-wave pattern cos(a_k . (x'-x)) on the left
       4.   the whole group -> hexagonal interference pattern
       5.   a second hex group at a different scale and rotation
       6.   every displayed vector with the full similarity map (the kernel)
       A third panel shows phasor dials, one per displayed row: a needle at
       angle a_k . x(t) (one dial, then three, then a grid for the full
       set). Because the encoding is phi(x) = ifft(e^{iAx}), the needles
       spin as the agent moves and each arrow's alpha pulses with
       cos(a_k . x) in the final phase. The map is the mean of cos over the
       shown rows -- for the full set that is the similarity phi(x).phi(x')
       up to the 1/d DC term.

modulate : scale modulation during a random walk. A gain vector over the
       scales morphs smoothly ones -> small-scales-only -> large-scales-only
       (one PHASE_TIME hold or morph per segment). Arrow alphas follow their
       scale's gain (modulated-out arrows fade to MOD_ALPHA_MIN) and the
       similarity map is the gain-weighted mean of the plane waves, so the
       kernel visibly broadens (small scales) or sharpens (large scales).

learned : the power of learning the encoding. The similarity at the origin
       is held with every displayed vector fully on, then the phase matrix
       morphs per pixel from the standard HexSSP A to the A(x) learned for
       fractal_target.jpg by kernel_image_fit.py -- the kernel deforms into
       the fractal image while the arrows glide to the learned A(0) rows.
       Requires results/kernel_image_fit/params.msgpack.

All colors live in COLORS below so the scheme is easy to swap.
Writes <out>/ssp_<name>.mp4 and a downsampled <out>/ssp_<name>.gif.
"""

import argparse
import time
from pathlib import Path as FSPath

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import imageio.v2 as imageio

from vsagym.spaces import HexagonalSSPSpace

# ---------------------------------------------------------------------------
# Color scheme -- edit here to restyle every animation
# ---------------------------------------------------------------------------

COLORS = {
    "background": "white",
    "path": "#9aa0a6",        # grey random-walk trail
    "agent": "black",         # current-location dot
    "arrow": "#12406b",       # rows of A (alpha modulated per frame)
    "arrow_edge": "white",    # outline so arrows pop off dark backgrounds
    "sim_cmap": "Blues",      # similarity map, low sim -> white
    "frame_edge": "#c8cdd3",  # thin axes border
}

ARROW_ALPHA_MIN = 0.3   # walk: floor of the cos(a_k . x) alpha pulse
MOD_ALPHA_MIN = 0.1     # modulate: alpha of a fully modulated-out arrow
PHASE_TIME = 3.0        # seconds each phase / gain segment is held


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_walk(rng, n_frames, world, margin, speed=0.07, turn=0.22, start=None):
    """Momentum random walk: heading diffuses, steers away from the walls."""
    pos = (np.array(start, float) if start is not None
           else rng.uniform(world * 0.3, world * 0.7, 2))
    ang = rng.uniform(0, 2 * np.pi)
    out = np.empty((n_frames, 2))
    for i in range(n_frames):
        ang += turn * rng.normal()
        d_wall = min(pos.min(), world - pos.max())
        if d_wall < margin:                      # blend heading toward center
            to_c = np.arctan2(world / 2 - pos[1], world / 2 - pos[0])
            diff = (to_c - ang + np.pi) % (2 * np.pi) - np.pi
            ang += np.clip(diff, -0.35, 0.35) * (1 - d_wall / margin)
        pos = pos + speed * np.array([np.cos(ang), np.sin(ang)])
        pos = np.clip(pos, 0.15, world - 0.15)
        out[i] = pos
    return out


def half_phase_matrix(ssp_space):
    """Effective half-spectrum frequencies a_k (rad/m), DC row excluded."""
    n_free = (ssp_space.ssp_dim - 1) // 2
    return ssp_space.phase_matrix[1:n_free + 1] / ssp_space.length_scale.flatten()


def arrow_subset(ssp_space, a_half, args):
    """Hex subgroup drawn as arrows: all simplex vertices at a few scales and
    rotations (rows ordered rotation -> scale -> vertex), plus the conjugate
    rows -a_k. Returns (row indices, arrow rows, s_pick, r_pick)."""
    nv, ns, nr = ssp_space.grid_basis_dim, ssp_space.n_scales, ssp_space.n_rotates
    s_pick = np.round(np.linspace(0, ns - 1, args.arrow_scales)).astype(int)
    # space rotations are linspace(0, 120deg, nr, endpoint=False); stride
    # through them so the picks stay equally spaced on [0, 120deg)
    r_pick = np.floor(np.arange(args.arrow_rotates) * nr
                      / args.arrow_rotates).astype(int)
    idx = np.array([r * ns * nv + s * nv + v
                    for r in r_pick for s in s_pick for v in range(nv)])
    return idx, np.vstack([a_half[idx], -a_half[idx]]), s_pick, r_pick


def arrow_display_uv(a_set, mags_max, args):
    """Display vectors: direction of a_k, length = arrow_len * (|a_k|/max)^pow.
    The compressive power keeps small-scale arrows visible: with the default
    1/9 scale ratio and pow 0.5 the smallest is 1/3 the longest."""
    mags = np.linalg.norm(a_set, axis=1)
    disp = args.arrow_len * (mags / mags_max) ** args.arrow_len_pow
    return a_set / mags[:, None] * disp[:, None]


def env_grid(world, gn):
    """pcolormesh cell edges and flat cell-center coordinates."""
    cell = np.linspace(0, world, gn + 1)
    cc = (cell[:-1] + cell[1:]) / 2
    GX, GY = np.meshgrid(cc, cc)
    return cell, np.column_stack([GX.ravel(), GY.ravel()])


def two_panel_fig(cell, gn, arrow_len, bounds, phasors=False):
    """Left: environment (sim mesh, path, agent dot). Middle: arrow hub.
    With phasors=True, a third square panel for phasor dials (axp)."""
    ratios = [1.0, 0.8, 0.6] if phasors else [1.0, 1.0]
    fig, axs = plt.subplots(1, len(ratios),
                            figsize=(5.6 * sum(ratios), 5.6), dpi=120,
                            gridspec_kw={"width_ratios": ratios},
                            facecolor=COLORS["background"])
    ax, axr = axs[0], axs[1]
    axp = axs[2] if phasors else None
    ax.set_facecolor(COLORS["background"])
    mesh = ax.pcolormesh(cell, cell, np.zeros((gn, gn)), cmap=COLORS["sim_cmap"],
                         vmin=0, vmax=1, zorder=0, shading="flat")
    path_ln, = ax.plot([], [], color=COLORS["path"], lw=1.6, zorder=2,
                       solid_capstyle="round")
    dot, = ax.plot([], [], "o", color=COLORS["agent"], ms=7, zorder=4)
    ax.set_xlim(*bounds); ax.set_ylim(*bounds)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])

    axr.set_facecolor(COLORS["background"])
    axr.plot([0], [0], "o", color=COLORS["agent"], ms=7, zorder=4)
    lim = 1.25 * arrow_len
    axr.set_xlim(-lim, lim); axr.set_ylim(-lim, lim)
    axr.set_aspect("equal")
    axr.set_xticks([]); axr.set_yticks([])

    if phasors:
        axp.set_facecolor(COLORS["background"])
        axp.set_xlim(-1, 1); axp.set_ylim(-1, 1)
        axp.set_aspect("equal")
        axp.set_xticks([]); axp.set_yticks([])

    for a_ in axs:
        for s in a_.spines.values():
            s.set_visible(False)
    fig.tight_layout(pad=0.4)
    return fig, ax, axr, axp, mesh, path_ln, dot


def hub_quiver(axr, uv):
    return axr.quiver(np.zeros(len(uv)), np.zeros(len(uv)), uv[:, 0], uv[:, 1],
                      angles="xy", scale_units="xy", scale=1, width=0.006,
                      headwidth=6, headlength=8, headaxislength=7,
                      linewidth=0.6, zorder=3)


def phasor_dials(axp, n):
    """Dial circles + a needle LineCollection for n phasors: one big dial,
    a column of up to 3, else a 3-column grid of small dials."""
    if n == 1:
        centers, rad = [(0.0, 0.0)], 0.55
    elif n <= 3:
        centers, rad = [(0.0, 0.65), (0.0, 0.0), (0.0, -0.65)][:n], 0.28
    else:
        cols = 3
        xs = np.linspace(-0.5, 0.5, cols)
        ys = np.linspace(0.78, -0.78, int(np.ceil(n / cols)))
        centers = [(xs[i % cols], ys[i // cols]) for i in range(n)]
        rad = 0.125
    circles = [axp.add_patch(plt.Circle(c, rad, fill=False, lw=1.4,
                                        ec=COLORS["frame_edge"]))
               for c in centers]
    needles = LineCollection([], colors=[COLORS["arrow"]], lw=2.2,
                             capstyle="round")
    axp.add_collection(needles)
    return {"centers": np.asarray(centers), "rad": rad,
            "artists": circles + [needles], "needles": needles}


def set_needles(ph, angles, alphas=None):
    """Point each dial's needle at its angle; optionally set per-needle
    alpha (used to show modulation gains)."""
    tips = ph["centers"] + ph["rad"] * np.column_stack(
        [np.cos(angles), np.sin(angles)])
    ph["needles"].set_segments(
        [[tuple(c), tuple(t)] for c, t in zip(ph["centers"], tips)])
    if alphas is not None:
        rgb = matplotlib.colors.to_rgb(COLORS["arrow"])
        ph["needles"].set_colors([rgb + (a,) for a in alphas])


def frame_of(fig):
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def write_outputs(frames_iter, n_frames, out_base, args):
    writer = imageio.get_writer(out_base.with_suffix(".mp4"), fps=args.fps,
                                macro_block_size=1)
    gif_frames = []
    t0 = time.time()
    for f, frame in enumerate(frames_iter):
        # libx264 needs even dimensions; figure size rounding can yield odd
        frame = frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2]
        writer.append_data(frame)
        if f % args.gif_stride == 0:
            gif_frames.append(frame[:: args.gif_scale, :: args.gif_scale])
        if f % 50 == 0:
            print(f"  frame {f}/{n_frames} ({time.time() - t0:.0f}s)")
    writer.close()
    gif_path = out_base.with_suffix(".gif")
    imageio.mimsave(gif_path, gif_frames,
                    fps=max(1, round(args.fps / args.gif_stride)), loop=0)
    print(f"wrote {out_base.with_suffix('.mp4')} and {gif_path} "
          f"({len(gif_frames)} gif frames)")


def smoothstep(u):
    return u * u * (3 - 2 * u)


# ---------------------------------------------------------------------------
# Animation 1: "walk" -- the idea of SSPs
# ---------------------------------------------------------------------------


def anim_walk(args, out_base):
    world = args.world
    rng = np.random.default_rng(args.seed)
    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=args.n_scales,
                                  n_rotates=args.n_rotates,
                                  scale_min=args.scale_min,
                                  scale_sampling="log",
                                  length_scale=args.length_scale, rng=args.seed)
    d = ssp_space.ssp_dim
    a_half = half_phase_matrix(ssp_space)             # (n_free, 2)
    center = np.array([world / 2, world / 2])

    nv, ns = ssp_space.grid_basis_dim, ssp_space.n_scales
    idx, a_show, s_pick, r_pick = arrow_subset(ssp_space, a_half, args)
    print(f"ssp_dim={d}, phases={len(a_half)}, arrows shown={len(a_show)}")

    def group_rows(r, s):
        """The 3 simplex-vertex rows of one hex group (rotation r, scale s)."""
        return a_half[[r * ns * nv + s * nv + v for v in range(nv)]]

    g1 = group_rows(r_pick[0], s_pick[-1])   # fine scale, rotation 0
    # 2nd group: rotated and coarser, but not the smallest scale (which is
    # too coarse to show a lattice in this arena)
    g2 = group_rows(r_pick[1 % len(r_pick)], ns - 2)

    # full-set phasor grid: columns = the 3 hex-group vertices, rows sorted
    # by scale (stable, so rotations stay in order within a scale)
    ph_idx = idx[np.argsort((idx // nv) % ns, kind="stable")]

    # intro phases: (arrows drawn, pattern rows, phasor rows)
    intro = [
        (g1[:1], g1[:1], g1[:1]),        # 1st vector + plane wave + phasor
        (g1[1:2], g1[1:2], g1[1:2]),     # 2nd
        (g1[2:], g1[2:], g1[2:]),        # 3rd
        (g1, g1, g1),                    # whole group -> hex interference
        (g2, g2, g2),                    # another group: new scale + rotation
        (a_show, a_half, a_half[ph_idx]),   # everything + full similarity
    ]
    holds = [PHASE_TIME] * (len(intro) - 1) + [5.0]   # last phase lingers

    hold_frames = [round(h * args.fps) for h in holds]
    total = sum(hold_frames) + args.frames
    # one continuous walk through every phase; the path never resets
    traj = make_walk(rng, total, world, margin=1.2, start=center)
    mags_max = np.linalg.norm(a_half, axis=1).max()

    gn = args.grid_n
    cell, grid = env_grid(world, gn)
    fig, ax, axr, axp, mesh, path_ln, dot = two_panel_fig(
        cell, gn, args.arrow_len, (0, world), phasors=True)
    base_rgb = matplotlib.colors.to_rgb(COLORS["arrow"])
    edge_rgb = matplotlib.colors.to_rgb(COLORS["arrow_edge"])
    quiv = None

    # -- phasor panel: one dial per shown row, needle at angle a_k . x(t) ----
    ph = {"rows": None, "artists": []}

    def set_phasors(rows):
        for art in ph["artists"]:
            art.remove()
        ph.update(phasor_dials(axp, len(rows)), rows=rows)

    def draw(x, pat_rows, arr_rows, alphas, path_pts):
        """Update every artist for one frame and grab it. The pattern is the
        mean plane wave of pat_rows; for the full row set this is (up to the
        1/d DC term) the similarity phi(x) . phi(x')."""
        nonlocal quiv
        pat = np.cos((grid - x) @ pat_rows.T).mean(-1)
        mesh.set_array(np.clip(pat, 0, 1))
        path_ln.set_data(path_pts[:, 0], path_pts[:, 1])
        dot.set_data([x[0]], [x[1]])
        if quiv is not None:
            quiv.remove()
        quiv = hub_quiver(axr, arrow_display_uv(arr_rows, mags_max, args))
        quiv.set_facecolor([base_rgb + (a,) for a in alphas])
        quiv.set_edgecolor([edge_rgb + (a,) for a in alphas])
        set_needles(ph, ph["rows"] @ x)    # phasor needles: angle a_k . x(t)
        return frame_of(fig)

    def frames():
        f = 0
        for (arr_rows, pat_rows, ph_rows), nf in zip(intro, hold_frames):
            set_phasors(ph_rows)
            for _ in range(nf):
                x = traj[f]
                yield draw(x, pat_rows, arr_rows,
                           np.ones(len(arr_rows)), traj[: f + 1])
                f += 1
        while f < total:
            x = traj[f]
            yield draw(x, a_half, a_show, np.ones(len(a_show)), traj[: f + 1])
            f += 1

    write_outputs(frames(), total, out_base, args)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Animation 2: "modulate" -- scale modulation during a walk
# ---------------------------------------------------------------------------


def anim_modulate(args, out_base):
    world = args.world
    rng = np.random.default_rng(args.seed)
    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=args.n_scales,
                                  n_rotates=args.n_rotates,
                                  scale_min=args.scale_min,
                                  scale_sampling="log",
                                  length_scale=args.length_scale, rng=args.seed)
    d = ssp_space.ssp_dim
    a_half = half_phase_matrix(ssp_space)
    center = np.array([world / 2, world / 2])

    nv, ns = ssp_space.grid_basis_dim, ssp_space.n_scales
    idx, a_show, _, _ = arrow_subset(ssp_space, a_half, args)
    mags_max = np.linalg.norm(a_half, axis=1).max()

    # per-scale gain vectors (rows ordered rotation -> scale -> vertex)
    scales = np.asarray(ssp_space.scales)
    g_ones = np.ones(ns)
    g_small = (scales <= np.median(scales)).astype(float)   # keep small scales
    g_large = (scales >= np.median(scales)).astype(float)   # keep large scales
    row_s = (np.arange(len(a_half)) // nv) % ns              # scale idx per row
    show_s = np.concatenate([(idx // nv) % ns] * 2)          # incl. conjugates
    print(f"ssp_dim={d}, scales={np.round(scales, 2)}, "
          f"g_small={g_small}, g_large={g_large}")

    # gain schedule: hold / morph segments, PHASE_TIME x multiplier each
    segs = [(g_ones, g_ones, 1.0), (g_ones, g_small, 1.0),
            (g_small, g_small, 1.0), (g_small, g_large, 1.5),
            (g_large, g_large, 1.5)]
    n_hold = round(PHASE_TIME * args.fps)
    seg_frames = [round(m * n_hold) for _, _, m in segs]
    total = sum(seg_frames)
    traj = make_walk(rng, total, world, margin=1.2, start=center)

    gn = args.grid_n
    cell, grid = env_grid(world, gn)
    fig, ax, axr, axp, mesh, path_ln, dot = two_panel_fig(
        cell, gn, args.arrow_len, (0, world), phasors=True)
    quiv = hub_quiver(axr, arrow_display_uv(a_show, mags_max, args))
    base_rgb = matplotlib.colors.to_rgb(COLORS["arrow"])
    edge_rgb = matplotlib.colors.to_rgb(COLORS["arrow_edge"])
    # phasor dials for the 18 shown base rows; the needle angle is the
    # modulated phase g_s * (a_k . x): a zeroed scale parks at e^{i0} = 1.
    # Dial columns = the 3 hex-group vertices, rows sorted by scale.
    ph_idx = idx[np.argsort((idx // nv) % ns, kind="stable")]
    ph_rows = a_half[ph_idx]
    ph_s = (ph_idx // nv) % ns
    ph = phasor_dials(axp, len(ph_rows))

    def frames():
        f = 0
        for (gA, gB, _), nf in zip(segs, seg_frames):
            for i in range(nf):
                g = (1 - smoothstep(i / (nf - 1))) * gA \
                    + smoothstep(i / (nf - 1)) * gB
                x = traj[f]
                # gain-weighted similarity (normalized so the peak stays 1)
                w = g[row_s]
                pat = (np.cos((grid - x) @ a_half.T) * w).sum(-1) / w.sum()
                mesh.set_array(np.clip(pat, 0, 1))
                path_ln.set_data(traj[: f + 1, 0], traj[: f + 1, 1])
                dot.set_data([x[0]], [x[1]])
                alpha = MOD_ALPHA_MIN + (1 - MOD_ALPHA_MIN) * g[show_s]
                quiv.set_facecolor([base_rgb + (a,) for a in alpha])
                quiv.set_edgecolor([edge_rgb + (a,) for a in alpha])
                set_needles(ph, g[ph_s] * (ph_rows @ x),
                            MOD_ALPHA_MIN + (1 - MOD_ALPHA_MIN) * g[ph_s])
                yield frame_of(fig)
                f += 1

    write_outputs(frames(), total, out_base, args)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Animation 3: "learned" -- morph the kernel into the learned A(x) fractal
# ---------------------------------------------------------------------------


def anim_learned(args, out_base):
    """Left: phi(0) . phi(x') over the image domain [-1,1]^2, morphing from
    the standard HexSSP kernel to the fractal-image fit of
    kernel_image_fit.py by interpolating the phases A_hex -> A_hex + dA(x)
    per pixel. Right: the displayed rows of A gliding from the hex rows to
    the learned A(0) rows, all fully opaque."""
    import jax
    import jax.numpy as jnp
    from flax import serialization
    import kernel_image_fit as kif

    # space + model exactly as in the kernel_image_fit training run
    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=8, n_rotates=8,
                                  length_scale=0.1, rng=0)
    d = ssp_space.ssp_dim
    a_base = np.asarray(half_phase_matrix(ssp_space), np.float32)
    nf = len(a_base)

    model = kif.PhaseNet(hidden=(128, 128), n_free=nf, act="sin", w0=30.0)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, 2)))
    params = serialization.from_bytes(
        params, (FSPath(args.learned_dir) / "params.msgpack").read_bytes())

    gn = args.grid_n
    cell = np.linspace(-1, 1, gn + 1)
    cc = (cell[:-1] + cell[1:]) / 2
    X, Y = np.meshgrid(cc, cc)
    grid = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)

    print(f"ssp_dim={d}; evaluating learned dA on the {gn}x{gn} grid ...")
    # the SIREN's forward is arithmetic-sensitive: only the jitted graph
    # (as used in training/eval) reproduces the fitted function
    apply_j = jax.jit(model.apply)
    dA = np.asarray(apply_j(params, jnp.asarray(grid)))             # (n, nf, 2)
    theta0 = grid @ a_base.T                                        # hex phases
    theta1 = 30.0 * np.einsum("pkj,pj->pk", dA, grid)               # residual

    def sim_map(u):
        return (1 + 2 * np.cos(theta0 + u * theta1).sum(-1)) / d

    # displayed arrows: hex subgroup, 4 evenly spaced rotations of the 8
    nv, ns, nr = ssp_space.grid_basis_dim, ssp_space.n_scales, ssp_space.n_rotates
    s_pick = np.round(np.linspace(0, ns - 1, args.arrow_scales)).astype(int)
    r_pick = np.floor(np.arange(4) * nr / 4).astype(int)
    idx = np.array([r * ns * nv + s * nv + v
                    for r in r_pick for s in s_pick for v in range(nv)])
    dA0 = np.asarray(apply_j(params, jnp.zeros((1, 2))))[0]
    a_learn = a_base + 30.0 * dA0
    arr0 = np.vstack([a_base[idx], -a_base[idx]])
    arr1 = np.vstack([a_learn[idx], -a_learn[idx]])
    mag_ref = max(np.linalg.norm(arr0, axis=1).max(),
                  np.linalg.norm(arr1, axis=1).max())
    print(f"arrows shown={len(arr0)}, |a| hex max "
          f"{np.linalg.norm(arr0, axis=1).max():.1f} -> learned max "
          f"{np.linalg.norm(arr1, axis=1).max():.1f}")

    fig, ax, axr, _, mesh, path_ln, dot = two_panel_fig(
        cell, gn, args.arrow_len, (-1.0, 1.0))
    dot.set_data([0.0], [0.0])
    base_rgb = matplotlib.colors.to_rgb(COLORS["arrow"])
    edge_rgb = matplotlib.colors.to_rgb(COLORS["arrow_edge"])
    quiv = None

    def draw(u):
        nonlocal quiv
        mesh.set_array(np.clip(sim_map(u), 0, 1))
        rows = (1 - u) * arr0 + u * arr1
        mags = np.linalg.norm(rows, axis=1)
        disp = args.arrow_len * (mags / mag_ref) ** args.arrow_len_pow
        uv = rows / mags[:, None] * disp[:, None]
        if quiv is not None:
            quiv.remove()
        quiv = hub_quiver(axr, uv)
        quiv.set_facecolor([base_rgb + (1.0,)] * len(uv))
        quiv.set_edgecolor([edge_rgb + (1.0,)] * len(uv))
        return frame_of(fig)

    n_hold = round(PHASE_TIME * args.fps)
    n_morph = 2 * n_hold

    def frames():
        f0 = draw(0.0)
        for _ in range(n_hold):
            yield f0
        for i in range(n_morph):
            yield draw(smoothstep(i / (n_morph - 1)))
        f1 = draw(1.0)
        for _ in range(n_hold):
            yield f1

    write_outputs(frames(), 2 * n_hold + n_morph, out_base, args)
    plt.close(fig)


# ---------------------------------------------------------------------------

ANIMATIONS = {"walk": anim_walk, "modulate": anim_modulate,
              "learned": anim_learned}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("anim", nargs="?", default="walk", choices=ANIMATIONS)
    ap.add_argument("--world", type=float, default=10.0, help="env side (m)")
    ap.add_argument("--n-scales", type=int, default=5)
    ap.add_argument("--n-rotates", type=int, default=9,
                    help="multiple of --arrow-rotates keeps shown arrows evenly rotated")
    ap.add_argument("--length-scale", type=float, default=1.0)
    ap.add_argument("--arrow-len", type=float, default=1.1,
                    help="display length of the longest a_k arrow (m)")
    ap.add_argument("--arrow-len-pow", type=float, default=0.5,
                    help="display length exponent on |a_k|/max")
    ap.add_argument("--scale-min", type=float, default=np.pi / 9,
                    help="smallest hex scale (log-spaced up to pi)")
    ap.add_argument("--arrow-scales", type=int, default=2,
                    help="scales in the displayed hex arrow subgroup")
    ap.add_argument("--arrow-rotates", type=int, default=3,
                    help="rotations in the displayed hex arrow subgroup")
    ap.add_argument("--grid-n", type=int, default=220)
    ap.add_argument("--frames", type=int, default=360,
                    help="walk-phase frames (walk animation)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif-stride", type=int, default=2)
    ap.add_argument("--gif-scale", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--learned-dir", type=str, default="results/kernel_image_fit",
                    help="learned animation: dir with params.msgpack")
    ap.add_argument("--out", type=str, default="results/animations")
    args = ap.parse_args()

    out = FSPath(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ANIMATIONS[args.anim](args, out / f"ssp_{args.anim}")


if __name__ == "__main__":
    main()
