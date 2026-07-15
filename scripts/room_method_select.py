#!/usr/bin/env python
"""Per-room best-of over the three gain parameterizations of
scripts/room_id_scale_modulation.py:

    scale-vec -- free per-scale gains (L1 0.08 + coarse 0.06)
    kernel    -- Gaussian over scales, learned (mu, sigma) (coarse 0.03)
    window    -- sliding boxcar of four 1s, learned center (coarse 0.03)

All three are trained on the same rooms (same env, codebook, integral
encodings), each room's map is decoded with each method's gains, and per room
the method with the LOWEST decode error wins (fewest objects outside their
footprints first, then lowest median error). Every selected modulation is
materialized as a plain n_scales gain vector, so downstream encoding/decoding
is identical regardless of which parameterization produced it. Selection runs
independently for the codebook-ID config and the trained-ID config (rooms are
independent bundles, so mixing methods across rooms is exact, and for the
trained config the winning method's jointly-trained IDs come along with its
gains).

Outputs (results/indoor_env_best/): unbound-similarity overlays for both
configs, the selected gain vectors (bars colored by winning method), decode
figures with each room's init grid, and json/npz with the per-method decode
scores behind each selection.

    python scripts/room_method_select.py
"""

import argparse
import json
import sys
import time
from pathlib import Path as FSPath
from types import SimpleNamespace

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(FSPath(__file__).parent))
from vsagym.spaces import HexagonalSSPSpace, SPSpace  # noqa: E402
from shape_scale_modulation import make_scale_index  # noqa: E402
from indoor_env import make_env  # noqa: E402
from room_id_scale_modulation import (RoomData, scale_collapse_matrix,  # noqa: E402
                                      train, train_ids, snap_window, decode_room,
                                      eval_room, overlay_figure, decode_figure)

METHODS = {"scale-vec": "free", "kernel": "gaussian", "window": "window"}
METHOD_COLOR = {"scale-vec": "#2a78d6", "kernel": "#008300", "window": "#e87ba4"}


def method_args(gain_param, base):
    """Per-method training args with that method's reg defaults."""
    return SimpleNamespace(
        gain_param=gain_param, window_width=base.window_width,
        reg_w=0.08 if gain_param == "free" else 0.0,
        coarse_w=0.06 if gain_param == "free" else 0.03,
        lr=base.lr, steps=base.steps, id_steps=base.id_steps, n_draws=1)


def dec_score(dec):
    """Selection key: (n objects outside their footprint, median error)."""
    return (len(dec["hits"]) - int(np.sum(dec["hits"])), float(np.median(dec["errs"])))


def gains_selected_figure(rooms_data, g_cb, m_cb, g_tr, m_tr, scales, path):
    """Selected per-room gain vectors: bars colored by the winning method
    (codebook config); gray step = selection for the trained-ID config."""
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True, facecolor="white")
    by_rc = {(rd.room.row, rd.room.col): ri for ri, rd in enumerate(rooms_data)}
    x = np.arange(len(scales))
    for (r, c), ri in by_rc.items():
        rd = rooms_data[ri]
        ax = axs[1 - r, c]
        ax.bar(x, g_cb[ri], color=METHOD_COLOR[m_cb[ri]])
        ax.step(x, g_tr[ri], where="mid", color="0.3", lw=1.4)
        ax.set_title(f"{rd.room.theme} -- {m_cb[ri]} (+IDs: {m_tr[ri]})", fontsize=10)
        ax.set_xlabel("scale (coarse -> fine)")
        if c == 0:
            ax.set_ylabel("gain")
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=col, label=m)
               for m, col in METHOD_COLOR.items()]
    handles.append(plt.Line2D([], [], color="0.3", lw=1.4, label="selected +IDs profile"))
    axs[0, 0].legend(handles=handles, fontsize=8, frameon=False)
    fig.suptitle("per-room selected scale gains (lowest decode error wins)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def print_selection(label, rooms_data, dec_by_method, winners):
    print(f"\nper-room decode error by method ({label}); * = selected")
    print(f"{'room':<10}" + "".join(f" {m:>18}" for m in METHODS))
    for ri, rd in enumerate(rooms_data):
        row = f"{rd.room.theme:<10}"
        for m in METHODS:
            dec = dec_by_method[m][rd.room.name]
            mark = "*" if winners[ri] == m else " "
            row += (f"  {np.median(dec['errs']):>7.3f} "
                    f"{int(np.sum(dec['hits']))}/{len(dec['hits']):<3}{mark}")
        print(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-scales", type=int, default=10)
    ap.add_argument("--n-rotates", type=int, default=33)
    ap.add_argument("--scale-min", type=float, default=1.0)
    ap.add_argument("--scale-max", type=float, default=10.0)
    ap.add_argument("--ls", type=float, default=0.9)
    ap.add_argument("--integral-res", type=float, default=0.04)
    ap.add_argument("--grid-n", type=int, default=50)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--id-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--window-width", type=int, default=4)
    ap.add_argument("--env-seed", type=int, default=7)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", type=str, default="results/indoor_env_best/room_id_modulation.png")
    args = ap.parse_args()

    out_path = FSPath(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(args.env_seed)
    ssp_space = HexagonalSSPSpace(domain_dim=2, n_scales=args.n_scales,
                                  n_rotates=args.n_rotates, scale_min=args.scale_min,
                                  scale_max=args.scale_max, scale_sampling="log",
                                  length_scale=args.ls, rng=0)
    d = ssp_space.ssp_dim
    n_sc = ssp_space.n_scales
    scale_idx = make_scale_index(ssp_space)
    W = scale_collapse_matrix(ssp_space, scale_idx)

    rng = np.random.default_rng(args.seed)
    rooms_data, ids_by_room = [], {}
    for room in env.rooms:
        items = [it for it in env.items if it.room == room.name]
        ids = SPSpace(len(items), d, rng=rng).vectors
        ids_by_room[room.name] = ids
        ids_hat = np.fft.rfft(ids, axis=-1)[:, :(d + 1) // 2]
        rooms_data.append(RoomData(room, items, ssp_space, W, ids_hat, args.grid_n,
                                   lambda it: 0, rng, 1, integral_res=args.integral_res))
    print(f"ssp_dim={d}, {len(env.items)} objects, methods: {list(METHODS)}")

    # train all methods, decode every room with each
    gains_cb, gains_tr, ids_tr = {}, {}, {}
    dec_cb = {m: {} for m in METHODS}
    dec_tr = {m: {} for m in METHODS}
    t0 = time.time()
    for name, gp in METHODS.items():
        a = method_args(gp, args)
        print(f"[{name}] training per-room gains ...")
        g, _, raw = train(rooms_data, ssp_space, a)
        if gp == "window":
            g = snap_window(raw, n_sc, args.window_width)
        gains_cb[name] = g
        print(f"[{name}] jointly training gains + IDs ...")
        gj, idsj, _, rawj = train_ids(rooms_data, ssp_space, scale_idx, a)
        if gp == "window":
            gj = snap_window(rawj, n_sc, args.window_width)
        gains_tr[name], ids_tr[name] = gj, idsj
        for ri, rd in enumerate(rooms_data):
            dec_cb[name][rd.room.name] = decode_room(
                ssp_space, g[ri], scale_idx, rd, ids_by_room[rd.room.name])
            dec_tr[name][rd.room.name] = decode_room(
                ssp_space, gj[ri], scale_idx, rd, idsj[rd.room.name])
    print(f"trained + decoded all methods in {time.time() - t0:.0f}s")

    # per-room selection, independently for both configs
    win_cb, win_tr = [], []
    for rd in rooms_data:
        win_cb.append(min(METHODS, key=lambda m: dec_score(dec_cb[m][rd.room.name])))
        win_tr.append(min(METHODS, key=lambda m: dec_score(dec_tr[m][rd.room.name])))
    g_sel_cb = np.stack([gains_cb[m][ri] for ri, m in enumerate(win_cb)])
    g_sel_tr = np.stack([gains_tr[m][ri] for ri, m in enumerate(win_tr)])
    ids_sel_tr = {rd.room.name: ids_tr[m][rd.room.name]
                  for rd, m in zip(rooms_data, win_tr)}
    dec_sel_cb = {rd.room.name: dec_cb[m][rd.room.name]
                  for rd, m in zip(rooms_data, win_cb)}
    dec_sel_tr = {rd.room.name: dec_tr[m][rd.room.name]
                  for rd, m in zip(rooms_data, win_tr)}

    print_selection("codebook IDs + per-room gains", rooms_data, dec_cb, win_cb)
    print_selection("trained IDs + joint gains", rooms_data, dec_tr, win_tr)

    # cosine of the selected combos + selected decode summary
    print(f"\n{'room':<10} {'chosen':>10} {'cos':>7} {'grid':>6} {'med err':>8}"
          f"   {'chosen+IDs':>10} {'cos':>7} {'grid':>6} {'med err':>8}")
    sims_cb, sims_tr, cos_cb, cos_tr = [], [], {}, {}
    for ri, rd in enumerate(rooms_data):
        s_c, c_c = eval_room(ssp_space, g_sel_cb[ri], scale_idx, W, rd,
                             ids_by_room[rd.room.name])
        s_t, c_t = eval_room(ssp_space, g_sel_tr[ri], scale_idx, W, rd,
                             ids_sel_tr[rd.room.name])
        sims_cb.append(s_c); sims_tr.append(s_t)
        cos_cb[rd.room.name], cos_tr[rd.room.name] = c_c, c_t
        dc, dt = dec_sel_cb[rd.room.name], dec_sel_tr[rd.room.name]
        print(f"{rd.room.theme:<10} {win_cb[ri]:>10} {np.mean(c_c):>7.3f} "
              f"{dc['n_pts']:>4}^2 {np.median(dc['errs']):>8.3f}   "
              f"{win_tr[ri]:>10} {np.mean(c_t):>7.3f} "
              f"{dt['n_pts']:>4}^2 {np.median(dt['errs']):>8.3f}")
    all_e_cb = np.concatenate([d["errs"] for d in dec_sel_cb.values()])
    all_e_tr = np.concatenate([d["errs"] for d in dec_sel_tr.values()])
    all_h_cb = sum(int(np.sum(d["hits"])) for d in dec_sel_cb.values())
    all_h_tr = sum(int(np.sum(d["hits"])) for d in dec_sel_tr.values())
    print(f"{'ALL':<10} {'':>10} {np.concatenate(list(cos_cb.values())).mean():>7.3f} "
          f"{all_h_cb:>3}/{len(all_e_cb):<3} {np.median(all_e_cb):>7.3f}   "
          f"{'':>10} {np.concatenate(list(cos_tr.values())).mean():>7.3f} "
          f"{all_h_tr:>3}/{len(all_e_tr):<3} {np.median(all_e_tr):>7.3f}")

    overlay_figure(env, rooms_data, sims_cb, cos_cb,
                   "unbound object similarity maps, per-room best-of gains "
                   "(one color per object, alpha ~ similarity)",
                   out_path, args.grid_n)
    overlay_figure(env, rooms_data, sims_tr, cos_tr,
                   "unbound object similarity maps, per-room best-of gains + "
                   "trained IDs (one color per object, alpha ~ similarity)",
                   out_path.with_name(out_path.stem + "_trainedIDs.png"), args.grid_n)
    gains_selected_figure(rooms_data, g_sel_cb, win_cb, g_sel_tr, win_tr,
                          np.asarray(ssp_space.scales),
                          out_path.with_name(out_path.stem + "_gains.png"))
    decode_figure(env, rooms_data, dec_sel_cb,
                  "direct-optim decoded positions (x), per-room best-of gains, "
                  "codebook IDs; init grids from each room's kernel FWHM",
                  out_path.with_name(out_path.stem + "_decode.png"))
    decode_figure(env, rooms_data, dec_sel_tr,
                  "direct-optim decoded positions (x), per-room best-of gains + "
                  "trained IDs; init grids from each room's kernel FWHM",
                  out_path.with_name(out_path.stem + "_decode_trainedIDs.png"))

    save = dict(gains_selected=g_sel_cb, gains_selected_trained=g_sel_tr,
                methods=win_cb, methods_trained=win_tr,
                scales=np.asarray(ssp_space.scales),
                rooms=[rd.room.theme for rd in rooms_data])
    for m in METHODS:
        save[f"gains_{m}"] = gains_cb[m]
        save[f"gains_trained_{m}"] = gains_tr[m]
    for rd in rooms_data:
        save[f"ids_{rd.room.theme}"] = ids_sel_tr[rd.room.name]
    np.savez(out_path.with_name(out_path.stem + "_gains.npz"), **save)

    def dec_summary(dec):
        return {"l_eff": float(dec["l_eff"]), "n_pts_per_dim": int(dec["n_pts"]),
                "hits": np.asarray(dec["hits"]).tolist(),
                "errs": np.asarray(dec["errs"]).tolist(), "nfev": int(dec["nfev"])}

    metrics = {}
    for ri, rd in enumerate(rooms_data):
        metrics[rd.room.theme] = {
            "method": win_cb[ri], "method_trained_ids": win_tr[ri],
            "cos_selected": cos_cb[rd.room.name].tolist(),
            "cos_selected_trained_ids": cos_tr[rd.room.name].tolist(),
            "decode_by_method": {m: dec_summary(dec_cb[m][rd.room.name])
                                 for m in METHODS},
            "decode_by_method_trained_ids": {m: dec_summary(dec_tr[m][rd.room.name])
                                             for m in METHODS}}
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump({"args": vars(args), "metrics": metrics}, f, indent=2)
    print(f"wrote {out_path.with_suffix('.json')} and _gains.npz")


if __name__ == "__main__":
    main()
