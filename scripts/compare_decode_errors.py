#!/usr/bin/env python
"""Compare per-room direct-optim decode errors across the three gain
parameterizations of scripts/room_id_scale_modulation.py:

    scale-vec -- free per-scale gains (results/indoor_env)
    kernel    -- Gaussian over scales, learned (mu, sigma) (results/indoor_env_kernel)
    window    -- sliding boxcar of four 1s, learned center (results/indoor_env_window)

Reads each run's room_id_modulation.json and draws a grouped dot plot
(median error per room, log x) for both decode configs. Hollow markers mark
rooms where not every object decoded inside its footprint.

    python scripts/compare_decode_errors.py --out results/decode_comparison.png
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {  # fixed categorical order + marker shape per version
    "scale-vec": ("results/indoor_env/room_id_modulation.json", "#2a78d6", "o"),
    "kernel": ("results/indoor_env_kernel/room_id_modulation.json", "#008300", "s"),
    "window": ("results/indoor_env_window/room_id_modulation.json", "#e87ba4", "^"),
}
CONFIGS = [("decode_learned", "codebook IDs + per-room gains"),
           ("decode_trained_ids", "trained IDs + joint gains")]
ROOM_ORDER = ["circle", "bedroom", "office", "living", "storage", "dining"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default="results/decode_comparison.png")
    args = ap.parse_args()

    data = {v: json.load(open(p))["metrics"] for v, (p, _, _) in RUNS.items()}
    rooms = [r for r in ROOM_ORDER if r in data["scale-vec"]]

    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, facecolor="white")
    for ax, (key, title) in zip(axs, CONFIGS):
        for yi, room in enumerate(rooms):
            ax.axhspan(yi - 0.42, yi + 0.42, color="0.96" if yi % 2 else "white",
                       zorder=0)
            for vi, (ver, (_, color, marker)) in enumerate(RUNS.items()):
                d = data[ver][room][key]
                med = float(np.median(d["errs"]))
                all_hit = sum(d["hits"]) == len(d["hits"])
                y = yi + (vi - 1) * 0.24
                ax.plot(med, y, marker, ms=8, color=color, zorder=3,
                        markerfacecolor=color if all_hit else "white",
                        markeredgecolor=color, markeredgewidth=1.6)
                if not all_hit:
                    ax.annotate(f"{sum(d['hits'])}/{len(d['hits'])}", (med, y),
                                textcoords="offset points", xytext=(8, -3),
                                fontsize=7.5, color="0.45", zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(7e-4, 1.2)
        ax.set_yticks(range(len(rooms)))
        ax.set_yticklabels(rooms)
        ax.set_ylim(len(rooms) - 0.5, -0.5)   # coarse rooms on top
        ax.set_xlabel("median decode error (m)")
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", color="0.88", lw=0.7, zorder=1)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    handles = [plt.Line2D([], [], marker=m, ls="none", ms=8, color=c, label=v)
               for v, (_, c, m) in RUNS.items()]
    handles.append(plt.Line2D([], [], marker="o", ls="none", ms=8,
                              markerfacecolor="white", markeredgecolor="0.4",
                              markeredgewidth=1.6, color="0.4",
                              label="hollow = missed objects"))
    axs[0].legend(handles=handles, loc="lower left", fontsize=8.5, frameon=False)
    fig.suptitle("per-room direct-optim decode error by gain parameterization",
                 fontsize=12)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
