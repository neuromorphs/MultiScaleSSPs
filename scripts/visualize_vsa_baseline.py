#!/usr/bin/env python
"""Visualize how the VSA baseline's decoded class maps change with length scale.

Fits ``VSASpatialMemory`` on the same room, with the same random seed (so the
SSP phase matrix and class vectors are identical across runs), once per
length scale. Produces, in --out-dir:

    class_maps_grid.png   - grid of decoded per-label probability maps, one row
                             per label, one column per length scale (plus a
                             ground-truth column for reference)
    kl_vs_length_scale.png - KL(estimated || ground truth) per label as a
                             function of length scale
    summary.json           - the raw per-label, per-length-scale KL values

Usage:
    python scripts/visualize_vsa_baseline.py --out-dir /path/to/output
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multiscalessps.envs.room import make_default_room
from multiscalessps.models.baseline import VSASpatialMemory

DEFAULT_LENGTH_SCALES = [0.05, 0.1, 0.15, 0.2]


def fit_models(room, length_scales, ssp_dim, seed, normalize_by_class):
    """Fit one VSASpatialMemory per length scale, all sharing the same seed."""
    return {
        ls: VSASpatialMemory.from_room(
            room,
            ssp_dim=ssp_dim,
            length_scale=ls,
            normalize_by_class=normalize_by_class,
            rng=seed,
        )
        for ls in length_scales
    }


def save_class_maps_grid(room, models, length_scales, out_dir: Path, normalize_by_class):
    gt_maps = room.label_probability_maps()
    labels = sorted(gt_maps, key=lambda label: label)
    (xmin, xmax), (ymin, ymax) = room.bounds
    extent = (xmin, xmax, ymin, ymax)

    n_cols = len(length_scales) + 1
    fig, axes = plt.subplots(len(labels), n_cols, figsize=(3 * n_cols, 3 * len(labels)))

    col_titles = ["ground truth"] + [f"length_scale={ls}" for ls in length_scales]
    for row, label in enumerate(labels):
        row_maps = [gt_maps[label]] + [
            models[ls].class_probability_maps(room)[label] for ls in length_scales
        ]

        for col, (m, title) in enumerate(zip(row_maps, col_titles)):
            ax = axes[row, col]
            im = ax.imshow(
                m, extent=extent, origin="upper", cmap="viridis", vmin=0, vmax=m.max()
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(title, fontsize=10)
            if col == 0:
                ax.set_ylabel(label, fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    title_suffix = " (normalized per class)" if normalize_by_class else ""
    fig.suptitle(
        f"Decoded class probability maps vs. length scale{title_suffix}\n"
        "(each panel's color scale is normalized to its own max)"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "class_maps_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_kl_vs_length_scale_plot(room, models, length_scales, out_dir: Path):
    label_colors = {"room": "#9ecae1", "floor": "#6baed6", "wall": "#08306b", "donut": "#e6550d"}
    kl_by_ls = {ls: models[ls].evaluate_kl(room) for ls in length_scales}
    labels = sorted(next(iter(kl_by_ls.values())))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for label in labels:
        ax.plot(
            length_scales,
            [kl_by_ls[ls][label] for ls in length_scales],
            marker="o",
            color=label_colors.get(label),
            label=label,
        )
    ax.set_xlabel("length scale")
    ax.set_ylabel("KL(estimated || ground truth)")
    ax.set_title("VSA baseline accuracy vs. length scale")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "kl_vs_length_scale.png", dpi=150)
    plt.close(fig)
    return kl_by_ls


def save_summary(kl_by_ls, length_scales, ssp_dim, seed, normalize_by_class, out_dir: Path):
    summary = {
        "length_scales": length_scales,
        "ssp_dim": ssp_dim,
        "seed": seed,
        "normalize_by_class": normalize_by_class,
        "kl_by_length_scale": {str(ls): kl_by_ls[ls] for ls in length_scales},
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(".scratch/vsa_baseline"))
    parser.add_argument("--length-scales", type=float, nargs="+", default=DEFAULT_LENGTH_SCALES)
    parser.add_argument("--ssp-dim", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normalize-by-class",
        action="store_true",
        help="Normalize each class's summed records to unit norm before combining into "
        "memory, so classes with more points don't dominate the bundle.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    room = make_default_room()

    models = fit_models(
        room, args.length_scales, args.ssp_dim, args.seed, args.normalize_by_class
    )
    save_class_maps_grid(
        room, models, args.length_scales, args.out_dir, args.normalize_by_class
    )
    kl_by_ls = save_kl_vs_length_scale_plot(room, models, args.length_scales, args.out_dir)
    save_summary(
        kl_by_ls, args.length_scales, args.ssp_dim, args.seed, args.normalize_by_class, args.out_dir
    )

    print(f"Saved visualizations and data to {args.out_dir}")


if __name__ == "__main__":
    main()
