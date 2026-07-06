#!/usr/bin/env python
"""Visualize a RoomEnv: its layout, rasterized data, and sampling behavior.

Produces, in --out-dir:
    room_layout.png           - region/object map colored by derived scale
    dense_positions.csv       - every dense-sampled point with its label + scale
    room_grid_labels.npy      - raw base-resolution label grid (for reuse without recomputing)
    dense_counts.png          - bar charts of dense-sample counts by label/scale
    dense_positions_scatter.png - the actual dense_positions() point cloud, colored by label
                                  (shows density differences the fixed-resolution grid plot can't)
    sampled_points.png        - rejection-sampled points per scale, overlaid on the room
    sampled_points_labels_vs_scale.png - the same rejection-sampled points, shown once
                                  colored by label and once colored by derived scale,
                                  side by side (shows which labels collapse into which scale)
    label_probability_maps.png - per-label ground-truth spatial probability map (heatmap),
                                  titled with its KL divergence from a uniform distribution
    label_kl_divergence.png   - bar chart of the same per-label KL-from-uniform values
    kl_noise_robustness.png  - sanity check: KL(true || corrupted) vs. corruption strength,
                                  for two corruption types (uniform-noise mixing, Gaussian blur)
    map_diff_uniform_noise.png - per label: true map, noise-mixed map, and their difference
    map_diff_blur.png        - per label: true map, blurred map, and their difference
    summary.json              - counts, areas, KL divergences, and the label/scale policy in effect

Usage:
    python scripts/visualize_room.py --out-dir /path/to/output
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from multiscalessps.envs.room import DEFAULT_LENGTH_SCALES, make_default_room
from multiscalessps.metrics import blur_map, kl_divergence, kl_from_uniform, mix_uniform_noise


def save_layout_plot(room, out_dir: Path):
    ax = room.render()
    ax.figure.savefig(out_dir / "room_layout.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def save_dense_data(room, out_dir: Path):
    positions = room.dense_positions()

    with open(out_dir / "dense_positions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label", "scale"])
        for (x, y), label in positions:
            writer.writerow([x, y, label, room.scale_of(label)])

    np.save(out_dir / "room_grid_labels.npy", room.grid)
    return positions


def save_counts_plot(room, positions, out_dir: Path):
    label_counts = Counter(label for _, label in positions)
    scale_counts = Counter(room.scale_of(label) for _, label in positions)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(label_counts.keys(), label_counts.values(), color="#6baed6")
    axes[0].set_title("Dense sample counts by label")
    axes[0].set_ylabel("count")

    scale_order = [s for s in DEFAULT_LENGTH_SCALES if s in scale_counts]
    axes[1].bar(scale_order, [scale_counts[s] for s in scale_order], color="#08306b")
    axes[1].set_title("Dense sample counts by derived scale")

    fig.tight_layout()
    fig.savefig(out_dir / "dense_counts.png", dpi=150)
    plt.close(fig)
    return label_counts, scale_counts


def save_dense_scatter_plot(room, positions, out_dir: Path):
    """Scatter the actual dense_positions() point cloud, colored by label.

    Unlike room_layout.png (which is capped at the base grid_size
    resolution), this reflects any per-object density overrides directly.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"room": "#d9d9d9", "floor": "#c6dbef", "wall": "#08306b", "donut": "#e6550d"}

    by_label = {}
    for pos, label in positions:
        by_label.setdefault(label, []).append(pos)

    # draw coarser/background labels first so denser/finer ones stay visible on top
    plot_order = sorted(by_label, key=lambda label: DEFAULT_LENGTH_SCALES[room.scale_of(label)])
    for label in plot_order:
        xs, ys = zip(*by_label[label])
        ax.scatter(xs, ys, s=4, color=colors.get(label), label=f"{label} (n={len(xs)})", alpha=0.7)

    (xmin, xmax), (ymin, ymax) = room.bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=7, markerscale=2)
    ax.set_title("dense_positions() point cloud by label")
    fig.savefig(out_dir / "dense_positions_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_sampling_plot(room, out_dir: Path, n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    ax = room.render(show_objects=True)

    colors = {"coarse": "#31a354", "medium": "#756bb1", "fine": "#e6550d"}
    present_scales = set(room.scale_of(label) for label in room.label_to_scale)
    for scale in DEFAULT_LENGTH_SCALES:
        if scale not in present_scales:
            continue
        pts = [room.sample_position(scale=scale, rng=rng) for _ in range(n_samples)]
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=8, color=colors.get(scale, "black"), label=scale, alpha=0.7)

    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Rejection-sampled points by scale")
    ax.figure.savefig(out_dir / "sampled_points.png", dpi=150, bbox_inches="tight")
    plt.close(ax.figure)


def save_labels_vs_scale_plot(room, out_dir: Path, n_per_label: int, seed: int):
    """Sample evenly per label, then plot the same points colored by label vs. by scale.

    Sampling evenly per label (rather than per scale) makes it visually clear
    which distinct labels (e.g. "room" and "floor") get collapsed into the
    same derived scale (e.g. both "coarse").
    """
    rng = np.random.default_rng(seed)
    labels = list(room.label_to_scale.keys())
    samples = [
        (room.sample_position(label=label, rng=rng), label, room.scale_of(label))
        for label in labels
        for _ in range(n_per_label)
    ]

    label_colors = {"room": "#9ecae1", "floor": "#6baed6", "wall": "#08306b", "donut": "#e6550d"}
    scale_colors = {"coarse": "#31a354", "medium": "#756bb1", "fine": "#e6550d"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    panels = [
        (axes[0], "label", label_colors, "Sampled points by label"),
        (axes[1], "scale", scale_colors, "Same points by derived scale"),
    ]
    for ax, key_idx, color_map, title in panels:
        room.render(ax=ax, show_objects=True)
        by_key = {}
        for pos, label, scale in samples:
            key = label if key_idx == "label" else scale
            by_key.setdefault(key, []).append(pos)
        for key, pts in by_key.items():
            xs, ys = zip(*pts)
            ax.scatter(
                xs, ys, s=16, color=color_map.get(key, "black"),
                label=f"{key} (n={len(xs)})", edgecolor="black", linewidth=0.3,
            )
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_dir / "sampled_points_labels_vs_scale.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_probability_maps_plot(room, out_dir: Path):
    """Heatmap of each label's ground-truth probability map, titled with its KL from uniform."""
    label_maps = room.label_probability_maps()
    kl = kl_from_uniform(label_maps)
    labels = sorted(label_maps, key=lambda label: kl[label])

    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 4))
    axes = np.atleast_1d(axes)

    (xmin, xmax), (ymin, ymax) = room.bounds
    for ax, label in zip(axes, labels):
        im = ax.imshow(label_maps[label], extent=(xmin, xmax, ymin, ymax), origin="upper", cmap="viridis")
        ax.set_title(f"{label}\nKL-from-uniform = {kl[label]:.2f}")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_dir / "label_probability_maps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return kl


def save_kl_bar_chart(kl, out_dir: Path):
    labels = sorted(kl, key=lambda label: kl[label])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, [kl[label] for label in labels], color="#3182bd")
    ax.set_ylabel("KL(P_label || Uniform)")
    ax.set_title("Spatial concentration per label")
    fig.tight_layout()
    fig.savefig(out_dir / "label_kl_divergence.png", dpi=150)
    plt.close(fig)


def save_noise_robustness_plot(room, out_dir: Path):
    """Sanity-check the KL metric: self-KL should be ~0, and KL should rise
    monotonically as each label's map is corrupted, under two corruption
    types (uniform-noise mixing, Gaussian blur).
    """
    label_maps = room.label_probability_maps()
    self_kl = {label: kl_divergence(p, p) for label, p in label_maps.items()}

    noise_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    sigmas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    label_colors = {"room": "#9ecae1", "floor": "#6baed6", "wall": "#08306b", "donut": "#e6550d"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, p in label_maps.items():
        noise_kl = [kl_divergence(p, mix_uniform_noise(p, a)) for a in noise_levels]
        blur_kl = [kl_divergence(p, blur_map(p, s)) for s in sigmas]
        color = label_colors.get(label)
        axes[0].plot(noise_levels, noise_kl, marker="o", color=color, label=label)
        axes[1].plot(sigmas, blur_kl, marker="o", color=color, label=label)

    axes[0].set_xlabel("uniform noise level")
    axes[0].set_ylabel("KL(true || corrupted)")
    axes[0].set_title("Corruption: uniform noise mixing")
    axes[1].set_xlabel("blur sigma (grid cells)")
    axes[1].set_title("Corruption: Gaussian blur")
    for ax in axes:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "kl_noise_robustness.png", dpi=150)
    plt.close(fig)
    return self_kl


def save_map_diff_plot(room, out_dir: Path, corruption_name: str, corrupt_fn, param_label: str):
    """For each label: true map, corrupted map, and (corrupted - true).

    The diff panel is what the single KL number summarizes as a scalar: red
    = probability mass the corruption added where it shouldn't be, blue =
    mass it removed from where it should be. Each label gets its own
    symmetric color scale, since donut's per-cell probabilities are much
    larger than room's (fewer occupied cells sharing the same total mass).
    """
    label_maps = room.label_probability_maps()
    labels = list(label_maps.keys())
    (xmin, xmax), (ymin, ymax) = room.bounds
    extent = (xmin, xmax, ymin, ymax)

    fig, axes = plt.subplots(len(labels), 3, figsize=(9, 3 * len(labels)))
    col_titles = ["true", f"corrupted ({param_label})", "corrupted - true"]
    for row, label in enumerate(labels):
        true_p = label_maps[label]
        corrupted_p = corrupt_fn(true_p)
        diff = corrupted_p - true_p
        vmax = max(np.abs(diff).max(), 1e-12)

        ax_true, ax_corrupt, ax_diff = axes[row]
        ax_true.imshow(true_p, extent=extent, origin="upper", cmap="viridis")
        ax_corrupt.imshow(corrupted_p, extent=extent, origin="upper", cmap="viridis")
        im = ax_diff.imshow(diff, extent=extent, origin="upper", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        fig.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)

        ax_true.set_ylabel(label, fontsize=11)
        for ax, title in zip((ax_true, ax_corrupt, ax_diff), col_titles):
            if row == 0:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")

    fig.suptitle(f"Ground truth vs. {corruption_name}")
    fig.tight_layout()
    fig.savefig(out_dir / f"map_diff_{corruption_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_label_areas(room, positions):
    """Area per label, accounting for per-object density overrides.

    Points don't all represent the same area: a base grid cell covers
    ``cell_area``, but a point from an object's ``density`` override covers
    a smaller/larger patch of its own. Weighting each point by the area it
    actually represents keeps this correct regardless of sampling density.
    """
    H, W = room.grid_size
    (xmin, xmax), (ymin, ymax) = room.bounds
    cell_area = (xmax - xmin) / W * (ymax - ymin) / H

    point_area = {
        obj.name: (2 * obj.patch_radius / obj.density) ** 2
        for obj in room.objects
        if obj.density is not None
    }

    areas = {}
    for _, label in positions:
        areas[label] = areas.get(label, 0.0) + point_area.get(label, cell_area)
    return areas, cell_area


def save_summary(room, positions, label_counts, scale_counts, label_kl, self_kl, out_dir: Path):
    label_areas, cell_area = compute_label_areas(room, positions)

    summary = {
        "bounds": room.bounds,
        "grid_size": room.grid_size,
        "cell_area": cell_area,
        "label_to_scale": room.label_to_scale,
        "length_scales": DEFAULT_LENGTH_SCALES,
        "label_counts": dict(label_counts),
        "label_areas": label_areas,
        "label_kl_from_uniform": label_kl,
        "label_self_kl_sanity_check": self_kl,
        "scale_counts": dict(scale_counts),
        "objects": [
            {
                "name": o.name,
                "position": o.position,
                "patch_radius": o.patch_radius,
                "density": o.density,
            }
            for o in room.objects
        ],
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(".scratch/room"))
    parser.add_argument("--n-samples", type=int, default=300)
    parser.add_argument("--n-per-label", type=int, default=150)
    parser.add_argument("--diff-noise-level", type=float, default=0.5)
    parser.add_argument("--diff-sigma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    room = make_default_room()

    save_layout_plot(room, args.out_dir)
    positions = save_dense_data(room, args.out_dir)
    label_counts, scale_counts = save_counts_plot(room, positions, args.out_dir)
    save_dense_scatter_plot(room, positions, args.out_dir)
    save_sampling_plot(room, args.out_dir, args.n_samples, args.seed)
    save_labels_vs_scale_plot(room, args.out_dir, args.n_per_label, args.seed)
    label_kl = save_probability_maps_plot(room, args.out_dir)
    save_kl_bar_chart(label_kl, args.out_dir)
    self_kl = save_noise_robustness_plot(room, args.out_dir)
    save_map_diff_plot(
        room, args.out_dir, "uniform_noise",
        lambda p: mix_uniform_noise(p, args.diff_noise_level),
        f"noise_level={args.diff_noise_level}",
    )
    save_map_diff_plot(
        room, args.out_dir, "blur",
        lambda p: blur_map(p, args.diff_sigma),
        f"sigma={args.diff_sigma}",
    )
    save_summary(room, positions, label_counts, scale_counts, label_kl, self_kl, args.out_dir)

    print(f"Saved visualizations and data to {args.out_dir}")


if __name__ == "__main__":
    main()
