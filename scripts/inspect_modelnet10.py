#!/usr/bin/env python
"""Load ModelNet10 and print shape/info about a single sample.

Usage:
    python scripts/inspect_modelnet10.py
    python scripts/inspect_modelnet10.py --root data/ModelNet10 --index 0 --num-points 2048
    python scripts/inspect_modelnet10.py --pair --category chair --num-points 256
"""

import argparse

import torch
from torch.utils.data import DataLoader

from multiscalessps.data import ModelNet10Dataset, ModelNet10PairDataset


def inspect_pair(args) -> None:
    cats = [args.category] if args.category else None
    dataset = ModelNet10PairDataset(
        root=args.root, split=args.split, categories=cats,
        num_full_points=args.num_full_points,
        num_input_points=args.num_points if args.num_points > 0 else 256,
    )
    print(f"Dataset: ModelNet10 pair — down-sampled vs full cloud ({args.split})")
    print(f"  num samples : {len(dataset)}")
    print(f"  categories  : {dataset.classes}")
    print()

    sample = dataset[args.index]
    pts, full = sample["points"], sample["full_points"]
    print(f"Sample at index {args.index}: {sample['class_name']}")
    print(f"  input (down-sampled) : shape={tuple(pts.shape)}, dtype={pts.dtype}")
    print(f"  full cloud (target)  : shape={tuple(full.shape)}, dtype={full.dtype}")
    print(f"  full min / max       : {full.min().item():.4f} / {full.max().item():.4f}")
    print(f"  path                 : {sample['path']}")

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    print()
    print("One DataLoader batch (batch_size=4):")
    print(f"  points      : {tuple(batch['points'].shape)}")
    print(f"  full_points : {tuple(batch['full_points'].shape)}")
    print(f"  labels      : {batch['label'].tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/ModelNet10",
                        help="Path to the extracted ModelNet10 directory.")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--index", type=int, default=0,
                        help="Index of the sample to inspect.")
    parser.add_argument("--num-points", type=int, default=1024,
                        help="Points sampled per mesh (use -1 for raw mesh).")
    parser.add_argument("--pair", action="store_true",
                        help="Inspect the down-sampled vs full-cloud pair dataset.")
    parser.add_argument("--category", default=None,
                        help="Restrict pair dataset to one category, e.g. 'chair'.")
    parser.add_argument("--num-full-points", type=int, default=2048,
                        help="Full/target cloud size (pair mode).")
    args = parser.parse_args()

    if args.pair:
        inspect_pair(args)
        return

    num_points = None if args.num_points < 0 else args.num_points
    dataset = ModelNet10Dataset(root=args.root, split=args.split,
                                num_points=num_points)

    print(f"Dataset: ModelNet10 ({args.split})")
    print(f"  num samples : {len(dataset)}")
    print(f"  num classes : {len(dataset.classes)}")
    print(f"  classes     : {dataset.classes}")
    print()

    sample = dataset[args.index]
    print(f"Sample at index {args.index}:")
    if num_points is None:
        v, faces = sample["vertices"], sample["faces"]
        print(f"  class       : {sample['class_name']} (label {sample['label']})")
        print(f"  vertices    : shape={tuple(v.shape)}, dtype={v.dtype}")
        print(f"  faces       : shape={tuple(faces.shape)}, dtype={faces.dtype}")
        print(f"  bbox min    : {v.min(dim=0).values.tolist()}")
        print(f"  bbox max    : {v.max(dim=0).values.tolist()}")
        print(f"  path        : {sample['path']}")
    else:
        points, label = sample
        print(f"  class       : {dataset.classes[label]} (label {label})")
        print(f"  points      : shape={tuple(points.shape)}, dtype={points.dtype}")
        print(f"  min / max   : {points.min().item():.4f} / {points.max().item():.4f}")
        print(f"  mean (xyz)  : {points.mean(dim=0).tolist()}")

    # Show that it batches through a DataLoader (only for fixed-size samples).
    if num_points is not None:
        loader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch_points, batch_labels = next(iter(loader))
        print()
        print("One DataLoader batch (batch_size=4):")
        print(f"  points : shape={tuple(batch_points.shape)}, dtype={batch_points.dtype}")
        print(f"  labels : {batch_labels.tolist()}")


if __name__ == "__main__":
    main()
