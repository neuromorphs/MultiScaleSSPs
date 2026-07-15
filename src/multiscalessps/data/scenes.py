"""Loaders for labeled RGB point-cloud scenes: SceneNN (indoor) and
Semantic3D (outdoor).

Both loaders expose the same interface:

    scene = SceneNNScene("data/scenenn/005")          # or Semantic3DScene(...)
    scene.points        (N, 3) float32, meters
    scene.colors        (N, 3) float32 in [0, 1]
    scene.labels        (N,)   int32 per-point semantic/instance label id
    scene.label_names   {label_id: name} (may be partial)
    scene.subsample(voxel=0.05)   -> new arrays, one point per voxel

Download with scripts/download_semantic_datasets.py.

SceneNN scenes are reconstructed triangle meshes; we load the vertices (with
per-vertex color and NYU40-style class label). Semantic3D scenes are raw
terrestrial laser scans stored as ASCII 'x y z intensity r g b' rows with a
parallel .labels file (semantic-8 classes). Parsed scenes are cached as .npz
next to the raw files, so the first load is slow (minutes for Semantic3D) and
subsequent loads are instant.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SEMANTIC3D_CLASSES = {
    0: "unlabeled",
    1: "man-made terrain",
    2: "natural terrain",
    3: "high vegetation",
    4: "low vegetation",
    5: "buildings",
    6: "hard scape",
    7: "scanning artefacts",
    8: "cars",
}

# NYU40 class ids used by SceneNN per-vertex labels.
NYU40_CLASSES = {
    0: "unlabeled", 1: "wall", 2: "floor", 3: "cabinet", 4: "bed",
    5: "chair", 6: "sofa", 7: "table", 8: "door", 9: "window",
    10: "bookshelf", 11: "picture", 12: "counter", 13: "blinds",
    14: "desk", 15: "shelves", 16: "curtain", 17: "dresser", 18: "pillow",
    19: "mirror", 20: "floor mat", 21: "clothes", 22: "ceiling",
    23: "books", 24: "refrigerator", 25: "television", 26: "paper",
    27: "towel", 28: "shower curtain", 29: "box", 30: "whiteboard",
    31: "person", 32: "night stand", 33: "toilet", 34: "sink", 35: "lamp",
    36: "bathtub", 37: "bag", 38: "otherstructure", 39: "otherfurniture",
    40: "otherprop",
}


class LabeledScene:
    """Common container: points (N,3), colors (N,3) in [0,1], labels (N,)."""

    def __init__(self, points, colors, labels, label_names):
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.colors = np.ascontiguousarray(colors, dtype=np.float32)
        self.labels = np.ascontiguousarray(labels, dtype=np.int32)
        self.label_names = label_names

    def __len__(self):
        return len(self.points)

    def subsample(self, voxel):
        """One point per occupied voxel (first hit); returns new arrays."""
        idx = np.floor((self.points - self.points.min(axis=0))
                       / voxel).astype(np.int64)
        _, first = np.unique(idx, axis=0, return_index=True)
        return (self.points[first], self.colors[first], self.labels[first])

    def class_counts(self):
        ids, counts = np.unique(self.labels, return_counts=True)
        return {self.label_names.get(int(i), f"label {i}"): int(c)
                for i, c in zip(ids, counts)}


class SceneNNScene(LabeledScene):
    """One SceneNN scene from data/scenenn/<id>/ (mesh vertices).

    The PLY per-vertex ``label`` is an INSTANCE id; names come from the
    scene's XML annotations (<label id=... text="bed" .../>), so
    ``label_names`` maps instance id -> text (empty texts fall back to
    "instance <id>"). The raw XML entries are kept in ``.instances``.
    """

    def __init__(self, scene_dir: str | Path):
        scene_dir = Path(scene_dir)
        sid = scene_dir.name
        cache = scene_dir / f"{sid}_cache.npz"
        if cache.exists():
            z = np.load(cache)
            pts, rgb, lab = z["points"], z["colors"], z["labels"]
        else:
            pts, rgb, lab = self._parse_ply(scene_dir / f"{sid}.ply")
            np.savez_compressed(cache, points=pts, colors=rgb, labels=lab)
        self.instances = self._parse_xml(scene_dir / f"{sid}.xml")
        names = {int(i["id"]): (i.get("text") or f"instance {i['id']}")
                 for i in self.instances if "id" in i}
        names.setdefault(0, "unlabeled")
        super().__init__(pts, rgb, lab, names)

    @staticmethod
    def _parse_ply(path):
        from plyfile import PlyData
        ply = PlyData.read(path)
        v = ply["vertex"].data
        names = v.dtype.names
        pts = np.column_stack([v["x"], v["y"], v["z"]])
        rgb = (np.column_stack([v["red"], v["green"], v["blue"]])
               .astype(np.float32) / 255.0
               if "red" in names else np.zeros((len(pts), 3), np.float32))
        label_field = next((n for n in ("label", "nyu_class", "class")
                            if n in names), None)
        lab = (v[label_field].astype(np.int32) if label_field
               else np.zeros(len(pts), np.int32))
        return pts, rgb, lab

    @staticmethod
    def _parse_xml(path):
        if not path.exists():
            return []
        out = []
        for el in ET.parse(path).getroot().iter("label"):
            out.append(dict(el.attrib))
        return out


class Semantic3DScene(LabeledScene):
    """One Semantic3D training scene from data/semantic3d/<scene>/.

    Raw files are ASCII 'x y z intensity r g b' plus a parallel .labels file
    (semantic-8 class per row, 0 = unlabeled). Parsing ~20-30M rows takes a
    few minutes once; the result is cached as .npz.
    """

    def __init__(self, scene_dir: str | Path):
        scene_dir = Path(scene_dir)
        scene = scene_dir.name
        cache = scene_dir / f"{scene}_cache.npz"
        if cache.exists():
            z = np.load(cache)
            pts, rgb, lab, inten = (z["points"], z["colors"], z["labels"],
                                    z["intensity"])
        else:
            txt = scene_dir / f"{scene}_xyz_intensity_rgb.txt"
            print(f"parsing {txt.name} (first load, takes a few minutes)...")
            raw = np.fromfile(txt, sep=" ", dtype=np.float32).reshape(-1, 7)
            pts, inten = raw[:, :3], raw[:, 3]
            rgb = raw[:, 4:7] / 255.0
            lab = np.fromfile(scene_dir / f"{scene}_xyz_intensity_rgb.labels",
                              sep=" ", dtype=np.int32)
            assert len(lab) == len(pts), (len(lab), len(pts))
            np.savez_compressed(cache, points=pts, colors=rgb, labels=lab,
                                intensity=inten)
        super().__init__(pts, rgb, lab, SEMANTIC3D_CLASSES)
        self.intensity = inten
