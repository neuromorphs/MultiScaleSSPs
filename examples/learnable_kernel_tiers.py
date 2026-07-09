#!/usr/bin/env python3
"""Learnable-kernel VSA encoding of the hierarchical building, three tiers.

Self-contained: includes the BuildingEnv ground truth (copied from
multiscalessps.envs.building), metrics, models, training, evaluation, and
figures. No external project imports -- just numpy / torch / matplotlib.

What's new vs. the hierarchical_encoding notebooks
---------------------------------------------------
1. **Learnable kernels.** The discrete 3-scale mixture (`scale_logits` over
   hand-picked lengthscales) is replaced by a continuous, per-class
   log-lengthscale: each class c owns `log_ls[c]`, and positions bound to c
   are encoded at that class's bandwidth: phase = (x / ls_c) @ axis_phase.
   Gradients flow through both memory construction and query scoring, so
   the kernel bandwidth itself is learned per class -- `floor` can go
   coarse while `tv_stand` goes fine, which one shared mixture could never
   express.
   Each class also owns a learnable log-gain on its atoms' contribution to
   the memory -- the differentiable generalization of the notebooks'
   per-class point capping. `floor` contributes ~25x more raw atoms than
   `stove`; a learned gain lets the optimizer rebalance that energy instead
   of a hand-picked cap.
2. **Temperature learned in the loss** (the fix the notebooks discovered):
   raw FHRR correlations are O(1/sqrt(dim)), so an untempered cross-entropy
   gives the kernel parameters almost no gradient. Each head owns a
   learnable log-temperature trained jointly. A cheap post-hoc grid
   calibration on the validation split is still applied to the *final*
   full memory before reporting probabilities.
3. **Tier 3 in two flavors.**
   - *flat*: one memory = sum(pos (*) room) + sum(pos (*) furniture) --
     the notebooks' design, now with learnable kernels.
   - *hierarchical*: furniture atoms are additionally bound with their
     room vector, memory = sum(pos (*) room) + sum(pos (*) room (*)
     furniture). Querying furniture unbinds with the room decoded from the
     same memory, so `bed`-in-`kitchen` is near-orthogonal by construction
     instead of merely down-weighted post-hoc.
4. **One shared evaluation.** All furniture-task methods (Tier 1->2
   cascade, Tier 3 flat, flat + room prior, hierarchical) are scored on the
   *same* validation points and the same non-wall grid, with both mean
   per-class recall and plain (query-weighted) accuracy -- mean recall
   alone hides a collapsed `floor` class.

Run:
    python learnable_kernel_tiers.py                 # full run (~5-10 min CPU)
    python learnable_kernel_tiers.py --quick         # small smoke test
Outputs: printed summary + figures and results.json in --outdir.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================================================================
# 1. Ground-truth environment (verbatim behavior from envs/building.py)
# ======================================================================

Bounds = Tuple[Tuple[float, float], Tuple[float, float]]


@dataclass
class Room:
    name: str
    bounds: Dict[str, float]  # {"xmin", "xmax", "ymin", "ymax"}

    def contains(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        b = self.bounds
        return (x >= b["xmin"]) & (x <= b["xmax"]) & (y >= b["ymin"]) & (y <= b["ymax"])


@dataclass
class Furniture:
    name: str
    room: str
    bounds: Dict[str, float]

    def contains(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        b = self.bounds
        return (x >= b["xmin"]) & (x <= b["xmax"]) & (y >= b["ymin"]) & (y <= b["ymax"])


class BuildingEnv:
    def __init__(
        self,
        bounds: Bounds = ((-1.0, 1.0), (-1.0, 1.0)),
        grid_size: Tuple[int, int] = (128, 128),
        rooms: Optional[List[Room]] = None,
        furniture: Optional[List[Furniture]] = None,
    ):
        self.bounds = bounds
        self.grid_size = grid_size
        self.rooms = list(rooms) if rooms else []
        self.furniture = list(furniture) if furniture else []
        self.room_names = [r.name for r in self.rooms]
        self.furniture_names = [f.name for f in self.furniture]
        self._room_grid = None
        self._furniture_grid = None

    @property
    def room_grid(self) -> np.ndarray:
        if self._room_grid is None:
            self._rasterize()
        return self._room_grid

    @property
    def furniture_grid(self) -> np.ndarray:
        if self._furniture_grid is None:
            self._rasterize()
        return self._furniture_grid

    def cell_to_coord(self, i: int, j: int) -> Tuple[float, float]:
        (xmin, xmax), (ymin, ymax) = self.bounds
        H, W = self.grid_size
        x = xmin + (j + 0.5) / W * (xmax - xmin)
        y = ymax - (i + 0.5) / H * (ymax - ymin)
        return x, y

    def dense_positions(self):
        H, W = self.grid_size
        room_grid = self.room_grid
        furniture_grid = self.furniture_grid
        positions = []
        for i in range(H):
            for j in range(W):
                furniture = furniture_grid[i, j] or None
                positions.append(
                    (self.cell_to_coord(i, j), {"room": room_grid[i, j], "furniture": furniture})
                )
        return positions

    def room_probability_maps(self) -> Dict[str, np.ndarray]:
        grid = self.room_grid
        maps = {}
        for name in np.unique(grid):
            mask = grid == name
            maps[str(name)] = mask.astype(float) / mask.sum()
        return maps

    def furniture_probability_maps(self) -> Dict[str, np.ndarray]:
        grid = self.furniture_grid
        maps = {}
        for name in np.unique(grid):
            if name == "":
                continue
            mask = grid == name
            maps[str(name)] = mask.astype(float) / mask.sum()
        return maps

    def _rasterize(self):
        H, W = self.grid_size
        (xmin, xmax), (ymin, ymax) = self.bounds
        xs = xmin + (np.arange(W) + 0.5) / W * (xmax - xmin)
        ys = ymax - (np.arange(H) + 0.5) / H * (ymax - ymin)
        xx, yy = np.meshgrid(xs, ys)

        room_grid = np.full((H, W), "wall", dtype=object)
        for room in self.rooms:
            room_grid[room.contains(xx, yy)] = room.name

        furniture_grid = np.full((H, W), "", dtype=object)
        for item in self.furniture:
            furniture_grid[item.contains(xx, yy)] = item.name

        self._room_grid = room_grid
        self._furniture_grid = furniture_grid

    def render(self, ax=None, show_furniture: bool = True, show_labels: bool = True):
        from matplotlib.patches import Rectangle

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        room_palette = ["#c6dbef", "#fdd0a2", "#c7e9c0", "#dadaeb", "#fbb4b9", "#ffffb3"]
        room_colors = {name: room_palette[i % len(room_palette)] for i, name in enumerate(self.room_names)}
        (xmin, xmax), (ymin, ymax) = self.bounds
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor="#8a8a8a", zorder=0))
        for room in self.rooms:
            b = room.bounds
            ax.add_patch(
                Rectangle(
                    (b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                    facecolor=room_colors.get(room.name, "#eeeeee"), edgecolor="none", zorder=1,
                )
            )
            if show_labels:
                cx = (b["xmin"] + b["xmax"]) / 2
                ax.text(cx, b["ymax"] - 0.025, room.name, ha="center", va="top",
                        fontsize=9, style="italic", color="#444444", zorder=4)
        if show_furniture:
            for item in self.furniture:
                b = item.bounds
                ax.add_patch(
                    Rectangle(
                        (b["xmin"], b["ymin"]), b["xmax"] - b["xmin"], b["ymax"] - b["ymin"],
                        facecolor="#6b3e26", edgecolor="black", linewidth=0.8, zorder=2,
                    )
                )
                if show_labels:
                    ax.text((b["xmin"] + b["xmax"]) / 2, (b["ymin"] + b["ymax"]) / 2, item.name,
                            ha="center", va="center", fontsize=7, color="white", zorder=3)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_title("Building: rooms and furniture")
        return ax


def make_default_building() -> BuildingEnv:
    rooms = [
        Room("living_room", {"xmin": -0.94, "xmax": -0.03, "ymin": 0.03, "ymax": 0.94}),
        Room("bedroom", {"xmin": 0.03, "xmax": 0.94, "ymin": 0.03, "ymax": 0.94}),
        Room("kitchen", {"xmin": -0.94, "xmax": -0.03, "ymin": -0.94, "ymax": -0.03}),
        Room("bathroom", {"xmin": 0.03, "xmax": 0.94, "ymin": -0.94, "ymax": -0.03}),
    ]
    furniture = [
        Furniture("sofa", "living_room", {"xmin": -0.9, "xmax": -0.5, "ymin": 0.1, "ymax": 0.35}),
        Furniture("tv_stand", "living_room", {"xmin": -0.5, "xmax": -0.2, "ymin": 0.68, "ymax": 0.82}),
        Furniture("bed", "bedroom", {"xmin": 0.15, "xmax": 0.65, "ymin": 0.5, "ymax": 0.82}),
        Furniture("wardrobe", "bedroom", {"xmin": 0.75, "xmax": 0.93, "ymin": 0.1, "ymax": 0.5}),
        Furniture("table", "kitchen", {"xmin": -0.7, "xmax": -0.35, "ymin": -0.65, "ymax": -0.35}),
        Furniture("stove", "kitchen", {"xmin": -0.93, "xmax": -0.75, "ymin": -0.93, "ymax": -0.75}),
        Furniture("sink", "bathroom", {"xmin": 0.1, "xmax": 0.3, "ymin": -0.3, "ymax": -0.1}),
        Furniture("tub", "bathroom", {"xmin": 0.5, "xmax": 0.9, "ymin": -0.9, "ymax": -0.55}),
    ]
    return BuildingEnv(rooms=rooms, furniture=furniture)


# ======================================================================
# 2. Metrics / small utilities
# ======================================================================

EPS = 1e-12


def kl_divergence(p, q, eps=EPS) -> float:
    """KL(p || q) for two same-shape probability maps."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))


def make_split(labels_idx: np.ndarray, num_classes: int, val_frac=0.2, seed=0):
    """Stratified train/val split of indices into labels_idx."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx = [], []
    for c in range(num_classes):
        cp = np.flatnonzero(labels_idx == c)
        rng.shuffle(cp)
        n_val = max(1, int(round(val_frac * len(cp))))
        val_idx.append(cp[:n_val])
        train_idx.append(cp[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


def balanced_class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    return counts.sum() / (num_classes * counts.clamp(min=1.0))


def per_class_recall(pred: np.ndarray, labels: np.ndarray, names: List[str]) -> Dict[str, float]:
    return {
        n: float((pred[labels == i] == i).mean()) if np.any(labels == i) else float("nan")
        for i, n in enumerate(names)
    }


def softmax_np(scores: np.ndarray, temperature: float) -> np.ndarray:
    z = scores / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def calibrate_temperature(scores_val: np.ndarray, labels_val: np.ndarray,
                          class_weights: np.ndarray, temperatures=None) -> float:
    """Grid-search softmax temperature minimizing class-weighted val NLL on
    raw scores (used post-hoc on the final full memory)."""
    if temperatures is None:
        temperatures = np.logspace(-4, 0.5, 60)
    best_t, best_nll = temperatures[0], np.inf
    w = class_weights[labels_val]
    for t in temperatures:
        p = softmax_np(scores_val, t)
        logp = np.log(p[np.arange(len(labels_val)), labels_val] + EPS)
        nll = -(w * logp).sum() / w.sum()
        if nll < best_nll:
            best_t, best_nll = t, nll
    return float(best_t)


# ======================================================================
# 3. Models: FHRR maps with per-class learnable lengthscales
# ======================================================================


def _rand_phase(shape, generator):
    return 2 * torch.pi * torch.rand(*shape, generator=generator) - torch.pi


class PerClassKernelMap(nn.Module):
    """FHRR spatial memory with a continuous, learnable lengthscale per class.

    Encoding of a position x for class c: exp(1j * (x @ axis_phase) / ls_c),
    bound (elementwise complex product) with the class vector. Because the
    phase is linear in 1/ls_c, gradients w.r.t. log_ls flow through both
    memory construction and query scoring -- the kernel bandwidth is learned,
    not selected from a preset menu.
    """

    def __init__(self, dim, spatial_dim, num_classes, init_lengthscale=0.2,
                 init_temp=0.01, seed=0, device="cpu"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.dim = dim
        self.num_classes = num_classes
        self.register_buffer("axis_phase", _rand_phase((spatial_dim, dim), g).to(device))
        self.register_buffer("class_vecs", torch.exp(1j * _rand_phase((num_classes, dim), g)).to(device))
        self.log_ls = nn.Parameter(torch.full((num_classes,), math.log(init_lengthscale), device=device))
        self.log_gain = nn.Parameter(torch.zeros(num_classes, device=device))
        self.log_temp = nn.Parameter(torch.tensor(math.log(init_temp), device=device))

    @property
    def lengthscales(self) -> torch.Tensor:
        return torch.exp(self.log_ls)

    @property
    def gains(self) -> torch.Tensor:
        return torch.exp(self.log_gain)

    @property
    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temp)

    def build_memory(self, points: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        base = points @ self.axis_phase                          # [N, D] real
        pos = torch.exp(1j * (base / self.lengthscales[labels][:, None]))
        atoms = pos * self.class_vecs[labels]                    # [N, D] complex
        atoms = atoms * self.gains[labels][:, None]              # learned energy rebalance
        memory = atoms.sum(dim=0)
        return memory / torch.linalg.norm(memory)

    def score(self, memory: torch.Tensor, query: torch.Tensor, chunk=2048) -> torch.Tensor:
        """Raw correlation scores [N, C]: Re(<probe_c(x), memory>) / dim."""
        mc = memory[None, :] * torch.conj(self.class_vecs)       # [C, D]
        outs = []
        for i in range(0, query.shape[0], chunk):
            q = query[i:i + chunk]
            base = q @ self.axis_phase                           # [n, D]
            phase = base[:, None, :] / self.lengthscales[None, :, None]  # [n, C, D]
            pos_conj = torch.exp(-1j * phase)
            sims = torch.einsum("ncd,cd->nc", pos_conj, mc).real / self.dim
            outs.append(sims)
        return torch.cat(outs, dim=0)

    @torch.no_grad()
    def predict_proba(self, memory, query, temperature: float) -> np.ndarray:
        scores = self.score(memory, query).cpu().numpy()
        return softmax_np(scores, temperature)


class HouseMap(nn.Module):
    """One whole-building memory with two heads (room / furniture), each with
    per-class learnable lengthscales and its own learnable temperature.

    hierarchical=False (flat):  memory = sum pos(*)room + sum pos(*)furn
    hierarchical=True:          memory = sum pos(*)room + sum pos(*)room(*)furn
    In the hierarchical variant, furniture queries must supply a room binding
    (decoded from the same memory at eval time), which makes cross-room
    furniture confusion near-orthogonal by construction.
    """

    def __init__(self, dim, spatial_dim, num_rooms, num_furn,
                 init_ls_room=0.4, init_ls_furn=0.15, init_temp=0.01,
                 hierarchical=False, seed=0, device="cpu"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.dim = dim
        self.hierarchical = hierarchical
        self.register_buffer("axis_room", _rand_phase((spatial_dim, dim), g).to(device))
        self.register_buffer("axis_furn", _rand_phase((spatial_dim, dim), g).to(device))
        self.register_buffer("room_vecs", torch.exp(1j * _rand_phase((num_rooms, dim), g)).to(device))
        self.register_buffer("furn_vecs", torch.exp(1j * _rand_phase((num_furn, dim), g)).to(device))
        self.log_ls_room = nn.Parameter(torch.full((num_rooms,), math.log(init_ls_room), device=device))
        self.log_ls_furn = nn.Parameter(torch.full((num_furn,), math.log(init_ls_furn), device=device))
        self.log_gain_room = nn.Parameter(torch.zeros(num_rooms, device=device))
        self.log_gain_furn = nn.Parameter(torch.zeros(num_furn, device=device))
        self.log_temp_room = nn.Parameter(torch.tensor(math.log(init_temp), device=device))
        self.log_temp_furn = nn.Parameter(torch.tensor(math.log(init_temp), device=device))

    @property
    def ls_room(self):
        return torch.exp(self.log_ls_room)

    @property
    def ls_furn(self):
        return torch.exp(self.log_ls_furn)

    @property
    def temp_room(self):
        return torch.exp(self.log_temp_room)

    @property
    def temp_furn(self):
        return torch.exp(self.log_temp_furn)

    def build_memory(self, room_pts, room_lbls, furn_pts, furn_lbls, furn_room_lbls=None):
        base_r = room_pts @ self.axis_room
        pos_r = torch.exp(1j * (base_r / self.ls_room[room_lbls][:, None]))
        atoms_r = pos_r * self.room_vecs[room_lbls]
        atoms_r = atoms_r * torch.exp(self.log_gain_room)[room_lbls][:, None]
        rec_r = atoms_r.sum(dim=0)

        base_f = furn_pts @ self.axis_furn
        pos_f = torch.exp(1j * (base_f / self.ls_furn[furn_lbls][:, None]))
        atoms_f = pos_f * self.furn_vecs[furn_lbls]
        atoms_f = atoms_f * torch.exp(self.log_gain_furn)[furn_lbls][:, None]
        if self.hierarchical:
            assert furn_room_lbls is not None, "hierarchical memory needs the room of each furniture point"
            atoms_f = atoms_f * self.room_vecs[furn_room_lbls]
        rec_f = atoms_f.sum(dim=0)

        memory = rec_r + rec_f
        return memory / torch.linalg.norm(memory)

    def score_rooms(self, memory, query, chunk=2048):
        mc = memory[None, :] * torch.conj(self.room_vecs)
        outs = []
        for i in range(0, query.shape[0], chunk):
            q = query[i:i + chunk]
            base = q @ self.axis_room
            pos_conj = torch.exp(-1j * (base[:, None, :] / self.ls_room[None, :, None]))
            outs.append(torch.einsum("ncd,cd->nc", pos_conj, mc).real / self.dim)
        return torch.cat(outs, dim=0)

    def score_furniture(self, memory, query, room_idx=None, chunk=2048):
        """room_idx: [N] long, required iff hierarchical (the room to unbind)."""
        if self.hierarchical and room_idx is None:
            raise ValueError("hierarchical HouseMap requires room_idx to score furniture")
        outs = []
        fc = torch.conj(self.furn_vecs)                          # [C, D]
        for i in range(0, query.shape[0], chunk):
            q = query[i:i + chunk]
            base = q @ self.axis_furn
            pos_conj = torch.exp(-1j * (base[:, None, :] / self.ls_furn[None, :, None]))
            if self.hierarchical:
                t = torch.conj(self.room_vecs[room_idx[i:i + chunk]]) * memory[None, :]  # [n, D]
                sims = torch.einsum("ncd,cd,nd->nc", pos_conj, fc, t).real / self.dim
            else:
                mc = memory[None, :] * fc
                sims = torch.einsum("ncd,cd->nc", pos_conj, mc).real / self.dim
            outs.append(sims)
        return torch.cat(outs, dim=0)


# ======================================================================
# 4. Training
# ======================================================================


def train_single_head(model: PerClassKernelMap, pts, lbls, class_weights,
                      epochs, lr, score_batch, mem_batch, seed, tag=""):
    """Joint training of per-class log-lengthscales + log-temperature.
    Each epoch: memory from a random subset (stochastic superposition),
    scores on a random minibatch, weighted CE at the learned temperature."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = pts.shape[0]
    hist = {"loss": [], "ls": [], "temp": []}
    for _ in range(epochs):
        opt.zero_grad()
        mi = torch.randperm(n, generator=g)[:mem_batch] if mem_batch < n else torch.arange(n)
        memory = model.build_memory(pts[mi], lbls[mi])
        si = torch.randperm(n, generator=g)[:score_batch]
        scores = model.score(memory, pts[si]) / model.temperature
        loss = F.cross_entropy(scores, lbls[si], weight=class_weights)
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        hist["ls"].append(model.lengthscales.detach().cpu().numpy().copy())
        hist["temp"].append(model.temperature.item())
    with torch.no_grad():
        memory = model.build_memory(pts, lbls)  # final memory from ALL training points
    hist["ls"] = np.array(hist["ls"])
    return memory, hist


def train_house(model: HouseMap, room_pts, room_lbls, w_room,
                furn_pts, furn_lbls, furn_room_lbls, w_furn,
                epochs, lr, score_batch, mem_batch, seed, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n_r, n_f = room_pts.shape[0], furn_pts.shape[0]
    hist = {"loss": [], "room_loss": [], "furn_loss": [],
            "ls_room": [], "ls_furn": [], "temp_room": [], "temp_furn": []}
    for _ in range(epochs):
        opt.zero_grad()
        mi_r = torch.randperm(n_r, generator=g)[:mem_batch] if mem_batch < n_r else torch.arange(n_r)
        mi_f = torch.randperm(n_f, generator=g)[:mem_batch] if mem_batch < n_f else torch.arange(n_f)
        memory = model.build_memory(
            room_pts[mi_r], room_lbls[mi_r], furn_pts[mi_f], furn_lbls[mi_f],
            furn_room_lbls[mi_f] if model.hierarchical else None,
        )
        si_r = torch.randperm(n_r, generator=g)[:score_batch]
        si_f = torch.randperm(n_f, generator=g)[:score_batch]
        room_scores = model.score_rooms(memory, room_pts[si_r]) / model.temp_room
        furn_scores = model.score_furniture(
            memory, furn_pts[si_f],
            room_idx=furn_room_lbls[si_f] if model.hierarchical else None,
        ) / model.temp_furn
        room_loss = F.cross_entropy(room_scores, room_lbls[si_r], weight=w_room)
        furn_loss = F.cross_entropy(furn_scores, furn_lbls[si_f], weight=w_furn)
        loss = room_loss + furn_loss
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        hist["room_loss"].append(room_loss.item())
        hist["furn_loss"].append(furn_loss.item())
        hist["ls_room"].append(model.ls_room.detach().cpu().numpy().copy())
        hist["ls_furn"].append(model.ls_furn.detach().cpu().numpy().copy())
        hist["temp_room"].append(model.temp_room.item())
        hist["temp_furn"].append(model.temp_furn.item())
    with torch.no_grad():
        memory = model.build_memory(room_pts, room_lbls, furn_pts, furn_lbls,
                                    furn_room_lbls if model.hierarchical else None)
    hist["ls_room"] = np.array(hist["ls_room"])
    hist["ls_furn"] = np.array(hist["ls_furn"])
    return memory, hist


# ======================================================================
# 5. Main experiment
# ======================================================================


def prepare_data(env: BuildingEnv, seed: int, device: str):
    positions = env.dense_positions()
    xy_all = np.array([p for p, _ in positions], dtype=np.float32)
    room_labels_all = np.array([str(lbl["room"]) for _, lbl in positions])
    furn_labels_all = np.array(
        ["" if lbl["furniture"] is None else str(lbl["furniture"]) for _, lbl in positions]
    )

    room_names = sorted(set(room_labels_all))                    # 4 rooms + 'wall'
    room_to_idx = {n: i for i, n in enumerate(room_names)}
    room_idx_all = np.array([room_to_idx[r] for r in room_labels_all])

    non_wall = room_labels_all != "wall"
    furn_or_floor = np.where(furn_labels_all[non_wall] == "", "floor", furn_labels_all[non_wall])
    furn_names = ["floor"] + sorted(n for n in set(furn_or_floor) if n != "floor")
    furn_to_idx = {n: i for i, n in enumerate(furn_names)}
    furn_idx_nw = np.array([furn_to_idx[f] for f in furn_or_floor])
    room_idx_nw = room_idx_all[non_wall]                         # room of every non-wall cell

    tr_room, va_room = make_split(room_idx_all, len(room_names), seed=seed)
    tr_furn, va_furn = make_split(furn_idx_nw, len(furn_names), seed=seed + 1)

    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    data = dict(
        xy_all=xy_all, room_labels_all=room_labels_all, non_wall=non_wall,
        room_names=room_names, room_to_idx=room_to_idx, room_idx_all=room_idx_all,
        furn_names=furn_names, furn_to_idx=furn_to_idx, furn_idx_nw=furn_idx_nw,
        room_idx_nw=room_idx_nw, xy_nw=xy_all[non_wall],
        tr_room=tr_room, va_room=va_room, tr_furn=tr_furn, va_furn=va_furn,
        # torch tensors
        pts_room_tr=t(xy_all[tr_room]), lbl_room_tr=t(room_idx_all[tr_room], torch.long),
        pts_room_va=t(xy_all[va_room]), lbl_room_va=room_idx_all[va_room],
        pts_furn_tr=t(xy_all[non_wall][tr_furn]), lbl_furn_tr=t(furn_idx_nw[tr_furn], torch.long),
        pts_furn_va=t(xy_all[non_wall][va_furn]), lbl_furn_va=furn_idx_nw[va_furn],
        roomof_furn_tr=t(room_idx_nw[tr_furn], torch.long),
        roomof_furn_va=room_idx_nw[va_furn],
        grid_pts=t(xy_all), grid_pts_nw=t(xy_all[non_wall]),
        furn_room_map={f.name: f.room for f in env.furniture},
    )
    return data


def footprints(env: BuildingEnv, data) -> Dict[str, float]:
    """sqrt(area) per class name -- the natural target scale for a kernel."""
    H, W = env.grid_size
    (xmin, xmax), (ymin, ymax) = env.bounds
    cell_area = (xmax - xmin) / W * (ymax - ymin) / H
    fp = {}
    for name in data["room_names"]:
        fp[name] = math.sqrt((data["room_labels_all"] == name).sum() * cell_area)
    grid_f = env.furniture_grid.reshape(-1)
    for name in data["furn_names"]:
        if name == "floor":
            n_cells = ((data["room_labels_all"] != "wall") & (grid_f == "")).sum()
        else:
            n_cells = (grid_f == name).sum()
        fp[name] = math.sqrt(n_cells * cell_area)
    return fp


def furn_probs_to_maps_kl(probs_nw: np.ndarray, data, env) -> Tuple[Dict, Dict]:
    """probs_nw: [N_non_wall_cells, 9] over the full non-wall grid."""
    H, W = env.grid_size
    gt = {"floor": None}
    fmaps = env.furniture_probability_maps()
    floor_mask = (env.room_grid != "wall") & (env.furniture_grid == "")
    gt["floor"] = floor_mask.astype(float) / floor_mask.sum()
    for n in data["furn_names"]:
        if n != "floor":
            gt[n] = fmaps[n]
    est, kl = {}, {}
    for n, i in data["furn_to_idx"].items():
        full = np.zeros(H * W)
        full[data["non_wall"]] = probs_nw[:, i] + EPS
        m = full.reshape(H, W)
        est[n] = m / m.sum()
        kl[n] = kl_divergence(est[n], gt[n])
    return est, kl, gt


def run(args):
    device = args.device
    t0 = time.time()
    env = make_default_building()
    data = prepare_data(env, args.seed, device)
    H, W = env.grid_size
    room_names, furn_names = data["room_names"], data["furn_names"]
    n_rooms, n_furn = len(room_names), len(furn_names)
    print(f"grid {env.grid_size}, rooms {room_names}, furniture classes {furn_names}")
    print(f"room split: {len(data['tr_room'])} train / {len(data['va_room'])} val;  "
          f"furniture split: {len(data['tr_furn'])} train / {len(data['va_furn'])} val\n")

    w_room = balanced_class_weights(data["lbl_room_tr"], n_rooms).to(device)
    w_furn = balanced_class_weights(data["lbl_furn_tr"], n_furn).to(device)

    results = {}

    # ------------------------------------------------------------------
    # Tier 1: room identity, one shared memory
    # ------------------------------------------------------------------
    print("=== Tier 1: room identity (one shared memory) ===")
    t1 = PerClassKernelMap(args.dim, 2, n_rooms, init_lengthscale=0.4,
                           seed=args.seed + 1, device=device)
    mem1, hist1 = train_single_head(t1, data["pts_room_tr"], data["lbl_room_tr"], w_room,
                                    args.epochs, args.lr, args.score_batch, args.mem_batch,
                                    seed=args.seed + 101)
    with torch.no_grad():
        raw1 = t1.score(mem1, data["pts_room_va"]).cpu().numpy()
    temp1 = calibrate_temperature(raw1, data["lbl_room_va"], w_room.cpu().numpy())
    pred1 = softmax_np(raw1, temp1).argmax(axis=1)
    rec1 = per_class_recall(pred1, data["lbl_room_va"], room_names)
    acc1 = float((pred1 == data["lbl_room_va"]).mean())
    ls1 = dict(zip(room_names, t1.lengthscales.detach().cpu().numpy().round(4).tolist()))
    gain1 = dict(zip(room_names, t1.gains.detach().cpu().numpy().round(3).tolist()))
    print(f"  learned lengthscales: {ls1}")
    print(f"  learned gains: {gain1}")
    print(f"  learned temp {hist1['temp'][-1]:.4g} -> post-hoc {temp1:.4g}")
    print(f"  val recall: { {k: round(v, 3) for k, v in rec1.items()} }  acc={acc1:.3f}\n")
    results["tier1"] = dict(recall=rec1, accuracy=acc1, mean_recall=float(np.mean(list(rec1.values()))),
                            lengthscales=ls1, gains=gain1, temp=temp1)

    # room probability maps for Tier 1 (full grid)
    room_probs1 = t1.predict_proba(mem1, data["grid_pts"], temp1)

    # ------------------------------------------------------------------
    # Tier 2: floor/furniture per room (4 small memories)
    # ------------------------------------------------------------------
    print("=== Tier 2: floor + furniture, one memory per room ===")
    tier2 = {}
    # per-room subsets of the GLOBAL furniture split, so every furniture-task
    # method is evaluated on identical validation points.
    roomof_tr = data["roomof_furn_tr"].cpu().numpy()
    for ri, room in enumerate(env.room_names):
        r_glob = data["room_to_idx"][room]
        local_names = ["floor"] + sorted(f.name for f in env.furniture if f.room == room)
        g2l = {data["furn_to_idx"][n]: li for li, n in enumerate(local_names)}
        # training subset for this room
        m_tr = roomof_tr == r_glob
        pts_tr = data["pts_furn_tr"][torch.tensor(np.flatnonzero(m_tr), device=device)]
        lbl_tr_glob = data["lbl_furn_tr"].cpu().numpy()[m_tr]
        lbl_tr = torch.tensor([g2l[g] for g in lbl_tr_glob], dtype=torch.long, device=device)
        w2 = balanced_class_weights(lbl_tr, len(local_names)).to(device)

        m2 = PerClassKernelMap(args.dim, 2, len(local_names), init_lengthscale=0.15,
                               seed=args.seed + 10 + ri, device=device)
        mem2, hist2 = train_single_head(m2, pts_tr, lbl_tr, w2, args.epochs, args.lr,
                                        args.score_batch, args.mem_batch, seed=args.seed + 110 + ri)
        # validation subset for this room
        m_va = data["roomof_furn_va"] == r_glob
        pts_va = data["pts_furn_va"][torch.tensor(np.flatnonzero(m_va), device=device)]
        lbl_va = np.array([g2l[g] for g in data["lbl_furn_va"][m_va]])
        with torch.no_grad():
            raw2 = m2.score(mem2, pts_va).cpu().numpy()
        temp2 = calibrate_temperature(raw2, lbl_va, w2.cpu().numpy())
        pred2 = softmax_np(raw2, temp2).argmax(axis=1)
        rec2 = per_class_recall(pred2, lbl_va, local_names)
        ls2 = dict(zip(local_names, m2.lengthscales.detach().cpu().numpy().round(4).tolist()))
        gain2 = dict(zip(local_names, m2.gains.detach().cpu().numpy().round(3).tolist()))
        tier2[room] = dict(model=m2, memory=mem2, temp=temp2, local_names=local_names,
                           recall=rec2, lengthscales=ls2, gains=gain2, hist=hist2)
        print(f"  {room:>12s}: ls={ls2}  gains={gain2}")
        print(f"  {'':>12s}  recall={ {k: round(v, 3) for k, v in rec2.items()} }")
    print()
    results["tier2"] = {room: dict(recall=d["recall"], lengthscales=d["lengthscales"],
                                   gains=d["gains"], temp=d["temp"])
                        for room, d in tier2.items()}

    # ------------------------------------------------------------------
    # Cascade T1 -> T2 (probabilistic): P(furn=c|x) = sum_r P(r|x) P(c|x, T2_r)
    # ------------------------------------------------------------------
    def cascade_probs(query_pts) -> np.ndarray:
        roomP = t1.predict_proba(mem1, query_pts, temp1)         # [N, n_rooms]
        combined = np.zeros((query_pts.shape[0], n_furn))
        for room, d in tier2.items():
            p2 = d["model"].predict_proba(d["memory"], query_pts, d["temp"])
            rcol = data["room_to_idx"][room]
            for li, n in enumerate(d["local_names"]):
                combined[:, data["furn_to_idx"][n]] += roomP[:, rcol] * p2[:, li]
        combined += EPS
        return combined / combined.sum(axis=1, keepdims=True)

    casc_va = cascade_probs(data["pts_furn_va"])
    pred_c = casc_va.argmax(axis=1)
    rec_c = per_class_recall(pred_c, data["lbl_furn_va"], furn_names)
    acc_c = float((pred_c == data["lbl_furn_va"]).mean())
    results["cascade"] = dict(recall=rec_c, accuracy=acc_c,
                              mean_recall=float(np.mean(list(rec_c.values()))))
    print(f"cascade T1->T2:  mean recall={results['cascade']['mean_recall']:.3f}  acc={acc_c:.3f}")

    # ------------------------------------------------------------------
    # Tier 3 flat: one house memory, flat superposition
    # ------------------------------------------------------------------
    print("\n=== Tier 3 (flat): one whole-building memory ===")
    t3f = HouseMap(args.dim, 2, n_rooms, n_furn, hierarchical=False,
                   seed=args.seed + 2, device=device)
    mem3f, hist3f = train_house(t3f, data["pts_room_tr"], data["lbl_room_tr"], w_room,
                                data["pts_furn_tr"], data["lbl_furn_tr"],
                                data["roomof_furn_tr"], w_furn,
                                args.epochs, args.lr, args.score_batch, args.mem_batch,
                                seed=args.seed + 102)
    with torch.no_grad():
        raw3f_room = t3f.score_rooms(mem3f, data["pts_room_va"]).cpu().numpy()
        raw3f_furn = t3f.score_furniture(mem3f, data["pts_furn_va"]).cpu().numpy()
    temp3f_room = calibrate_temperature(raw3f_room, data["lbl_room_va"], w_room.cpu().numpy())
    temp3f_furn = calibrate_temperature(raw3f_furn, data["lbl_furn_va"], w_furn.cpu().numpy())
    pred3f_room = softmax_np(raw3f_room, temp3f_room).argmax(axis=1)
    rec3f_room = per_class_recall(pred3f_room, data["lbl_room_va"], room_names)
    furnP3f_va = softmax_np(raw3f_furn, temp3f_furn)
    pred3f = furnP3f_va.argmax(axis=1)
    rec3f = per_class_recall(pred3f, data["lbl_furn_va"], furn_names)
    acc3f = float((pred3f == data["lbl_furn_va"]).mean())
    ls3f_room = dict(zip(room_names, t3f.ls_room.detach().cpu().numpy().round(4).tolist()))
    ls3f_furn = dict(zip(furn_names, t3f.ls_furn.detach().cpu().numpy().round(4).tolist()))
    gain3f_furn = dict(zip(furn_names, torch.exp(t3f.log_gain_furn).detach().cpu().numpy().round(3).tolist()))
    print(f"  room ls={ls3f_room}")
    print(f"  furn ls={ls3f_furn}")
    print(f"  furn gains={gain3f_furn}")
    print(f"  room recall={ {k: round(v, 3) for k, v in rec3f_room.items()} }")
    print(f"  furn recall={ {k: round(v, 3) for k, v in rec3f.items()} }  acc={acc3f:.3f}")

    # flat + room prior (the notebooks' joint-then-marginalize fix)
    roomP3f_va = softmax_np(t3f.score_rooms(mem3f, data["pts_furn_va"]).detach().cpu().numpy(), temp3f_room)
    wall_col = data["room_to_idx"]["wall"]
    prior_va = np.zeros_like(furnP3f_va)
    for n, i in data["furn_to_idx"].items():
        compat = (1.0 - roomP3f_va[:, wall_col]) if n == "floor" \
            else roomP3f_va[:, data["room_to_idx"][data["furn_room_map"][n]]]
        prior_va[:, i] = furnP3f_va[:, i] * compat
    prior_va += EPS
    prior_va /= prior_va.sum(axis=1, keepdims=True)
    pred3fp = prior_va.argmax(axis=1)
    rec3fp = per_class_recall(pred3fp, data["lbl_furn_va"], furn_names)
    acc3fp = float((pred3fp == data["lbl_furn_va"]).mean())
    print(f"  furn recall (+room prior)={ {k: round(v, 3) for k, v in rec3fp.items()} }  acc={acc3fp:.3f}")
    results["tier3_flat"] = dict(
        room_recall=rec3f_room, room_accuracy=float((pred3f_room == data["lbl_room_va"]).mean()),
        recall=rec3f, accuracy=acc3f, mean_recall=float(np.mean(list(rec3f.values()))),
        recall_prior=rec3fp, accuracy_prior=acc3fp,
        mean_recall_prior=float(np.mean(list(rec3fp.values()))),
        ls_room=ls3f_room, ls_furn=ls3f_furn, gains_furn=gain3f_furn,
    )

    # ------------------------------------------------------------------
    # Tier 3 hierarchical: furniture bound with room vector
    # ------------------------------------------------------------------
    print("\n=== Tier 3 (hierarchical): furniture bound with room vector ===")
    t3h = HouseMap(args.dim, 2, n_rooms, n_furn, hierarchical=True,
                   seed=args.seed + 3, device=device)
    mem3h, hist3h = train_house(t3h, data["pts_room_tr"], data["lbl_room_tr"], w_room,
                                data["pts_furn_tr"], data["lbl_furn_tr"],
                                data["roomof_furn_tr"], w_furn,
                                args.epochs, args.lr, args.score_batch, args.mem_batch,
                                seed=args.seed + 103)
    with torch.no_grad():
        raw3h_room = t3h.score_rooms(mem3h, data["pts_room_va"]).cpu().numpy()
        oracle_rooms_va = torch.tensor(data["roomof_furn_va"], dtype=torch.long, device=device)
        raw3h_furn_oracle = t3h.score_furniture(mem3h, data["pts_furn_va"],
                                                room_idx=oracle_rooms_va).cpu().numpy()
    temp3h_room = calibrate_temperature(raw3h_room, data["lbl_room_va"], w_room.cpu().numpy())
    temp3h_furn = calibrate_temperature(raw3h_furn_oracle, data["lbl_furn_va"], w_furn.cpu().numpy())
    pred3h_room = softmax_np(raw3h_room, temp3h_room).argmax(axis=1)
    rec3h_room = per_class_recall(pred3h_room, data["lbl_room_va"], room_names)

    def hier_furn_probs(query_pts) -> np.ndarray:
        """Marginalize over room bindings weighted by the room head's own
        decode (renormalized over non-wall rooms): P(c|x) = sum_r P(r|x) P(c|x,r)."""
        with torch.no_grad():
            roomP = softmax_np(t3h.score_rooms(mem3h, query_pts).cpu().numpy(), temp3h_room)
        nw_cols = [data["room_to_idx"][r] for r in env.room_names]
        roomP_nw = roomP[:, nw_cols]
        roomP_nw = roomP_nw / (roomP_nw.sum(axis=1, keepdims=True) + EPS)
        combined = np.zeros((query_pts.shape[0], n_furn))
        for k, r in enumerate(env.room_names):
            ridx = torch.full((query_pts.shape[0],), data["room_to_idx"][r],
                              dtype=torch.long, device=device)
            with torch.no_grad():
                p = softmax_np(t3h.score_furniture(mem3h, query_pts, room_idx=ridx).cpu().numpy(),
                               temp3h_furn)
            combined += roomP_nw[:, [k]] * p
        combined += EPS
        return combined / combined.sum(axis=1, keepdims=True)

    hierP_va = hier_furn_probs(data["pts_furn_va"])
    pred3h = hierP_va.argmax(axis=1)
    rec3h = per_class_recall(pred3h, data["lbl_furn_va"], furn_names)
    acc3h = float((pred3h == data["lbl_furn_va"]).mean())
    pred3h_or = softmax_np(raw3h_furn_oracle, temp3h_furn).argmax(axis=1)
    rec3h_or = per_class_recall(pred3h_or, data["lbl_furn_va"], furn_names)
    acc3h_or = float((pred3h_or == data["lbl_furn_va"]).mean())
    ls3h_room = dict(zip(room_names, t3h.ls_room.detach().cpu().numpy().round(4).tolist()))
    ls3h_furn = dict(zip(furn_names, t3h.ls_furn.detach().cpu().numpy().round(4).tolist()))
    gain3h_furn = dict(zip(furn_names, torch.exp(t3h.log_gain_furn).detach().cpu().numpy().round(3).tolist()))
    print(f"  room ls={ls3h_room}")
    print(f"  furn ls={ls3h_furn}")
    print(f"  furn gains={gain3h_furn}")
    print(f"  room recall={ {k: round(v, 3) for k, v in rec3h_room.items()} }")
    print(f"  furn recall (decoded room)={ {k: round(v, 3) for k, v in rec3h.items()} }  acc={acc3h:.3f}")
    print(f"  furn recall (oracle room) ={ {k: round(v, 3) for k, v in rec3h_or.items()} }  acc={acc3h_or:.3f}")
    results["tier3_hier"] = dict(
        room_recall=rec3h_room, room_accuracy=float((pred3h_room == data["lbl_room_va"]).mean()),
        recall=rec3h, accuracy=acc3h, mean_recall=float(np.mean(list(rec3h.values()))),
        recall_oracle=rec3h_or, accuracy_oracle=acc3h_or,
        mean_recall_oracle=float(np.mean(list(rec3h_or.values()))),
        ls_room=ls3h_room, ls_furn=ls3h_furn, gains_furn=gain3h_furn,
    )

    # ------------------------------------------------------------------
    # Grid decodes for maps + KL (all furniture methods on the same grid)
    # ------------------------------------------------------------------
    print("\ncomputing grid decodes for maps/KL ...")
    grid_nw = data["grid_pts_nw"]
    casc_grid = cascade_probs(grid_nw)
    with torch.no_grad():
        furnP3f_grid = softmax_np(t3f.score_furniture(mem3f, grid_nw).cpu().numpy(), temp3f_furn)
        roomP3f_grid = softmax_np(t3f.score_rooms(mem3f, grid_nw).cpu().numpy(), temp3f_room)
    prior_grid = np.zeros_like(furnP3f_grid)
    for n, i in data["furn_to_idx"].items():
        compat = (1.0 - roomP3f_grid[:, wall_col]) if n == "floor" \
            else roomP3f_grid[:, data["room_to_idx"][data["furn_room_map"][n]]]
        prior_grid[:, i] = furnP3f_grid[:, i] * compat
    prior_grid += EPS
    prior_grid /= prior_grid.sum(axis=1, keepdims=True)
    hier_grid = hier_furn_probs(grid_nw)

    maps = {}
    kls = {}
    for name, probs in [("cascade", casc_grid), ("flat", furnP3f_grid),
                        ("flat+prior", prior_grid), ("hier", hier_grid)]:
        est, kl, gt_furn = furn_probs_to_maps_kl(probs, data, env)
        maps[name], kls[name] = est, kl
        results.setdefault("kl", {})[name] = {k: round(v, 3) for k, v in kl.items()}

    # room maps
    gt_room = env.room_probability_maps()
    with torch.no_grad():
        roomP3f_full = softmax_np(t3f.score_rooms(mem3f, data["grid_pts"]).cpu().numpy(), temp3f_room)
        roomP3h_full = softmax_np(t3h.score_rooms(mem3h, data["grid_pts"]).cpu().numpy(), temp3h_room)
    room_maps = {"tier1": {}, "flat": {}, "hier": {}}
    room_kl = {"tier1": {}, "flat": {}, "hier": {}}
    for src, probs in [("tier1", room_probs1), ("flat", roomP3f_full), ("hier", roomP3h_full)]:
        for n, i in data["room_to_idx"].items():
            m = probs[:, i].reshape(H, W) + EPS
            room_maps[src][n] = m / m.sum()
            room_kl[src][n] = kl_divergence(room_maps[src][n], gt_room[n])
    results["room_kl"] = {s: {k: round(v, 3) for k, v in d.items()} for s, d in room_kl.items()}

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    import os
    os.makedirs(args.outdir, exist_ok=True)
    ext = env.bounds[0] + env.bounds[1]
    extent = (env.bounds[0][0], env.bounds[0][1], env.bounds[1][0], env.bounds[1][1])

    fig, ax = plt.subplots(figsize=(6, 6))
    env.render(ax=ax)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/layout.png", dpi=130)
    plt.close(fig)

    # room decode figure: gt / T1 / T3flat / T3hier
    rows = [("ground truth", None, gt_room), ("Tier 1", room_kl["tier1"], room_maps["tier1"]),
            ("Tier 3 flat", room_kl["flat"], room_maps["flat"]),
            ("Tier 3 hier", room_kl["hier"], room_maps["hier"])]
    fig, axes = plt.subplots(len(rows), n_rooms, figsize=(2.4 * n_rooms, 2.4 * len(rows)))
    for r, (rname, kl_d, maps_d) in enumerate(rows):
        for c, n in enumerate(room_names):
            axes[r, c].imshow(maps_d[n], extent=extent, origin="upper", cmap="viridis")
            title = f"{n}" if r == 0 else (f"KL={kl_d[n]:.2f}" if kl_d else "")
            axes[r, c].set_title(title, fontsize=8)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(rname, fontsize=9)
    fig.suptitle("Room decode: ground truth vs. Tier 1 vs. Tier 3 heads")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/room_decode.png", dpi=130)
    plt.close(fig)

    # furniture decode figure: gt + 4 methods x 9 classes
    method_rows = [("ground truth", None, gt_furn), ("cascade T1->T2", kls["cascade"], maps["cascade"]),
                   ("Tier 3 flat", kls["flat"], maps["flat"]),
                   ("flat + room prior", kls["flat+prior"], maps["flat+prior"]),
                   ("Tier 3 hier", kls["hier"], maps["hier"])]
    fig, axes = plt.subplots(len(method_rows), n_furn, figsize=(1.9 * n_furn, 2.0 * len(method_rows)))
    for r, (rname, kl_d, maps_d) in enumerate(method_rows):
        for c, n in enumerate(furn_names):
            axes[r, c].imshow(maps_d[n], extent=extent, origin="upper", cmap="viridis")
            title = n if r == 0 else f"KL={kl_d[n]:.1f}"
            axes[r, c].set_title(title, fontsize=7)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(rname, fontsize=8)
    fig.suptitle("Furniture decode across methods (same non-wall grid)")
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/furniture_decode.png", dpi=130)
    plt.close(fig)

    # summary: recall bars + learned lengthscales vs footprints
    fp = footprints(env, data)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    methods = ["cascade", "tier3_flat", "tier3_flat_prior", "tier3_hier"]
    labels = ["cascade\nT1->T2", "T3 flat", "T3 flat\n+prior", "T3 hier"]
    mean_rec = [results["cascade"]["mean_recall"], results["tier3_flat"]["mean_recall"],
                results["tier3_flat"]["mean_recall_prior"], results["tier3_hier"]["mean_recall"]]
    accs = [results["cascade"]["accuracy"], results["tier3_flat"]["accuracy"],
            results["tier3_flat"]["accuracy_prior"], results["tier3_hier"]["accuracy"]]
    x = np.arange(len(methods))
    axes[0].bar(x - 0.2, mean_rec, 0.4, label="mean per-class recall")
    axes[0].bar(x + 0.2, accs, 0.4, label="accuracy (query-weighted)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(1.0, color="gray", lw=0.5)
    axes[0].set_title("Furniture task (9-way, same val points)")
    axes[0].legend(fontsize=8)

    room_methods = [("Tier 1", results["tier1"]), ("T3 flat", results["tier3_flat"]),
                    ("T3 hier", results["tier3_hier"])]
    mr = [np.mean(list((d.get("recall") if nm == "Tier 1" else d["room_recall"]).values()))
          for nm, d in room_methods]
    ra = [d.get("accuracy") if nm == "Tier 1" else d["room_accuracy"] for nm, d in room_methods]
    x = np.arange(3)
    axes[1].bar(x - 0.2, mr, 0.4, label="mean per-class recall")
    axes[1].bar(x + 0.2, ra, 0.4, label="accuracy")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([nm for nm, _ in room_methods], fontsize=8)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Room task (5-way incl. wall)")
    axes[1].legend(fontsize=8)

    # learned lengthscale vs footprint
    pts = []
    for n, v in ls1.items():
        pts.append((fp[n], v, f"T1:{n}", "o"))
    for room, d in tier2.items():
        for n, v in d["lengthscales"].items():
            key = n if n != "floor" else room  # floor footprint ~ per-room; approximate with room
            pts.append((fp[n] if n != "floor" else fp[room], v, f"T2:{n}", "s"))
    for n, v in ls3f_furn.items():
        pts.append((fp[n], v, f"T3f:{n}", "^"))
    for n, v in ls3h_furn.items():
        pts.append((fp[n], v, f"T3h:{n}", "v"))
    for marker, lab in [("o", "Tier 1 (rooms)"), ("s", "Tier 2"), ("^", "T3 flat furn"), ("v", "T3 hier furn")]:
        xs = [p[0] for p in pts if p[3] == marker]
        ys = [p[1] for p in pts if p[3] == marker]
        axes[2].scatter(xs, ys, marker=marker, alpha=0.7, label=lab)
    lims = [0.05, 2.0]
    axes[2].plot(lims, lims, "k--", lw=0.6, label="ls = footprint")
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("class footprint sqrt(area)")
    axes[2].set_ylabel("learned lengthscale")
    axes[2].set_title("Learned kernel bandwidth vs. object size")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(f"{args.outdir}/summary.png", dpi=130)
    plt.close(fig)

    # training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(hist1["loss"], label="Tier 1")
    axes[0].plot(hist3f["loss"], label="T3 flat (total)")
    axes[0].plot(hist3h["loss"], label="T3 hier (total)")
    axes[0].set_title("training loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend(fontsize=8)
    for i, n in enumerate(furn_names):
        axes[1].plot(hist3f["ls_furn"][:, i], label=n)
    axes[1].set_title("T3 flat: furniture lengthscales")
    axes[1].set_xlabel("epoch")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=6, ncol=2)
    for i, n in enumerate(furn_names):
        axes[2].plot(hist3h["ls_furn"][:, i], label=n)
    axes[2].set_title("T3 hier: furniture lengthscales")
    axes[2].set_xlabel("epoch")
    axes[2].set_yscale("log")
    axes[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{args.outdir}/training_curves.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY (all furniture methods share identical validation points)")
    print("=" * 78)
    print("\nRoom task (5-way, incl. wall):")
    print(f"  {'method':<22s} {'mean recall':>12s} {'accuracy':>10s}")
    print(f"  {'Tier 1':<22s} {results['tier1']['mean_recall']:>12.3f} {results['tier1']['accuracy']:>10.3f}")
    print(f"  {'Tier 3 flat (head)':<22s} {np.mean(list(rec3f_room.values())):>12.3f} "
          f"{results['tier3_flat']['room_accuracy']:>10.3f}")
    print(f"  {'Tier 3 hier (head)':<22s} {np.mean(list(rec3h_room.values())):>12.3f} "
          f"{results['tier3_hier']['room_accuracy']:>10.3f}")

    print("\nFurniture task (9-way):")
    print(f"  {'method':<22s} {'mean recall':>12s} {'accuracy':>10s} {'mean KL':>9s}")
    rows_f = [
        ("cascade T1->T2", results["cascade"]["mean_recall"], results["cascade"]["accuracy"], "cascade"),
        ("Tier 3 flat", results["tier3_flat"]["mean_recall"], results["tier3_flat"]["accuracy"], "flat"),
        ("flat + room prior", results["tier3_flat"]["mean_recall_prior"],
         results["tier3_flat"]["accuracy_prior"], "flat+prior"),
        ("Tier 3 hier", results["tier3_hier"]["mean_recall"], results["tier3_hier"]["accuracy"], "hier"),
    ]
    for nm, mrec, acc, key in rows_f:
        mkl = np.mean(list(kls[key].values()))
        print(f"  {nm:<22s} {mrec:>12.3f} {acc:>10.3f} {mkl:>9.2f}")
    print(f"  {'T3 hier (oracle room)':<22s} {results['tier3_hier']['mean_recall_oracle']:>12.3f} "
          f"{results['tier3_hier']['accuracy_oracle']:>10.3f} {'-':>9s}")

    print("\nPer-class recall (furniture task):")
    hdr = f"  {'class':<10s} {'cascade':>8s} {'flat':>8s} {'+prior':>8s} {'hier':>8s} {'hier(or)':>9s}"
    print(hdr)
    for n in furn_names:
        print(f"  {n:<10s} {rec_c[n]:>8.3f} {rec3f[n]:>8.3f} {rec3fp[n]:>8.3f} "
              f"{rec3h[n]:>8.3f} {rec3h_or[n]:>9.3f}")

    print("\nLearned lengthscales vs. class footprint sqrt(area):")
    print(f"  {'class':<12s} {'footprint':>9s} {'T1/T2':>8s} {'T3 flat':>8s} {'T3 hier':>8s}")
    for n in room_names:
        print(f"  {n:<12s} {fp[n]:>9.3f} {ls1[n]:>8.3f} {ls3f_room[n]:>8.3f} {ls3h_room[n]:>8.3f}")
    t2_ls_lookup = {}
    for room, d in tier2.items():
        for n, v in d["lengthscales"].items():
            t2_ls_lookup.setdefault(n, []).append(v)
    for n in furn_names:
        t2v = np.mean(t2_ls_lookup[n]) if n in t2_ls_lookup else float("nan")
        print(f"  {n:<12s} {fp[n]:>9.3f} {t2v:>8.3f} {ls3f_furn[n]:>8.3f} {ls3h_furn[n]:>8.3f}")

    results["footprints"] = {k: round(v, 4) for k, v in fp.items()}
    results["runtime_sec"] = round(time.time() - t0, 1)
    results["args"] = {k: v for k, v in vars(args).items()}
    with open(f"{args.outdir}/results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\ndone in {results['runtime_sec']}s -- figures + results.json in {args.outdir}/")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--score-batch", type=int, default=1024,
                    help="points scored per epoch (gradient minibatch)")
    ap.add_argument("--mem-batch", type=int, default=6000,
                    help="points superposed into the memory per training epoch "
                         "(final memory always uses all training points)")
    ap.add_argument("--outdir", type=str, default="lk_results")
    ap.add_argument("--quick", action="store_true", help="small/fast smoke test")
    args = ap.parse_args()
    if args.quick:
        args.dim = 256
        args.epochs = 60
    torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
