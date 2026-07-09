#!/usr/bin/env python3
"""Numpy forward-math validation for learnable_kernel_tiers.py (no torch).

Loads the real script with a stub torch module so we exercise its actual
env / metric / split / footprint code, then mirrors the FHRR forward math
(encode -> memory -> score, flat and hierarchical binding, cascade
combination) in numpy with FIXED lengthscales. Verifies the design decodes
sensibly before the learnable version runs anywhere with real torch.
"""
import math
import sys
import types
import numpy as np


def install_torch_stub():
    t = types.ModuleType("torch")

    class _NoGrad:
        def __call__(self, f):
            return f

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    t.no_grad = lambda: _NoGrad()
    t.pi = math.pi
    nn = types.ModuleType("torch.nn")

    class Module:
        pass

    nn.Module = Module
    nnF = types.ModuleType("torch.nn.functional")
    t.nn = nn
    t.Tensor = object
    sys.modules["torch"] = t
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = nnF


install_torch_stub()

import importlib.util

spec = importlib.util.spec_from_file_location("lkt", sys.argv[1])
lkt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lkt)
print("script module loaded with stub torch OK")

env = lkt.make_default_building()
H, W = env.grid_size

# ---- numpy data prep (mirrors prepare_data without torch tensors) ----
SEED = 0
positions = env.dense_positions()
xy_all = np.array([p for p, _ in positions], dtype=np.float64)
room_labels_all = np.array([str(l["room"]) for _, l in positions])
furn_labels_all = np.array(["" if l["furniture"] is None else str(l["furniture"]) for _, l in positions])

room_names = sorted(set(room_labels_all))
room_to_idx = {n: i for i, n in enumerate(room_names)}
room_idx_all = np.array([room_to_idx[r] for r in room_labels_all])
non_wall = room_labels_all != "wall"
furn_or_floor = np.where(furn_labels_all[non_wall] == "", "floor", furn_labels_all[non_wall])
furn_names = ["floor"] + sorted(n for n in set(furn_or_floor) if n != "floor")
furn_to_idx = {n: i for i, n in enumerate(furn_names)}
furn_idx_nw = np.array([furn_to_idx[f] for f in furn_or_floor])
room_idx_nw = room_idx_all[non_wall]
xy_nw = xy_all[non_wall]

tr_room, va_room = lkt.make_split(room_idx_all, len(room_names), seed=SEED)
tr_furn, va_furn = lkt.make_split(furn_idx_nw, len(furn_names), seed=SEED + 1)
print(f"splits: room {len(tr_room)}/{len(va_room)}  furn {len(tr_furn)}/{len(va_furn)}")

DIM = 1024
rng = np.random.default_rng(SEED)


def rand_phase(*shape):
    return rng.uniform(-np.pi, np.pi, size=shape)


def build_memory(pts, lbls, ls_per_class, axis, class_vecs, bind=None):
    base = pts @ axis                                    # [N, D]
    pos = np.exp(1j * base / ls_per_class[lbls][:, None])
    atoms = pos * class_vecs[lbls]
    if bind is not None:
        atoms = atoms * bind
    m = atoms.sum(axis=0)
    return m / np.linalg.norm(m)


def score(memory, query, ls_per_class, axis, class_vecs, bind=None, chunk=512):
    """bind: optional [N, D] per-query binding (e.g. conj-room for hier)."""
    C = class_vecs.shape[0]
    out = np.empty((query.shape[0], C))
    mc = memory[None, :] * np.conj(class_vecs)           # [C, D]
    for i in range(0, query.shape[0], chunk):
        q = query[i:i + chunk]
        base = q @ axis
        pos_conj = np.exp(-1j * base[:, None, :] / ls_per_class[None, :, None])
        if bind is None:
            out[i:i + chunk] = np.einsum("ncd,cd->nc", pos_conj, mc).real / DIM
        else:
            t = np.conj(bind[i:i + chunk]) * memory[None, :]
            out[i:i + chunk] = np.einsum("ncd,cd,nd->nc", pos_conj, np.conj(class_vecs), t).real / DIM
    return out


def recall_line(tag, pred, lbls, names):
    r = lkt.per_class_recall(pred, lbls, names)
    print(f"  {tag}: mean_recall={np.mean(list(r.values())):.3f} acc={(pred == lbls).mean():.3f}  "
          f"{ {k: round(v, 2) for k, v in r.items()} }")
    return r


w_room = np.ones(len(room_names))
w_furn = np.ones(len(furn_names))

# ---- Tier 1 (fixed ls) ----
axis1 = rand_phase(2, DIM)
cv1 = np.exp(1j * rand_phase(len(room_names), DIM))
ls1 = np.full(len(room_names), 0.35)
mem1 = build_memory(xy_all[tr_room], room_idx_all[tr_room], ls1, axis1, cv1)
raw = score(mem1, xy_all[va_room], ls1, axis1, cv1)
temp1 = lkt.calibrate_temperature(raw, room_idx_all[va_room], w_room)
pred = lkt.softmax_np(raw, temp1).argmax(1)
print("\nTier 1 (fixed ls=0.35):")
recall_line("T1", pred, room_idx_all[va_room], room_names)


def t1_probs(q):
    return lkt.softmax_np(score(mem1, q, ls1, axis1, cv1), temp1)


# ---- Tier 2 per room (fixed ls) ----
print("\nTier 2 (fixed ls: floor 0.3, furniture 0.12):")
tier2 = {}
roomof_tr = room_idx_nw[tr_furn]
roomof_va = room_idx_nw[va_furn]
pts_furn_tr, lbl_furn_tr = xy_nw[tr_furn], furn_idx_nw[tr_furn]
pts_furn_va, lbl_furn_va = xy_nw[va_furn], furn_idx_nw[va_furn]
for room in env.room_names:
    rg = room_to_idx[room]
    local_names = ["floor"] + sorted(f.name for f in env.furniture if f.room == room)
    g2l = {furn_to_idx[n]: li for li, n in enumerate(local_names)}
    m_tr = roomof_tr == rg
    pts = pts_furn_tr[m_tr]
    lbl = np.array([g2l[g] for g in lbl_furn_tr[m_tr]])
    axis2 = rand_phase(2, DIM)
    cv2 = np.exp(1j * rand_phase(len(local_names), DIM))
    ls2 = np.array([0.1] + [0.15] * (len(local_names) - 1))
    mem2 = build_memory(pts, lbl, ls2, axis2, cv2)
    m_va = roomof_va == rg
    raw = score(mem2, pts_furn_va[m_va], ls2, axis2, cv2)
    lblv = np.array([g2l[g] for g in lbl_furn_va[m_va]])
    temp2 = lkt.calibrate_temperature(raw, lblv, np.ones(len(local_names)))
    pred = lkt.softmax_np(raw, temp2).argmax(1)
    recall_line(room, pred, lblv, local_names)
    tier2[room] = (axis2, cv2, ls2, mem2, temp2, local_names)

# ---- cascade ----
def cascade_probs(q):
    roomP = t1_probs(q)
    combined = np.zeros((q.shape[0], len(furn_names)))
    for room, (axis2, cv2, ls2, mem2, temp2, local_names) in tier2.items():
        p2 = lkt.softmax_np(score(mem2, q, ls2, axis2, cv2), temp2)
        for li, n in enumerate(local_names):
            combined[:, furn_to_idx[n]] += roomP[:, room_to_idx[room]] * p2[:, li]
    combined += 1e-12
    return combined / combined.sum(1, keepdims=True)


print("\nCascade T1->T2:")
recall_line("cascade", cascade_probs(pts_furn_va).argmax(1), lbl_furn_va, furn_names)

# ---- Tier 3 flat & hier ----
axis_r = rand_phase(2, DIM)
axis_f = rand_phase(2, DIM)
rv = np.exp(1j * rand_phase(len(room_names), DIM))
fv = np.exp(1j * rand_phase(len(furn_names), DIM))
ls_r = np.full(len(room_names), 0.35)
ls_f = np.array([0.1] + [0.15] * (len(furn_names) - 1))

for hier in [False, True]:
    tag = "hier" if hier else "flat"
    base_f = pts_furn_tr @ axis_f
    pos_f = np.exp(1j * base_f / ls_f[lbl_furn_tr][:, None])
    atoms_f = pos_f * fv[lbl_furn_tr]
    if hier:
        atoms_f = atoms_f * rv[roomof_tr]
    base_r = xy_all[tr_room] @ axis_r
    pos_r = np.exp(1j * base_r / ls_r[room_idx_all[tr_room]][:, None])
    rec_r_un = (pos_r * rv[room_idx_all[tr_room]]).sum(0)
    mem = rec_r_un + atoms_f.sum(0)
    mem = mem / np.linalg.norm(mem)

    raw_room = score(mem, xy_all[va_room], ls_r, axis_r, rv)
    tr_ = lkt.calibrate_temperature(raw_room, room_idx_all[va_room], w_room)
    pred_room = lkt.softmax_np(raw_room, tr_).argmax(1)
    print(f"\nTier 3 {tag}:")
    recall_line("room head", pred_room, room_idx_all[va_room], room_names)

    if not hier:
        raw_f = score(mem, pts_furn_va, ls_f, axis_f, fv)
        tf_ = lkt.calibrate_temperature(raw_f, lbl_furn_va, w_furn)
        recall_line("furn (flat)", lkt.softmax_np(raw_f, tf_).argmax(1), lbl_furn_va, furn_names)
    else:
        raw_f = score(mem, pts_furn_va, ls_f, axis_f, fv, bind=rv[roomof_va])
        tf_ = lkt.calibrate_temperature(raw_f, lbl_furn_va, w_furn)
        recall_line("furn (oracle room)", lkt.softmax_np(raw_f, tf_).argmax(1), lbl_furn_va, furn_names)
        # marginalized over decoded room
        roomP = lkt.softmax_np(score(mem, pts_furn_va, ls_r, axis_r, rv), tr_)
        nw_cols = [room_to_idx[r] for r in env.room_names]
        roomP_nw = roomP[:, nw_cols]
        roomP_nw /= roomP_nw.sum(1, keepdims=True) + 1e-12
        combined = np.zeros((pts_furn_va.shape[0], len(furn_names)))
        for k, r in enumerate(env.room_names):
            bindr = np.tile(rv[room_to_idx[r]], (pts_furn_va.shape[0], 1))
            p = lkt.softmax_np(score(mem, pts_furn_va, ls_f, axis_f, fv, bind=bindr), tf_)
            combined += roomP_nw[:, [k]] * p
        recall_line("furn (marginalized)", combined.argmax(1), lbl_furn_va, furn_names)

# footprints sanity
data_stub = dict(room_names=room_names, furn_names=furn_names,
                 room_labels_all=room_labels_all)
fp = lkt.footprints(env, data_stub)
print("\nfootprints:", {k: round(v, 3) for k, v in fp.items()})
print("\nvalidation complete")
