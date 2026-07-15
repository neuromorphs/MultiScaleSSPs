"""Export Robot@Home homes as synthetic CARMEN logs for hwSSPslam.

Robot@Home's database has no odometry or localized trajectory, but it DOES
ship a localized, room-labeled 2D geometric map per home (rh_twodgeomap).
This exporter turns each home into a synthetic-but-grounded SLAM dataset:

  1. rasterize the labeled geomap into an occupancy grid,
  2. plan a smooth pseudo-path visiting every room (A* between room
     centroids on the inflated free grid),
  3. sample poses along it at constant speed (exact GT velocities),
  4. integrate a diff-drive noise model on the exact deltas -> odometry,
  5. raycast a 180 deg x 360-beam scan at every pose (matching hwSSPslam's
     CARMEN sensor convention: beam_i = -90 + i*(180/n) deg),
  6. write hwSSPslam-compatible files:
       <out>/rathome_<home>.log       FLASER lines with NOISY odometry poses
       <out>/rathome_<home>.gfs.log   same scans with TRUE poses (eval ref)
       <out>/rathome_<home>_labels.npz  per-beam room-type labels + GT poses

The .log/.gfs.log pair drops straight into hwSSPslam
(`python3 ssp_bounded_carmen.py data/rathome_anto.log`); the sidecar npz
carries the semantics for the later binding experiments.

Usage:
    python scripts/export_robotathome_carmen.py --home anto \
        --out ../hwSSPslam/data
    python scripts/export_robotathome_carmen.py --home all --out ../hwSSPslam/data
"""

import argparse
import heapq
from pathlib import Path

import numpy as np

from multiscalessps.data import RobotAtHome

NO_RETURN = 81.91          # > VALID_MAX(40) -> treated as no-return
SENSOR_FOV = np.pi         # 180 deg, hwSSPslam carmen convention
N_BEAMS = 360


# ---------------------------------------------------------------------------
# occupancy grid from the labeled geomap
# ---------------------------------------------------------------------------

def build_grids(pts, labels, res=0.05, pad=1.0, dilate=2):
    """Occupancy + label grids from geomap points. Returns dict with
    origin, res, occ (H,W) bool, lab (H,W) int8 (-1 = free)."""
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    shape = np.ceil((hi - lo) / res).astype(int)[::-1]  # (rows=y, cols=x)
    occ = np.zeros(shape, bool)
    lab = np.full(shape, -1, np.int8)
    ij = np.floor((pts - lo) / res).astype(int)
    occ[ij[:, 1], ij[:, 0]] = True
    lab[ij[:, 1], ij[:, 0]] = labels
    # dilate occupancy (close sampling gaps in walls) and spread labels
    for _ in range(dilate):
        o = occ.copy()
        l = lab.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh = np.roll(np.roll(occ, dy, 0), dx, 1)
            newly = sh & ~o
            o |= sh
            l[newly] = np.roll(np.roll(lab, dy, 0), dx, 1)[newly]
        occ, lab = o, l
    return dict(origin=lo, res=res, occ=occ, lab=lab)


def inflate(occ, cells):
    out = occ.copy()
    for _ in range(cells):
        o = out.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            o |= np.roll(np.roll(out, dy, 0), dx, 1)
        out = o
    return out


# ---------------------------------------------------------------------------
# path planning: A* between room centroids on the inflated free grid
# ---------------------------------------------------------------------------

def astar(free, start, goal):
    """4/8-connected A* on boolean free grid; start/goal (row, col)."""
    h = lambda a: np.hypot(a[0] - goal[0], a[1] - goal[1])
    openq = [(h(start), 0.0, start, None)]
    came, cost = {}, {start: 0.0}
    while openq:
        _, g, cur, parent = heapq.heappop(openq)
        if cur in came:
            continue
        came[cur] = parent
        if cur == goal:
            path = [cur]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return path[::-1]
        for dy, dx, w in ((1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
                          (1, 1, 1.414), (1, -1, 1.414),
                          (-1, 1, 1.414), (-1, -1, 1.414)):
            nxt = (cur[0] + dy, cur[1] + dx)
            if not (0 <= nxt[0] < free.shape[0]
                    and 0 <= nxt[1] < free.shape[1]) or not free[nxt]:
                continue
            ng = g + w
            if ng < cost.get(nxt, np.inf):
                cost[nxt] = ng
                heapq.heappush(openq, (ng + h(nxt), ng, nxt, cur))
    return None


def nearest_free(free, cell):
    """Closest free cell to `cell` (BFS ring search)."""
    if free[cell]:
        return cell
    fy, fx = np.nonzero(free)
    k = np.argmin((fy - cell[0]) ** 2 + (fx - cell[1]) ** 2)
    return (int(fy[k]), int(fx[k]))


def room_tour(rh, home, grids, free):
    """Waypoint cells: room centroids (snapped to free space), ordered as a
    greedy nearest-neighbor tour from the largest room."""
    home_id = rh._home_id(home)
    rooms = [rid for rid, r in rh.rooms.items() if r["home_id"] == home_id]
    cents, sizes = [], []
    for rid in rooms:
        p, _ = rh.geomap(room_id=rid)
        if len(p) < 20:
            continue
        cents.append(p.mean(axis=0))
        sizes.append(len(p))
    cents = np.array(cents)
    order = [int(np.argmax(sizes))]
    left = set(range(len(cents))) - set(order)
    while left:
        cur = cents[order[-1]]
        nxt = min(left, key=lambda i: np.linalg.norm(cents[i] - cur))
        order.append(nxt)
        left.remove(nxt)
    cells = []
    for i in order:
        c = np.floor((cents[i] - grids["origin"]) / grids["res"]).astype(int)
        cells.append(nearest_free(free, (int(c[1]), int(c[0]))))
    return cells


def smooth_path(cells, grids, free, iters=200, alpha=0.15):
    """Grid path -> smoothed metric polyline (reverts moves that leave free
    space)."""
    pts = np.array([(c[1] + 0.5, c[0] + 0.5) for c in cells]) * grids["res"] \
        + grids["origin"]
    p = pts.copy()
    for _ in range(iters):
        prop = p.copy()
        prop[1:-1] += alpha * (p[:-2] + p[2:] - 2 * p[1:-1])
        ij = np.floor((prop - grids["origin"]) / grids["res"]).astype(int)
        ok = free[ij[:, 1], ij[:, 0]]
        p[ok] = prop[ok]
    return p


def sample_trajectory(polyline, speed=0.30, dt=0.10):
    """Constant-speed poses along the polyline; heading = smoothed tangent.
    Returns (n,3) [x, y, theta]."""
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = int(s[-1] / (speed * dt))
    si = np.linspace(0, s[-1], n)
    xy = np.stack([np.interp(si, s, polyline[:, k]) for k in range(2)], 1)
    d = np.gradient(xy, axis=0)
    # smooth the tangent so heading (and omega) stay physical
    k = 15
    ker = np.ones(k) / k
    d = np.stack([np.convolve(d[:, 0], ker, "same"),
                  np.convolve(d[:, 1], ker, "same")], 1)
    th = np.arctan2(d[:, 1], d[:, 0])
    th = np.unwrap(th)
    return np.column_stack([xy, th])


# ---------------------------------------------------------------------------
# odometry noise + raycasting
# ---------------------------------------------------------------------------

def noisy_odometry(gt, rng, a_trans=0.03, a_rot=0.02, a_drift=0.002):
    """Integrate exact SE(2) deltas with diff-drive-style noise:
    translation scaled by (1+eps), rotation with additive noise plus a
    slowly-varying bias -- returns odometry track starting at gt[0]."""
    odom = gt.copy()
    bias = 0.0
    for k in range(1, len(gt)):
        c, s = np.cos(gt[k - 1, 2]), np.sin(gt[k - 1, 2])
        R = np.array([[c, s], [-s, c]])
        dloc = R @ (gt[k, :2] - gt[k - 1, :2])
        dth = np.arctan2(np.sin(gt[k, 2] - gt[k - 1, 2]),
                         np.cos(gt[k, 2] - gt[k - 1, 2]))
        dn = np.linalg.norm(dloc)
        bias += rng.normal(0, a_drift * np.sqrt(max(dn, 1e-9)))
        dloc = dloc * (1 + rng.normal(0, a_trans))
        dth = dth * (1 + rng.normal(0, a_rot)) + bias * dn
        c, s = np.cos(odom[k - 1, 2]), np.sin(odom[k - 1, 2])
        odom[k, :2] = odom[k - 1, :2] + np.array([[c, -s], [s, c]]) @ dloc
        odom[k, 2] = odom[k - 1, 2] + dth
    return odom


def raycast(grids, pose, beam, max_range=15.0, sigma=0.01, rng=None):
    """DDA raycast for all beams at once. Returns (ranges, hit_labels)."""
    res, origin, occ, lab = (grids["res"], grids["origin"], grids["occ"],
                             grids["lab"])
    ang = pose[2] + beam
    step = res * 0.7
    n_steps = int(max_range / step)
    dxy = np.stack([np.cos(ang), np.sin(ang)], 1) * step
    cur = np.repeat(pose[None, :2], len(beam), 0).astype(float)
    ranges = np.full(len(beam), NO_RETURN)
    labels = np.full(len(beam), -1, np.int8)
    alive = np.ones(len(beam), bool)
    for i in range(1, n_steps + 1):
        cur[alive] += dxy[alive]
        ij = np.floor((cur[alive] - origin) / res).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, occ.shape[1] - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, occ.shape[0] - 1)
        hit = occ[ij[:, 1], ij[:, 0]]
        if hit.any():
            idx = np.flatnonzero(alive)[hit]
            r = i * step
            ranges[idx] = r + (rng.normal(0, sigma, len(idx)) if rng is not
                               None else 0)
            labels[idx] = lab[ij[hit, 1], ij[hit, 0]]
            alive[idx] = False
        if not alive.any():
            break
    return ranges, labels


# ---------------------------------------------------------------------------
# CARMEN output
# ---------------------------------------------------------------------------

def write_flaser(path, scans, poses, ts):
    with open(path, "w") as f:
        for r, p, t in zip(scans, poses, ts):
            vals = " ".join(f"{v:.3f}" for v in r)
            f.write(f"FLASER {len(r)} {vals} "
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{t:.6f} synth {t:.6f}\n")


def export_home(rh, home, out_dir, args):
    print(f"[{home}]")
    pts, labels = rh.geomap(home=home)
    print(f"  geomap {len(pts):,} pts, extent {np.ptp(pts, axis=0).round(1)}")
    grids = build_grids(pts, labels, res=args.res, dilate=args.dilate)
    free = ~inflate(grids["occ"], int(np.ceil(args.robot_radius / args.res)))
    # Plan only on the largest connected free component -- room centroids
    # otherwise snap into furniture-interior pockets A* can't escape.
    from scipy import ndimage
    cc, _ = ndimage.label(free)
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    free = cc == sizes.argmax()
    print(f"  grid {grids['occ'].shape}, occupied {grids['occ'].mean():.1%}, "
          f"free (main component) {free.mean():.1%}")

    cells = room_tour(rh, home, grids, free)
    # Ping-pong laps: out through all rooms and back, repeated -- gives the
    # loop-closure machinery genuine revisits.
    seq = cells + cells[-2::-1]
    seq = seq * args.laps
    full = []
    for a, b in zip(seq[:-1], seq[1:]):
        seg = astar(free, a, b)
        if seg is None:
            print(f"  WARNING: no path between rooms at {a}->{b}, skipping leg")
            continue
        full.extend(seg if not full else seg[1:])
    if len(full) < 10:
        print("  FAILED: no usable tour; try --res/--robot-radius")
        return
    poly = smooth_path(full, grids, free)
    gt = sample_trajectory(poly, speed=args.speed, dt=args.dt)
    print(f"  tour {len(cells)} rooms, path {len(gt)} poses "
          f"({len(gt) * args.dt:.0f} s @ {args.speed} m/s)")

    rng = np.random.default_rng(args.seed)
    odom = noisy_odometry(gt, rng, a_trans=args.noise_trans,
                          a_rot=args.noise_rot, a_drift=args.noise_drift)
    drift = np.linalg.norm(odom[:, :2] - gt[:, :2], axis=1)
    print(f"  odometry drift: final {drift[-1]:.2f} m, max {drift.max():.2f} m")

    beam = np.deg2rad(-90.0 + np.arange(N_BEAMS) * (180.0 / N_BEAMS))
    scans = np.empty((len(gt), N_BEAMS))
    blab = np.empty((len(gt), N_BEAMS), np.int8)
    for k, p in enumerate(gt):
        scans[k], blab[k] = raycast(grids, p, beam, sigma=args.range_sigma,
                                    rng=rng)
    hit_frac = (scans < 40).mean()
    print(f"  scans: {hit_frac:.1%} beams return")

    ts = np.arange(len(gt)) * args.dt
    out_dir.mkdir(parents=True, exist_ok=True)
    write_flaser(out_dir / f"rathome_{home}.log", scans, odom, ts)
    write_flaser(out_dir / f"rathome_{home}.gfs.log", scans, gt, ts)
    np.savez_compressed(
        out_dir / f"rathome_{home}_labels.npz",
        beam_labels=blab, gt=gt, odom=odom, ts=ts,
        room_types=np.array([rh.room_types.get(i, "?") for i in
                             range(max(rh.room_types) + 1)]))
    print(f"  wrote rathome_{home}.log / .gfs.log / _labels.npz")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=Path("data/robotathome/rh.db"))
    ap.add_argument("--home", default="anto",
                    help="home name or 'all'")
    ap.add_argument("--out", type=Path,
                    default=Path("hwSSPslam/data"))
    ap.add_argument("--res", type=float, default=0.05)
    ap.add_argument("--dilate", type=int, default=2,
                    help="occupancy dilation cells (closes wall gaps)")
    ap.add_argument("--robot-radius", type=float, default=0.25)
    ap.add_argument("--speed", type=float, default=0.30)
    ap.add_argument("--dt", type=float, default=0.10)
    ap.add_argument("--range-sigma", type=float, default=0.01)
    ap.add_argument("--laps", type=int, default=2,
                    help="Ping-pong laps through the room tour (revisits "
                         "give loop closure real work).")
    ap.add_argument("--noise-trans", type=float, default=0.15)
    ap.add_argument("--noise-rot", type=float, default=0.10)
    ap.add_argument("--noise-drift", type=float, default=0.015)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rh = RobotAtHome(args.db)
    homes = list(rh.homes.values()) if args.home == "all" else [args.home]
    for h in homes:
        export_home(rh, h, args.out, args)


if __name__ == "__main__":
    main()
