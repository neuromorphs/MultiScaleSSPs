"""Loader for the Robot@Home2 dataset (rh.db SQLite database).

Robot@Home2: a mobile robot with a 2D Hokuyo laser (682 beams, ~4.18 rad
aperture, 5.6 m range) driven through 5 apartments (41 rooms, 9 room
categories), with per-observation sensor poses and room annotations.
Download with scripts/download_robotathome.py.

Typical use:

    rh = RobotAtHome()                        # data/robotathome/rh.db
    rh.homes                                  # {0: 'alma', ...}
    rh.room_types                             # {0: 'bathroom', ...}
    obs = rh.laser_observations(home="alma")  # structured array, time-ordered
    pts, labels, obs_idx = rh.scan_points(obs["id"][::10])   # robot frame
    map_pts, map_labels = rh.geomap(home="alma")   # localized labeled 2D map

FRAMES -- important: the database contains NO localized robot trajectory.
The sensor_pose_* fields of every laser observation hold the same constant
robot-frame mounting offset (x=0.205, yaw=0), so `scan_points` yields points
in each scan's own ROBOT frame (concatenated; split them with obs_idx), each
labeled with the room type the robot was in -- suitable for per-scan
encoding/classification, not for stitching a global map. The localized,
semantically labeled 2D map of each home is the separate `geomap` table
(built by the dataset authors), which IS in a consistent home frame.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

OBS_FIELDS = ("id", "home_id", "room_id", "home_session_id",
              "home_subsession_id", "time_stamp", "sensor_pose_x",
              "sensor_pose_y", "sensor_pose_yaw", "laser_aperture",
              "laser_max_range", "laser_num_of_scans")


class RobotAtHome:
    def __init__(self, db_path: str | Path = "data/robotathome/rh.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"{db_path} not found -- run scripts/download_robotathome.py")
        self.con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self.homes = dict(self.con.execute("SELECT id, name FROM rh_homes"))
        self.room_types = dict(
            self.con.execute("SELECT id, name FROM rh_room_types"))
        self.rooms = {
            rid: {"name": name, "home_id": hid, "room_type_id": rt}
            for rid, name, hid, rt in self.con.execute(
                "SELECT id, name, home_id, room_type_id FROM rh_rooms")}
        self._room_type_of = np.zeros(max(self.rooms) + 1, dtype=np.int32)
        for rid, r in self.rooms.items():
            self._room_type_of[rid] = r["room_type_id"]

    def _home_id(self, home):
        if isinstance(home, str):
            return {v: k for k, v in self.homes.items()}[home]
        return home

    def laser_observations(self, home=None, room_id=None, session=None):
        """Laser observations (structured array, time-ordered), optionally
        filtered by home (name or id), room id, or home_session id."""
        where, params = [], []
        if home is not None:
            where.append("home_id = ?")
            params.append(self._home_id(home))
        if room_id is not None:
            where.append("room_id = ?")
            params.append(room_id)
        if session is not None:
            where.append("home_session_id = ?")
            params.append(session)
        q = f"SELECT {', '.join(OBS_FIELDS)} FROM rh_lsrscan"
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY time_stamp"
        rows = self.con.execute(q, params).fetchall()
        dtype = [(f, np.int64 if f in ("id", "home_id", "room_id",
                                       "home_session_id",
                                       "home_subsession_id", "time_stamp",
                                       "laser_num_of_scans") else np.float64)
                 for f in OBS_FIELDS]
        return np.array(rows, dtype=dtype)

    def ranges(self, obs_id):
        """Raw ranges (n_beams,) and validity mask for one observation."""
        rows = self.con.execute(
            "SELECT shot_id, scan, valid_scan FROM rh_lsrscan_scans "
            "WHERE sensor_observation_id = ? ORDER BY shot_id",
            (int(obs_id),)).fetchall()
        arr = np.asarray(rows, dtype=np.float64)
        return arr[:, 1], arr[:, 2] > 0

    def scan_points(self, obs_ids, min_range=0.05):
        """Robot-frame 2D points for many observations at once (the db has
        no localized trajectory; see module docstring).

        Returns (points (N,2), room_type_labels (N,), obs_index (N,)) where
        obs_index maps each point back to its row in obs_ids -- use it to
        split the concatenated points into individual scans. Invalid beams
        and ranges below min_range are dropped.
        """
        obs_ids = np.atleast_1d(np.asarray(obs_ids, dtype=np.int64))
        marks = ",".join("?" * len(obs_ids))
        obs = {int(r[0]): r for r in self.con.execute(
            f"SELECT {', '.join(OBS_FIELDS)} FROM rh_lsrscan "
            f"WHERE id IN ({marks})", obs_ids.tolist())}
        rows = self.con.execute(
            "SELECT sensor_observation_id, shot_id, scan, valid_scan "
            f"FROM rh_lsrscan_scans WHERE sensor_observation_id IN ({marks}) "
            "ORDER BY sensor_observation_id, shot_id",
            obs_ids.tolist()).fetchall()
        raw = np.asarray(rows, dtype=np.float64)
        order = {int(o): k for k, o in enumerate(obs_ids)}

        pts, labels, obs_idx = [], [], []
        for oid in obs_ids:
            o = obs[int(oid)]
            sel = raw[raw[:, 0] == oid]
            r, valid = sel[:, 2], sel[:, 3] > 0
            n, aperture = int(o[11]), float(o[9])
            beam = np.linspace(-aperture / 2, aperture / 2, n)
            keep = valid & (r > min_range) & (r < float(o[10]))
            ang = beam[sel[:, 1].astype(int)] + float(o[8])   # + yaw
            x = float(o[6]) + r * np.cos(ang)
            y = float(o[7]) + r * np.sin(ang)
            pts.append(np.column_stack([x[keep], y[keep]]))
            labels.append(np.full(keep.sum(),
                                  self._room_type_of[int(o[2])], np.int32))
            obs_idx.append(np.full(keep.sum(), order[int(oid)], np.int32))
        return (np.concatenate(pts).astype(np.float32),
                np.concatenate(labels), np.concatenate(obs_idx))

    def geomap(self, home=None, room_id=None):
        """Prebuilt 2D geometric map points with room-type labels.

        Returns (points (N,2), room_type_labels (N,)).
        """
        where, params = [], []
        if home is not None:
            where.append("home_id = ?")
            params.append(self._home_id(home))
        if room_id is not None:
            where.append("room_id = ?")
            params.append(room_id)
        q = "SELECT x, y, room_id FROM rh_twodgeomap"
        if where:
            q += " WHERE " + " AND ".join(where)
        arr = np.asarray(self.con.execute(q, params).fetchall(),
                         dtype=np.float64)
        if not len(arr):
            return (np.empty((0, 2), np.float32), np.empty(0, np.int32))
        return (arr[:, :2].astype(np.float32),
                self._room_type_of[arr[:, 2].astype(int)])

    def room_label_name(self, label_id):
        return self.room_types.get(int(label_id), f"label {label_id}")
