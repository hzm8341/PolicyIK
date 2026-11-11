"""Independent (simulator-free) Franka Panda kinematics and collision screening for the P2 task sets.

* Batched forward kinematics parsed from a URDF (revolute/prismatic/fixed joints, xyz + rpy origins), numpy float64.
* Sphere-based self-collision and floor screening with the curobo Franka sphere model and its self-collision
  ignore pairs and buffers. The curobo files are read in place (not redistributed); their path is configurable.

The simulator is not used here on purpose: the sim-vs-URDF FK check in check_fk_sim.py compares the two.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import yaml

CUROBO_CONTENT = os.environ.get("CUROBO_CONTENT", "/mnt/data/GitHub/UniVTAC/third_party/curobo/src/curobo/content")
URDF = os.path.join(CUROBO_CONTENT, "assets/robot/franka_description/franka_panda.urdf")
ROBOT_YML = os.path.join(CUROBO_CONTENT, "configs/robot/franka.yml")
SPHERES_YML = os.path.join(CUROBO_CONTENT, "configs/robot/spheres/franka_mesh.yml")
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_OPEN = 0.04


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, ang):
    """(B,) angles about a unit axis -> (B,3,3)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s, c = np.sin(ang)[:, None, None], np.cos(ang)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


class FrankaKinematics:
    def __init__(self, urdf: str = URDF):
        root = ET.parse(urdf).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            lim = j.find("limit")
            T = np.eye(4)
            T[:3, :3] = _rpy(*rpy)
            T[:3, 3] = xyz
            self.joints[j.get("name")] = dict(
                type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                T=T, axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3),
                lower=float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                upper=float(lim.get("upper")) if lim is not None and lim.get("upper") else None)
        self.child_of = {v["child"]: k for k, v in self.joints.items()}
        self.limits = np.array([[self.joints[n]["lower"], self.joints[n]["upper"]] for n in ARM_JOINTS])

    def link_poses(self, q: np.ndarray, links=None, base: str = "panda_link0") -> dict:
        """q (B,7) arm joints (fingers fixed open) -> {link: (B,4,4)} in the base-link frame."""
        q = np.atleast_2d(np.asarray(q, float))
        B = len(q)
        vals = {n: q[:, i] for i, n in enumerate(ARM_JOINTS)}
        vals.update({"panda_finger_joint1": np.full(B, FINGER_OPEN), "panda_finger_joint2": np.full(B, FINGER_OPEN)})
        cache = {base: np.broadcast_to(np.eye(4), (B, 4, 4)).copy()}

        def pose(link):
            if link in cache:
                return cache[link]
            j = self.joints[self.child_of[link]]
            Tp = pose(j["parent"]) @ j["T"][None]
            if j["type"] in ("revolute", "continuous"):
                M = np.broadcast_to(np.eye(4), (B, 4, 4)).copy()
                M[:, :3, :3] = _axis_angle(j["axis"], vals[self.child_of[link]])
                Tp = Tp @ M
            elif j["type"] == "prismatic":
                M = np.broadcast_to(np.eye(4), (B, 4, 4)).copy()
                M[:, :3, 3] = vals[self.child_of[link]][:, None] * j["axis"][None]
                Tp = Tp @ M
            cache[link] = Tp
            return Tp

        links = links or [j["child"] for j in self.joints.values()]
        return {l: pose(l) for l in links}

    def fk(self, q: np.ndarray, link: str = "panda_hand") -> tuple[np.ndarray, np.ndarray]:
        """-> position (B,3), quaternion wxyz (B,4) of `link` in the panda_link0 frame."""
        T = self.link_poses(q, [link])[link]
        return T[:, :3, 3], rotmat_to_quat_wxyz(T[:, :3, :3])


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    B = R.shape[0]
    q = np.empty((B, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for b in range(B):
        m = R[b]
        if tr[b] > 0:
            s = 2 * np.sqrt(tr[b] + 1)
            q[b] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        else:
            i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
            j, k = (i + 1) % 3, (i + 2) % 3
            s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
            v = np.zeros(3)
            v[i] = 0.25 * s
            v[j] = (m[j, i] + m[i, j]) / s
            v[k] = (m[k, i] + m[i, k]) / s
            q[b] = [(m[k, j] - m[j, k]) / s, *v]
    q *= np.where(q[:, :1] < 0, -1.0, 1.0)          # w >= 0 (unique)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


class SphereCollision:
    """curobo sphere model: self-collision with ignore pairs and per-link buffers, and floor contact (z < 0)."""

    def __init__(self, kin: FrankaKinematics):
        self.kin = kin
        sph = yaml.safe_load(open(SPHERES_YML))["collision_spheres"]
        rc = yaml.safe_load(open(ROBOT_YML))["robot_cfg"]["kinematics"]
        self.links = [l for l in rc["collision_link_names"] if l in sph]
        self.ignore = {(a, b) for a, bs in rc["self_collision_ignore"].items() for b in bs}
        self.ignore |= {(b, a) for a, b in self.ignore}
        buf = rc.get("self_collision_buffer", {})
        self.centers, self.radii, self.owner = [], [], []
        for l in self.links:
            for s in sph[l]:
                if s["radius"] <= 0:
                    continue
                self.centers.append(s["center"])
                self.radii.append(s["radius"] + buf.get(l, 0.0))
                self.owner.append(l)
        self.centers = np.array(self.centers)
        self.radii = np.array(self.radii)
        self.raw_radii = np.array([s["radius"] for l in self.links for s in sph[l] if s["radius"] > 0])
        own = np.array(self.owner)
        n = len(own)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if own[i] != own[j] and (own[i], own[j]) not in self.ignore]
        self.pi, self.pj = np.array(pairs).T

    def world_spheres(self, q):
        T = self.kin.link_poses(q, self.links)
        c = np.empty((len(q), len(self.owner), 3))
        for k, (l, p) in enumerate(zip(self.owner, self.centers)):
            c[:, k] = T[l][:, :3, :3] @ p + T[l][:, :3, 3]
        return c

    def check(self, q, floor_ignore=("panda_link0", "panda_link1"), chunk=4096):
        """-> (self_collision (B,), floor_contact (B,)) booleans."""
        q = np.atleast_2d(q)
        sc, fc = np.zeros(len(q), bool), np.zeros(len(q), bool)
        fmask = ~np.isin(np.array(self.owner), floor_ignore)
        for s in range(0, len(q), chunk):
            c = self.world_spheres(q[s:s + chunk])
            d = np.linalg.norm(c[:, self.pi] - c[:, self.pj], axis=-1)
            sc[s:s + chunk] = (d < self.radii[self.pi] + self.radii[self.pj]).any(1)
            fc[s:s + chunk] = ((c[:, fmask, 2] - self.raw_radii[fmask]) < 0.0).any(1)
        return sc, fc
