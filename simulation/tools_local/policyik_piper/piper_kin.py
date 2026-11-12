"""Simulator-free AgileX Piper kinematics and floor screening for the Piper task sets (plan S0).

* Batched forward kinematics from the arm's URDF (numpy float64). The URDF and its collision meshes are read in
  place from the user's Intelligence_AOP checkout (vendor files, not copied into this repository); the path is
  configurable with ``PIPER_URDF``.
* Floor screening from the URDF's own collision meshes: a link touches the floor iff a vertex of its convex hull
  is below z = 0 (exact for the hull; the lowest point of a mesh is a hull vertex). No sphere model.
* ``quat_angle``: the angle between two rotations from the vector part of their relative quaternion. The
  ``2 * arccos(|dot|)`` form used by ``policyik_franka/check_fk_sim.py`` cannot resolve its own 0.01 deg threshold
  on float32 quaternions (2026-11-11: identical rotations, one side stored as float32, read up to 0.032 deg).

Self-collision is not screened here (plan section 6, choice 4): it is checked against the simulator in S1.
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "policyik_franka"))
from urdf_fk import _axis_angle, _rpy, rotmat_to_quat_wxyz  # noqa: E402  (robot-independent helpers)

URDF = os.environ.get("PIPER_URDF", "/home/hzm/code/Intelligence_AOP/sim/assets/piper_agx/piper_right.urdf")
ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
EE_LINK = "link6"          # the flange (flange_link is link6 through a zero fixed joint)
BASE_LINK = "base_link"
FINGER_POS = 0.0           # both prismatic finger joints at their URDF zero
FLOOR_IGNORE = ("base_link", "link1")   # they stand on the mounting plane z = 0


def _origin(el) -> np.ndarray:
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = _rpy(*[float(v) for v in el.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


def _stl_vertices(path: str) -> np.ndarray:
    """Vertices (N,3) of a binary STL."""
    raw = open(path, "rb").read()
    n = int(np.frombuffer(raw, "<u4", 1, 80)[0])
    if len(raw) != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL ({len(raw)} bytes for {n} triangles)")
    tri = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
    return tri["v"].reshape(-1, 3).astype(float)


class PiperKinematics:
    def __init__(self, urdf: str = URDF):
        self.urdf = urdf
        root = ET.parse(urdf).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            ax, lim = j.find("axis"), j.find("limit")
            self.joints[j.get("name")] = dict(
                type=j.get("type"), parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                T=_origin(j.find("origin")),
                axis=np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3),
                lower=float(lim.get("lower")) if lim is not None else None,
                upper=float(lim.get("upper")) if lim is not None else None)
        self.child_of = {v["child"]: k for k, v in self.joints.items()}
        self.limits = np.array([[self.joints[n]["lower"], self.joints[n]["upper"]] for n in ARM_JOINTS])
        # convex-hull vertices of every collision mesh, in its link's frame
        self.hull = {}
        for link in root.findall("link"):
            pts = []
            for c in link.findall("collision"):
                mesh = c.find("geometry").find("mesh")
                if mesh is None:
                    raise ValueError(f"{link.get('name')}: only mesh collisions are handled")
                s = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
                v = _stl_vertices(mesh.get("filename")) * s
                T = _origin(c.find("origin"))
                pts.append(v @ T[:3, :3].T + T[:3, 3])
            if pts:
                v = np.concatenate(pts)
                self.hull[link.get("name")] = v[ConvexHull(v).vertices]

    def link_poses(self, q: np.ndarray, links) -> dict:
        """q (B,6) arm joints (fingers at FINGER_POS) -> {link: (B,4,4)} in the base_link frame."""
        q = np.atleast_2d(np.asarray(q, float))
        B = len(q)
        vals = {n: q[:, i] for i, n in enumerate(ARM_JOINTS)}
        cache = {BASE_LINK: np.broadcast_to(np.eye(4), (B, 4, 4)).copy()}

        def pose(link):
            if link in cache:
                return cache[link]
            name = self.child_of[link]
            j = self.joints[name]
            Tp = pose(j["parent"]) @ j["T"][None]
            if j["type"] in ("revolute", "prismatic"):
                v = vals.get(name, np.full(B, FINGER_POS))
                M = np.broadcast_to(np.eye(4), (B, 4, 4)).copy()
                if j["type"] == "revolute":
                    M[:, :3, :3] = _axis_angle(j["axis"], v)
                else:
                    M[:, :3, 3] = v[:, None] * j["axis"][None]
                Tp = Tp @ M
            elif j["type"] != "fixed":
                raise ValueError(f"joint {name}: type {j['type']} not handled")
            cache[link] = Tp
            return Tp

        return {l: pose(l) for l in links}

    def fk(self, q: np.ndarray, link: str = EE_LINK) -> tuple[np.ndarray, np.ndarray]:
        """-> position (B,3), quaternion wxyz (B,4) of ``link`` in the base_link frame."""
        T = self.link_poses(q, [link])[link]
        return T[:, :3, 3], rotmat_to_quat_wxyz(T[:, :3, :3])

    def lowest_point(self, q: np.ndarray, chunk: int = 20000) -> np.ndarray:
        """(B,) z of the lowest collision-hull vertex over all links except FLOOR_IGNORE."""
        q = np.atleast_2d(q)
        links = [l for l in self.hull if l not in FLOOR_IGNORE]
        low = np.full(len(q), np.inf)
        for s in range(0, len(q), chunk):
            T = self.link_poses(q[s:s + chunk], links)
            for l in links:
                z = np.einsum("bj,nj->bn", T[l][:, 2, :3], self.hull[l]) + T[l][:, 2, 3:4]
                low[s:s + chunk] = np.minimum(low[s:s + chunk], z.min(1))
        return low

    def floor_contact(self, q: np.ndarray) -> np.ndarray:
        return self.lowest_point(q) < 0.0


def quat_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (rad) between unit quaternions a, b (wxyz, (B,4)); well conditioned near zero."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    w = np.sum(a * b, -1)
    v = a[:, :1] * b[:, 1:] - b[:, :1] * a[:, 1:] - np.cross(a[:, 1:], b[:, 1:])
    return 2.0 * np.arctan2(np.linalg.norm(v, axis=1), np.abs(w))
