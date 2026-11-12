"""pytest tools_local/policyik_piper/test_piper_tasks.py  (from simulation/, env_local.sh sourced; no simulator)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_piper_tasks as M  # noqa: E402
from piper_kin import PiperKinematics, quat_angle  # noqa: E402
from urdf_fk import rotmat_to_quat_wxyz  # noqa: E402

import policyik_sim.splits as S  # noqa: E402


@pytest.fixture(scope="module")
def kin():
    return PiperKinematics()


@pytest.fixture(scope="module")
def files(kin):
    return M.build(40_000, kin)


def _rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def test_limits_are_the_arms(kin):
    # joint2's lower and joint3's upper limit are exactly 0 on this arm
    assert kin.limits.shape == (6, 2) and kin.limits[1, 0] == 0.0 and kin.limits[2, 1] == 0.0
    np.testing.assert_allclose(kin.limits[0], [-2.6179938, 2.6179938])


def test_joint1_turns_the_whole_arm_about_base_z(kin):
    q = np.random.default_rng(0).uniform(kin.limits[:, 0], kin.limits[:, 1], (50, 6))
    q0 = q.copy()
    q0[:, 0] = 0.0
    T, T0 = kin.link_poses(q, ["link6"])["link6"], kin.link_poses(q0, ["link6"])["link6"]
    for a, t, t0 in zip(q[:, 0], T, T0):
        np.testing.assert_allclose(t[:3, 3], _rz(a) @ t0[:3, 3], atol=1e-12)
        np.testing.assert_allclose(t[:3, :3], _rz(a) @ t0[:3, :3], atol=1e-12)


def test_flange_is_link6_and_the_gripper_base_sits_4_5_mm_out(kin):
    q = np.random.default_rng(1).uniform(kin.limits[:, 0], kin.limits[:, 1], (20, 6))
    T = kin.link_poses(q, ["link6", "flange_link", "gripper_base"])
    np.testing.assert_allclose(T["flange_link"], T["link6"], atol=1e-15)
    np.testing.assert_allclose(T["gripper_base"][:, :3, 3], T["link6"][:, :3, 3] + 0.0045 * T["link6"][:, :3, 2], atol=1e-12)


def test_quat_angle_resolves_what_the_arccos_form_cannot():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(2000, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q32 = q.astype(np.float32).astype(float)
    arccos = 2 * np.arccos(np.abs(np.sum(q32 * q, -1)).clip(0, 1))
    assert np.degrees(arccos).max() > 0.01            # the franka gate's 0.01 deg is below this metric's noise
    assert np.degrees(quat_angle(q32, q)).max() < 1e-4
    half = np.radians(0.02) / 2                        # a real 0.02 deg rotation about x is read as 0.02 deg
    d = np.array([[np.cos(half), np.sin(half), 0, 0]])
    assert np.degrees(quat_angle(d, np.array([[1.0, 0, 0, 0]])))[0] == pytest.approx(0.02, rel=1e-6)


def test_floor_contact(kin):
    assert not kin.floor_contact(np.zeros((1, 6)))[0]
    q = np.random.default_rng(2).uniform(kin.limits[:, 0], kin.limits[:, 1], (5000, 6))
    low, fc = kin.lowest_point(q), kin.floor_contact(q)
    assert fc.any() and (~fc).any() and np.array_equal(fc, low < 0)
    # the flange origin is a point of the arm: wherever it is under the floor, a hull vertex is too
    assert fc[kin.fk(q)[0][:, 2] < -0.02].all()
    # link origins of the moving links are inside their own meshes' hulls to within the mesh size
    z6 = kin.link_poses(q, ["link6"])["link6"][:, 2, 3]
    assert (low <= z6 + 1e-9).mean() > 0.99


def test_sets_are_disjoint_valid_and_inside_the_workspace(files, kin):
    src, ev, tune = files["source_set"], files["eval_tasks"], files["tune_tasks"]
    assert len(ev["q_start"]) == 4120 and len(tune["q_start"]) == 1024
    assert not src["self_collision"].any() and not bool(src["self_collision_screened"])
    key = lambda a: {r.tobytes() for r in a}
    test_q = key(ev["q_start"]) | key(files["eval_goal_q"]["q_goal"])
    assert len(test_q) == 2 * 4120
    for other in (files["train_start_pool"]["q"], files["train_goal_pool"]["q"], tune["q_start"], tune["q_goal"]):
        assert not key(other) & test_q
    assert S.in_workspace(ev["goal_pos"].astype(float)).all() and S.in_workspace(tune["goal_pos"].astype(float)).all()
    for q in (ev["q_start"], files["eval_goal_q"]["q_goal"], tune["q_start"], tune["q_goal"]):
        q = q.astype(float)
        assert not kin.floor_contact(q).any()
        assert (q >= kin.limits[:, 0] - 1e-6).all() and (q <= kin.limits[:, 1] + 1e-6).all()
    # goals are the poses of their own goal configurations
    p, o = kin.fk(files["eval_goal_q"]["q_goal"].astype(float))
    assert np.abs(p - ev["goal_pos"]).max() < 1e-6 and np.degrees(quat_angle(o, ev["goal_quat"])).max() < 1e-3


def test_generation_is_repeatable(files, kin):
    assert M.array_hashes(M.build(40_000, kin)) == M.array_hashes(files)


def test_workspace_constants_are_patched_in(files):
    assert (S.R_MAX, S.Z_MIN, S.Z_MAX, S.R_KEEPOUT, S.Z_KEEPOUT) == tuple(M.WORKSPACE.values())
