import numpy as np

import pydrake
from pydrake.all import (
    RigidTransform, PiecewiseQuaternionSlerp, RotationMatrix
)

"""
Throwing related trajectory stuff
"""
def orientation_slerp(r_A, r_B):
    # assume t is between 0 and 1; responsibility of caller to scale time
    traj = PiecewiseQuaternionSlerp()
    traj.Append(0.0, r_A)
    traj.Append(1.0, r_B)
    return traj

def interpolatePosesLinear(T_world_poseA, T_world_poseB, t):
    # assume t is between 0 and 1; responsibility of caller to scale time
    p_A, r_A = T_world_poseA.translation(), T_world_poseA.rotation()
    p_B, r_B = T_world_poseB.translation(), T_world_poseB.rotation()

    # rotation is a clean slerp
    r_slerp = orientation_slerp(r_A, r_B)
    r_curr = RotationMatrix(r_slerp.value(t))

    # position is a straight line
    p_curr = p_A + t * (p_B - p_A)

    T_world_currGripper = RigidTransform(r_curr, p_curr)
    return T_world_currGripper

def interpolatePosesArcMotion(T_world_poseA, T_world_poseB, t):
    # assume t is between 0 and 1; responsibility of caller to scale time
    p_A, r_A = T_world_poseA.translation(), T_world_poseA.rotation()
    p_B, r_B = T_world_poseB.translation(), T_world_poseB.rotation()

    # rotation is a clean slerp
    r_slerp = orientation_slerp(r_A, r_B)
    r_curr = RotationMatrix(r_slerp.value(t))

    # position is an arc that isn't necessarily axis aligned
    arc_radius = p_B[2] - p_A[2]
    phi = np.arctan2(p_B[1] - p_A[1], p_B[0] - p_A[0]) #xy direction heading
    theta = (t - 1) * np.pi / 2 # -90 to 0 degrees
    p_curr = p_A + np.array([
        arc_radius * np.cos(theta) * np.cos(phi),
        arc_radius * np.cos(theta) * np.sin(phi),
        arc_radius * np.sin(theta) + arc_radius
    ])

    T_world_currGripper = RigidTransform(r_curr, p_curr)
    return T_world_currGripper

def interpolatePosesArcMotion_pdot(T_world_poseA, T_world_poseB, t):
    # assume t is between 0 and 1; responsibility of caller to scale time
    p_A, r_A = T_world_poseA.translation(), T_world_poseA.rotation()
    p_B, r_B = T_world_poseB.translation(), T_world_poseB.rotation()

    # position is an arc that isn't necessarily axis aligned
    arc_radius = p_B[2] - p_A[2]
    phi = np.arctan2(p_B[1] - p_A[1], p_B[0] - p_A[0]) #xy direction heading
    theta = (t - 1) * np.pi / 2 # -90 to 0 degrees
    pdot_curr = (np.pi / 2) * np.array([
      - arc_radius * np.sin(theta) * np.cos(phi),
      - arc_radius * np.sin(theta) * np.sin(phi),
        arc_radius * np.cos(theta)
    ])

    return pdot_curr


def interpolate_joint_angle(ja1, ja2, time_interval, num_samples, include_end=False):
    # constant joint velocity over the time interval
    # naive - let's try not to deal with wrap around and joint velocity limits
    delta = ja2 - ja1
    joint_velocities = delta / time_interval

    ja_list = []
    t_lst = np.linspace(0, time_interval, num_samples, endpoint=include_end)
    for t in t_lst:
        ja_list.append(ja1 + t * joint_velocities)

    return t_lst, ja_list


def get_launch_speed_required(theta, x, y, g=9.81):
    """
    Assuming we launch from (0, 0) at theta (radians),
    how fast do we need to launch to hit (x, y)?
    """
    assert 0 < theta < np.pi
    assert 0 < x
    assert y < np.tan(theta) * x

    time_to_target = np.sqrt(2.0 / g * (np.tan(theta) * x - y))
    speed_required = x / (np.cos(theta) * time_to_target)

    return speed_required