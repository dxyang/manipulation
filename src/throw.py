import numpy as np

from pydrake.all import (
    PiecewisePolynomial, RigidTransform, RotationMatrix
)

from .ik import (
    create_q_knots,
    pose_to_jointangles,
)
from .trajectory import (
    get_launch_speed_required,
    interpolate_joint_angle,
    interpolatePosesLinear,
    interpolatePosesArcMotion,
    interpolatePosesArcMotion_pdot,
)
from .drake_helpers import visualize_transform

def plan_pickup(
    T_world_robotInitial,
    T_world_gripperObject,
    t_goToObj = 1.0,
    t_holdObj = 0.5,
    t_goToPreobj = 1.0,
):
    # returns timings, poses, and q knots necessary for grabbing the object and returning to the original position

    total_pickup_time = t_goToObj + t_holdObj + t_goToPreobj

    # Create pose trajectory
    t_lst = np.linspace(0, total_pickup_time, 100, endpoint=False)
    pose_lst = []
    for t in t_lst:
        if t < t_goToObj + t_holdObj:
            pose = interpolatePosesLinear(T_world_robotInitial, T_world_gripperObject, min(t / t_goToObj, 1.0))
        elif t < t_goToObj + t_holdObj + t_goToPreobj:
            pose = interpolatePosesLinear(T_world_gripperObject, T_world_robotInitial, (t - (t_goToObj + t_holdObj)) / t_goToPreobj)
        else:
            break
        pose_lst.append(pose)

    q_knots = create_q_knots(pose_lst)

    return t_lst, q_knots, total_pickup_time

def add_go_to_ja_via_jointinterpolation(
    ja1,
    ja2,
    t_start,
    t_lst,
    q_knots,
    time_interval_s=10.0,
    hold_time_s=0.0,
    num_samples = 100,
    include_end=False,
):
    # print(f"--------")
    # print(f"ja1: {ja1.squeeze()}")
    # print(f"ja2: {ja2.squeeze()}")

    ts, jas = interpolate_joint_angle(ja1, ja2, time_interval_s - hold_time_s, num_samples, include_end=include_end)

    # add in the hold time at the end position to make sure we've stabilized
    if not np.isclose(hold_time_s, 0.0):
        num_hold_samples = 10
        hold_ts = np.linspace(time_interval_s - hold_time_s, time_interval_s, num_hold_samples, endpoint=False)
        hold_jas = [ja2 for _ in range(num_hold_samples)]
        ts = np.append(ts, hold_ts)
        jas.extend(hold_jas)

    # t start should be the next value in t_list
    ts += t_start
    t_lst = np.append(t_lst, ts)
    q_knots.extend(jas)

    return t_lst, q_knots

def add_go_to_pose_via_jointinterpolation(
    T_world_robotInitial,
    T_world_robotFinal,
    t_start,
    t_lst,
    q_knots,
    time_interval_s=10.0,
    hold_time_s=0.0,
    do_magic=False
):
    ja1 = pose_to_jointangles(T_world_robotInitial)
    ja2 = pose_to_jointangles(T_world_robotFinal)
    if do_magic:
        ja2 = np.copy(ja1)
        ja2[3] += np.pi / 2
        ja2[5] += np.pi / 2

    return add_go_to_ja_via_jointinterpolation(
        ja1,
        ja2,
        t_start=t_start,
        t_lst=t_lst,
        q_knots=q_knots,
        time_interval_s=time_interval_s,
        hold_time_s=hold_time_s
    )


def plan_prethrow_pose(
    T_world_robotInitial,
    p_world_target, # np.array of shape (3,)
    gripper_to_object_dist,
    throw_height=0.5, # meters
    prethrow_height=0.2,
    prethrow_radius=0.4,
    throw_angle=np.pi/4.0,
    meshcat=None,
    throw_speed_adjustment_factor=1.0,
):
    """
    only works with the "back portion" of the clutter station until we figure out how to move the bins around
    motion moves along an arc from a "pre throw" to a "throw" position
    """
    theta = 1.0 * np.arctan2(p_world_target[1], p_world_target[0])
    print(f"theta={theta}")

    T_world_prethrow = RigidTransform(
        p=np.array([
            prethrow_radius * np.cos(theta),
            prethrow_radius * np.sin(theta),
            prethrow_height
        ]),
        R=RotationMatrix.MakeXRotation(-np.pi/2).multiply(
            RotationMatrix.MakeYRotation((theta - np.pi / 2))
        )
    )

    throw_radius = throw_height - prethrow_height
    T_world_throw = RigidTransform(
        p=T_world_prethrow.translation() + np.array([
            throw_radius * np.cos(theta),
            throw_radius * np.sin(theta),
            throw_height - prethrow_height
        ]),
        R=RotationMatrix.MakeXRotation(-np.pi/2).multiply(
            RotationMatrix.MakeYRotation((theta - np.pi / 2)).multiply(
                RotationMatrix.MakeXRotation(-np.pi/2)
            )
        )
    )

    if meshcat:
        visualize_transform(meshcat, "T_world_prethrow", T_world_prethrow)
        visualize_transform(meshcat, "T_world_throw", T_world_throw)

    p_world_object_at_launch = interpolatePosesArcMotion(
        T_world_prethrow, T_world_throw,
        t=throw_angle / (np.pi / 2.)
    ).translation() + np.array([0, 0, -gripper_to_object_dist])
    pdot_world_launch = interpolatePosesArcMotion_pdot(
        T_world_prethrow, T_world_throw,
        t=throw_angle / (np.pi / 2.)
    )
    launch_speed_base = np.linalg.norm(pdot_world_launch)
    launch_speed_required = get_launch_speed_required(
        theta=throw_angle,
        x=np.linalg.norm(p_world_target[:2]) - np.linalg.norm(p_world_object_at_launch[:2]),
        y=p_world_target[2] - p_world_object_at_launch[2]
    )
    total_throw_time = launch_speed_base / launch_speed_required / throw_speed_adjustment_factor
    # print(f"p_world_object_at_launch={p_world_object_at_launch}")
    # print(f"target={p_world_target}")
    # print(f"dx={np.linalg.norm(p_world_target[:2]) - np.linalg.norm(p_world_object_at_launch[:2])}")
    # print(f"dy={p_world_target[2] - p_world_object_at_launch[2]}")
    # print(f"pdot_world_launch={pdot_world_launch}")
    # print(f"total_throw_time={total_throw_time}")

    return T_world_prethrow, T_world_throw

def plan_throw(
    T_world_robotInitial,
    T_world_gripperObject,
    p_world_target, # np.array of shape (3,)
    gripper_to_object_dist,
    throw_height=0.5, # meters
    prethrow_height=0.2,
    prethrow_radius=0.4,
    throw_angle=np.pi/4.0,
    meshcat=None,
    throw_speed_adjustment_factor=1.0,
):
    """
    only works with the "back portion" of the clutter station until we figure out how to move the bins around
    motion moves along an arc from a "pre throw" to a "throw" position
    """
    theta = 1.0 * np.arctan2(p_world_target[1], p_world_target[0])
    print(f"theta={theta}")

    T_world_prethrow = RigidTransform(
        p=np.array([
            prethrow_radius * np.cos(theta),
            prethrow_radius * np.sin(theta),
            prethrow_height
        ]),
        R=RotationMatrix.MakeXRotation(-np.pi/2).multiply(
            RotationMatrix.MakeYRotation((theta - np.pi / 2))
        )
    )

    throw_radius = throw_height - prethrow_height
    T_world_throw = RigidTransform(
        p=T_world_prethrow.translation() + np.array([
            throw_radius * np.cos(theta),
            throw_radius * np.sin(theta),
            throw_height - prethrow_height
        ]),
        R=RotationMatrix.MakeXRotation(-np.pi/2).multiply(
            RotationMatrix.MakeYRotation((theta - np.pi / 2)).multiply(
                RotationMatrix.MakeXRotation(-np.pi/2)
            )
        )
    )

    if meshcat:
        visualize_transform(meshcat, "T_world_prethrow", T_world_prethrow)
        visualize_transform(meshcat, "T_world_throw", T_world_throw)

    p_world_object_at_launch = interpolatePosesArcMotion(
        T_world_prethrow, T_world_throw,
        t=throw_angle / (np.pi / 2.)
    ).translation() + np.array([0, 0, -gripper_to_object_dist])
    pdot_world_launch = interpolatePosesArcMotion_pdot(
        T_world_prethrow, T_world_throw,
        t=throw_angle / (np.pi / 2.)
    )
    launch_speed_base = np.linalg.norm(pdot_world_launch)
    launch_speed_required = get_launch_speed_required(
        theta=throw_angle,
        x=np.linalg.norm(p_world_target[:2]) - np.linalg.norm(p_world_object_at_launch[:2]),
        y=p_world_target[2] - p_world_object_at_launch[2]
    )
    total_throw_time = launch_speed_base / launch_speed_required / throw_speed_adjustment_factor
    print(f"p_world_object_at_launch={p_world_object_at_launch}")
    print(f"target={p_world_target}")
    print(f"dx={np.linalg.norm(p_world_target[:2]) - np.linalg.norm(p_world_object_at_launch[:2])}")
    print(f"dy={p_world_target[2] - p_world_object_at_launch[2]}")
    print(f"pdot_world_launch={pdot_world_launch}")
    print(f"total_throw_time={total_throw_time}")

    T_world_hackyWayPoint = RigidTransform(
        p=[-.6, -0.0, 0.6],
        R=RotationMatrix.MakeXRotation(-np.pi/2.0), #R_WORLD_PRETHROW, #RotationMatrix.MakeXRotation(-np.pi/2.0),
    )

    # event timings (not all are relevant to pose and gripper)
    # initial pose => prethrow => throw => yays
    t_goToObj = 1.0
    t_holdObj = 0.5
    t_goToPreobj = 1.0
    t_goToWaypoint = 1.0
    t_goToPrethrow = 1.0
    t_goToRelease = total_throw_time * throw_angle / (np.pi / 2.)
    t_goToThrowEnd = total_throw_time * (1 - throw_angle / (np.pi / 2.))
    t_throwEndHold = 3.0
    ts = np.array([
        t_goToObj,
        t_holdObj,
        t_goToPreobj,
        t_goToWaypoint,
        t_goToPrethrow,
        t_goToRelease,
        t_goToThrowEnd,
        t_throwEndHold
    ])
    cum_pose_ts = np.cumsum(ts)
    print(cum_pose_ts)

    # Create pose trajectory
    t_lst = np.linspace(0, cum_pose_ts[-1], 1000)
    pose_lst = []
    for t in t_lst:
        if t < cum_pose_ts[1]:
            pose = interpolatePosesLinear(T_world_robotInitial, T_world_gripperObject, min(t / ts[0], 1.0))
        elif t < cum_pose_ts[2]:
            pose = interpolatePosesLinear(T_world_gripperObject, T_world_robotInitial, (t - cum_pose_ts[1]) / ts[2])
        elif t < cum_pose_ts[3]:
            pose = interpolatePosesLinear(T_world_robotInitial, T_world_hackyWayPoint, (t - cum_pose_ts[2]) / ts[3])
        elif t <= cum_pose_ts[4]:
            pose = interpolatePosesLinear(T_world_hackyWayPoint, T_world_prethrow, (t - cum_pose_ts[3]) / ts[4])
        else:
            pose = interpolatePosesArcMotion(T_world_prethrow, T_world_throw, min((t - cum_pose_ts[4]) / (ts[5] + ts[6]), 1.0))
        pose_lst.append(pose)

    # Create gripper trajectory.
    gripper_times_lst = np.array([
        0.,
        t_goToObj,
        t_holdObj,
        t_goToPreobj,
        t_goToWaypoint,
        t_goToPrethrow,
        t_goToRelease,
        t_goToThrowEnd,
        t_throwEndHold
    ])
    gripper_cumulative_times_lst = np.cumsum(gripper_times_lst)
    GRIPPER_OPEN = 0.5
    GRIPPER_CLOSED = 0.0
    gripper_knots = np.array([
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        GRIPPER_OPEN,
        GRIPPER_CLOSED
    ]).reshape(1,gripper_times_lst.shape[0])
    g_traj = PiecewisePolynomial.FirstOrderHold(gripper_cumulative_times_lst, gripper_knots)

    return t_lst, pose_lst, g_traj
