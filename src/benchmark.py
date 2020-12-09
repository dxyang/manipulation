import numpy as np
import scipy.optimize

import pydrake
from pydrake.all import (
    PiecewisePolynomial, RigidTransform, RotationMatrix, JacobianWrtVariable,
    MathematicalProgram
)

from .drake_helpers import (
    get_basic_manip_station,
    BuildAndSimulateTrajectory,
    setup_manipulation_station,
    visualize_transform,
    GripperControllerUsingIiwaState,
    GripperControllerUsingIiwaStateV2,
    GripperControllerUsingIiwaStateV3,
)
from .ik import (
    create_q_knots,
    pose_to_jointangles,
    jointangles_to_pose,
    spatial_velocity_jacobian_at_jointangles,
)
from .throw import (
    plan_pickup,
    plan_prethrow_pose,
    add_go_to_ja_via_jointinterpolation,
    add_go_to_pose_via_jointinterpolation,
    plan_throw,
)
from .trajectory import (
    get_launch_angle_required,
    get_launch_speed_required,
)

def run_benchmark(
    zmq_url=None,
    P_WORLD_TARGET=np.array([-1, 1, 0]),
    MAX_APPROACH_ANGLE=-45 / 180.0 * np.pi,
    OBJECT_TO_TOSS="ball",
    GRIPPER_TO_OBJECT_COM_DIST=0.11,
    LAUNCH_ANGLE_THRESH=3 / 180.0 * np.pi, # 3 seems to work well?
    verbose=False,
    **kwargs,
):
    # Initialize global plants ONCE for IK calculations
    # This speed up exectuation greatly.
    GLOBAL_PLANT, GLOBAL_CONTEXT = get_basic_manip_station()

    """
    https://schunk.com/fileadmin/pim/docs/IM0026091.PDF - gripper frame to tip is about .115 m for reference
    GRIPPER_TO_OBJECT_FRAME_DIST = 0.12 # meters, this is how much "above" the balls origin we must send the gripper body frame in order to successfully grasp the object
    OBJECT_FRAME_TO_COM_DIST = 0.05 / 2
    """
    T_world_target = RigidTransform(RotationMatrix(), P_WORLD_TARGET)
    T_world_objectInitial = RigidTransform(
        #p=[-.1, -.69, 1.04998503e-01], # sphere
        p=[-0.1  , -0.69 ,  0.09], # foam_brick
        R=RotationMatrix.MakeZRotation(np.pi/2.0)
    )
    T_world_gripperObject = RigidTransform(
        p=T_world_objectInitial.translation() + np.array([0, 0, 0.025 + GRIPPER_TO_OBJECT_COM_DIST]),
        R=RotationMatrix.MakeXRotation(-np.pi/2.0)
    )
    T_world_objectCOM = RigidTransform(
        T_world_objectInitial.rotation(),
        T_world_objectInitial.translation() + np.array([0, 0, 0.025])
    )

    # Set starting and ending joint angles for throw
    throw_heading = np.arctan2(P_WORLD_TARGET[1], P_WORLD_TARGET[0])
    ja1 = throw_heading - np.pi
    # TODO: edit these to work better for large angles.
    PRETHROW_JA = np.array([ja1, 0, 0, 1.9, 0, -1.9, 0, 0, 0])
    THROWEND_JA = np.array([ja1, 0, 0, 0.4, 0, -0.4, 0, 0, 0])
    T_world_prethrowPose = jointangles_to_pose(
        plant=GLOBAL_PLANT,
        context=GLOBAL_CONTEXT,
        jointangles=PRETHROW_JA[:7],
    )

    T_world_robotInitial, meshcat = setup_manipulation_station(
        T_world_objectInitial, zmq_url, T_world_target, manipuland=OBJECT_TO_TOSS
    )

    #object frame viz
    if meshcat:
        visualize_transform(meshcat, "T_world_obj0", T_world_objectInitial)
        visualize_transform(meshcat, "T_world_objectCOM", T_world_objectCOM)
        visualize_transform(meshcat, "T_world_gripperObject", T_world_gripperObject)
        T_world_target = RigidTransform(
            p=P_WORLD_TARGET,
            R=RotationMatrix()
        )
        visualize_transform(meshcat, "target", T_world_target)

    def throw_objective(inp, g=9.81, alpha=1, return_other=None):
        throw_motion_time, release_frac = inp

        release_ja = PRETHROW_JA + release_frac * (THROWEND_JA - PRETHROW_JA)

        T_world_releasePose = jointangles_to_pose(
            plant=GLOBAL_PLANT,
            context=GLOBAL_CONTEXT,
            jointangles=release_ja[:7]
        )
        p_release = (
            T_world_releasePose.translation()
        + T_world_releasePose.rotation().multiply([0, GRIPPER_TO_OBJECT_COM_DIST, 0])
        )

        J_release = spatial_velocity_jacobian_at_jointangles(
            plant=GLOBAL_PLANT,
            context=GLOBAL_CONTEXT,
            jointangles=release_ja[:7],
            gripper_to_object_dist=GRIPPER_TO_OBJECT_COM_DIST # <==== important
        )[3:6]
        v_release = J_release @ ((THROWEND_JA - PRETHROW_JA) / throw_motion_time)[:7]

        if v_release[:2] @ (P_WORLD_TARGET - p_release)[:2] <= 0:
            return 1000

        x = np.linalg.norm((P_WORLD_TARGET - p_release)[:2])
        y = (P_WORLD_TARGET - p_release)[2]
        vx = np.linalg.norm(v_release[:2])
        vy = v_release[2]

        tta = x / vx
        y_hat = vy * tta - 0.5 * g * tta ** 2
        phi_hat = np.arctan((vy - g * tta) / vx)

        objective = (
            (y_hat - y) ** 2
          + np.maximum(phi_hat - MAX_APPROACH_ANGLE, 0) ** 2
        #+ (phi_hat - MAX_APPROACH_ANGLE) ** 2
        )
        if objective < 1e-6:
            # if we hit target at correct angle
            # then try to move as slow as possible
            objective -= throw_motion_time ** 2

        if return_other == "launch":
            return np.arctan2(vy, vx)
        elif return_other == "land":
            return phi_hat
        elif return_other == "tta":
            return tta
        else:
            return objective

    res = scipy.optimize.differential_evolution(
        throw_objective,
        bounds=[(1e-3, 3), (0.1, 0.9)],
        seed=43
    )

    throw_motion_time, release_frac = res.x
    assert throw_motion_time > 0
    assert 0 < release_frac < 1

    plan_launch_angle = throw_objective(res.x, return_other="launch")
    plan_land_angle = throw_objective(res.x, return_other="land")
    tta = throw_objective(res.x, return_other="tta")
    if verbose:
        print(f"Throw motion will take: {throw_motion_time:.4f} seconds")
        print(f"Releasing at {release_frac:.4f} along the motion")
        print(f"Launch angle (degrees): {plan_launch_angle / np.pi * 180.0}")
        print(f"Plan land angle (degrees): {plan_land_angle / np.pi * 180.0}")
        print(f"time to arrival: {tta}")

    '''
    timings
    '''
    t_goToObj = 1.0
    t_holdObj = 2.0
    t_goToPreobj = 1.0
    t_goToWaypoint = 2.0
    t_goToPrethrow = 4.0 # must be greater than 1.0 for a 1 second hold to stabilize
    t_goToThrowEnd = throw_motion_time

    # plan pickup
    t_lst, q_knots, total_time = plan_pickup(T_world_robotInitial, T_world_gripperObject,
        t_goToObj=t_goToObj,
        t_holdObj=t_holdObj,
        t_goToPreobj=t_goToPreobj
    )

    # clear the bins via a waypoint
    T_world_hackyWayPoint = RigidTransform(
        p=[-.6, -0.0, 0.6],
        R=RotationMatrix.MakeXRotation(-np.pi/2.0), #R_WORLD_PRETHROW, #RotationMatrix.MakeXRotation(-np.pi/2.0),
    )
    t_lst, q_knots = add_go_to_pose_via_jointinterpolation(
        T_world_robotInitial,
        T_world_hackyWayPoint,
        t_start=total_time,
        t_lst=t_lst,
        q_knots=q_knots,
        time_interval_s=t_goToWaypoint
    )

    # go to prethrow
    t_lst, q_knots = add_go_to_ja_via_jointinterpolation(
        pose_to_jointangles(T_world_hackyWayPoint),
        PRETHROW_JA,
        t_start=total_time + t_goToWaypoint,
        t_lst=t_lst,
        q_knots=q_knots,
        time_interval_s=t_goToPrethrow,
        hold_time_s=1.0,
    )

    # go to throw
    t_lst, q_knots = add_go_to_ja_via_jointinterpolation(
        PRETHROW_JA,
        THROWEND_JA,
        t_start=total_time + t_goToWaypoint + t_goToPrethrow,
        t_lst=t_lst,
        q_knots=q_knots,
        time_interval_s=t_goToThrowEnd,
        num_samples=30,
        include_end=True
    )

    # turn trajectory into joint space
    q_knots = np.array(q_knots)
    q_traj = PiecewisePolynomial.CubicShapePreserving(t_lst, q_knots[:, 0:7].T)


    '''
    Gripper reference trajectory (not used, we use a closed loop controller instead)
    '''
    # make gripper trajectory
    assert t_holdObj > 1.5

    gripper_times_lst = np.array([
        0.,
        t_goToObj,
        1.0, # pickup object
        0.25, # pickup object
        t_holdObj - 1.25, # pickup object
        t_goToPreobj,
        t_goToWaypoint,
        t_goToPrethrow,
        release_frac * t_goToThrowEnd,
        1e-9,
        (1 - release_frac) * t_goToThrowEnd,
    ])
    gripper_cumulative_times_lst = np.cumsum(gripper_times_lst)
    GRIPPER_OPEN = 0.5
    GRIPPER_CLOSED = 0.0
    gripper_knots = np.array([
        GRIPPER_OPEN,
        GRIPPER_OPEN,
        GRIPPER_OPEN, # pickup object
        GRIPPER_CLOSED, # pickup object
        GRIPPER_CLOSED, # pickup object
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        GRIPPER_CLOSED,
        GRIPPER_OPEN,
        GRIPPER_OPEN,
    ]).reshape(1, gripper_times_lst.shape[0])
    g_traj = PiecewisePolynomial.FirstOrderHold(gripper_cumulative_times_lst, gripper_knots)

    get_gripper_controller_3 = lambda station_plant: GripperControllerUsingIiwaStateV3(
        plant=station_plant,
        gripper_to_object_dist=GRIPPER_TO_OBJECT_COM_DIST,
        T_world_objectPickup=T_world_gripperObject,
        T_world_prethrow=T_world_prethrowPose,
        planned_launch_angle=plan_launch_angle,
        launch_angle_thresh=LAUNCH_ANGLE_THRESH,
        dbg_state_prints=verbose
    )

    # do the thing
    simulator, _, meshcat, loggers = BuildAndSimulateTrajectory(
        q_traj=q_traj,
        g_traj=g_traj,
        get_gripper_controller=get_gripper_controller_3,
        T_world_objectInitial=T_world_objectInitial, # where to init the object in the world
        T_world_targetBin=T_world_target, # where the ball should hit - aka where the bin will catch it
        zmq_url=zmq_url,
        time_step=1e-3, # target (-6, 6, -1). 1e-3 => overshoot barely, 1e-4 => undershoot barely, look around 7.92-7.94 s in sim
        include_target_bin=False,
        manipuland=OBJECT_TO_TOSS
    )

    fly_time = max(tta + 1, 2)
    if verbose:
        print(f"Throw motion should happen from 9.5 seconds to {10 + throw_motion_time} seconds")
        print(f"Running for {q_traj.end_time() + fly_time} seconds")
    if meshcat:
        meshcat.start_recording()
    simulator.AdvanceTo(q_traj.end_time() + fly_time)
    if meshcat:
        meshcat.stop_recording()
        meshcat.publish_recording()

    all_log_times = loggers["state"].sample_times()
    chosen_idxs = (9 < all_log_times) & (all_log_times < 100)

    log_times = loggers["state"].sample_times()[chosen_idxs]
    log_states = loggers["state"].data()[:, chosen_idxs]

    p_objects = np.zeros((len(log_times), 3))
    p_objects[:, 0] = log_states[4]
    p_objects[:, 1] = log_states[5]
    p_objects[:, 2] = log_states[6]

    deltas = p_objects - P_WORLD_TARGET
    land_idx = np.where(deltas[:, 2] > 0)[0][-1]

    p_land = p_objects[land_idx]
    is_overthrow = np.linalg.norm(p_land[:2]) > np.linalg.norm(P_WORLD_TARGET[:2])

    delta_land = deltas[land_idx]
    land_pos_error = np.linalg.norm(delta_land) * (1 if is_overthrow else -1)
    aim_angle_error = (
        np.arctan2(p_land[1], p_land[0])
      - np.arctan2(P_WORLD_TARGET[1], P_WORLD_TARGET[0])
    )

    v_land = (
        (p_objects[land_idx] - p_objects[land_idx - 1])
      / (log_times[land_idx] - log_times[land_idx - 1])
    )
    sim_land_angle = np.arctan(v_land[2] / np.linalg.norm(v_land[:2]))
    land_angle_error = sim_land_angle - plan_land_angle

    ret_dict=dict(
        land_pos_error=land_pos_error,
        land_angle_error=land_angle_error,
        aim_angle_error=aim_angle_error,
        time_to_arrival=tta,
        land_time=log_times[land_idx],
        land_x=p_land[0],
        land_y=p_land[1],
        land_z=p_land[2],
        sim_land_angle=sim_land_angle,
        plan_land_angle=plan_land_angle,
        plan_launch_angle=plan_launch_angle,
        release_frac=release_frac,
        throw_motion_time=throw_motion_time,
    )

    return ret_dict
