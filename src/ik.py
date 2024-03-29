import numpy as np

import pydrake
from pydrake.all import DiagramBuilder, JacobianWrtVariable, RigidTransform, RotationMatrix, Solve
from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.multibody import inverse_kinematics

from .drake_helpers import CreateIiwaControllerPlant, visualize_transform

def spatial_velocity_jacobian_at_jointangles(
    plant, context,
    jointangles,
    gripper_to_object_dist
):
    # returns a 6 x 7 matrix
    # [dw_x, dw_y, dw_z, dv_x, dv_y, dv_z] for each of the 7 joints

    # jointangles should be a numpy array or list of joint angles of length 7
    ja = np.array(jointangles).squeeze()
    assert(ja.shape == (7,))

    plant.SetPositions(
        plant.GetMyContextFromRoot(context),
        plant.GetModelInstanceByName("iiwa"),
        ja
    )
    plant_context = plant.GetMyContextFromRoot(context)

    J_G = plant.CalcJacobianSpatialVelocity(
        plant_context,
        JacobianWrtVariable.kQDot,
        plant.GetBodyByName("body").body_frame(),
        [0, gripper_to_object_dist, 0],
        plant.world_frame(),
        plant.world_frame()
    )
    return J_G[:, :7]

def jointangles_to_pose(plant, context, jointangles):
    # jointangles should be a numpy array or list of joint angles of length 7
    #ja = np.array(jointangles).squeeze()
    ja = jointangles
    assert(ja.shape == (7,))

    plant.SetPositions(
        plant.GetMyContextFromRoot(context),
        plant.GetModelInstanceByName("iiwa"),
        ja
    )
    plant_context = plant.GetMyContextFromRoot(context)
    pose = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("body"))
    return pose

def pose_to_jointangles(T_world_robotPose):
    plant, _ = CreateIiwaControllerPlant()
    plant_context = plant.CreateDefaultContext()
    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("body")
    #q_nominal = np.array([ 0., 0.6, 0., -1.75, 0., 1., 0., 0., 0.]) # nominal joint for joint-centering.
    q_nominal = np.array([-1.57, 0.1, 0.00, -1.2, 0.00, 1.60, 0.00, 0.00, 0.00])

    def AddOrientationConstraint(ik, R_WG, bounds):
        ik.AddOrientationConstraint(
            frameAbar=world_frame, R_AbarA=R_WG,
            frameBbar=gripper_frame, R_BbarB=RotationMatrix(),
            theta_bound=bounds
        )

    def AddPositionConstraint(ik, p_WG_lower, p_WG_upper):
        ik.AddPositionConstraint(
            frameA=world_frame, frameB=gripper_frame, p_BQ=np.zeros(3),
            p_AQ_lower=p_WG_lower, p_AQ_upper=p_WG_upper)

    # def AddJacobianConstraint_Joint_To_Plane(ik):
    #     # calculate the jacobian
    #     J_G = plant.CalcJacobianSpatialVelocity(
    #         ik.context(),
    #         JacobianWrtVariable.kQDot,
    #         gripper_frame,
    #         [0,0,0],
    #         world_frame,
    #         world_frame
    #     )

    #     # ensure that when joints 4 and 6 move, they keep the gripper in the desired plane
    #     prog = ik.get_mutable_prog()
    #     prog.AddConstraint()
    #     joint_4 = J_G[:, 3]
    #     joint_6 = J_G[:, 5]

    ik = inverse_kinematics.InverseKinematics(plant)
    q_variables = ik.q() # Get variables for MathematicalProgram
    prog = ik.prog() # Get MathematicalProgram

    p_WG = T_world_robotPose.translation()
    r_WG = T_world_robotPose.rotation()

    # must be an exact solution
    z_slack = 0
    degrees_slack = 0
    AddPositionConstraint(ik, p_WG - np.array([0, 0, z_slack]), p_WG + np.array([0, 0, z_slack]))
    AddOrientationConstraint(ik, r_WG, degrees_slack * np.pi / 180)

    # todo: add some sort of constraint so that jacobian is 0 in certain directions
    # (so just joints 4 and 6 move when swung)

    # initial guess
    prog.SetInitialGuess(q_variables, q_nominal)
    diff = q_variables - q_nominal

    prog.AddCost(np.sum(diff.dot(diff)))

    result = Solve(prog)
    if not result.is_success():
        #visualize_transform(meshcat, "FAIL", X_WG, prefix='', length=0.3, radius=0.02)
        assert(False) # no IK solution for target

    jas = result.GetSolution(q_variables)

    return jas

def create_q_knots(pose_lst):
    """Convert end-effector pose list to joint position list using series of
    InverseKinematics problems. Note that q is 9-dimensional because the last 2 dimensions
    contain gripper joints, but these should not matter to the constraints.
    @param: pose_lst (python list): post_lst[i] contains keyframe X_WG at index i.
    @return: q_knots (python_list): q_knots[i] contains IK solution that will give f(q_knots[i]) \approx pose_lst[i].
    """
    q_knots = []
    plant, _ = CreateIiwaControllerPlant()
    world_frame = plant.world_frame()
    gripper_frame = plant.GetFrameByName("body")
    #q_nominal = np.array([ 0., 0.6, 0., -1.75, 0., 1., 0., 0., 0.]) # nominal joint for joint-centering.
    q_nominal = np.array([-1.57, 0.1, 0.00, -1.2, 0.00, 1.60, 0.00, 0.00, 0.00])

    def AddOrientationConstraint(ik, R_WG, bounds):
        """Add orientation constraint to the ik problem. Implements an inequality
        constraint where the axis-angle difference between f_R(q) and R_WG must be
        within bounds. Can be translated to:
        ik.prog().AddBoundingBoxConstraint(angle_diff(f_R(q), R_WG), -bounds, bounds)
        """
        ik.AddOrientationConstraint(
            frameAbar=world_frame, R_AbarA=R_WG,
            frameBbar=gripper_frame, R_BbarB=RotationMatrix(),
            theta_bound=bounds
        )

    def AddPositionConstraint(ik, p_WG_lower, p_WG_upper):
        """Add position constraint to the ik problem. Implements an inequality
        constraint where f_p(q) must lie between p_WG_lower and p_WG_upper. Can be
        translated to
        ik.prog().AddBoundingBoxConstraint(f_p(q), p_WG_lower, p_WG_upper)
        """
        ik.AddPositionConstraint(
            frameA=world_frame, frameB=gripper_frame, p_BQ=np.zeros(3),
            p_AQ_lower=p_WG_lower, p_AQ_upper=p_WG_upper)

    for i in range(len(pose_lst)):
        # if i % 100 == 0:
        #     print(i)
        ik = inverse_kinematics.InverseKinematics(plant)
        q_variables = ik.q() # Get variables for MathematicalProgram
        prog = ik.prog() # Get MathematicalProgram

        #### Modify here ###############################
        X_WG = pose_lst[i]
        p_WG = X_WG.translation()
        r_WG = X_WG.rotation()

        z_slack = 0
        degrees_slack = 0
        AddPositionConstraint(ik, p_WG - np.array([0, 0, z_slack]), p_WG + np.array([0, 0, z_slack]))
        AddOrientationConstraint(ik, r_WG, degrees_slack * np.pi / 180)

        # initial guess
        if i == 0:
            prog.SetInitialGuess(q_variables, q_nominal)
            diff = q_variables - q_nominal
        else:
            prog.SetInitialGuess(q_variables, q_knots[i - 1])
            diff = q_variables - q_knots[i - 1]

        prog.AddCost(np.sum(diff.dot(diff)))

        ################################################

        result = Solve(prog)
        if not result.is_success():
            visualize_transform(meshcat, "FAIL", X_WG, prefix='', length=0.3, radius=0.02)
            print(f"Failed at {i}")
            break
            #raise RuntimeError
        tmp = result.GetSolution(q_variables)
        q_knots.append(tmp)

    return q_knots
