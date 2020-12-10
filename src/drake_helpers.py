import collections
import numpy as np
from pydrake.examples.manipulation_station import ManipulationStation

import pydrake
from pydrake.all import (
    BasicVector, DiagramBuilder, ConnectMeshcatVisualizer, Simulator, FindResourceOrThrow,
    LeafSystem, Parser, RigidTransform,
    MultibodyPlant, RollPitchYaw, AddTriad,
    PiecewisePolynomial, PiecewiseQuaternionSlerp, RotationMatrix,
    TrajectorySource, SignalLogger,
    JacobianWrtVariable
)

from .trajectory import get_proj_height_at_x

'''
Visualization
'''
def visualize_transform(meshcat, name, transform, prefix='', length=0.15, radius=0.006):
    # Support RigidTransform as well as 4x4 homogeneous matrix.
    if isinstance(transform, RigidTransform):
        transform = transform.GetAsMatrix4()
    AddTriad(meshcat.vis, name=name, prefix=prefix, length=length, radius=0.005, opacity=0.2)
    meshcat.vis[prefix][name].set_transform(transform)

'''
Core model and environment setup
'''
class GripperControllerUsingIiwaState(LeafSystem):
    def __init__(
        self,
        plant,
        T_world_objectPickup,
        T_world_prethrow,
        T_world_targetRelease,
        dbg_state_prints=False # turn this to true to get some helfpul dbg prints
    ):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self.gripper = plant.GetBodyByName("body")

        self.q_port = self.DeclareVectorInputPort("iiwa_position", BasicVector(7))
        self.qdot_port = self.DeclareVectorInputPort("iiwa_velocity", BasicVector(7))
        self.DeclareVectorOutputPort("wsg_position", BasicVector(1),
                                     self.CalcGripperTarget)

        # important poses
        self.T_world_objectPickup = T_world_objectPickup
        self.T_world_prethrow = T_world_prethrow
        self.T_world_targetRelease = T_world_targetRelease

        # some constants
        self.GRIPPER_OPEN = np.array([0.5])
        self.GRIPPER_CLOSED = np.array([0.0])

        # states
        self.gripper_picked_up_object = False
        self.reached_prethrow = False
        self.at_or_passed_release = False

        # this helps make sure we're in the right state
        self.is_robot_stationary = False
        self.pose_translation_history = collections.deque(maxlen=10)

        # determines gripper control based on above logic
        self.should_close_gripper = False
        self.dbg_state_prints = dbg_state_prints

    def rigid_transforms_close(self, T_world_poseA, T_world_poseB):
        r_A, t_A = T_world_poseA.rotation(), T_world_poseA.translation()
        r_B, t_B = T_world_poseB.rotation(), T_world_poseB.translation()

        # hack, being positionally close should be sufficient
        return np.allclose(t_A, t_B, rtol=1e-3, atol=1e-3)

    def CalcGripperTarget(self, context, output):
        q = self.q_port.Eval(context)
        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        T_world_robot = self._plant.EvalBodyPoseInWorld(self._plant_context, self.gripper)
        t = T_world_robot.translation()
        self.pose_translation_history.append(t)

        # location history to check if the robot is stationary (i.e., pick up object!)
        if len(self.pose_translation_history) == self.pose_translation_history.maxlen:
            all_same = True
            for i in range(len(self.pose_translation_history)):
                t_0 = self.pose_translation_history[0]
                t_i = self.pose_translation_history[i]
                same = np.allclose(t_0, t_i, rtol=1e-3, atol=1e-3)
                all_same = all_same and same
            if all_same and not self.is_robot_stationary:
                if self.dbg_state_prints:
                    print("ROBOT BECAME STATIONARY")
            elif not all_same and self.is_robot_stationary:
                if self.dbg_state_prints:
                    print("ROBOT STARTED MOVING")
            self.is_robot_stationary = all_same

        # FSM covering picking up the object, getting to the prethrow, and getting to release
        if not self.gripper_picked_up_object:
            # gripper control until we've got the object
            at_pickup = self.rigid_transforms_close(T_world_robot, self.T_world_objectPickup)
            if at_pickup:
                if self.dbg_state_prints:
                    print(f"WE REACHED THE PICKUP POINT")
            if at_pickup and self.is_robot_stationary:
                if self.dbg_state_prints:
                    print(f"AT PICK UP + ROBOT IS STATIONARY")
                self.gripper_picked_up_object = True
                self.should_close_gripper = True
        elif not self.reached_prethrow:
            at_prethrow = self.rigid_transforms_close(T_world_robot, self.T_world_prethrow)
            if at_prethrow:
                if self.dbg_state_prints:
                    print(f"WE REACHED THE PRETHROW POINT")
            if at_prethrow and self.is_robot_stationary:
                if self.dbg_state_prints:
                    print(f"AT PRETHROW + ROBOT IS STATIONARY")
                self.reached_prethrow = True
        elif not self.at_or_passed_release:
            at_release = self.rigid_transforms_close(T_world_robot, self.T_world_targetRelease)
            if at_release:
                if self.dbg_state_prints:
                    print(f"AT THE RELEASE POSE")
                self.at_or_passed_release = True
                self.should_close_gripper = False

        if self.should_close_gripper:
            output.SetFromVector(self.GRIPPER_CLOSED)
        else:
            output.SetFromVector(self.GRIPPER_OPEN)

class GripperControllerUsingIiwaStateV2(LeafSystem):
    def __init__(
        self,
        plant,
        gripper_to_object_dist,
        T_world_objectPickup,
        T_world_prethrow,
        p_world_target,
        planned_launch_angle,
        height_thresh=0.03,
        launch_angle_thresh=0.1,
        dbg_state_prints=False # turn this to true to get some helfpul dbg prints
    ):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self.gripper = plant.GetBodyByName("body")

        self.q_port = self.DeclareVectorInputPort("iiwa_position", BasicVector(7))
        self.qdot_port = self.DeclareVectorInputPort("iiwa_velocity", BasicVector(7))
        self.DeclareVectorOutputPort("wsg_position", BasicVector(1),
                                     self.CalcGripperTarget)

        # important pose info
        self.gripper_to_object_dist = gripper_to_object_dist
        self.T_world_objectPickup = T_world_objectPickup
        self.T_world_prethrow = T_world_prethrow
        self.p_world_target = p_world_target
        self.planned_launch_angle = planned_launch_angle
        self.height_thresh = height_thresh
        self.launch_angle_thresh = launch_angle_thresh

        # some constants
        self.GRIPPER_OPEN = np.array([0.5])
        self.GRIPPER_CLOSED = np.array([0.0])

        # states
        self.gripper_picked_up_object = False
        self.reached_prethrow = False
        self.at_or_passed_release = False

        # this helps make sure we're in the right state
        self.is_robot_stationary = False
        self.translation_history = np.zeros(shape=(10, 3))
        self.translation_history_idx = 0

        # determines gripper control based on above logic
        self.should_close_gripper = False
        self.dbg_state_prints = dbg_state_prints

    def rigid_transforms_close(self, T_world_poseA, T_world_poseB, tol=1e-3):
        r_A, t_A = T_world_poseA.rotation(), T_world_poseA.translation()
        r_B, t_B = T_world_poseB.rotation(), T_world_poseB.translation()

        # hack, being positionally close should be sufficient
        return np.allclose(t_A, t_B, rtol=tol, atol=tol)

    def CalcGripperTarget(self, context, output):
        q = self.q_port.Eval(context)
        qdot = self.qdot_port.Eval(context)

        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        T_world_robot = self._plant.EvalBodyPoseInWorld(self._plant_context, self.gripper)

        # location history to check if the robot is stationary (i.e., pick up object!)
        self.translation_history[
            self.translation_history_idx % self.translation_history.shape[0]
        ] = T_world_robot.translation()
        self.translation_history_idx += 1
        if self.translation_history_idx >= self.translation_history.shape[0]:
            all_same = np.abs(
                self.translation_history.max(axis=0)
              - self.translation_history.min(axis=0)
            ).max() < 1e-3

            if self.dbg_state_prints:
                if all_same != self.is_robot_stationary:
                    print(f"ROBOT BECAME {'STATIONARY' if all_same else 'MOVING'}")

            self.is_robot_stationary = all_same

        # FSM covering picking up the object, getting to the prethrow, and getting to release
        if not self.gripper_picked_up_object:
            # gripper control until we've got the object
            at_pickup = self.rigid_transforms_close(T_world_robot, self.T_world_objectPickup)
            if at_pickup and self.is_robot_stationary:
                self.gripper_picked_up_object = True
                self.should_close_gripper = True
        elif not self.reached_prethrow:
            at_prethrow = self.rigid_transforms_close(T_world_robot, self.T_world_prethrow)
            if at_prethrow and self.is_robot_stationary:
                self.reached_prethrow = True
        elif not self.at_or_passed_release:
            J_robot = self._plant.CalcJacobianTranslationalVelocity(
                self._plant_context,
                JacobianWrtVariable.kQDot,
                self.gripper.body_frame(),
                [0, self.gripper_to_object_dist, 0],
                self._plant.world_frame(),
                self._plant.world_frame()
            )[:, 7:14] # TODO: Fixme: This is hardcoded and is flaky.
            v_proj = J_robot @ qdot
            p_proj = (
                T_world_robot.translation()
              + T_world_robot.rotation().multiply([0, self.gripper_to_object_dist, 0])
            )

            # Assume p_proj, v_proj, self.p_world_target are aligned
            # assert np.abs(np.cross(v_porj, self.p_world_target - p_proj)).max() < 1e-3
            # Then convert problem to 2d
            launch_vx = np.linalg.norm(v_proj[:2])
            launch_vy = np.linalg.norm(v_proj[2])
            launch_angle = np.arctan2(launch_vy, launch_vx)
            dx = np.linalg.norm(self.p_world_target[:2] - p_proj[:2])

            proj_height_at_target = get_proj_height_at_x(
                launch_vx=launch_vx, launch_vy=launch_vy, target_x=dx
            ) + p_proj[2]

            # Only launch if our angle is close enough to the desired angle
            if launch_angle > self.planned_launch_angle - self.launch_angle_thresh:
                if self.dbg_state_prints:
                    if (
                        np.abs(proj_height_at_target - self.p_world_target[2])
                        < self.height_thresh + 1
                    ):
                        print(
                            f"p={p_proj}",
                            f"v={v_proj}",
                            f"v2d=({launch_vx:.2f}, {launch_vy:.2f})",
                            f"x={dx:.2f}",
                            f"ys=({proj_height_at_target:.2f}, {self.p_world_target[2]})",
                        )

                # Release if our projected throw lands close to target
                if (
                    np.abs(proj_height_at_target - self.p_world_target[2])
                    < self.height_thresh
                ):
                    self.at_or_passed_release = True
                    self.should_close_gripper = False
                    if self.dbg_state_prints:
                        print(f"RELEASING!")


        if self.should_close_gripper:
            output.SetFromVector(self.GRIPPER_CLOSED)
        else:
            output.SetFromVector(self.GRIPPER_OPEN)

class GripperControllerUsingIiwaStateV3(LeafSystem):
    def __init__(
        self,
        plant,
        gripper_to_object_dist,
        T_world_objectPickup,
        T_world_prethrow,
        planned_launch_angle,
        launch_angle_thresh=0,
        dbg_state_prints=False # turn this to true to get some helfpul dbg prints
    ):
        LeafSystem.__init__(self)
        self._plant = plant
        self._plant_context = plant.CreateDefaultContext()
        self._iiwa = plant.GetModelInstanceByName("iiwa")
        self.gripper = plant.GetBodyByName("body")

        self.q_port = self.DeclareVectorInputPort("iiwa_position", BasicVector(7))
        self.qdot_port = self.DeclareVectorInputPort("iiwa_velocity", BasicVector(7))
        self.DeclareVectorOutputPort("wsg_position", BasicVector(1),
                                     self.CalcGripperTarget)

        # important pose info
        self.gripper_to_object_dist = gripper_to_object_dist
        self.T_world_objectPickup = T_world_objectPickup
        self.T_world_prethrow = T_world_prethrow
        self.planned_launch_angle = planned_launch_angle
        self.launch_angle_thresh = launch_angle_thresh

        # some constants
        self.GRIPPER_OPEN = np.array([0.5])
        self.GRIPPER_CLOSED = np.array([0.0])

        # states
        self.gripper_picked_up_object = False
        self.reached_prethrow = False
        self.at_or_passed_release = False

        # this helps make sure we're in the right state
        self.is_robot_stationary = False
        self.translation_history = np.zeros(shape=(10, 3))
        self.translation_history_idx = 0

        # determines gripper control based on above logic
        self.should_close_gripper = False
        self.dbg_state_prints = dbg_state_prints

    def translations_close(self, T_world_poseA, T_world_poseB, tol=1e-3):
        t_A = T_world_poseA.translation()
        t_B = T_world_poseB.translation()
        return np.allclose(t_A, t_B, rtol=tol, atol=tol)

    def CalcGripperTarget(self, context, output):
        q = self.q_port.Eval(context)
        qdot = self.qdot_port.Eval(context)

        self._plant.SetPositions(self._plant_context, self._iiwa, q)
        T_world_robot = self._plant.EvalBodyPoseInWorld(self._plant_context, self.gripper)

        # location history to check if the robot is stationary (i.e., pick up object!)
        self.translation_history[
            self.translation_history_idx % self.translation_history.shape[0]
        ] = T_world_robot.translation()
        self.translation_history_idx += 1
        if self.translation_history_idx >= self.translation_history.shape[0]:
            all_same = np.abs(
                self.translation_history.max(axis=0)
              - self.translation_history.min(axis=0)
            ).max() < 1e-3

            if self.dbg_state_prints:
                if all_same != self.is_robot_stationary:
                    print(f"ROBOT BECAME {'STATIONARY' if all_same else 'MOVING'}")

            self.is_robot_stationary = all_same

        # FSM covering picking up the object, getting to the prethrow, and getting to release
        if not self.gripper_picked_up_object:
            # gripper control until we've got the object
            at_pickup = self.translations_close(T_world_robot, self.T_world_objectPickup)
            if at_pickup and self.is_robot_stationary:
                self.gripper_picked_up_object = True
                self.should_close_gripper = True
        elif not self.reached_prethrow:
            at_prethrow = self.translations_close(T_world_robot, self.T_world_prethrow)
            if at_prethrow and self.is_robot_stationary:
                self.reached_prethrow = True
        elif not self.at_or_passed_release:
            J_robot = self._plant.CalcJacobianTranslationalVelocity(
                self._plant_context,
                JacobianWrtVariable.kQDot,
                self.gripper.body_frame(),
                [0, self.gripper_to_object_dist, 0],
                self._plant.world_frame(),
                self._plant.world_frame()
            )[:, 7:14] # TODO: Fixme: This is hardcoded and is flaky.
            v_proj = J_robot @ qdot

            # Assume p_proj, v_proj, self.p_world_target are aligned
            # assert np.abs(np.cross(v_porj, self.p_world_target - p_proj)).max() < 1e-3
            # Then convert problem to 2d
            launch_vx = np.linalg.norm(v_proj[:2])
            launch_vy = np.linalg.norm(v_proj[2])
            launch_angle = np.arctan2(launch_vy, launch_vx)

            if self.dbg_state_prints:
                if (
                    launch_angle - self.planned_launch_angle
                  > self.launch_angle_thresh - 0.1
                ):
                    print(
                        #f"p={p_proj}",
                        #f"v={v_proj}",
                        f"v2d=({launch_vx:.2f}, {launch_vy:.2f})",
                        f"launch_angles=({launch_angle*180/np.pi:.2f}, {self.planned_launch_angle*180/np.pi:.2f})",
                    )

            # Launch when angle is high enough
            if launch_angle - self.planned_launch_angle > self.launch_angle_thresh:
                self.at_or_passed_release = True
                self.should_close_gripper = False
                if self.dbg_state_prints:
                    print(f"RELEASING!")


        if self.should_close_gripper:
            output.SetFromVector(self.GRIPPER_CLOSED)
        else:
            output.SetFromVector(self.GRIPPER_OPEN)

def get_basic_manip_station():
    """Used for IK"""
    builder = DiagramBuilder()
    station = builder.AddSystem(ManipulationStation())
    station.SetupClutterClearingStation()
    station.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant = station.get_multibody_plant()

    return plant, context


# manipulation station has a lot of extra things that we don't need for IK
def CreateIiwaControllerPlant():
    """creates plant that includes only the robot and gripper, used for controllers."""
    robot_sdf_path = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/iiwa7_no_collision.sdf")
    gripper_sdf_path = FindResourceOrThrow(
        "drake/manipulation/models/wsg_50_description/sdf/schunk_wsg_50_no_tip.sdf")
    sim_timestep = 1e-3
    plant_robot = MultibodyPlant(sim_timestep)
    parser = Parser(plant=plant_robot)
    parser.AddModelFromFile(robot_sdf_path)
    parser.AddModelFromFile(gripper_sdf_path)
    plant_robot.WeldFrames(
        A=plant_robot.world_frame(),
        B=plant_robot.GetFrameByName("iiwa_link_0"))
    plant_robot.WeldFrames(
        A=plant_robot.GetFrameByName("iiwa_link_7"),
        B=plant_robot.GetFrameByName("body"),
        X_AB=RigidTransform(RollPitchYaw(np.pi/2, 0, np.pi/2), np.array([0, 0, 0.114]))
    )
    plant_robot.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    plant_robot.Finalize()

    link_frame_indices = []
    for i in range(8):
        link_frame_indices.append(
            plant_robot.GetFrameByName("iiwa_link_" + str(i)).index())

    return plant_robot, link_frame_indices


# used for getting the initial pose of the robot
def setup_manipulation_station(T_world_objectInitial, zmq_url, T_world_targetBin, manipuland="foam", include_bin=True, include_hoop=False):
    builder = DiagramBuilder()
    station = builder.AddSystem(ManipulationStation(time_step=1e-3))
    station.SetupClutterClearingStation()
    if manipuland is "foam":
        station.AddManipulandFromFile(
            "drake/examples/manipulation_station/models/061_foam_brick.sdf",
            T_world_objectInitial)
    elif manipuland is "ball":
        station.AddManipulandFromFile(
            "drake/examples/manipulation_station/models/sphere.sdf",
            T_world_objectInitial)
    elif manipuland is "bball":
        station.AddManipulandFromFile(
            "drake/../../../../../../manipulation/sdfs/bball.sdf", # this is some path hackery
            T_world_objectInitial)
    elif manipuland is "rod":
        station.AddManipulandFromFile(
            "drake/examples/manipulation_station/models/rod.sdf",
            T_world_objectInitial)
    station_plant = station.get_multibody_plant()
    parser = Parser(station_plant)
    if include_bin:
        parser.AddModelFromFile("extra_bin.sdf")
        station_plant.WeldFrames(station_plant.world_frame(), station_plant.GetFrameByName("extra_bin_base"), T_world_targetBin)
    if include_hoop:
        parser.AddModelFromFile("sdfs/hoop.sdf")
        station_plant.WeldFrames(station_plant.world_frame(), station_plant.GetFrameByName("base_link_hoop"), T_world_targetBin)
    station.Finalize()

    frames_to_draw = {"gripper": {"body"}}
    meshcat = None
    if zmq_url is not None:
        meshcat = ConnectMeshcatVisualizer(builder,
            station.get_scene_graph(),
            output_port=station.GetOutputPort("pose_bundle"),
            delete_prefix_on_load=False,
            frames_to_draw=frames_to_draw,
            zmq_url=zmq_url)

    diagram = builder.Build()

    plant = station.get_multibody_plant()
    context = plant.CreateDefaultContext()
    gripper = plant.GetBodyByName("body")

    initial_pose = plant.EvalBodyPoseInWorld(context, gripper)

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(0.01)

    return initial_pose, meshcat

def BuildAndSimulateTrajectory(
    q_traj,
    g_traj,
    get_gripper_controller,
    T_world_objectInitial,
    T_world_targetBin,
    zmq_url,
    time_step,
    include_target_bin=True,
    include_hoop=False,
    manipuland="foam"
):
    """Simulate trajectory for manipulation station.
    @param q_traj: Trajectory class used to initialize TrajectorySource for joints.
    @param g_traj: Trajectory class used to initialize TrajectorySource for gripper.
    """
    builder = DiagramBuilder()
    station = builder.AddSystem(ManipulationStation(time_step=time_step)) #1e-3 or 1e-4 probably
    station.SetupClutterClearingStation()
    if manipuland is "foam":
        station.AddManipulandFromFile(
            "drake/examples/manipulation_station/models/061_foam_brick.sdf",
            T_world_objectInitial)
    elif manipuland is "ball":
        station.AddManipulandFromFile(
            "drake/examples/manipulation_station/models/sphere.sdf",
            T_world_objectInitial)
    elif manipuland is "bball":
        station.AddManipulandFromFile(
            "drake/../../../../../../manipulation/sdfs/bball.sdf", # this is some path hackery
            T_world_objectInitial)
    elif manipuland is "rod":
        station.AddManipulandFromFile(
            "drake/examples/manipulation_station/models/rod.sdf",
            T_world_objectInitial)
    station_plant = station.get_multibody_plant()
    parser = Parser(station_plant)
    if include_target_bin:
        parser.AddModelFromFile("sdfs/extra_bin.sdf")
        station_plant.WeldFrames(station_plant.world_frame(), station_plant.GetFrameByName("extra_bin_base"), T_world_targetBin)
    if include_hoop:
        parser.AddModelFromFile("sdfs/hoop.sdf")
        station_plant.WeldFrames(station_plant.world_frame(), station_plant.GetFrameByName("base_link_hoop"), T_world_targetBin)
    station.Finalize()

    # iiwa joint trajectory - predetermined trajectory
    q_traj_system = builder.AddSystem(TrajectorySource(q_traj))
    builder.Connect(q_traj_system.get_output_port(),
                    station.GetInputPort("iiwa_position"))

    # gripper - closed loop controller
    gctlr = builder.AddSystem(get_gripper_controller(station_plant))
    gctlr.set_name("GripperControllerUsingIiwaState")
    builder.Connect(station.GetOutputPort("iiwa_position_measured"),  gctlr.GetInputPort("iiwa_position"))
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"), gctlr.GetInputPort("iiwa_velocity"))
    builder.Connect(gctlr.get_output_port(), station.GetInputPort("wsg_position"))

    loggers = dict(
        state=builder.AddSystem(SignalLogger(31)),
        v_est=builder.AddSystem(SignalLogger(7))
    )
    builder.Connect(station.GetOutputPort("plant_continuous_state"),
                    loggers["state"].get_input_port())
    builder.Connect(station.GetOutputPort("iiwa_velocity_estimated"),
                    loggers["v_est"].get_input_port())

    meshcat = None
    if zmq_url is not None:
        meshcat = ConnectMeshcatVisualizer(builder,
            station.get_scene_graph(),
            output_port=station.GetOutputPort("pose_bundle"),
            delete_prefix_on_load=True,
            frames_to_draw={"gripper":{"body"}},
            zmq_url=zmq_url)

    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    return simulator, station_plant, meshcat, loggers

if __name__ == "__main__":
    pass