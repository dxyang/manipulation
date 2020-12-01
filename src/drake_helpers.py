import numpy as np
from pydrake.examples.manipulation_station import ManipulationStation

import pydrake
from pydrake.all import (
    DiagramBuilder, ConnectMeshcatVisualizer, Simulator, FindResourceOrThrow,
    Parser, MultibodyPlant, RigidTransform,
    RollPitchYaw, AddTriad,
    PiecewisePolynomial, PiecewiseQuaternionSlerp, RotationMatrix,
    TrajectorySource, SignalLogger
)

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
# used for getting the initial pose of the robot
def setup_manipulation_station(T_world_objectInitial, zmq_url):
    builder = DiagramBuilder()
    station = builder.AddSystem(ManipulationStation(time_step=1e-3))
    station.SetupClutterClearingStation()
    station.AddManipulandFromFile(
        #"drake/examples/manipulation_station/models/061_foam_brick.sdf",
        "drake/examples/manipulation_station/models/sphere.sdf",
        T_world_objectInitial)
    station.Finalize()

    frames_to_draw = {"gripper": {"body"}}
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

def BuildAndSimulateTrajectory(q_traj, g_traj, T_world_objectInitial, T_world_target, zmq_url):
    """Simulate trajectory for manipulation station.
    @param q_traj: Trajectory class used to initialize TrajectorySource for joints.
    @param g_traj: Trajectory class used to initialize TrajectorySource for gripper.
    """
    builder = DiagramBuilder()
    station = builder.AddSystem(ManipulationStation(time_step=1e-4))
    station.SetupClutterClearingStation()
    station.AddManipulandFromFile(
        #"drake/examples/manipulation_station/models/061_foam_brick.sdf",
        "drake/examples/manipulation_station/models/sphere.sdf",
        T_world_objectInitial)
    station.Finalize()

    station_plant = station.get_multibody_plant()

    q_traj_system = builder.AddSystem(TrajectorySource(q_traj))
    g_traj_system = builder.AddSystem(TrajectorySource(g_traj))
    builder.Connect(q_traj_system.get_output_port(),
                    station.GetInputPort("iiwa_position"))
    builder.Connect(g_traj_system.get_output_port(),
                    station.GetInputPort("wsg_position"))

    state_logger = builder.AddSystem(SignalLogger(31))
    #builder.Connect(station_plant.GetOutputPort("continuous_state"),
    #                state_logger.get_input_port())
    builder.Connect(station.GetOutputPort("plant_continuous_state"),
                    state_logger.get_input_port())

    meshcat = ConnectMeshcatVisualizer(builder,
          station.get_scene_graph(),
          output_port=station.GetOutputPort("pose_bundle"),
          delete_prefix_on_load=True,
          frames_to_draw={"gripper":{"body"}},
          zmq_url=zmq_url)

    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)

    return simulator, station_plant, meshcat, state_logger