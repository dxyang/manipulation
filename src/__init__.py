import numpy as np
from pydrake.examples.manipulation_station import ManipulationStation

import pydrake
from pydrake.all import (
    DiagramBuilder, ConnectMeshcatVisualizer, Simulator, FindResourceOrThrow,
    Parser, MultibodyPlant, RigidTransform, #LeafSystem, BasicVector,
    RollPitchYaw, AddTriad, #JacobianWrtVariable, SignalLogger,
    PiecewisePolynomial, PiecewiseQuaternionSlerp, RotationMatrix, Solve,
    TrajectorySource, #BodyIndex
)
from pydrake.multibody import inverse_kinematics