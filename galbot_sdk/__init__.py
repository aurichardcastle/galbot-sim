"""galbot_sdk — GalbotSim: a pure-Python, robot-free backend for the Galbot
SDK v1.9.0 Python API, backed by MuJoCo and Galbot's official S1 model.

Mirrors the official package layout (api_contract.md §1): the top-level
package owns the full base-module namespace (the real loader copies every
public symbol of the compiled ``galbot_sdk.cpython-*.so`` into package
globals); ``galbot_sdk.g1`` / ``galbot_sdk.s1`` re-export it via
``from .. import *`` and add the pre-bound singleton wrapper classes.

Import-time self-check: every name of the verbatim api_contract §2 ``__all__``
must be present in this namespace, else ``ImportError`` is raised naming the
missing symbols (mirrors the real loader's fail-at-import contract and
catches inter-module drift instantly).
"""

__version__ = "1.9.0"
version = __version__

# Logger re-exports (api_contract §1/§3)
from .galbot_sdk_logger import (
    init_logger,
    debug,
    info,
    warning,
    error,
    critical,
)

# Base-module namespace (DESIGN.md §5.1)
from ._enums import *          # noqa: F401,F403 — enums, constants, aliases (§4-§5)
from ._types import *          # noqa: F401,F403 — value types (§7)
from ._robot import GalbotRobot                     # noqa: F401 — §8.1
from ._motion import (                              # noqa: F401 — §8.2-§8.4, §6
    GalbotMotion,
    GalbotNavigation,
    GalbotPerception,
    check_motion_status,
    create_joint_state,
    create_pose_state,
    create_parameter,
)

# ---------------------------------------------------------------------------
# api_contract.md §2 base-module __all__ — VERBATIM, do not edit by hand.
# ---------------------------------------------------------------------------
__all__ = [
    "AudioData", "CLOSE_TO_OBSTACLE", "COLLISION", "COMM_DISCONNECTED", "CYLINDER",
    "CollisionCheckOption", "ControlStatus", "DATA_FETCH_FAILED", "DepthData",
    "DetectionAndSegmentationResult", "DetectionResult", "DexHandType", "DexhandState",
    "EUCLIDEAN_DISTANCE", "EffortInfo", "Error", "ErrorInfo", "FAILED", "FAULT", "FOUNDATION_STEREO",
    "ForceData", "FrameTriad", "G1ControllerName", "G1JointGroup", "GalbotMotion", "GalbotNavigation",
    "GalbotOneFoxtrotSensor", "GalbotPerception", "GalbotRobot", "GripperState", "GroupCommand",
    "Header", "IKSolverConfig", "INIT_FAILED", "INTERRUPTED", "INVALID_INPUT", "IN_PROGRESS", "ImuData",
    "JOINT", "JointCommand", "JointState", "JointStateMessage", "JointStates", "KinematicsBoundary",
    "LIGHT_STEREO", "LINE", "LidarData", "LineTrajCheckPrimitive", "LogLevel", "MachineType",
    "MotionPlanConfig", "MotionStatus", "NavigationTaskSnapshot", "NavigationTaskStatus", "OCCUPIED",
    "OdomData", "POSE", "PUBLISH_FAIL", "Parameter", "PerceptionModule", "Point", "Point2d", "PointField",
    "PointFieldDataType", "Pose", "Pose2d", "PoseState", "PrimitiveType", "Quaternion", "RADIAN_DISTANCE",
    "RANDOM_PROGRESSIVE_SEED", "RANDOM_SEED", "ROBOT_STATES", "RUNNING", "RgbData", "RobotStates",
    "RobotStatesType", "S1ControllerName", "S1JointGroup", "STATUS_NUM", "STOPPED_UNREACHED", "SUCCESS",
    "SUCTION_ACTION_STATE", "SamplerConfig", "SeedType", "SensorType", "SingoriXTarget", "StateCheckType",
    "SuctionCupState", "SyncedObservation", "TARGET_DATA_DEFAULT", "TARGET_DATA_FRAME_POSE",
    "TARGET_DATA_FRAME_TWIST", "TARGET_DATA_FRAME_WRENCH", "TARGET_DATA_JOINT_ACCELERATION",
    "TARGET_DATA_JOINT_EFFORT", "TARGET_DATA_JOINT_POSITION", "TARGET_DATA_JOINT_VELOCITY",
    "TARGET_DATA_NONE", "TARGET_TYPE_APPEND", "TARGET_TYPE_CLEAR", "TARGET_TYPE_DEFAULT",
    "TARGET_TYPE_NONE", "TARGET_TYPE_OVERRIDE", "TARGET_TYPE_PREPENDNOW", "TARGET_TYPE_PROVERRIDE",
    "TARGET_TYPE_TOUCH", "TIMEOUT", "TIMEOUT_AND_EXACT_SOLUTION", "TargetConfig",
    "TargetGroupTrajectory", "TargetSampling", "TargetTaskTrajectory", "TaskCommand", "TaskHandle",
    "TerminationConditionType", "Timestamp", "Trajectory", "TrajectoryControlStatus",
    "TrajectoryFeasibilityCheckOption", "TrajectoryPlanConfig", "TrajectoryPoint", "Twist", "UNKNOWN",
    "UNSUPPORTED_FUNCRION", "USER_DEFINED_SEED", "UltrasonicData", "UltrasonicType", "Vector3",
    "WBCException", "Waypoint", "WaypointParams", "Wrench", "check_motion_status", "create_joint_state",
    "create_parameter", "create_pose_state",
]

# Import-time completeness check (DESIGN.md §5.1): fail loudly, name the drift.
_missing = [_name for _name in __all__ if _name not in globals()]
if _missing:
    raise ImportError(
        "galbot_sdk (GalbotSim): base-module namespace is missing "
        f"{len(_missing)} required public symbol(s) declared by "
        "api_contract.md §2 __all__: " + ", ".join(sorted(_missing))
    )
del _missing
