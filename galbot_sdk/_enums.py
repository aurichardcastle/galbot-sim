"""galbot_sdk._enums — GalbotSim enum surface (api_contract.md §4–§5).

Pybind-lookalike enums, string-constant classes, module-level int constants,
and the module-level enum aliases declared by the official stub tail.

Fidelity notes:
- Every member value is verbatim from the official ``galbot_sdk.pyi`` stubs,
  including the SDK's own typo ``MotionStatus.UNSUPPORTED_FUNCRION``.
- The real SDK enums are pybind11 enum types (NOT ``enum.Enum`` subclasses).
  We use ``enum.IntEnum`` with pybind-lookalike ``__str__``/``__repr__``:
  members provide ``.name``/``.value``/``__members__``, ``int()``/``__index__``,
  hashing, equality with ints, and construction from int — the whole surface
  the tutorials (and the api_contract §4 header) observe.
  Accepted delta (DESIGN.md §5.3/§6): members are ``int`` subclasses, which is
  observable only via ``isinstance(x, int)`` — nothing exercises it.
"""

import enum


class _SdkEnum(enum.IntEnum):
    """Base class for pybind11-lookalike SDK enums.

    ``str()``  -> ``"ClassName.MEMBER"``       (pybind __str__)
    ``repr()`` -> ``"<ClassName.MEMBER: v>"``  (pybind __repr__)
    """

    def __str__(self):
        return f"{self.__class__.__name__}.{self.name}"

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {int(self)}>"


# ---------------------------------------------------------------------------
# §4 enums (exact values)
# ---------------------------------------------------------------------------

class ControlStatus(_SdkEnum):
    """Returned by nearly all GalbotRobot control APIs."""
    SUCCESS = 0             # Execution successful
    TIMEOUT = 1             # Execution timeout
    FAULT = 2               # Fault occurred, cannot continue execution
    INVALID_INPUT = 3       # Input parameters do not meet requirements
    INIT_FAILED = 4         # Internal communication component creation failed
    IN_PROGRESS = 5         # Motion in progress but not reached target
    STOPPED_UNREACHED = 6   # Stopped but not reached target
    DATA_FETCH_FAILED = 7   # Data fetch failed
    PUBLISH_FAIL = 8        # Data sending failed
    COMM_DISCONNECTED = 9   # Connection failed


class MotionStatus(_SdkEnum):
    """Returned by GalbotMotion APIs (values 0-9 mirror ControlStatus)."""
    SUCCESS = 0
    TIMEOUT = 1
    FAULT = 2
    INVALID_INPUT = 3
    INIT_FAILED = 4
    IN_PROGRESS = 5
    STOPPED_UNREACHED = 6
    DATA_FETCH_FAILED = 7
    PUBLISH_FAIL = 8
    COMM_DISCONNECTED = 9
    STATUS_NUM = 10
    UNSUPPORTED_FUNCRION = 11   # official SDK typo — preserved verbatim


class TrajectoryControlStatus(_SdkEnum):
    INVALID_INPUT = 0
    RUNNING = 1
    COMPLETED = 2
    STOPPED_UNREACHED = 3
    ERROR = 4
    DATA_FETCH_FAILED = 5


class NavigationTaskStatus(_SdkEnum):
    UNKNOWN = 0
    RUNNING = 1
    SUCCESS = 2
    FAILED = 3
    INTERRUPTED = 4
    OCCUPIED = 5
    COLLISION = 6
    CLOSE_TO_OBSTACLE = 7


class MachineType(_SdkEnum):
    G1 = 0
    S1 = 1


class SensorType(_SdkEnum):
    HEAD_LEFT_CAMERA = 0
    HEAD_RIGHT_CAMERA = 1
    LEFT_ARM_CAMERA = 2
    RIGHT_ARM_CAMERA = 3
    LEFT_ARM_DEPTH_CAMERA = 4
    RIGHT_ARM_DEPTH_CAMERA = 5
    BASE_LIDAR = 6
    HEAD_LIDAR = 7
    BACK_LIDAR = 8
    CHASSIS_LIDAR = 9
    HEAD_IMU = 10
    BACK_IMU = 11
    CHASSIS_IMU = 12
    TORSO_IMU = 13
    LIDAR_IMU = 14
    BASE_ULTRASONIC = 15
    LEFT_FRONT_SURROUND_CAMERA = 16
    RIGHT_FRONT_SURROUND_CAMERA = 17
    LEFT_REAR_SURROUND_CAMERA = 18
    RIGHT_REAR_SURROUND_CAMERA = 19


class DexHandType(_SdkEnum):
    INSPIRE = 0
    BRAINCO = 1
    SHARPA = 2


class GalbotOneFoxtrotSensor(_SdkEnum):
    """G1 only (absent from the s1 stub, present in the base module)."""
    LEFT_WRIST_FORCE = 0
    RIGHT_WRIST_FORCE = 1


class UltrasonicType(_SdkEnum):
    FRONT_LEFT = 0
    FRONT_RIGHT = 1
    RIGHT_LEFT = 2
    RIGHT_RIGHT = 3
    BACK_LEFT = 4
    BACK_RIGHT = 5
    LEFT_LEFT = 6
    LEFT_RIGHT = 7


class LogLevel(_SdkEnum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    CRITICAL = 5


class PerceptionModule(_SdkEnum):
    FOUNDATION_STEREO = 0   # High-precision stereo depth
    LIGHT_STEREO = 1        # Lightweight stereo depth


class SUCTION_ACTION_STATE(_SdkEnum):
    IDLE = 0
    SUCKING = 1
    SUCCESS = 2
    FAILED = 3


class RobotStatesType(_SdkEnum):
    POSE = 0
    JOINT = 1
    ROBOT_STATES = 2


class PrimitiveType(_SdkEnum):
    LINE = 0
    CYLINDER = 1


class SeedType(_SdkEnum):
    RANDOM_SEED = 0
    RANDOM_PROGRESSIVE_SEED = 1
    USER_DEFINED_SEED = 2


class StateCheckType(_SdkEnum):
    EUCLIDEAN_DISTANCE = 0
    RADIAN_DISTANCE = 1


class TerminationConditionType(_SdkEnum):
    TIMEOUT = 0
    TIMEOUT_AND_EXACT_SOLUTION = 1


class PointFieldDataType(_SdkEnum):
    UNKNOWN = 0
    INT8 = 1
    UINT8 = 2
    INT16 = 3
    UINT16 = 4
    INT32 = 5
    UINT32 = 6
    FLOAT32 = 7
    FLOAT64 = 8


class TargetSampling(_SdkEnum):
    TARGET_SAMPLING_DEFAULT = 0
    TARGET_SAMPLING_DIRECT_PASS = 1
    TARGET_SAMPLING_LINEAR_INTERPOLATE = 2
    TARGET_SAMPLING_TRAPEZOIDAL_PROFILE = 3
    TARGET_SAMPLING_S_CURVE_PROFILE = 4
    TARGET_SAMPLING_CUBIC_SPLINES = 5
    TARGET_SAMPLING_QUINTIC_SPLINES = 6
    TARGET_SAMPLING_B_SPLINES = 7
    TARGET_SAMPLING_CUSTOM = 15


# ---------------------------------------------------------------------------
# §5 string-constant classes (plain classes of ClassVar[str], verbatim)
# ---------------------------------------------------------------------------

class G1ControllerName:
    CHASSIS_POSE_CTRL = 'chassis_pose_ctrl'
    CHASSIS_TWIST_CTRL = 'chassis_twist_ctrl'
    CONTROLLER_NAME_NUM = 'CONTROLLER_NAME_NUM'
    HEAD_PVT_BYPASS_CTRL = 'head_pvt_bypass_ctrl'
    HEAD_PVT_CTRL = 'head_pvt_ctrl'
    LEFT_ARM_PVT_BYPASS_CTRL = 'left_arm_pvt_bypass_ctrl'
    LEFT_ARM_PVT_CTRL = 'left_arm_pvt_ctrl'
    LEFT_DEXHAND_CTRL = 'left_dexhand_ctrl'
    LEFT_GRIPPER_CTRL = 'left_gripper_ctrl'
    LEG_PVT_BYPASS_CTRL = 'leg_pvt_bypass_ctrl'
    LEG_PVT_CTRL = 'leg_pvt_ctrl'
    RIGHT_ARM_PVT_BYPASS_CTRL = 'right_arm_pvt_bypass_ctrl'
    RIGHT_ARM_PVT_CTRL = 'right_arm_pvt_ctrl'
    RIGHT_DEXHAND_CTRL = 'right_dexhand_ctrl'
    RIGHT_GRIPPER_CTRL = 'right_gripper_ctrl'


class G1JointGroup:
    chassis = 'chassis'
    head = 'head'
    left_arm = 'left_arm'
    left_dexhand = 'left_dexhand'
    left_gripper = 'left_gripper'
    left_suction_cup = 'left_suction_cup'
    leg = 'leg'
    right_arm = 'right_arm'
    right_dexhand = 'right_dexhand'
    right_gripper = 'right_gripper'
    right_suction_cup = 'right_suction_cup'


class S1ControllerName:
    ELEVATOR_CTRL = 'elevator_ctrl'
    HEAD_PVT_CTRL = 'head_pvt_ctrl'
    LEFT_ARM_PVT_CTRL = 'left_arm_pvt_ctrl'
    LEFT_CAMERA_CTRL = 'left_camera_ctrl'
    LEFT_GRIPPER_CTRL = 'left_gripper_ctrl'
    RIGHT_ARM_PVT_CTRL = 'right_arm_pvt_ctrl'
    RIGHT_CAMERA_CTRL = 'right_camera_ctrl'
    RIGHT_GRIPPER_CTRL = 'right_gripper_ctrl'
    SWERVE_CHASSIS_POSE_CTRL = 'swerve_chassis_pose_ctrl'
    SWERVE_CHASSIS_TWIST_CTRL = 'swerve_chassis_twist_ctrl'


class S1JointGroup:
    head = 'head'
    left_arm = 'left_arm'
    left_camera = 'left_camera'
    left_gripper = 'left_gripper'
    right_arm = 'right_arm'
    right_camera = 'right_camera'
    right_gripper = 'right_gripper'
    swerve_chassis = 'swerve_chassis'
    torso = 'torso'


# ---------------------------------------------------------------------------
# §5 module-level int constants (SingoriXTarget bitmasks)
# ---------------------------------------------------------------------------

TARGET_DATA_NONE = 0
TARGET_DATA_JOINT_POSITION = 1
TARGET_DATA_JOINT_VELOCITY = 2
TARGET_DATA_JOINT_ACCELERATION = 4
TARGET_DATA_JOINT_EFFORT = 8
TARGET_DATA_FRAME_POSE = 16
TARGET_DATA_FRAME_TWIST = 32
TARGET_DATA_FRAME_WRENCH = 64
TARGET_DATA_DEFAULT = 255

TARGET_TYPE_NONE = 0
TARGET_TYPE_TOUCH = 1
TARGET_TYPE_CLEAR = 2
TARGET_TYPE_PREPENDNOW = 4
TARGET_TYPE_APPEND = 8
TARGET_TYPE_OVERRIDE = 10
TARGET_TYPE_PROVERRIDE = 14
TARGET_TYPE_DEFAULT = 255


# ---------------------------------------------------------------------------
# Module-level enum aliases — exactly per the §4 stub tail.
#
# Every MotionStatus member EXCEPT SUCCESS/TIMEOUT binds to MotionStatus; the
# stub tail then declares:
#   SUCCESS: NavigationTaskStatus = <NavigationTaskStatus.SUCCESS: 2>
#   TIMEOUT: TerminationConditionType = <TerminationConditionType.TIMEOUT: 0>
# ---------------------------------------------------------------------------

# MotionStatus aliases (SUCCESS/TIMEOUT deliberately excluded — see stub tail)
FAULT = MotionStatus.FAULT
INVALID_INPUT = MotionStatus.INVALID_INPUT
INIT_FAILED = MotionStatus.INIT_FAILED
IN_PROGRESS = MotionStatus.IN_PROGRESS
STOPPED_UNREACHED = MotionStatus.STOPPED_UNREACHED
DATA_FETCH_FAILED = MotionStatus.DATA_FETCH_FAILED
PUBLISH_FAIL = MotionStatus.PUBLISH_FAIL
COMM_DISCONNECTED = MotionStatus.COMM_DISCONNECTED
STATUS_NUM = MotionStatus.STATUS_NUM
UNSUPPORTED_FUNCRION = MotionStatus.UNSUPPORTED_FUNCRION

# NavigationTaskStatus aliases (incl. the stub tail's SUCCESS binding)
UNKNOWN = NavigationTaskStatus.UNKNOWN
RUNNING = NavigationTaskStatus.RUNNING
SUCCESS = NavigationTaskStatus.SUCCESS
FAILED = NavigationTaskStatus.FAILED
INTERRUPTED = NavigationTaskStatus.INTERRUPTED
OCCUPIED = NavigationTaskStatus.OCCUPIED
COLLISION = NavigationTaskStatus.COLLISION
CLOSE_TO_OBSTACLE = NavigationTaskStatus.CLOSE_TO_OBSTACLE

# PrimitiveType aliases
LINE = PrimitiveType.LINE
CYLINDER = PrimitiveType.CYLINDER

# StateCheckType aliases
EUCLIDEAN_DISTANCE = StateCheckType.EUCLIDEAN_DISTANCE
RADIAN_DISTANCE = StateCheckType.RADIAN_DISTANCE

# PerceptionModule aliases
FOUNDATION_STEREO = PerceptionModule.FOUNDATION_STEREO
LIGHT_STEREO = PerceptionModule.LIGHT_STEREO

# RobotStatesType aliases
POSE = RobotStatesType.POSE
JOINT = RobotStatesType.JOINT
ROBOT_STATES = RobotStatesType.ROBOT_STATES

# SeedType aliases
RANDOM_SEED = SeedType.RANDOM_SEED
RANDOM_PROGRESSIVE_SEED = SeedType.RANDOM_PROGRESSIVE_SEED
USER_DEFINED_SEED = SeedType.USER_DEFINED_SEED

# TerminationConditionType aliases (incl. the stub tail's TIMEOUT binding)
TIMEOUT = TerminationConditionType.TIMEOUT
TIMEOUT_AND_EXACT_SOLUTION = TerminationConditionType.TIMEOUT_AND_EXACT_SOLUTION


__all__ = [
    # enum classes (§4)
    "ControlStatus", "MotionStatus", "TrajectoryControlStatus",
    "NavigationTaskStatus", "MachineType", "SensorType", "DexHandType",
    "GalbotOneFoxtrotSensor", "UltrasonicType", "LogLevel", "PerceptionModule",
    "SUCTION_ACTION_STATE", "RobotStatesType", "PrimitiveType", "SeedType",
    "StateCheckType", "TerminationConditionType", "PointFieldDataType",
    "TargetSampling",
    # string-constant classes (§5)
    "G1ControllerName", "G1JointGroup", "S1ControllerName", "S1JointGroup",
    # int constants (§5)
    "TARGET_DATA_NONE", "TARGET_DATA_JOINT_POSITION",
    "TARGET_DATA_JOINT_VELOCITY", "TARGET_DATA_JOINT_ACCELERATION",
    "TARGET_DATA_JOINT_EFFORT", "TARGET_DATA_FRAME_POSE",
    "TARGET_DATA_FRAME_TWIST", "TARGET_DATA_FRAME_WRENCH",
    "TARGET_DATA_DEFAULT", "TARGET_TYPE_NONE", "TARGET_TYPE_TOUCH",
    "TARGET_TYPE_CLEAR", "TARGET_TYPE_PREPENDNOW", "TARGET_TYPE_APPEND",
    "TARGET_TYPE_OVERRIDE", "TARGET_TYPE_PROVERRIDE", "TARGET_TYPE_DEFAULT",
    # module-level enum aliases (§4 stub tail)
    "FAULT", "INVALID_INPUT", "INIT_FAILED", "IN_PROGRESS",
    "STOPPED_UNREACHED", "DATA_FETCH_FAILED", "PUBLISH_FAIL",
    "COMM_DISCONNECTED", "STATUS_NUM", "UNSUPPORTED_FUNCRION",
    "UNKNOWN", "RUNNING", "SUCCESS", "FAILED", "INTERRUPTED", "OCCUPIED",
    "COLLISION", "CLOSE_TO_OBSTACLE",
    "LINE", "CYLINDER",
    "EUCLIDEAN_DISTANCE", "RADIAN_DISTANCE",
    "FOUNDATION_STEREO", "LIGHT_STEREO",
    "POSE", "JOINT", "ROBOT_STATES",
    "RANDOM_SEED", "RANDOM_PROGRESSIVE_SEED", "USER_DEFINED_SEED",
    "TIMEOUT", "TIMEOUT_AND_EXACT_SOLUTION",
]
