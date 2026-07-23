"""galbot_sdk._types — GalbotSim value-type surface (api_contract.md §7).

Every §7 name with exact attribute names (including the SDK's own typo
``Error.commpent``), pybind-style overloaded ``__init__`` where the stub has
overloads, the motion-plan-config family via a get/set accessor generator,
``WBCException``, and the shim-only ``GalbotSimNotImplemented`` +
``_unsupported()`` helper (DESIGN.md §5.4).

Fidelity notes / accepted deltas (DESIGN.md §6):
- Plain-attr value types are ``@dataclass``-style Python classes, not pybind
  value types; extra keyword construction is a permissive superset.
- Stub-``(ro)`` byte buffers on RgbData/DepthData are plain read/write attrs
  here so the sim world can populate them (DESIGN.md §5.4 "plain-attr types").
- Default field values not stated by the stub are chosen to be benign and are
  never observed by the tutorial contract.
"""

import copy
from dataclasses import dataclass, field

from ._enums import (
    NavigationTaskStatus,
    PointFieldDataType,
    PrimitiveType,
    RobotStatesType,
    SUCTION_ACTION_STATE,
    SeedType,
    StateCheckType,
    TARGET_DATA_DEFAULT,
    TARGET_TYPE_DEFAULT,
    TargetSampling,
    TerminationConditionType,
)


# ---------------------------------------------------------------------------
# Shim-only error surface
# ---------------------------------------------------------------------------

class WBCException(Exception):
    """Official SDK exception type (empty body, api_contract §7)."""


class GalbotSimNotImplemented(NotImplementedError):
    """Raised by GalbotSim for the few SDK methods that have no benign
    failure shape (DESIGN.md tier T3). Never reached by any tutorial path."""


def _unsupported(method, section, why):
    """Raise :class:`GalbotSimNotImplemented` with a self-describing message."""
    raise GalbotSimNotImplemented(
        f"GalbotSim does not implement {method}() (api_contract.md {section}): {why}. "
        "GalbotSim is a robot-free MuJoCo backend for the Galbot SDK Python API; "
        "see planning/DESIGN.md and planning/api_contract.md."
    )


# ---------------------------------------------------------------------------
# Geometry primitives (overloaded __init__, pybind style)
# ---------------------------------------------------------------------------

class Point:
    """§7 Point: ``__init__()`` | ``__init__(x=0.0, y=0.0, z=0.0)``."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, z={self.z})"


class Point2d:
    """§7 Point2d: ``__init__()`` | ``__init__(x=0.0, y=0.0)``."""

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def __repr__(self):
        return f"Point2d(x={self.x}, y={self.y})"


class Quaternion:
    """§7 Quaternion: ``__init__()`` | ``__init__(x=0.0, y=0.0, z=0.0, w=1.0)``."""

    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)

    def __repr__(self):
        return f"Quaternion(x={self.x}, y={self.y}, z={self.z}, w={self.w})"


class Pose:
    """§7 Pose: attrs ``position: Point``, ``orientation: Quaternion``.

    Overloads: ``__init__()`` | ``__init__(pos, quat)`` | ``__init__(vec)``
    with ``vec`` a len-7 ``[x, y, z, qx, qy, qz, qw]``.
    """

    def __init__(self, *args, pos=None, quat=None, vec=None):
        if args and (pos is not None or quat is not None or vec is not None):
            raise TypeError("Pose(): mixed positional and keyword overload use")
        if len(args) == 1:
            vec = args[0]
        elif len(args) == 2:
            pos, quat = args
        elif len(args) > 2:
            raise TypeError(f"Pose(): expected 0-2 positional args, got {len(args)}")

        if vec is not None:
            v = [float(x) for x in vec]
            if len(v) != 7:
                raise TypeError(f"Pose(vec): expected 7 elements, got {len(v)}")
            self.position = Point(*v[:3])
            self.orientation = Quaternion(*v[3:7])
        elif pos is not None or quat is not None:
            p = [float(x) for x in (pos if pos is not None else (0.0, 0.0, 0.0))]
            q = [float(x) for x in (quat if quat is not None else (0.0, 0.0, 0.0, 1.0))]
            if len(p) != 3 or len(q) != 4:
                raise TypeError("Pose(pos, quat): expected len-3 pos and len-4 quat")
            self.position = Point(*p)
            self.orientation = Quaternion(*q)
        else:
            self.position = Point()
            self.orientation = Quaternion()

    def __repr__(self):
        return f"Pose(position={self.position!r}, orientation={self.orientation!r})"


class Pose2d:
    """§7 Pose2d: attr ``position: Point2d``, prop ``theta: float``.

    Overloads: ``__init__()`` | ``__init__(vec)`` | ``__init__(x=0.0, y=0.0, theta=0.0)``.
    """

    def __init__(self, *args, x=0.0, y=0.0, theta=0.0):
        if len(args) == 1 and not isinstance(args[0], (int, float)):
            v = [float(e) for e in args[0]]
            if len(v) != 3:
                raise TypeError(f"Pose2d(vec): expected 3 elements, got {len(v)}")
            x, y, theta = v
        elif args:
            if len(args) > 3:
                raise TypeError(f"Pose2d(): expected 0-3 positional args, got {len(args)}")
            vals = [x, y, theta]
            for i, a in enumerate(args):
                vals[i] = a
            x, y, theta = vals
        self.position = Point2d(x, y)
        self.theta = float(theta)

    def __repr__(self):
        return f"Pose2d(position={self.position!r}, theta={self.theta})"


def _pose_to_list7(pose):
    """Pose | len-7 sequence -> ``[x, y, z, qx, qy, qz, qw]`` list of floats."""
    if isinstance(pose, Pose):
        p, q = pose.position, pose.orientation
        return [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    return [float(v) for v in pose]


# ---------------------------------------------------------------------------
# Plain-attribute data classes (§7, dataclass style)
# ---------------------------------------------------------------------------

@dataclass
class Header:
    frame_id: str = ""
    timestamp_ns: int = 0


@dataclass
class Timestamp:
    sec: int = 0
    nanosec: int = 0


@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Twist:
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)


@dataclass
class Wrench:
    force: Vector3 = field(default_factory=Vector3)
    torque: Vector3 = field(default_factory=Vector3)


@dataclass
class EffortInfo:
    force: Vector3 = field(default_factory=Vector3)
    torque: Vector3 = field(default_factory=Vector3)
    timestamp_ns: int = 0


@dataclass
class ForceData:
    force: Vector3 = field(default_factory=Vector3)
    torque: Vector3 = field(default_factory=Vector3)
    timestamp_ns: int = 0


@dataclass
class ImuData:
    accel: Vector3 = field(default_factory=Vector3)
    gyro: Vector3 = field(default_factory=Vector3)
    magnet: Vector3 = field(default_factory=Vector3)
    timestamp_ns: int = 0


@dataclass
class OdomData:
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])  # xyzw
    linear_velocity: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    angular_velocity: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    timestamp_ns: int = 0


@dataclass
class UltrasonicData:
    distance: float = 0.0  # meters
    timestamp_ns: int = 0


@dataclass
class RgbData:
    data: bytes = b""      # compressed image bytes
    format: str = ""       # e.g. 'jpeg'/'png' ('rgb8' observed on-wire)
    header: Header = field(default_factory=Header)


@dataclass
class DepthData:
    data: bytes = b""
    depth_scale: int = 1000
    format: str = ""       # e.g. 'depth16' / '16UC1'
    header: Header = field(default_factory=Header)
    height: int = 0
    width: int = 0


@dataclass
class AudioData:
    data: list = field(default_factory=list)
    format: str = "pcm"    # 'pcm' 16000Hz/16-bit/mono | 'json'
    header: Header = field(default_factory=Header)
    type: str = ""         # 'waken_up'|'denoise_chunk'|'vad_begin'|'vad_chunk'|'vad_end'


@dataclass
class PointField:
    name: str = ""
    offset: int = 0
    datatype: PointFieldDataType = PointFieldDataType.UNKNOWN
    count: int = 1


@dataclass
class LidarData:
    data: list = field(default_factory=list)
    fields: list = field(default_factory=list)  # list[PointField]
    header: Header = field(default_factory=Header)
    height: int = 0
    width: int = 0
    is_bigendian: bool = False
    is_dense: bool = True
    point_step: int = 0
    row_step: int = 0


@dataclass
class JointState:
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    effort: float = 0.0
    current: float = 0.0
    timestamp_ns: int = 0


@dataclass
class JointCommand:
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    effort: float = 0.0


@dataclass
class JointStateMessage:
    joint_state_vec: list = field(default_factory=list)  # list[JointState]
    timestamp_ns: int = 0


@dataclass
class GripperState:
    width: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0
    is_moving: bool = False
    joint_positions: list = field(default_factory=list)
    timestamp_ns: int = 0


@dataclass
class SuctionCupState:
    action_state: SUCTION_ACTION_STATE = SUCTION_ACTION_STATE.IDLE
    activation: bool = False
    pressure: float = 0.0
    timestamp_ns: int = 0


@dataclass
class DexhandState:
    force_sensor_map: dict = field(default_factory=dict)  # dict[str, EffortInfo] (Sharpa only)
    joint_state: JointStateMessage = field(default_factory=JointStateMessage)
    timestamp_ns: int = 0


class Error:
    """§7 Error: ``__init__()`` | ``__init__(commpent, error_code, description)``.

    ``commpent`` is the official SDK typo for "component" — preserved verbatim.
    """

    def __init__(self, commpent="", error_code=0, description=""):
        self.commpent = str(commpent)
        self.error_code = int(error_code)
        self.description = str(description)

    def __repr__(self):
        return (f"Error(commpent={self.commpent!r}, error_code={self.error_code}, "
                f"description={self.description!r})")


@dataclass
class ErrorInfo:
    error_vec: list = field(default_factory=list)  # list[Error]
    timestamp_ns: int = 0


@dataclass
class SyncedObservation:
    rgb_data_map: dict = field(default_factory=dict)    # dict[SensorType, RgbData]
    depth_data_map: dict = field(default_factory=dict)  # dict[SensorType, DepthData]
    joint_state: JointStateMessage = field(default_factory=JointStateMessage)


@dataclass
class FrameTriad:
    body_frame_id: str = ""
    reference_frame_id: str = ""
    header: Header = field(default_factory=Header)
    pose: Pose = None
    twist: Twist = None
    wrench: Wrench = None


@dataclass
class Trajectory:
    joint_groups: list = field(default_factory=list)  # list[str]
    joint_names: list = field(default_factory=list)   # list[str]
    points: list = field(default_factory=list)        # list[TrajectoryPoint]


@dataclass
class TrajectoryPoint:
    joint_command_vec: list = field(default_factory=list)  # list[JointCommand]
    time_from_start_second: float = 0.0


@dataclass
class GroupCommand:
    joint_commands: list = field(default_factory=list)  # list[JointCommand]
    time_from_start_s: float = 0.0


@dataclass
class TaskCommand:
    subtask_commands: list = field(default_factory=list)  # list[FrameTriad]
    time_from_start_s: float = 0.0


@dataclass
class TargetConfig:
    target_id: str = ""
    target_data: int = TARGET_DATA_DEFAULT
    target_type: int = TARGET_TYPE_DEFAULT
    target_priority: int = 0
    target_sampling: TargetSampling = TargetSampling.TARGET_SAMPLING_DEFAULT
    target_ts: Timestamp = field(default_factory=Timestamp)


@dataclass
class TargetGroupTrajectory:
    group_commands: list = field(default_factory=list)  # list[GroupCommand]
    joint_names: list = field(default_factory=list)
    target_config: TargetConfig = field(default_factory=TargetConfig)


@dataclass
class TargetTaskTrajectory:
    group_names: list = field(default_factory=list)
    joint_names: list = field(default_factory=list)
    subtask_names: list = field(default_factory=list)
    target_config: TargetConfig = field(default_factory=TargetConfig)
    task_commands: list = field(default_factory=list)  # list[TaskCommand]


@dataclass
class SingoriXTarget:
    header: Header = field(default_factory=Header)
    target_group_trajectory_map: dict = field(default_factory=dict)
    target_task_trajectory_map: dict = field(default_factory=dict)


@dataclass
class TaskHandle:
    task_id: str = ""
    request_sent: bool = False
    msg: str = ""


@dataclass
class NavigationTaskSnapshot:
    task_id: str = ""
    status: NavigationTaskStatus = NavigationTaskStatus.UNKNOWN
    msg: str = ""


class WaypointParams:
    """§7 WaypointParams: no-arg ctor, six float read/write properties."""

    def __init__(self):
        self.velocity_scale = 1.0
        self.acceleration_scale = 1.0
        self.jerk_scale = 1.0
        self.arrival_orientation_threshold = 0.1
        self.arrival_position_threshold_x = 0.1
        self.arrival_position_threshold_y = 0.1

    def __repr__(self):
        return (f"WaypointParams(velocity_scale={self.velocity_scale}, "
                f"acceleration_scale={self.acceleration_scale}, "
                f"jerk_scale={self.jerk_scale})")


class Waypoint:
    """§7 Waypoint: ``__init__(pose, params=...)`` — sentinel ``...`` means a
    default :class:`WaypointParams` (api_contract §10.5)."""

    def __init__(self, pose, params=...):
        self.pose = pose
        self.params = WaypointParams() if params is Ellipsis or params is None else params

    def __repr__(self):
        return f"Waypoint(pose={self.pose!r}, params={self.params!r})"


# ---------------------------------------------------------------------------
# Perception result types (§7)
# ---------------------------------------------------------------------------

class DetectionAndSegmentationResult:
    """§7: props ``bbox`` (ro), ``class_index``, ``class_name``, ``confidence``,
    ``keypoints`` (ro)."""

    def __init__(self):
        self._bbox = (0, 0, 0, 0)  # x, y, w, h
        self.class_index = -1
        self.class_name = ""
        self.confidence = 0.0
        self._keypoints = []       # list[tuple[float, float]]

    @property
    def bbox(self):
        return self._bbox

    @property
    def keypoints(self):
        return self._keypoints

    def __repr__(self):
        return (f"DetectionAndSegmentationResult(class_name={self.class_name!r}, "
                f"class_index={self.class_index}, confidence={self.confidence}, "
                f"bbox={self._bbox})")


class DetectionResult:
    """§7 DetectionResult: ``clear()``, ``get_result_info()``, rw + ro props."""

    def __init__(self):
        self.clear()

    def clear(self):
        self._bounding_boxes = []       # ro: list[tuple[int,int,int,int]]
        self.class_indices = []
        self.class_names = []
        self.confidences = []
        self.detection_results = []     # list[DetectionAndSegmentationResult]
        self.grasp_pose_result = []     # list[list[float]]
        self._instance_mask = None      # ro: numpy HxW/HxWxC or None
        self.ocr_string = []
        self._point_clouds = []         # ro: Nx3 numpy arrays
        self.running_info = ""
        self.sensor_name = ""
        self._target_point_poses = []   # ro: list[4x4 float32]
        self._target_poses = []         # ro: list[4x4 float32]
        self.timestamp_ns = 0

    def get_result_info(self):
        return (f"DetectionResult(sensor_name={self.sensor_name!r}, "
                f"classes={self.class_names}, confidences={self.confidences}, "
                f"detections={len(self.detection_results)}, "
                f"running_info={self.running_info!r}, "
                f"timestamp_ns={self.timestamp_ns})")

    @property
    def bounding_boxes(self):
        return self._bounding_boxes

    @property
    def instance_mask(self):
        return self._instance_mask

    @property
    def point_clouds(self):
        return self._point_clouds

    @property
    def target_point_poses(self):
        return self._target_point_poses

    @property
    def target_poses(self):
        return self._target_poses

    def __repr__(self):
        return self.get_result_info()


# ---------------------------------------------------------------------------
# Robot-state types for motion planning (§7)
# ---------------------------------------------------------------------------

class RobotStates:
    """§7 RobotStates: ``__init__()`` | ``__init__(chain, whole_joint, base_pose)``.

    ``base_state``/``whole_body_joint`` are settable properties (example6
    assigns both directly); ``chain_name`` is a permissive plain attribute
    (DESIGN.md §5.4, example9).
    """

    def __init__(self, chain="", whole_joint=None, base_pose=None):
        self.chain_name = str(chain)
        self._whole_body_joint = (
            [float(v) for v in whole_joint] if whole_joint is not None else []
        )
        self._base_state = (
            _pose_to_list7(base_pose) if base_pose is not None
            else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )

    def get_type(self):
        return RobotStatesType.ROBOT_STATES

    def set_base_state(self, base_pose):
        self.base_state = base_pose

    def set_whole_body_joint(self, joint_positions):
        self.whole_body_joint = joint_positions

    @property
    def base_state(self):
        return self._base_state

    @base_state.setter
    def base_state(self, value):
        self._base_state = _pose_to_list7(value)

    @property
    def whole_body_joint(self):
        return self._whole_body_joint

    @whole_body_joint.setter
    def whole_body_joint(self, value):
        self._whole_body_joint = [float(v) for v in value]

    def __repr__(self):
        return (f"{type(self).__name__}(chain_name={self.chain_name!r}, "
                f"type={self.get_type()!s})")


class JointStates(RobotStates):
    """§7 JointStates(RobotStates): joint-space target state (example9)."""

    def __init__(self):
        super().__init__()
        self._joint_positions = []

    def get_type(self):
        return RobotStatesType.JOINT

    def set_joint(self, index, val):
        # Stub quirk (api_contract §10.4): `val` is typed SupportsInt; be
        # permissive and store as float. Grows the vector when needed.
        i = int(index)
        if i < 0:
            raise IndexError(f"JointStates.set_joint: negative index {i}")
        if i >= len(self._joint_positions):
            self._joint_positions.extend(
                0.0 for _ in range(i + 1 - len(self._joint_positions))
            )
        self._joint_positions[i] = float(val)

    def set_joint_positions(self, joints):
        self.joint_positions = joints

    @property
    def joint_positions(self):
        return self._joint_positions

    @joint_positions.setter
    def joint_positions(self, value):
        self._joint_positions = [float(v) for v in value]


class PoseState(RobotStates):
    """§7 PoseState(RobotStates): Cartesian target state (example9).

    Plain attrs ``frame_id``, ``pose``, ``reference_frame`` (defaults mirror
    the SDK-wide frame defaults: 'EndEffector' on 'base_link').
    """

    def __init__(self):
        super().__init__()
        self.frame_id = "EndEffector"
        self.pose = Pose()
        self.reference_frame = "base_link"

    def get_type(self):
        return RobotStatesType.POSE


# ---------------------------------------------------------------------------
# Parameter (motion-plan options, §7)
# ---------------------------------------------------------------------------

class Parameter:
    """§7 Parameter: ``__init__()`` | ``__init__(direct_execute, blocking,
    timeout, actuate, tool_pose, check_collision, frame='base_link')``.

    Plain attrs: ``actuate_type`` (untyped — stub quirk), ``is_blocking``,
    ``is_check_collision``, ``is_direct_execute``, ``is_tool_pose``,
    ``reference_frame``; props ``joint_state``, ``timeout_second``.
    """

    def __init__(self, direct_execute=True, blocking=True, timeout=15.0,
                 actuate="position", tool_pose=False, check_collision=True,
                 frame="base_link"):
        self.is_direct_execute = bool(direct_execute)
        self.is_blocking = bool(blocking)
        self.timeout_second = float(timeout)
        self.actuate_type = actuate            # untyped in the stub — preserved
        self.is_tool_pose = bool(tool_pose)
        self.is_check_collision = bool(check_collision)
        self.reference_frame = str(frame)
        self.joint_state = {}                  # dict[str, list[float]]
        self._move_line = False

    # -- getters ------------------------------------------------------------
    def get_actuate_type(self):
        return self.actuate_type

    def get_blocking(self):
        return self.is_blocking

    def get_check_collision(self):
        return self.is_check_collision

    def get_direct_execute(self):
        return self.is_direct_execute

    def get_reference_frame(self):
        return self.reference_frame

    def get_timeout(self):
        return self.timeout_second

    def get_tool_pose(self):
        return self.is_tool_pose

    # -- setters ------------------------------------------------------------
    def set_actuate(self, actuate):
        self.actuate_type = actuate

    def set_blocking(self, blocking):
        self.is_blocking = bool(blocking)

    def set_check_collision(self, check_collision):
        self.is_check_collision = bool(check_collision)

    def set_direct_execute(self, direct_execute):
        self.is_direct_execute = bool(direct_execute)

    def set_move_line(self, move_line):
        self._move_line = bool(move_line)

    def set_reference_frame(self, frame):
        self.reference_frame = str(frame)

    def set_timeout(self, timeout):
        self.timeout_second = float(timeout)

    def set_tool_pose(self, tool_pose):
        self.is_tool_pose = bool(tool_pose)

    def __repr__(self):
        return (f"Parameter(is_direct_execute={self.is_direct_execute}, "
                f"is_blocking={self.is_blocking}, "
                f"timeout_second={self.timeout_second}, "
                f"actuate_type={self.actuate_type!r}, "
                f"is_tool_pose={self.is_tool_pose}, "
                f"is_check_collision={self.is_check_collision}, "
                f"reference_frame={self.reference_frame!r})")


# ---------------------------------------------------------------------------
# Motion-plan config family (§7) — accessor generator keeps this compact
# ---------------------------------------------------------------------------

class _ConfigBase:
    """Base for the MotionPlanConfig sub-config family.

    Subclasses declare ``_FIELDS = {name: default_factory}`` (+ optional
    ``_GET_ONLY = (names,)``); :func:`_gen_accessors` attaches the
    ``get_<name>``/``set_<name>`` pairs. All have ``print() -> None``.
    """

    _FIELDS = {}
    _GET_ONLY = ()

    def __init__(self):
        for name, factory in type(self)._FIELDS.items():
            setattr(self, "_" + name, factory())

    def print(self):
        print(repr(self))

    def __repr__(self):
        parts = ", ".join(
            f"{n}={getattr(self, '_' + n)!r}" for n in type(self)._FIELDS
        )
        return f"{type(self).__name__}({parts})"


def _gen_accessors(cls):
    """Attach ``get_<field>`` (all fields) and ``set_<field>`` (unless the
    field is in ``cls._GET_ONLY``) for every entry of ``cls._FIELDS``."""
    for fname in cls._FIELDS:
        def _get(self, _n=fname):
            return getattr(self, "_" + _n)

        _get.__name__ = "get_" + fname
        setattr(cls, "get_" + fname, _get)

        if fname not in cls._GET_ONLY:
            def _set(self, value, _n=fname):
                setattr(self, "_" + _n, value)

            _set.__name__ = "set_" + fname
            setattr(cls, "set_" + fname, _set)
    return cls


@_gen_accessors
class CollisionCheckOption(_ConfigBase):
    _FIELDS = {
        "disable_env_collision_check": lambda: False,
        "disable_self_collision_check": lambda: False,
    }


@_gen_accessors
class IKSolverConfig(_ConfigBase):
    _FIELDS = {
        "col_aware_ik_joint_limit_bias": lambda: 0.0,
        "col_aware_ik_timeout": lambda: 0.05,
        "enable_collision_check_log": lambda: False,
        "rotation_eps": lambda: [1e-3, 1e-3, 1e-3],
        "translation_eps": lambda: [1e-3, 1e-3, 1e-3],
        "seed_type": lambda: SeedType.RANDOM_SEED,
    }


@_gen_accessors
class LineTrajCheckPrimitive(_ConfigBase):
    _FIELDS = {
        "cylinder_prim_radius": lambda: 0.0,
        "line_check_primitive_type": lambda: PrimitiveType.LINE,
        "line_prim_curvature": lambda: 0.0,
    }


@_gen_accessors
class SamplerConfig(_ConfigBase):
    _FIELDS = {
        "interpolate": lambda: True,
        "interpolation_cnt": lambda: 10,
        "max_planning_time": lambda: 5.0,
        "max_simplification_time": lambda: 1.0,
        "simplify": lambda: True,
        "state_check_resolution": lambda: 0.01,
        "state_check_type": lambda: StateCheckType.EUCLIDEAN_DISTANCE,
        "termination_condition_type": lambda: TerminationConditionType.TIMEOUT,
    }


@_gen_accessors
class TrajectoryFeasibilityCheckOption(_ConfigBase):
    _FIELDS = {
        "disable_collision_check": lambda: False,
        "disable_joint_limit_check": lambda: False,
        "disable_velocity_feasibility_check": lambda: False,
    }


@_gen_accessors
class TrajectoryPlanConfig(_ConfigBase):
    _FIELDS = {
        "min_move_time": lambda: 0.0,
        "move_line_intermediate_point": lambda: 0.0,
        "way_point_plan_expected_time": lambda: 0.0,
    }


@_gen_accessors
class KinematicsBoundary(_ConfigBase):
    # Stub fidelity (api_contract §7): NO set_acc_* setters exist.
    _FIELDS = {
        "chain_name": lambda: "",
        "acc_lower_limit": lambda: [],
        "acc_upper_limit": lambda: [],
        "jerk_lower_limit": lambda: [],
        "jerk_upper_limit": lambda: [],
        "lower_limit": lambda: [],
        "upper_limit": lambda: [],
        "vel_lower_limit": lambda: [],
        "vel_upper_limit": lambda: [],
    }
    _GET_ONLY = ("acc_lower_limit", "acc_upper_limit")


_MPC_SUBCONFIGS = {
    "collision_check_option": CollisionCheckOption,
    "ik_solver_config": IKSolverConfig,
    "line_traj_check_primitive": LineTrajCheckPrimitive,
    "sampler_config": SamplerConfig,
    "trajectory_feasibility_check_option": TrajectoryFeasibilityCheckOption,
    "trajectory_plan_config": TrajectoryPlanConfig,
}


@_gen_accessors
class MotionPlanConfig(_ConfigBase):
    """§7 MotionPlanConfig: ``create_*`` factories; ``get_*`` (by value),
    ``get_*_ref`` (live reference), ``set_*`` for each sub-config; plain
    get/set for the boundary/limit/revert/update fields; ``print()``."""

    _FIELDS = {
        "feasibility_boundary": lambda: [],
        "hard_joint_limit": lambda: [],
        "ik_joint_limit": lambda: [],
        "sampler_joint_limit": lambda: [],
        "revert_ik_joint_limit": lambda: False,
        "revert_ik_joint_limit_chains": lambda: [],
        "update_time": lambda: 0,
    }

    def __init__(self):
        super().__init__()
        for name, sub_cls in _MPC_SUBCONFIGS.items():
            setattr(self, "_" + name, sub_cls())

    def __repr__(self):
        parts = ", ".join(
            f"{n}={getattr(self, '_' + n)!r}"
            for n in list(_MPC_SUBCONFIGS) + list(type(self)._FIELDS)
        )
        return f"MotionPlanConfig({parts})"


def _mpc_attach_subconfig_methods():
    """Attach create_/get_/get_*_ref/set_ methods for each sub-config."""
    for name, sub_cls in _MPC_SUBCONFIGS.items():
        def _create(self, _c=sub_cls):
            return _c()

        def _get(self, _n=name):
            return copy.deepcopy(getattr(self, "_" + _n))  # pybind by-value get

        def _get_ref(self, _n=name):
            return getattr(self, "_" + _n)                 # live reference

        def _set(self, value, _n=name):
            setattr(self, "_" + _n, copy.deepcopy(value))  # pybind by-value set

        _create.__name__ = "create_" + name
        _get.__name__ = "get_" + name
        _get_ref.__name__ = "get_" + name + "_ref"
        _set.__name__ = "set_" + name
        setattr(MotionPlanConfig, _create.__name__, _create)
        setattr(MotionPlanConfig, _get.__name__, _get)
        setattr(MotionPlanConfig, _get_ref.__name__, _get_ref)
        setattr(MotionPlanConfig, _set.__name__, _set)


_mpc_attach_subconfig_methods()


__all__ = [
    # geometry
    "Point", "Point2d", "Quaternion", "Pose", "Pose2d",
    # plain-attr data classes
    "Header", "Timestamp", "Vector3", "Twist", "Wrench", "EffortInfo",
    "ForceData", "ImuData", "OdomData", "UltrasonicData", "RgbData",
    "DepthData", "AudioData", "PointField", "LidarData", "JointState",
    "JointCommand", "JointStateMessage", "GripperState", "SuctionCupState",
    "DexhandState", "Error", "ErrorInfo", "SyncedObservation", "FrameTriad",
    "Trajectory", "TrajectoryPoint", "GroupCommand", "TaskCommand",
    "TargetConfig", "TargetGroupTrajectory", "TargetTaskTrajectory",
    "SingoriXTarget", "TaskHandle", "NavigationTaskSnapshot",
    "WaypointParams", "Waypoint",
    # perception results
    "DetectionAndSegmentationResult", "DetectionResult",
    # robot states
    "RobotStates", "JointStates", "PoseState",
    # motion-plan options / config family
    "Parameter", "CollisionCheckOption", "IKSolverConfig",
    "LineTrajCheckPrimitive", "SamplerConfig",
    "TrajectoryFeasibilityCheckOption", "TrajectoryPlanConfig",
    "KinematicsBoundary", "MotionPlanConfig",
    # exceptions
    "WBCException", "GalbotSimNotImplemented",
]
