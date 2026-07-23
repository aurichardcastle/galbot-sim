"""galbot_sdk._robot — base ``GalbotRobot`` manager (GalbotSim core group).

Implements DESIGN.md §5.6 against the official surface in api_contract.md §8.1:
the pybind-style blocked constructor, the per-``MachineType`` ``get_instance``
cache, and all 58 instance methods with stub-exact signatures and defaults.

Behavior tiers (DESIGN.md §0/§5.6):
- T1 implemented: lifecycle, joint/base/gripper control, joint-state getters,
  trajectory execution, sensor getters, ``get_transform``/``get_odom``.
- T1-lite no-op-success: controller management, volume, WBC end-effector
  command, dexhand command.
- T2 benign-failure stubs (one logged warning per method per process):
  BMS/ultrasonic/force/log getters, dexhand state, suction cup, audio output.
- T3 no honest failure shape exists -> raise ``GalbotSimNotImplemented``:
  ``publish_target``, ``request_target``, ``start_microphone_stream_input``.

ControlStatus gate (DESIGN.md §4, normative), precedence
lifecycle gate -> validation -> execution:
- before ``init()`` succeeded            -> ``ControlStatus.INIT_FAILED``
- after ``request_shutdown``/``destroy`` -> ``ControlStatus.FAULT``
- bad names / wrong vector length / non-finite values -> ``INVALID_INPUT``
- valid commands complete fully, synchronously        -> ``SUCCESS``
- ``TIMEOUT`` is NEVER emitted; ``timeout_s``/speeds are accepted and ignored
  (enforcing the speed×timeout race would break tutorial acceptance — §0).
- Out-of-range joint targets are silently clipped by the world (then SUCCESS).

World interface consumed (DESIGN.md §3 growth seam / §5.5 — this module never
imports mujoco). Required primitives on the ``_world.get_world()`` singleton:
    init() -> bool                         (RuntimeError if the model XML is missing)
    request_shutdown()/wait_for_shutdown()/destroy()   (safe in any state)
    resolve(joint_groups, joint_names[, machine_type]) -> list[JointRef]
        raises LookupError/ValueError on unknown names; refs expose ``.name``
    get_positions(refs) -> list[float]
    set_positions(refs, values)            (clips to joint range, mj_forward)
    frame_pose(frame, ref) -> pose7 [x,y,z,qx,qy,qz,qw]  (raises on unknown frame)
    base_pose7() -> pose7 ; set_base_pose7(pose7)
    rgb(sensor)/depth(sensor)/lidar(sensor)/imu(sensor)/intrinsic(sensor) -> dict
    now_ns() -> int
Optional (robot-local fallbacks are used when absent):
    set_gripper_width(end_effector, width_m), gripper_state(end_effector),
    frame_names(), sensor_extrinsic(sensor, reference_frame), ik(chain, pose7, ref)
"""

import inspect
import math
import time

from ._enums import (
    ControlStatus,
    DexHandType,
    MachineType,
    MotionStatus,
    SUCTION_ACTION_STATE,
    TrajectoryControlStatus,
)
from ._types import (
    DepthData,
    GripperState,
    Header,
    JointState,
    JointStateMessage,
    RgbData,
    SuctionCupState,
    SyncedObservation,
    _unsupported,
)
from . import galbot_sdk_logger as _log

__all__ = ["GalbotRobot"]


# ---------------------------------------------------------------------------
# Constants (verified against planning/model_mapping.json)
# ---------------------------------------------------------------------------

# Gripper width domain (gripper_width_model LUT endpoints) and drive range.
_GRIPPER_WIDTH_MIN = 0.002756   # m, fully closed
_GRIPPER_WIDTH_MAX = 0.082774   # m, fully open
_GRIPPER_THETA_MAX = 1.6357     # rad, active_joint1 at fully closed
_GRIPPER_END_EFFECTORS = ("left_gripper", "right_gripper")

# Machine default joint-read order (api_contract.md §10.10).
_DEFAULT_GROUP_ORDER = {
    MachineType.G1: ("chassis", "head", "left_arm", "right_arm", "leg"),
    MachineType.S1: ("torso", "head", "left_arm", "right_arm"),
}

# Informational group list served by get_joint_group_names().
_ALL_GROUPS = {
    MachineType.G1: ("chassis", "head", "left_arm", "right_arm", "leg",
                     "left_gripper", "right_gripper"),
    MachineType.S1: ("torso", "head", "left_arm", "right_arm",
                     "left_gripper", "right_gripper"),
}

# WBC end-effector pose keys -> concrete S1 model frames (model_mapping.json).
_WBC_EE_FRAMES = {
    "lee_pose": "left_arm_end_effector_mount_link",
    "ree_pose": "right_arm_end_effector_mount_link",
    "head_pose": "head_end_effector_mount_link",
}

# Default active controller per group (S1ControllerName/G1ControllerName vocab).
_DEFAULT_CONTROLLERS = {
    "left_arm": "left_arm_pvt_ctrl",
    "right_arm": "right_arm_pvt_ctrl",
    "head": "head_pvt_ctrl",
    "leg": "leg_pvt_ctrl",
    "torso": "elevator_ctrl",
    "chassis": "chassis_pose_ctrl",
    "swerve_chassis": "swerve_chassis_pose_ctrl",
    "left_gripper": "left_gripper_ctrl",
    "right_gripper": "right_gripper_ctrl",
}

# One-log-line-per-method bookkeeping (per process, shared by all instances).
_logged_once = set()


def _log_once(key, message, level="warning"):
    """Emit ``message`` at most once per process for ``key``."""
    if key in _logged_once:
        return
    _logged_once.add(key)
    getattr(_log, level, _log.warning)(message)


# ---------------------------------------------------------------------------
# Small math / coercion helpers (SDK pose7 quaternions are xyzw)
# ---------------------------------------------------------------------------

def _yaw_of_quat_xyzw(q):
    x, y, z, w = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_xyzw_of_yaw(yaw):
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _finite_floats(values):
    """Coerce ``values`` into a list of finite floats.

    Raises ValueError/TypeError on non-numeric or non-finite input.
    """
    out = [float(v) for v in values]
    for v in out:
        if not math.isfinite(v):
            raise ValueError("non-finite value")
    return out


def _is_pose_like(obj):
    """Duck-typed check for a galbot_sdk ``Pose`` (position + orientation)."""
    return hasattr(obj, "position") and hasattr(obj, "orientation")


def _pose_to_pose7(pose):
    """galbot_sdk ``Pose`` -> [x, y, z, qx, qy, qz, qw]."""
    p, q = pose.position, pose.orientation
    return _finite_floats(
        [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
    )


def _sensor_frame_candidates(sensor_id):
    """Best-effort model frames for a SensorType (extrinsic fallback)."""
    name = getattr(sensor_id, "name", str(sensor_id))
    if "CAMERA" in name:
        if "LEFT_ARM" in name:
            return ("left_arm_camera_color_optical_frame",
                    "left_arm_end_effector_mount_link")
        if "RIGHT_ARM" in name:
            return ("right_arm_camera_color_optical_frame",
                    "right_arm_end_effector_mount_link")
        return ("head_camera", "head_end_effector_mount_link")
    return ("base_link",)


class GalbotRobot:
    """Base robot manager (pybind-lookalike).

    There is no public constructor — obtain the per-machine singleton via::

        robot = GalbotRobot.get_instance(MachineType.S1)

    Documented lifecycle (api_contract.md §1)::

        robot.init(); ...use...
        robot.request_shutdown(); robot.wait_for_shutdown(); robot.destroy()

    ``init()`` is effective only once; after ``destroy()`` the SDK cannot be
    re-initialized in the same process.
    """

    _instances = {}   # MachineType -> GalbotRobot

    # -- construction -------------------------------------------------------

    def __init__(self, *args, **kwargs):
        raise TypeError("galbot_sdk.GalbotRobot: No constructor defined!")

    def __repr__(self):
        return "<galbot_sdk.GalbotRobot object at 0x{:x}>".format(id(self))

    @staticmethod
    def get_instance(machine_type):
        """Return the process-wide ``GalbotRobot`` singleton for ``machine_type``."""
        try:
            mt = MachineType(int(machine_type))
        except (TypeError, ValueError):
            raise TypeError(
                "get_instance(): incompatible function arguments. "
                "Expected a galbot_sdk.MachineType, got {!r}".format(machine_type)
            )
        inst = GalbotRobot._instances.get(mt)
        if inst is None:
            inst = object.__new__(GalbotRobot)
            inst._setup(mt)
            GalbotRobot._instances[mt] = inst
        return inst

    def _setup(self, machine_type):
        """Private initializer (bypasses the blocked public constructor)."""
        self._machine_type = machine_type
        self._world_ref = None
        self._inited = False
        self._shutdown = False
        self._destroyed = False
        self._resolve_mode = None            # lazily probed world.resolve arity
        self._enabled_sensors = set()
        self._sync_mode = False
        self._traj_status = {}               # group -> TrajectoryControlStatus
        self._gripper_width = {}             # end_effector -> commanded width (m)
        self._suction = {}                   # end_effector -> activation bool
        self._dexhand = {}                   # end_effector -> last command list
        self._volume = 50.0
        self._controllers = {}               # group -> active controller name
        self._ee_command = None              # last WBC EE command tuple

    # -- world adapter ------------------------------------------------------

    def _now_ns(self):
        w = self._world_ref
        if w is not None:
            fn = getattr(w, "now_ns", None)
            if callable(fn):
                return int(fn())
        return time.time_ns()

    def _resolve(self, joint_groups, joint_names):
        """world.resolve with machine-default injection and arity adaptation.

        DESIGN.md §5.5 declares ``resolve(groups, names)``; a world that also
        takes ``machine_type`` (needed to disambiguate per-machine defaults)
        is detected once via ``inspect.signature`` and called accordingly.
        The robot always injects its own machine default when both arguments
        are empty, so a 2-argument world stays unambiguous.
        """
        w = self._world_ref
        groups = list(joint_groups or [])
        names = list(joint_names or [])
        if not groups and not names:
            groups = list(_DEFAULT_GROUP_ORDER[self._machine_type])
        if self._resolve_mode is None:
            mode = "pos2"
            try:
                sig = inspect.signature(w.resolve)
                if "machine_type" in sig.parameters:
                    mode = "kw"
                else:
                    positional = [
                        p for p in sig.parameters.values()
                        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                    ]
                    if len(positional) >= 3:
                        mode = "pos3"
            except (TypeError, ValueError):
                mode = "pos2"
            self._resolve_mode = mode
        if self._resolve_mode == "kw":
            return w.resolve(groups, names, machine_type=self._machine_type)
        if self._resolve_mode == "pos3":
            return w.resolve(groups, names, self._machine_type)
        return w.resolve(groups, names)

    def _world_dict(self, method, candidates, sensor):
        """Call the first available world sensor primitive; ``{}`` on failure."""
        if not self._ready(method):
            return {}
        w = self._world_ref
        for name in candidates:
            fn = getattr(w, name, None)
            if callable(fn):
                try:
                    d = fn(sensor)
                except Exception:
                    _log_once("err:" + method,
                              "GalbotSim: {}() could not synthesize data; "
                              "returning {{}}".format(method))
                    return {}
                return d if isinstance(d, dict) else {}
        _log_once("miss:" + method,
                  "GalbotSim: world backend has no primitive for {}(); "
                  "returning {{}}".format(method))
        return {}

    # -- gates --------------------------------------------------------------

    def _gate(self, method):
        """Lifecycle gate for ControlStatus-returning APIs (DESIGN.md §4).

        Returns None when the call may proceed, else the ControlStatus to
        return (INIT_FAILED before init, FAULT after shutdown/destroy).
        """
        if not self._inited:
            _log_once(
                "gate-init:" + method,
                "GalbotSim: {}() called before init() succeeded -> "
                "ControlStatus.INIT_FAILED".format(method))
            return ControlStatus.INIT_FAILED
        if self._shutdown or self._destroyed:
            _log_once(
                "gate-fault:" + method,
                "GalbotSim: {}() called after request_shutdown()/destroy() -> "
                "ControlStatus.FAULT".format(method))
            return ControlStatus.FAULT
        return None

    def _ready(self, method):
        """Lifecycle gate for data getters: empty defaults outside RUNNING."""
        if self._inited and not self._shutdown and not self._destroyed:
            return True
        _log_once(
            "ready:" + method,
            "GalbotSim: {}() called outside the RUNNING lifecycle state; "
            "returning empty data".format(method),
            level="debug")
        return False

    def _invalid(self, method, why):
        _log.warning("GalbotSim: {}() rejected input ({}) -> "
                     "ControlStatus.INVALID_INPUT".format(method, why))
        return ControlStatus.INVALID_INPUT

    def _stub_warn(self, method, shape):
        _log_once(
            "stub:" + method,
            "GalbotSim: {}() has no simulation backend; returning {} "
            "(benign stub, api_contract.md §8.1)".format(method, shape))

    def _apply_positions(self, method, values, joint_groups, joint_names):
        """Shared validate-and-write path for every joint position command."""
        try:
            vals = _finite_floats(values)
        except (TypeError, ValueError):
            return self._invalid(
                method, "joint positions must be a sequence of finite numbers")
        try:
            refs = self._resolve(joint_groups, joint_names)
        except (LookupError, ValueError):
            return self._invalid(method, "unknown joint group or joint name")
        if len(vals) != len(refs):
            return self._invalid(
                method, "expected {} values, got {}".format(len(refs), len(vals)))
        # Out-of-range targets are clipped inside world.set_positions (§4).
        self._world_ref.set_positions(refs, vals)
        return ControlStatus.SUCCESS

    # ======================================================================
    # Lifecycle / system (api_contract.md §8.1)
    # ======================================================================

    def init(self, enable_sensor_set=..., enable_sync_mode=False):
        """Initialize the robot (loads the MuJoCo world on first success).

        Only the first call has effect; repeat calls return True. After
        ``destroy()`` re-initialization is refused (returns False).
        ``enable_sensor_set`` sentinel ``...`` = default sensors; the sim
        serves every sensor getter regardless of the enable set (§5.5).
        """
        if self._destroyed:
            _log_once("init-after-destroy",
                      "GalbotSim: init() after destroy() is not supported in "
                      "the same process; returning False")
            return False
        if self._inited:
            return True
        if enable_sensor_set is Ellipsis or enable_sensor_set is None:
            sensors = set()
        else:
            sensors = set(enable_sensor_set)
        from ._world import get_world
        world = get_world()
        try:
            sig = inspect.signature(world.init)
            kwargs = {}
            if "machine_type" in sig.parameters:
                kwargs["machine_type"] = self._machine_type
            if "enable_sensor_set" in sig.parameters:
                kwargs["enable_sensor_set"] = sensors
            ok = bool(world.init(**kwargs))
        except (TypeError, ValueError):
            ok = bool(world.init())
        if not ok:
            return False
        self._world_ref = world
        self._enabled_sensors = sensors
        self._sync_mode = bool(enable_sync_mode)
        self._inited = True
        return True

    def is_running(self):
        """True while initialized and no shutdown signal has been captured."""
        return bool(self._inited and not self._shutdown and not self._destroyed)

    def request_shutdown(self):
        """Signal shutdown (non-blocking; step 1 of the shutdown contract)."""
        self._shutdown = True
        w = self._world_ref
        if w is not None:
            fn = getattr(w, "request_shutdown", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def wait_for_shutdown(self):
        """Block until shutdown completes (immediate: nothing runs async)."""
        w = self._world_ref
        if w is not None:
            fn = getattr(w, "wait_for_shutdown", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    def destroy(self):
        """Release resources (terminal; no re-init in the same process)."""
        self._destroyed = True
        w = self._world_ref
        if w is not None:
            fn = getattr(w, "destroy", None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    # ======================================================================
    # Joint state / kinematic info
    # ======================================================================

    def get_joint_group_names(self):
        """Names of the SDK joint groups for this machine type."""
        return list(_ALL_GROUPS[self._machine_type])

    def get_joint_names(self, only_active_joint=True, joint_groups=[]):
        """Joint names, ordered by ``joint_groups`` (machine default if empty).

        ``only_active_joint`` is accepted for signature fidelity; the sim's
        group tables already contain only SDK-active joints.
        """
        if not self._ready("get_joint_names"):
            return []
        try:
            refs = self._resolve(joint_groups, [])
        except (LookupError, ValueError):
            _log.warning("GalbotSim: get_joint_names() got an unknown joint "
                         "group; returning []")
            return []
        return [getattr(r, "name", str(r)) for r in refs]

    def get_joint_positions(self, joint_groups=[], joint_names=[]):
        """Joint positions; order joint_names > joint_groups > machine default."""
        if not self._ready("get_joint_positions"):
            return []
        try:
            refs = self._resolve(joint_groups, joint_names)
        except (LookupError, ValueError):
            _log.warning("GalbotSim: get_joint_positions() got an unknown "
                         "group/joint name; returning []")
            return []
        return [float(v) for v in self._world_ref.get_positions(refs)]

    def get_joint_states(self, joint_groups=[], joint_names=[]):
        """Per-joint ``JointState`` list (velocity/accel/effort/current = 0)."""
        if not self._ready("get_joint_states"):
            return []
        try:
            refs = self._resolve(joint_groups, joint_names)
        except (LookupError, ValueError):
            _log.warning("GalbotSim: get_joint_states() got an unknown "
                         "group/joint name; returning []")
            return []
        now = self._now_ns()
        states = []
        for pos in self._world_ref.get_positions(refs):
            js = JointState()
            js.position = float(pos)
            js.velocity = 0.0
            js.acceleration = 0.0
            js.effort = 0.0
            js.current = 0.0
            js.timestamp_ns = now
            states.append(js)
        return states

    def get_frame_names(self):
        """All frame names known to the world (bodies + aliases)."""
        if not self._ready("get_frame_names"):
            return []
        w = self._world_ref
        for name in ("frame_names", "get_frame_names"):
            fn = getattr(w, name, None)
            if callable(fn):
                try:
                    return list(fn())
                except Exception:
                    break
        _log_once("miss:get_frame_names",
                  "GalbotSim: world backend has no frame_names(); returning []")
        return []

    def get_transform(self, target_frame, source_frame, timestamp_ns=0,
                      timeout_ms=100):
        """``(pose7, timestamp_ns)`` of ``source_frame`` expressed in
        ``target_frame`` (ROS lookup semantics); ``([], ts)`` on failure.

        ``timestamp_ns``/``timeout_ms`` are accepted and ignored — the sim's
        kinematic state is always current (no TF history).
        """
        now = self._now_ns()
        if not self._ready("get_transform"):
            return ([], now)
        try:
            pose = self._world_ref.frame_pose(str(source_frame),
                                              str(target_frame))
            return ([float(v) for v in pose], now)
        except Exception:
            _log.warning("GalbotSim: get_transform() unknown frame pair "
                         "({!r}, {!r}); returning ([], ts)".format(
                             target_frame, source_frame))
            return ([], now)

    def get_sensor_extrinsic(self, sensor_id, reference_frame='base_link'):
        """``(pose7, timestamp_ns)`` of the sensor frame in ``reference_frame``;
        ``([], ts)`` on failure."""
        now = self._now_ns()
        if not self._ready("get_sensor_extrinsic"):
            return ([], now)
        w = self._world_ref
        fn = getattr(w, "sensor_extrinsic", None)
        if callable(fn):
            try:
                pose = fn(sensor_id, reference_frame)
                if pose:
                    return ([float(v) for v in pose], now)
            except Exception:
                pass
        for frame in _sensor_frame_candidates(sensor_id):
            try:
                pose = w.frame_pose(frame, str(reference_frame))
                return ([float(v) for v in pose], now)
            except Exception:
                continue
        _log_once("miss:get_sensor_extrinsic",
                  "GalbotSim: no extrinsic frame available for {!r}; "
                  "returning ([], ts)".format(sensor_id))
        return ([], now)

    def get_wbc_end_effector_poses(self):
        """``{'lee_pose'|'ree_pose'|'head_pose': pose7}`` via FK in base_link."""
        if not self._ready("get_wbc_end_effector_poses"):
            return {}
        poses = {}
        for key, frame in _WBC_EE_FRAMES.items():
            try:
                pose = self._world_ref.frame_pose(frame, "base_link")
                poses[key] = [float(v) for v in pose]
            except Exception:
                poses[key] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return poses

    # ======================================================================
    # Joint / trajectory control
    # ======================================================================

    def set_joint_positions(self, joint_positions, joint_groups=[],
                            joint_names=[], is_blocking=True, speed_rad_s=0.2,
                            timeout_s=15.0, *, max_speed=None):
        """Move the resolved joints to ``joint_positions`` (concatenated
        multi-group vectors accepted, ordered by ``joint_groups``).

        Stub-exact signature; the trailing keyword-only ``max_speed`` is a
        GalbotSim insurance alias for ``speed_rad_s`` (invisible to positional
        callers and to the stub). Speeds and ``timeout_s`` are accepted and
        ignored: the sim completes synchronously and never emits TIMEOUT (§4).
        ``is_blocking=False`` behaves identically (effect already applied).
        """
        gate = self._gate("set_joint_positions")
        if gate is not None:
            return gate
        del is_blocking, speed_rad_s, timeout_s, max_speed  # accepted, ignored
        return self._apply_positions(
            "set_joint_positions", joint_positions, joint_groups, joint_names)

    def set_joint_commands(self, joint_commands, joint_groups=[],
                           joint_names=[], time_from_start_s=0.0):
        """High-frequency command path; only ``.position`` is honored (§8.1)."""
        gate = self._gate("set_joint_commands")
        if gate is not None:
            return gate
        try:
            values = [float(cmd.position) for cmd in joint_commands]
        except (AttributeError, TypeError, ValueError):
            return self._invalid(
                "set_joint_commands",
                "joint_commands must be a sequence of JointCommand")
        return self._apply_positions(
            "set_joint_commands", values, joint_groups, joint_names)

    def set_joint_commands_batch(self, trajectory):
        """Non-blocking batch: applies the trajectory's final point."""
        gate = self._gate("set_joint_commands_batch")
        if gate is not None:
            return gate
        return self._execute_trajectory("set_joint_commands_batch", trajectory)

    def execute_joint_trajectory(self, trajectory, is_blocking=True):
        """Execute a ``Trajectory``: the final ``TrajectoryPoint``'s positions
        are applied synchronously (interpolation happens inside the world).

        INVALID_INPUT if ``trajectory.joint_groups`` and ``joint_names`` are
        both empty (§8.1) or the trajectory carries no points.
        """
        gate = self._gate("execute_joint_trajectory")
        if gate is not None:
            return gate
        del is_blocking  # sync engine: effect is fully applied either way
        return self._execute_trajectory("execute_joint_trajectory", trajectory)

    def _execute_trajectory(self, method, trajectory):
        groups = list(getattr(trajectory, "joint_groups", None) or [])
        names = list(getattr(trajectory, "joint_names", None) or [])
        points = list(getattr(trajectory, "points", None) or [])
        if not groups and not names:
            return self._invalid(
                method, "trajectory.joint_groups and joint_names are both empty")
        if not points:
            return self._invalid(method, "trajectory has no points")
        final = points[-1]
        try:
            values = [float(cmd.position) for cmd in final.joint_command_vec]
        except (AttributeError, TypeError, ValueError):
            return self._invalid(
                method, "final TrajectoryPoint has no valid joint_command_vec")
        status = self._apply_positions(method, values, groups, names)
        if status == ControlStatus.SUCCESS:
            for group in groups:
                self._traj_status[group] = TrajectoryControlStatus.COMPLETED
        return status

    def check_trajectory_execution_status(self, joint_groups=[]):
        """Per-group ``TrajectoryControlStatus`` list. Accepts ANY iterable
        (ex9 passes ``dict_keys``). RUNNING is unreachable in the synchronous
        engine; the registry defaults to COMPLETED (§4)."""
        try:
            groups = list(joint_groups)
        except TypeError:
            groups = []
        if not self._ready("check_trajectory_execution_status"):
            return [TrajectoryControlStatus.DATA_FETCH_FAILED] * len(groups)
        return [self._traj_status.get(g, TrajectoryControlStatus.COMPLETED)
                for g in groups]

    def stop_trajectory_execution(self):
        """SUCCESS no-op: nothing is ever in flight in the sync engine (§4)."""
        gate = self._gate("stop_trajectory_execution")
        if gate is not None:
            return gate
        return ControlStatus.SUCCESS

    # ======================================================================
    # Base control
    # ======================================================================

    def set_base_pose(self, *args, **kwargs):
        """Move the base. Three official overloads, dispatched on arg types:

        1. ``set_base_pose(base_pose: Pose, is_blocking=True, timeout_s=15.0)``
        2. ``set_base_pose(x, y, yaw, frame_id='odom', reference_frame_id='odom',
           is_blocking=True, timeout_s=15.0)``
        3. ``set_base_pose(x, y, yaw, frame_id, reference_frame_id,
           time_from_start_s, is_blocking=True, timeout_s=15.0)``

        Frames are all world-fixed in the sim (odom ≡ map ≡ world); timing
        parameters are accepted and ignored (synchronous completion, §4).
        """
        gate = self._gate("set_base_pose")
        if gate is not None:
            return gate
        a = list(args)
        if "base_pose" in kwargs or (a and _is_pose_like(a[0])):
            pose_arg = kwargs.get("base_pose", a[0] if a else None)
            try:
                pose7 = _pose_to_pose7(pose_arg)
            except (AttributeError, TypeError, ValueError):
                return self._invalid("set_base_pose", "invalid Pose argument")
            self._world_ref.set_base_pose7(pose7)
            return ControlStatus.SUCCESS
        # Numeric overloads (2/3): only x, y, yaw affect the sim; the frame
        # and timing slots differ between overloads but are ignored either way.
        def _arg(idx, key):
            if key in kwargs:
                return kwargs[key]
            if len(a) > idx:
                return a[idx]
            raise TypeError(
                "set_base_pose(): incompatible function arguments "
                "(missing {!r})".format(key))
        try:
            xyy = _finite_floats([_arg(0, "x"), _arg(1, "y"), _arg(2, "yaw")])
        except (TypeError, ValueError):
            return self._invalid("set_base_pose", "non-finite x/y/yaw")
        x, y, yaw = xyy
        current = self._world_ref.base_pose7()
        pose7 = [x, y, float(current[2])] + _quat_xyzw_of_yaw(yaw)
        self._world_ref.set_base_pose7(pose7)
        return ControlStatus.SUCCESS

    def set_base_velocity(self, linear_velocity, angular_velocity,
                          duration_s=0.0):
        """Integrate a body-frame velocity once over ``duration_s`` (single
        synchronous step of v×duration). ``duration_s <= 0`` means "no
        auto-stop" — a no-op in the synchronous engine (SUCCESS)."""
        gate = self._gate("set_base_velocity")
        if gate is not None:
            return gate
        try:
            lv = _finite_floats(linear_velocity)
            av = _finite_floats(angular_velocity)
            dt = float(duration_s)
        except (TypeError, ValueError):
            return self._invalid("set_base_velocity",
                                 "velocities must be finite 3-vectors")
        if len(lv) != 3 or len(av) != 3 or not math.isfinite(dt):
            return self._invalid("set_base_velocity",
                                 "expected FixedSize(3) velocity vectors")
        if dt <= 0.0:
            return ControlStatus.SUCCESS
        pose = [float(v) for v in self._world_ref.base_pose7()]
        yaw = _yaw_of_quat_xyzw(pose[3:7])
        wx = lv[0] * math.cos(yaw) - lv[1] * math.sin(yaw)
        wy = lv[0] * math.sin(yaw) + lv[1] * math.cos(yaw)
        new_yaw = yaw + av[2] * dt
        pose7 = ([pose[0] + wx * dt, pose[1] + wy * dt, pose[2]]
                 + _quat_xyzw_of_yaw(new_yaw))
        self._world_ref.set_base_pose7(pose7)
        return ControlStatus.SUCCESS

    def stop_base(self):
        """SUCCESS no-op: base motion always completes synchronously (§4)."""
        gate = self._gate("stop_base")
        if gate is not None:
            return gate
        return ControlStatus.SUCCESS

    def zero_whole_body_and_base(self, *args, **kwargs):
        """Zero arms/head/torso-or-leg and drive the base to the zero pose.

        Overload 1: ``(base_zero_pose: Pose, is_blocking=True,
        leg_head_speed_rad_s=0.2, leg_head_timeout_s=15.0, params=None)``.
        Overload 2: ``(frame_id='odom', reference_frame_id='odom', ...)`` —
        base goes to the origin (current height preserved).
        Returns ``(MotionStatus, ControlStatus)`` (§4/§8.1).
        """
        gate = self._gate("zero_whole_body_and_base")
        if gate is not None:
            return (MotionStatus(int(gate)), gate)
        a = list(args)
        base_pose7 = None
        if "base_zero_pose" in kwargs or (a and _is_pose_like(a[0])):
            try:
                base_pose7 = _pose_to_pose7(
                    kwargs.get("base_zero_pose", a[0] if a else None))
            except (AttributeError, TypeError, ValueError):
                return (MotionStatus.INVALID_INPUT, ControlStatus.INVALID_INPUT)
        try:
            groups = [g for g in _DEFAULT_GROUP_ORDER[self._machine_type]
                      if g != "chassis"]
            refs = self._resolve(groups, [])
            self._world_ref.set_positions(refs, [0.0] * len(refs))
            current = [float(v) for v in self._world_ref.base_pose7()]
            if base_pose7 is None:
                base_pose7 = [0.0, 0.0, current[2], 0.0, 0.0, 0.0, 1.0]
            self._world_ref.set_base_pose7(list(base_pose7))
        except (LookupError, ValueError):
            return (MotionStatus.INVALID_INPUT, ControlStatus.INVALID_INPUT)
        return (MotionStatus.SUCCESS, ControlStatus.SUCCESS)

    # ======================================================================
    # End effectors
    # ======================================================================

    def set_gripper_command(self, end_effector, width_m, velocity_mps=0.03,
                            effort=5, is_blocking=True):
        """Command a gripper width. ``width_m`` is clamped into the LUT domain
        [{:.6f}, {:.6f}] m, then SUCCESS (§4 — ex5/ex10 command 0.1 m).
        Velocity/effort/blocking accepted and ignored (synchronous engine).
        """.format(_GRIPPER_WIDTH_MIN, _GRIPPER_WIDTH_MAX)
        gate = self._gate("set_gripper_command")
        if gate is not None:
            return gate
        if end_effector not in _GRIPPER_END_EFFECTORS:
            return self._invalid(
                "set_gripper_command",
                "unknown end effector {!r} (expected one of {})".format(
                    end_effector, list(_GRIPPER_END_EFFECTORS)))
        try:
            width = float(width_m)
        except (TypeError, ValueError):
            return self._invalid("set_gripper_command",
                                 "width_m must be a finite number")
        if not math.isfinite(width):
            return self._invalid("set_gripper_command",
                                 "width_m must be a finite number")
        width = min(_GRIPPER_WIDTH_MAX, max(_GRIPPER_WIDTH_MIN, width))
        try:
            self._apply_gripper(end_effector, width)
        except (LookupError, ValueError):
            return self._invalid("set_gripper_command",
                                 "gripper group is not resolvable")
        self._gripper_width[end_effector] = width
        return ControlStatus.SUCCESS

    def _apply_gripper(self, end_effector, width_m):
        """Drive the gripper via the world's LUT primitive when available,
        else fall back to an endpoint-linear theta approximation."""
        w = self._world_ref
        for name in ("set_gripper_width", "set_gripper", "gripper_command"):
            fn = getattr(w, name, None)
            if callable(fn):
                fn(end_effector, width_m)
                return
        span = _GRIPPER_WIDTH_MAX - _GRIPPER_WIDTH_MIN
        frac = (_GRIPPER_WIDTH_MAX - width_m) / span
        theta = min(1.0, max(0.0, frac)) * _GRIPPER_THETA_MAX
        refs = self._resolve([end_effector], [])
        w.set_positions(refs, [theta] * len(refs))

    def get_gripper_state(self, end_effector):
        """``GripperState`` for 'left_gripper'/'right_gripper'; ``is_moving``
        is always False (nothing is ever in flight)."""
        now = self._now_ns()
        state = GripperState()
        width = self._gripper_width.get(end_effector, _GRIPPER_WIDTH_MAX)
        joints = []
        if self._ready("get_gripper_state") and \
                end_effector in _GRIPPER_END_EFFECTORS:
            w = self._world_ref
            read = None
            for name in ("gripper_state", "get_gripper_state"):
                fn = getattr(w, name, None)
                if callable(fn):
                    read = fn
                    break
            try:
                if read is not None:
                    res = read(end_effector)
                    if isinstance(res, dict):
                        width = float(res.get("width", width))
                        joints = [float(v) for v in
                                  res.get("joint_positions", [])]
                    else:
                        width = float(res[0])
                        joints = [float(v) for v in res[1]]
                else:
                    refs = self._resolve([end_effector], [])
                    theta = float(self._world_ref.get_positions(refs)[0])
                    joints = [theta, -theta, theta]
            except Exception:
                joints = []
        if not joints:
            span = _GRIPPER_WIDTH_MAX - _GRIPPER_WIDTH_MIN
            theta = min(1.0, max(0.0, (_GRIPPER_WIDTH_MAX - width) / span)) \
                * _GRIPPER_THETA_MAX
            joints = [theta, -theta, theta]
        state.width = float(width)
        state.velocity = 0.0
        state.effort = 0.0
        state.is_moving = False
        state.joint_positions = joints
        state.timestamp_ns = now
        return state

    def set_suction_cup_command(self, end_effector, activate):
        """T2 benign stub (G1 hardware): stores the activation, SUCCESS."""
        gate = self._gate("set_suction_cup_command")
        if gate is not None:
            return gate
        self._stub_warn("set_suction_cup_command", "ControlStatus.SUCCESS")
        self._suction[end_effector] = bool(activate)
        return ControlStatus.SUCCESS

    def get_suction_cup_state(self, end_effector):
        """T2 benign stub: default ``SuctionCupState`` (IDLE, no pressure)."""
        self._stub_warn("get_suction_cup_state", "a default SuctionCupState")
        state = SuctionCupState()
        try:
            state.action_state = SUCTION_ACTION_STATE.IDLE
            state.activation = self._suction.get(end_effector, False)
            state.pressure = 0.0
            state.timestamp_ns = self._now_ns()
        except AttributeError:
            pass
        return state

    def set_dexhand_command(self, end_effector, dexhand_command,
                            dexhand_type=..., is_blocking=True):
        """Store the dexhand command in a virtual register (no dexhand in the
        S1 model); sentinel ``dexhand_type=...`` -> ``DexHandType.INSPIRE``."""
        gate = self._gate("set_dexhand_command")
        if gate is not None:
            return gate
        if dexhand_type is Ellipsis or dexhand_type is None:
            dexhand_type = DexHandType.INSPIRE
        try:
            positions = [float(getattr(cmd, "position", cmd))
                         for cmd in dexhand_command]
        except (TypeError, ValueError):
            return self._invalid(
                "set_dexhand_command",
                "dexhand_command must be a sequence of JointCommand")
        self._dexhand[end_effector] = positions
        return ControlStatus.SUCCESS

    def get_dexhand_state(self, end_effector, dexhand_type=...):
        """T2 benign stub: None (the documented failure shape)."""
        self._stub_warn("get_dexhand_state", "None")
        return None

    def set_end_effector_command(self, poses, end_effector_frames,
                                 reference_frames=[]):
        """WBC pose command: rows ``[x,y,z,qx,qy,qz,qw]``. Applied via IK
        best-effort (chain guessed from the frame name); always SUCCESS."""
        gate = self._gate("set_end_effector_command")
        if gate is not None:
            return gate
        try:
            rows = [_finite_floats(row) for row in poses]
            frames = [str(f) for f in end_effector_frames]
        except (TypeError, ValueError):
            return self._invalid(
                "set_end_effector_command",
                "poses must be rows of 7 finite floats")
        for row, frame in zip(rows, frames):
            chain = None
            if "left" in frame:
                chain = "left_arm"
            elif "right" in frame:
                chain = "right_arm"
            if chain is None or len(row) != 7:
                continue
            try:
                ik = getattr(self._world_ref, "ik", None)
                if not callable(ik):
                    break
                solution = ik(chain, row, "base_link")
                if solution is None:
                    continue
                refs = self._resolve([chain], [])
                self._world_ref.set_positions(refs, list(solution))
            except Exception:
                continue   # best-effort per DESIGN §5.6 (no-op on failure)
        self._ee_command = (rows, frames, list(reference_frames or []))
        return ControlStatus.SUCCESS

    def clear_end_effector_command(self):
        """Clear the stored WBC command; SUCCESS."""
        gate = self._gate("clear_end_effector_command")
        if gate is not None:
            return gate
        self._ee_command = None
        return ControlStatus.SUCCESS

    # ======================================================================
    # Controller management (T1-lite: virtual registry, always SUCCESS)
    # ======================================================================

    def _controller_group(self, controller_name):
        name = str(controller_name)
        for group, default in _DEFAULT_CONTROLLERS.items():
            if name == default or name.startswith(group + "_"):
                return group
        return "all"

    def acquire_controller(self, controller_name):
        gate = self._gate("acquire_controller")
        if gate is not None:
            return gate
        self._controllers[self._controller_group(controller_name)] = \
            str(controller_name)
        return ControlStatus.SUCCESS

    def release_controller(self, group_name='all'):
        gate = self._gate("release_controller")
        if gate is not None:
            return gate
        if group_name == 'all':
            self._controllers.clear()
        else:
            self._controllers.pop(str(group_name), None)
        return ControlStatus.SUCCESS

    def reload_controller(self, group_name='all'):
        gate = self._gate("reload_controller")
        if gate is not None:
            return gate
        return ControlStatus.SUCCESS

    def start_controller(self, group_name='all'):
        gate = self._gate("start_controller")
        if gate is not None:
            return gate
        return ControlStatus.SUCCESS

    def stop_controller(self, group_name='all'):
        gate = self._gate("stop_controller")
        if gate is not None:
            return gate
        return ControlStatus.SUCCESS

    def switch_controller(self, controller_name):
        gate = self._gate("switch_controller")
        if gate is not None:
            return gate
        self._controllers[self._controller_group(controller_name)] = \
            str(controller_name)
        return ControlStatus.SUCCESS

    def get_active_controller(self, group_name):
        """Stored controller for the group, else the machine's default name."""
        group = str(group_name)
        return self._controllers.get(
            group, _DEFAULT_CONTROLLERS.get(group, ""))

    # ======================================================================
    # Sensors (dict-returning; {} on failure — api_contract.md §10.6)
    # Served regardless of the init enable-set (ex4 queries TORSO_IMU
    # without enabling it — DESIGN §5.5).
    # ======================================================================

    def get_rgb_data(self, camera_id):
        """``{header, format, data}`` with cv2-decodable compressed bytes."""
        return self._world_dict("get_rgb_data",
                                ("rgb", "get_rgb", "rgb_data"), camera_id)

    def get_depth_data(self, camera_id):
        """``{header, format, data, height, width, depth_scale}`` — data is a
        raw 720x1280 uint16 buffer (hard tutorial requirement)."""
        return self._world_dict("get_depth_data",
                                ("depth", "get_depth", "depth_data"), camera_id)

    def get_camera_intrinsic(self, camera_id):
        """``{header, width, height, distortion_model, D, K, ...}``."""
        return self._world_dict(
            "get_camera_intrinsic",
            ("intrinsic", "get_intrinsic", "camera_intrinsic"), camera_id)

    def get_imu_data(self, sensor_id):
        """``{timestamp_ns, accel, gyro, magnet}`` (gravity-only accel)."""
        return self._world_dict("get_imu_data",
                                ("imu", "get_imu", "imu_data"), sensor_id)

    def get_lidar_data(self, sensor_id):
        """PointCloud2-style dict (fields/point_step/row_step/data)."""
        return self._world_dict("get_lidar_data",
                                ("lidar", "get_lidar", "lidar_data"), sensor_id)

    def get_ultrasonic_data(self, ultrasonic_type):
        """T2 benign stub: {} (no ultrasonic array in the S1 model)."""
        self._stub_warn("get_ultrasonic_data", "{}")
        return {}

    def get_force_sensor_data(self, sensor_type):
        """T2 benign stub: {} (G1-only wrist force sensors)."""
        self._stub_warn("get_force_sensor_data", "{}")
        return {}

    def get_odom(self):
        """``{timestamp_ns, position, orientation, linear_velocity,
        angular_velocity}`` from the free joint; twist is always zero."""
        now = self._now_ns()
        if not self._ready("get_odom"):
            return {}
        try:
            pose = [float(v) for v in self._world_ref.base_pose7()]
        except Exception:
            _log_once("err:get_odom",
                      "GalbotSim: get_odom() could not read the base pose; "
                      "returning {}")
            return {}
        return {
            "timestamp_ns": now,
            "position": pose[:3],
            "orientation": pose[3:7],
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
        }

    def get_synced_observation(self, cameras, with_joint_state=True):
        """``SyncedObservation`` built from per-camera maps (+ joint state);
        None on failure (§10.6). All frames share one timestamp — the sim is
        trivially synchronized."""
        if not self._ready("get_synced_observation"):
            return None
        try:
            cams = list(cameras)
        except TypeError:
            return None
        try:
            obs = SyncedObservation()
            rgb_map, depth_map = {}, {}
            for cam in cams:
                name = getattr(cam, "name", str(cam))
                if "DEPTH" in name:
                    data = self.get_depth_data(cam)
                    if data:
                        depth_map[cam] = self._wrap_sensor_dict(
                            DepthData, data,
                            ("data", "format", "depth_scale", "height",
                             "width"))
                else:
                    data = self.get_rgb_data(cam)
                    if data:
                        rgb_map[cam] = self._wrap_sensor_dict(
                            RgbData, data, ("data", "format"))
            obs.rgb_data_map = rgb_map
            obs.depth_data_map = depth_map
            if with_joint_state:
                message = JointStateMessage()
                message.joint_state_vec = self.get_joint_states([], [])
                message.timestamp_ns = self._now_ns()
                obs.joint_state = message
            return obs
        except Exception:
            _log_once("err:get_synced_observation",
                      "GalbotSim: get_synced_observation() could not build "
                      "the observation; returning None")
            return None

    def _wrap_sensor_dict(self, cls, data, fields):
        """dict -> value-type instance (falls back to the raw dict if the
        target type refuses attribute writes)."""
        obj = cls()
        try:
            for field in fields:
                if field in data:
                    setattr(obj, field, data[field])
            header = data.get("header")
            if isinstance(header, Header):
                obj.header = header
            elif isinstance(header, dict):
                h = Header()
                h.frame_id = str(header.get("frame_id", ""))
                h.timestamp_ns = int(header.get("timestamp_ns", 0))
                obj.header = h
        except AttributeError:
            return data
        return obj

    def get_bms_information(self):
        """T2 benign stub: {} (no battery model in the sim)."""
        self._stub_warn("get_bms_information", "{}")
        return {}

    def get_device_information(self):
        """Static GalbotSim identity dict (DESIGN §5.6)."""
        model = "Galbot G1" if self._machine_type == MachineType.G1 \
            else "Galbot S1"
        return {
            "model": "{} (GalbotSim)".format(model),
            "serial_number": "GALBOTSIM-000001",
            "firmware_version": "1.9.0-sim",
            "hardware_version": "GalbotSim/MuJoCo",
            "manufacturer": "GalbotSim",
        }

    def get_log_information(self, timewindow_s, log_level):
        """T2 benign stub: {} (no on-robot log store in the sim)."""
        self._stub_warn("get_log_information", "{}")
        return {}

    # ======================================================================
    # Audio
    # ======================================================================

    def get_volume(self):
        """Virtual speaker volume, 0.0-100.0 (default 50.0)."""
        return float(self._volume)

    def set_volume(self, volume):
        """Store the virtual volume (clamped into [0, 100]); True on success."""
        try:
            value = float(volume)
        except (TypeError, ValueError):
            return False
        if not math.isfinite(value):
            return False
        self._volume = min(100.0, max(0.0, value))
        return True

    def start_microphone_stream_input(self, callback, chunk_size=2560,
                                      use_raw_audio=False):
        """T3: a returned stream_id would promise a live callback stream; no
        honest failure shape exists -> ``GalbotSimNotImplemented``."""
        _unsupported(
            "GalbotRobot.start_microphone_stream_input",
            "api_contract.md §8.1 (Audio)",
            "a stream_id return would promise live microphone callbacks the "
            "simulator cannot deliver")

    def stop_microphone_stream_input(self, stream_id=''):
        """T2 benign stub: no-op (nothing is ever streaming)."""
        self._stub_warn("stop_microphone_stream_input", "None (no-op)")
        return None

    def write_audio_stream_output(self, audio_chunk, stream_id=''):
        """T2 benign stub: False (the documented could-not-write shape)."""
        self._stub_warn("write_audio_stream_output", "False")
        return False

    def stop_audio_stream_output(self, stream_id=''):
        """T2 benign stub: no-op (nothing is ever playing)."""
        self._stub_warn("stop_audio_stream_output", "None (no-op)")
        return None

    # ======================================================================
    # Advanced WBC channel (T3 — no benign failure shape)
    # ======================================================================

    def publish_target(self, target):
        """T3: high-frequency WBC publish path -> ``GalbotSimNotImplemented``."""
        _unsupported(
            "GalbotRobot.publish_target",
            "api_contract.md §8.1 (Advanced WBC channel)",
            "SingoriXTarget streaming has no synchronous-sim equivalent")

    def request_target(self, target):
        """T3: WBC service path -> ``GalbotSimNotImplemented``."""
        _unsupported(
            "GalbotRobot.request_target",
            "api_contract.md §8.1 (Advanced WBC channel)",
            "SingoriXTarget service round-trips have no synchronous-sim "
            "equivalent")
