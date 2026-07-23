"""galbot_sdk._motion — base GalbotMotion / GalbotNavigation / GalbotPerception.

GalbotSim "motion" group (DESIGN.md §5.7, api_contract.md §6 / §8.2–§8.4).

All three manager classes follow the pybind pattern of the real SDK:
no public constructor (``__init__`` raises TypeError) and a per-``MachineType``
cached ``get_instance`` staticmethod.  The ``g1``/``s1`` wrapper packages bind
these bases to a fixed machine type and delegate via ``__getattr__``.

Engine model (DESIGN.md §3): synchronous, step-on-demand, zero threads.
Every blocking call performs its entire effect inline and returns SUCCESS;
``is_blocking=False`` behaves identically.  TIMEOUT is never emitted for a
reachable target (load-bearing for the tutorial PASS matrix).

Documented fidelity deltas (accepted by DESIGN.md §6):
- ``check_collision`` is registry-only: obstacles/tools are stored but not
  geometrically tested → always ``(SUCCESS, [False] * len(start))``.
- ``get_jacobian(..., joint_state=...)`` / ``*_by_state`` compute the live
  Jacobian (the reference state is accepted and ignored, one debug log).
- Waypoint IK in ``motion_plan_multi_waypoints`` is best-effort: an
  unreachable waypoint reuses the previous joint solution (logged) so a plan
  is always non-empty for reachable-neighbourhood requests (tutorial 9).
- Navigation is instantaneous: goals are applied synchronously to the base
  free joint; ``is_localized()`` is True from init (kills the ex3/ex10
  localization polling loops); ``check_goal_arrival()`` compares the stored
  goal to the live base pose.
"""

import json
import math
import uuid

from ._enums import MachineType, MotionStatus, NavigationTaskStatus
from ._types import (
    DetectionResult,
    JointStates,
    MotionPlanConfig,
    NavigationTaskSnapshot,
    Parameter,
    PoseState,
    RobotStates,
    TaskHandle,
)
from ._world import get_world
from .galbot_sdk_logger import debug as _log_debug
from .galbot_sdk_logger import warning as _log_warning

# ---------------------------------------------------------------------------
# Model vocabulary (verified against planning/model_mapping.json)
# ---------------------------------------------------------------------------

#: Kinematic-chain DOF counts.  "leg" is the emulated G1 group (5 virtual
#: joints, knee -> torso lift); "torso" is the native S1 prismatic lift.
_CHAIN_DOF = {
    "left_arm": 7,
    "right_arm": 7,
    "head": 2,
    "leg": 5,
    "torso": 1,
}

#: Whole-body group order used by every tutorial (21 DOF total).
_WHOLE_BODY_GROUPS = ("leg", "head", "left_arm", "right_arm")

#: Per-machine kinematic chains (contract §10.10 read orders, chain flavour).
_MACHINE_CHAINS = {
    MachineType.G1: ("leg", "head", "left_arm", "right_arm"),
    MachineType.S1: ("torso", "head", "left_arm", "right_arm"),
}

#: Chain -> concrete end-effector frame in the S1 MJCF.
_CHAIN_EE_FRAME = {
    "left_arm": "left_arm_end_effector_mount_link",
    "right_arm": "right_arm_end_effector_mount_link",
    "head": "head_end_effector_mount_link",
    "torso": "torso_lift_link1",
    "leg": "torso_lift_link1",
}

#: S1 link names (bodies from model_mapping.json plus the EE mount links).
_S1_LINKS = (
    "base_link",
    "torso_lift_link1",
    "left_arm_link1", "left_arm_link2", "left_arm_link3", "left_arm_link4",
    "left_arm_link5", "left_arm_link6", "left_arm_link7",
    "left_active_link1", "left_active_link2", "left_passive_link",
    "right_arm_link1", "right_arm_link2", "right_arm_link3", "right_arm_link4",
    "right_arm_link5", "right_arm_link6", "right_arm_link7",
    "right_active_link1", "right_active_link2", "right_passive_link",
    "head_link1", "head_link2",
    "wheel_steering_link1", "wheel_driving_link1",
    "wheel_steering_link2", "wheel_driving_link2",
    "wheel_steering_link3", "wheel_driving_link3",
    "wheel_steering_link4", "wheel_driving_link4",
    "left_arm_end_effector_mount_link",
    "right_arm_end_effector_mount_link",
    "head_end_effector_mount_link",
)

_EE_FRAMES = (
    "left_arm_end_effector_mount_link",
    "right_arm_end_effector_mount_link",
    "head_end_effector_mount_link",
)

_OBSTACLE_TYPES = ("box", "sphere", "cylinder", "mesh", "point_cloud", "depth_image")

_TOOL_LIST = ("galbot_gripper",)

#: Points per linear-interpolation segment (DESIGN.md §5.7).
_PLAN_POINTS = 10

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_WORLD_READY = False
_WARNED = set()


def _warn_once(key, msg):
    """Emit one warning per process for *key* (T2 stub etiquette)."""
    if key not in _WARNED:
        _WARNED.add(key)
        _log_warning(msg)


def _require_world():
    """Return the initialized world singleton.

    Idempotent; raises RuntimeError (from the world) if the MuJoCo model
    cannot be loaded — never at import time (DESIGN.md §5.5).
    """
    global _WORLD_READY
    w = get_world()
    if not _WORLD_READY or not w.is_running():
        if not w.is_running() and _WORLD_READY:
            _WORLD_READY = False
        if not _WORLD_READY:
            if not w.init():
                raise RuntimeError(
                    "GalbotSim world failed to initialize (see log for details)"
                )
            _WORLD_READY = True
    return w


def _registry(world, name):
    """Fetch a plain-dict registry off the world, creating it if absent."""
    reg = getattr(world, name, None)
    if reg is None:
        reg = {}
        try:
            setattr(world, name, reg)
        except Exception:  # pragma: no cover - world with __slots__
            pass
    return reg


def _as_parameter(params):
    """Resolve the stub's ``params: Parameter = ...`` sentinel (§10.5)."""
    return params if isinstance(params, Parameter) else Parameter()


def _finite(values):
    try:
        return all(math.isfinite(float(v)) for v in values)
    except (TypeError, ValueError):
        return False


def _pose7(obj):
    """Coerce a Pose-like object / len-7 sequence to ``[x,y,z,qx,qy,qz,qw]``.

    Returns None when *obj* is not a valid pose.
    """
    if obj is None:
        return None
    if hasattr(obj, "pose") and not hasattr(obj, "position"):
        # Waypoint-like wrapper (has .pose, .params)
        return _pose7(obj.pose)
    if hasattr(obj, "position") and hasattr(obj, "orientation"):
        p, q = obj.position, obj.orientation
        vec = [p.x, p.y, p.z, q.x, q.y, q.z, q.w]
        return [float(v) for v in vec] if _finite(vec) else None
    try:
        vec = [float(v) for v in obj]
    except (TypeError, ValueError):
        return None
    if len(vec) == 7 and _finite(vec):
        return vec
    if len(vec) == 3 and _finite(vec):  # [x, y, yaw] convenience (nav goals)
        x, y, yaw = vec
        return [x, y, 0.0] + _quat_from_yaw(yaw)
    return None


def _yaw_of(pose7):
    x, y, z, w = pose7[3], pose7[4], pose7[5], pose7[6]
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_from_yaw(yaw):
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]


def _wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _interp(start, target, n=_PLAN_POINTS):
    """*n* linearly interpolated points from *start* to *target* (inclusive)."""
    return [
        [s + (t - s) * (k / float(n)) for s, t in zip(start, target)]
        for k in range(1, n + 1)
    ]


def _chain_positions(world, chain):
    """Live joint positions of a chain via world primitives."""
    refs = world.resolve([chain], [])
    return [float(v) for v in world.get_positions(refs)]


def _resolve_frame(frame, chain=None):
    """Map SDK frame vocabulary onto a concrete model frame name.

    ``EndEffector`` resolves through *chain*; a bare chain name resolves to
    its end-effector mount link; anything else passes through untouched.
    """
    if frame == "EndEffector":
        return _CHAIN_EE_FRAME.get(chain) if chain else None
    if frame in _CHAIN_EE_FRAME and frame in _CHAIN_DOF:
        return _CHAIN_EE_FRAME[frame]
    return frame


def _chain_for_ee_frame(end_effector_frame):
    """Chain owning an end-effector frame; None when not derivable."""
    if end_effector_frame in _CHAIN_DOF:
        return end_effector_frame
    suffix = "_end_effector_mount_link"
    if end_effector_frame.endswith(suffix):
        prefix = end_effector_frame[: -len(suffix)]
        if prefix in _CHAIN_DOF:
            return prefix
    return None


def _split_whole_body(values):
    """Split a concatenated whole-body vector into per-chain vectors.

    Accepts the 21-DOF tutorial layout (leg,head,left_arm,right_arm) and the
    17-DOF native S1 layout (torso,head,left_arm,right_arm).
    """
    vec = [float(v) for v in values]
    if len(vec) == 21:
        order = _WHOLE_BODY_GROUPS
    elif len(vec) == 17:
        order = ("torso", "head", "left_arm", "right_arm")
    else:
        return {}
    out, i = {}, 0
    for chain in order:
        n = _CHAIN_DOF[chain]
        out[chain] = vec[i:i + n]
        i += n
    return out


def _states_to_joint_map(states):
    """RobotStates (any flavour) -> ``{chain: joint_positions}``."""
    if states is None:
        return {}
    if isinstance(states, PoseState):
        return {}
    if isinstance(states, JointStates):
        chain = getattr(states, "chain_name", "") or ""
        joints = list(getattr(states, "joint_positions", []) or [])
        return {chain: [float(v) for v in joints]} if chain and joints else {}
    whole = list(getattr(states, "whole_body_joint", []) or [])
    return _split_whole_body(whole) if whole else {}


def _solve_ik(world, chain, pose7, reference_frame, seed=None):
    """Damped-least-squares IK via the world; None on failure."""
    try:
        result = world.ik(chain, pose7, reference_frame, seed=seed)
    except Exception as exc:
        _log_debug(f"[galbot_sdk] IK failed for chain '{chain}': {exc}")
        return None
    if result is None:
        return None
    return [float(v) for v in result]


# ---------------------------------------------------------------------------
# GalbotMotion (api_contract §8.2 — 35 instance methods + get_instance)
# ---------------------------------------------------------------------------

class GalbotMotion:
    """Motion planning / kinematics manager (MuJoCo-backed simulation).

    No public constructor — use ``GalbotMotion.get_instance(machine_type)``
    (the ``galbot_sdk.g1`` / ``.s1`` wrappers do this for you).
    """

    _instances = {}

    def __init__(self):
        raise TypeError("galbot_sdk.GalbotMotion: No constructor defined!")

    @staticmethod
    def get_instance(machine_type) -> "GalbotMotion":
        # NOTE: the official stub leaves this parameter untyped (`...`) —
        # quirk preserved by accepting anything int-coercible (§10.4).
        mt = MachineType(int(machine_type))
        inst = GalbotMotion._instances.get(mt)
        if inst is None:
            inst = object.__new__(GalbotMotion)
            inst._machine_type = mt
            inst._inited = False
            inst._plan_config = MotionPlanConfig()
            GalbotMotion._instances[mt] = inst
        return inst

    def __repr__(self) -> str:
        return (
            f"<galbot_sdk.GalbotMotion(machine_type="
            f"{self._machine_type!s}) object at 0x{id(self):x}>"
        )

    # -- lifecycle ----------------------------------------------------------

    def init(self) -> bool:
        _require_world()  # missing model raises a clear RuntimeError here
        self._inited = True
        return True

    # -- obstacle / attached-object / tool registries -----------------------

    def add_obstacle(self, obstacle_id: str, obstacle_type: str, pose,
                     scale=[0.0, 0.0, 0.0], key: str = '',
                     target_frame: str = 'world', ee_frame: str = 'ee_base',
                     reference_joint_positions=[], reference_base_pose=[],
                     ignore_collision_link_names=[], safe_margin: float = 0.0,
                     resolution: float = 0.01) -> MotionStatus:
        return self._register(
            "obstacles", obstacle_id, obstacle_type, pose, scale, key,
            target_frame, ee_frame, reference_joint_positions,
            reference_base_pose, ignore_collision_link_names, safe_margin,
            resolution)

    def attach_target_object(self, obstacle_id: str, obstacle_type: str, pose,
                             scale=[0.0, 0.0, 0.0], key: str = '',
                             target_frame: str = 'world',
                             ee_frame: str = 'ee_base',
                             reference_joint_positions=[],
                             reference_base_pose=[],
                             ignore_collision_link_names=[],
                             safe_margin: float = 0.0,
                             resolution: float = 0.01) -> MotionStatus:
        return self._register(
            "attached_objects", obstacle_id, obstacle_type, pose, scale, key,
            target_frame, ee_frame, reference_joint_positions,
            reference_base_pose, ignore_collision_link_names, safe_margin,
            resolution)

    def _register(self, registry_name, obstacle_id, obstacle_type, pose,
                  scale, key, target_frame, ee_frame,
                  reference_joint_positions, reference_base_pose,
                  ignore_collision_link_names, safe_margin, resolution):
        try:
            w = _require_world()
        except Exception:
            return MotionStatus.INIT_FAILED
        if not obstacle_id or not isinstance(obstacle_id, str):
            return MotionStatus.INVALID_INPUT
        if obstacle_type not in _OBSTACLE_TYPES:
            return MotionStatus.INVALID_INPUT
        pose_vec = _pose7(pose)
        if pose_vec is None:
            return MotionStatus.INVALID_INPUT
        _registry(w, registry_name)[obstacle_id] = {
            "obstacle_type": obstacle_type,
            "pose": pose_vec,
            "scale": [float(v) for v in scale],
            "key": key,
            "target_frame": target_frame,
            "ee_frame": ee_frame,
            "reference_joint_positions": [float(v) for v in
                                          reference_joint_positions],
            "reference_base_pose": [float(v) for v in reference_base_pose],
            "ignore_collision_link_names": list(ignore_collision_link_names),
            "safe_margin": float(safe_margin),
            "resolution": float(resolution),
        }
        return MotionStatus.SUCCESS

    def detach_target_object(self, obstacle_id: str) -> MotionStatus:
        try:
            w = _require_world()
        except Exception:
            return MotionStatus.INIT_FAILED
        if _registry(w, "attached_objects").pop(obstacle_id, None) is None:
            return MotionStatus.INVALID_INPUT
        return MotionStatus.SUCCESS

    def remove_obstacle(self, obstacle_id: str) -> MotionStatus:
        try:
            w = _require_world()
        except Exception:
            return MotionStatus.INIT_FAILED
        if _registry(w, "obstacles").pop(obstacle_id, None) is None:
            return MotionStatus.INVALID_INPUT
        return MotionStatus.SUCCESS

    def clear_obstacle(self) -> MotionStatus:
        try:
            w = _require_world()
        except Exception:
            return MotionStatus.INIT_FAILED
        _registry(w, "obstacles").clear()
        return MotionStatus.SUCCESS

    def get_built_obstacles_list(self) -> list:
        try:
            w = _require_world()
        except Exception:
            return []
        return list(_registry(w, "obstacles").keys())

    def attach_tool(self, chain: str, tool: str) -> MotionStatus:
        try:
            w = _require_world()
        except Exception:
            return MotionStatus.INIT_FAILED
        if chain not in _CHAIN_DOF or not tool or not isinstance(tool, str):
            return MotionStatus.INVALID_INPUT
        _registry(w, "tools")[chain] = tool
        return MotionStatus.SUCCESS

    def detach_tool(self, chain: str) -> MotionStatus:
        try:
            w = _require_world()
        except Exception:
            return MotionStatus.INIT_FAILED
        if _registry(w, "tools").pop(chain, None) is None:
            return MotionStatus.INVALID_INPUT
        return MotionStatus.SUCCESS

    # -- collision ----------------------------------------------------------

    def check_collision(self, start, enable_collision_check: bool = True,
                        params: Parameter = ...) -> tuple:
        """Registry-only collision check → ``(SUCCESS, [False] * len(start))``.

        Honest for tutorial scenes (empty collision worlds); the geometric
        seam (``mj_collision``) is a documented later upgrade.
        """
        try:
            _require_world()
        except Exception:
            return (MotionStatus.INIT_FAILED, [])
        if isinstance(start, RobotStates):
            start = [start]
        try:
            n = len(list(start))
        except TypeError:
            return (MotionStatus.INVALID_INPUT, [])
        return (MotionStatus.SUCCESS, [False] * n)

    # -- FK / IK / Jacobian -------------------------------------------------

    def forward_kinematics(self, target_frame: str,
                           reference_frame: str = 'base_link',
                           joint_state={}, params: Parameter = ...) -> tuple:
        try:
            w = _require_world()
        except Exception:
            return (MotionStatus.INIT_FAILED, [])
        frame = _resolve_frame(target_frame)
        if frame is None:
            return (MotionStatus.INVALID_INPUT, [])
        try:
            if joint_state:
                jmap = {str(k): [float(x) for x in v]
                        for k, v in dict(joint_state).items()}
                pose = w.fk_with_joint_state(frame, reference_frame, jmap)
            else:
                pose = w.frame_pose(frame, reference_frame)
        except Exception as exc:
            _log_debug(f"[galbot_sdk] forward_kinematics({target_frame}): {exc}")
            return (MotionStatus.INVALID_INPUT, [])
        return (MotionStatus.SUCCESS, [float(v) for v in pose])

    def forward_kinematics_by_state(self, target_frame: str,
                                    reference_robot_states: RobotStates = None,
                                    reference_frame: str = 'base_link',
                                    params: Parameter = ...) -> tuple:
        joint_state = _states_to_joint_map(reference_robot_states)
        return self.forward_kinematics(target_frame, reference_frame,
                                       joint_state, params)

    def inverse_kinematics(self, target_pose, chain_names,
                           target_frame: str = 'EndEffector',
                           reference_frame: str = 'base_link',
                           initial_joint_positions={},
                           enable_collision_check: bool = True,
                           params: Parameter = ...) -> tuple:
        """IK per chain → ``(SUCCESS, {chain: joints})``; failure → ``(FAULT, {})``."""
        try:
            w = _require_world()
        except Exception:
            return (MotionStatus.INIT_FAILED, {})
        pose = _pose7(target_pose)
        chains = list(chain_names)
        if pose is None or not chains:
            return (MotionStatus.INVALID_INPUT, {})
        seeds = dict(initial_joint_positions) if initial_joint_positions else {}
        solution = {}
        for chain in chains:
            if chain not in _CHAIN_DOF:
                return (MotionStatus.INVALID_INPUT, {})
            seed = seeds.get(chain)
            seed = [float(v) for v in seed] if seed else None
            joints = _solve_ik(w, chain, pose, reference_frame, seed)
            if joints is None:
                return (MotionStatus.FAULT, {})
            solution[chain] = joints
        return (MotionStatus.SUCCESS, solution)

    def inverse_kinematics_by_state(self, target_pose, chain_names,
                                    target_frame: str = 'EndEffector',
                                    reference_frame: str = 'base_link',
                                    reference_robot_states: RobotStates = None,
                                    enable_collision_check: bool = True,
                                    params: Parameter = ...) -> tuple:
        seeds = _states_to_joint_map(reference_robot_states)
        return self.inverse_kinematics(
            target_pose, chain_names, target_frame, reference_frame,
            seeds, enable_collision_check, params)

    def get_jacobian(self, chain_name: str, target_frame: str = 'EndEffector',
                     reference_frame: str = 'base_link', joint_state={},
                     params: Parameter = ...) -> tuple:
        try:
            w = _require_world()
        except Exception:
            return (MotionStatus.INIT_FAILED, [])
        if chain_name not in _CHAIN_DOF:
            return (MotionStatus.INVALID_INPUT, [])
        if joint_state:
            _warn_once("jacobian_joint_state",
                       "[galbot_sdk] get_jacobian: joint_state reference is "
                       "accepted but the live Jacobian is returned "
                       "(GalbotSim fidelity delta)")
        frame = _resolve_frame(target_frame, chain=chain_name)
        if frame is None:
            return (MotionStatus.INVALID_INPUT, [])
        try:
            jac = w.jacobian(chain_name, frame, reference_frame)
        except Exception as exc:
            _log_debug(f"[galbot_sdk] get_jacobian({chain_name}): {exc}")
            return (MotionStatus.FAULT, [])
        return (MotionStatus.SUCCESS,
                [[float(x) for x in row] for row in jac])

    def get_jacobian_by_state(self, chain_name: str,
                              target_frame: str = 'EndEffector',
                              reference_frame: str = 'base_link',
                              reference_robot_states: RobotStates = None,
                              params: Parameter = ...) -> tuple:
        joint_state = _states_to_joint_map(reference_robot_states)
        return self.get_jacobian(chain_name, target_frame, reference_frame,
                                 joint_state, params)

    # -- pose / state getters ----------------------------------------------

    def get_end_effector_pose(self, end_effector_frame: str,
                              reference_frame: str = 'base_link') -> tuple:
        try:
            w = _require_world()
        except Exception:
            return (MotionStatus.INIT_FAILED, [])
        frame = _resolve_frame(end_effector_frame)
        if frame is None:
            return (MotionStatus.INVALID_INPUT, [])
        try:
            pose = w.frame_pose(frame, reference_frame)
        except Exception as exc:
            _log_debug(
                f"[galbot_sdk] get_end_effector_pose({end_effector_frame}): {exc}")
            return (MotionStatus.INVALID_INPUT, [])
        return (MotionStatus.SUCCESS, [float(v) for v in pose])

    def get_end_effector_pose_on_chain(self, chain_name: str,
                                       frame_id: str = 'EndEffector',
                                       reference_frame: str = 'base_link'
                                       ) -> tuple:
        if chain_name not in _CHAIN_DOF:
            return (MotionStatus.INVALID_INPUT, [])
        frame = _resolve_frame(frame_id, chain=chain_name)
        if frame is None:
            return (MotionStatus.INVALID_INPUT, [])
        return self.get_end_effector_pose(frame, reference_frame)

    def get_chain_joint_state(self) -> dict:
        try:
            w = _require_world()
        except Exception:
            return {}
        out = {}
        for chain in _MACHINE_CHAINS[self._machine_type]:
            try:
                out[chain] = _chain_positions(w, chain)
            except Exception as exc:
                _log_debug(f"[galbot_sdk] get_chain_joint_state({chain}): {exc}")
        return out

    def get_robot_states(self) -> RobotStates:
        states = RobotStates()
        try:
            w = _require_world()
        except Exception:
            return states
        whole = []
        for chain in _WHOLE_BODY_GROUPS:
            try:
                whole.extend(_chain_positions(w, chain))
            except Exception:
                whole.extend([0.0] * _CHAIN_DOF[chain])
        states.whole_body_joint = whole
        try:
            states.base_state = [float(v) for v in w.base_pose7()]
        except Exception:
            pass
        return states

    def get_link_names(self, only_end_effector: bool = False) -> list:
        if only_end_effector:
            return list(_EE_FRAMES)
        return list(_S1_LINKS)

    def get_supported_chains(self) -> set:
        return set(_MACHINE_CHAINS[self._machine_type])

    def get_supported_ee_frames(self) -> set:
        return set(_EE_FRAMES) | {"EndEffector", "ee_base"}

    def get_supported_frames(self) -> set:
        frames = set(_S1_LINKS) | {"world", "map", "odom", "EndEffector",
                                   "ee_base"}
        try:
            w = _require_world()
            names = getattr(w, "frame_names", None)
            if callable(names):
                frames |= set(names())
        except Exception:
            pass
        return frames

    def get_supported_links(self) -> set:
        return set(_S1_LINKS)

    def get_supported_obstacle_types(self) -> set:
        return set(_OBSTACLE_TYPES)

    def get_supported_tool_list(self) -> set:
        return set(_TOOL_LIST)

    # -- planning (IK + linear joint-space interpolation) -------------------

    def motion_plan(self, target: RobotStates, start: RobotStates = None,
                    reference_robot_states: RobotStates = None,
                    enable_collision_check: bool = True,
                    params: Parameter = ...) -> tuple:
        try:
            w = _require_world()
        except Exception:
            return (MotionStatus.INIT_FAILED, {})
        return self._plan_to_target(w, target, waypoints=None,
                                    start_map=_states_to_joint_map(start),
                                    best_effort=False)

    def motion_plan_multi_waypoints(self, target=None, waypoint_poses=(),
                                    start=None,
                                    reference_robot_states: RobotStates = None,
                                    enable_collision_check: bool = True,
                                    params: Parameter = ..., targets=None
                                    ) -> tuple:
        """Both stub overloads: RobotStates+waypoints, or a targets Mapping."""
        try:
            w = _require_world()
        except Exception:
            return (MotionStatus.INIT_FAILED, {})

        # Overload 2: Mapping[RobotStates, Seq[Seq[float]]]
        mapping = targets if targets is not None else (
            target if isinstance(target, dict) else None)
        if mapping is not None:
            start_map = {}
            for st in (list(start) if start else []):
                start_map.update(_states_to_joint_map(st))
            merged = {}
            for tgt, wps in mapping.items():
                status, traj = self._plan_to_target(
                    w, tgt, waypoints=list(wps) if wps is not None else [],
                    start_map=start_map, best_effort=True)
                if status != MotionStatus.SUCCESS:
                    return (status, {})
                for chain, points in traj.items():
                    merged.setdefault(chain, []).extend(points)
            return (MotionStatus.SUCCESS, merged)

        # Overload 1: single RobotStates target + waypoint list
        waypoints = list(waypoint_poses) if waypoint_poses is not None else []
        return self._plan_to_target(w, target, waypoints=waypoints,
                                    start_map=_states_to_joint_map(start),
                                    best_effort=True)

    def _plan_to_target(self, w, target, waypoints, start_map, best_effort):
        """Core planner.

        *waypoints* None → plan to the target state itself (motion_plan);
        otherwise plan through the waypoint list, whose meaning depends on
        the target flavour (PoseState → pose7s to IK; JointStates → joint
        vectors; plain RobotStates → whole-body vectors).
        """
        if not isinstance(target, RobotStates):
            return (MotionStatus.INVALID_INPUT, {})
        start_map = dict(start_map or {})

        def start_joints(chain):
            if chain in start_map:
                return [float(v) for v in start_map[chain]]
            return _chain_positions(w, chain)

        # --- PoseState target: Cartesian waypoints, chained-seed IK --------
        if isinstance(target, PoseState):
            chain = getattr(target, "chain_name", "") or ""
            if chain not in _CHAIN_DOF:
                return (MotionStatus.INVALID_INPUT, {})
            reference = getattr(target, "reference_frame", "") or "base_link"
            if waypoints is None:
                pose = _pose7(getattr(target, "pose", None))
                if pose is None:
                    return (MotionStatus.INVALID_INPUT, {})
                pose_list = [pose]
            else:
                pose_list = [_pose7(wp) for wp in waypoints]
                if not pose_list or any(p is None for p in pose_list):
                    return (MotionStatus.INVALID_INPUT, {})
            try:
                seed = start_joints(chain)
            except Exception:
                return (MotionStatus.FAULT, {})
            joint_targets = []
            for i, pose in enumerate(pose_list):
                joints = _solve_ik(w, chain, pose, reference, seed)
                if joints is None:
                    if not best_effort:
                        return (MotionStatus.FAULT, {})
                    _warn_once(
                        f"plan_ik_{chain}",
                        f"[galbot_sdk] motion_plan_multi_waypoints: IK did "
                        f"not converge for waypoint {i} on chain '{chain}'; "
                        f"reusing previous joint solution (GalbotSim "
                        f"best-effort planning)")
                    joints = list(seed)
                joint_targets.append(joints)
                seed = joints
            points, cursor = [], start_joints(chain)
            for joints in joint_targets:
                points.extend(_interp(cursor, joints))
                cursor = joints
            return (MotionStatus.SUCCESS, {chain: points})

        # --- JointStates target: joint-space waypoints ---------------------
        if isinstance(target, JointStates):
            chain = getattr(target, "chain_name", "") or ""
            if chain not in _CHAIN_DOF:
                return (MotionStatus.INVALID_INPUT, {})
            dof = _CHAIN_DOF[chain]
            if waypoints is None:
                vecs = [list(getattr(target, "joint_positions", []) or [])]
            else:
                vecs = [list(wp) for wp in waypoints]
            if not vecs:
                return (MotionStatus.INVALID_INPUT, {})
            for vec in vecs:
                if len(vec) != dof or not _finite(vec):
                    return (MotionStatus.INVALID_INPUT, {})
            try:
                cursor = start_joints(chain)
            except Exception:
                return (MotionStatus.FAULT, {})
            points = []
            for vec in vecs:
                joints = [float(v) for v in vec]
                points.extend(_interp(cursor, joints))
                cursor = joints
            return (MotionStatus.SUCCESS, {chain: points})

        # --- plain RobotStates target: whole-body vectors ------------------
        if waypoints is None:
            vecs = [list(getattr(target, "whole_body_joint", []) or [])]
        else:
            vecs = [list(wp) for wp in waypoints]
        splits = [_split_whole_body(vec) for vec in vecs]
        if not splits or any(not s for s in splits):
            return (MotionStatus.INVALID_INPUT, {})
        traj = {}
        for chain in splits[0]:
            try:
                cursor = start_joints(chain)
            except Exception:
                return (MotionStatus.FAULT, {})
            points = []
            for split in splits:
                joints = split[chain]
                points.extend(_interp(cursor, joints))
                cursor = joints
            traj[chain] = points
        return (MotionStatus.SUCCESS, traj)

    # -- direct actuation ---------------------------------------------------

    def set_end_effector_pose(self, target_pose, end_effector_frame: str,
                              reference_frame: str = 'base_link',
                              reference_robot_states: RobotStates = None,
                              enable_collision_check: bool = True,
                              is_blocking: bool = True,
                              timeout: float = -1.0,
                              params: Parameter = ...) -> MotionStatus:
        """IK then synchronous joint interpolation; SUCCESS on completion.

        ``timeout`` is accepted and never slept on (§4 hard rule); the
        tutorials pass a chain name ("left_arm") as *end_effector_frame*.
        """
        try:
            w = _require_world()
        except Exception:
            return MotionStatus.INIT_FAILED
        chain = _chain_for_ee_frame(end_effector_frame)
        pose = _pose7(target_pose)
        if chain is None or pose is None:
            return MotionStatus.INVALID_INPUT
        seed_map = _states_to_joint_map(reference_robot_states)
        seed = seed_map.get(chain)
        if seed is None:
            try:
                seed = _chain_positions(w, chain)
            except Exception:
                return MotionStatus.FAULT
        joints = _solve_ik(w, chain, pose, reference_frame, seed)
        if joints is None:
            joints = _solve_ik(w, chain, pose, reference_frame, None)
        if joints is None:
            return MotionStatus.FAULT
        try:
            refs = w.resolve([chain], [])
            w.set_positions(refs, joints)
        except Exception as exc:
            _log_debug(f"[galbot_sdk] set_end_effector_pose apply: {exc}")
            return MotionStatus.FAULT
        return MotionStatus.SUCCESS

    def move_whole_body_joint_zero(self, is_blocking: bool = True,
                                   leg_head_speed_rad_s: float = 0.2,
                                   leg_head_timeout_s: float = 15.0,
                                   params: Parameter = ...) -> MotionStatus:
        """Arms/head/leg-emulation to zero; speed/timeout accepted, ignored."""
        try:
            w = _require_world()
        except Exception:
            return MotionStatus.INIT_FAILED
        moved_any = False
        for group in _WHOLE_BODY_GROUPS:
            try:
                refs = w.resolve([group], [])
                w.set_positions(refs, [0.0] * len(list(refs)))
                moved_any = True
            except Exception as exc:
                _log_debug(
                    f"[galbot_sdk] move_whole_body_joint_zero({group}): {exc}")
        return MotionStatus.SUCCESS if moved_any else MotionStatus.FAULT

    # -- config -------------------------------------------------------------

    def get_motion_plan_config(self) -> tuple:
        return (MotionStatus.SUCCESS, self._plan_config)

    def set_motion_plan_config(self, config: MotionPlanConfig) -> MotionStatus:
        if not isinstance(config, MotionPlanConfig):
            return MotionStatus.INVALID_INPUT
        self._plan_config = config
        return MotionStatus.SUCCESS

    def status_to_string(self, status) -> str:
        """Convert MotionStatus to a human-readable string."""
        return check_motion_status(status)


# ---------------------------------------------------------------------------
# GalbotNavigation (api_contract §8.3 — 26 instance methods + get_instance)
# ---------------------------------------------------------------------------

class GalbotNavigation:
    """Navigation manager: synchronous goal application on the free base.

    Tuple-returning APIs yield ``(True, "SUCCESS")`` on success and
    ``(False, "<reason>")`` on failure (contract §10.6).
    """

    _instances = {}

    def __init__(self):
        raise TypeError("galbot_sdk.GalbotNavigation: No constructor defined!")

    @staticmethod
    def get_instance(machine_type) -> "GalbotNavigation":
        mt = MachineType(int(machine_type))
        inst = GalbotNavigation._instances.get(mt)
        if inst is None:
            inst = object.__new__(GalbotNavigation)
            inst._machine_type = mt
            inst._inited = False
            inst._last_goal = None
            inst._last_status = NavigationTaskStatus.UNKNOWN
            inst._tasks = {}
            inst._attached_boxes = {}
            inst._arrival_threshold = [1e-3, 1e-3, 1e-3]
            inst._config = {
                "arrival_threshold": [1e-3, 1e-3, 1e-3],
                "velocity_limit": [1.0, 1.0, 1.0],
                "kinematics_limits": {
                    "vel_limit": [1.0, 1.0, 1.0],
                    "acc_limit": [1.0, 1.0, 1.0],
                    "jerk_limit": [1.0, 1.0, 1.0],
                },
                "timeout_s": 0.0,
            }
            GalbotNavigation._instances[mt] = inst
        return inst

    def __repr__(self) -> str:
        return (
            f"<galbot_sdk.GalbotNavigation(machine_type="
            f"{self._machine_type!s}) object at 0x{id(self):x}>"
        )

    # -- lifecycle / localization ------------------------------------------

    def init(self) -> bool:
        w = _require_world()
        try:
            w.localized = True  # localized from init (kills ex3/ex10 loops)
        except Exception:
            pass
        self._inited = True
        return True

    def is_localized(self) -> bool:
        try:
            w = _require_world()
        except Exception:
            return False
        return bool(getattr(w, "localized", True))

    def relocalize(self, init_pose) -> tuple:
        pose = _pose7(init_pose)
        if pose is None:
            return (False, "INVALID_INPUT")
        try:
            w = _require_world()
            w.set_base_pose7(pose)
        except Exception:
            return (False, "INIT_FAILED")
        try:
            w.localized = True
        except Exception:
            pass
        return (True, "SUCCESS")

    def get_current_pose(self) -> list:
        w = _require_world()
        return [float(v) for v in w.base_pose7()]

    # -- goal navigation (synchronous) --------------------------------------

    def _apply_goal(self, goal):
        """Teleport-interpolate the base to *goal* pose7 and store it."""
        pose = _pose7(goal)
        if pose is None:
            return (False, "INVALID_INPUT")
        try:
            w = _require_world()
            w.set_base_pose7(pose)
        except Exception:
            return (False, "INIT_FAILED")
        self._last_goal = pose
        try:
            w.last_goal = pose
        except Exception:
            pass
        self._last_status = NavigationTaskStatus.SUCCESS
        return (True, "SUCCESS")

    def navigate_to_goal(self, goal_pose, enable_collision_check: bool = True,
                         is_blocking: bool = False, timeout: float = 8,
                         omni_plan: bool = False) -> tuple:
        return self._apply_goal(goal_pose)

    def navigate_to_goal_v2(self, goal_pose, max_vel, pose_frame: str = 'map',
                            enable_collision_check: bool = True,
                            is_blocking: bool = False, timeout: float = 5.0,
                            omni_plan: bool = False) -> tuple:
        return self._apply_goal(goal_pose)

    def move_straight_to(self, goal_pose, is_blocking: bool = True,
                         timeout: float = 8) -> tuple:
        return self._apply_goal(goal_pose)

    def navigate_with_velocity(self, vx: float, vy: float, vyaw: float,
                               duration_s: float = 3.0,
                               enable_collision_check: bool = True) -> tuple:
        """Single synchronous integration step of body-frame v x duration."""
        values = (vx, vy, vyaw, duration_s)
        if not _finite(values):
            return (False, "INVALID_INPUT")
        try:
            w = _require_world()
            cur = [float(v) for v in w.base_pose7()]
        except Exception:
            return (False, "INIT_FAILED")
        vx, vy, vyaw, dt = (float(v) for v in values)
        yaw = _yaw_of(cur)
        nx = cur[0] + (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        ny = cur[1] + (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        nyaw = yaw + vyaw * dt
        try:
            w.set_base_pose7([nx, ny, cur[2]] + _quat_from_yaw(nyaw))
        except Exception:
            return (False, "INIT_FAILED")
        return (True, "SUCCESS")

    # -- task-handle navigation ---------------------------------------------

    def _run_task(self, final_pose):
        """Apply the final waypoint synchronously; register a TaskHandle."""
        handle = TaskHandle()
        handle.task_id = uuid.uuid4().hex
        ok, msg = (self._apply_goal(final_pose) if final_pose is not None
                   else (False, "INVALID_INPUT"))
        handle.request_sent = bool(ok)
        handle.msg = msg
        snapshot = NavigationTaskSnapshot()
        snapshot.task_id = handle.task_id
        snapshot.status = (NavigationTaskStatus.SUCCESS if ok
                           else NavigationTaskStatus.FAILED)
        snapshot.msg = msg
        self._tasks[handle.task_id] = snapshot
        return handle

    def navigate_along_trajectory(self, waypoints, frame_id: str = 'map',
                                  speed_ratio: float = 1.0,
                                  enable_collision_check: bool = True
                                  ) -> TaskHandle:
        wps = list(waypoints) if waypoints is not None else []
        return self._run_task(wps[-1] if wps else None)

    def navigate_through_waypoints(self, waypoints, frame_id: str = 'map',
                                   enable_collision_check: bool = True
                                   ) -> TaskHandle:
        wps = list(waypoints) if waypoints is not None else []
        return self._run_task(wps[-1] if wps else None)

    def set_navigation_target(self, target, frame: str = 'map',
                              speed_ratio: float = 1.0,
                              enable_collision_check: bool = True
                              ) -> TaskHandle:
        return self._run_task(target)

    def stop_navigation(self) -> tuple:
        # Nothing is ever in flight in the synchronous engine.
        return (True, "SUCCESS")

    # -- status / checks -----------------------------------------------------

    def check_goal_arrival(self) -> bool:
        """Compare the last stored goal to the live base pose."""
        try:
            w = _require_world()
        except Exception:
            return False
        goal = getattr(w, "last_goal", None) or self._last_goal
        if goal is None:
            return False
        cur = [float(v) for v in w.base_pose7()]
        tx, ty, tyaw = self._arrival_threshold
        return (abs(goal[0] - cur[0]) <= tx
                and abs(goal[1] - cur[1]) <= ty
                and abs(_wrap_angle(_yaw_of(goal) - _yaw_of(cur))) <= tyaw)

    def check_path_reachability(self, goal_pose, start_pose) -> bool:
        # Empty-world planner: any finite pose pair is reachable.
        return _pose7(goal_pose) is not None and _pose7(start_pose) is not None

    def get_navigation_status(self) -> NavigationTaskStatus:
        return self._last_status

    def get_navigation_target_status(self, task_id: str
                                     ) -> NavigationTaskSnapshot:
        snapshot = self._tasks.get(task_id)
        if snapshot is None:
            snapshot = NavigationTaskSnapshot()
            snapshot.task_id = task_id
            snapshot.status = NavigationTaskStatus.UNKNOWN
            snapshot.msg = f"unknown task_id: {task_id!r}"
        return snapshot

    # -- bounding-box registry ----------------------------------------------

    def _boxes(self):
        return _registry(_require_world(), "bounding_boxes")

    def add_bounding_box(self, box_info: dict) -> tuple:
        if not isinstance(box_info, dict):
            return (False, "INVALID_INPUT")
        try:
            boxes = self._boxes()
        except Exception:
            return (False, "INIT_FAILED")
        tag = box_info.get("box_tag", len(boxes))
        boxes[int(tag)] = dict(box_info)
        return (True, "SUCCESS")

    def get_bounding_box(self) -> list:
        try:
            return [dict(v) for v in self._boxes().values()]
        except Exception:
            return []

    def remove_bounding_box(self, box_tag) -> tuple:
        try:
            boxes = self._boxes()
        except Exception:
            return (False, "INIT_FAILED")
        if boxes.pop(int(box_tag), None) is None:
            return (False, "INVALID_INPUT")
        return (True, "SUCCESS")

    def attach_box_to_link(self, box_info: dict,
                           ignore_collision_links=[]) -> tuple:
        if not isinstance(box_info, dict):
            return (False, "INVALID_INPUT")
        tag = int(box_info.get("box_tag", len(self._attached_boxes)))
        self._attached_boxes[tag] = {
            "box_info": dict(box_info),
            "ignore_collision_links": list(ignore_collision_links),
        }
        return (True, "SUCCESS")

    def detach_box_from_link(self, box_tag) -> tuple:
        if self._attached_boxes.pop(int(box_tag), None) is None:
            return (False, "INVALID_INPUT")
        return (True, "SUCCESS")

    # -- configuration -------------------------------------------------------

    def set_navigation_arrival_threshold(self, threshold) -> tuple:
        try:
            values = [float(v) for v in threshold]
        except (TypeError, ValueError):
            return (False, "INVALID_INPUT")
        if len(values) != 3 or not _finite(values):
            return (False, "INVALID_INPUT")
        self._arrival_threshold = values
        self._config["arrival_threshold"] = values
        return (True, "SUCCESS")

    def set_navigation_kinematics_limits(self, vel_limit, acc_limit,
                                         jerk_limit) -> tuple:
        try:
            limits = {
                "vel_limit": [float(v) for v in vel_limit],
                "acc_limit": [float(v) for v in acc_limit],
                "jerk_limit": [float(v) for v in jerk_limit],
            }
        except (TypeError, ValueError):
            return (False, "INVALID_INPUT")
        self._config["kinematics_limits"] = limits
        return (True, "SUCCESS")

    def set_navigation_velocity_limit(self, vel_limit) -> tuple:
        try:
            values = [float(v) for v in vel_limit]
        except (TypeError, ValueError):
            return (False, "INVALID_INPUT")
        self._config["velocity_limit"] = values
        return (True, "SUCCESS")

    def set_navigation_timeout(self, timeout_s) -> tuple:
        try:
            self._config["timeout_s"] = float(timeout_s)
        except (TypeError, ValueError):
            return (False, "INVALID_INPUT")
        return (True, "SUCCESS")

    def dump_navigation_configs(self) -> tuple:
        return (True, json.dumps(self._config))


# ---------------------------------------------------------------------------
# GalbotPerception (api_contract §8.4 — G1 only)
# ---------------------------------------------------------------------------

class GalbotPerception:
    """Perception manager. G1 only: ``get_instance(MachineType.S1)`` raises.

    GalbotSim has no perception backends; ``wait_for_new_result`` /
    ``get_latest_result`` report the honest no-result shapes with one
    warning per method per process.
    """

    _instances = {}

    def __init__(self):
        raise TypeError("galbot_sdk.GalbotPerception: No constructor defined!")

    @staticmethod
    def get_instance(machine_type: MachineType) -> "GalbotPerception":
        mt = MachineType(int(machine_type))
        if mt == MachineType.S1:
            raise RuntimeError(
                "GalbotPerception is not supported on MachineType.S1")
        inst = GalbotPerception._instances.get(mt)
        if inst is None:
            inst = object.__new__(GalbotPerception)
            inst._machine_type = mt
            inst._enabled_modules = frozenset()
            GalbotPerception._instances[mt] = inst
        return inst

    def init(self, enabled_modules) -> bool:
        self._enabled_modules = frozenset(enabled_modules)
        return True

    def run_once(self, module) -> bool:
        return True

    def get_latest_result(self, module) -> tuple:
        _warn_once("perception_get_latest_result",
                   "[galbot_sdk] GalbotPerception.get_latest_result: "
                   "GalbotSim has no perception backend; returning "
                   "(False, DetectionResult())")
        return (False, DetectionResult())

    def wait_for_new_result(self, module, timeout_s: float = 5.0) -> bool:
        _warn_once("perception_wait_for_new_result",
                   "[galbot_sdk] GalbotPerception.wait_for_new_result: "
                   "GalbotSim has no perception backend; returning False")
        return False


# ---------------------------------------------------------------------------
# Module-level free functions (api_contract §6)
# ---------------------------------------------------------------------------

_MOTION_STATUS_TEXT = {
    MotionStatus.SUCCESS: "Execution successful",
    MotionStatus.TIMEOUT: "Execution timeout",
    MotionStatus.FAULT: "Fault occurred, cannot continue execution",
    MotionStatus.INVALID_INPUT: "Input parameters do not meet requirements",
    MotionStatus.INIT_FAILED:
        "Internal communication component creation failed",
    MotionStatus.IN_PROGRESS: "Motion in progress but not reached target",
    MotionStatus.STOPPED_UNREACHED: "Stopped but not reached target",
    MotionStatus.DATA_FETCH_FAILED: "Data fetch failed",
    MotionStatus.PUBLISH_FAIL: "Data sending failed",
    MotionStatus.COMM_DISCONNECTED: "Connection failed",
    MotionStatus.STATUS_NUM: "Status count sentinel",
    MotionStatus.UNSUPPORTED_FUNCRION: "Unsupported function",
}


def check_motion_status(status) -> str:
    """Convert a MotionStatus enum value to a string."""
    try:
        member = MotionStatus(int(status))
    except (TypeError, ValueError):
        return f"UNKNOWN({status!r})"
    return f"{member.name}: {_MOTION_STATUS_TEXT[member]}"


def create_joint_state() -> JointStates:
    """Create a JointStates instance."""
    return JointStates()


def create_pose_state() -> PoseState:
    """Create a PoseState instance."""
    return PoseState()


def create_parameter(direct_execute: bool, blocking: bool, timeout: float,
                     actuate: str, tool_pose: bool, check_collision: bool,
                     frame: str = 'base_link') -> Parameter:
    """Create a Parameter instance (actuate: position/velocity/torque)."""
    return Parameter(direct_execute, blocking, timeout, actuate, tool_pose,
                     check_collision, frame)


__all__ = [
    "GalbotMotion",
    "GalbotNavigation",
    "GalbotPerception",
    "check_motion_status",
    "create_joint_state",
    "create_pose_state",
    "create_parameter",
]
