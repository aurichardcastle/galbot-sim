"""GalbotSim world — THE MuJoCo singleton behind the galbot_sdk shim.

This is the ONLY module in the shim that imports :mod:`mujoco` (DESIGN.md §5.5).
Managers (``_robot.py`` / ``_motion.py``) never touch mujoco; they call the
primitives exposed here::

    resolve / get_positions / set_positions / frame_pose / ik /
    fk_with_joint_state / jacobian / base_pose7 / set_base_pose7 /
    apply_base_velocity / rgb / depth / lidar / imu / intrinsic / now_ns
    + lifecycle (init / is_running / request_shutdown / wait_for_shutdown / destroy)
    + registries (obstacles / tools / attached_objects / bounding_boxes)

Engine model (DESIGN.md §3, normative):
  * Synchronous, step-on-demand, zero threads, zero locks, **no mj_step ever**.
  * Motion = clip targets to joint range -> linear qpos interpolation over
    ``substeps`` -> ``mj_forward`` per substep -> done (milliseconds per call).
  * Timestamps are wall-clock ``time.time_ns()``; there is no sim clock.
  * IK is damped least-squares on a scratch ``MjData`` (never mutates live data).
  * Sensors are procedural by default (deterministic, pure functions of
    (sensor, resolution)); real rendering is a named seam (``GALBOTSIM_RENDER``).

Quaternion convention (DESIGN.md §6): SDK pose7 is ``[x, y, z, qx, qy, qz, qw]``
(quat xyzw); MuJoCo is wxyz.  The conversion lives in exactly two helpers here
(:func:`_wxyz_to_xyzw`, :func:`_xyzw_to_wxyz`) and nowhere else.

Accepted fidelity deltas (documented): one world is shared by all machine types
in a process; the base is assumed flat (writing yaw rebuilds the free-joint
quaternion about +Z); collision registries are bookkeeping only.
"""

from __future__ import annotations

import dataclasses
import glob
import logging
import math
import os
import struct
import time
import zlib

import mujoco
import numpy as np

__all__ = [
    "DEFAULT_MJCF_DIR", "MODEL_XML_ENV", "RENDER_ENV",
    "MACHINE_G1", "MACHINE_S1",
    "UNINIT", "RUNNING", "SHUTDOWN_REQUESTED", "DESTROYED",
    "LEG2_SCALE", "GRIPPER_LUT", "GRIPPER_WIDTH_MIN", "GRIPPER_WIDTH_MAX",
    "GRIPPER_THETA_MIN", "GRIPPER_THETA_MAX",
    "SDK_GROUPS", "MACHINE_SPECS", "MachineSpec", "JointRef",
    "CHAIN_EE_BODY", "SENSOR_NAMES", "CAMERA_WIDTH", "CAMERA_HEIGHT", "CAMERA_FOVY_DEG",
    "gripper_width_from_theta", "gripper_theta_from_width",
    "World", "get_world",
]

_LOG = logging.getLogger("galbot_user_logger")
if os.environ.get("GALBOTSIM_LOG_LEVEL"):
    try:  # optional env knob (DESIGN.md §6); never fatal
        _LOG.setLevel(os.environ["GALBOTSIM_LOG_LEVEL"].upper())
    except (ValueError, TypeError):
        pass

# --------------------------------------------------------------------------- #
# Model location                                                              #
# --------------------------------------------------------------------------- #

MODEL_XML_ENV = "GALBOTSIM_MODEL_XML"
RENDER_ENV = "GALBOTSIM_RENDER"  # =1 -> real-rendering seam (deferred; procedural default)

#: Where quickstart.sh checks the description repo out, relative to the repo root.
DEFAULT_MJCF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "third_party", "galbot_s1_description", "mjcf",
)


def _discover_model_xml(mjcf_dir: str = DEFAULT_MJCF_DIR) -> str | None:
    """Find the S1 MJCF in a checkout of ``galbot_s1_description``.

    Upstream has renamed this file before (``galbot_s1_v1_1_0.xml`` ->
    ``galbot_s1.xml``), so the filename is discovered rather than hard-coded:
    any ``galbot_s1*.xml`` in ``mjcf/`` is accepted, shortest name first so a
    plain ``galbot_s1.xml`` wins over a versioned sibling. Returns None when
    the directory is absent or holds no candidate.
    """
    if not os.path.isdir(mjcf_dir):
        return None
    candidates = sorted(glob.glob(os.path.join(mjcf_dir, "galbot_s1*.xml")), key=lambda p: (len(p), p))
    return candidates[0] if candidates else None

# --------------------------------------------------------------------------- #
# Machine / group tables (verified against planning/model_mapping.json)       #
# --------------------------------------------------------------------------- #

MACHINE_G1 = 0  # == MachineType.G1
MACHINE_S1 = 1  # == MachineType.S1

# Lifecycle flags (DESIGN.md §5.5): UNINIT -> RUNNING -> SHUTDOWN_REQUESTED -> DESTROYED
UNINIT = "uninit"
RUNNING = "running"
SHUTDOWN_REQUESTED = "shutdown_requested"
DESTROYED = "destroyed"

_LEFT_ARM_JOINTS = tuple(f"left_arm_joint{i}" for i in range(1, 8))
_RIGHT_ARM_JOINTS = tuple(f"right_arm_joint{i}" for i in range(1, 8))
_HEAD_JOINTS = ("head_joint1", "head_joint2")
_TORSO_JOINTS = ("torso_lift_joint1",)
_LEG_JOINTS = tuple(f"leg_joint{i}" for i in range(1, 6))  # G1 compat, emulated
_CHASSIS_JOINTS = ("chassis_x", "chassis_y", "chassis_yaw")  # planar view of the free base

#: SDK joint-group -> ordered SDK joint names.  Groups with no S1 counterpart
#: (dexhand / suction / camera) are resolvable with zero DOF.
SDK_GROUPS: dict[str, tuple[str, ...]] = {
    "left_arm": _LEFT_ARM_JOINTS,
    "right_arm": _RIGHT_ARM_JOINTS,
    "head": _HEAD_JOINTS,
    "torso": _TORSO_JOINTS,
    "leg": _LEG_JOINTS,
    "chassis": _CHASSIS_JOINTS,
    "swerve_chassis": _CHASSIS_JOINTS,
    "left_gripper": ("left_gripper_joint1",),
    "right_gripper": ("right_gripper_joint1",),
    "left_dexhand": (),
    "right_dexhand": (),
    "left_suction_cup": (),
    "right_suction_cup": (),
    "left_camera": (),
    "right_camera": (),
}


@dataclasses.dataclass(frozen=True)
class MachineSpec:
    """Per-MachineType group vocabulary and empty-args read order (§10.10)."""

    name: str
    default_order: tuple[str, ...]  # get_joint_positions([], []) group order
    group_names: tuple[str, ...]    # get_joint_group_names() vocabulary


MACHINE_SPECS: dict[int, MachineSpec] = {
    MACHINE_G1: MachineSpec(
        name="G1",
        default_order=("chassis", "head", "left_arm", "right_arm", "leg"),
        group_names=(
            "chassis", "head", "left_arm", "left_dexhand", "left_gripper",
            "left_suction_cup", "leg", "right_arm", "right_dexhand",
            "right_gripper", "right_suction_cup",
        ),
    ),
    MACHINE_S1: MachineSpec(
        name="S1",
        default_order=("torso", "head", "left_arm", "right_arm"),
        group_names=(
            "head", "left_arm", "left_camera", "left_gripper", "right_arm",
            "right_camera", "right_gripper", "swerve_chassis", "torso",
        ),
    ),
}

# Leg emulation (model_mapping.json sdk_groups.leg.policy): S1 has no legs.
# leg_joint2 (G1 knee, primary height DOF) maps linearly onto torso_lift_joint1:
#   lift_m = clip(leg_joint2 / LEG2_SCALE, 0, 0.75);  read: leg_joint2 = lift_m * LEG2_SCALE
# leg_joint1/3/4/5 are virtual-static: stored, echoed, never touch qpos.
LEG2_SCALE = 1.0 / 0.75  # rad per meter (convention: 1.0 rad over the full 0.75 m stroke)

# Gripper theta <-> width LUT (model_mapping.json gripper_width_model, 21 pairs,
# strictly monotonic decreasing width; identical for left and right grippers).
GRIPPER_LUT: tuple[tuple[float, float], ...] = (
    (0.0, 0.082774), (0.08178, 0.079398), (0.16357, 0.075792), (0.24535, 0.071979),
    (0.32714, 0.067985), (0.40892, 0.063837), (0.49071, 0.059562), (0.57249, 0.05519),
    (0.65428, 0.050748), (0.73606, 0.046267), (0.81785, 0.041778), (0.89963, 0.037309),
    (0.98142, 0.032891), (1.0632, 0.028553), (1.14499, 0.024324), (1.22677, 0.020233),
    (1.30856, 0.016307), (1.39034, 0.012572), (1.47213, 0.009053), (1.55391, 0.005774),
    (1.6357, 0.002756),
)
_LUT_THETA = np.array([t for t, _ in GRIPPER_LUT])
_LUT_WIDTH = np.array([w for _, w in GRIPPER_LUT])
GRIPPER_THETA_MIN = float(_LUT_THETA[0])
GRIPPER_THETA_MAX = float(_LUT_THETA[-1])
GRIPPER_WIDTH_MIN = float(_LUT_WIDTH[-1])   # 0.002756 m (closed)
GRIPPER_WIDTH_MAX = float(_LUT_WIDTH[0])    # 0.082774 m (open)


def gripper_width_from_theta(theta: float) -> float:
    """Finger-pad separation (m) for a drive-joint angle (rad); LUT interpolation."""
    t = min(max(float(theta), GRIPPER_THETA_MIN), GRIPPER_THETA_MAX)
    return float(np.interp(t, _LUT_THETA, _LUT_WIDTH))


def gripper_theta_from_width(width_m: float) -> float:
    """Drive-joint angle (rad) for a pad separation (m); clamps into the LUT domain."""
    w = min(max(float(width_m), GRIPPER_WIDTH_MIN), GRIPPER_WIDTH_MAX)
    # width is strictly decreasing in theta -> flip for np.interp (needs ascending x)
    return float(np.interp(w, _LUT_WIDTH[::-1], _LUT_THETA[::-1]))


# Chain -> end-effector body (all verified present in galbot_s1_v1_1_0.xml).
CHAIN_EE_BODY: dict[str, str] = {
    "left_arm": "left_arm_end_effector_mount_link",
    "right_arm": "right_arm_end_effector_mount_link",
    "head": "head_end_effector_mount_link",
    "torso": "torso_lift_link1",
    "left_gripper": "left_gripper_base_link",
    "right_gripper": "right_gripper_base_link",
}

# Synthetic camera convention (model_mapping.json synthetic_camera): 5 cm forward,
# 5 cm up from the mount, optical axis along parent +X, quat wxyz [.5,.5,-.5,-.5].
_CAM_OFF_POS = (0.05, 0.0, 0.05)
_CAM_OFF_QUAT = (0.5, 0.5, -0.5, -0.5)  # wxyz

# Frame alias -> (host body, offset pos, offset quat wxyz).  DESIGN.md §5.5.
_OPTICAL_FRAMES: dict[str, str] = {
    "head_camera_color_optical_frame": "head_end_effector_mount_link",
    "head_camera": "head_end_effector_mount_link",
    "front_head": "head_end_effector_mount_link",
    "left_arm_camera_color_optical_frame": "left_arm_end_effector_mount_link",
    "right_arm_camera_color_optical_frame": "right_arm_end_effector_mount_link",
}
_WORLD_FRAMES = ("world", "map", "odom")

# SensorType value -> name (api_contract.md §4 SensorType; embedded so this
# module never has to import the enum module).
SENSOR_NAMES: dict[int, str] = {
    0: "HEAD_LEFT_CAMERA", 1: "HEAD_RIGHT_CAMERA", 2: "LEFT_ARM_CAMERA",
    3: "RIGHT_ARM_CAMERA", 4: "LEFT_ARM_DEPTH_CAMERA", 5: "RIGHT_ARM_DEPTH_CAMERA",
    6: "BASE_LIDAR", 7: "HEAD_LIDAR", 8: "BACK_LIDAR", 9: "CHASSIS_LIDAR",
    10: "HEAD_IMU", 11: "BACK_IMU", 12: "CHASSIS_IMU", 13: "TORSO_IMU",
    14: "LIDAR_IMU", 15: "BASE_ULTRASONIC", 16: "LEFT_FRONT_SURROUND_CAMERA",
    17: "RIGHT_FRONT_SURROUND_CAMERA", 18: "LEFT_REAR_SURROUND_CAMERA",
    19: "RIGHT_REAR_SURROUND_CAMERA",
}
_SENSOR_FRAME_ID: dict[int, str] = {
    0: "head_camera_color_optical_frame", 1: "head_camera_color_optical_frame",
    2: "left_arm_camera_color_optical_frame", 3: "right_arm_camera_color_optical_frame",
    4: "left_arm_camera_color_optical_frame", 5: "right_arm_camera_color_optical_frame",
}

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720   # 720x1280 shape is load-bearing: ex5/ex10 hardcode the reshape
CAMERA_FOVY_DEG = 58.0
_LIDAR_POINTS = 360


@dataclasses.dataclass(frozen=True)
class JointRef:
    """One SDK-visible joint resolved onto the model (or an emulation).

    kind:
      * ``model``   — 1-DOF model joint, SDK value == qpos value
      * ``leg2``    — G1 leg_joint2 emulated on torso_lift_joint1 via LEG2_SCALE
      * ``virtual`` — stored & echoed only (leg_joint1/3/4/5); never touches qpos
      * ``gripper`` — drive joint qpos + coupled (-theta, +theta) qpos triple
      * ``base_x`` / ``base_y`` / ``base_yaw`` — planar components of the free base
    """

    name: str
    kind: str
    qpos: int = -1                      # qpos address of the driven model joint
    lo: float = -math.inf               # SDK-side clip range
    hi: float = math.inf
    coupled: tuple = ()                 # ((qpos_adr, sign), ...) for kind == "gripper"


# --------------------------------------------------------------------------- #
# Quaternion helpers.  The wxyz<->xyzw conversion lives in exactly these two   #
# functions and nowhere else in the shim (DESIGN.md §6).                       #
# --------------------------------------------------------------------------- #

def _wxyz_to_xyzw(q) -> list[float]:
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _xyzw_to_wxyz(q) -> np.ndarray:
    return np.array([float(q[3]), float(q[0]), float(q[1]), float(q[2])])


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # wxyz
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def _quat_conj(q: np.ndarray) -> np.ndarray:  # wxyz
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 0.0 else np.array([1.0, 0.0, 0.0, 0.0])


def _quat_to_mat(q: np.ndarray) -> np.ndarray:  # wxyz -> 3x3
    w, x, y, z = _quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _quat_rotvec(q: np.ndarray) -> np.ndarray:
    """Rotation vector (angle * axis) of a wxyz quaternion."""
    q = _quat_normalize(q)
    if q[0] < 0.0:
        q = -q
    v = q[1:4]
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.zeros(3)
    angle = 2.0 * math.atan2(s, float(q[0]))
    return v * (angle / s)


def _yaw_quat(yaw: float) -> np.ndarray:  # wxyz, rotation about +Z
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _quat_yaw(q) -> float:  # wxyz
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


# --------------------------------------------------------------------------- #
# Stdlib PNG encoder (cv2.imdecode-compatible; no cv2 dependency in the shim) #
# --------------------------------------------------------------------------- #

def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + tag + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))


def _encode_png_rgb8(img: np.ndarray) -> bytes:
    """Encode an (H, W, 3) uint8 RGB array as a valid truecolor PNG."""
    h, w = img.shape[:2]
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, color type 2 (RGB)
    rows = np.concatenate(
        [np.zeros((h, 1), dtype=np.uint8), img.reshape(h, w * 3)], axis=1
    )  # filter byte 0 per scanline
    return (b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"IHDR", ihdr)
            + _png_chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
            + _png_chunk(b"IEND", b""))


# --------------------------------------------------------------------------- #
# The world                                                                   #
# --------------------------------------------------------------------------- #

class World:
    """Singleton MuJoCo world.  Use :func:`get_world`; do not construct twice."""

    def __init__(self) -> None:
        self.state: str = UNINIT
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._scratch: mujoco.MjData | None = None
        self._refs_by_name: dict[str, JointRef] = {}
        self._group_refs: dict[str, tuple[JointRef, ...]] = {}
        self._chains: dict[str, dict] = {}       # chain -> {"refs", "qpos", "dof", "lo", "hi", "body"}
        self._body_ids: dict[str, int] = {}
        self._virtual: dict[str, float] = {}     # leg_joint1/3/4/5 stored commands
        self._png_cache: dict[tuple, bytes] = {}
        self._depth_cache: dict[tuple, bytes] = {}
        self._lidar_cache: dict[int, bytes] = {}
        # Navigation bookkeeping (DESIGN.md §5.5 "Base")
        self.localized: bool = False
        self.last_goal: list[float] | None = None
        # Registries (plain dicts; bookkeeping only — DESIGN.md accepted delta)
        self.obstacles: dict[str, dict] = {}
        self.tools: dict[str, str] = {}          # chain -> tool name
        self.attached_objects: dict[str, dict] = {}
        self.bounding_boxes: dict[int, dict] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle (flag-based; DESIGN.md §4 / §5.5)                        #
    # ------------------------------------------------------------------ #

    def init(self) -> bool:
        """Load the model and enter RUNNING.  Idempotent-True while RUNNING;
        False after destroy (no re-init in one process, contract §10.3).
        Missing model file raises RuntimeError here — never at import."""
        if self.state == RUNNING:
            return True
        if self.state in (SHUTDOWN_REQUESTED, DESTROYED):
            _LOG.warning("GalbotSim world: init() after shutdown/destroy is not supported")
            return False
        xml_path = os.environ.get(MODEL_XML_ENV) or _discover_model_xml()
        if not xml_path or not os.path.isfile(xml_path):
            found = sorted(os.path.basename(p) for p in glob.glob(os.path.join(DEFAULT_MJCF_DIR, "*.xml")))
            raise RuntimeError(
                f"GalbotSim: S1 MJCF model not found. Looked for galbot_s1*.xml in "
                f"'{DEFAULT_MJCF_DIR}' (found: {found or 'nothing'}). Run ./quickstart.sh, "
                f"or set {MODEL_XML_ENV} to an MJCF from "
                f"github.com/GalaxyGeneralRobotics/galbot_s1_description."
            )
        try:
            self._model = mujoco.MjModel.from_xml_path(xml_path)
        except Exception as exc:  # surface a clear, actionable failure
            raise RuntimeError(f"GalbotSim: failed to load MJCF '{xml_path}': {exc}") from exc
        self._data = mujoco.MjData(self._model)
        self._scratch = mujoco.MjData(self._model)
        self._build_tables()
        mujoco.mj_forward(self._model, self._data)
        self.state = RUNNING
        self.localized = True  # kills the ex3/ex10 is_localized() polling loops
        _LOG.info("GalbotSim world initialized from %s", xml_path)
        return True

    def is_running(self) -> bool:
        return self.state == RUNNING

    def request_shutdown(self) -> None:
        """Non-blocking flag flip; safe in any state."""
        if self.state == RUNNING:
            self.state = SHUTDOWN_REQUESTED

    def wait_for_shutdown(self) -> None:
        """Returns immediately — nothing runs in the background (DESIGN.md §3)."""
        return None

    def destroy(self) -> None:
        """Terminal; safe in any state, including after a failed init."""
        self.state = DESTROYED

    def _require_model(self) -> None:
        if self._model is None or self._data is None:
            raise RuntimeError("GalbotSim world is not initialized (call init() first)")

    # ------------------------------------------------------------------ #
    # Table construction (indices resolved via mj_name2id at load)       #
    # ------------------------------------------------------------------ #

    def _jid(self, joint_name: str) -> int:
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            raise RuntimeError(f"GalbotSim: model joint '{joint_name}' missing from MJCF")
        return jid

    def _model_ref(self, sdk_name: str, model_joint: str | None = None) -> JointRef:
        jid = self._jid(model_joint or sdk_name)
        adr = int(self._model.jnt_qposadr[jid])
        if self._model.jnt_limited[jid]:
            lo, hi = (float(v) for v in self._model.jnt_range[jid])
        else:
            lo, hi = -math.inf, math.inf
        return JointRef(name=sdk_name, kind="model", qpos=adr, lo=lo, hi=hi)

    def _build_tables(self) -> None:
        m = self._model
        for i in range(m.nbody):
            self._body_ids[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)] = i

        refs: dict[str, JointRef] = {}
        for name in (*_LEFT_ARM_JOINTS, *_RIGHT_ARM_JOINTS, *_HEAD_JOINTS, *_TORSO_JOINTS):
            refs[name] = self._model_ref(name)

        # Leg emulation: leg_joint2 -> torso lift; 1/3/4/5 virtual-static.
        lift = refs["torso_lift_joint1"]
        refs["leg_joint2"] = JointRef(
            name="leg_joint2", kind="leg2", qpos=lift.qpos,
            lo=lift.lo * LEG2_SCALE, hi=lift.hi * LEG2_SCALE,
        )
        for name in ("leg_joint1", "leg_joint3", "leg_joint4", "leg_joint5"):
            refs[name] = JointRef(name=name, kind="virtual")
            self._virtual.setdefault(name, 0.0)

        # Grippers: drive joint + coupled (-theta, +theta) qpos triple.  The
        # position actuators are kp=1.0 (too weak) — write qpos directly.
        for side in ("left", "right"):
            drive = self._model_ref(f"{side}_gripper_joint1", f"{side}_active_joint1")
            coupled = (
                (int(m.jnt_qposadr[self._jid(f"{side}_active_joint2")]), -1.0),
                (int(m.jnt_qposadr[self._jid(f"{side}_passive_joint")]), +1.0),
            )
            refs[drive.name] = dataclasses.replace(drive, kind="gripper", coupled=coupled)

        # Planar view of the free base (G1 'chassis' / S1 'swerve_chassis').
        refs["chassis_x"] = JointRef(name="chassis_x", kind="base_x")
        refs["chassis_y"] = JointRef(name="chassis_y", kind="base_y")
        refs["chassis_yaw"] = JointRef(name="chassis_yaw", kind="base_yaw")

        self._refs_by_name = refs
        self._group_refs = {
            group: tuple(refs[j] for j in joints) for group, joints in SDK_GROUPS.items()
        }

        # IK/FK chains: groups made purely of model joints, with an EE body.
        for chain in ("left_arm", "right_arm", "head", "torso"):
            crefs = self._group_refs[chain]
            jids = [self._jid(r.name) for r in crefs]
            self._chains[chain] = {
                "refs": crefs,
                "qpos": np.array([r.qpos for r in crefs], dtype=int),
                "dof": np.array([int(m.jnt_dofadr[j]) for j in jids], dtype=int),
                "lo": np.array([r.lo for r in crefs]),
                "hi": np.array([r.hi for r in crefs]),
                "body": self._body_ids[CHAIN_EE_BODY[chain]],
            }

    # ------------------------------------------------------------------ #
    # Joint resolution / read / write (the synchronous choke point)      #
    # ------------------------------------------------------------------ #

    def resolve(self, joint_groups=(), joint_names=(), machine: int = MACHINE_G1) -> list[JointRef]:
        """Name-precedence resolution: joint_names > joint_groups > machine default.

        Multi-group lists concatenate in list order (14 for both arms, 21 for
        ["leg","head","left_arm","right_arm"]).  Unknown names/groups raise
        KeyError (managers map that to INVALID_INPUT).
        """
        self._require_model()
        names = list(joint_names or ())
        if names:
            try:
                return [self._refs_by_name[n] for n in names]
            except KeyError as exc:
                raise KeyError(f"unknown joint name {exc.args[0]!r}") from None
        groups = list(joint_groups or ())
        if not groups:
            spec = MACHINE_SPECS.get(int(machine))
            if spec is None:
                raise KeyError(f"unknown machine type {machine!r}")
            groups = list(spec.default_order)
        out: list[JointRef] = []
        for g in groups:
            try:
                out.extend(self._group_refs[g])
            except KeyError:
                raise KeyError(f"unknown joint group {g!r}") from None
        return out

    def _read_ref(self, ref: JointRef) -> float:
        d = self._data
        if ref.kind in ("model", "gripper"):
            return float(d.qpos[ref.qpos])
        if ref.kind == "leg2":
            return float(d.qpos[ref.qpos]) * LEG2_SCALE
        if ref.kind == "virtual":
            return self._virtual.get(ref.name, 0.0)
        if ref.kind == "base_x":
            return float(d.qpos[0])
        if ref.kind == "base_y":
            return float(d.qpos[1])
        if ref.kind == "base_yaw":
            return _quat_yaw(d.qpos[3:7])
        raise AssertionError(f"unhandled JointRef kind {ref.kind!r}")

    @staticmethod
    def _write_ref(qpos: np.ndarray, ref: JointRef, value: float) -> None:
        """Write one SDK-space value into a qpos vector (live or scratch)."""
        if ref.kind == "model":
            qpos[ref.qpos] = value
        elif ref.kind == "leg2":
            qpos[ref.qpos] = value / LEG2_SCALE
        elif ref.kind == "gripper":
            qpos[ref.qpos] = value
            for adr, sign in ref.coupled:
                qpos[adr] = sign * value
        elif ref.kind == "base_x":
            qpos[0] = value
        elif ref.kind == "base_y":
            qpos[1] = value
        elif ref.kind == "base_yaw":
            qpos[3:7] = _yaw_quat(value)  # flat-base assumption (documented)
        # "virtual" is handled by callers (no physical effect)

    def get_positions(self, refs) -> list[float]:
        self._require_model()
        return [self._read_ref(r) for r in refs]

    def set_positions(self, refs, values, substeps: int = 20) -> None:
        """THE synchronous motion choke point (DESIGN.md §3/§5.5).

        Clip each target to its SDK-side range (silently, debug log — the
        example1 heart pose commands arm_joint6 = ±0.82 against ±0.7854 and
        must still SUCCESS), linearly interpolate qpos from current to target
        over ``substeps``, ``mj_forward`` per substep.  Never sleeps, never
        calls mj_step.  Raises ValueError on length mismatch / non-finite.
        """
        self._require_model()
        refs = list(refs)
        vals = [float(v) for v in values]
        if len(vals) != len(refs):
            raise ValueError(f"expected {len(refs)} values, got {len(vals)}")
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("non-finite joint target")
        plan: list[tuple[JointRef, float, float]] = []
        for ref, v in zip(refs, vals):
            target = min(max(v, ref.lo), ref.hi)
            if target != v:
                _LOG.debug("GalbotSim: clipped %s target %.6g -> %.6g", ref.name, v, target)
            if ref.kind == "virtual":
                self._virtual[ref.name] = target
                continue
            plan.append((ref, self._read_ref(ref), target))
        if not plan:
            return
        n = max(1, int(substeps))
        d = self._data
        for k in range(1, n + 1):
            a = k / n
            for ref, start, target in plan:
                self._write_ref(d.qpos, ref, start + (target - start) * a)
            mujoco.mj_forward(self._model, d)

    # ------------------------------------------------------------------ #
    # Frames / kinematics                                                #
    # ------------------------------------------------------------------ #

    def _resolve_frame(self, frame, chain: str | None = None):
        """frame name -> (body_id, offset_pos(3), offset_quat wxyz(4))."""
        zero = np.zeros(3)
        ident = np.array([1.0, 0.0, 0.0, 0.0])
        if frame is None or frame == "":
            if chain and chain in CHAIN_EE_BODY:
                return self._body_ids[CHAIN_EE_BODY[chain]], zero, ident
            raise KeyError("empty frame with no chain context")
        frame = str(frame)
        if frame in _WORLD_FRAMES:
            return 0, zero, ident  # mujoco world body
        bid = self._body_ids.get(frame)
        if bid is not None:
            return bid, zero, ident
        if frame in ("EndEffector", "ee_base"):
            if chain and chain in CHAIN_EE_BODY:
                return self._body_ids[CHAIN_EE_BODY[chain]], zero, ident
            raise KeyError(f"frame {frame!r} requires a chain context")
        if frame in CHAIN_EE_BODY:  # chain name used as a frame (ex2 step 8)
            return self._body_ids[CHAIN_EE_BODY[frame]], zero, ident
        host = _OPTICAL_FRAMES.get(frame)
        if host is not None:
            return self._body_ids[host], np.array(_CAM_OFF_POS), np.array(_CAM_OFF_QUAT)
        raise KeyError(f"unknown frame {frame!r}")

    def _world_pose_on(self, data, frame, chain: str | None = None):
        bid, off_p, off_q = self._resolve_frame(frame, chain)
        p = np.array(data.xpos[bid])
        q = np.array(data.xquat[bid])
        if off_p.any() or off_q[0] != 1.0:
            p = p + _quat_to_mat(q) @ off_p
            q = _quat_normalize(_quat_mul(q, off_q))
        return p, q

    def _pose_on(self, data, frame, ref="base_link", chain: str | None = None) -> list[float]:
        pf, qf = self._world_pose_on(data, frame, chain)
        pr, qr = self._world_pose_on(data, ref, chain)
        p_rel = _quat_to_mat(qr).T @ (pf - pr)
        q_rel = _quat_normalize(_quat_mul(_quat_conj(qr), qf))
        return [float(p_rel[0]), float(p_rel[1]), float(p_rel[2]), *_wxyz_to_xyzw(q_rel)]

    def frame_pose(self, frame, ref: str = "base_link", chain: str | None = None) -> list[float]:
        """pose7 [x,y,z,qx,qy,qz,qw] of ``frame`` expressed in ``ref`` (live data)."""
        self._require_model()
        return self._pose_on(self._data, frame, ref, chain)

    def frame_names(self) -> list[str]:
        """Model body names plus the alias vocabulary."""
        self._require_model()
        names = [n for n in self._body_ids if n != "world"]
        for extra in (*_WORLD_FRAMES, "EndEffector", *_OPTICAL_FRAMES):
            if extra not in names:
                names.append(extra)
        return names

    def chain_names(self) -> list[str]:
        return list(self._chains) if self._chains else ["left_arm", "right_arm", "head", "torso"]

    def _scratch_with(self, joint_state=None) -> mujoco.MjData:
        """Scratch MjData primed from live qpos with optional {chain: joints} overrides."""
        sd = self._scratch
        sd.qpos[:] = self._data.qpos
        for chain, vals in (joint_state or {}).items():
            refs = self._group_refs.get(chain)
            if refs is None:
                raise KeyError(f"unknown chain {chain!r}")
            vals = [float(v) for v in vals]
            if len(vals) != len(refs):
                raise ValueError(f"chain {chain!r} expects {len(refs)} joints, got {len(vals)}")
            for ref, v in zip(refs, vals):
                if ref.kind == "virtual":
                    continue  # no physical effect; never mutate stored state from FK
                self._write_ref(sd.qpos, ref, min(max(v, ref.lo), ref.hi))
        mujoco.mj_kinematics(self._model, sd)
        return sd

    def fk_with_joint_state(self, frame, ref: str = "base_link",
                            joint_state=None, chain: str | None = None) -> list[float]:
        """FK on scratch data: pose7 of ``frame`` in ``ref`` for the given
        {chain: joint_values} overrides (unspecified joints = live values)."""
        self._require_model()
        sd = self._scratch_with(joint_state)
        return self._pose_on(sd, frame, ref, chain)

    def jacobian(self, chain: str, frame: str = "EndEffector",
                 ref: str = "base_link", joint_state=None) -> list[list[float]]:
        """6xN geometric jacobian of ``frame`` w.r.t. the chain's joints, rows
        expressed in ``ref`` ([3 translational; 3 rotational])."""
        self._require_model()
        info = self._chains.get(chain)
        if info is None:
            raise KeyError(f"unknown chain {chain!r}")
        sd = self._scratch_with(joint_state)
        mujoco.mj_comPos(self._model, sd)
        bid, off_p, _ = self._resolve_frame(frame, chain)
        point = np.array(sd.xpos[bid]) + _quat_to_mat(np.array(sd.xquat[bid])) @ off_p
        nv = self._model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jac(self._model, sd, jacp, jacr, point, bid)
        cols = info["dof"]
        J = np.vstack([jacp[:, cols], jacr[:, cols]])
        _, qr = self._world_pose_on(sd, ref, chain)
        R_t = _quat_to_mat(qr).T
        J = np.vstack([R_t @ J[0:3], R_t @ J[3:6]])
        return [[float(v) for v in row] for row in J]

    def ik(self, chain: str, target_pose7, ref: str = "base_link",
           seed=None, target_frame: str = "EndEffector"):
        """Damped-least-squares IK on scratch data (never mutates live state).

        lambda=1e-2, <=200 iterations, tolerance 1e-4 m / 1e-3 rad, joint-range
        clamp, seeded from live qpos unless ``seed`` is given.  Returns the
        chain's joint values (list[float]) or None if unconverged.
        """
        self._require_model()
        info = self._chains.get(chain)
        if info is None:
            raise KeyError(f"unknown chain {chain!r}")
        m = self._model
        sd = self._scratch
        sd.qpos[:] = self._data.qpos
        qadr, dofs, lo, hi = info["qpos"], info["dof"], info["lo"], info["hi"]
        if seed is not None:
            s = np.clip(np.asarray([float(v) for v in seed]), lo, hi)
            if s.shape[0] != qadr.shape[0]:
                raise ValueError(f"seed length {s.shape[0]} != chain DOF {qadr.shape[0]}")
            sd.qpos[qadr] = s
        mujoco.mj_kinematics(m, sd)

        # Target in world coordinates: ref pose (on scratch) composed with pose7.
        p_ref, q_ref = self._world_pose_on(sd, ref, chain)
        tp = np.asarray([float(v) for v in target_pose7])
        p_t = p_ref + _quat_to_mat(q_ref) @ tp[0:3]
        q_t = _quat_normalize(_quat_mul(q_ref, _xyzw_to_wxyz(tp[3:7])))

        bid, off_p, off_q = self._resolve_frame(target_frame, chain)
        nv = m.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        lam2 = 1e-2 ** 2
        eye6 = np.eye(6)
        for _ in range(200):
            R_b = _quat_to_mat(np.array(sd.xquat[bid]))
            p_c = np.array(sd.xpos[bid]) + R_b @ off_p
            q_c = _quat_normalize(_quat_mul(np.array(sd.xquat[bid]), off_q))
            e_p = p_t - p_c
            e_r = _quat_rotvec(_quat_mul(q_t, _quat_conj(q_c)))
            if np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(e_r) < 1e-3:
                return [float(v) for v in sd.qpos[qadr]]
            mujoco.mj_comPos(m, sd)
            mujoco.mj_jac(m, sd, jacp, jacr, p_c, bid)
            J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
            e = np.concatenate([e_p, e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + lam2 * eye6, e)
            dq = np.clip(dq, -0.3, 0.3)  # step limit for stability
            sd.qpos[qadr] = np.clip(sd.qpos[qadr] + dq, lo, hi)
            mujoco.mj_kinematics(m, sd)
        return None

    # ------------------------------------------------------------------ #
    # Free-joint base ops                                                #
    # ------------------------------------------------------------------ #

    def base_pose7(self) -> list[float]:
        """pose7 [x,y,z,qx,qy,qz,qw] of the free-joint base (world/odom/map frame)."""
        self._require_model()
        q = self._data.qpos
        return [float(q[0]), float(q[1]), float(q[2]), *_wxyz_to_xyzw(q[3:7])]

    def set_base_pose7(self, pose7, substeps: int = 20) -> None:
        """Teleport-by-interpolation of the free-joint base (qpos 0-6)."""
        self._require_model()
        tp = [float(v) for v in pose7]
        if len(tp) != 7 or not all(math.isfinite(v) for v in tp):
            raise ValueError("base pose must be 7 finite floats [x,y,z,qx,qy,qz,qw]")
        d = self._data
        p0 = np.array(d.qpos[0:3])
        q0 = np.array(d.qpos[3:7])
        p1 = np.array(tp[0:3])
        q1 = _xyzw_to_wxyz(tp[3:7])
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1  # shortest-arc nlerp
        n = max(1, int(substeps))
        for k in range(1, n + 1):
            a = k / n
            d.qpos[0:3] = p0 + (p1 - p0) * a
            d.qpos[3:7] = _quat_normalize(q0 + (q1 - q0) * a)
            mujoco.mj_forward(self._model, d)

    def base_pose2d(self) -> tuple[float, float, float]:
        """(x, y, yaw) planar view of the base."""
        self._require_model()
        q = self._data.qpos
        return float(q[0]), float(q[1]), _quat_yaw(q[3:7])

    def set_base_pose2d(self, x: float, y: float, yaw: float, substeps: int = 20) -> None:
        """Planar base motion; keeps current z, rebuilds the quat about +Z."""
        self._require_model()
        z = float(self._data.qpos[2])
        yq = _yaw_quat(float(yaw))
        self.set_base_pose7([float(x), float(y), z, *_wxyz_to_xyzw(yq)], substeps=substeps)

    def apply_base_velocity(self, vx: float, vy: float, wyaw: float,
                            duration_s: float, substeps: int = 20) -> None:
        """Single synchronous integration of a body-frame twist over duration_s
        (DESIGN.md §5.5).  duration <= 0 integrates nothing (the sync engine has
        no notion of an open-ended velocity command)."""
        self._require_model()
        duration = max(0.0, float(duration_s))
        if duration == 0.0:
            return
        x, y, yaw = self.base_pose2d()
        n = max(1, int(substeps))
        dt = duration / n
        d = self._data
        for _ in range(n):
            x += (float(vx) * math.cos(yaw) - float(vy) * math.sin(yaw)) * dt
            y += (float(vx) * math.sin(yaw) + float(vy) * math.cos(yaw)) * dt
            yaw += float(wyaw) * dt
            d.qpos[0] = x
            d.qpos[1] = y
            d.qpos[3:7] = _yaw_quat(yaw)
            mujoco.mj_forward(self._model, d)

    # ------------------------------------------------------------------ #
    # Gripper convenience                                                #
    # ------------------------------------------------------------------ #

    def gripper_state(self, end_effector: str) -> tuple[float, list[float]]:
        """(width_m, [theta, -theta, +theta]) for 'left_gripper'/'right_gripper'."""
        self._require_model()
        refs = self._group_refs.get(end_effector)
        if not refs or refs[0].kind != "gripper":
            raise KeyError(f"unknown gripper {end_effector!r}")
        ref = refs[0]
        theta = float(self._data.qpos[ref.qpos])
        joints = [theta] + [sign * theta for _, sign in ref.coupled]
        return gripper_width_from_theta(theta), joints

    # ------------------------------------------------------------------ #
    # Machine vocabulary helpers                                         #
    # ------------------------------------------------------------------ #

    def group_names(self, machine: int = MACHINE_G1) -> list[str]:
        spec = MACHINE_SPECS.get(int(machine))
        if spec is None:
            raise KeyError(f"unknown machine type {machine!r}")
        return list(spec.group_names)

    def sdk_joint_names(self, machine: int = MACHINE_G1, joint_groups=()) -> list[str]:
        groups = list(joint_groups or ())
        if not groups:
            spec = MACHINE_SPECS.get(int(machine))
            if spec is None:
                raise KeyError(f"unknown machine type {machine!r}")
            groups = list(spec.default_order)
        out: list[str] = []
        for g in groups:
            try:
                out.extend(SDK_GROUPS[g])
            except KeyError:
                raise KeyError(f"unknown joint group {g!r}") from None
        return out

    # ------------------------------------------------------------------ #
    # Procedural sensors (deterministic; no GL, no cv2 — DESIGN.md §5.5) #
    # Served regardless of the init enable-set (ex4 queries TORSO_IMU    #
    # without enabling it).                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sensor_id(sensor) -> int:
        try:
            return int(sensor)
        except (TypeError, ValueError):
            return 0

    def _header(self, sid: int, default_frame: str | None = None) -> dict:
        frame = _SENSOR_FRAME_ID.get(sid) or default_frame or SENSOR_NAMES.get(sid, f"sensor_{sid}").lower()
        return {"frame_id": frame, "timestamp_ns": self.now_ns()}

    def rgb(self, sensor) -> dict:
        """1280x720 deterministic gradient+checker RGB, stdlib-PNG-encoded bytes.
        format value is 'rgb8' (the tutorial-observed value, DESIGN.md §5.5)."""
        sid = self._sensor_id(sensor)
        key = (sid, CAMERA_WIDTH, CAMERA_HEIGHT)
        png = self._png_cache.get(key)
        if png is None:
            w, h = CAMERA_WIDTH, CAMERA_HEIGHT
            xs = np.arange(w, dtype=np.uint32)
            ys = np.arange(h, dtype=np.uint32)
            r = np.broadcast_to((xs * 255 // max(1, w - 1)).astype(np.uint8), (h, w))
            g = np.broadcast_to((ys * 255 // max(1, h - 1)).astype(np.uint8)[:, None], (h, w))
            checker = (((xs[None, :] // 40) ^ (ys[:, None] // 40)) & 1).astype(np.uint8)
            b = (checker * 200 + (sid * 13) % 55).astype(np.uint8)
            img = np.stack([r, g, b], axis=-1)
            png = _encode_png_rgb8(img)
            self._png_cache[key] = png
        return {"header": self._header(sid), "format": "rgb8", "data": png}

    def depth(self, sensor) -> dict:
        """720x1280 uint16 raw depth (mm), depth_scale 1000.  The 720x1280 shape
        is a hard requirement — ex5/ex10 hardcode the reshape."""
        sid = self._sensor_id(sensor)
        key = (sid, CAMERA_WIDTH, CAMERA_HEIGHT)
        raw = self._depth_cache.get(key)
        if raw is None:
            w, h = CAMERA_WIDTH, CAMERA_HEIGHT
            x = np.arange(w)[None, :]
            y = np.arange(h)[:, None]
            plane = 1200.0 + 300.0 * np.sin(x / 97.0) + 200.0 * np.cos(y / 71.0) + 7.0 * sid
            raw = np.clip(plane, 300.0, 5000.0).astype("<u2").tobytes()
            self._depth_cache[key] = raw
        return {
            "header": self._header(sid),
            "format": "16UC1",
            "data": raw,
            "height": CAMERA_HEIGHT,
            "width": CAMERA_WIDTH,
            "depth_scale": 1000,
        }

    def lidar(self, sensor) -> dict:
        """PointCloud2-style dict: one 360-point ring of float32 x/y/z."""
        sid = self._sensor_id(sensor)
        raw = self._lidar_cache.get(sid)
        if raw is None:
            n = _LIDAR_POINTS
            theta = np.arange(n) * (2.0 * math.pi / n)
            radius = 2.0 + 0.25 * np.sin(4.0 * theta + sid)
            pts = np.zeros((n, 3), dtype="<f4")
            pts[:, 0] = radius * np.cos(theta)
            pts[:, 1] = radius * np.sin(theta)
            pts[:, 2] = 0.1
            raw = pts.tobytes()
            self._lidar_cache[sid] = raw
        point_step = 12
        width = _LIDAR_POINTS
        return {
            "header": self._header(sid, default_frame="base_link"),
            "height": 1,
            "width": width,
            "fields": [
                {"name": "x", "offset": 0, "datatype": 7, "count": 1},  # 7 = FLOAT32
                {"name": "y", "offset": 4, "datatype": 7, "count": 1},
                {"name": "z", "offset": 8, "datatype": 7, "count": 1},
            ],
            "is_bigendian": False,
            "point_step": point_step,
            "row_step": point_step * width,
            "is_dense": True,
            "data": raw,
        }

    def imu(self, sensor) -> dict:
        """Static gravity-aligned IMU: accel (0, 0, 9.81), zero gyro/magnet."""
        sid = self._sensor_id(sensor)
        return {
            "timestamp_ns": self.now_ns(),
            "accel": {"x": 0.0, "y": 0.0, "z": 9.81},
            "gyro": {"x": 0.0, "y": 0.0, "z": 0.0},
            "magnet": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

    def intrinsic(self, sensor) -> dict:
        """Pinhole intrinsics for the synthetic 1280x720 fovy-58deg camera."""
        sid = self._sensor_id(sensor)
        f = (CAMERA_HEIGHT / 2.0) / math.tan(math.radians(CAMERA_FOVY_DEG) / 2.0)
        cx, cy = CAMERA_WIDTH / 2.0, CAMERA_HEIGHT / 2.0
        name = SENSOR_NAMES.get(sid, "")
        return {
            "header": self._header(sid),
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "distortion_model": "plumb_bob",
            "D": [0.0, 0.0, 0.0, 0.0, 0.0],
            "K": [f, 0.0, cx, 0.0, f, cy, 0.0, 0.0, 1.0],
            "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "P": [f, 0.0, cx, 0.0, 0.0, f, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
            "binning_x": 0,
            "binning_y": 0,
            "roi": [0, 0, 0, 0],
            "camera_type": "depth" if "DEPTH" in name else "rgb",
        }

    # ------------------------------------------------------------------ #
    # Time                                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def now_ns() -> int:
        """Wall-clock nanoseconds — there is no sim clock (DESIGN.md §3)."""
        return time.time_ns()


# --------------------------------------------------------------------------- #
# Singleton access                                                            #
# --------------------------------------------------------------------------- #

_WORLD: World | None = None


def get_world() -> World:
    """The process-wide world singleton (shared by all machine types — an
    accepted, documented limitation; DESIGN.md §5.5)."""
    global _WORLD
    if _WORLD is None:
        _WORLD = World()
    return _WORLD
