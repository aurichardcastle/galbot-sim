"""GalbotSim showcase: smooth robot motion with orbiting camera."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import imageio.v2 as imageio

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_demo.mp4")
MODEL = os.environ.get("GALBOTSIM_MODEL_XML",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "third_party", "galbot_s1_description", "mjcf",
                 "galbot_s1_v1_1_0.xml"))

spec = mujoco.MjSpec.from_file(MODEL)
spec.visual.global_.offwidth = 1280
spec.visual.global_.offheight = 720
model = spec.compile()
data = mujoco.MjData(model)

W, H, FPS = 1280, 720, 30
renderer = mujoco.Renderer(model, height=H, width=W)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.0, 0.0, 0.85]
cam.distance = 2.4
cam.elevation = -15

joint_map = {}
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    if name:
        joint_map[name] = model.jnt_qposadr[i]

LEFT_ARM = [f"left_arm_joint{i}" for i in range(1, 8)]
RIGHT_ARM = [f"right_arm_joint{i}" for i in range(1, 8)]
HEAD = ["head_joint1", "head_joint2"]

def set_joints(names, values):
    for name, val in zip(names, values):
        if name in joint_map:
            data.qpos[joint_map[name]] = val

def get_joints(names):
    return [float(data.qpos[joint_map[n]]) for n in names if n in joint_map]

def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)

frames = []

POSES = {
    "rest":       {"left": [0]*7, "right": [0]*7, "head": [0, 0]},
    "heart":      {"left": [1.53, 0.36, -2.54, -1.80, 0.12, -0.82, 0.09],
                   "right": [-1.53, -0.36, 2.54, 1.80, -0.12, 0.82, -0.09],
                   "head": [0, 0.1]},
    "wide":       {"left": [0.0, -1.3, 0.0, -0.3, 0.0, 0.0, 0.0],
                   "right": [0.0, 1.3, 0.0, 0.3, 0.0, 0.0, 0.0],
                   "head": [0, 0]},
    "reach":      {"left": [0.4, 0.0, 0.0, -1.4, 0.0, 0.6, 0.0],
                   "right": [-0.4, 0.0, 0.0, 1.4, 0.0, -0.6, 0.0],
                   "head": [0, 0.15]},
    "look_left":  {"left": [0]*7, "right": [0]*7, "head": [0.6, 0.1]},
    "look_right": {"left": [0]*7, "right": [0]*7, "head": [-0.6, 0.1]},
}

SEQUENCE = [
    ("rest",       40,  90),   # 1.3s hold, start azimuth
    ("heart",      60, 135),   # 2s transition
    ("heart",      45, 160),   # 1.5s hold
    ("wide",       50, 200),   # 1.7s transition
    ("wide",       30, 220),   # 1s hold
    ("reach",      50, 270),   # 1.7s transition
    ("reach",      30, 290),   # 1s hold
    ("look_left",  30, 320),   # 1s
    ("look_right", 30, 360),   # 1s
    ("rest",       50, 405),   # 1.7s transition
    ("rest",       30, 430),   # 1s hold
]

all_names = LEFT_ARM + RIGHT_ARM + HEAD

current_vals = np.array(get_joints(all_names), dtype=float)
current_az = 90.0

for pose_name, n_frames, target_az in SEQUENCE:
    pose = POSES[pose_name]
    target_vals = np.array(pose["left"] + pose["right"] + pose["head"], dtype=float)
    start_vals = current_vals.copy()
    start_az = current_az

    for f in range(n_frames):
        t = smoothstep((f + 1) / n_frames)
        interp = start_vals + (target_vals - start_vals) * t
        set_joints(all_names, interp)

        cam.azimuth = start_az + (target_az - start_az) * t

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render().copy())

    current_vals = target_vals.copy()
    current_az = target_az

imageio.mimsave(OUT, frames, fps=FPS, macro_block_size=1)
print(f"Wrote {OUT} ({len(frames)} frames @ {FPS}fps = {len(frames)/FPS:.1f}s)")
