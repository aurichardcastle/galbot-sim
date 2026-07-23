"""GalbotSim demo — heart gesture rendered to MP4.

Uses the galbot_sdk shim exactly as a real Galbot tutorial would:
GalbotRobot from galbot_sdk.g1, init, set_joint_positions. After each
SDK call the MuJoCo world already has the target qpos applied (the sim
is synchronous), so we grab the model/data from the world singleton
and render frames with mujoco.Renderer.

Joint targets are the same values as the official example1 tutorial
(example1_galbot_control_started.py).
"""

import pathlib

import imageio.v2 as imageio
import mujoco
import numpy as np

from galbot_sdk.g1 import GalbotRobot, ControlStatus

# ── output ──────────────────────────────────────────────────────────
OUT = pathlib.Path(__file__).parent / "heart_gesture.mp4"
WIDTH, HEIGHT = 640, 480
FPS = 30
INTERP_FRAMES = 60  # frames to interpolate between each pair of poses

# ── heart gesture targets (from example1) ───────────────────────────
JOINT_GROUPS = ["left_arm", "right_arm"]
#                      LA1    LA2    LA3    LA4    LA5    LA6    LA7
#                      RA1    RA2    RA3    RA4    RA5    RA6    RA7
HEART_POSE = [
     1.53,  0.36, -2.54, -1.80,  0.12, -0.82,  0.09,   # left_arm
    -1.53, -0.36,  2.54,  1.80, -0.12,  0.82, -0.09,   # right_arm
]

# ── camera (good angle for the S1) ─────────────────────────────────
def make_camera():
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.0, 0.0, 0.9]
    cam.distance = 2.6
    cam.azimuth = 135
    cam.elevation = -12
    return cam


def render_frame(renderer, data, cam):
    """Render one RGB frame from the current MuJoCo state."""
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def interpolate_and_render(renderer, model, data, cam, start_qpos, end_qpos, n_frames):
    """Linearly interpolate full qpos from start to end, rendering each step."""
    frames = []
    for i in range(n_frames):
        alpha = (i + 1) / n_frames
        data.qpos[:] = start_qpos + alpha * (end_qpos - start_qpos)
        mujoco.mj_forward(model, data)
        frames.append(render_frame(renderer, data, cam))
    return frames


def main():
    # ── 1. SDK init (exactly like a real tutorial) ──────────────────
    robot = GalbotRobot()
    ok = robot.init()
    assert ok, "GalbotRobot.init() failed"
    print(f"Robot initialized, running={robot.is_running()}")

    joint_names = robot.get_joint_names()
    print(f"Joint names ({len(joint_names)}): {joint_names}")

    # ── 2. Grab the world singleton for rendering ───────────────────
    from galbot_sdk._world import get_world
    world = get_world()
    model = world._model
    data = world._data

    cam = make_camera()
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    frames = []

    # ── 3. Capture the rest pose ────────────────────────────────────
    rest_qpos = data.qpos.copy()
    print("Captured rest pose")

    # hold rest pose for a moment (15 frames = 0.5s)
    for _ in range(15):
        frames.append(render_frame(renderer, data, cam))

    # ── 4. Interpolate rest -> heart pose ───────────────────────────
    # Set heart pose via SDK to get the target qpos
    status = robot.set_joint_positions(
        HEART_POSE, JOINT_GROUPS, [],
        is_blocking=True, max_speed=0.1, timeout_s=20.0,
    )
    assert status == ControlStatus.SUCCESS, f"set_joint_positions failed: {status}"
    heart_qpos = data.qpos.copy()
    print("Heart pose set via SDK")

    # Reset to rest and interpolate visually
    data.qpos[:] = rest_qpos
    mujoco.mj_forward(model, data)
    frames.extend(
        interpolate_and_render(renderer, model, data, cam,
                               rest_qpos, heart_qpos, INTERP_FRAMES)
    )

    # ── 5. Hold heart pose (30 frames = 1s) ─────────────────────────
    for _ in range(30):
        frames.append(render_frame(renderer, data, cam))

    # ── 6. Interpolate heart -> rest ────────────────────────────────
    frames.extend(
        interpolate_and_render(renderer, model, data, cam,
                               heart_qpos, rest_qpos, INTERP_FRAMES)
    )

    # ── 7. Hold rest pose at end (15 frames) ────────────────────────
    for _ in range(15):
        frames.append(render_frame(renderer, data, cam))

    # ── 8. Save MP4 ─────────────────────────────────────────────────
    imageio.mimsave(str(OUT), frames, fps=FPS, macro_block_size=1)
    print(f"Wrote {OUT} ({len(frames)} frames @ {FPS} fps, "
          f"{len(frames)/FPS:.1f}s)")

    # ── 9. Shutdown ─────────────────────────────────────────────────
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()
    print("Done.")


if __name__ == "__main__":
    main()
