# galbot-sim

[![tutorials](https://github.com/aurichardcastle/galbot-sim/actions/workflows/tutorials.yml/badge.svg)](https://github.com/aurichardcastle/galbot-sim/actions/workflows/tutorials.yml)

A small offline backend for the [Galbot SDK](https://github.com/GalaxyGeneralRobotics/GalbotSDK) Python API: Galbot's official G1 tutorials run on a laptop with no robot, backed by MuJoCo and Galbot's public [S1 description](https://github.com/GalaxyGeneralRobotics/galbot_s1_description). Mac/Linux, `mujoco` + `numpy` only.

All 10 official G1 tutorials pass **byte-unmodified**. That claim runs in CI on every push — clean Ubuntu machine, SDK examples and S1 model cloned fresh from Galbot's repos: [latest run](https://github.com/aurichardcastle/galbot-sim/actions/workflows/tutorials.yml).

## What "pass" asserts (and what it doesn't)

Passing means the SDK's API contract is satisfied: imports resolve, calls return the documented status codes and data shapes, joint targets are reached and read back consistently. It does **not** assert physics outcomes. Motion is kinematic — positions are interpolated and written directly, with no contact dynamics — grippers set width but do not grasp, and camera frames are procedural (the public S1 model ships with `ncam=0`, so RGB/depth/lidar are synthesized in the documented formats). This is a tool for developing and testing SDK code paths without hardware, not a physics simulator.

## Quickstart

One command — creates a venv, fetches Galbot's SDK examples and S1 model, runs the whole suite (about 5 minutes):

```bash
git clone https://github.com/aurichardcastle/galbot-sim && cd galbot-sim && bash quickstart.sh
```

To run your own SDK scripts against it:

```bash
PYTHONPATH=/path/to/galbot-sim \
python your_script.py
```

No code changes: `from galbot_sdk.g1 import GalbotRobot` resolves to this backend.

## Where this fits

Galbot's developer portal offers a browser-based online simulation with a deploy path to real robots. This backend covers the complementary cases: local development without an account or network, CI pipelines, and running the GitHub tutorials exactly as shipped. As far as I could find, no other offline mock of the `galbot_sdk` package exists; if one does, I'd be glad to link it.

## Limitations

- Kinematic control only: no `mj_step`, no gravity, no contact. Grippers set width; they do not hold objects.
- Blocking calls always complete and return `SUCCESS`; `speed`/`timeout` parameters are accepted but not raced (see below for why).
- Sensors are procedural — deterministic frames in the documented formats, not rendered images.
- Collision checking is registry-based, not geometric.
- `is_blocking=False` completes immediately.
- The G1 API is emulated on Galbot's public S1 model (leg mapping below).

## Behaviors discovered while building this

Documented because they are load-bearing for anyone who tries the same:

- **Speed/timeout semantics.** `example1` commands a 2.54 rad move at `speed=0.1 rad/s` with `timeout=20 s` — honest arithmetic needs 25.4 s. Emulating that race returns `TIMEOUT` and the tutorial fails; blocking calls here complete fully instead.
- **Out-of-range targets.** The `example1` heart pose commands `arm_joint6 = ±0.82` against a ±0.7854 joint range. Rejecting it fails the tutorial; targets are clipped to range, then succeed.
- **Actuator order ≠ joint order** in the S1 MJCF (left arm actuators 1–7, right arm 16–22, grippers 25–26). All mapping here is resolved by joint name at load time.
- **Gripper actuators can't servo** (kp = 1.0 vs 50,000 on the arms). Gripper control writes positions kinematically through the four-bar coupling (active2 = −θ, passive = +θ) and a 21-point width↔angle table derived from the model.
- **Leg emulation.** G1 tutorials command `leg_joint1–5`; the S1 model has a prismatic torso lift. `leg_joint2` maps to the lift (scaled); the others are stored and echoed back.

## Repo layout

```
galbot_sdk/     the backend (9 files: enums, types, world, robot, motion, wrappers)
tests/          run_tutorials.sh (the 10-tutorial suite) + import surface tests
demo/           two short videos of the backend driving the S1 model
quickstart.sh   one-command setup + proof
```

## License

MIT.
