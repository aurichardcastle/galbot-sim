"""test_import_surface.py — pytest tests for galbot_sdk import surface.

Validates that the GalbotSim shim exposes the same public API surface as
the official Galbot SDK v1.9.0.
"""

import importlib


def test_import_galbot_sdk():
    """Top-level galbot_sdk imports without error."""
    import galbot_sdk
    assert galbot_sdk is not None


def test_import_galbot_sdk_g1():
    """galbot_sdk.g1 subpackage imports without error."""
    import galbot_sdk.g1
    assert galbot_sdk.g1 is not None


def test_import_galbot_sdk_s1():
    """galbot_sdk.s1 subpackage imports without error."""
    import galbot_sdk.s1
    assert galbot_sdk.s1 is not None


def test_version():
    """__version__ matches the target SDK version."""
    import galbot_sdk
    assert galbot_sdk.__version__ == "1.9.0"
    assert galbot_sdk.version == "1.9.0"


def test_g1_galbot_robot_import():
    """GalbotRobot is importable from galbot_sdk.g1."""
    from galbot_sdk.g1 import GalbotRobot
    assert GalbotRobot is not None


def test_g1_control_status_import():
    """ControlStatus is importable from galbot_sdk.g1."""
    from galbot_sdk.g1 import ControlStatus
    assert ControlStatus is not None


def test_all_names_importable():
    """Every name in galbot_sdk.__all__ resolves to an actual object."""
    import galbot_sdk
    missing = []
    for name in galbot_sdk.__all__:
        obj = getattr(galbot_sdk, name, None)
        if obj is None:
            missing.append(name)
    assert missing == [], f"Names in __all__ not importable: {missing}"


def test_control_status_values():
    """ControlStatus enum values match the official SDK."""
    from galbot_sdk import ControlStatus
    assert int(ControlStatus.SUCCESS) == 0
    assert int(ControlStatus.TIMEOUT) == 1
    assert int(ControlStatus.FAULT) == 2
    assert int(ControlStatus.INVALID_INPUT) == 3
    assert int(ControlStatus.IN_PROGRESS) == 5
    assert int(ControlStatus.COMM_DISCONNECTED) == 9


def test_motion_status_values():
    """MotionStatus enum values match the official SDK."""
    from galbot_sdk import MotionStatus
    assert int(MotionStatus.SUCCESS) == 0
    assert int(MotionStatus.IN_PROGRESS) == 5
    assert int(MotionStatus.STATUS_NUM) == 10
    assert int(MotionStatus.UNSUPPORTED_FUNCRION) == 11  # official typo


def test_machine_type_values():
    """MachineType enum values match the official SDK."""
    from galbot_sdk import MachineType
    assert int(MachineType.G1) == 0
    assert int(MachineType.S1) == 1


def test_navigation_task_status_values():
    """NavigationTaskStatus enum values match the official SDK."""
    from galbot_sdk import NavigationTaskStatus
    assert int(NavigationTaskStatus.UNKNOWN) == 0
    assert int(NavigationTaskStatus.SUCCESS) == 2
    assert int(NavigationTaskStatus.FAILED) == 3


def test_g1_all_populated():
    """galbot_sdk.g1.__all__ is non-empty and contains core names."""
    import galbot_sdk.g1
    assert len(galbot_sdk.g1.__all__) > 0
    assert "GalbotRobot" in galbot_sdk.g1.__all__
    assert "ControlStatus" in galbot_sdk.g1.__all__


def test_s1_all_populated():
    """galbot_sdk.s1.__all__ is non-empty and contains core names."""
    import galbot_sdk.s1
    assert len(galbot_sdk.s1.__all__) > 0
    assert "GalbotRobot" in galbot_sdk.s1.__all__
    assert "ControlStatus" in galbot_sdk.s1.__all__
